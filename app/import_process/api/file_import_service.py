"""
文件导入服务 (File Import Service)

这是导入链对外的 Web 入口，核心职责不是“处理文档”，而是“编排一次导入任务”：
1. 接收上传文件并安全落盘；
2. 可选上传到 MinIO 做对象存储；
3. 初始化导入状态；
4. 将真正的导入工作交给 LangGraph 后台执行；
5. 对外提供任务状态查询接口。

阅读建议：
先看 `upload_files()` 理解任务是如何创建的，再看 `run_graph_task()` 理解
FastAPI 是如何把 Web 请求桥接到 `import_process.agent.main_graph` 的。
"""

import os
import shutil
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
from app.core.langsmith import (
    add_langsmith_middleware,
    bootstrap_langsmith,
    build_tracing_metadata,
    get_langsmith_project,
    maybe_tracing_context,
)

bootstrap_langsmith()
# 第三方库
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_result_json,
    set_task_result,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".md"}

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
def _sanitize_uploaded_filename(raw_filename: str) -> str:
    """
    清理客户端上传的文件名。

    这里必须把 filename 当成不可信输入处理，因为客户端完全可以传：
    - ../a.txt
    - ..\\..\\windows\\system32\\x
    - /absolute/path/file.pdf

    处理策略很直接：
    1. 把 Windows 反斜杠统一成正斜杠；
    2. 只取最后一段名字，彻底去掉目录信息；
    3. 去掉首尾空白；
    4. 空值时给一个安全兜底名。
    """
    normalized = (raw_filename or "").replace("\\", "/")
    safe_name = normalized.split("/")[-1].strip()
    return safe_name or "uploaded_file"


def _build_safe_local_upload_path(task_local_dir: str, raw_filename: str) -> tuple[str, str]:
    """
    构造最终落盘路径，并验证路径没有逃出任务目录。

    仅仅做 basename 截断还不够稳妥，所以这里再做一次 resolve + commonpath 校验：
    - task_local_dir 是允许写入的边界目录；
    - final_path 必须解析后仍然位于这个目录之内；
    - 一旦不满足，就拒绝这次上传。
    """
    safe_name = _sanitize_uploaded_filename(raw_filename)
    task_dir_resolved = Path(task_local_dir).resolve()
    final_path = (task_dir_resolved / safe_name).resolve()

    if os.path.commonpath([str(task_dir_resolved), str(final_path)]) != str(task_dir_resolved):
        raise HTTPException(status_code=400, detail="Invalid upload filename")

    return str(final_path), safe_name


def _validate_upload_file(file: UploadFile) -> None:
    safe_name = _sanitize_uploaded_filename(file.filename)
    suffix = Path(safe_name).suffix.lower()

    if not safe_name or safe_name == "uploaded_file":
        raise HTTPException(status_code=400, detail="Upload filename is required")

    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {safe_name}. Only {allowed} files are allowed.",
        )


app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)
add_langsmith_middleware(app)

# --------------------------
# 静态页面路由：返回文件导入前端页面import.html
# 访问地址：http://localhost:8000/import.html
# --------------------------
@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """返回文件导入前端页面：import.html"""
    # 拼接HTML文件绝对路径，基于项目根目录定位
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    # 日志记录页面访问的文件路径，方便排查文件不存在问题
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    # 校验文件是否存在，不存在则抛出404异常
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    # 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
    )


@app.get("/health")
async def health():
    return {"ok": True}


# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(
        task_id: str,
        local_dir: str,
        local_file_path: str,
        external_doc_id: str = "",
):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    :param external_doc_id: 可选业务文档ID；传入后优先作为 doc_id 使用
    """
    # 这里相当于 Web 层和 LangGraph 层之间的“适配器”：
    # Web 层只知道 task_id / 文件路径；
    # 图执行层需要的是完整 state。
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径
        init_state["external_doc_id"] = (external_doc_id or "").strip()
        final_state = init_state

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        invoke_metadata = build_tracing_metadata(
            service="file_import_service",
            operation="run_graph_task",
            task_id=task_id,
            extra={"local_file_path": local_file_path},
        )
        invoke_config = {
            "run_name": "import_graph_run",
            "tags": ["import-service", "langgraph", "import"],
            "metadata": invoke_metadata,
        }
        with maybe_tracing_context(
            project_name=get_langsmith_project(),
            tags=["import-service", "langgraph", "import"],
            metadata=invoke_metadata,
        ):
            for event in kb_import_app.stream(init_state, config=invoke_config):
                for node_name, node_result in event.items():
                    # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                    logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                    # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                    add_done_task(task_id, node_name)
                    if isinstance(node_result, dict):
                        final_state.update(node_result)
                        import_summary = node_result.get("import_summary")
                        if import_summary:
                            # 将结构化导入摘要写入任务结果，供状态接口和 SSE 直接复用。
                            set_task_result(
                                task_id,
                                "import_summary",
                                json.dumps(import_summary, ensure_ascii=False),
                            )
                        item_name = node_result.get("item_name")
                        if item_name:
                            set_task_result(task_id, "item_name", item_name)
                        current_external_doc_id = node_result.get("external_doc_id")
                        if current_external_doc_id:
                            set_task_result(task_id, "external_doc_id", current_external_doc_id)

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        final_summary = final_state.get("import_summary")
        if final_summary:
            set_task_result(
                task_id,
                "import_summary",
                json.dumps(final_summary, ensure_ascii=False),
            )
        if final_state.get("item_name"):
            set_task_result(task_id, "item_name", final_state["item_name"])
        if final_state.get("external_doc_id"):
            set_task_result(task_id, "external_doc_id", final_state["external_doc_id"])
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        set_task_result(task_id, "error", str(e))
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)


# --------------------------
# 核心接口：文件上传接口
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        external_doc_ids: List[str] = Form(default=[]),
):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 这个接口的阅读重点不在 FastAPI 语法，而在“一个上传文件如何变成一个独立任务”。
    # 每个文件都会生成独立 task_id、独立本地目录、独立 MinIO 对象路径。
    # 这样后续即使多文件并发上传，也不会互相污染状态。
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []
    normalized_external_doc_ids = [(value or "").strip() for value in (external_doc_ids or [])]

    # 业务文档 ID 允许两种模式：
    # 1. 不传：全部走默认路径 hash
    # 2. 传和文件数相同的列表：逐个文件绑定稳定业务主键
    if normalized_external_doc_ids and len(normalized_external_doc_ids) != len(files):
        raise HTTPException(
            status_code=400,
            detail="external_doc_ids count must match files count",
        )

    for file in files:
        _validate_upload_file(file)

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for index, file in enumerate(files):
        external_doc_id = normalized_external_doc_ids[index] if normalized_external_doc_ids else ""
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        update_task_status(task_id, "pending")
        logger.info(
            f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}，"
            f"external_doc_id={external_doc_id or '-'}"
        )
        if external_doc_id:
            # 在任务结果里提前记录，便于状态接口在图执行前也能回显这个业务ID。
            set_task_result(task_id, "external_doc_id", external_doc_id)

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)  # 目录不存在则创建，存在则不做处理
        # 构建上传文件的本地保存绝对路径
        # 上传文件名来自客户端，不能直接信任。
        # 这里统一做文件名清理和目录边界校验，确保文件只能写入当前 task 目录。
        local_file_abs_path, safe_filename = _build_safe_local_upload_path(task_local_dir, file.filename)

        # 5. 将上传的文件保存到本地临时目录（后续MinIO上传/文件解析均基于此文件）
        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 6. 将本地文件上传至MinIO对象存储，做持久化保存
        # 从环境变量获取MinIO的PDF存储目录配置
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")  # 缺省值：pdf_files
        # 构建MinIO中的文件对象名：配置目录/YYYYMMDD/文件名（按日期分层，和本地一致）
        # MinIO 对象名必须带 task_id。
        # 如果只用“日期/文件名”，同一天不同任务上传同名文件时会互相覆盖。
        # 这里和本地目录保持一致，使用 task_id 做隔离。
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{task_id}/{safe_filename}"
        try:
            # 获取MinIO客户端实例
            minio_client = get_minio_client()
            if minio_client is None:
                # MinIO客户端获取失败，抛出500服务异常
                raise HTTPException(status_code=500,
                                    detail="MinIO service connection failed, please check MinIO config")
            # 从环境变量获取MinIO的桶名配置
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")  # 缺省值：kb-import-bucket

            # 本地文件上传至MinIO（同名文件会自动覆盖，保证文件最新）
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type  # 传递文件原始MIME类型
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")
        except Exception as e:
            # MinIO上传失败，记录警告日志（不中断后续流程，本地文件仍可继续处理）
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)

        # 7. 标记「文件上传」阶段为「已完成」，前端轮询可查
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        background_tasks.add_task(
            run_graph_task,
            task_id,
            task_local_dir,
            local_file_abs_path,
            external_doc_id,
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID（前端基于TaskID轮询进度）
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }
# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id),  # 正在运行的节点/阶段列表
        # 导入摘要会告诉调用方这次是全量新增、文档跳过，还是 chunk 级跳过/删除。
        "import_summary": get_task_result_json(task_id, "import_summary", {}),
        "item_name": get_task_result(task_id, "item_name", ""),
        "external_doc_id": get_task_result(task_id, "external_doc_id", ""),
        "error": get_task_result(task_id, "error", ""),
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info
# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8001  # 服务端口
    )

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/import_process/agent/main_graph.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
