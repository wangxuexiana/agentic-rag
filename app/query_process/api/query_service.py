"""
查询服务 (Query Service) — FastAPI 应用

本模块是「Knowledge RAG」查询流程的 Web 入口，负责接收前端请求并调度 LangGraph 查询图。

核心职责：
1. 暴露 REST API 端点，支持同步/流式两种查询模式
2. 管理 SSE（Server-Sent Events）长连接，实现 LLM 答案的实时推送
3. 提供会话历史的查询与清除接口
4. 内嵌一个简易聊天页面（chat.html），方便本地调试

服务端口：8002（独立于文件导入服务的 8001）

主要端点：
- POST /query          — 提交查询（支持流式/同步）
- GET  /stream/{id}    — SSE 流式结果推送
- GET  /history/{id}   — 查询会话历史
- DELETE /history/{id} — 清除会话历史
- GET  /chat.html      — 聊天调试页面
- GET  /health         — 健康检查
"""

from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain.agents.middleware.todo import Todo
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.core.langsmith import (
    add_langsmith_middleware,
    bootstrap_langsmith,
    build_tracing_metadata,
    get_langsmith_project,
    maybe_tracing_context,
)
from app.query_process.agent.main_graph import query_app

bootstrap_langsmith()

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="Knowledge RAG 查询服务")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
add_langsmith_middleware(app)

# 返回chat.html页面
@app.get("/chat.html")  # 对外访问地址
async def chat():
    # 从 api -> query_process
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    # 定义chat.html位置
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    # 如果不存在，抛出404异常
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")
    return FileResponse(chat_html_path)

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")  # ...必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")



# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


def build_default_query_state(session_id: str, user_query: str, is_stream: bool) -> dict:
    return {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream,

        # 初始化 Agent 状态
        "rewritten_query": user_query,
        "history": [],
        "item_names": [],

        # 规划层默认值
        "intent_type": "unknown",
        "task_type": "full_agentic",
        "retrieval_plan": {},
        "selected_tools": [],
        "need_clarify": False,
        "clarification_question": "",

        # Router 开关默认值
        "run_embedding": False,
        "run_hyde": False,
        "run_kg": False,
        "run_web_search": False,
        "router_reason": "",

        # 检索轮次
        "retrieval_round": 1,
        "max_retrieval_rounds": 2,
        "followup_query": "",
        "retry_intent": "",

        # 检索结果
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "kg_chunks": [],
        "web_search_docs": [],

        # 融合结果
        "rrf_chunks": [],
        "reranked_docs": [],

        # 证据判断
        "evidence_status": "unknown",
        "reflection_reason": "",
        "missing_facts": [],
        "citations": [],
        "final_confidence": 0.0,
        "support_score": 0.0,
        "coverage_score": 0.0,
        "consistency_score": 0.0,

        # 输出
        "prompt": "",
        "answer": "",
        "cache_stats": {"stages": {}, "backend": {}},
    }


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    """
    执行 LangGraph 查询流程图的核心调度函数。

    工作流程：
    1. 构造完整的默认状态字典（覆盖 QueryGraphState 的所有字段）
    2. 调用 query_app.invoke() 执行整条查询图
    3. 执行成功后更新任务状态为 COMPLETED
    4. 异常时更新任务状态为 FAILED，并通过 SSE 推送错误事件

    :param session_id: 会话唯一标识，贯穿整个查询流程
    :param user_query: 用户原始查询文本
    :param is_stream: 是否为流式响应模式
    """
    # 这个函数相当于查询侧的“状态装配器”：
    # API 层收到的是 query/session_id/is_stream 这几个轻量参数，
    # LangGraph 需要的是一整份 QueryGraphState 初始状态。
    # 入口阅读提示：
    # 1. 这里先把 HTTP 请求参数装配成 QueryGraphState 初始字典
    # 2. 再把整份状态交给 query_app.invoke() 进入 LangGraph
    # 3. 后续所有节点都不再层层传参，而是统一读写这份共享 state
    # 4. invoke 结束后，这里负责收尾：更新任务状态、处理错误、推送 SSE
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    # 这里把所有状态字段显式列出来，阅读时不要把它看成冗余初始化。
    # 它的价值在于：
    # 1. 一眼看清查询链里有哪些状态会被节点读写；
    # 2. 避免节点依赖“某个字段也许存在”的隐式约定；
    # 3. 为后续调试单个节点提供稳定起点。
    default_state = build_default_query_state(session_id=session_id, user_query=user_query, is_stream=is_stream)

    try:
        # 后期运行
        invoke_metadata = build_tracing_metadata(
            service="query_service",
            operation="run_query_graph",
            session_id=session_id,
            is_stream=is_stream,
            extra={"query_length": len(user_query)},
        )
        invoke_config = {
            "run_name": "query_graph_run",
            "tags": ["query-service", "langgraph", "query"],
            "metadata": invoke_metadata,
        }
        with maybe_tracing_context(
            project_name=get_langsmith_project(),
            tags=["query-service", "langgraph", "query"],
            metadata=invoke_metadata,
        ):
            final_state = query_app.invoke(default_state, config=invoke_config)
        # 整体任务就更新完了！ 接下来就是数据的更新了！
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        return final_state
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        raise



@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph, session_id,user_query,is_stream)
        # 返回结果
        print("开始处理结果....")
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }
    else:
        # 同步运行
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "done_list": get_done_task_list(session_id),
            "cache_stats": get_task_result_json(session_id, "cache_stats", {}),
        }



@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print("调用流式/stream...")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/agent/main_graph.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
