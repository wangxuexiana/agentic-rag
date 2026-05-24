# Zhiku

Zhiku 是一个面向企业设备文档的 Agentic RAG 知识库系统。项目围绕“文档导入”和“智能问答”两条主流程构建：导入侧负责解析 PDF/Markdown、处理图片、切分文档、识别产品名称、生成向量并写入 Milvus；查询侧基于 LangGraph 编排规划、产品确认、多路检索、RRF 融合、重排序、证据反思和答案生成。

## 功能特性

- 文档导入：支持 PDF 和 Markdown 文件上传，自动进入知识库导入流程。
- 文档解析：通过 MinerU 将 PDF 转为 Markdown，并对文档图片进行摘要和 MinIO 存储。
- 增量导入：基于文档快照对 chunk 做差异检测，减少重复向量化和重复入库。
- 混合检索：使用 BGE-M3 dense/sparse embedding 在 Milvus 中执行混合检索。
- 多路召回：支持本地 embedding 检索、HyDE 检索、知识图谱检索占位和百炼 MCP 联网搜索。
- Agentic 查询：通过 Planner、Tool Router、Evidence Reflection 和 Dynamic Retrieval 进行查询规划与补检索。
- 结果治理：RRF 融合、reranker 精排、证据充分性判断、引用和缓存统计。
- 流式输出：查询服务提供 SSE 流式事件，便于前端实时展示进度和回答。
- 离线评测：内置本地评测、RAGAS 评测和评分报告脚本。

## 技术栈

- Web 服务：FastAPI、Uvicorn
- 工作流编排：LangGraph
- LLM 接入：LangChain、OpenAI 兼容 API、阿里云百炼 DashScope
- 向量模型：BGE-M3
- 重排序模型：BGE Reranker
- 向量数据库：Milvus
- 对象存储：MinIO
- 会话历史：MongoDB
- 联网搜索：阿里云百炼 MCP WebSearch
- 观测追踪：LangSmith

## 项目结构

```text
.
├── app/
│   ├── clients/                  # MongoDB、Milvus、MinIO 等客户端封装
│   ├── config/                   # 环境变量配置对象
│   ├── core/                     # 日志、Prompt 加载、LangSmith 初始化
│   ├── import_process/           # 文档导入服务和 LangGraph 导入流程
│   ├── lm/                       # LLM、Embedding、Reranker 工具
│   ├── query_process/            # 查询服务和 LangGraph 查询流程
│   ├── tool/                     # 离线评测、模型下载、调试工具
│   └── utils/                    # SSE、任务状态、路径和格式化工具
├── doc/                          # 示例设备手册和阅读文档
├── prompts/                      # 查询、HyDE、图片摘要等 Prompt 模板
├── static_flowcharts/            # 主要流程图
├── test/offline_eval/            # 离线评测数据和报告
├── main.py                       # 基础健康检查入口
├── pyproject.toml                # Python 项目依赖
└── .env.example                  # 环境变量示例
```

## 环境要求

- Python 3.11+
- Milvus
- MongoDB
- MinIO
- 可用的 OpenAI 兼容大模型服务
- MinerU API Token
- 可选：LangSmith、DashScope MCP WebSearch、Redis 缓存

建议使用 `uv` 管理依赖：

```bash
uv sync
```

如果不使用 `uv`，也可以根据 `pyproject.toml` 手动安装依赖。

## 配置

复制环境变量模板：

```bash
copy .env.example .env
```

然后按本地环境修改 `.env`。核心配置包括：

- `OPENAI_API_KEY`、`OPENAI_API_BASE`、`LLM_DEFAULT_MODEL`
- `MINERU_API_TOKEN`、`MINERU_BASE_URL`
- `MONGO_URL`、`MONGO_DB_NAME`
- `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME`
- `MILVUS_URL`、`CHUNKS_COLLECTION`、`ITEM_NAME_COLLECTION`
- `BGE_M3_PATH`、`BGE_DEVICE`
- `BGE_RERANKER_LARGE`、`BGE_RERANKER_DEVICE`
- `DASHSCOPE_API_KEY`、`MCP_DASHSCOPE_BASE_URL`
- `LANGSMITH_API_KEY`、`LANGSMITH_PROJECT`

注意：`.env` 包含密钥，已在 `.gitignore` 中忽略，不要提交到仓库。

## 启动服务

### 1. 文档导入服务

```bash
uv run python app/import_process/api/file_import_service.py
```

默认地址：

- API: `http://127.0.0.1:8001`
- 上传页: `http://127.0.0.1:8001/import.html`
- 健康检查: `http://127.0.0.1:8001/health`

主要接口：

- `POST /upload`：上传 PDF/Markdown 并启动导入任务
- `GET /status/{task_id}`：查询导入任务状态

### 2. 查询问答服务

```bash
uv run python app/query_process/api/query_service.py
```

默认地址：

- API: `http://127.0.0.1:8002`
- 调试页: `http://127.0.0.1:8002/chat.html`
- 健康检查: `http://127.0.0.1:8002/health`

主要接口：

- `POST /query`：提交查询，支持同步和流式模式
- `GET /stream/{session_id}`：获取 SSE 流式事件
- `GET /history/{session_id}`：查询会话历史
- `DELETE /history/{session_id}`：清理会话历史

同步查询示例：

```bash
curl -X POST http://127.0.0.1:8002/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"HAK180 如何设置局部烫金？\",\"is_stream\":false}"
```

## 查询流程

查询链路由 `app/query_process/agent/main_graph.py` 编排：

```text
用户问题
  -> Planner
  -> 产品名称确认
  -> Tool Router
  -> Embedding / HyDE / WebSearch / KG 多路检索
  -> RRF 融合
  -> Rerank 精排
  -> Evidence Reflection
  -> Dynamic Retrieval 或 Answer Output
```

当证据不足或冲突时，系统会在最大轮次限制内生成补检索意图并重新进入检索链路。

## 文档导入流程

导入链路由 `app/import_process/agent/main_graph.py` 编排：

```text
上传文件
  -> 入口检查
  -> PDF 转 Markdown 或读取 Markdown
  -> 图片摘要与 MinIO 上传
  -> 文档切分
  -> 产品名称识别
  -> chunk diff
  -> BGE-M3 向量化
  -> Milvus 入库
```

## 离线评测

项目提供离线评测脚本，入口位于 `app/tool/`：

```bash
uv run python app/tool/run_local_eval.py
uv run python app/tool/run_offline_eval.py app/tool/offline_eval_cases.sample.jsonl
uv run python app/tool/score_offline_eval.py <run_result.jsonl>
uv run python app/tool/run_ragas_eval.py <run_result.jsonl>
```

评测样例和历史报告位于 `test/offline_eval/`。

## 开发建议

- 优先通过 `.env.example` 对齐配置，再启动外部依赖。
- 本地开发默认使用单进程服务，任务状态和 SSE 队列保存在进程内存中。
- 生产环境建议将任务状态、缓存和流式事件迁移到 Redis 或消息队列，并将导入任务拆到独立 worker。
- 大模型、向量模型、Milvus、MongoDB、MinIO 均属于运行时依赖，单元测试建议使用 mock 或 fake client 隔离。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动导入服务
uv run python app/import_process/api/file_import_service.py

# 启动查询服务
uv run python app/query_process/api/query_service.py

# Python 语法检查
uv run python -m compileall app main.py
```

## 许可证

当前仓库尚未声明许可证。若需要开源发布，请补充 `LICENSE` 文件。
