"""
掌柜智库（zhiku）- 企业级RAG知识库系统

项目概述：
    本项目是一个基于 LangGraph 的 Agentic RAG（检索增强生成）系统，
    核心能力是将产品手册/说明书等文档自动解析、切分、向量化后入库，
    并通过多路检索 + 证据反思 + 动态补检索的智能流程，为用户提供精准的问答服务。

两大核心流程：
    1. 文档导入流程（import_process）：
       PDF/MD → MinerU解析 → 图片处理 → 文档切分 → 商品名识别 → BGE-M3向量化 → Milvus入库
    2. 智能查询流程（query_process）：
       用户问题 → Planner规划 → 商品名确认 → 多路检索(Embedding/HyDE/KG/Web) → RRF融合 → Rerank重排
       → 证据反思 → (不足则动态补检索) → LLM生成答案

技术栈：
    - Web框架：FastAPI + Uvicorn
    - 工作流引擎：LangGraph（状态图驱动）
    - LLM：LangChain + OpenAI兼容API（通义千问等）
    - 向量数据库：Milvus（稠密+稀疏混合检索）
    - 对象存储：MinIO（图片/文件存储）
    - 文档数据库：MongoDB（会话历史记录）
    - 图数据库：Neo4j（知识图谱，预留）
    - Embedding模型：BGE-M3（多语言双向量）
    - Rerank模型：BGE-Reranker-Large
    - PDF解析：MinerU云端API
    - 网络搜索：百炼MCP服务

项目入口：
    - 文件导入服务：app/import_process/api/file_import_service.py（端口8001）
    - 查询问答服务：app/query_process/api/query_service.py（端口8002）

本文件（main.py）为FastAPI基础入口，仅提供健康检查接口。
正式业务请使用上述两个独立服务入口。
"""
from fastapi import FastAPI

app = FastAPI(
    title="掌柜智库 - 企业级RAG知识库系统",
    description="Agentic RAG系统：文档导入 + 智能查询"
)

@app.get("/")
def read_root():
    """健康检查接口，确认服务启动状态"""
    return {"Hello": "World", "service": "zhiku"}


# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/api/query_service.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
