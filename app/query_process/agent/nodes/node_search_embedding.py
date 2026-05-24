"""
本地知识库 embedding 检索节点。

第一版 retrieval cache 先接在这里：
1. 先查缓存
2. 未命中时再生成 query embedding 并调用 Milvus
3. 命中时直接复用历史检索结果
"""

from __future__ import annotations

import os
import sys

from dotenv import find_dotenv, load_dotenv

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.services.cache_decorator import RetrievalCache
from app.utils.debug_trace_utils import append_trace_event
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task


load_dotenv(find_dotenv())


RETRIEVAL_TOPK = 5
RETRIEVAL_REQ_LIMIT = 10
RETRIEVAL_RANKER_WEIGHTS = (0.8, 0.2)


def node_search_embedding(state):
    """
    基于 rewritten_query 和 item_names 执行本地 Milvus 混合检索。
    """
    logger.info("--- node_search_embedding 开始执行 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    if not state.get("run_embedding", False):
        logger.info("Router 未启用 embedding 检索，当前节点跳过")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": []}

    query = state.get("rewritten_query") or state.get("original_query") or ""
    item_names = state.get("item_names") or []
    logger.info(f"本地检索输入: query='{query}', item_names={item_names}")

    embedding_chunks, cache_hit, _ = RetrievalCache.execute(
        state=state,
        retrieval_type="embedding",
        topk=RETRIEVAL_TOPK,
        compute_fn=lambda: _run_embedding_search(query=query, item_names=item_names),
        result_key="embedding_chunks",
    )
    hit_count = len(embedding_chunks)
    logger.info(f"node_search_embedding 执行完成，命中 {hit_count} 条结果，cache_hit={cache_hit}")

    append_trace_event(
        session_id=state["session_id"],
        node="node_search_embedding",
        retrieval_round=int(state.get("retrieval_round", 1)),
        payload={
            "query": query,
            "item_names": item_names,
            "hit_count": hit_count,
            "retrieval_cache_hit": cache_hit,
            "top_chunk_ids": [
                (hit.get("entity") or {}).get("chunk_id") or hit.get("id")
                for hit in embedding_chunks[:5]
            ],
        },
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": embedding_chunks}


def _run_embedding_search(*, query: str, item_names: list):
    """
    真实执行 query embedding 生成和 Milvus 混合检索。
    """
    if not query:
        logger.warning("Embedding retrieval 跳过：query 为空")
        return []

    logger.info("开始为 query 生成 dense/sparse embedding")
    embeddings = generate_embeddings([query])
    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]
    logger.debug(f"向量生成完成: dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}")

    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("缺少 CHUNKS_COLLECTION 配置，无法执行本地 embedding 检索")
        return []

    expr = None
    if item_names:
        quoted = ", ".join(f'"{escape_milvus_string(value)}"' for value in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"本轮检索使用 item_name 过滤: {expr}")
    else:
        logger.info("本轮未识别到 item_name，执行全库召回")

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=RETRIEVAL_REQ_LIMIT,
    )

    client = get_milvus_client()
    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        norm_score=True,
        limit=RETRIEVAL_TOPK,
        output_fields=["chunk_id", "content", "item_name"],
    )

    if not res:
        return []

    hit_count = len(res[0]) if len(res) > 0 else 0
    logger.info(f"Milvus 本地混合检索完成，召回 {hit_count} 条结果")
    if hit_count > 0:
        logger.debug(f"Top1 检索结果示例: {res[0][0]}")
    return res[0]
