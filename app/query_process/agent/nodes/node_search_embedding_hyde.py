"""
HyDE 检索节点。

这条分支会先生成假设性文档，再把“query + hyde_doc”一起做 embedding 检索。
本次接入 retrieval cache 后，命中缓存时会直接跳过：
1. HyDE 文档生成
2. query 向量生成
3. Milvus 混合检索
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
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.services.cache_decorator import RetrievalCache
from app.utils.debug_trace_utils import append_trace_event
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task


load_dotenv(find_dotenv())


HYDE_TOPK = 5
HYDE_REQ_LIMIT = 10
HYDE_RANKER_WEIGHTS = (0.8, 0.2)


def node_search_embedding_hyde(state):
    """
    HyDE 检索主流程。
    """
    logger.info("--- node_search_embedding_hyde 开始执行 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    if not state.get("run_hyde", False):
        logger.info("Router 未启用 HyDE 检索，当前节点跳过")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": []}

    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    item_names = state.get("item_names") or []
    logger.info(f"HyDE 检索输入: query='{rewritten_query}', item_names={item_names}")

    def _compute_hyde():
        """生成 HyDE 文档并执行检索，返回 (chunks, extra_data)。"""
        doc = step_1_create_hyde_doc(rewritten_query)
        chunks = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=doc,
            item_names=item_names,
            req_limit=HYDE_REQ_LIMIT,
            top_k=HYDE_TOPK,
            ranker_weights=HYDE_RANKER_WEIGHTS,
        )
        return chunks, {"hyde_doc": doc}

    (hyde_embedding_chunks, cache_hit, extra) = RetrievalCache.execute(
        state=state,
        retrieval_type="hyde",
        topk=HYDE_TOPK,
        compute_fn=_compute_hyde,
        result_key="hyde_embedding_chunks",
        extra_cache_keys=["hyde_doc"],
    )
    hyde_doc = extra.get("hyde_doc", "")
    hit_count = len(hyde_embedding_chunks)
    logger.info(
        f"node_search_embedding_hyde 执行完成，命中 {hit_count} 条结果，"
        f"hyde_doc_length={len(hyde_doc or '')}, cache_hit={cache_hit}"
    )

    append_trace_event(
        session_id=state["session_id"],
        node="node_search_embedding_hyde",
        retrieval_round=int(state.get("retrieval_round", 1)),
        payload={
            "query": rewritten_query,
            "item_names": item_names,
            "hit_count": hit_count,
            "hyde_doc_length": len(hyde_doc or ""),
            "retrieval_cache_hit": cache_hit,
            "top_chunk_ids": [
                (hit.get("entity") or {}).get("chunk_id") or hit.get("id")
                for hit in hyde_embedding_chunks[:5]
            ],
        },
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "hyde_embedding_chunks": hyde_embedding_chunks,
        "hyde_doc": hyde_doc,
    }


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    先让 LLM 基于 query 生成一段更完整的假设性文档。
    """
    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")

    logger.info(f"Step 1: 开始生成 HyDE 文档, query={rewritten_query}")
    llm = get_llm_client()
    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    response = llm.invoke(hyde_prompt)
    hyde_doc = response.content

    logger.info(f"Step 1: HyDE 文档生成完成, length={len(hyde_doc)}")
    logger.debug(f"Step 1: HyDE 预览: {hyde_doc[:80]}...")
    return hyde_doc


def step_2_search_embedding_hyde(
    *,
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),
    norm_score: bool = True,
    output_fields=("chunk_id", "content", "item_name"),
):
    """
    用 “query + hyde_doc” 生成 embedding，并执行 Milvus 混合检索。
    """
    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")
    if not hyde_doc:
        raise ValueError("hyde_doc 不能为空")

    combined_text = f"{rewritten_query} {hyde_doc}"
    logger.info(f"Step 2: 组合 query 与 HyDE 文档, total_length={len(combined_text)}")

    embeddings = generate_embeddings([combined_text])
    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("缺少 CHUNKS_COLLECTION 配置，无法执行 HyDE 检索")
        return []

    expr = None
    if item_names:
        quoted = ", ".join(f'"{escape_milvus_string(value)}"' for value in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"Step 2: HyDE 检索使用 item_name 过滤: {expr}")
    else:
        logger.info("Step 2: HyDE 检索本轮不做 item_name 过滤")

    reqs = create_hybrid_search_requests(
        dense_vector=embeddings.get("dense")[0],
        sparse_vector=embeddings.get("sparse")[0],
        expr=expr,
        limit=req_limit,
    )

    client = get_milvus_client()
    if not client:
        logger.error("无法连接到 Milvus，HyDE 检索直接返回空结果")
        return []

    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=ranker_weights,
        norm_score=norm_score,
        limit=top_k,
        output_fields=list(output_fields),
    )
    if not res:
        return []

    hit_count = len(res[0]) if len(res) > 0 else 0
    logger.info(f"Step 2: HyDE 检索完成，召回 {hit_count} 条结果")
    if hit_count > 0:
        first_hit = res[0][0]
        logger.debug(f"Step 2: Top1 命中示例: {first_hit}")
    return res[0]
