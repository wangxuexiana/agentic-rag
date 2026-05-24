"""
Rerank 节点。

职责：
1. 合并本地 RRF 结果与 web 检索结果
2. 用 reranker 模型计算相关性分数
3. 基于分数做动态 TopK 截断
4. 接入 rerank cache，避免同一批候选被重复精排
"""

from __future__ import annotations

import sys
from typing import Dict, List

from app.core.logger import logger
from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.services.cache_decorator import RerankCache
from app.utils.debug_trace_utils import append_trace_event
from app.utils.task_utils import add_done_task, add_running_task


RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.5


def node_rerank(state):
    """
    Rerank 主流程。

    先把候选文档统一成标准结构，再尝试命中 rerank cache。
    未命中时执行真实 rerank，命中时直接复用上次结果。
    """
    logger.info("--- Rerank 节点开始执行 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info(
        f"进入 Rerank 前路由状态: "
        f"selected_tools={state.get('selected_tools')}, "
        f"run_embedding={state.get('run_embedding')}, "
        f"run_hyde={state.get('run_hyde')}, "
        f"run_kg={state.get('run_kg')}, "
        f"run_web_search={state.get('run_web_search')}"
    )

    doc_items = step_1_merge_docs(state)

    def _compute_rerank():
        """执行 rerank + 动态 TopK 截断，返回截断后的文档列表。"""
        scored = step_2_rerank_docs(state, doc_items)
        return step_3_topk(scored)

    topk_docs, cache_hit = RerankCache.execute(
        state=state,
        candidate_docs=doc_items,
        compute_fn=_compute_rerank,
    )
    scored_docs = topk_docs
    logger.info(f"Rerank 节点处理结束，最终输出 {len(topk_docs)} 条文档")

    append_trace_event(
        session_id=state["session_id"],
        node="node_rerank",
        retrieval_round=int(state.get("retrieval_round", 1)),
        payload={
            "merged_doc_count": len(doc_items),
            "reranked_doc_count": len(scored_docs),
            "topk_doc_count": len(topk_docs),
            "rerank_cache_hit": cache_hit,
            "top_chunk_ids": [
                (doc.get("chunk_id") or doc.get("doc_id") or doc.get("url"))
                for doc in topk_docs[:5]
            ],
            "top_scores": [float(doc.get("score", 0.0) or 0.0) for doc in topk_docs[:5]],
        },
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs": topk_docs}


def step_1_merge_docs(state) -> List[Dict]:
    """
    把本地知识库候选和 web 搜索候选统一成 reranker 可处理的结构。
    """
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []

    logger.info(f"Step 1: 合并文档，本地 {len(rrf_docs)} 条，Web {len(web_docs)} 条")
    doc_items: List[Dict] = []

    for index, doc in enumerate(rrf_docs):
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc
        if not isinstance(entity, dict):
            logger.warning(f"本地文档格式异常(index={index}): {type(entity)}")
            continue

        content = entity.get("content")
        if not content:
            logger.debug(f"跳过空内容本地文档(index={index})")
            continue

        doc_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name") or ""

        doc_items.append(
            {
                "text": content,
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "title": title,
                "url": "",
                "source": "local",
            }
        )

    for index, doc in enumerate(web_docs):
        if not isinstance(doc, dict):
            preview = str(doc)
            if len(preview) > 160:
                preview = preview[:160] + "..."
            logger.warning(
                f"web_search_docs[{index}] type mismatch: "
                f"expected dict, got {type(doc).__name__}, value={preview}"
            )
            continue

        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        if not text:
            logger.debug(f"跳过空内容 web 文档(index={index})")
            continue

        doc_items.append(
            {
                "text": text,
                "doc_id": None,
                "chunk_id": None,
                "title": title,
                "url": url,
                "source": "web",
            }
        )

    logger.info(f"Step 1: 合并完成，共输出 {len(doc_items)} 条标准化文档")
    return doc_items


def step_2_rerank_docs(state, doc_items: List[Dict]) -> List[Dict]:
    """
    调用 reranker 计算 query 与每条候选文档的相关性分数。
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""
    if not doc_items or not question:
        logger.warning("Step 2: 跳过 rerank，原因是没有候选文档或没有 query")
        return []

    logger.info(f"Step 2: 开始 rerank，待排序文档数: {len(doc_items)}")
    texts = [item["text"] for item in doc_items]

    try:
        reranker = get_reranker_model()
        sentence_pairs = [[question, text] for text in texts]
        logger.info("Step 2: 正在计算 rerank 分数")
        scores = reranker.compute_score(sentence_pairs)

        scored_docs: List[Dict] = []
        for item, text, score in zip(doc_items, texts, scores):
            scored_docs.append(
                {
                    "text": text,
                    "score": float(score),
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                }
            )

        scored_docs.sort(key=lambda value: value["score"], reverse=True)
        return scored_docs
    except Exception as exc:
        logger.error(f"Step 2: rerank 发生异常: {exc}", exc_info=True)
        return [
            {
                "text": item.get("text"),
                "score": 0.0,
                "source": item.get("source") or "",
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "url": item.get("url") or "",
                "title": item.get("title") or "",
            }
            for item in doc_items
        ]


def step_3_topk(scored_docs: List[Dict]) -> List[Dict]:
    """
    对 rerank 分数做动态 TopK 截断，避免机械固定取前 N 条。
    """
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS

    topk = max_topk
    if topk > min_topk:
        for index in range(min_topk - 1, max_topk - 1):
            score_current = scored_docs[index].get("score")
            score_next = scored_docs[index + 1].get("score")
            gap = score_current - score_next
            rel = gap / (abs(score_current) + 1e-6)
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(
                    f"Step 3: 触发断崖截断 @ index={index} "
                    f"(score {score_current:.4f} -> {score_next:.4f}, gap={gap:.4f})"
                )
                topk = index + 1
                break

    topk_docs = scored_docs[:topk]
    logger.info(f"Step 3: 截断完成，保留前 {len(topk_docs)} 条文档")

    if topk_docs:
        preview = ", ".join(
            f"{doc.get('chunk_id') or 'Web'}({doc.get('score'):.3f})"
            for doc in topk_docs[:3]
        )
        logger.debug(f"Step 3: Top3 预览: {preview}")

    return topk_docs
