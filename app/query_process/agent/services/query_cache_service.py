from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

from dotenv import load_dotenv

from app.clients.cache_client import get_cache, get_cache_stats_snapshot, set_cache
from app.query_process.agent.services.cache_key_service import (
    build_answer_cache_key,
    build_item_name_confirm_cache_key,
    build_planner_cache_key,
    build_retrieval_cache_key,
    build_rerank_cache_key,
)


load_dotenv()


PLANNER_CACHE_TTL_SECONDS = int(os.getenv("PLANNER_CACHE_TTL_SECONDS", "900"))
PLANNER_CACHE_PROMPT_VERSION = os.getenv("PLANNER_CACHE_PROMPT_VERSION", "v1")
PLANNER_CACHE_MODEL_VERSION = os.getenv("PLANNER_CACHE_MODEL_VERSION", "default")
RERANK_CACHE_TTL_SECONDS = int(os.getenv("RERANK_CACHE_TTL_SECONDS", "900"))
RERANK_CACHE_MODEL_VERSION = os.getenv("RERANK_CACHE_MODEL_VERSION", "bge-reranker-large")
RERANK_CACHE_TOPK_VERSION = os.getenv("RERANK_CACHE_TOPK_VERSION", "v1")
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "900"))
RETRIEVAL_CACHE_INDEX_VERSION = os.getenv("RETRIEVAL_CACHE_INDEX_VERSION", "default")
ITEM_NAME_CONFIRM_CACHE_TTL_SECONDS = int(os.getenv("ITEM_NAME_CONFIRM_CACHE_TTL_SECONDS", "900"))
ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "900"))
ANSWER_CACHE_PROMPT_VERSION = os.getenv("ANSWER_CACHE_PROMPT_VERSION", "v1")
ANSWER_CACHE_MODEL_VERSION = os.getenv("ANSWER_CACHE_MODEL_VERSION", "default")


def ensure_cache_stats_dict(state: Dict) -> Dict:
    cache_stats = state.get("cache_stats")
    if not isinstance(cache_stats, dict):
        cache_stats = {"stages": {}, "backend": {}}
        state["cache_stats"] = cache_stats
    cache_stats.setdefault("stages", {})
    cache_stats.setdefault("backend", {})
    return cache_stats


def record_stage_cache_result(state: Dict, *, stage: str, cache_hit: bool, detail: Optional[Dict] = None) -> Dict:
    """
    记录当前 query 在某个阶段的缓存命中情况。
    """
    cache_stats = ensure_cache_stats_dict(state)
    stage_stats = cache_stats["stages"].setdefault(stage, {})
    stage_stats["cache_hit"] = bool(cache_hit)
    if detail:
        stage_stats.update(detail)
    cache_stats["backend"] = get_cache_stats_snapshot()
    return cache_stats


def get_planner_cache(
    *,
    query: str,
    history: Iterable[Dict],
    item_names: Iterable[str],
    doc_scope: str = "",
) -> Optional[Dict]:
    key = build_planner_cache_key(
        query=query,
        history=history,
        item_names=item_names,
        prompt_version=PLANNER_CACHE_PROMPT_VERSION,
        model_version=PLANNER_CACHE_MODEL_VERSION,
        doc_scope=doc_scope,
    )
    value = get_cache(key)
    return value if isinstance(value, dict) else None


def set_planner_cache(
    *,
    query: str,
    history: Iterable[Dict],
    item_names: Iterable[str],
    plan: Dict,
    doc_scope: str = "",
    ttl_seconds: Optional[int] = None,
) -> bool:
    if not isinstance(plan, dict) or not plan:
        return False

    key = build_planner_cache_key(
        query=query,
        history=history,
        item_names=item_names,
        prompt_version=PLANNER_CACHE_PROMPT_VERSION,
        model_version=PLANNER_CACHE_MODEL_VERSION,
        doc_scope=doc_scope,
    )
    payload = {
        "intent_type": plan.get("intent_type", "unknown"),
        "task_type": plan.get("task_type", "full_agentic"),
        "selected_tools": list(plan.get("selected_tools") or []),
        "need_clarify": bool(plan.get("need_clarify", False)),
        "success_criteria": plan.get("success_criteria", ""),
        "notes": plan.get("notes", ""),
    }
    return set_cache(
        key,
        payload,
        ttl_seconds=ttl_seconds or PLANNER_CACHE_TTL_SECONDS,
    )


def get_rerank_cache(
    *,
    query: str,
    candidate_docs: Iterable[Dict],
) -> Optional[Dict]:
    key = build_rerank_cache_key(
        query=query,
        candidate_docs=candidate_docs,
        reranker_version=RERANK_CACHE_MODEL_VERSION,
        topk_version=RERANK_CACHE_TOPK_VERSION,
    )
    value = get_cache(key)
    return value if isinstance(value, dict) else None


def set_rerank_cache(
    *,
    query: str,
    candidate_docs: Iterable[Dict],
    rerank_result: Dict,
    ttl_seconds: Optional[int] = None,
) -> bool:
    if not isinstance(rerank_result, dict) or not isinstance(rerank_result.get("reranked_docs"), list):
        return False

    key = build_rerank_cache_key(
        query=query,
        candidate_docs=candidate_docs,
        reranker_version=RERANK_CACHE_MODEL_VERSION,
        topk_version=RERANK_CACHE_TOPK_VERSION,
    )
    return set_cache(
        key,
        {"reranked_docs": rerank_result.get("reranked_docs", [])},
        ttl_seconds=ttl_seconds or RERANK_CACHE_TTL_SECONDS,
    )


def get_retrieval_cache(
    *,
    retrieval_type: str,
    query: str,
    item_names: Iterable[str],
    topk: int,
) -> Optional[Dict]:
    key = build_retrieval_cache_key(
        retrieval_type=retrieval_type,
        query=query,
        item_names=item_names,
        topk=topk,
        index_version=RETRIEVAL_CACHE_INDEX_VERSION,
    )
    value = get_cache(key)
    return value if isinstance(value, dict) else None


def set_retrieval_cache(
    *,
    retrieval_type: str,
    query: str,
    item_names: Iterable[str],
    topk: int,
    retrieval_result: Dict,
    ttl_seconds: Optional[int] = None,
) -> bool:
    if not isinstance(retrieval_result, dict):
        return False

    key = build_retrieval_cache_key(
        retrieval_type=retrieval_type,
        query=query,
        item_names=item_names,
        topk=topk,
        index_version=RETRIEVAL_CACHE_INDEX_VERSION,
    )
    return set_cache(
        key,
        retrieval_result,
        ttl_seconds=ttl_seconds or RETRIEVAL_CACHE_TTL_SECONDS,
    )


def get_item_name_confirm_cache(
    *,
    query: str,
    history: Iterable[Dict],
    task_type: str = "",
    need_clarify: bool = False,
) -> Optional[Dict]:
    key = build_item_name_confirm_cache_key(
        query=query,
        history=history,
        task_type=task_type,
        need_clarify=need_clarify,
    )
    value = get_cache(key)
    return value if isinstance(value, dict) else None


def set_item_name_confirm_cache(
    *,
    query: str,
    history: Iterable[Dict],
    task_type: str = "",
    need_clarify: bool = False,
    result: Dict,
    ttl_seconds: Optional[int] = None,
) -> bool:
    if not isinstance(result, dict) or not result:
        return False
    key = build_item_name_confirm_cache_key(
        query=query,
        history=history,
        task_type=task_type,
        need_clarify=need_clarify,
    )
    payload = {
        "rewritten_query": result.get("rewritten_query", ""),
        "item_names": list(result.get("item_names") or []),
        "answer": result.get("answer", ""),
    }
    return set_cache(
        key,
        payload,
        ttl_seconds=ttl_seconds or ITEM_NAME_CONFIRM_CACHE_TTL_SECONDS,
    )


def get_answer_cache(
    *,
    query: str,
    item_names: Iterable[str],
    reranked_docs: Iterable[Dict],
    evidence_status: str,
) -> Optional[Dict]:
    key = build_answer_cache_key(
        query=query,
        item_names=item_names,
        reranked_docs=reranked_docs,
        evidence_status=evidence_status,
        prompt_version=ANSWER_CACHE_PROMPT_VERSION,
        model_version=ANSWER_CACHE_MODEL_VERSION,
    )
    value = get_cache(key)
    return value if isinstance(value, dict) else None


def set_answer_cache(
    *,
    query: str,
    item_names: Iterable[str],
    reranked_docs: Iterable[Dict],
    evidence_status: str,
    answer_result: Dict,
    ttl_seconds: Optional[int] = None,
) -> bool:
    if not isinstance(answer_result, dict) or not answer_result.get("answer"):
        return False
    key = build_answer_cache_key(
        query=query,
        item_names=item_names,
        reranked_docs=reranked_docs,
        evidence_status=evidence_status,
        prompt_version=ANSWER_CACHE_PROMPT_VERSION,
        model_version=ANSWER_CACHE_MODEL_VERSION,
    )
    payload = {
        "answer": answer_result.get("answer", ""),
        "prompt": answer_result.get("prompt", ""),
    }
    return set_cache(
        key,
        payload,
        ttl_seconds=ttl_seconds or ANSWER_CACHE_TTL_SECONDS,
    )
