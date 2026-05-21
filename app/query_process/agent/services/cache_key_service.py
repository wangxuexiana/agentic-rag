import hashlib
import json
from typing import Any, Dict, Iterable, Optional


def _normalize_text(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def resolve_stable_query_text(
    *,
    original_query: str,
    rewritten_query: str,
    item_names: Iterable[str],
) -> str:
    """
    为 retrieval / rerank 生成更稳定的 query key 输入。

    设计原因：
    - `rewritten_query` 可能被 LLM 改写成不同表达，导致相同问题二次请求也无法命中缓存。
    - 当 `item_names` 已经确认时，检索范围已经被实体过滤收敛，此时优先使用 `original_query`
      作为 key 输入更稳定。
    - 若尚未确认实体，则退回 `rewritten_query`，保留改写对检索策略的影响。
    """
    normalized_original = _normalize_text(original_query)
    normalized_rewritten = _normalize_text(rewritten_query)
    has_item_names = any(str(name).strip() for name in (item_names or []))
    if has_item_names and normalized_original:
        return normalized_original
    return normalized_rewritten or normalized_original


def _normalize_history(history: Iterable[Dict[str, Any]]) -> list:
    normalized = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "role": _normalize_text(item.get("role")),
                "content": _normalize_text(item.get("content")),
            }
        )
    return normalized


def build_planner_cache_key(
    *,
    query: str,
    history: Iterable[Dict[str, Any]],
    item_names: Iterable[str],
    prompt_version: str = "v1",
    model_version: str = "default",
    doc_scope: str = "",
) -> str:
    """
    Planner 层的缓存 key。

    这里把 query、history、item_names、prompt/model 版本一起算进去，
    目的是保证：
    - 相同输入可稳定命中
    - prompt 或模型升级后自动失效
    - 产品范围变化时不复用旧规划结果
    """
    payload = {
        "stage": "planner",
        "query": _normalize_text(query),
        "history": _normalize_history(history),
        "item_names": sorted(
            _normalize_text(name)
            for name in (item_names or [])
            if str(name).strip()
        ),
        "prompt_version": prompt_version,
        "model_version": model_version,
        "doc_scope": _normalize_text(doc_scope),
    }
    return f"query:planner:{_stable_json_hash(payload)}"


def build_rerank_cache_key(
    *,
    query: str,
    candidate_docs: Iterable[Dict[str, Any]],
    reranker_version: str = "default",
    topk_version: str = "v1",
) -> str:
    """
    Rerank 层的缓存 key。

    优先使用候选文档的稳定标识来构造 key：
    - chunk_id
    - doc_id
    - url

    如果都没有，再退回到 text hash，避免无法区分不同候选集。
    """
    normalized_candidates = []
    for item in candidate_docs or []:
        if not isinstance(item, dict):
            continue
        identity = (
            item.get("chunk_id")
            or item.get("doc_id")
            or item.get("url")
            or _stable_json_hash({"text": _normalize_text(item.get("text"))})
        )
        normalized_candidates.append(
            {
                "id": str(identity),
                "source": _normalize_text(item.get("source")),
                "title": _normalize_text(item.get("title")),
            }
        )

    payload = {
        "stage": "rerank",
        "query": _normalize_text(query),
        "candidates": normalized_candidates,
        "reranker_version": reranker_version,
        "topk_version": topk_version,
    }
    return f"query:rerank:{_stable_json_hash(payload)}"


def build_retrieval_cache_key(
    *,
    retrieval_type: str,
    query: str,
    item_names: Iterable[str],
    topk: int,
    index_version: str = "default",
) -> str:
    """
    Retrieval 层缓存 key。

    第一版先覆盖本地 embedding 检索，所以 key 里带：
    - retrieval_type
    - query
    - item_names 过滤范围
    - topk
    - index_version
    """
    payload = {
        "stage": "retrieval",
        "retrieval_type": _normalize_text(retrieval_type),
        "query": _normalize_text(query),
        "item_names": sorted(
            _normalize_text(name)
            for name in (item_names or [])
            if str(name).strip()
        ),
        "topk": int(topk or 0),
        "index_version": _normalize_text(index_version),
    }
    return f"query:retrieval:{_stable_json_hash(payload)}"


def build_item_name_confirm_cache_key(
    *,
    query: str,
    history: Iterable[Dict[str, Any]],
    task_type: str = "",
    need_clarify: bool = False,
) -> str:
    """
    item_name_confirm 节点缓存 key。
    这里绑定原始 query、最近历史和 planner 给出的澄清语义，避免在上下文变化时误复用。
    """
    payload = {
        "stage": "item_name_confirm",
        "query": _normalize_text(query),
        "history": _normalize_history(history),
        "task_type": _normalize_text(task_type),
        "need_clarify": bool(need_clarify),
    }
    return f"query:item-name-confirm:{_stable_json_hash(payload)}"


def build_answer_cache_key(
    *,
    query: str,
    item_names: Iterable[str],
    reranked_docs: Iterable[Dict[str, Any]],
    evidence_status: str,
    prompt_version: str = "v1",
    model_version: str = "default",
) -> str:
    """
    最终答案缓存 key。
    只在证据集合与实体范围稳定时复用答案，避免单纯按 query 命中旧答案。
    """
    normalized_docs = []
    for item in reranked_docs or []:
        if not isinstance(item, dict):
            continue
        identity = (
            item.get("chunk_id")
            or item.get("doc_id")
            or item.get("url")
            or _stable_json_hash({"text": _normalize_text(item.get("text"))})
        )
        normalized_docs.append(
            {
                "id": str(identity),
                "source": _normalize_text(item.get("source")),
                "title": _normalize_text(item.get("title")),
                "score": round(float(item.get("score") or 0.0), 6),
            }
        )

    payload = {
        "stage": "answer",
        "query": _normalize_text(query),
        "item_names": sorted(
            _normalize_text(name)
            for name in (item_names or [])
            if str(name).strip()
        ),
        "reranked_docs": normalized_docs,
        "evidence_status": _normalize_text(evidence_status),
        "prompt_version": _normalize_text(prompt_version),
        "model_version": _normalize_text(model_version),
    }
    return f"query:answer:{_stable_json_hash(payload)}"
