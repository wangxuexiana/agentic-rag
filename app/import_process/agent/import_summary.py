from __future__ import annotations

from typing import Any, Dict


def build_import_summary(state: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """
    统一生成本次导入的增量摘要。

    这个摘要的目标不是替代详细日志，而是让调用方一眼看出：
    - 这次是不是跳过了
    - 跳过发生在文档级还是 chunk 级
    - 实际新增/更新/删除了多少 chunk
    - 节省了多少 embedding / 入库工作量
    """
    added_count = len(state.get("added_chunks") or [])
    updated_count = len(state.get("updated_chunks") or [])
    deleted_count = len(state.get("deleted_chunks") or [])
    unchanged_count = len(state.get("unchanged_chunks") or [])
    active_count = len(state.get("chunks") or [])
    all_chunk_count = len(state.get("all_chunks") or [])
    changed_count = added_count + updated_count
    skip_reason = state.get("skip_reason", "")
    skip_import = bool(state.get("skip_import"))

    if skip_import and skip_reason == "source_hash_unchanged":
        action = "skip_document"
    elif skip_import and skip_reason == "chunk_diff_unchanged":
        action = "skip_chunks"
    elif deleted_count and not changed_count:
        action = "delete_only"
    elif changed_count:
        action = "incremental_upsert"
    else:
        action = "noop"

    return {
        "phase": phase,
        "action": action,
        "skip_import": skip_import,
        "skip_reason": skip_reason,
        # external_doc_id 用于区分“业务上的文档身份”和“本地临时路径”。
        # 当上游传了稳定业务 ID 时，这里会直接带出来，便于接口层核对。
        "external_doc_id": state.get("external_doc_id", ""),
        "doc_id": state.get("doc_id", ""),
        "file_title": state.get("file_title", ""),
        "item_name": state.get("item_name", ""),
        "source_hash": state.get("source_hash", ""),
        "doc_version": state.get("doc_version", ""),
        "all_chunk_count": all_chunk_count,
        "active_chunk_count": active_count,
        "added_count": added_count,
        "updated_count": updated_count,
        "deleted_count": deleted_count,
        "unchanged_count": unchanged_count,
        # 这些字段更偏“成本视角”，用来判断增量到底省了多少事。
        "embedding_recompute_count": changed_count,
        "embedding_skipped_count": unchanged_count,
        "milvus_upsert_count": changed_count,
        "milvus_delete_count": deleted_count,
    }
