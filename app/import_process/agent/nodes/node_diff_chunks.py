import sys
from typing import Any, Dict, List, Tuple

from app.clients.document_registry import get_document_snapshot
from app.core.logger import logger
from app.import_process.agent.import_summary import build_import_summary
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task


def node_diff_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    对当前切分结果和历史快照做 chunk 级 diff。

    这里会把 chunk 分成：
    - added
    - updated
    - deleted
    - unchanged

    下游 embedding 节点只处理 added + updated。
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> start node: {node_name}")
    add_running_task(state.get("task_id", ""), node_name)

    current_chunks = state.get("chunks") or []
    previous_document, previous_chunks = get_document_snapshot(state.get("doc_id", ""))
    added, updated, deleted, unchanged = diff_chunks(current_chunks, previous_chunks)

    state["previous_document"] = previous_document
    state["previous_chunks"] = previous_chunks
    state["added_chunks"] = added
    state["updated_chunks"] = updated
    state["deleted_chunks"] = deleted
    state["unchanged_chunks"] = unchanged
    state["all_chunks"] = current_chunks
    state["chunks"] = added + updated

    if not added and not updated and not deleted:
        state["skip_import"] = True
        state["skip_reason"] = "chunk_diff_unchanged"
    elif state.get("skip_reason") != "source_hash_unchanged":
        state["skip_reason"] = ""

    state["import_summary"] = build_import_summary(state, phase=node_name)

    logger.info(
        f"{node_name}: added={len(added)} updated={len(updated)} "
        f"deleted={len(deleted)} unchanged={len(unchanged)}"
    )
    return state


def diff_chunks(
    current_chunks: List[Dict[str, Any]],
    previous_chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    prev_map = {
        chunk.get("chunk_key"): chunk
        for chunk in previous_chunks
        if isinstance(chunk, dict) and chunk.get("chunk_key")
    }
    curr_map = {
        chunk.get("chunk_key"): chunk
        for chunk in current_chunks
        if isinstance(chunk, dict) and chunk.get("chunk_key")
    }

    added: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []
    deleted: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []

    for chunk_key, current in curr_map.items():
        previous = prev_map.get(chunk_key)
        if previous is None:
            added.append(current)
            continue

        merged = previous.copy()
        merged.update(current)
        if current.get("chunk_hash") != previous.get("chunk_hash"):
            updated.append(merged)
        else:
            unchanged.append(merged)

    for chunk_key, previous in prev_map.items():
        if chunk_key not in curr_map:
            deleted.append(previous)

    return added, updated, deleted, unchanged

