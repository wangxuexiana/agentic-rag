from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

from app.clients.document_registry import delete_document_snapshot
from app.clients.milvus_utils import get_milvus_client
from app.config.milvus_config import milvus_config
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.utils.escape_milvus_string_utils import escape_milvus_string

CHUNK_METADATA_COLLECTION_NAME = f"{milvus_config.chunks_collection}_meta"


def run_once(
        target_file: Path,
        local_dir: Path,
        task_id: str,
        external_doc_id: str = "",
) -> Dict[str, Any]:
    state = create_default_state(
        task_id=task_id,
        local_file_path=str(target_file),
        local_dir=str(local_dir),
        external_doc_id=(external_doc_id or "").strip(),
    )
    final_state = None
    for event in kb_import_app.stream(state, stream_mode="values"):
        final_state = event
    return final_state or state


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "skip_import": bool(state.get("skip_import")),
        "skip_reason": state.get("skip_reason", ""),
        "external_doc_id": state.get("external_doc_id", ""),
        "doc_id": state.get("doc_id", ""),
        "source_hash": state.get("source_hash", ""),
        "item_name": state.get("item_name", ""),
        "chunks_len": len(state.get("chunks") or []),
        "added": len(state.get("added_chunks") or []),
        "updated": len(state.get("updated_chunks") or []),
        "deleted": len(state.get("deleted_chunks") or []),
        "unchanged": len(state.get("unchanged_chunks") or []),
        "import_summary": state.get("import_summary") or {},
    }


def remove_heading_block(markdown_text: str, heading_keyword: str) -> str:
    """
    删除指定标题所在的整段内容，直到下一个同级或更高层级标题。
    """
    lines = markdown_text.splitlines()
    start_index = None
    start_level = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        title = stripped.lstrip("#").strip()
        level = len(stripped) - len(stripped.lstrip("#"))
        if heading_keyword in title:
            start_index = index
            start_level = level
            break

    if start_index is None or start_level is None:
        return markdown_text

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        if level <= start_level:
            end_index = index
            break

    new_lines = lines[:start_index] + lines[end_index:]
    return "\n".join(new_lines).strip() + "\n"


def cleanup_import_artifacts(states: list[Dict[str, Any]]) -> Dict[str, Any]:
    client = get_milvus_client()
    all_chunks = []
    item_names = set()
    doc_ids = set()

    for state in states:
        if not state:
            continue
        all_chunks.extend(state.get("chunks") or [])
        item_name = (state.get("item_name") or "").strip()
        if item_name:
            item_names.add(item_name)
        doc_id = (state.get("doc_id") or "").strip()
        if doc_id:
            doc_ids.add(doc_id)

    deleted_chunk_ids = 0
    chunk_ids = [str(chunk.get("chunk_id", "")).strip() for chunk in all_chunks if chunk.get("chunk_id")]
    if client and chunk_ids:
        unique_chunk_ids = sorted(set(chunk_ids))
        client.delete(
            collection_name=milvus_config.chunks_collection,
            filter=f"chunk_id in [{', '.join(unique_chunk_ids)}]",
        )
        deleted_chunk_ids = len(unique_chunk_ids)

    deleted_item_names = []
    if client and milvus_config.item_name_collection:
        for item_name in sorted(item_names):
            safe_item_name = escape_milvus_string(item_name)
            client.delete(
                collection_name=milvus_config.item_name_collection,
                filter=f'item_name=="{safe_item_name}"',
            )
            deleted_item_names.append(item_name)

    deleted_metadata_doc_ids = []
    if client and doc_ids and client.has_collection(collection_name=CHUNK_METADATA_COLLECTION_NAME):
        for doc_id in sorted(doc_ids):
            safe_doc_id = escape_milvus_string(doc_id)
            client.delete(
                collection_name=CHUNK_METADATA_COLLECTION_NAME,
                filter=f'doc_id=="{safe_doc_id}"',
            )
            deleted_metadata_doc_ids.append(doc_id)

    removed_doc_ids = []
    for doc_id in sorted(doc_ids):
        delete_document_snapshot(doc_id)
        removed_doc_ids.append(doc_id)

    return {
        "chunk_ids_deleted": deleted_chunk_ids,
        "item_names_deleted": deleted_item_names,
        "metadata_doc_ids_deleted": deleted_metadata_doc_ids,
        "doc_ids_removed_from_registry": removed_doc_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证增量导入链路")
    parser.add_argument("--old", required=True, help="旧版 Markdown/PDF 路径")
    parser.add_argument("--new", help="新版 Markdown/PDF 路径")
    parser.add_argument(
        "--delete-heading",
        default="11.2.3",
        help="删除场景里要移除的标题关键字，默认删除 11.2.3 这一节",
    )
    parser.add_argument(
        "--external-doc-id",
        default="",
        help="可选业务文档ID；传入后会优先作为增量同步的 doc_id",
    )
    args = parser.parse_args()

    old_path = Path(args.old).resolve()
    new_path = Path(args.new).resolve() if args.new else None
    external_doc_id = (args.external_doc_id or "").strip()

    temp_root = Path("output") / "incremental_verify" / str(uuid.uuid4())
    temp_root.mkdir(parents=True, exist_ok=True)
    target_file = temp_root / f"verify{old_path.suffix}"
    relocated_root = temp_root / "relocated"
    relocated_root.mkdir(parents=True, exist_ok=True)
    relocated_file = relocated_root / f"verify-relocated{old_path.suffix}"

    results: Dict[str, Any] = {
        "old_path": str(old_path),
        "new_path": str(new_path) if new_path else "",
        "delete_heading": args.delete_heading,
        "external_doc_id": external_doc_id,
    }

    first_state: Dict[str, Any] | None = None
    second_state: Dict[str, Any] | None = None
    third_state: Dict[str, Any] | None = None
    fourth_state: Dict[str, Any] | None = None
    relocated_state: Dict[str, Any] | None = None

    try:
        shutil.copyfile(old_path, target_file)
        first_state = run_once(
            target_file,
            temp_root,
            "verify-first-import",
            external_doc_id=external_doc_id,
        )
        results["first_import"] = summarize_state(first_state)

        second_state = run_once(
            target_file,
            temp_root,
            "verify-second-import",
            external_doc_id=external_doc_id,
        )
        results["second_import_same_file"] = summarize_state(second_state)

        if external_doc_id:
            shutil.copyfile(target_file, relocated_file)
            relocated_state = run_once(
                relocated_file,
                relocated_root,
                "verify-relocated-import",
                external_doc_id=external_doc_id,
            )
            results["relocated_import_same_external_doc_id"] = summarize_state(relocated_state)

        if new_path:
            shutil.copyfile(new_path, target_file)
            third_state = run_once(
                target_file,
                temp_root,
                "verify-third-import",
                external_doc_id=external_doc_id,
            )
            results["third_import_updated_file"] = summarize_state(third_state)

        if target_file.suffix.lower() == ".md":
            base_text = (new_path or old_path).read_text(encoding="utf-8")
            deleted_text = remove_heading_block(base_text, args.delete_heading)
            target_file.write_text(deleted_text, encoding="utf-8")
            fourth_state = run_once(
                target_file,
                temp_root,
                "verify-delete-import",
                external_doc_id=external_doc_id,
            )
            results["fourth_import_deleted_section"] = summarize_state(fourth_state)
    finally:
        results["cleanup"] = cleanup_import_artifacts([
            first_state or {},
            relocated_state or {},
            third_state or {},
            fourth_state or {},
        ])
        shutil.rmtree(temp_root, ignore_errors=True)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
