import sys
from typing import Any, Dict, List

from pymilvus import DataType

from app.clients.document_registry import save_document_snapshot
from app.clients.milvus_utils import get_milvus_client
from app.config.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.import_summary import build_import_summary
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task


CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection
CHUNK_METADATA_COLLECTION_NAME = f"{CHUNKS_COLLECTION_NAME}_meta"


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    # 这个节点不再做“按 item_name 全删全插”，而是执行真正的增量同步：
    # - deleted_chunks：定向删除
    # - changed_chunks(added + updated)：重新写入
    # - unchanged_chunks：直接沿用，不动向量库
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> start node: {node_name}")
    add_running_task(state.get("task_id", ""), node_name)

    changed_chunks = state.get("chunks") or []
    unchanged_chunks = state.get("unchanged_chunks") or []
    deleted_chunks = state.get("deleted_chunks") or []

    client = prepare_collection(changed_chunks or unchanged_chunks)

    if deleted_chunks:
        delete_chunks(client, deleted_chunks)
    if changed_chunks:
        changed_chunks = upsert_chunks(client, changed_chunks)

    # 无论主向量集合是不是旧 schema，正式元数据都统一写入独立 metadata collection。
    sync_metadata_collection(client, changed_chunks, deleted_chunks)

    # 入库完成后，把“未变化 + 已更新”的 chunk 重新拼回当前文档的有效全集。
    active_chunks = merge_active_chunks(unchanged_chunks, changed_chunks)
    state["chunks"] = active_chunks

    # 注册表快照必须在 Milvus 操作成功后再更新，否则会出现“快照比向量库更新”的不一致。
    save_document_snapshot(
        state.get("doc_id", ""),
        {
            "doc_id": state.get("doc_id", ""),
            "doc_version": state.get("doc_version", ""),
            "file_title": state.get("file_title", ""),
            "item_name": state.get("item_name", ""),
            "source_path": state.get("local_file_path", ""),
            "source_hash": state.get("source_hash", ""),
            "md_hash": state.get("md_hash", ""),
        },
        active_chunks,
    )

    logger.info(
        f"{node_name}: upserted={len(changed_chunks)} "
        f"deleted={len(deleted_chunks)} active={len(active_chunks)}"
    )
    state["import_summary"] = build_import_summary(state, phase=node_name)
    return state


def prepare_collection(reference_chunks: List[Dict[str, Any]]):
    # 保留自动建表逻辑，避免新环境首次导入失败。
    client = get_milvus_client()
    if client is None:
        raise ValueError("Milvus client unavailable")
    if not CHUNKS_COLLECTION_NAME:
        raise ValueError("CHUNKS_COLLECTION_NAME not configured")

    vector_dimension = 1024
    if reference_chunks and reference_chunks[0].get("dense_vector"):
        vector_dimension = len(reference_chunks[0]["dense_vector"])

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
    ensure_metadata_collection(client)
    return client


def get_collection_compatibility(client) -> tuple[set[str], bool]:
    """
    兼容旧版集合 schema。
    如果当前集合还是历史结构，就不能直接插入 doc_id / chunk_key 等新增字段。
    """
    description = client.describe_collection(collection_name=CHUNKS_COLLECTION_NAME)
    field_names = {field["name"] for field in description.get("fields", [])}
    enable_dynamic_field = bool(description.get("enable_dynamic_field", False))
    return field_names, enable_dynamic_field


def create_collection(client, collection_name: str, vector_dimension: int) -> None:
    # 继续沿用 auto_id 主键，原因是当前项目已有这一套使用方式。
    # 业务层真正稳定识别 chunk 的字段是 chunk_key，而不是 Milvus 的自增主键。
    schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="doc_version", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="chunk_key", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="chunk_hash", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT64)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"},
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def ensure_metadata_collection(client) -> None:
    """
    为增量同步准备正式 metadata collection。

    背景：
    - 历史 chunk 主集合里不一定有 doc_id/chunk_key/chunk_hash 等字段；
    - 直接重建主集合风险太高，会影响查询链路；
    - 因此将“增量同步所需元数据”单独沉淀到一个正式集合中。
    """
    if client.has_collection(collection_name=CHUNK_METADATA_COLLECTION_NAME):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_fields=True)
    schema.add_field(field_name="chunk_key", datatype=DataType.VARCHAR, max_length=255, is_primary=True)
    # MilvusClient 的 create_collection 需要至少一个向量字段。
    # 这里加一个 2 维占位向量，只为承载正式元数据 schema，不参与任何检索逻辑。
    schema.add_field(field_name="meta_vector", datatype=DataType.FLOAT_VECTOR, dim=2)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="doc_version", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="chunk_hash", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT64)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    client.create_collection(
        collection_name=CHUNK_METADATA_COLLECTION_NAME,
        schema=schema,
        index_params=_build_metadata_index_params(client),
    )


def _build_metadata_index_params(client):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="meta_vector",
        index_name="meta_vector_index",
        index_type="FLAT",
        metric_type="L2",
    )
    return index_params


def delete_chunks(client, deleted_chunks: List[Dict[str, Any]]) -> None:
    # 优先按历史 chunk_id 删除，最快也最直接。
    hydrate_chunk_ids_from_metadata(client, deleted_chunks)
    ids = [str(chunk.get("chunk_id", "")).strip() for chunk in deleted_chunks if chunk.get("chunk_id")]
    if ids:
        client.delete(collection_name=CHUNKS_COLLECTION_NAME, filter=f"chunk_id in [{', '.join(ids)}]")
        return

    logger.warning("delete_chunks: no chunk_id resolved from metadata, skip main collection delete")


def upsert_chunks(client, changed_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 对于 updated chunk，先删旧的再插新的。
    # 当前 schema 还是 auto_id 模式，所以“更新”本质上是 delete + insert。
    hydrate_chunk_ids_from_metadata(client, changed_chunks)
    old_ids = [str(chunk.get("chunk_id", "")).strip() for chunk in changed_chunks if chunk.get("chunk_id")]
    if old_ids:
        client.delete(collection_name=CHUNKS_COLLECTION_NAME, filter=f"chunk_id in [{', '.join(old_ids)}]")

    supported_fields, enable_dynamic_field = get_collection_compatibility(client)
    data_to_insert = []
    for chunk in changed_chunks:
        item = chunk.copy()
        # chunk_id 由 Milvus 自动生成，这里不能把旧值直接带进去。
        item.pop("chunk_id", None)
        if not enable_dynamic_field:
            # 旧集合不支持动态字段时，只保留 schema 已声明的字段。
            item = {key: value for key, value in item.items() if key in supported_fields}
        data_to_insert.append(item)

    insert_result = client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=data_to_insert)
    inserted_ids = insert_result.get("ids", [])
    for index, chunk in enumerate(changed_chunks):
        if index < len(inserted_ids):
            # 回填新的 chunk_id，供下一次增量更新时定向删除使用。
            chunk["chunk_id"] = str(inserted_ids[index])
    return changed_chunks


def sync_metadata_collection(
    client,
    changed_chunks: List[Dict[str, Any]],
    deleted_chunks: List[Dict[str, Any]],
) -> None:
    """
    同步正式 metadata collection。

    这样做之后，哪怕主 chunk 集合还是历史 schema，也可以：
    - 按 doc_id 查询一份文档有哪些 chunk
    - 按 chunk_key/chunk_hash 追踪某个切片的历史身份
    - 为未来迁移到“主集合自带完整业务字段”做准备
    """
    delete_metadata_rows(client, deleted_chunks)
    upsert_metadata_rows(client, changed_chunks)


def delete_metadata_rows(client, chunks: List[Dict[str, Any]]) -> None:
    chunk_keys = sorted({
        str(chunk.get("chunk_key", "")).strip()
        for chunk in chunks
        if str(chunk.get("chunk_key", "")).strip()
    })
    if not chunk_keys:
        return

    quoted = ", ".join(f'"{escape_milvus_string(chunk_key)}"' for chunk_key in chunk_keys)
    client.delete(
        collection_name=CHUNK_METADATA_COLLECTION_NAME,
        filter=f"chunk_key in [{quoted}]",
    )


def upsert_metadata_rows(client, chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return

    delete_metadata_rows(client, chunks)
    rows = []
    for chunk in chunks:
        chunk_key = str(chunk.get("chunk_key", "")).strip()
        if not chunk_key:
            continue
        rows.append({
            "chunk_key": chunk_key,
            "meta_vector": [0.0, 0.0],
            "chunk_id": int(chunk.get("chunk_id")) if str(chunk.get("chunk_id", "")).strip() else None,
            "doc_id": str(chunk.get("doc_id", "")).strip(),
            "doc_version": str(chunk.get("doc_version", "")).strip(),
            "chunk_hash": str(chunk.get("chunk_hash", "")).strip(),
            "section_path": str(chunk.get("section_path", "")).strip(),
            "part": int(chunk.get("part", 0) or 0),
            "file_title": str(chunk.get("file_title", "")).strip(),
            "item_name": str(chunk.get("item_name", "")).strip(),
            "title": str(chunk.get("title", "")).strip(),
            "parent_title": str(chunk.get("parent_title", "")).strip(),
            "content": str(chunk.get("content", "")).strip(),
        })
    if rows:
        client.insert(collection_name=CHUNK_METADATA_COLLECTION_NAME, data=rows)


def hydrate_chunk_ids_from_metadata(client, chunks: List[Dict[str, Any]]) -> None:
    """
    通过 metadata collection 反查旧 chunk_id。

    作用：
    - 让 delete/update 不再依赖主集合是否带 doc_id/chunk_key 字段；
    - 历史快照里没保存 chunk_id 时，也能靠 chunk_key 找回旧向量主键。
    """
    pending_keys = sorted({
        str(chunk.get("chunk_key", "")).strip()
        for chunk in chunks
        if not str(chunk.get("chunk_id", "")).strip() and str(chunk.get("chunk_key", "")).strip()
    })
    if not pending_keys:
        return
    if not client.has_collection(collection_name=CHUNK_METADATA_COLLECTION_NAME):
        return

    quoted = ", ".join(f'"{escape_milvus_string(chunk_key)}"' for chunk_key in pending_keys)
    rows = client.query(
        collection_name=CHUNK_METADATA_COLLECTION_NAME,
        filter=f"chunk_key in [{quoted}]",
        output_fields=["chunk_key", "chunk_id"],
    )
    chunk_id_map = {
        str(row.get("chunk_key", "")).strip(): str(row.get("chunk_id", "")).strip()
        for row in (rows or [])
        if str(row.get("chunk_key", "")).strip() and str(row.get("chunk_id", "")).strip()
    }
    for chunk in chunks:
        if chunk.get("chunk_id"):
            continue
        chunk_key = str(chunk.get("chunk_key", "")).strip()
        if chunk_key in chunk_id_map:
            chunk["chunk_id"] = chunk_id_map[chunk_key]


def merge_active_chunks(
    unchanged_chunks: List[Dict[str, Any]],
    changed_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # 最终结果按 section_path + part 排序，便于后续查看和调试。
    merged = unchanged_chunks + changed_chunks
    merged.sort(key=lambda item: (item.get("section_path", ""), item.get("part", 0)))
    return merged
