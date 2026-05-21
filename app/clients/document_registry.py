import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient

from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT


load_dotenv()

# 注册表的职责：
# 1. 保存文档级快照（source_hash / md_hash / item_name / doc_version）
# 2. 保存 chunk 级快照（chunk_key / chunk_hash / section_path / chunk_id）
#
# 现在优先使用 Mongo 作为正式存储，解决：
# - 多进程下 JSON 文件覆盖
# - 多实例无法共享
# - 异常中断时文件状态不一致
#
# 如果环境里没有配置 Mongo，则自动回退到本地 JSON，保证开发环境仍可直接运行。
REGISTRY_DIR = PROJECT_ROOT / "output" / ".document_registry"
REGISTRY_PATH = REGISTRY_DIR / "document_registry.json"
DEFAULT_DOCUMENTS_COLLECTION = os.getenv("IMPORT_DOCUMENTS_COLLECTION", "import_documents")
DEFAULT_CHUNKS_COLLECTION = os.getenv("IMPORT_DOCUMENT_CHUNKS_COLLECTION", "import_document_chunks")
SNAPSHOT_CHUNK_FIELDS = {
    "chunk_id",
    "doc_id",
    "doc_version",
    "chunk_key",
    "chunk_hash",
    "section_path",
    "part",
    "file_title",
    "item_name",
    "title",
    "parent_title",
    "content",
}


def _empty_registry() -> Dict[str, Dict[str, Any]]:
    return {"documents": {}, "chunks": {}}


def _ensure_registry_file() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text(
            json.dumps(_empty_registry(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _load_registry() -> Dict[str, Dict[str, Any]]:
    _ensure_registry_file()
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"failed to load registry json, fallback to empty: {exc}")
        return _empty_registry()


def _save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    _ensure_registry_file()
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_chunk_snapshot(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    注册表只保存增量同步必需字段，不保存向量正文外的大对象。

    这样可以避免：
    - Mongo 因稀疏向量的非字符串 key 写入失败
    - JSON 文件体积无意义膨胀
    - registry 和 Milvus 主数据重复存太多内容
    """
    normalized = {}
    for key, value in (chunk or {}).items():
        if key not in SNAPSHOT_CHUNK_FIELDS:
            continue
        normalized[str(key)] = value
    return normalized


class _JsonRegistryBackend:
    backend_name = "json"

    def get_document_snapshot(self, doc_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not doc_id:
            return {}, []
        registry = _load_registry()
        document = deepcopy(registry.get("documents", {}).get(doc_id, {}))
        chunks = list(deepcopy(registry.get("chunks", {}).get(doc_id, {})).values())
        chunks.sort(key=lambda item: (item.get("section_path", ""), item.get("part", 0)))
        return document, chunks

    def save_document_snapshot(self, doc_id: str, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        if not doc_id:
            return
        registry = _load_registry()
        registry.setdefault("documents", {})[doc_id] = deepcopy(document)
        registry.setdefault("chunks", {})[doc_id] = {
            chunk["chunk_key"]: _normalize_chunk_snapshot(chunk)
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("chunk_key")
        }
        _save_registry(registry)

    def delete_document_snapshot(self, doc_id: str) -> None:
        if not doc_id:
            return
        registry = _load_registry()
        registry.get("documents", {}).pop(doc_id, None)
        registry.get("chunks", {}).pop(doc_id, None)
        _save_registry(registry)


class _MongoRegistryBackend:
    backend_name = "mongo"

    def __init__(self) -> None:
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        if not mongo_url or not db_name:
            raise ValueError("missing MONGO_URL or MONGO_DB_NAME")

        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.documents = self.db[DEFAULT_DOCUMENTS_COLLECTION]
        self.chunks = self.db[DEFAULT_CHUNKS_COLLECTION]

        # 文档快照按 doc_id 唯一；chunk 快照按 (doc_id, chunk_key) 唯一。
        self.documents.create_index("doc_id", unique=True)
        self.chunks.create_index([("doc_id", 1), ("chunk_key", 1)], unique=True)
        self.chunks.create_index([("doc_id", 1), ("section_path", 1), ("part", 1)])

    def get_document_snapshot(self, doc_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not doc_id:
            return {}, []

        document = self.documents.find_one({"doc_id": doc_id}, {"_id": 0}) or {}
        chunks = list(
            self.chunks.find(
                {"doc_id": doc_id},
                {"_id": 0},
            ).sort([("section_path", 1), ("part", 1)])
        )
        return deepcopy(document), deepcopy(chunks)

    def save_document_snapshot(self, doc_id: str, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        if not doc_id:
            return

        safe_document = deepcopy(document)
        safe_document["doc_id"] = doc_id
        self.documents.replace_one(
            {"doc_id": doc_id},
            safe_document,
            upsert=True,
        )

        # 用“先删后插”保持当前文档快照的完整一致性。
        self.chunks.delete_many({"doc_id": doc_id})
        chunk_docs = []
        for chunk in chunks:
            if not isinstance(chunk, dict) or not chunk.get("chunk_key"):
                continue
            item = _normalize_chunk_snapshot(chunk)
            item["doc_id"] = doc_id
            chunk_docs.append(item)
        if chunk_docs:
            self.chunks.insert_many(chunk_docs, ordered=False)

    def delete_document_snapshot(self, doc_id: str) -> None:
        if not doc_id:
            return
        self.documents.delete_one({"doc_id": doc_id})
        self.chunks.delete_many({"doc_id": doc_id})


_registry_backend = None


def get_registry_backend():
    global _registry_backend
    if _registry_backend is not None:
        return _registry_backend

    try:
        _registry_backend = _MongoRegistryBackend()
        logger.info(
            f"document registry backend initialized: mongo "
            f"({DEFAULT_DOCUMENTS_COLLECTION}/{DEFAULT_CHUNKS_COLLECTION})"
        )
    except Exception as exc:
        logger.warning(f"document registry fallback to json backend: {exc}")
        _registry_backend = _JsonRegistryBackend()
    return _registry_backend


def get_document_snapshot(doc_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # 对外暴露统一接口，业务层无需关心底层是 Mongo 还是 JSON。
    return get_registry_backend().get_document_snapshot(doc_id)


def save_document_snapshot(doc_id: str, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
    get_registry_backend().save_document_snapshot(doc_id, document, chunks)


def delete_document_snapshot(doc_id: str) -> None:
    get_registry_backend().delete_document_snapshot(doc_id)
