from typing import TypedDict
import copy

from app.core.logger import logger


class ImportGraphState(TypedDict):
    # 任务级元数据，用于日志追踪和任务展示。
    task_id: str

    # 入口分流开关。
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # 兼容原有流程里的切分和服务开关。
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # 导入过程里的路径字段。
    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str
    split_path: str
    embeddings_path: str

    # 文档内容与切分结果。
    md_content: str
    chunks: list
    item_name: str

    # 预留业务主键入口。
    # 当前默认还是走路径 hash，但如果上游已经有稳定文档 ID，
    # 可以直接传 external_doc_id，避免“换路径就变成新文档”。
    external_doc_id: str

    # 增量同步核心字段。
    doc_id: str
    doc_version: str
    source_hash: str
    md_hash: str
    skip_import: bool
    skip_reason: str

    # diff 阶段产物。
    previous_document: dict
    previous_chunks: list
    added_chunks: list
    updated_chunks: list
    deleted_chunks: list
    unchanged_chunks: list
    all_chunks: list

    # 结构化导入摘要，供日志、脚本和上层接口直接读取。
    import_summary: dict

    embeddings_content: list


graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "external_doc_id": "",
    "doc_id": "",
    "doc_version": "",
    "source_hash": "",
    "md_hash": "",
    "skip_import": False,
    "skip_reason": "",
    "previous_document": {},
    "previous_chunks": [],
    "added_chunks": [],
    "updated_chunks": [],
    "deleted_chunks": [],
    "unchanged_chunks": [],
    "all_chunks": [],
    "import_summary": {},
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    # 每次都深拷贝，避免多个任务共享同一份默认列表/字典对象。
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    state = create_default_state(local_file_path="doc.pdf")
    logger.info(state)

