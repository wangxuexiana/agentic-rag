import hashlib
import os
import sys
from os.path import splitext

from app.clients.document_registry import get_document_snapshot
from app.core.logger import logger
from app.import_process.agent.import_summary import build_import_summary
from app.import_process.agent.state import ImportGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_done_task, add_running_task


ALLOWED_IMPORT_SUFFIXES = {".pdf", ".md"}


def node_entry(state: ImportGraphState) -> ImportGraphState:
    # 入口节点新增了两件增量同步相关的事：
    # 1. 生成稳定 doc_id
    # 2. 计算源文件 hash，并和历史快照对比，决定是否可以直接跳过
    func_name = sys._getframe().f_code.co_name
    logger.debug(f"[{func_name}] start\nstate={format_state(state)}")
    add_running_task(state["task_id"], func_name)

    # 先做最基础的文件存在性校验。
    document_path = state.get("local_file_path", "")
    if not document_path:
        raise ValueError("local_file_path is required")
    if not os.path.exists(document_path):
        raise FileNotFoundError(document_path)

    # doc_id 基于绝对路径生成，保证同一路径反复导入时身份稳定。
    normalized_path = os.path.abspath(document_path)
    external_doc_id = (state.get("external_doc_id") or "").strip()
    if external_doc_id:
        # 预留更稳定的业务主键入口。
        # 当上游已经有“同一文档”的外部 ID 时，直接优先使用它。
        state["doc_id"] = external_doc_id
    else:
        state["doc_id"] = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()
    # source_hash 是“这个源文件当前版本”的指纹。
    state["source_hash"] = _hash_file(normalized_path)
    state["doc_version"] = state["source_hash"]

    # 仍然保留原来的 PDF / MD 分流逻辑。
    suffix = splitext(document_path)[1].lower()
    if suffix == ".pdf":
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = document_path
    elif suffix == ".md":
        state["is_md_read_enabled"] = True
        state["md_path"] = document_path
        # MD 输入可以直接读内容并记录 md_hash，后续节点无需再从文件回读。
        with open(document_path, "r", encoding="utf-8") as file_obj:
            state["md_content"] = file_obj.read()
        state["md_hash"] = hashlib.sha1(state["md_content"].encode("utf-8")).hexdigest()
    else:
        raise ValueError(f"unsupported file type: {suffix}, allowed={sorted(ALLOWED_IMPORT_SUFFIXES)}")

    file_name = os.path.basename(document_path)
    state["file_title"] = splitext(file_name)[0]

    # 如果源文件 hash 没变，说明这份文档没有任何更新，可以整条导入链直接短路。
    previous_document, _ = get_document_snapshot(state["doc_id"])
    if previous_document and previous_document.get("source_hash") == state["source_hash"]:
        state["skip_import"] = True
        state["skip_reason"] = "source_hash_unchanged"
        state["import_summary"] = build_import_summary(state, phase=func_name)
        logger.info(f"[{func_name}] source unchanged, skip import: {state['file_title']}")
    else:
        state["skip_reason"] = ""

    add_done_task(state["task_id"], func_name)
    logger.debug(f"[{func_name}] done\nstate={format_state(state)}")
    return state


def _hash_file(path: str) -> str:
    # 按块读取，避免大文件一次性读入内存。
    digest = hashlib.sha1()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
