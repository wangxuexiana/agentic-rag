from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_diff_chunks import node_diff_chunks
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState


load_dotenv()

# 导入主图的关键变化：
# 以前是 “切分 -> 识别 -> 向量化 -> 全量入库”
# 现在变成 “切分 -> 识别 -> diff -> 只对变化部分向量化/入库”
workflow = StateGraph(ImportGraphState)

workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_diff_chunks", node_diff_chunks)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

workflow.set_entry_point("node_entry")


def route_after_entry(state: ImportGraphState) -> str:
    # 入口处就允许直接短路，避免未变化文档还走一遍 PDF 解析和切分。
    if state.get("skip_import"):
        return END
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    return END


def route_after_diff(state: ImportGraphState) -> str:
    # diff 之后有 3 种情况：
    # 1. 有新增/修改 chunk：先做 embedding，再入库
    # 2. 只有 deleted chunk：直接入库节点做删除
    # 3. 什么都没变：结束
    if state.get("skip_import"):
        return END
    if state.get("chunks"):
        return "node_bge_embedding"
    if state.get("deleted_chunks"):
        return "node_import_milvus"
    return END


workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)

workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
# 新增 diff 节点，放在 item_name 之后。
# 原因：chunk 元数据里需要包含 item_name，一起进入快照和 Milvus。
workflow.add_edge("node_item_name_recognition", "node_diff_chunks")
workflow.add_conditional_edges(
    "node_diff_chunks",
    route_after_diff,
    {
        "node_bge_embedding": "node_bge_embedding",
        "node_import_milvus": "node_import_milvus",
        END: END,
    },
)
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

kb_import_app = workflow.compile()
