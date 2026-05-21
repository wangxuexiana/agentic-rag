"""
知识图谱查询节点 (Knowledge Graph Query Node)

本模块实现 Agentic RAG 查询流程中的「知识图谱查询」环节。
当前阶段为预留占位实现，仅当 Router 启用 KG 工具时才执行。

后续计划：
- 对接 Neo4j 图数据库，基于实体关系进行结构化查询
- 支持设备间的拓扑关系、上下游依赖等关系型问答
- 与 Milvus 向量检索互补，覆盖结构化知识需求

当前状态：骨架实现，直接返回空结果（kg_chunks=[]）
"""

import time
import sys
from app.utils.task_utils import add_running_task, add_done_task


# ==================== Java 开发者阅读提示 ====================
# 这是 KG（知识图谱）查询节点。
# 但按当前项目真实状态，它仍然是“占位实现”：
# - 路由层可以把请求导到这里
# - 但真正的图谱查询逻辑还没有落地
#
# 所以你读这段代码时，要把它理解成“流程占位”，不是最终实现。
# ===========================================================

def node_query_kg(state):
    """
    节点功能：在 Neo4j 知识图谱中查询实体关系。
    """
    # 当前说明：
    # 1. 这是图谱检索支路的占位节点
    # 2. 主图已预留 run_kg 开关与 kg_chunks 输出
    # 3. 但当前版本没有真正的 Neo4j 查询逻辑
    # 4. 因此即便进入本节点，最终也只返回空结果
    print("=== node_query_kg 图谱查询处理 ===")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 如果 Router 没有启用 KG 工具，这个节点直接跳过
    if not state.get("run_kg", False):
        print("Router 未启用 KG 检索，当前节点跳过执行。")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"kg_chunks": []}


    time.sleep(1)
    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"kg_chunks": []}
