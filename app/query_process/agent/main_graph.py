"""
查询流程主图定义模块（Query Main Graph）

功能概述：
    定义查询流程的 LangGraph 状态图，编排"规划→检索→融合→反思→生成"全链路节点与路由逻辑。
    编译后生成可执行的 query_app 实例，供 query_service.py 调用。

流程拓扑（核心执行路径）：
    node_planner(规划) → node_item_name_confirm(商品确认)
        ├─ [有答案] → node_answer_output(直接输出澄清/拒答)
        └─ [无答案] → node_tool_router(工具路由) → node_multi_search(虚拟分叉点)
            ├─ node_search_embedding(向量检索)
            ├─ node_search_embedding_hyde(HyDE检索)
            ├─ node_web_search_mcp(网络搜索)
            └─ node_query_kg(图谱查询)
                ↓ (四路合并至 node_join)
            node_rrf(RRF融合) → node_rerank(重排序) → node_evidence_reflection(证据反思)
                ├─ [证据充足] → node_answer_output(生成答案)
                └─ [证据不足且未超轮次] → node_dynamic_reretrieval(补检索) → node_tool_router(重新路由)

关键设计：
    - 虚拟节点：node_multi_search/node_join 用于组织多路并发搜索的分叉与合并，无业务逻辑
    - 条件路由：route_after_item_confirm 控制是否跳过检索直接输出
    - 动态补检索：route_after_reflection 控制证据不足时是否进入下一轮检索
    - 最大检索轮次：默认2轮，防止无限循环
"""
from langgraph.graph import END, StateGraph

from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_planner import node_planner
from app.query_process.agent.nodes.node_tool_router import node_tool_router
from app.core.logger import logger
from app.query_process.agent.nodes.node_evidence_reflection import node_evidence_reflection
from app.query_process.agent.nodes.node_dynamic_retrieval import node_dynamic_reretrieval


# ==================== Java 开发者阅读提示 ====================
# 1. StateGraph(QueryGraphState)
#    可以把它理解成“一个带状态的流程编排器”。
#    类比 Java 时，更像工作流引擎里的流程定义，或者“状态机 + 多个处理节点”。
#
# 2. builder.add_node("node_xxx", func)
#    注册一个节点。节点执行时，本质上就是调用 func(state)。
#
# 3. builder.add_edge("A", "B")
#    固定流转：A 执行完后一定进入 B。
#
# 4. builder.add_conditional_edges("A", route_func)
#    条件流转：A 执行完后，先执行 route_func(state)，再决定下一跳。
#    你可以把它类比成 Java 里的 if/else 分支路由。
#
# 5. lambda x: x
#    Python 匿名函数，含义是“输入什么就原样返回什么”。
#    这里主要用于占位节点/虚拟节点，本身不做业务计算。
#
# 6. END
#    LangGraph 提供的结束标记；流程走到这里就算完成。
# ===========================================================




# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
builder.add_node("node_planner", node_planner)  # 规划节点：先决定怎么查，再进入后续流程
builder.add_node("node_item_name_confirm", node_item_name_confirm)  # 确认商品
builder.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点
builder.add_node("node_search_embedding", node_search_embedding)  # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_query_kg", node_query_kg)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
builder.add_node("node_rrf", node_rrf)  # 排序
builder.add_node("node_rerank", node_rerank)  # 重排
builder.add_node("node_answer_output", node_answer_output)  # 生成
builder.add_node("node_tool_router", node_tool_router)  # 工具路由节点：把 Planner 计划转成执行开关
builder.add_node("node_evidence_reflection", node_evidence_reflection)  # 证据反思节点：检索后先判断证据是否足够
builder.add_node("node_dynamic_reretrieval", node_dynamic_reretrieval)  # 补检索节点：证据不足时准备下一轮检索


# 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑；
# lambda x:x 含义：接收 state 并原样返回，是最轻便的 “无逻辑传递” 方式；
# 普通函数替换：定义 def 函数名(state): return state 即可完全等价，优势是易扩展、易调试；

# 设置起点
builder.set_entry_point("node_planner")



def route_after_item_confirm(state: QueryGraphState):
    """
    商品确认节点执行完后的分支函数。

    这里的判断逻辑是：
    1. 如果 item_confirm 阶段已经直接写出了 answer，
       说明这次请求不应该继续进入检索链路。
       常见场景：
       - 需要澄清：例如候选商品有多个，需要反问用户
       - 直接拒答：例如商品完全找不到，没有必要继续检索
    2. 如果还没有 answer，才继续进入 tool_router -> retrieval 链路。

    Python 语法说明：
    - state.get("answer")
      从 dict 中取值；如果 key 不存在，返回 None
    - bool(x)
      Python 内置函数，把任意对象转成布尔值：
      None / "" / 0 / [] / {} 会变成 False
      非空字符串、非空列表等会变成 True
    """
    # 路由步骤：
    # 1. 看 item_name_confirm 是否已经在 state["answer"] 中写了澄清/拒答内容
    # 2. 如果已经有 answer，本轮就不需要再继续检索，直接去输出节点
    # 3. 如果没有 answer，说明商品名已确认或允许全库召回，继续进入 tool_router
    # 如果已有答案（Branch B/C），直接跳到输出
    logger.info(
        f"item_confirm 后路由判断: "
        f"has_answer={bool(state.get('answer'))}, "
        f"item_names={state.get('item_names')}, "
        f"task_type={state.get('task_type')}, "
        f"need_clarify={state.get('need_clarify')}"
    )
    if state.get("answer"):
        """
        这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
        具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
        1. 多选一（反问用户） ：
        - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，且置信度都不足以直接确认。
        - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
        - 结果 ：此时不需要再去检索文档了，直接把这句话发给用户让他选。
        2. 查无此人（拒绝回答） ：

        - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，但库里只有华为的数据），或者评分过低（<0.6）。
        - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
        - 结果 ：同样不需要后续检索，直接结束流程。
        """
        return "node_answer_output"
    # 否则继续搜索流程
    return "node_tool_router"


def route_after_reflection(state: QueryGraphState):
    """
    Reflection 之后的路由函数。

    可以把它理解成：
    - 如果“证据已经足够”，就直接去最终答案节点
    - 如果“证据还不够”，并且补检索次数没超限，就再补一轮检索

    这相当于工作流里的一个 if/else 判断节点。
    """
    """
    Reflection 之后的路由判断：
    - sufficient / unknown -> 直接生成答案
    - insufficient / conflicting 且未超过最大轮次 -> 动态补检索
    - insufficient / conflicting 但已到上限 -> 直接生成答案
    """
    # 路由步骤：
    # 1. 读取证据状态 evidence_status
    # 2. 读取当前检索轮次 retrieval_round 与最大轮次 max_retrieval_rounds
    # 3. 只有“证据不足/冲突”且“还有下一轮机会”时，才进入动态补检索
    # 4. 其他情况都进入最终答案输出节点
    # get(key, default) 相当于 Java 的 Map#getOrDefault。
    evidence_status = state.get("evidence_status", "unknown")
    # int(...) 是 Python 内置类型转换函数，这里用于保证轮次字段一定是整数。
    retrieval_round = int(state.get("retrieval_round", 1))
    max_retrieval_rounds = int(state.get("max_retrieval_rounds", 2))

    logger.info(
        f"reflection 后路由判断: "
        f"evidence_status={evidence_status}, "
        f"retrieval_round={retrieval_round}, "
        f"max_retrieval_rounds={max_retrieval_rounds}"
    )

    # “in { ... }” 是成员判断，可类比 Java 的 Set.contains(...)。
    # 这里的意思是：只有“证据不足/冲突”且“补检索次数还没超限”时，才继续补检索。
    if evidence_status in {"insufficient", "conflicting"} and retrieval_round < max_retrieval_rounds:
        return "node_dynamic_reretrieval"

    return "node_answer_output"


builder.add_edge("node_planner", "node_item_name_confirm")
builder.add_edge("node_tool_router", "node_multi_search")



# 1. 意图确认 -> (条件分叉) -> 多路搜索 / 答案输出
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm
)

# 2. 并发执行四路搜索
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")
builder.add_edge("node_multi_search", "node_query_kg")

# 3. 四路搜索 -> 结果合并
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 4. 合并 -> 排序 -> 重排 -> 生成 -> 结束
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_evidence_reflection")
builder.add_conditional_edges(
    "node_evidence_reflection",
    route_after_reflection
)
builder.add_edge("node_dynamic_reretrieval", "node_tool_router")
builder.add_edge("node_answer_output", END)

# 编译生成可执行的 Runnable 应用
query_app = builder.compile()

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/agent/state.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
