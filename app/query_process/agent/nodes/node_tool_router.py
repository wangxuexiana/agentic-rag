"""
工具路由节点 (Tool Router Node)

本模块实现 Agentic RAG 查询流程中的「工具路由」环节。
它不做检索，只负责把 Planner 的「检索计划」翻译成「执行开关」。

核心逻辑：
- 读取 Planner 输出的 selected_tools 列表
- 将每个工具名映射为 state 中的布尔开关（run_embedding / run_hyde / run_kg / run_web_search）
- 对 selected_tools 做标准化校验（过滤非法工具名、去重）
- 如果 Planner 未给出工具但当前任务不是 clarification，默认启用 embedding + hyde 兜底

下游每个检索节点只需判断自己的 run_xxx 开关，即可决定是否执行。
这种设计将路由逻辑与执行逻辑解耦，便于后续演进到更复杂的动态分支。
"""

import sys

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.query_process.agent.tool_registry import (
    apply_tool_switches,
    enrich_tools_for_task,
    get_allowed_tools,
)
from app.utils.debug_trace_utils import append_trace_event
from app.utils.task_utils import add_done_task, add_running_task


# ==================== Java 开发者阅读提示 ====================
# 这个节点不做真正检索，只做"开关翻译"。
# Planner 给出的结果更偏"计划"：
# - selected_tools = ["embedding", "hyde"]
#
# Tool Router 会通过 ToolRegistry 批量翻译成布尔开关：
# - run_embedding = True
# - run_hyde = True
# - run_kg = False
# - run_web_search = False
#
# 这样后面的各个检索节点就只需要判断自己的 run_xxx 开关即可。
#
# 【解耦变更】: 不再硬编码 allowed_tools 和逐个手写 run_xxx 赋值。
# 改为通过 tool_registry.apply_tool_switches() 批量设置。
# 新增工具只需在 TOOL_REGISTRY 注册即可。
# ===========================================================


def _ensure_tool_list(selected_tools) -> list:
    """
    对 selected_tools 做一次轻量标准化。

    为什么这里还要做一次：
    1. Planner 可能输出空值或脏值
    2. 某些中间节点以后可能会修改 selected_tools
    3. Router 是真正执行前的最后一道关口，适合再兜底一次
    """
    allowed_tools = get_allowed_tools()

    if not isinstance(selected_tools, list):
        return []

    normalized = []
    for tool in selected_tools:
        if tool in allowed_tools and tool not in normalized:
            normalized.append(tool)

    return normalized


def node_tool_router(state: QueryGraphState) -> QueryGraphState:
    """
    Tool Router 节点：
    它不做检索，只负责把"计划"翻译成"执行开关"。

    这一版我们不直接做复杂的 LangGraph 动态边分叉，
    而是先把每个工具是否启用写入 state，让下游节点自己判断要不要执行。

    这样做的好处：
    1. 路由逻辑已经成立
    2. 图结构改动小
    3. 便于逐步演进到更复杂的动态分支
    4. 【新增】通过 ToolRegistry 解耦，新工具一键注册
    """
    logger.info("--- node_tool_router 开始执行 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    task_type = state.get("task_type", "full_agentic")
    need_clarify = bool(state.get("need_clarify", False))
    selected_tools = _ensure_tool_list(state.get("selected_tools", []))

    # 如果 Planner 没给出工具，但当前任务又不是 clarification，
    # 那么给一个稳定的兜底，避免后续所有检索都空跑。
    if not selected_tools and task_type != "clarification":
        selected_tools = ["embedding", "hyde"]

    # 根据 task_type 补充默认工具（如 kb_with_web 自动加 web_search）
    selected_tools = enrich_tools_for_task(selected_tools, task_type)

    # 通过 ToolRegistry 批量设置所有 run_xxx 开关
    apply_tool_switches(state, selected_tools)

    # 给调试和后续 reflection 留一段文字说明
    if need_clarify:
        state["router_reason"] = "Planner 判定当前问题需要澄清，默认不主动扩展工具检索。"
    elif state.get("selected_tools"):
        state["router_reason"] = f"根据 Planner 结果启用工具: {state['selected_tools']}"
    else:
        state["router_reason"] = "未启用任何工具。"

    logger.info(
        f"Router 执行完成: "
        f"task_type={task_type}, "
        f"need_clarify={need_clarify}, "
        f"selected_tools={state.get('selected_tools')}, "
        f"run_embedding={state.get('run_embedding')}, "
        f"run_hyde={state.get('run_hyde')}, "
        f"run_kg={state.get('run_kg')}, "
        f"run_web_search={state.get('run_web_search')}"
    )

    append_trace_event(
        session_id=state["session_id"],
        node="node_tool_router",
        retrieval_round=int(state.get("retrieval_round", 1)),
        payload={
            "task_type": task_type,
            "selected_tools": state.get("selected_tools"),
            "run_embedding": state.get("run_embedding"),
            "run_hyde": state.get("run_hyde"),
            "run_kg": state.get("run_kg"),
            "run_web_search": state.get("run_web_search"),
            "router_reason": state.get("router_reason", ""),
        },
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

# ──────────────────────────────────────────────────────────
# 阅读导航
# 下一篇: app/query_process/agent/nodes/node_search_embedding.py
# 总导航: doc/代码阅读顺序.md
# ──────────────────────────────────────────────────────────
