"""
Planner 节点。

本次继续保留“节点只负责编排、service 负责规则”的边界，
并在节点层先接入第一层查询缓存：planner cache。
"""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.services import planner_service, query_cache_service
from app.query_process.agent.state import QueryGraphState
from app.utils.debug_trace_utils import append_trace_event
from app.utils.task_utils import add_done_task, add_running_task


def _build_history_text(history: list) -> str:
    return planner_service.build_history_text(history)


def _build_fallback_plan(query: str) -> dict:
    return planner_service.build_fallback_plan(query)


def _normalize_selected_tools(selected_tools, task_type: str) -> list:
    return planner_service.normalize_selected_tools(selected_tools, task_type)


def _parse_planner_output(raw_text: str, query: str) -> dict:
    return planner_service.parse_planner_output(raw_text, query, logger)


def _apply_plan_to_state(state: QueryGraphState, plan: dict) -> None:
    state["intent_type"] = plan.get("intent_type", "unknown")
    state["task_type"] = plan.get("task_type", "full_agentic")
    state["retrieval_plan"] = plan
    state["selected_tools"] = _normalize_selected_tools(
        plan.get("selected_tools"),
        plan.get("task_type", "full_agentic"),
    )
    state["need_clarify"] = bool(plan.get("need_clarify", False))

    if state["need_clarify"]:
        state["clarification_question"] = planner_service.build_clarification_question(
            state["intent_type"]
        )
    else:
        state["clarification_question"] = ""


def node_planner(state: QueryGraphState) -> QueryGraphState:
    """
    Planner 主流程：
    1. 读取 query、history、item_names
    2. 先尝试命中 planner cache
    3. 未命中时再调 LLM
    4. 将计划写回 state
    """
    logger.info("--- node_planner 开始执行 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "") or original_query
    history = state.get("history", [])
    item_names = state.get("item_names", [])
    doc_scope = ",".join(sorted(str(name).strip() for name in item_names if str(name).strip()))

    cached_plan = query_cache_service.get_planner_cache(
        query=rewritten_query,
        history=history,
        item_names=item_names,
        doc_scope=doc_scope,
    )

    if cached_plan:
        logger.info("Planner 命中缓存，跳过 LLM 调用")
        plan = cached_plan
        cache_hit = True
    else:
        cache_hit = False
        history_text = _build_history_text(history)
        planner_prompt = planner_service.build_planner_prompt(history_text, rewritten_query)

        try:
            llm = get_llm_client(json_mode=True)
            messages = [
                SystemMessage(content="你是一个严谨的 Agentic RAG Planner，只输出 JSON。"),
                HumanMessage(content=planner_prompt),
            ]

            logger.info("Planner 正在调用 LLM 生成检索计划")
            response = llm.invoke(messages)
            raw_text = response.content

            logger.debug(f"Planner 原始输出: {raw_text}")
            plan = _parse_planner_output(raw_text, rewritten_query)
            logger.info(f"Planner 纠偏后的最终计划: {plan}")
        except Exception as exc:
            logger.error(f"Planner 调用失败，进入 fallback: {exc}", exc_info=True)
            plan = _build_fallback_plan(rewritten_query)

        query_cache_service.set_planner_cache(
            query=rewritten_query,
            history=history,
            item_names=item_names,
            plan=plan,
            doc_scope=doc_scope,
        )

    query_cache_service.record_stage_cache_result(
        state,
        stage="planner",
        cache_hit=cache_hit,
        detail={"query": rewritten_query, "doc_scope": doc_scope},
    )
    _apply_plan_to_state(state, plan)

    logger.info(
        f"Planner 执行完成: intent={state['intent_type']}, "
        f"task_type={state['task_type']}, "
        f"selected_tools={state['selected_tools']}, "
        f"need_clarify={state['need_clarify']}, "
        f"cache_hit={cache_hit}"
    )

    append_trace_event(
        session_id=state["session_id"],
        node="node_planner",
        retrieval_round=int(state.get("retrieval_round", 1)),
        payload={
            "intent_type": state["intent_type"],
            "task_type": state["task_type"],
            "selected_tools": state["selected_tools"],
            "need_clarify": state["need_clarify"],
            "planner_cache_hit": cache_hit,
            "success_criteria": plan.get("success_criteria", ""),
            "notes": plan.get("notes", ""),
        },
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
