"""
实体确认节点。

职责：
1. 提取 query 中的候选实体名并重写 query
2. 对候选实体做向量召回与标准实体对齐
3. 决定是直接确认实体、返回澄清问题，还是放行到全库检索
4. 为节点结果增加一层缓存，避免相同 query 重复做实体确认
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.core.logger import logger
from app.query_process.agent.services import item_name_confirm_service, query_cache_service
from app.query_process.agent.state import QueryGraphState
from app.utils.debug_trace_utils import append_trace_event
from app.utils.task_utils import add_done_task, add_running_task

load_dotenv()


def step_3_extract_info(query: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """调用 service 抽取实体名并重写 query。"""
    return item_name_confirm_service.extract_query_info(query, history, logger)


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict[str, Any]]:
    """对候选实体名做向量化并到实体集合中召回标准候选。"""
    return item_name_confirm_service.vectorize_and_query_item_names(item_names, logger)


def step_5_align_item_names(query_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """把召回结果整理成已确认实体和待澄清实体两类。"""
    return item_name_confirm_service.align_item_names(query_results, logger)


def step_6_check_confirmation(
    state: QueryGraphState,
    item_results: Dict[str, List[str]],
    history_chats: List[Dict[str, Any]],
    rewritten_query: str,
    raw_item_names: List[str],
) -> QueryGraphState:
    """根据实体对齐结果，把确认状态写回图状态。"""
    return item_name_confirm_service.apply_confirmation_result(
        state=state,
        align_result=item_results,
        history=history_chats,
        rewritten_query=rewritten_query,
        raw_item_names=raw_item_names,
        logger=logger,
    )


def step_7_write_history(
    state: QueryGraphState,
    session_id: str,
    rewritten_query: str,
    message_id: str,
) -> QueryGraphState:
    """把本轮节点产出的 assistant/user 信息写回历史。"""
    answer = (state.get("answer") or "").strip()
    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            rewritten_query="",
            item_names=state.get("item_names") or [],
        )

    save_chat_message(
        session_id=session_id,
        role="user",
        text=state.get("original_query") or "",
        rewritten_query=rewritten_query,
        item_names=state.get("item_names") or [],
        message_id=message_id,
    )
    return state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    实体确认主流程。

    缓存策略：
    - 命中时直接复用 `rewritten_query / item_names / answer`
    - 未命中时执行原有实体抽取和对齐流程，并把结果写入缓存
    """
    session_id = state.get("session_id")
    node_name = sys._getframe().f_code.co_name
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, node_name, is_stream)
    logger.info("----node_item_name_confirm----start")

    original_query = state.get("original_query") or ""
    history = get_recent_messages(session_id, limit=10)
    planner_requires_clarify = bool(state.get("need_clarify")) or state.get("task_type") == "clarification"

    cached_result = query_cache_service.get_item_name_confirm_cache(
        query=original_query,
        history=history,
        task_type=state.get("task_type") or "",
        need_clarify=planner_requires_clarify,
    )
    if cached_result:
        state["rewritten_query"] = cached_result.get("rewritten_query") or original_query
        state["item_names"] = list(cached_result.get("item_names") or [])
        state["answer"] = cached_result.get("answer") or ""
        state["history"] = history

        query_cache_service.record_stage_cache_result(
            state,
            stage="item_name_confirm",
            cache_hit=True,
            detail={
                "item_name_count": len(state.get("item_names") or []),
                "has_answer": bool(state.get("answer")),
            },
        )
        append_trace_event(
            session_id=session_id,
            node=node_name,
            payload={
                "original_query": original_query,
                "rewritten_query": state.get("rewritten_query") or "",
                "confirmed_item_names": state.get("item_names") or [],
                "item_name_confirm_cache_hit": True,
                "answer_preview": (state.get("answer") or "")[:200],
            },
            retrieval_round=state.get("retrieval_round", 1),
        )

        add_done_task(session_id, node_name, is_stream)
        logger.info("----node_item_name_confirm----end (cache hit)")
        return state

    # 未命中时，沿用原逻辑，先记一条原始 user 消息，后面再补回 rewritten_query/item_names。
    message_id = save_chat_message(
        session_id=session_id,
        role="user",
        text=original_query,
        rewritten_query="",
        item_names=[],
    )

    extract_result = step_3_extract_info(original_query, history)
    rewritten_query = extract_result.get("rewritten_query") or original_query
    raw_item_names = extract_result.get("item_names") or []
    state["rewritten_query"] = rewritten_query

    item_results: Dict[str, List[str]] = {
        "confirmed_item_names": [],
        "options_item_names": [],
    }
    if raw_item_names and not planner_requires_clarify:
        query_milvus_results = step_4_vectorize_and_query(raw_item_names)
        item_results = step_5_align_item_names(query_milvus_results)

    state = step_6_check_confirmation(
        state=state,
        item_results=item_results,
        history_chats=history,
        rewritten_query=rewritten_query,
        raw_item_names=raw_item_names,
    )

    query_cache_service.set_item_name_confirm_cache(
        query=original_query,
        history=history,
        task_type=state.get("task_type") or "",
        need_clarify=planner_requires_clarify,
        result={
            "rewritten_query": state.get("rewritten_query") or "",
            "item_names": state.get("item_names") or [],
            "answer": state.get("answer") or "",
        },
    )
    query_cache_service.record_stage_cache_result(
        state,
        stage="item_name_confirm",
        cache_hit=False,
        detail={
            "item_name_count": len(state.get("item_names") or []),
            "has_answer": bool(state.get("answer")),
        },
    )

    append_trace_event(
        session_id=session_id,
        node=node_name,
        payload={
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "raw_item_names": raw_item_names,
            "planner_requires_clarify": planner_requires_clarify,
            "confirmed_item_names": state.get("item_names") or [],
            "item_name_confirm_cache_hit": False,
            "answer_preview": (state.get("answer") or "")[:200],
        },
        retrieval_round=state.get("retrieval_round", 1),
    )

    final_state = step_7_write_history(
        state=state,
        session_id=session_id,
        rewritten_query=rewritten_query,
        message_id=message_id,
    )
    final_state["history"] = history

    add_done_task(session_id, node_name, is_stream)
    logger.info("----node_item_name_confirm----end")
    return final_state
