"""
答案输出节点。

职责：
1. 收口整个查询链，生成最终答案
2. 在答案生成前接入 answer cache
3. 写回 task result、trace、历史记录和 SSE final 事件
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from app.clients.mongo_history_utils import save_chat_message
from app.core.logger import logger
from app.query_process.agent.services import answer_output_service, query_cache_service
from app.query_process.agent.state import QueryGraphState
from app.utils.debug_trace_utils import append_trace_event
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import add_done_task, add_running_task, set_task_result


def _normalize_doc_dicts(docs, field_name: str):
    return answer_output_service.normalize_doc_dicts(docs, field_name, logger)


def _format_evidence_status_label(evidence_status: str) -> str:
    return answer_output_service.format_evidence_status_label(evidence_status)


def _collect_display_citations(state: QueryGraphState, max_items: int = 3) -> List[Dict[str, Any]]:
    return answer_output_service.collect_display_citations(state, logger, max_items=max_items)


def _normalize_match_text(text: str) -> str:
    return answer_output_service.normalize_match_text(text)


def _count_aligned_local_docs(state: QueryGraphState) -> int:
    return answer_output_service.count_aligned_local_docs(state, logger)


def _should_force_abstain(state: QueryGraphState) -> bool:
    return answer_output_service.should_force_abstain(state, logger)


def _build_insufficient_evidence_answer(state: QueryGraphState) -> str:
    return answer_output_service.build_insufficient_evidence_answer(state, logger)


def step_1_check_answer(state: QueryGraphState) -> bool:
    """如果上游节点已经产出了答案，就直接复用。"""
    return answer_output_service.check_existing_answer(state, logger)


def step_2_construct_prompt(state: QueryGraphState) -> str:
    """组装最终回答 prompt。"""
    return answer_output_service.construct_answer_prompt(state, logger)


def step_3_generate_response(state: QueryGraphState, prompt: str) -> str:
    """调用 LLM 生成答案文本。"""
    return answer_output_service.generate_response(state, prompt, logger)


def _append_answer_meta(state: QueryGraphState) -> QueryGraphState:
    """给答案追加证据状态、引用说明等展示信息。"""
    state["answer"] = answer_output_service.append_answer_meta(state, logger)
    return state


def _extract_images_from_docs(docs: List[Dict[str, Any]]) -> List[str]:
    return answer_output_service.extract_images_from_docs(docs, logger)


def _normalize_answer_images(answer: str, image_urls: List[str]) -> str:
    return answer_output_service.normalize_answer_images(answer, image_urls)


def step_4_write_history(state: QueryGraphState, image_urls: List[str] | None = None) -> QueryGraphState:
    """把 assistant 的最终回答写回历史。"""
    session_id = state.get("session_id")
    answer = (state.get("answer") or "").strip()
    if not session_id or not answer:
        return state

    save_chat_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query=state.get("rewritten_query") or "",
        item_names=state.get("item_names") or [],
        image_urls=image_urls or [],
    )
    return state


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    最终答案输出主流程。

    answer cache 策略：
    - 命中时直接复用最终答案和 prompt，不再重新组 prompt / 调 LLM
    - 未命中时正常生成答案，并在生成后把最终答案写入缓存
    """
    session_id = state.get("session_id")
    node_name = sys._getframe().f_code.co_name
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, node_name, is_stream)
    logger.info("----node_answer_output----start")

    answer_exists = step_1_check_answer(state)
    if answer_exists:
        query_cache_service.record_stage_cache_result(
            state,
            stage="answer",
            cache_hit=False,
            detail={"skipped": True, "reason": "upstream_answer_exists"},
        )
    else:
        answer_cache_query = state.get("original_query") or state.get("rewritten_query") or ""
        cached_answer = query_cache_service.get_answer_cache(
            query=answer_cache_query,
            item_names=state.get("item_names") or [],
            reranked_docs=state.get("reranked_docs") or [],
            evidence_status=state.get("evidence_status") or "unknown",
        )
        if cached_answer:
            state["answer"] = cached_answer.get("answer") or ""
            state["prompt"] = cached_answer.get("prompt") or ""
            query_cache_service.record_stage_cache_result(
                state,
                stage="answer",
                cache_hit=True,
                detail={
                    "has_prompt": bool(state.get("prompt")),
                    "answer_length": len(state.get("answer") or ""),
                },
            )
        else:
            query_cache_service.record_stage_cache_result(
                state,
                stage="answer",
                cache_hit=False,
                detail={},
            )

            guarded_answer = _build_insufficient_evidence_answer(state)
            if guarded_answer:
                state["answer"] = guarded_answer
                set_task_result(session_id, "answer", guarded_answer)
            else:
                prompt = step_2_construct_prompt(state)
                state["prompt"] = prompt
                generated = step_3_generate_response(state, prompt)
                # 兼容旧 service 返回整个 state 的历史行为，避免把 dict 写进 answer。
                if isinstance(generated, dict):
                    state = generated
                else:
                    state["answer"] = generated

            state = _append_answer_meta(state)
            image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
            state["answer"] = _normalize_answer_images(state.get("answer") or "", image_urls)
            query_cache_service.set_answer_cache(
                query=answer_cache_query,
                item_names=state.get("item_names") or [],
                reranked_docs=state.get("reranked_docs") or [],
                evidence_status=state.get("evidence_status") or "unknown",
                answer_result={
                    "answer": state.get("answer") or "",
                    "prompt": state.get("prompt") or "",
                },
            )

    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
    if state.get("answer"):
        state["answer"] = _normalize_answer_images(state.get("answer") or "", image_urls)

    set_task_result(session_id, "answer", state.get("answer") or "")
    set_task_result(session_id, "cache_stats", json.dumps(state.get("cache_stats", {}), ensure_ascii=False))

    append_trace_event(
        session_id=session_id,
        node=node_name,
        payload={
            "evidence_status": state.get("evidence_status") or "",
            "final_confidence": state.get("final_confidence"),
            "citations_count": len(state.get("citations") or []),
            "image_count": len(image_urls),
            "cache_stats": state.get("cache_stats", {}),
            "answer_cache_hit": bool(
                ((state.get("cache_stats") or {}).get("stages") or {}).get("answer", {}).get("cache_hit")
            ),
            "answer_preview": (state.get("answer") or "")[:300],
        },
        retrieval_round=state.get("retrieval_round", 1),
    )

    step_4_write_history(state, image_urls=image_urls)

    if is_stream and session_id:
        push_to_session(session_id, SSEEvent.FINAL, {"answer": state.get("answer") or ""})

    add_done_task(session_id, node_name, is_stream)
    logger.info("----node_answer_output----end")
    return state
