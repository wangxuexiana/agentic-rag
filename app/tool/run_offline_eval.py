"""
Offline eval runner for the local query pipeline.

Usage:
    python app/tool/run_offline_eval.py <dataset.jsonl> [output.jsonl]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.query_process.api.query_service import run_query_graph


TRACE_PATH = Path("logs") / "agent_trace.jsonl"
DEFAULT_OUTPUT_DIR = Path("test") / "offline_eval"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"invalid jsonl at line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"line {lineno} is not a json object")
            records.append(obj)
    return records


def read_trace_events(session_id: str) -> List[Dict[str, Any]]:
    if not TRACE_PATH.exists():
        return []

    matched: List[Dict[str, Any]] = []
    with TRACE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except Exception:
                continue
            if event.get("session_id") == session_id:
                matched.append(event)

    matched.sort(key=lambda x: (x.get("retrieval_round", 1), x.get("ts", "")))
    return matched


def summarize_trace(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_rounds: List[Dict[str, Any]] = []
    planner: Dict[str, Any] = {}
    item_confirm: Dict[str, Any] = {}
    embedding_retrieval: Dict[str, Any] = {}
    hyde_retrieval: Dict[str, Any] = {}
    rrf_summary: Dict[str, Any] = {}
    rerank_summary: Dict[str, Any] = {}
    final_reflection: Dict[str, Any] = {}
    final_answer: Dict[str, Any] = {}
    dynamic_retry: Dict[str, Any] = {}

    for event in events:
        node = event.get("node")
        payload = event.get("payload") or {}
        retrieval_round = int(event.get("retrieval_round", 1) or 1)

        if node == "node_planner":
            planner = {
                "retrieval_round": retrieval_round,
                "intent_type": payload.get("intent_type", "unknown"),
                "task_type": payload.get("task_type", "unknown"),
                "selected_tools": payload.get("selected_tools", []),
                "need_clarify": bool(payload.get("need_clarify", False)),
                "success_criteria": payload.get("success_criteria", ""),
                "notes": payload.get("notes", ""),
            }
        elif node == "node_item_name_confirm":
            item_confirm = {
                "retrieval_round": retrieval_round,
                "extracted_item_names": payload.get("extracted_item_names", []),
                "confirmed_item_names": payload.get("confirmed_item_names", []),
                "candidate_options": payload.get("candidate_options", []),
                "final_item_names": payload.get("final_item_names", []),
                "confirmation_mode": payload.get("confirmation_mode", ""),
                "has_answer": bool(payload.get("has_answer", False)),
                "need_clarify": bool(payload.get("need_clarify", False)),
                "rewritten_query": payload.get("rewritten_query", ""),
            }
        elif node == "node_tool_router":
            tool_rounds.append(
                {
                    "retrieval_round": retrieval_round,
                    "selected_tools": payload.get("selected_tools", []),
                    "router_reason": payload.get("router_reason", ""),
                }
            )
        elif node == "node_search_embedding":
            embedding_retrieval = {
                "retrieval_round": retrieval_round,
                "query": payload.get("query", ""),
                "item_names": payload.get("item_names", []),
                "hit_count": payload.get("hit_count", 0),
                "top_chunk_ids": payload.get("top_chunk_ids", []),
            }
        elif node == "node_search_embedding_hyde":
            hyde_retrieval = {
                "retrieval_round": retrieval_round,
                "query": payload.get("query", ""),
                "item_names": payload.get("item_names", []),
                "hit_count": payload.get("hit_count", 0),
                "hyde_doc_length": payload.get("hyde_doc_length", 0),
                "top_chunk_ids": payload.get("top_chunk_ids", []),
            }
        elif node == "node_rrf":
            rrf_summary = {
                "retrieval_round": retrieval_round,
                "embedding_input_count": payload.get("embedding_input_count", 0),
                "hyde_input_count": payload.get("hyde_input_count", 0),
                "rrf_output_count": payload.get("rrf_output_count", 0),
                "top_chunk_ids": payload.get("top_chunk_ids", []),
            }
        elif node == "node_rerank":
            rerank_summary = {
                "retrieval_round": retrieval_round,
                "merged_doc_count": payload.get("merged_doc_count", 0),
                "reranked_doc_count": payload.get("reranked_doc_count", 0),
                "topk_doc_count": payload.get("topk_doc_count", 0),
                "top_chunk_ids": payload.get("top_chunk_ids", []),
                "top_scores": payload.get("top_scores", []),
            }
        elif node == "node_evidence_reflection":
            final_reflection = {
                "retrieval_round": retrieval_round,
                "evidence_status": payload.get("evidence_status", "unknown"),
                "final_confidence": payload.get("final_confidence", 0.0),
                "support_score": payload.get("support_score", 0.0),
                "coverage_score": payload.get("coverage_score", 0.0),
                "consistency_score": payload.get("consistency_score", 0.0),
                "reflection_reason": payload.get("reflection_reason", ""),
                "missing_facts": payload.get("missing_facts", []),
            }
        elif node == "node_dynamic_reretrieval":
            dynamic_retry = {
                "retrieval_round": retrieval_round,
                "followup_query": payload.get("followup_query", ""),
                "selected_tools": payload.get("selected_tools", []),
                "retry_intent": payload.get("retry_intent", ""),
                "missing_facts": payload.get("missing_facts", []),
            }
        elif node == "node_answer_output":
            final_answer = {
                "retrieval_round": retrieval_round,
                "evidence_status": payload.get("evidence_status", "unknown"),
                "final_confidence": payload.get("final_confidence", 0.0),
                "support_score": payload.get("support_score", 0.0),
                "coverage_score": payload.get("coverage_score", 0.0),
                "consistency_score": payload.get("consistency_score", 0.0),
                "answer_preview": payload.get("answer_preview", ""),
            }

    return {
        "planner": planner,
        "item_confirm": item_confirm,
        "tool_rounds": tool_rounds,
        "embedding_retrieval": embedding_retrieval,
        "hyde_retrieval": hyde_retrieval,
        "rrf_summary": rrf_summary,
        "rerank_summary": rerank_summary,
        "final_reflection": final_reflection,
        "dynamic_retry": dynamic_retry,
        "final_answer": final_answer,
        "retrieval_round_count": max([1] + [int(x.get("retrieval_round", 1) or 1) for x in events]),
    }


def build_session_id(case_id: str, idx: int) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_case_id = (case_id or f"case_{idx}").replace(" ", "_")
    return f"offline_eval_{safe_case_id}_{idx}_{ts}"


def normalize_case(case: Dict[str, Any], idx: int) -> Dict[str, Any]:
    case_id = str(case.get("case_id") or f"case_{idx}")
    query = str(case.get("query") or "").strip()
    if not query:
        raise ValueError(f"case[{idx}] missing query")

    return {
        "case_id": case_id,
        "category": case.get("category", ""),
        "query": query,
        "gold_answer": case.get("gold_answer", ""),
        "gold_item_names": case.get("gold_item_names", []),
        "expected_tools": case.get("expected_tools", []),
        "acceptable_first_round_tools_any_of": case.get("acceptable_first_round_tools_any_of", []),
        "expected_intent_type": case.get("expected_intent_type", ""),
        "acceptable_intent_types": case.get("acceptable_intent_types", []),
        "expected_task_type": case.get("expected_task_type", ""),
        "acceptable_task_types": case.get("acceptable_task_types", []),
        "expected_need_clarify": case.get("expected_need_clarify"),
        "expected_evidence_status": case.get("expected_evidence_status", ""),
        "expected_answer_mode": case.get("expected_answer_mode", ""),
        "expected_item_names": case.get("expected_item_names", []),
        "require_item_resolution": bool(case.get("require_item_resolution", False)),
        "expected_retry_tools": case.get("expected_retry_tools", []),
        "acceptable_retry_tools_any_of": case.get("acceptable_retry_tools_any_of", []),
        "require_second_round": bool(case.get("require_second_round", False)),
        "gold_chunk_ids": case.get("gold_chunk_ids", []),
        "require_retrieval_hit": bool(case.get("require_retrieval_hit", False)),
        "require_rerank_hit": bool(case.get("require_rerank_hit", False)),
        "min_final_confidence": case.get("min_final_confidence"),
        "max_final_confidence": case.get("max_final_confidence"),
        "min_ragas_faithfulness": case.get("min_ragas_faithfulness"),
        "min_ragas_response_relevancy": case.get("min_ragas_response_relevancy"),
        "min_ragas_context_recall": case.get("min_ragas_context_recall"),
        "min_ragas_context_precision": case.get("min_ragas_context_precision"),
        "must_have_keywords": case.get("must_have_keywords", []),
        "must_not_have_keywords": case.get("must_not_have_keywords", []),
        "should_abstain": bool(case.get("should_abstain", False)),
        "notes": case.get("notes", ""),
    }


def normalize_reranked_docs(reranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for doc in reranked_docs or []:
        if not isinstance(doc, dict):
            continue
        normalized.append(
            {
                "text": str(doc.get("text") or ""),
                "title": str(doc.get("title") or ""),
                "source": str(doc.get("source") or ""),
                "chunk_id": doc.get("chunk_id"),
                "doc_id": doc.get("doc_id"),
                "url": str(doc.get("url") or ""),
                "score": doc.get("score"),
            }
        )
    return normalized


def run_case(case: Dict[str, Any], idx: int) -> Dict[str, Any]:
    session_id = build_session_id(case["case_id"], idx)
    final_state = run_query_graph(session_id=session_id, user_query=case["query"], is_stream=False) or {}
    reranked_docs = normalize_reranked_docs(final_state.get("reranked_docs") or [])
    body = {
        "message": "处理完成",
        "session_id": session_id,
        "answer": str(final_state.get("answer") or ""),
        "cache_stats": final_state.get("cache_stats", {}),
    }

    events = read_trace_events(session_id)
    trace_summary = summarize_trace(events)

    result = {
        "case_id": case["case_id"],
        "category": case.get("category", ""),
        "session_id": session_id,
        "query": case["query"],
        "gold_answer": case["gold_answer"],
        "gold_item_names": case["gold_item_names"],
        "expected_tools": case["expected_tools"],
        "acceptable_first_round_tools_any_of": case.get("acceptable_first_round_tools_any_of", []),
        "expected_intent_type": case.get("expected_intent_type", ""),
        "acceptable_intent_types": case.get("acceptable_intent_types", []),
        "expected_task_type": case.get("expected_task_type", ""),
        "acceptable_task_types": case.get("acceptable_task_types", []),
        "expected_need_clarify": case.get("expected_need_clarify"),
        "expected_evidence_status": case.get("expected_evidence_status", ""),
        "expected_answer_mode": case.get("expected_answer_mode", ""),
        "expected_item_names": case.get("expected_item_names", []),
        "require_item_resolution": case.get("require_item_resolution", False),
        "expected_retry_tools": case.get("expected_retry_tools", []),
        "acceptable_retry_tools_any_of": case.get("acceptable_retry_tools_any_of", []),
        "require_second_round": case.get("require_second_round", False),
        "gold_chunk_ids": case.get("gold_chunk_ids", []),
        "require_retrieval_hit": case.get("require_retrieval_hit", False),
        "require_rerank_hit": case.get("require_rerank_hit", False),
        "min_final_confidence": case.get("min_final_confidence"),
        "max_final_confidence": case.get("max_final_confidence"),
        "min_ragas_faithfulness": case.get("min_ragas_faithfulness"),
        "min_ragas_response_relevancy": case.get("min_ragas_response_relevancy"),
        "min_ragas_context_recall": case.get("min_ragas_context_recall"),
        "min_ragas_context_precision": case.get("min_ragas_context_precision"),
        "must_have_keywords": case["must_have_keywords"],
        "must_not_have_keywords": case["must_not_have_keywords"],
        "should_abstain": case["should_abstain"],
        "notes": case["notes"],
        "http_status": 200,
        "response": body,
        "reranked_docs": reranked_docs,
        "retrieved_contexts": [doc["text"] for doc in reranked_docs if doc.get("text")],
        "trace_summary": trace_summary,
        "trace_events": events,
    }
    return result


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_output_path(dataset_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{stem}.run.{ts}.jsonl"


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python app/tool/run_offline_eval.py <dataset.jsonl> [output.jsonl]")
        return 1

    dataset_path = Path(argv[1])
    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}")
        return 1

    output_path = resolve_output_path(dataset_path, argv[2] if len(argv) > 2 else None)
    raw_cases = load_jsonl(dataset_path)
    cases = [normalize_case(case, idx) for idx, case in enumerate(raw_cases, start=1)]

    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] running case_id={case['case_id']}")
        try:
            result = run_case(case, idx)
        except Exception as exc:
            result = {
                "case_id": case["case_id"],
                "query": case["query"],
                "error": str(exc),
            }
        results.append(result)

    write_jsonl(output_path, results)
    print(f"offline eval run saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
