"""
Rule-based scorer for offline eval run results.

Usage:
    python app/tool/score_offline_eval.py <run_result.jsonl> [report.json] [ragas_report.json]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REPORT_DIR = Path("test") / "offline_eval"
DEFAULT_RAGAS_THRESHOLDS = {
    "faithfulness": 0.7,
    "response_relevancy": 0.75,
    "context_recall": 0.6,
    "context_precision": 0.6,
}
DEFAULT_ABSTAIN_HINTS = [
    "不能确认",
    "暂时不能确认",
    "证据不足",
    "未找到",
    "建议补充",
]


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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("json report must be an object")
    return obj


def contains_all(text: str, keywords: List[str]) -> bool:
    return all(str(keyword) in text for keyword in keywords if str(keyword))


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(str(keyword) in text for keyword in keywords if str(keyword))


def first_round_tools(record: Dict[str, Any]) -> List[str]:
    summary = record.get("trace_summary") or {}
    tool_rounds = summary.get("tool_rounds") or []
    if not tool_rounds:
        return []
    first = sorted(tool_rounds, key=lambda x: int(x.get("retrieval_round", 1) or 1))[0]
    return list(first.get("selected_tools") or [])


def retry_tools(record: Dict[str, Any]) -> List[str]:
    summary = record.get("trace_summary") or {}
    dynamic_retry = summary.get("dynamic_retry") or {}
    return list(dynamic_retry.get("selected_tools") or [])


def final_reflection(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("final_reflection") or {}


def planner_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("planner") or {}


def item_confirm_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("item_confirm") or {}


def embedding_retrieval_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("embedding_retrieval") or {}


def hyde_retrieval_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("hyde_retrieval") or {}


def rrf_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("rrf_summary") or {}


def rerank_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return (record.get("trace_summary") or {}).get("rerank_summary") or {}


def answer_text(record: Dict[str, Any]) -> str:
    response = record.get("response") or {}
    return str(response.get("answer") or "")


def infer_answer_mode(record: Dict[str, Any], answer: str) -> str:
    item_confirm = item_confirm_summary(record)
    if item_confirm.get("confirmation_mode") in {"clarify_options", "clarify_required"}:
        return "clarify"

    reflection = final_reflection(record)
    evidence_status = str(reflection.get("evidence_status", "unknown"))
    final_confidence = float(reflection.get("final_confidence", 0.0) or 0.0)
    if evidence_status in {"insufficient", "conflicting"} and (
        final_confidence < 0.5 or contains_any(answer, DEFAULT_ABSTAIN_HINTS)
    ):
        return "abstain"

    return "direct_answer"


def expected_bool(record: Dict[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is None:
        return None
    return bool(value)


def matches_any_tool_expectation(selected_tools: List[str], record: Dict[str, Any], primary_key: str, alt_key: str) -> bool:
    primary = list(record.get(primary_key) or [])
    alternatives = list(record.get(alt_key) or [])

    if alternatives:
        return any(set(option).issubset(set(selected_tools)) for option in alternatives if isinstance(option, list))

    if primary:
        return set(primary).issubset(set(selected_tools))

    return True


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    answer = answer_text(record)
    planner = planner_summary(record)
    item_confirm = item_confirm_summary(record)
    reflection = final_reflection(record)
    embedding_retrieval = embedding_retrieval_summary(record)
    hyde_retrieval = hyde_retrieval_summary(record)
    rrf = rrf_summary(record)
    rerank = rerank_summary(record)

    expected_tools = list(record.get("expected_tools") or [])
    expected_retry_tools = list(record.get("expected_retry_tools") or [])
    expected_item_names = list(record.get("expected_item_names") or record.get("gold_item_names") or [])
    require_item_resolution = bool(record.get("require_item_resolution", False))
    gold_chunk_ids = list(record.get("gold_chunk_ids") or [])
    require_retrieval_hit = bool(record.get("require_retrieval_hit", False))
    require_rerank_hit = bool(record.get("require_rerank_hit", False))
    must_have_keywords = list(record.get("must_have_keywords") or [])
    must_not_have_keywords = list(record.get("must_not_have_keywords") or [])
    should_abstain = bool(record.get("should_abstain", False))

    selected_tools = first_round_tools(record)
    selected_retry_tools = retry_tools(record)
    final_item_names = list(item_confirm.get("final_item_names") or [])
    retrieval_round_count = int((record.get("trace_summary") or {}).get("retrieval_round_count", 1) or 1)
    evidence_status = str(reflection.get("evidence_status", "unknown"))
    final_confidence = float(reflection.get("final_confidence", 0.0) or 0.0)
    answer_mode = infer_answer_mode(record, answer)
    retrieval_chunk_ids = set(embedding_retrieval.get("top_chunk_ids") or []) | set(hyde_retrieval.get("top_chunk_ids") or [])
    rerank_chunk_ids = set(rerank.get("top_chunk_ids") or [])

    expected_intent_type = str(record.get("expected_intent_type") or "")
    expected_task_type = str(record.get("expected_task_type") or "")
    acceptable_intent_types = list(record.get("acceptable_intent_types") or [])
    acceptable_task_types = list(record.get("acceptable_task_types") or [])
    expected_need_clarify = expected_bool(record, "expected_need_clarify")
    expected_evidence_status = str(record.get("expected_evidence_status") or "")
    expected_answer_mode = str(record.get("expected_answer_mode") or "")
    require_second_round = bool(record.get("require_second_round", False))
    min_final_confidence = record.get("min_final_confidence")
    max_final_confidence = record.get("max_final_confidence")

    http_ok = int(record.get("http_status", 0) or 0) == 200
    tool_ok = matches_any_tool_expectation(selected_tools, record, "expected_tools", "acceptable_first_round_tools_any_of")
    retry_tool_ok = matches_any_tool_expectation(selected_retry_tools, record, "expected_retry_tools", "acceptable_retry_tools_any_of")
    must_have_ok = contains_all(answer, must_have_keywords) if must_have_keywords else True
    must_not_have_ok = not contains_any(answer, must_not_have_keywords) if must_not_have_keywords else True
    abstain_ok = contains_any(answer, DEFAULT_ABSTAIN_HINTS) if should_abstain else True
    if acceptable_intent_types:
        intent_ok = planner.get("intent_type") in acceptable_intent_types
    else:
        intent_ok = planner.get("intent_type") == expected_intent_type if expected_intent_type else True
    if acceptable_task_types:
        task_ok = planner.get("task_type") in acceptable_task_types
    else:
        task_ok = planner.get("task_type") == expected_task_type if expected_task_type else True
    clarify_ok = (
        bool(planner.get("need_clarify", False)) == expected_need_clarify
        if expected_need_clarify is not None
        else True
    )
    evidence_ok = evidence_status == expected_evidence_status if expected_evidence_status else True
    if require_item_resolution and expected_item_names and item_confirm:
        item_name_ok = set(expected_item_names).issubset(set(final_item_names))
    else:
        item_name_ok = True
    answer_mode_ok = answer_mode == expected_answer_mode if expected_answer_mode else True
    second_round_ok = retrieval_round_count > 1 if require_second_round else True
    retrieval_hit_ok = bool(retrieval_chunk_ids & set(gold_chunk_ids)) if (require_retrieval_hit and gold_chunk_ids) else True
    rerank_hit_ok = bool(rerank_chunk_ids & set(gold_chunk_ids)) if (require_rerank_hit and gold_chunk_ids) else True
    second_round_rescue_ok = True
    if require_second_round:
        second_round_rescue_ok = retrieval_round_count > 1 and evidence_status in {"sufficient", "insufficient", "conflicting"}

    confidence_min_ok = True if min_final_confidence is None else final_confidence >= float(min_final_confidence)
    confidence_max_ok = True if max_final_confidence is None else final_confidence <= float(max_final_confidence)
    confidence_ok = confidence_min_ok and confidence_max_ok

    passed = all(
        [
            http_ok,
            tool_ok,
            retry_tool_ok,
            must_have_ok,
            must_not_have_ok,
            abstain_ok,
            intent_ok,
            task_ok,
            clarify_ok,
            evidence_ok,
            item_name_ok,
            answer_mode_ok,
            second_round_ok,
            retrieval_hit_ok,
            rerank_hit_ok,
            second_round_rescue_ok,
            confidence_ok,
        ]
    )

    return {
        "case_id": record.get("case_id", ""),
        "category": record.get("category", ""),
        "http_ok": http_ok,
        "tool_ok": tool_ok,
        "retry_tool_ok": retry_tool_ok,
        "must_have_ok": must_have_ok,
        "must_not_have_ok": must_not_have_ok,
        "abstain_ok": abstain_ok,
        "intent_ok": intent_ok,
        "task_ok": task_ok,
        "clarify_ok": clarify_ok,
        "evidence_ok": evidence_ok,
        "item_name_ok": item_name_ok,
        "answer_mode_ok": answer_mode_ok,
        "second_round_ok": second_round_ok,
        "retrieval_hit_ok": retrieval_hit_ok,
        "rerank_hit_ok": rerank_hit_ok,
        "second_round_rescue_ok": second_round_rescue_ok,
        "retrieval_hit_applicable": bool(require_retrieval_hit and gold_chunk_ids),
        "rerank_hit_applicable": bool(require_rerank_hit and gold_chunk_ids),
        "second_round_rescue_applicable": bool(require_second_round),
        "confidence_ok": confidence_ok,
        "passed": passed,
        "selected_tools_round1": selected_tools,
        "selected_tools_retry": selected_retry_tools,
        "expected_tools": expected_tools,
        "expected_retry_tools": expected_retry_tools,
        "planner_intent_type": planner.get("intent_type", ""),
        "planner_task_type": planner.get("task_type", ""),
        "planner_need_clarify": bool(planner.get("need_clarify", False)),
        "expected_intent_type": expected_intent_type,
        "expected_task_type": expected_task_type,
        "acceptable_intent_types": acceptable_intent_types,
        "acceptable_task_types": acceptable_task_types,
        "expected_need_clarify": expected_need_clarify,
        "final_item_names": final_item_names,
        "expected_item_names": expected_item_names,
        "require_item_resolution": require_item_resolution,
        "confirmation_mode": item_confirm.get("confirmation_mode", ""),
        "retrieval_round_count": retrieval_round_count,
        "embedding_hit_count": int(embedding_retrieval.get("hit_count", 0) or 0),
        "hyde_hit_count": int(hyde_retrieval.get("hit_count", 0) or 0),
        "rrf_output_count": int(rrf.get("rrf_output_count", 0) or 0),
        "rerank_topk_count": int(rerank.get("topk_doc_count", 0) or 0),
        "retrieval_chunk_ids": sorted((str(x) for x in retrieval_chunk_ids if x), key=str),
        "rerank_chunk_ids": sorted((str(x) for x in rerank_chunk_ids if x), key=str),
        "gold_chunk_ids": gold_chunk_ids,
        "evidence_status": evidence_status,
        "expected_evidence_status": expected_evidence_status,
        "answer_mode": answer_mode,
        "expected_answer_mode": expected_answer_mode,
        "final_confidence": final_confidence,
        "min_ragas_faithfulness": record.get("min_ragas_faithfulness"),
        "min_ragas_response_relevancy": record.get("min_ragas_response_relevancy"),
        "min_ragas_context_recall": record.get("min_ragas_context_recall"),
        "min_ragas_context_precision": record.get("min_ragas_context_precision"),
        "answer_preview": answer[:300],
    }


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {"total": 0, "pass_rate": 0.0}

    def ratio(key: str) -> float:
        return round(sum(1 for x in scores if x[key]) / total, 4)

    def ratio_applicable(ok_key: str, applicable_key: str):
        applicable = [x for x in scores if x.get(applicable_key)]
        if not applicable:
            return None
        return round(sum(1 for x in applicable if x[ok_key]) / len(applicable), 4)

    passed = sum(1 for x in scores if x["passed"])
    avg_confidence = sum(float(x["final_confidence"]) for x in scores) / total
    second_round_cases = [x for x in scores if int(x["retrieval_round_count"]) > 1]
    required_second_round_cases = [x for x in scores if x["expected_retry_tools"] or x["expected_answer_mode"] == "abstain"]

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4),
        "tool_selection_accuracy": ratio("tool_ok"),
        "retry_tool_accuracy": ratio("retry_tool_ok"),
        "intent_accuracy": ratio("intent_ok"),
        "task_type_accuracy": ratio("task_ok"),
        "clarify_accuracy": ratio("clarify_ok"),
        "item_name_accuracy": ratio("item_name_ok"),
        "evidence_status_accuracy": ratio("evidence_ok"),
        "answer_mode_accuracy": ratio("answer_mode_ok"),
        "retrieval_hit_accuracy": ratio_applicable("retrieval_hit_ok", "retrieval_hit_applicable"),
        "rerank_hit_accuracy": ratio_applicable("rerank_hit_ok", "rerank_hit_applicable"),
        "second_round_rescue_accuracy": ratio_applicable("second_round_rescue_ok", "second_round_rescue_applicable"),
        "must_have_accuracy": ratio("must_have_ok"),
        "must_not_have_accuracy": ratio("must_not_have_ok"),
        "abstain_accuracy": ratio("abstain_ok"),
        "second_round_requirement_accuracy": ratio("second_round_ok"),
        "confidence_range_accuracy": ratio("confidence_ok"),
        "avg_final_confidence": round(avg_confidence, 4),
        "second_round_trigger_rate": round(len(second_round_cases) / total, 4),
        "second_round_required_case_count": len(required_second_round_cases),
        "retrieval_hit_case_count": sum(1 for x in scores if x.get("retrieval_hit_applicable")),
        "rerank_hit_case_count": sum(1 for x in scores if x.get("rerank_hit_applicable")),
        "second_round_rescue_case_count": sum(1 for x in scores if x.get("second_round_rescue_applicable")),
    }


def group_by_category(scores: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for score in scores:
        category = str(score.get("category") or "uncategorized")
        grouped.setdefault(category, []).append(score)

    category_summary: Dict[str, Dict[str, Any]] = {}
    for category, items in grouped.items():
        category_summary[category] = aggregate(items)
    return category_summary


def build_field_comments() -> Dict[str, Any]:
    return {
        "summary": {
            "total": "总测试 case 数量",
            "passed": "通过的 case 数量",
            "pass_rate": "通过率 = passed / total",
            "tool_selection_accuracy": "首轮工具选择准确率，selected_tools 包含 expected_tools 的比例",
            "retry_tool_accuracy": "重试轮工具选择准确率，selected_tools_retry 包含 expected_retry_tools 的比例",
            "intent_accuracy": "Planner 识别意图准确率，planner_intent_type 等于 expected_intent_type 的比例",
            "task_type_accuracy": "Planner 识别任务类型准确率，planner_task_type 等于 expected_task_type 的比例",
            "clarify_accuracy": "是否需要澄清判断准确率，planner_need_clarify 等于 expected_need_clarify 的比例",
            "item_name_accuracy": "产品名称识别准确率，require_item_resolution=true 时 case 才参与计算",
            "evidence_status_accuracy": "证据充分性判断准确率，evidence_status 等于 expected_evidence_status 的比例",
            "answer_mode_accuracy": "回答模式准确率，direct_answer / clarify / abstain",
            "retrieval_hit_accuracy": "检索是否命中 gold_chunk_ids，require_retrieval_hit=true 时 case 才参与计算",
            "rerank_hit_accuracy": "重排是否命中 gold_chunk_ids，require_rerank_hit=true 时 case 才参与计算",
            "second_round_rescue_accuracy": "需二轮检索的 case 在二轮后是否成功得到有效证据状态",
            "must_have_accuracy": "回答是否包含全部必有关键词的比例",
            "must_not_have_accuracy": "回答是否不包含任何禁用关键词的比例",
            "abstain_accuracy": "应放弃时是否正确放弃回答的比例",
            "second_round_requirement_accuracy": "是否满足二轮检索要求的比例",
            "confidence_range_accuracy": "置信度在预期范围内的 case 比例",
            "avg_final_confidence": "全部 case 的最终置信度均值",
            "second_round_trigger_rate": "触发二轮检索的比例",
            "second_round_required_case_count": "需要二轮检索的 case 数量",
            "retrieval_hit_case_count": "参与检索命中评估的 case 数量",
            "rerank_hit_case_count": "参与重排命中评估的 case 数量",
            "second_round_rescue_case_count": "参与二轮挽救评估的 case 数量",
        },
        "category_summary": "按 category 分组的 summary，各字段含义与 summary 相同",
        "failed_cases": {
            "case_id": "测试 case 唯一标识",
            "category": "测试 case 所属分类",
            "http_ok": "HTTP 响应是否为 200",
            "tool_ok": "首轮工具选择是否正确",
            "retry_tool_ok": "重试轮工具选择是否正确",
            "must_have_ok": "回答是否包含全部必有关键词",
            "must_not_have_ok": "回答是否不包含任何禁用关键词",
            "abstain_ok": "应放弃时是否正确放弃回答",
            "intent_ok": "意图识别是否正确",
            "task_ok": "任务类型识别是否正确",
            "clarify_ok": "是否需要澄清判断是否正确",
            "evidence_ok": "证据充分性判断是否正确",
            "item_name_ok": "产品名称识别是否正确",
            "answer_mode_ok": "回答模式是否正确",
            "second_round_ok": "是否满足二轮检索要求",
            "retrieval_hit_ok": "检索是否命中 gold_chunk_ids",
            "rerank_hit_ok": "重排是否命中 gold_chunk_ids",
            "second_round_rescue_ok": "二轮检索挽救是否成功",
            "confidence_ok": "置信度是否在预期范围内",
            "passed": "该 case 是否全部检查项通过",
            "selected_tools_round1": "首轮实际选择的工具列表",
            "selected_tools_retry": "重试轮实际选择的工具列表",
            "expected_tools": "首轮期望选择的工具列表",
            "expected_retry_tools": "重试轮期望选择的工具列表",
            "planner_intent_type": "Planner 识别出的意图类型",
            "planner_task_type": "Planner 识别出的任务类型",
            "planner_need_clarify": "Planner 判断是否需要澄清",
            "expected_intent_type": "该 case 期望的意图类型",
            "expected_task_type": "该 case 期望的任务类型",
            "expected_need_clarify": "该 case 期望是否需要澄清",
            "final_item_names": "产品确认后最终识别的产品名列表",
            "expected_item_names": "期望识别的产品名列表",
            "confirmation_mode": "产品确认模式（confirmed / clarify_options / clarify_required / no_item_continue）",
            "retrieval_round_count": "检索总轮数",
            "embedding_hit_count": "Embedding 检索命中数",
            "hyde_hit_count": "HyDE 检索命中数",
            "rrf_output_count": "RRF 融合后输出结果数",
            "rerank_topk_count": "Rerank 重排后 topk 结果数",
            "retrieval_chunk_ids": "Embedding/HyDE 检索返回的 chunk_id 列表",
            "rerank_chunk_ids": "Rerank 重排后返回的 chunk_id 列表",
            "gold_chunk_ids": "标注的金标 chunk_id 列表",
            "evidence_status": "实际判断的证据充分性状态",
            "expected_evidence_status": "期望的证据充分性状态",
            "answer_mode": "实际回答模式（direct_answer / clarify / abstain）",
            "expected_answer_mode": "期望的回答模式",
            "final_confidence": "最终置信度",
            "answer_preview": "回答前 300 字预览",
        },
        "all_case_scores": "全部 case 的评分详情，字段含义同 failed_cases",
    }
def build_ragas_field_comments() -> Dict[str, Any]:
    return {
        "ragas_threshold_defaults": {
            "faithfulness": "默认最低阈值 0.7，可由 case 的 min_ragas_faithfulness 覆盖",
            "response_relevancy": "默认最低阈值 0.75，可由 case 的 min_ragas_response_relevancy 覆盖",
            "context_recall": "默认最低阈值 0.6，可由 case 的 min_ragas_context_recall 覆盖",
            "context_precision": "默认最低阈值 0.6，可由 case 的 min_ragas_context_precision 覆盖",
        },
        "ragas_summary": {
            "total": "参与 ragas 评测的 case 总数",
            "faithfulness_count": "成功产出 faithfulness 分数的 case 数量",
            "faithfulness_mean": "faithfulness 平均分",
            "response_relevancy_count": "成功产出 response_relevancy 分数的 case 数量",
            "response_relevancy_mean": "response_relevancy 平均分",
            "context_recall_count": "成功产出 context_recall 分数的 case 数量",
            "context_recall_mean": "context_recall 平均分",
            "context_precision_count": "成功产出 context_precision 分数的 case 数量",
            "context_precision_mean": "context_precision 平均分",
            "faithfulness_pass_rate": "faithfulness 达到阈值的比例",
            "response_relevancy_pass_rate": "response_relevancy 达到阈值的比例",
            "context_recall_pass_rate": "context_recall 达到阈值的比例",
            "context_precision_pass_rate": "context_precision 达到阈值的比例",
        },
        "ragas_cases": {
            "case_id": "与离线评测 case 对齐的唯一标识",
            "category": "case 分类",
            "query": "用户问题",
            "retrieved_context_count": "传给 ragas 的检索上下文条数",
            "has_reference": "是否提供 gold_answer/reference",
            "has_response": "是否有系统回答",
            "faithfulness": "回答是否忠于检索上下文",
            "response_relevancy": "回答与问题的相关性",
            "context_recall": "检索上下文对参考答案的召回程度",
            "context_precision": "检索上下文相对参考答案的精确度",
            "thresholds": "该 case 生效的 ragas 阈值配置",
            "faithfulness_ok": "faithfulness 是否达到阈值",
            "response_relevancy_ok": "response_relevancy 是否达到阈值",
            "context_recall_ok": "context_recall 是否达到阈值",
            "context_precision_ok": "context_precision 是否达到阈值",
        },
    }


def merge_ragas_scores(
    scores: List[Dict[str, Any]],
    ragas_report: Dict[str, Any] | None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, List[Dict[str, Any]]]:
    if not ragas_report:
        return scores, None, []

    ragas_cases = ragas_report.get("cases") or []
    ragas_by_case_id = {
        str(case.get("case_id") or ""): case
        for case in ragas_cases
        if isinstance(case, dict) and str(case.get("case_id") or "")
    }

    merged_scores: List[Dict[str, Any]] = []
    for score in scores:
        case_id = str(score.get("case_id") or "")
        ragas_case = ragas_by_case_id.get(case_id) or {}
        thresholds = {
            "faithfulness": score.get("min_ragas_faithfulness"),
            "response_relevancy": score.get("min_ragas_response_relevancy"),
            "context_recall": score.get("min_ragas_context_recall"),
            "context_precision": score.get("min_ragas_context_precision"),
        }
        for metric_name, default_value in DEFAULT_RAGAS_THRESHOLDS.items():
            if thresholds.get(metric_name) is None:
                thresholds[metric_name] = default_value

        faithfulness = ragas_case.get("faithfulness")
        response_relevancy = ragas_case.get("response_relevancy")
        context_recall = ragas_case.get("context_recall")
        context_precision = ragas_case.get("context_precision")
        merged = dict(score)
        merged["ragas"] = {
            "faithfulness": faithfulness,
            "response_relevancy": response_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "retrieved_context_count": ragas_case.get("retrieved_context_count"),
            "has_reference": ragas_case.get("has_reference"),
            "has_response": ragas_case.get("has_response"),
            "thresholds": thresholds,
            "faithfulness_ok": faithfulness is not None and float(faithfulness) >= float(thresholds["faithfulness"]),
            "response_relevancy_ok": response_relevancy is not None and float(response_relevancy) >= float(thresholds["response_relevancy"]),
            "context_recall_ok": context_recall is not None and float(context_recall) >= float(thresholds["context_recall"]),
            "context_precision_ok": context_precision is not None and float(context_precision) >= float(thresholds["context_precision"]),
        }
        merged_scores.append(merged)

    ragas_summary = dict(ragas_report.get("summary") or {})
    if merged_scores:
        for metric_name in DEFAULT_RAGAS_THRESHOLDS:
            ok_key = f"{metric_name}_ok"
            applicable = [item for item in merged_scores if item.get("ragas", {}).get(metric_name) is not None]
            ragas_summary[f"{metric_name}_pass_rate"] = (
                round(sum(1 for item in applicable if item.get("ragas", {}).get(ok_key)) / len(applicable), 4)
                if applicable
                else None
            )

    return merged_scores, ragas_summary, ragas_cases


def resolve_report_path(run_path: Path, report_arg: str | None) -> Path:
    if report_arg:
        return Path(report_arg)
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_REPORT_DIR / f"{run_path.stem}.score.{ts}.json"


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python app/tool/score_offline_eval.py <run_result.jsonl> [report.json] [ragas_report.json]")
        return 1

    run_path = Path(argv[1])
    if not run_path.exists():
        print(f"run result not found: {run_path}")
        return 1

    ragas_report = None
    if len(argv) > 3:
        ragas_path = Path(argv[3])
        if not ragas_path.exists():
            print(f"ragas report not found: {ragas_path}")
            return 1
        ragas_report = load_json(ragas_path)

    records = load_jsonl(run_path)
    scores = [score_record(record) for record in records]
    summary = aggregate(scores)
    category_summary = group_by_category(scores)
    merged_scores, ragas_summary, ragas_cases = merge_ragas_scores(scores, ragas_report)
    failed_cases = [x for x in merged_scores if not x["passed"]]
    field_comments = build_field_comments()
    field_comments.update(build_ragas_field_comments())

    report = {
        "summary": summary,
        "ragas_summary": ragas_summary,
        "category_summary": category_summary,
        "field_comments": field_comments,
        "failed_cases": failed_cases,
        "all_case_scores": merged_scores,
        "ragas_cases": ragas_cases,
    }

    report_path = resolve_report_path(run_path, argv[2] if len(argv) > 2 else None)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"offline eval score report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
