"""
本地离线测评服务。

职责：
1. 运行评测集并生成 run 结果
2. 对 run 结果打分并生成 report
3. 对 report 做简要总结
4. 提供失败 case、报告对比、trace、趋势等本地能力

说明：
- 这是从原先的 MCP 服务层中抽回到 app/tool 的本地版本
- 目的是让本地测评脚本不再依赖 app/mcp 目录
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from app.query_process.api.query_service import app as query_app
from app.tool.run_offline_eval import load_jsonl, normalize_case, run_case, write_jsonl
from app.tool.score_offline_eval import aggregate, build_field_comments, group_by_category, score_record


SUMMARY_DIFF_KEYS = [
    "pass_rate",
    "tool_selection_accuracy",
    "retry_tool_accuracy",
    "intent_accuracy",
    "task_type_accuracy",
    "clarify_accuracy",
    "item_name_accuracy",
    "evidence_status_accuracy",
    "answer_mode_accuracy",
    "must_have_accuracy",
    "must_not_have_accuracy",
    "abstain_accuracy",
    "second_round_requirement_accuracy",
    "confidence_range_accuracy",
    "avg_final_confidence",
    "second_round_trigger_rate",
]

TREND_KEYS = [
    "pass_rate",
    "intent_accuracy",
    "task_type_accuracy",
    "evidence_status_accuracy",
    "answer_mode_accuracy",
    "must_have_accuracy",
]


def _filter_cases(
    cases: List[Dict[str, Any]],
    category: str | None = None,
    case_ids: List[str] | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """按 category、case_id、limit 对 case 做筛选。"""
    filtered = list(cases)

    if category:
        filtered = [case for case in filtered if str(case.get("category") or "") == category]

    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        filtered = [case for case in filtered if str(case.get("case_id") or "") in wanted]

    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    return filtered


def _load_report(report_path: Path) -> Dict[str, Any]:
    """读取 report JSON。"""
    return json.loads(report_path.read_text(encoding="utf-8"))


def _build_run_id() -> str:
    """生成本次离线测评运行 ID。"""
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_eval(
    dataset_path: Path,
    output_path: Path,
    category: str | None = None,
    case_ids: List[str] | None = None,
    limit: int | None = None,
) -> Dict[str, Any]:
    """执行离线测评并输出 JSONL 运行结果。"""
    raw_cases = load_jsonl(dataset_path)
    normalized_cases = [normalize_case(case, idx) for idx, case in enumerate(raw_cases, start=1)]
    selected_cases = _filter_cases(normalized_cases, category=category, case_ids=case_ids, limit=limit)

    client = TestClient(query_app)
    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(selected_cases, start=1):
        try:
            result = run_case(client, case, idx)
        except Exception as exc:
            result = {
                "case_id": case["case_id"],
                "query": case["query"],
                "category": case.get("category", ""),
                "error": str(exc),
            }
        results.append(result)

    write_jsonl(output_path, results)

    return {
        "run_id": _build_run_id(),
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "total_cases": len(normalized_cases),
        "selected_cases": len(selected_cases),
    }


def score_eval(run_output_path: Path, report_output_path: Path) -> Dict[str, Any]:
    """对离线运行结果打分，并生成标准化 report JSON。"""
    records = load_jsonl(run_output_path)
    scores = [score_record(record) for record in records]
    summary = aggregate(scores)
    category_summary = group_by_category(scores)
    failed_cases = [score for score in scores if not score.get("passed", False)]

    report: Dict[str, Any] = {
        "summary": summary,
        "category_summary": category_summary,
        "field_comments": build_field_comments(),
        "failed_cases": failed_cases,
        "all_case_scores": scores,
    }

    report_output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "report_path": str(report_output_path),
        "summary": summary,
        "failed_case_count": len(failed_cases),
    }


def list_failed_cases(
    report_path: Path,
    category: str | None = None,
    failed_check: str | None = None,
    limit: int | None = None,
) -> Dict[str, Any]:
    """从报告中筛选失败 case。"""
    report = _load_report(report_path)
    failed_cases: List[Dict[str, Any]] = list(report.get("failed_cases") or [])

    if category:
        failed_cases = [case for case in failed_cases if str(case.get("category") or "") == category]

    if failed_check:
        failed_cases = [case for case in failed_cases if case.get(failed_check) is False]

    if limit is not None and limit > 0:
        failed_cases = failed_cases[:limit]

    simplified = []
    for case in failed_cases:
        failed_checks = [key for key, value in case.items() if key.endswith("_ok") and value is False]
        simplified.append(
            {
                "case_id": case.get("case_id", ""),
                "query": case.get("query", ""),
                "category": case.get("category", ""),
                "failed_checks": failed_checks,
                "planner_intent_type": case.get("planner_intent_type", ""),
                "planner_task_type": case.get("planner_task_type", ""),
                "answer_mode": case.get("answer_mode", ""),
                "evidence_status": case.get("evidence_status", ""),
                "answer_preview": case.get("answer_preview", ""),
            }
        )

    return {
        "report_path": str(report_path),
        "count": len(simplified),
        "failed_cases": simplified,
    }


def rerun_failed_subset(
    dataset_path: Path,
    report_path: Path,
    output_path: Path,
    category: str | None = None,
    failed_check: str | None = None,
    limit: int | None = None,
) -> Dict[str, Any]:
    """根据 report 中的失败 case 重跑子集。"""
    report = _load_report(report_path)
    failed_cases: List[Dict[str, Any]] = list(report.get("failed_cases") or [])

    if category:
        failed_cases = [case for case in failed_cases if str(case.get("category") or "") == category]

    if failed_check:
        failed_cases = [case for case in failed_cases if case.get(failed_check) is False]

    failed_case_ids = [str(case.get("case_id") or "") for case in failed_cases if str(case.get("case_id") or "")]
    if limit is not None and limit > 0:
        failed_case_ids = failed_case_ids[:limit]

    result = run_eval(
        dataset_path=dataset_path,
        output_path=output_path,
        case_ids=failed_case_ids,
    )
    result["source_report_path"] = str(report_path)
    result["selected_failed_case_ids"] = failed_case_ids
    return result


def _diff_number(base_value: Any, new_value: Any) -> Dict[str, Any]:
    """对两个数值型指标做差异计算。"""
    if base_value is None or new_value is None:
        return {"base": base_value, "new": new_value, "delta": None}
    return {
        "base": round(float(base_value), 4),
        "new": round(float(new_value), 4),
        "delta": round(float(new_value) - float(base_value), 4),
    }


def diff_reports(base_report_path: Path, new_report_path: Path) -> Dict[str, Any]:
    """比较两份评测报告。"""
    base_report = _load_report(base_report_path)
    new_report = _load_report(new_report_path)

    base_summary = base_report.get("summary") or {}
    new_summary = new_report.get("summary") or {}

    summary_diff = {
        key: _diff_number(base_summary.get(key), new_summary.get(key))
        for key in SUMMARY_DIFF_KEYS
        if key in base_summary or key in new_summary
    }

    base_category_summary = base_report.get("category_summary") or {}
    new_category_summary = new_report.get("category_summary") or {}
    all_categories = sorted(set(base_category_summary.keys()) | set(new_category_summary.keys()))
    category_diff = {}
    for category in all_categories:
        base_pass_rate = (base_category_summary.get(category) or {}).get("pass_rate")
        new_pass_rate = (new_category_summary.get(category) or {}).get("pass_rate")
        category_diff[category] = _diff_number(base_pass_rate, new_pass_rate)

    base_scores = {
        str(case.get("case_id") or ""): bool(case.get("passed", False))
        for case in base_report.get("all_case_scores") or []
    }
    new_scores = {
        str(case.get("case_id") or ""): bool(case.get("passed", False))
        for case in new_report.get("all_case_scores") or []
    }

    all_case_ids = sorted(set(base_scores.keys()) | set(new_scores.keys()))
    fixed_cases = [case_id for case_id in all_case_ids if base_scores.get(case_id) is False and new_scores.get(case_id) is True]
    regressed_cases = [case_id for case_id in all_case_ids if base_scores.get(case_id) is True and new_scores.get(case_id) is False]

    return {
        "base_report_path": str(base_report_path),
        "new_report_path": str(new_report_path),
        "summary_diff": summary_diff,
        "category_diff": category_diff,
        "fixed_cases": fixed_cases,
        "regressed_cases": regressed_cases,
    }


def get_case_trace(
    run_output_path: Path,
    case_id: str = "",
    session_id: str = "",
    include_events: bool = False,
    max_events: int = 30,
) -> Dict[str, Any]:
    """从 run 结果中读取单个 case 的 trace 信息。"""
    records = load_jsonl(run_output_path)

    target = None
    for record in records:
        if case_id and str(record.get("case_id") or "") == case_id:
            target = record
            break
        if session_id and str(record.get("session_id") or "") == session_id:
            target = record
            break

    if not target:
        raise ValueError("未找到对应的 case_id 或 session_id")

    trace_summary = target.get("trace_summary") or {}
    trace_events = list(target.get("trace_events") or [])
    if include_events and max_events > 0:
        trace_events = trace_events[:max_events]
    elif not include_events:
        trace_events = []

    return {
        "case_id": target.get("case_id", ""),
        "session_id": target.get("session_id", ""),
        "query": target.get("query", ""),
        "http_status": target.get("http_status", 0),
        "response": target.get("response", {}),
        "trace_summary": trace_summary,
        "trace_events": trace_events,
    }


def summarize_report_trends(
    report_paths: List[Path] | None = None,
    report_dir: Path | None = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """汇总多份评测报告的趋势。"""
    collected = list(report_paths or [])
    if report_dir is not None:
        collected.extend(sorted(report_dir.glob("*.json"), key=lambda p: p.stat().st_mtime))

    unique: List[Path] = []
    seen = set()
    for path in collected:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        unique.append(path)

    if limit > 0:
        unique = unique[-limit:]

    if not unique:
        raise ValueError("未找到可用于趋势分析的报告文件")

    items = []
    for path in unique:
        summary = (_load_report(path).get("summary") or {})
        items.append(
            {
                "report_path": str(path),
                "pass_rate": summary.get("pass_rate"),
                "intent_accuracy": summary.get("intent_accuracy"),
                "task_type_accuracy": summary.get("task_type_accuracy"),
                "evidence_status_accuracy": summary.get("evidence_status_accuracy"),
                "answer_mode_accuracy": summary.get("answer_mode_accuracy"),
                "must_have_accuracy": summary.get("must_have_accuracy"),
            }
        )

    latest = items[-1]
    previous = items[-2] if len(items) >= 2 else None

    latest_vs_previous = {}
    if previous:
        for key in TREND_KEYS:
            latest_vs_previous[key] = _diff_number(previous.get(key), latest.get(key))

    best_report = max(items, key=lambda item: float(item.get("pass_rate") or 0.0))
    worst_report = min(items, key=lambda item: float(item.get("pass_rate") or 0.0))

    return {
        "report_count": len(items),
        "reports": items,
        "latest_report": latest,
        "previous_report": previous,
        "latest_vs_previous": latest_vs_previous,
        "best_report": best_report,
        "worst_report": worst_report,
    }


def summarize_eval(report_path: Path) -> Dict[str, Any]:
    """对单份评测报告做高层总结。"""
    report = _load_report(report_path)
    summary = report.get("summary") or {}
    category_summary = report.get("category_summary") or {}
    failed_cases: List[Dict[str, Any]] = list(report.get("failed_cases") or [])

    pass_rate = float(summary.get("pass_rate", 0.0) or 0.0)
    intent_accuracy = float(summary.get("intent_accuracy", 0.0) or 0.0)
    task_type_accuracy = float(summary.get("task_type_accuracy", 0.0) or 0.0)
    answer_mode_accuracy = float(summary.get("answer_mode_accuracy", 0.0) or 0.0)
    evidence_accuracy = float(summary.get("evidence_status_accuracy", 0.0) or 0.0)

    sorted_categories = sorted(
        (
            {
                "category": category,
                "pass_rate": float((metrics or {}).get("pass_rate", 0.0) or 0.0),
            }
            for category, metrics in category_summary.items()
        ),
        key=lambda item: item["pass_rate"],
    )

    weakest_categories = sorted_categories[:3]
    strongest_categories = list(reversed(sorted_categories[-3:])) if sorted_categories else []

    failed_check_counter: Dict[str, int] = {}
    for case in failed_cases:
        for key, value in case.items():
            if key.endswith("_ok") and value is False:
                failed_check_counter[key] = failed_check_counter.get(key, 0) + 1

    top_failed_checks = [
        {"check": check, "count": count}
        for check, count in sorted(failed_check_counter.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    highlights = []
    risks = []
    recommendations = []

    if intent_accuracy >= 0.8:
        highlights.append(f"入口层意图识别较稳，intent_accuracy={intent_accuracy:.2f}")
    else:
        risks.append(f"入口层意图识别仍偏弱，intent_accuracy={intent_accuracy:.2f}")

    if task_type_accuracy >= 0.85:
        highlights.append(f"任务类型规划较稳，task_type_accuracy={task_type_accuracy:.2f}")
    else:
        risks.append(f"任务类型规划仍不够稳，task_type_accuracy={task_type_accuracy:.2f}")

    if answer_mode_accuracy < 0.8:
        risks.append(f"回答模式控制仍是短板，answer_mode_accuracy={answer_mode_accuracy:.2f}")
        recommendations.append("优先分析 answer_mode_ok 失败 case，收敛过度 abstain 或误答问题")

    if evidence_accuracy < 0.9:
        risks.append(f"证据层稳定性不足，evidence_status_accuracy={evidence_accuracy:.2f}")
        recommendations.append("优先分析证据判断与 rerank 保留结果是否一致")

    if weakest_categories:
        weakest_text = "、".join(f"{item['category']}({item['pass_rate']:.2f})" for item in weakest_categories)
        risks.append(f"当前最弱分类为：{weakest_text}")

    if strongest_categories:
        strongest_text = "、".join(f"{item['category']}({item['pass_rate']:.2f})" for item in strongest_categories)
        highlights.append(f"当前表现较好的分类为：{strongest_text}")

    if pass_rate < 0.6:
        summary_text = (
            f"本轮总通过率为 {pass_rate:.2f}，主链路可用但仍有明显短板。"
            f"当前更适合按弱分类和高频失败项做专题优化。"
        )
    else:
        summary_text = (
            f"本轮总通过率为 {pass_rate:.2f}，整体表现已进入可持续优化阶段。"
            f"当前重点应转向边界 case 和证据层质量。"
        )

    if not recommendations:
        recommendations.append("优先从最弱分类和 top failed checks 入手做下一轮优化")

    return {
        "report_path": str(report_path),
        "summary_text": summary_text,
        "highlights": highlights,
        "risks": risks,
        "recommended_next_steps": recommendations,
        "weakest_categories": weakest_categories,
        "strongest_categories": strongest_categories,
        "top_failed_checks": top_failed_checks,
    }
