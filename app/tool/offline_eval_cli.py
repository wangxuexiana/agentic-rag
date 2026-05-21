"""
本地版离线评测全功能脚本。

目标：
1. 保留脚本形态，不引入 MCP server / client
2. 提供接近原先 MCP 服务的完整能力
3. 统一成一个命令行入口，便于本地调试和日常使用

支持的子命令：
- full-run
- run
- score
- list-failed
- diff
- summarize
- rerun-failed
- trace
- trends
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 兼容直接以脚本路径运行：python app\tool\offline_eval_cli.py ...
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tool.offline_eval_local_services import (
    diff_reports,
    get_case_trace,
    list_failed_cases,
    rerun_failed_subset,
    run_eval,
    score_eval,
    summarize_eval,
    summarize_report_trends,
)


def _resolve_path(path_str: str) -> Path:
    """把相对路径解析到项目根目录下。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _default_paths(dataset_path: Path) -> tuple[Path, Path]:
    """根据数据集名称和时间生成默认输出文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_path.stem
    run_output = PROJECT_ROOT / "test" / "offline_eval" / f"{dataset_name}_{timestamp}.run.jsonl"
    report_output = PROJECT_ROOT / "test" / "offline_eval" / f"{dataset_name}_{timestamp}.report.json"
    return run_output, report_output


def _print_json(payload: dict) -> None:
    """统一的 JSON 输出。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="本地版离线评测全功能脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_run = subparsers.add_parser("full-run", help="一条命令完成运行、打分和摘要输出")
    full_run.add_argument("dataset_path", help="评测集 JSONL 路径")
    full_run.add_argument("--run-output", default="", help="运行结果输出路径")
    full_run.add_argument("--report-output", default="", help="评分报告输出路径")
    full_run.add_argument("--category", default="", help="只跑某个 category")
    full_run.add_argument("--case-id", action="append", default=[], help="只跑指定 case_id，可重复传入")
    full_run.add_argument("--limit", type=int, default=0, help="最多跑多少条 case")

    run_cmd = subparsers.add_parser("run", help="只运行评测集，生成 run JSONL")
    run_cmd.add_argument("dataset_path", help="评测集 JSONL 路径")
    run_cmd.add_argument("--output", default="", help="运行结果输出路径")
    run_cmd.add_argument("--category", default="", help="只跑某个 category")
    run_cmd.add_argument("--case-id", action="append", default=[], help="只跑指定 case_id，可重复传入")
    run_cmd.add_argument("--limit", type=int, default=0, help="最多跑多少条 case")

    score_cmd = subparsers.add_parser("score", help="对 run JSONL 打分，生成 report JSON")
    score_cmd.add_argument("run_output_path", help="run JSONL 路径")
    score_cmd.add_argument("--output", default="", help="报告输出路径")

    failed_cmd = subparsers.add_parser("list-failed", help="查看失败 case")
    failed_cmd.add_argument("report_path", help="report JSON 路径")
    failed_cmd.add_argument("--category", default="", help="只看某个 category")
    failed_cmd.add_argument("--failed-check", default="", help="只看某个失败检查项")
    failed_cmd.add_argument("--limit", type=int, default=20, help="最多返回多少条")

    diff_cmd = subparsers.add_parser("diff", help="比较两份 report")
    diff_cmd.add_argument("base_report_path", help="基准 report 路径")
    diff_cmd.add_argument("new_report_path", help="新 report 路径")

    summarize_cmd = subparsers.add_parser("summarize", help="总结单份 report")
    summarize_cmd.add_argument("report_path", help="report JSON 路径")

    rerun_cmd = subparsers.add_parser("rerun-failed", help="根据 report 重跑失败子集")
    rerun_cmd.add_argument("dataset_path", help="原始评测集路径")
    rerun_cmd.add_argument("report_path", help="report JSON 路径")
    rerun_cmd.add_argument("--output", default="", help="重跑结果输出路径")
    rerun_cmd.add_argument("--category", default="", help="只重跑某个 category")
    rerun_cmd.add_argument("--failed-check", default="", help="只重跑某个失败检查项")
    rerun_cmd.add_argument("--limit", type=int, default=0, help="最多重跑多少条")

    trace_cmd = subparsers.add_parser("trace", help="查看单个 case 的 trace")
    trace_cmd.add_argument("run_output_path", help="run JSONL 路径")
    trace_cmd.add_argument("--case-id", default="", help="目标 case_id")
    trace_cmd.add_argument("--session-id", default="", help="目标 session_id")
    trace_cmd.add_argument("--include-events", action="store_true", help="是否包含原始 trace_events")
    trace_cmd.add_argument("--max-events", type=int, default=30, help="最多返回多少条 trace_events")

    trends_cmd = subparsers.add_parser("trends", help="汇总多份 report 的趋势")
    trends_cmd.add_argument("--report-path", action="append", default=[], help="直接传入 report 路径，可重复传入")
    trends_cmd.add_argument("--report-dir", default="", help="按目录扫描 report")
    trends_cmd.add_argument("--limit", type=int, default=10, help="最多纳入多少份报告")

    return parser


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "full-run":
        dataset_path = _resolve_path(args.dataset_path)
        run_output_path, report_output_path = _default_paths(dataset_path)
        if args.run_output:
            run_output_path = _resolve_path(args.run_output)
        if args.report_output:
            report_output_path = _resolve_path(args.report_output)

        run_output_path.parent.mkdir(parents=True, exist_ok=True)
        report_output_path.parent.mkdir(parents=True, exist_ok=True)

        run_result = run_eval(
            dataset_path=dataset_path,
            output_path=run_output_path,
            category=args.category or None,
            case_ids=args.case_id or None,
            limit=args.limit or None,
        )
        score_result = score_eval(run_output_path=run_output_path, report_output_path=report_output_path)
        summary = summarize_eval(report_output_path)
        _print_json(
            {
                "run_result": run_result,
                "score_result": score_result,
                "summary": summary,
            }
        )
        return 0

    if args.command == "run":
        dataset_path = _resolve_path(args.dataset_path)
        output_path = _resolve_path(args.output) if args.output else _default_paths(dataset_path)[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_eval(
            dataset_path=dataset_path,
            output_path=output_path,
            category=args.category or None,
            case_ids=args.case_id or None,
            limit=args.limit or None,
        )
        _print_json(result)
        return 0

    if args.command == "score":
        run_output_path = _resolve_path(args.run_output_path)
        output_path = _resolve_path(args.output) if args.output else run_output_path.with_suffix(".report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = score_eval(run_output_path=run_output_path, report_output_path=output_path)
        _print_json(result)
        return 0

    if args.command == "list-failed":
        report_path = _resolve_path(args.report_path)
        result = list_failed_cases(
            report_path=report_path,
            category=args.category or None,
            failed_check=args.failed_check or None,
            limit=args.limit,
        )
        _print_json(result)
        return 0

    if args.command == "diff":
        result = diff_reports(
            base_report_path=_resolve_path(args.base_report_path),
            new_report_path=_resolve_path(args.new_report_path),
        )
        _print_json(result)
        return 0

    if args.command == "summarize":
        result = summarize_eval(_resolve_path(args.report_path))
        _print_json(result)
        return 0

    if args.command == "rerun-failed":
        dataset_path = _resolve_path(args.dataset_path)
        report_path = _resolve_path(args.report_path)
        if args.output:
            output_path = _resolve_path(args.output)
        else:
            output_path = PROJECT_ROOT / "test" / "offline_eval" / "failed_subset.run.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = rerun_failed_subset(
            dataset_path=dataset_path,
            report_path=report_path,
            output_path=output_path,
            category=args.category or None,
            failed_check=args.failed_check or None,
            limit=args.limit or None,
        )
        _print_json(result)
        return 0

    if args.command == "trace":
        result = get_case_trace(
            run_output_path=_resolve_path(args.run_output_path),
            case_id=args.case_id,
            session_id=args.session_id,
            include_events=args.include_events,
            max_events=args.max_events,
        )
        _print_json(result)
        return 0

    if args.command == "trends":
        report_paths = [_resolve_path(path) for path in (args.report_path or [])]
        report_dir = _resolve_path(args.report_dir) if args.report_dir else None
        result = summarize_report_trends(
            report_paths=report_paths,
            report_dir=report_dir,
            limit=args.limit,
        )
        _print_json(result)
        return 0

    raise SystemExit(f"不支持的命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
