"""
本地离线测评一键脚本。

用途：
1. 运行评测集
2. 生成 run 结果 JSONL
3. 生成评分报告 JSON
4. 在终端输出一份简要摘要

适用场景：
- 只想在本地手工跑评测
- 不需要 MCP server / client 这一层
- 希望用一条命令完成“运行 + 打分”
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 兼容直接以脚本路径运行：python app\tool\run_local_eval.py ...
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tool.offline_eval_local_services import summarize_eval
from app.tool.offline_eval_local_services import run_eval
from app.tool.offline_eval_local_services import score_eval



def _resolve_path(path_str: str) -> Path:
    """把相对路径解析到项目根目录下。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _build_default_output_paths(dataset_path: Path) -> tuple[Path, Path]:
    """根据数据集名称和当前时间生成默认输出路径。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_path.stem
    run_output = PROJECT_ROOT / "test" / "offline_eval" / f"{dataset_name}_{timestamp}.run.jsonl"
    report_output = PROJECT_ROOT / "test" / "offline_eval" / f"{dataset_name}_{timestamp}.report.json"
    return run_output, report_output


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="本地离线测评一键脚本")
    parser.add_argument("dataset_path", help="评测集 JSONL 路径")
    parser.add_argument("--run-output", default="", help="运行结果输出路径，默认自动生成")
    parser.add_argument("--report-output", default="", help="评分报告输出路径，默认自动生成")
    parser.add_argument("--category", default="", help="只跑某个 category")
    parser.add_argument("--case-id", action="append", default=[], help="只跑指定 case_id，可重复传入")
    parser.add_argument("--limit", type=int, default=0, help="最多跑多少条 case")
    return parser


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = _resolve_path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"评测集不存在: {dataset_path}")

    if args.run_output:
        run_output_path = _resolve_path(args.run_output)
    else:
        run_output_path, _ = _build_default_output_paths(dataset_path)

    if args.report_output:
        report_output_path = _resolve_path(args.report_output)
    else:
        _, report_output_path = _build_default_output_paths(dataset_path)

    run_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    run_result = run_eval(
        dataset_path=dataset_path,
        output_path=run_output_path,
        category=args.category or None,
        case_ids=args.case_id or None,
        limit=args.limit or None,
    )
    score_result = score_eval(
        run_output_path=run_output_path,
        report_output_path=report_output_path,
    )
    summary = summarize_eval(report_output_path)

    print("离线测评已完成")
    print(f"数据集: {dataset_path}")
    print(f"运行结果: {run_output_path}")
    print(f"评分报告: {report_output_path}")
    print("")
    print("运行摘要")
    print(f"- 总 case 数: {run_result['total_cases']}")
    print(f"- 实际运行数: {run_result['selected_cases']}")
    print(f"- 通过数: {score_result['summary'].get('passed')}")
    print(f"- 通过率: {score_result['summary'].get('pass_rate')}")
    print("")
    print("报告总结")
    print(summary["summary_text"])

    highlights = list(summary.get("highlights") or [])
    if highlights:
        print("")
        print("亮点")
        for item in highlights:
            print(f"- {item}")

    risks = list(summary.get("risks") or [])
    if risks:
        print("")
        print("风险")
        for item in risks:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
