"""
Run ragas evaluation on offline eval run results.

Usage:
    python app/tool/run_ragas_eval.py <run_result.jsonl> [report.json]
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import Embeddings

from app.lm.embedding_utils import generate_embeddings
from app.lm.llm_utils import get_llm_client


DEFAULT_REPORT_DIR = Path("test") / "offline_eval"
METRIC_MAX_ATTEMPTS = 4
METRIC_RETRY_BACKOFF_SECONDS = 2.0


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


class BgeM3LangChainEmbeddings(Embeddings):
    """Minimal LangChain embeddings adapter backed by the project's BGE-M3 helper."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return list(generate_embeddings(texts).get("dense") or [])

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []


def load_ragas_symbols():
    try:
        from ragas import SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import SingleTurnSample

    try:
        from ragas.metrics.collections import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

    return {
        "SingleTurnSample": SingleTurnSample,
        "Faithfulness": Faithfulness,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextRecall": LLMContextRecall,
        "ResponseRelevancy": ResponseRelevancy,
    }


def resolve_report_path(run_path: Path, report_arg: Optional[str]) -> Path:
    if report_arg:
        return Path(report_arg)
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_REPORT_DIR / f"{run_path.stem}.ragas.{ts}.json"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def _should_retry_metric_error(message: str) -> bool:
    lowered = message.lower()
    retry_markers = [
        "connection error",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    ]
    return any(marker in lowered for marker in retry_markers)


def score_metric(metric: Any, sample: Any) -> tuple[Optional[float], Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(1, METRIC_MAX_ATTEMPTS + 1):
        try:
            result = asyncio.run(metric.single_turn_ascore(sample))
            return _as_float(result), None
        except Exception as exc:
            last_error = str(exc)
            if attempt >= METRIC_MAX_ATTEMPTS or not _should_retry_metric_error(last_error):
                return None, last_error
            sleep_seconds = METRIC_RETRY_BACKOFF_SECONDS * attempt
            print(
                f"metric retry {attempt}/{METRIC_MAX_ATTEMPTS - 1} after error: {last_error}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    return None, last_error


def extract_primary_answer(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    markers = [
        "\n参考说明：",
        "\n【参考说明】",
        "\n参考来源：",
        "\n【图片】",
        "\n图片：",
        "\n置信度：",
        "\n来源：",
    ]
    cut_positions = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if cut_positions:
        text = text[: min(cut_positions)].rstrip()

    text = re.split(r"\n(?:参考说明|参考来源|来源|置信度)\s*[：:]", text, maxsplit=1)[0].rstrip()
    text = re.split(r"\n【(?:参考说明|图片|参考来源)】", text, maxsplit=1)[0].rstrip()
    return text


def build_sample(sample_cls: Any, record: Dict[str, Any]) -> Any:
    response = record.get("response") or {}
    primary_answer = extract_primary_answer(str(response.get("answer") or ""))
    return sample_cls(
        user_input=str(record.get("query") or ""),
        response=primary_answer,
        reference=str(record.get("gold_answer") or ""),
        retrieved_contexts=[
            str(text) for text in (record.get("retrieved_contexts") or []) if str(text).strip()
        ],
    )


def aggregate_case_scores(case_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "faithfulness",
        "response_relevancy",
        "context_recall",
        "context_precision",
    ]
    summary: Dict[str, Any] = {"total": len(case_scores)}
    for metric_name in metric_names:
        values = [
            float(item[metric_name])
            for item in case_scores
            if item.get(metric_name) is not None
        ]
        summary[f"{metric_name}_count"] = len(values)
        summary[f"{metric_name}_mean"] = round(mean(values), 4) if values else None
    return summary


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python app/tool/run_ragas_eval.py <run_result.jsonl> [report.json]")
        return 1

    run_path = Path(argv[1])
    if not run_path.exists():
        print(f"run result not found: {run_path}")
        return 1

    try:
        ragas = load_ragas_symbols()
    except Exception as exc:
        print(f"failed to import ragas: {exc}")
        print("install dependency first: pip install ragas")
        return 1

    records = load_jsonl(run_path)
    llm = get_llm_client()
    embeddings = BgeM3LangChainEmbeddings()

    faithfulness = ragas["Faithfulness"](llm=llm)
    response_relevancy = ragas["ResponseRelevancy"](llm=llm, embeddings=embeddings)
    context_recall = ragas["LLMContextRecall"](llm=llm)
    context_precision = ragas["LLMContextPrecisionWithReference"](llm=llm)
    sample_cls = ragas["SingleTurnSample"]

    case_scores: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] ragas case_id={record.get('case_id', '')}")
        sample = build_sample(sample_cls, record)
        retrieved_contexts = record.get("retrieved_contexts") or []
        gold_answer = str(record.get("gold_answer") or "").strip()
        answer = extract_primary_answer(str((record.get("response") or {}).get("answer") or ""))

        result = {
            "case_id": record.get("case_id", ""),
            "category": record.get("category", ""),
            "query": record.get("query", ""),
            "retrieved_context_count": len(retrieved_contexts),
            "has_reference": bool(gold_answer),
            "has_response": bool(answer),
            "response_for_eval_preview": answer[:300],
            "faithfulness": None,
            "response_relevancy": None,
            "context_recall": None,
            "context_precision": None,
            "metric_errors": {},
        }

        if answer and retrieved_contexts:
            result["faithfulness"], err = score_metric(faithfulness, sample)
            if err:
                result["metric_errors"]["faithfulness"] = err
        if answer:
            result["response_relevancy"], err = score_metric(response_relevancy, sample)
            if err:
                result["metric_errors"]["response_relevancy"] = err
        if gold_answer and retrieved_contexts:
            result["context_recall"], err = score_metric(context_recall, sample)
            if err:
                result["metric_errors"]["context_recall"] = err
            result["context_precision"], err = score_metric(context_precision, sample)
            if err:
                result["metric_errors"]["context_precision"] = err

        case_scores.append(result)

    report = {
        "summary": aggregate_case_scores(case_scores),
        "cases": case_scores,
    }

    report_path = resolve_report_path(run_path, argv[2] if len(argv) > 2 else None)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"ragas eval report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
