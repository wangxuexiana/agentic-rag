import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


_TRACE_PATH = Path("logs") / "agent_trace.jsonl"


def append_trace_event(session_id: str, node: str, payload: Dict[str, Any], retrieval_round: int = 1) -> None:
    """
    以 JSONL 形式追加一条调试轨迹，供后续回放脚本使用。
    """
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "retrieval_round": retrieval_round,
        "node": node,
        "payload": payload,
    }

    with _TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────
# 📖 阅读导航
# 上一篇: app/utils/rate_limit_utils.py
# 下一篇: app/utils/escape_milvus_string_utils.py
# ──────────────────────────────────────────────────────────
