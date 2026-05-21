"""
调试轨迹回放工具 (Debug Trace Replay)

本脚本用于回放指定会话的 LangGraph 调试轨迹，方便离线排查查询流程问题。

工作原理：
1. 从 logs/agent_trace.jsonl 中按 session_id 过滤出所有相关事件
2. 按 retrieval_round（检索轮次）和时间戳排序
3. 逐条打印节点名称、时间戳和事件载荷

运行方式：
    python app/tool/replay_debug_trace.py <session_id>

示例：
    python app/tool/replay_debug_trace.py abc123-def456
"""

import json
import sys
from pathlib import Path


TRACE_PATH = Path("logs") / "agent_trace.jsonl"


def replay_trace(session_id: str) -> None:
    """
    回放指定会话的调试轨迹。

    :param session_id: 目标会话 ID，用于从日志文件中过滤相关事件
    """
    if not TRACE_PATH.exists():
        print(f"trace file not found: {TRACE_PATH}")
        return

    matched = []
    with TRACE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("session_id") == session_id:
                matched.append(event)

    if not matched:
        print(f"no trace events found for session_id={session_id}")
        return

    matched.sort(key=lambda x: (x.get("retrieval_round", 1), x.get("ts", "")))

    print(f"trace replay for session_id={session_id}")
    print("=" * 80)
    for event in matched:
        print(f"[round={event.get('retrieval_round')}] {event.get('node')} @ {event.get('ts')}")
        print(json.dumps(event.get("payload", {}), ensure_ascii=False, indent=2))
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python app/tool/replay_debug_trace.py <session_id>")
        raise SystemExit(1)

    replay_trace(sys.argv[1])
