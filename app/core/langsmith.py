import os
from typing import Any, Dict, Optional

from app.core.logger import logger

try:
    from langsmith import tracing_context
    from langsmith.middleware import TracingMiddleware
except Exception:  # pragma: no cover - graceful fallback when optional deps are unavailable
    tracing_context = None
    TracingMiddleware = None


def _normalize_bool_env(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_langsmith_enabled() -> bool:
    return _normalize_bool_env(
        os.getenv("LANGSMITH_TRACING_V2", os.getenv("LANGSMITH_TRACING"))
    )


def get_langsmith_project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "default")


def bootstrap_langsmith() -> bool:
    """
    Normalize LangSmith env vars at startup so the project can keep using the
    simpler LANGSMITH_TRACING flag in .env while satisfying the SDK's V2 check.
    """
    enabled = is_langsmith_enabled()
    os.environ["LANGSMITH_TRACING_V2"] = "true" if enabled else "false"

    if enabled:
        logger.info(
            f"[LangSmith] tracing enabled, project={get_langsmith_project()}"
        )
    else:
        logger.info("[LangSmith] tracing disabled")
    return enabled


def add_langsmith_middleware(app: Any) -> None:
    if not is_langsmith_enabled():
        return
    if TracingMiddleware is None:
        logger.warning("[LangSmith] middleware unavailable, skip FastAPI middleware")
        return

    app.add_middleware(TracingMiddleware)
    logger.info("[LangSmith] FastAPI tracing middleware installed")


def build_tracing_metadata(
    *,
    service: str,
    operation: str,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    is_stream: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "service": service,
        "operation": operation,
    }
    if session_id:
        metadata["session_id"] = session_id
    if task_id:
        metadata["task_id"] = task_id
    if is_stream is not None:
        metadata["is_stream"] = is_stream
    if extra:
        metadata.update(extra)
    return metadata


def maybe_tracing_context(
    *,
    project_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    if not is_langsmith_enabled() or tracing_context is None:
        from contextlib import nullcontext

        return nullcontext()

    return tracing_context(
        project_name=project_name or get_langsmith_project(),
        tags=tags,
        metadata=metadata,
        enabled=True,
    )
