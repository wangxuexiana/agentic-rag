import json
import os
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from app.core.logger import logger


load_dotenv()


def _normalize_ttl(ttl_seconds: Optional[int]) -> Optional[int]:
    if ttl_seconds is None:
        return None
    try:
        ttl = int(ttl_seconds)
    except (TypeError, ValueError):
        return None
    return ttl if ttl > 0 else None


class BaseCacheBackend:
    """
    查询侧缓存统一接口。
    """

    backend_name = "base"

    def get(self, key: str, default: Any = None) -> Any:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def clear(self) -> bool:
        raise NotImplementedError

    def get_stats_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError


class InMemoryCacheBackend(BaseCacheBackend):
    """
    开发期默认缓存实现。
    """

    backend_name = "memory"

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stats: Dict[str, Any] = {
            "backend": self.backend_name,
            "get_hit": 0,
            "get_miss": 0,
            "set_count": 0,
            "delete_count": 0,
            "clear_count": 0,
        }

    def _purge_if_expired(self, key: str) -> None:
        payload = self._store.get(key)
        if not payload:
            return
        expire_at = payload.get("expire_at")
        if expire_at is None:
            return
        if expire_at <= time.time():
            self._store.pop(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        with self._lock:
            self._purge_if_expired(key)
            payload = self._store.get(key)
            if not payload:
                self._stats["get_miss"] += 1
                return default
            self._stats["get_hit"] += 1
            return payload.get("value", default)

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        if not key:
            return False
        ttl = _normalize_ttl(ttl_seconds)
        expire_at = time.time() + ttl if ttl else None
        with self._lock:
            self._store[key] = {"value": value, "expire_at": expire_at}
            self._stats["set_count"] += 1
        return True

    def delete(self, key: str) -> bool:
        if not key:
            return False
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            if existed:
                self._stats["delete_count"] += 1
        return existed

    def clear(self) -> bool:
        with self._lock:
            self._store.clear()
            self._stats["clear_count"] += 1
        return True

    def get_stats_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = deepcopy(self._stats)
            snapshot["store_size"] = len(self._store)
        return snapshot


class RedisCacheBackend(BaseCacheBackend):
    """
    Redis 缓存实现。
    """

    backend_name = "redis"

    def __init__(self) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis backend requested but redis package is not installed") from exc

        redis_url = os.getenv("CACHE_REDIS_URL") or os.getenv("REDIS_URL")
        if not redis_url:
            raise ValueError("missing CACHE_REDIS_URL or REDIS_URL")

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._redis.ping()
        self._lock = threading.RLock()
        self._stats: Dict[str, Any] = {
            "backend": self.backend_name,
            "get_hit": 0,
            "get_miss": 0,
            "set_count": 0,
            "delete_count": 0,
            "clear_count": 0,
        }

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        raw = self._redis.get(key)
        with self._lock:
            if raw is None:
                self._stats["get_miss"] += 1
                return default
            self._stats["get_hit"] += 1
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("cache redis payload is not valid json, fallback to raw string")
            return raw

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        if not key:
            return False
        payload = json.dumps(value, ensure_ascii=False)
        ttl = _normalize_ttl(ttl_seconds)
        if ttl:
            ok = bool(self._redis.set(key, payload, ex=ttl))
        else:
            ok = bool(self._redis.set(key, payload))
        if ok:
            with self._lock:
                self._stats["set_count"] += 1
        return ok

    def delete(self, key: str) -> bool:
        if not key:
            return False
        deleted = bool(self._redis.delete(key))
        if deleted:
            with self._lock:
                self._stats["delete_count"] += 1
        return deleted

    def clear(self) -> bool:
        self._redis.flushdb()
        with self._lock:
            self._stats["clear_count"] += 1
        return True

    def get_stats_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = deepcopy(self._stats)
        try:
            snapshot["store_size"] = int(self._redis.dbsize())
        except Exception:
            snapshot["store_size"] = -1
        return snapshot


_cache_backend: Optional[BaseCacheBackend] = None


def _build_backend() -> BaseCacheBackend:
    backend_name = (os.getenv("CACHE_BACKEND") or "memory").strip().lower()
    if backend_name == "redis":
        return RedisCacheBackend()
    return InMemoryCacheBackend()


def get_cache_backend() -> BaseCacheBackend:
    global _cache_backend
    if _cache_backend is not None:
        return _cache_backend
    try:
        _cache_backend = _build_backend()
        logger.info(f"cache backend initialized: {_cache_backend.backend_name}")
    except Exception as exc:
        logger.warning(f"cache backend fallback to memory: {exc}")
        _cache_backend = InMemoryCacheBackend()
    return _cache_backend


def get_cache(key: str, default: Any = None) -> Any:
    return get_cache_backend().get(key, default=default)


def set_cache(key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    return get_cache_backend().set(key, value, ttl_seconds=ttl_seconds)


def delete_cache(key: str) -> bool:
    return get_cache_backend().delete(key)


def clear_cache() -> bool:
    return get_cache_backend().clear()


def get_cache_stats_snapshot() -> Dict[str, Any]:
    return get_cache_backend().get_stats_snapshot()
