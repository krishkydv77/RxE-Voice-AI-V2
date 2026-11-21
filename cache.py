"""Async cache: Redis if REDIS_URL is set and reachable, else in-memory LRU.

Used for: answer cache, retrieval cache, SerpAPI cache, TTS audio cache.
Redis down ho jaye to app crash nahi hota - 30s ke liye memory fallback use hota hai.

V3 metrics (sab logger me dikhte hain):
  - har get/set: hit/miss, ms, remaining TTL, backend  -> aget_ex() / aset() ka info dict
  - per-request ops list (contextvar)                    -> begin_request() / ops_summary()
  - global counters: per-kind hit rate, avg/p95 get ms, avg set ms -> snapshot()
  - Redis INFO (memory, server hits/misses, clients)     -> redis_info()
"""
import asyncio
import contextvars
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque

from dotenv import load_dotenv

from logger import logger, log_metric

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "")          # e.g. redis://localhost:6379/0
PREFIX = os.getenv("CACHE_PREFIX", "hcvb")
MEM_MAX = int(os.getenv("CACHE_MEM_MAX", "400"))
REDIS_RETRY_S = 30
KBVER_LOCAL_TTL_S = float(os.getenv("KBVER_LOCAL_TTL_S", "5"))   # kb version ko 5s process me yaad rakho (1 Redis RTT bachta hai)
STATS_LOG_INTERVAL_S = float(os.getenv("CACHE_STATS_LOG_S", "300"))

_mem: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_mem_lock = threading.Lock()
_mem_ver = "0"
_kbver_memo: tuple[float, str] = (0.0, "0")
_aredis = None
_down_until = 0.0
_stats_task: asyncio.Task | None = None
_started = time.time()

# ---- metrics state
_cnt: dict = defaultdict(lambda: {"hit": 0, "miss": 0, "set": 0, "err": 0})
_get_ms: deque = deque(maxlen=500)
_set_ms: deque = deque(maxlen=500)
_ops_var: contextvars.ContextVar = contextvars.ContextVar("cache_ops", default=None)


def make_key(kind: str, *parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{PREFIX}:{kind}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()}"


def kind_of(key: str) -> str:
    p = key.split(":", 2)
    return p[1] if len(p) > 2 else "generic"


# ------------------------------------------------------------------ memory backend
def _mem_get_ex(key: str):
    """returns (value, remaining_ttl_sec)"""
    with _mem_lock:
        item = _mem.get(key)
        if not item:
            return None, None
        exp, val = item
        now = time.time()
        if exp < now:
            _mem.pop(key, None)
            return None, None
        _mem.move_to_end(key)
        return val, int(exp - now)


def _mem_set(key: str, val: str, ttl: int) -> None:
    with _mem_lock:
        _mem[key] = (time.time() + ttl, val)
        _mem.move_to_end(key)
        while len(_mem) > MEM_MAX:
            _mem.popitem(last=False)


def _mark_down(e: Exception) -> None:
    global _down_until
    _down_until = time.time() + REDIS_RETRY_S
    _cnt["_redis"]["err"] += 1
    logger.warning("CACHE redis unavailable -> memory fallback for %ds | %s: %s", REDIS_RETRY_S, type(e).__name__, e)


async def _redis():
    global _aredis
    if not REDIS_URL or time.time() < _down_until:
        return None
    if _aredis is None:
        try:
            import redis.asyncio as aioredis

            # protocol=2: purane Redis (<6, e.g. Windows 3.0.x) HELLO command nahi samajhte
            _aredis = aioredis.from_url(REDIS_URL, protocol=2, socket_timeout=0.3,
                                        socket_connect_timeout=0.3, decode_responses=True)
            logger.info("CACHE redis client created | url=%s", REDIS_URL.split("@")[-1])
        except Exception as e:  # noqa: BLE001
            _mark_down(e)
            return None
    return _aredis


# ------------------------------------------------------------------ metrics helpers
def begin_request() -> list:
    """Har request ke start me call karo; is request ke saare cache ops (child tasks ke bhi) isme jama hote hain."""
    lst: list = []
    _ops_var.set(lst)
    return lst


def _record(op: str, kind: str, info: dict) -> None:
    c = _cnt[kind]
    if op == "get":
        c["hit" if info["hit"] else "miss"] += 1
        _get_ms.append(info["ms"])
    else:
        c["set"] += 1
        _set_ms.append(info["ms"])
    ops = _ops_var.get()
    if ops is not None:
        ops.append({"op": op, **info})
    if op == "get":
        logger.info("CACHE get | kind=%s hit=%s backend=%s ms=%.2f ttl_left=%ss",
                    kind, info["hit"], info["backend"], info["ms"], info["ttl_sec"])
    else:
        logger.info("CACHE set | kind=%s backend=%s ms=%.2f ttl=%ss bytes=%s",
                    kind, info["backend"], info["ms"], info["ttl_sec"], info.get("bytes"))


def ops_summary(ops: list) -> dict:
    gets = [o for o in ops if o["op"] == "get"]
    sets = [o for o in ops if o["op"] == "set"]
    hits = sum(1 for o in gets if o["hit"])
    by_kind: dict = {}
    for o in gets:
        by_kind.setdefault(o["kind"], []).append(bool(o["hit"]))
    return {
        "get_ops": len(gets), "get_hits": hits, "get_misses": len(gets) - hits,
        "get_ms_total": round(sum(o["ms"] for o in gets), 2),
        "set_ops_done": len(sets), "set_ms_total": round(sum(o["ms"] for o in sets), 2),
        "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()},
    }


def hit_rate(kind: str | None = None) -> float | None:
    items = [(k, c) for k, c in _cnt.items() if not k.startswith("_") and (kind is None or k == kind)]
    hit = sum(c["hit"] for _, c in items)
    tot = hit + sum(c["miss"] for _, c in items)
    return round(100.0 * hit / tot, 1) if tot else None


def avg_get_ms() -> float | None:
    return round(sum(_get_ms) / len(_get_ms), 2) if _get_ms else None


def avg_set_ms() -> float | None:
    return round(sum(_set_ms) / len(_set_ms), 2) if _set_ms else None


def _p95(d: deque) -> float | None:
    if not d:
        return None
    s = sorted(d)
    return round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)


def snapshot() -> dict:
    kinds = {}
    for k, c in _cnt.items():
        if k.startswith("_"):
            continue
        n = c["hit"] + c["miss"]
        kinds[k] = {**c, "hit_rate": round(100.0 * c["hit"] / n, 1) if n else None}
    return {
        "backend": backend(), "hit_rate": hit_rate(), "avg_get_ms": avg_get_ms(), "p95_get_ms": _p95(_get_ms),
        "avg_set_ms": avg_set_ms(), "p95_set_ms": _p95(_set_ms), "mem_entries": len(_mem), "mem_max": MEM_MAX,
        "redis_errors": _cnt["_redis"]["err"], "redis_down_for_s": max(0, round(_down_until - time.time())),
        "uptime_s": round(time.time() - _started), "by_kind": kinds,
    }


async def redis_info() -> dict:
    """Redis ki apni health: ping latency, memory, server-side hit ratio, clients."""
    r = await _redis()
    if r is None:
        return {"available": False, "backend": backend(), "configured": bool(REDIS_URL)}
    try:
        t0 = time.perf_counter()
        await r.ping()
        ping_ms = round((time.perf_counter() - t0) * 1000, 2)
        info = await r.info()
        keys = await r.dbsize()
        h, m = int(info.get("keyspace_hits", 0)), int(info.get("keyspace_misses", 0))
        return {
            "available": True, "ping_ms": ping_ms, "version": info.get("redis_version"),
            "used_memory": info.get("used_memory_human"), "maxmemory": info.get("maxmemory_human"),
            "keys": keys, "connected_clients": info.get("connected_clients"),
            "ops_per_sec": info.get("instantaneous_ops_per_sec"), "evicted_keys": info.get("evicted_keys"),
            "expired_keys": info.get("expired_keys"), "server_hits": h, "server_misses": m,
            "server_hit_ratio": round(100.0 * h / (h + m), 1) if (h + m) else None,
        }
    except Exception as e:  # noqa: BLE001
        _mark_down(e)
        return {"available": False, "backend": "memory", "error": f"{type(e).__name__}: {e}"}


async def _stats_loop(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            snap, ri = snapshot(), await redis_info()
            logger.info("CACHE STATS | app=%s | redis=%s", json.dumps(snap, ensure_ascii=False), json.dumps(ri, ensure_ascii=False))
            log_metric("cache_stats", hit_rate=snap["hit_rate"], avg_get_ms=snap["avg_get_ms"],
                       avg_set_ms=snap["avg_set_ms"], backend=snap["backend"], redis=ri)
        except Exception as e:  # noqa: BLE001
            logger.warning("CACHE stats loop error: %s", e)


def start_stats_logger() -> None:
    global _stats_task
    if _stats_task is None and STATS_LOG_INTERVAL_S > 0:
        _stats_task = asyncio.create_task(_stats_loop(STATS_LOG_INTERVAL_S))


# ------------------------------------------------------------------ get / set
async def aget_ex(key: str, kind: str | None = None):
    """returns (value|None, info). info = {hit, ms, ttl_sec, backend, kind}. GET + TTL ek hi pipeline (1 RTT)."""
    kind = kind or kind_of(key)
    t0 = time.perf_counter()
    r = await _redis()
    raw, ttl, backend_used = None, None, "memory"
    if r is not None:
        backend_used = "redis"
        try:
            pipe = r.pipeline(transaction=False)
            pipe.get(key)
            pipe.ttl(key)
            raw, ttl = await pipe.execute()
            if ttl is not None and ttl < 0:
                ttl = None
        except Exception as e:  # noqa: BLE001
            _mark_down(e)
            backend_used = "memory"
            raw, ttl = _mem_get_ex(key)
    else:
        raw, ttl = _mem_get_ex(key)
    value = None
    if raw is not None:
        try:
            value = json.loads(raw)
        except ValueError:
            value = None
    info = {"hit": value is not None, "ms": round((time.perf_counter() - t0) * 1000, 2), "ttl_sec": ttl,
            "backend": backend_used, "kind": kind}
    _record("get", kind, info)
    return value, info


async def aget(key: str):
    """Backward-compatible: sirf value."""
    return (await aget_ex(key))[0]


async def aset(key: str, value, ttl: int) -> dict:
    raw = json.dumps(value, ensure_ascii=False)
    kind = kind_of(key)
    t0 = time.perf_counter()
    r = await _redis()
    stored, backend_used = False, "memory"
    if r is not None:
        try:
            await r.set(key, raw, ex=ttl)
            stored, backend_used = True, "redis"
        except Exception as e:  # noqa: BLE001
            _mark_down(e)
    if not stored:
        _mem_set(key, raw, ttl)
    info = {"hit": None, "ms": round((time.perf_counter() - t0) * 1000, 2), "ttl_sec": ttl,
            "backend": backend_used, "kind": kind, "bytes": len(raw)}
    _record("set", kind, info)
    return info


# ------------------------------------------------------------------ KB version
async def akb_version() -> str:
    """KB version - reindex hone par badalta hai, taaki purane cached answers/retrievals use na hon."""
    global _kbver_memo
    now = time.time()
    if now - _kbver_memo[0] < KBVER_LOCAL_TTL_S:
        return _kbver_memo[1]
    ver = _mem_ver
    r = await _redis()
    if r is not None:
        try:
            v = await r.get(f"{PREFIX}:kbver")
            if v:
                ver = v
        except Exception as e:  # noqa: BLE001
            _mark_down(e)
    _kbver_memo = (now, ver)
    return ver


def bump_kb_version() -> None:
    """Sync (rag threads se call hota hai)."""
    global _mem_ver, _kbver_memo
    _mem_ver = str(time.time_ns())
    _kbver_memo = (time.time(), _mem_ver)
    logger.info("CACHE kb version bumped -> %s", _mem_ver)
    if not REDIS_URL:
        return
    try:
        import redis

        c = redis.Redis.from_url(REDIS_URL, protocol=2, socket_timeout=1, socket_connect_timeout=1)
        c.set(f"{PREFIX}:kbver", _mem_ver)
        c.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("CACHE bump_kb_version redis fail: %s", e)


def backend() -> str:
    return "redis" if REDIS_URL and time.time() >= _down_until else "memory"


async def close() -> None:
    global _aredis, _stats_task
    if _stats_task is not None:
        _stats_task.cancel()
        _stats_task = None
    if _aredis is not None:
        try:
            await _aredis.aclose()
        except Exception:  # noqa: BLE001
            pass
        _aredis = None
