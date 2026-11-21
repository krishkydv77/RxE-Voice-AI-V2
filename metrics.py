"""Daily cost / usage accumulator (file-backed: logs/daily_cost.json, restart ke baad bhi bacha rehta hai)."""
import json
import threading
from datetime import date

from logger import LOG_DIR, logger

_PATH = LOG_DIR / "daily_cost.json"
_lock = threading.Lock()
KEEP_DAYS = 60


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_data: dict = _load()


def _blank() -> dict:
    return {"requests": 0, "cost_inr": 0.0, "words": 0, "audio_sec": 0.0,
            "stt": 0.0, "llm": 0.0, "tts": 0.0, "web": 0.0, "other": 0.0, "paths": {}}


def _persist() -> None:
    try:
        for d in sorted(_data)[:-KEEP_DAYS]:
            _data.pop(d, None)
        tmp = _PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(_PATH)
    except OSError as e:
        logger.warning("METRICS daily_cost persist failed: %s", e)


def add_request(*, stt: float = 0.0, llm: float = 0.0, tts: float = 0.0, web: float = 0.0, other: float = 0.0,
                words: int = 0, audio_sec: float = 0.0, path: str = "?", count_request: bool = True) -> dict:
    """Ek request ka real cost aaj ke total me jodo. Returns aaj ke totals."""
    total = stt + llm + tts + web + other
    day = date.today().isoformat()
    with _lock:
        d = _data.setdefault(day, _blank())
        d["requests"] += 1 if count_request else 0
        d["cost_inr"] = round(d["cost_inr"] + total, 6)
        d["words"] += words
        d["audio_sec"] = round(d["audio_sec"] + audio_sec, 2)
        for k, v in (("stt", stt), ("llm", llm), ("tts", tts), ("web", web), ("other", other)):
            d[k] = round(d[k] + v, 6)
        d["paths"][path] = d["paths"].get(path, 0) + (1 if count_request else 0)
        _persist()
        out = {"daily_cost": round(d["cost_inr"], 4), "daily_requests": d["requests"],
               "daily_avg_cost": round(d["cost_inr"] / d["requests"], 4) if d["requests"] else 0.0}
    logger.info("METRICS daily | day=%s cost=₹%.4f requests=%d avg=₹%.4f", day, out["daily_cost"],
                out["daily_requests"], out["daily_avg_cost"])
    return out


def today() -> dict:
    return _data.get(date.today().isoformat(), _blank())


def recent(days: int = 7) -> dict:
    return {d: _data[d] for d in sorted(_data)[-days:]}
