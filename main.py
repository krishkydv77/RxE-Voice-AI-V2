import asyncio
import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

import agent
import cache
import metrics
import rag
from logger import logger, log_metric, request_id_var, LOG_DIR

BASE_DIR = Path(__file__).resolve().parent
MAX_SESSIONS = 200
MAX_AUDIO_MB = 10

_sessions: "OrderedDict[str, list[dict]]" = OrderedDict()


def get_history(session_id: str) -> list[dict]:
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
    else:
        _sessions[session_id] = []
        logger.info("SESSION new | id=%s active=%d", session_id, len(_sessions))
        while len(_sessions) > MAX_SESSIONS:
            old, _ = _sessions.popitem(last=False)
            logger.info("SESSION evicted | id=%s", old)
    return _sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    request_id_var.set("startup")
    logger.info("=" * 70)
    logger.info("APP start | log_dir=%s data_dir=%s", LOG_DIR.resolve(), rag.DATA_DIR)
    for key in ("GROQ_API_KEY", "SARVAM_API_KEY", "SERPAPI_API_KEY"):
        if not (os.getenv(key) or (key == "SERPAPI_API_KEY" and os.getenv("SERPAPI_KEY"))):
            logger.error("ENV missing | %s is not set in .env", key)
    logger.info("CONFIG | llm=%s stt=%s/%s tts=%s/%s speaker=%s embed=%s chunk=%d/%d top_k=%d core_target=%.0f ms",
                agent.voice.GROQ_MODEL, agent.voice.STT_PROVIDER, agent.voice.STT_MODEL, agent.voice.TTS_PROVIDER,
                agent.voice.TTS_MODEL, agent.voice.TTS_SPEAKER, rag.EMBED_MODEL, rag.CHUNK_SIZE, rag.CHUNK_OVERLAP,
                rag.TOP_K, agent.CORE_TARGET_MS)
    logger.info("CONFIG | cache=%s rerank=%s(%s) web_strict=%s trusted_domains=%d web_ttl_days=%d",
                cache.backend(), rag.RERANK_ENABLED, rag.RERANK_MODEL, agent.mcp_tool.WEB_STRICT,
                len(agent.mcp_tool.TRUSTED_DOMAINS), agent.mcp_tool.WEB_TTL_DAYS)

    logger.info("COST RATES (real, INR) | %s", json.dumps(agent.rates_summary(), ensure_ascii=False))
    logger.info("CONFIG | tts_stream=%s codec=%s prompt_budget=%d ttft_target=%.0f ms ttft_timeout=%.1fs history_msgs=%d "
                "ctx_max_chars=%d", agent.voice.TTS_USE_STREAM, agent.voice.TTS_CODEC, agent.PROMPT_TOKEN_BUDGET,
                agent.TTFT_TARGET_MS, agent.LLM_TTFT_TIMEOUT_S, agent.HISTORY_MSGS, rag.CONTEXT_MAX_CHARS)
    cache.start_stats_logger()
    if agent.USE_MCP and await agent.mcp.connect():
        logger.info("TOOLS mode = MCP server (%s)", agent.mcp._session and agent.mcp.url)
    else:
        if agent.USE_MCP:
            logger.error("MCP server not found -> switching to in-process mode")
        logger.info("TOOLS mode = in-process")
        rag.sync_data_dir()
        rag.purge_expired()
    await agent.warmup()
    logger.info("APP ready")
    yield
    logger.info("APP shutdown")
    await agent.mcp.close()
    await agent.voice.http_client.aclose()
    await cache.close()


app = FastAPI(title="Healthcare & Medical E-Commerce Voice Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.middleware("http")
async def request_logger(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    start = time.perf_counter()
    logger.info("REQUEST start | %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("REQUEST crashed | %s %s", request.method, request.url.path)
        raise
    ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    if request.url.path.startswith("/api"):
        log_metric("request", method=request.method, path=request.url.path, status=response.status_code, total_ms=ms)
    logger.info("REQUEST end | %s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health():
    if agent.mcp.active:
        return {"status": "ok", "tools": "mcp", "sources": await agent.mcp.call("kb_list_sources", {}),
                "sessions": len(_sessions), "cache": cache.backend()}
    s = rag.stats()
    return {"status": "ok", "tools": "in-process", "chunks": s["total_chunks"],
            "files": list(s["files"].keys()), "sessions": len(_sessions), "cache": cache.backend()}


@app.post("/api/reindex")
async def reindex(force: bool = False):
    logger.info("API reindex | force=%s", force)
    if agent.mcp.active:
        result = {"result": await agent.mcp.call("kb_reindex", {"force": force})}
    else:
        result = await asyncio.to_thread(rag.sync_data_dir, force)
        result["purge"] = await asyncio.to_thread(rag.purge_expired)
    cache.bump_kb_version()          # purane cached answers/retrievals invalid
    return result


@app.get("/api/metrics")
async def api_metrics():
    """Cache (app + Redis INFO), aaj ka real cost, last 7 din ka daily cost, rates."""
    return {"cache": cache.snapshot(), "redis": await cache.redis_info(), "today": metrics.today(),
            "last_7_days": metrics.recent(7), "rates": agent.rates_summary()}


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    existed = _sessions.pop(session_id, None) is not None
    logger.info("SESSION cleared | id=%s existed=%s", session_id, existed)
    return {"cleared": existed}


async def _read_audio(audio: UploadFile, session_id: str, route: str) -> bytes:
    data = await audio.read()
    logger.info("API %s | session=%s file=%s type=%s size=%.1f KB",
                route, session_id, audio.filename, audio.content_type, len(data) / 1024)
    if not data:
        raise HTTPException(400, "Empty audio")
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, f"Audio file must be smaller than {MAX_AUDIO_MB} MB")
    return data


@app.post("/api/voice/stream")
async def voice_stream(audio: UploadFile = File(...), session_id: str = Form("default")):
    """NDJSON stream: transcript -> audio(index 0..n, pehli sentence ki audio turant) -> done.
    Frontend pehli audio aate hi play kare = minimum time-to-first-audio."""
    data = await _read_audio(audio, session_id, "voice/stream")
    history = get_history(session_id)
    q: asyncio.Queue = asyncio.Queue()

    async def runner():
        try:
            result = await agent.run_voice(data, history, audio.filename or "audio.webm",
                                           audio.content_type or "audio/webm", on_event=q.put_nowait)
            meta = result.get("metadata", {})
            logger.info("API voice/stream completed | path=%s ttft=%s first_audio_ms=%s total_cost=₹%.4f redis_hit=%s lang_mismatch=%s",
                        result.get("path"), meta.get("ttft_ms"), meta.get("first_audio_ms"), meta.get("total_cost_inr", 0.0),
                        meta.get("redis_hit"), meta.get("language_mismatch"))
            q.put_nowait({"type": "done", **{k: v for k, v in result.items() if k != "audios"}})
        except Exception as e:  # noqa: BLE001
            logger.exception("API voice/stream failed")
            q.put_nowait({"type": "error", "error": str(e)})
        finally:
            q.put_nowait(None)

    task = asyncio.create_task(runner())

    async def gen():
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        finally:
            if not task.done():
                task.cancel()          # client disconnect

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/voice")
async def voice_endpoint(audio: UploadFile = File(...), session_id: str = Form("default")):
    data = await _read_audio(audio, session_id, "voice")

    try:
        result = await agent.run_voice(data, get_history(session_id),
                                       audio.filename or "audio.webm", audio.content_type or "audio/webm")
        
        # (Optional) Response aane ke baad yahan metadata log dekh sakte hain
        meta = result.get("metadata", {})
        logger.info("API voice completed | path=%s total_cost=₹%.4f total_ms=%.1f", 
                    result.get("path"), meta.get("total_cost_inr", 0.0), meta.get("total_ms", 0.0))

    except Exception as e:
        logger.exception("API voice failed")
        return JSONResponse({"error": str(e)}, status_code=502)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")),
                reload=False, log_level="warning")