import asyncio
import base64
import math
import os
import re
import time

import httpx
from dotenv import load_dotenv
from groq import AsyncGroq

import cache
from logger import logger, log_metric, Timer

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# ---- STT: Whisper (Groq pe, sasta) -> fail ho to Sarvam Saarika fallback
STT_PROVIDER = os.getenv("STT_PROVIDER", "groq_whisper").lower()          # groq_whisper | sarvam
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip()              # "hi"/"en" force; blank = auto-detect
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "").strip()                  # optional vocabulary hint (medicine names)
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
STT_MODEL = WHISPER_MODEL if STT_PROVIDER == "groq_whisper" else SARVAM_STT_MODEL

# ---- TTS: sirf Sarvam, Bulbul V4 Flash (HTTP streaming endpoint)
# V4 Flash persona speaker IDs use karta hai (voice_language_style); V3 ke short naam (shubh, priya) 400 dete hain.
TTS_PROVIDER = "sarvam"
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh_enhi_ads")          # docs ka default v4-flash persona
TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.1"))
TTS_USE_STREAM = os.getenv("SARVAM_TTS_STREAM", "1") == "1"               # 1 = /text-to-speech/stream, 0 = REST JSON
TTS_CODEC = os.getenv("SARVAM_TTS_CODEC", "mp3").lower()                  # mp3 = chhota payload, streaming-safe
TTS_BITRATE = os.getenv("SARVAM_TTS_BITRATE", "64k")                      # speech ke liye kaafi, base64 size kam
TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "22050"))
TTS_CACHE_TTL = int(os.getenv("TTS_CACHE_TTL", str(7 * 86400)))
SARVAM_TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "aac": "audio/aac", "flac": "audio/flac", "opus": "audio/ogg"}
_BITRATE_BPS = int(re.sub(r"\D", "", TTS_BITRATE) or "64") * 1000
# docs me stream endpoint ka language field `language_code` likha hai, REST me `target_language_code` tha.
# Pehle dono bhejte hain; 422 aaye to automatically simple variant par aa jaate hain (jo chalta hai yaad rakhta hai).
_LANG_VARIANTS = [("language_code", "target_language_code"), ("language_code",), ("target_language_code",)]
_lang_variant = 0

# ---- LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")   # long-query summarise ke liye
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "15"))

SLOW_STT_MS = float(os.getenv("SLOW_STT_MS", "2000"))
SLOW_LLM_MS = float(os.getenv("SLOW_LLM_MS", "2000"))
SLOW_TTS_MS = float(os.getenv("SLOW_TTS_MS", "3000"))

# ---- cost constants: REAL published rates (verify date: 2026-11-21) - sab env se override ho sakte hain
USD_INR = float(os.getenv("USD_INR", "94"))                                # mid-market ~93.9 (Nov 2026)
WHISPER_USD_PER_HOUR = float(os.getenv("WHISPER_USD_PER_HOUR", "0.04"))   # Groq whisper-large-v3-turbo
WHISPER_MIN_BILLED_SEC = float(os.getenv("WHISPER_MIN_BILLED_SEC", "10")) # Groq: har request min 10s billed
SARVAM_STT_INR_PER_MIN = float(os.getenv("SARVAM_STT_INR_PER_MIN", "0.49"))     # pay-as-you-go (~Rs30/hr), per-second billed
SARVAM_TTS_INR_PER_10K_CHARS = float(os.getenv("SARVAM_TTS_INR_PER_10K_CHARS", "20"))  # Bulbul V4 Flash PAYG (Growth: 17)
AUDIO_KB_PER_SEC = float(os.getenv("AUDIO_KB_PER_SEC", "16"))             # sirf tab jab API duration na de (webm/opus approx)

groq_client = AsyncGroq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT, max_retries=1)

http_client = httpx.AsyncClient(
    timeout=60,
    http2=True,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
)

LANGUAGES = {
    "hi-IN": "Hindi", "en-IN": "English", "bn-IN": "Bengali",
    "gu-IN": "Gujarati", "kn-IN": "Kannada", "ml-IN": "Malayalam",
    "mr-IN": "Marathi", "od-IN": "Odia", "pa-IN": "Punjabi",
    "ta-IN": "Tamil", "te-IN": "Telugu",
}
DEFAULT_LANG = "hi-IN"

# Whisper language name/ISO -> Sarvam language code
_WHISPER_LANG = {
    "hindi": "hi-IN", "hi": "hi-IN", "english": "en-IN", "en": "en-IN", "bengali": "bn-IN", "bn": "bn-IN",
    "gujarati": "gu-IN", "gu": "gu-IN", "kannada": "kn-IN", "kn": "kn-IN", "malayalam": "ml-IN", "ml": "ml-IN",
    "marathi": "mr-IN", "mr": "mr-IN", "punjabi": "pa-IN", "pa": "pa-IN", "tamil": "ta-IN", "ta": "ta-IN",
    "telugu": "te-IN", "te": "te-IN",
}

_bg: set = set()


def _warn_if_slow(stage: str, ms: float, limit: float) -> None:
    if ms > limit:
        logger.warning("%s slow: %.1f ms (limit %.0f ms)", stage.upper(), ms, limit)


# ---------------------------------------------------------------- cost helpers
def estimate_duration_sec(audio_bytes: bytes) -> float:
    return (len(audio_bytes) / 1024) / AUDIO_KB_PER_SEC


def calculate_whisper_cost(duration_sec: float) -> float:
    billed = max(duration_sec, WHISPER_MIN_BILLED_SEC)          # Groq minimum 10 sec billing
    return round(billed / 3600.0 * WHISPER_USD_PER_HOUR * USD_INR, 6)


def calculate_sarvam_stt_cost(duration_sec: float) -> float:
    return round(math.ceil(duration_sec) * SARVAM_STT_INR_PER_MIN / 60.0, 6)     # per-second, round up


def calculate_sarvam_tts_cost(text: str) -> float:
    return round(len(text) * SARVAM_TTS_INR_PER_10K_CHARS / 10000.0, 6)          # per character sent


def audio_seconds(nbytes: int, codec: str | None = None) -> float:
    """Output audio ki length (real bytes se): cost_per_second_audio metric ke liye."""
    codec = (codec or TTS_CODEC).lower()
    if codec == "wav":
        return round(max(0, nbytes - 44) / (TTS_SAMPLE_RATE * 2), 3)
    if codec == "linear16":
        return round(nbytes / (TTS_SAMPLE_RATE * 2), 3)
    return round(nbytes * 8 / _BITRATE_BPS, 3)


def rates_summary() -> dict:
    return {"usd_inr": USD_INR, "whisper_usd_per_hour": WHISPER_USD_PER_HOUR, "whisper_min_billed_s": WHISPER_MIN_BILLED_SEC,
            "sarvam_stt_inr_per_min": SARVAM_STT_INR_PER_MIN, "tts_model": TTS_MODEL,
            "tts_inr_per_10k_chars": SARVAM_TTS_INR_PER_10K_CHARS, "tts_speaker": TTS_SPEAKER,
            "tts_transport": "http_stream" if TTS_USE_STREAM else "rest", "tts_codec": TTS_CODEC}


# ---------------------------------------------------------------- STT: Groq Whisper
def _norm_ct(content_type: str) -> str:
    ct = (content_type or "audio/webm").split(";")[0].strip().lower()
    return "audio/webm" if ct in ("", "application/octet-stream") else ct


async def _stt_whisper(audio_bytes: bytes, filename: str, content_type: str, force_lang: str = "") -> tuple[str, str, dict]:
    ct = _norm_ct(content_type)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": WHISPER_MODEL, "response_format": "verbose_json", "temperature": "0"}
    lang_hint = force_lang or WHISPER_LANGUAGE
    if lang_hint:
        data["language"] = lang_hint
    if WHISPER_PROMPT:
        data["prompt"] = WHISPER_PROMPT
    logger.info("STT start | provider=groq_whisper model=%s lang_hint=%s file=%s type=%s size=%.1f KB",
                WHISPER_MODEL, lang_hint or "auto", filename, ct, len(audio_bytes) / 1024)

    with Timer() as t:
        r = await http_client.post(GROQ_STT_URL, headers=headers, data=data, files={"file": (filename, audio_bytes, ct)})
    if r.status_code != 200:
        log_metric("stt", status="error", provider="groq_whisper", http_status=r.status_code,
                   latency_ms=t.ms, error=r.text[:300])
        raise RuntimeError(f"Groq Whisper STT error {r.status_code}: {r.text[:300]}")

    j = r.json()
    transcript = (j.get("text") or "").strip()
    segs = j.get("segments") or []
    if transcript and segs and all(float(s.get("no_speech_prob", 0) or 0) > 0.85 for s in segs):
        logger.warning("STT whisper: no_speech_prob high on all segments -> treated as silence (%r)", transcript[:60])
        transcript = ""

    wname = str(j.get("language") or "").lower()
    lang = _WHISPER_LANG.get(wname)
    if lang is None:
        lang = "en-IN" if transcript and transcript.isascii() else DEFAULT_LANG
    duration = float(j.get("duration") or estimate_duration_sec(audio_bytes))
    cost = calculate_whisper_cost(duration)

    meta = {
        "status": "ok", "latency_ms": t.ms, "model": WHISPER_MODEL, "provider": "groq_whisper",
        "http_status": r.status_code, "audio_kb": round(len(audio_bytes) / 1024, 1), "audio_type": ct,
        "duration_s": round(duration, 2), "detected_language": wname, "language_used": lang,
        "language_fallback": wname not in _WHISPER_LANG, "transcript_chars": len(transcript),
        "transcript_words": len(transcript.split()), "stt_cost_inr": cost,
    }
    log_metric("stt", **meta)
    logger.info("STT done | provider=groq_whisper took=%.1f ms lang=%s dur=%.1fs cost=₹%.5f", t.ms, lang, duration, cost)
    return transcript, lang, meta


# ---------------------------------------------------------------- STT: Sarvam (fallback)
async def _stt_sarvam(audio_bytes: bytes, filename: str, content_type: str) -> tuple[str, str, dict]:
    ct = _norm_ct(content_type)
    headers = {"api-subscription-key": SARVAM_API_KEY}
    data = {"model": SARVAM_STT_MODEL, "language_code": "unknown"}
    logger.info("STT start | provider=sarvam model=%s file=%s type=%s size=%.1f KB",
                SARVAM_STT_MODEL, filename, ct, len(audio_bytes) / 1024)
    with Timer() as t:
        r = await http_client.post(SARVAM_STT_URL, headers=headers, data=data, files={"file": (filename, audio_bytes, ct)})
    if r.status_code != 200:
        log_metric("stt", status="error", provider="sarvam", http_status=r.status_code, latency_ms=t.ms, error=r.text[:300])
        raise RuntimeError(f"Sarvam STT error {r.status_code}: {r.text}")

    j = r.json()
    transcript = (j.get("transcript") or "").strip()
    detected = j.get("language_code") or DEFAULT_LANG
    lang = detected if detected in LANGUAGES else DEFAULT_LANG
    cost = calculate_sarvam_stt_cost(estimate_duration_sec(audio_bytes))
    meta = {
        "status": "ok", "latency_ms": t.ms, "model": SARVAM_STT_MODEL, "provider": "sarvam",
        "http_status": r.status_code, "sarvam_request_id": j.get("request_id"),
        "audio_kb": round(len(audio_bytes) / 1024, 1), "audio_type": ct, "detected_language": detected,
        "language_used": lang, "language_fallback": detected != lang, "transcript_chars": len(transcript),
        "transcript_words": len(transcript.split()), "stt_cost_inr": cost,
    }
    log_metric("stt", **meta)
    logger.info("STT done | provider=sarvam took=%.1f ms lang=%s cost=₹%.5f", t.ms, lang, cost)
    return transcript, lang, meta


async def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm",
                         content_type: str = "audio/webm") -> tuple[str, str, dict]:
    if STT_PROVIDER == "groq_whisper" and GROQ_API_KEY:
        try:
            transcript, lang, meta = await _stt_whisper(audio_bytes, filename, content_type)
            # Hindi speech kabhi "Urdu" detect ho jaati hai -> ek baar language=hi ke saath retry
            if meta["detected_language"] == "urdu" and not WHISPER_LANGUAGE:
                logger.info("STT whisper detected Urdu -> retry with language=hi")
                first_cost = meta.get("stt_cost_inr", 0.0)
                transcript, lang, meta = await _stt_whisper(audio_bytes, filename, content_type, "hi")
                meta["retried_as_hindi"] = True
                meta["stt_cost_inr"] = round(meta.get("stt_cost_inr", 0.0) + first_cost, 6)   # dono calls billed hain
                meta["stt_calls"] = 2
            _warn_if_slow("stt", meta["latency_ms"], SLOW_STT_MS)
            return transcript, lang, meta
        except Exception as e:  # noqa: BLE001
            if not SARVAM_API_KEY:
                raise
            logger.warning("STT whisper failed -> Sarvam fallback | %s: %s", type(e).__name__, e)
            log_metric("stt_fallback", reason=str(e)[:200])
    elif STT_PROVIDER == "groq_whisper":
        logger.warning("STT_PROVIDER=groq_whisper but GROQ_API_KEY missing -> Sarvam STT use ho raha hai")
    transcript, lang, meta = await _stt_sarvam(audio_bytes, filename, content_type)
    _warn_if_slow("stt", meta["latency_ms"], SLOW_STT_MS)
    return transcript, lang, meta


# ---------------------------------------------------------------- TTS: Sarvam only
def split_text(text: str, limit: int = 400) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= limit:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s[:limit]
    if cur:
        chunks.append(cur)
    return chunks


async def _tts_stream_chunk(chunk: str, lang_code: str) -> tuple[bytes, float | None, float]:
    """Sarvam /text-to-speech/stream: binary audio progressively aata hai. returns (bytes, ttfb_ms, total_ms)."""
    global _lang_variant
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    t0 = time.perf_counter()
    last_err = ""
    for attempt in range(1, 4):
        payload = {"text": chunk, "speaker": TTS_SPEAKER, "model": TTS_MODEL, "pace": TTS_PACE,
                   "output_audio_codec": TTS_CODEC, "speech_sample_rate": TTS_SAMPLE_RATE,
                   "enable_preprocessing": True}
        if TTS_CODEC in ("mp3", "aac", "opus"):
            payload["output_audio_bitrate"] = TTS_BITRATE
        for f in _LANG_VARIANTS[_lang_variant]:
            payload[f] = lang_code
        try:
            async with http_client.stream("POST", SARVAM_TTS_STREAM_URL, headers=headers, json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "ignore")[:300]
                    last_err = f"http {r.status_code}: {body}"
                    logger.warning("TTS stream http %s (attempt %d, lang_variant=%s) | %s", r.status_code, attempt,
                                   _LANG_VARIANTS[_lang_variant], body[:160])
                    if r.status_code == 422 and _lang_variant < len(_LANG_VARIANTS) - 1:
                        _lang_variant += 1
                        continue
                    if (r.status_code == 429 or r.status_code >= 500) and attempt < 2:
                        await asyncio.sleep(0.15)          # sirf 1 retry; phir REST fallback (latency bachao)
                        continue
                    break
                parts, ttfb = [], None
                async for b in r.aiter_bytes():
                    if b:
                        if ttfb is None:
                            ttfb = round((time.perf_counter() - t0) * 1000, 1)
                        parts.append(b)
                data = b"".join(parts)
                if not data:
                    last_err = "empty audio stream"
                    continue
                return data, ttfb, round((time.perf_counter() - t0) * 1000, 1)
        except httpx.HTTPError as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("TTS stream network error (attempt %d) | %s", attempt, last_err)
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Sarvam TTS stream error {last_err}")


async def _tts_rest_chunk(chunk: str, lang_code: str) -> tuple[bytes, float]:
    """Fallback: REST /text-to-speech (JSON + base64 wav), same V4 Flash model."""
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    payload = {"text": chunk, "target_language_code": lang_code, "speaker": TTS_SPEAKER, "model": TTS_MODEL,
               "pace": TTS_PACE, "speech_sample_rate": TTS_SAMPLE_RATE, "enable_preprocessing": True}
    t0 = time.perf_counter()
    for attempt in (1, 2):
        r = await http_client.post(SARVAM_TTS_URL, headers=headers, json=payload)
        if r.status_code == 200 or (r.status_code < 500 and r.status_code != 429) or attempt == 2:
            break
        await asyncio.sleep(0.2)
    if r.status_code != 200:
        raise RuntimeError(f"Sarvam TTS error {r.status_code}: {r.text[:300]}")
    raw = b"".join(base64.b64decode(a) for a in r.json().get("audios", []))
    return raw, round((time.perf_counter() - t0) * 1000, 1)


async def text_to_speech(text: str, lang_code: str) -> tuple[list[str], dict]:
    chunks = split_text(text)
    transport = "http_stream" if TTS_USE_STREAM else "rest"
    logger.info("TTS start | provider=sarvam model=%s speaker=%s lang=%s transport=%s codec=%s chunks=%d chars=%d",
                TTS_MODEL, TTS_SPEAKER, lang_code, transport, TTS_CODEC, len(chunks), len(text))
    ttfb, fallback = None, False
    codec = TTS_CODEC
    with Timer() as total:
        results = None
        if TTS_USE_STREAM:
            try:
                results = await asyncio.gather(*[_tts_stream_chunk(c, lang_code) for c in chunks])
                ttfb = min((r[1] for r in results if r[1] is not None), default=None)
                raws = [r[0] for r in results]
                latencies = [r[2] for r in results]
            except Exception as e:  # noqa: BLE001
                fallback = True
                logger.warning("TTS stream failed -> REST fallback | %s: %s", type(e).__name__, e)
                log_metric("tts_fallback", reason=str(e)[:200])
        if not TTS_USE_STREAM or fallback:
            transport, codec = ("rest_fallback" if fallback else "rest"), "wav"
            results = await asyncio.gather(*[_tts_rest_chunk(c, lang_code) for c in chunks])
            raws = [r[0] for r in results]
            latencies = [r[1] for r in results]
            ttfb = min(latencies) if latencies else None      # REST me TTFB = poora response

    audios = [base64.b64encode(raw).decode("ascii") for raw in raws]
    audio_sec = round(sum(audio_seconds(len(raw), codec) for raw in raws), 3)
    cost = calculate_sarvam_tts_cost(text)
    meta = {
        "status": "ok", "latency_ms": total.ms, "ttfb_ms": ttfb, "chunk_latencies_ms": latencies,
        "model": TTS_MODEL, "provider": "sarvam", "speaker": TTS_SPEAKER, "language": lang_code,
        "transport": transport, "codec": codec, "mime": _MIME.get(codec, "audio/mpeg"), "sample_rate": TTS_SAMPLE_RATE,
        "chunks": len(chunks), "input_chars": len(text), "audio_clips": len(audios), "audio_sec": audio_sec,
        "audio_b64_kb": round(sum(len(a) for a in audios) / 1024, 1),
        "tts_cost_inr": cost, "billed_chars": len(text), "cached": False,
    }
    log_metric("tts", **meta)
    logger.info("TTS done | transport=%s ttfb=%s ms total=%.1f ms audio=%.2fs chars=%d cost=₹%.5f",
                transport, ttfb, total.ms, audio_sec, len(text), cost)
    _warn_if_slow("tts", total.ms, SLOW_TTS_MS)
    return audios, meta


async def text_to_speech_cached(text: str, lang_code: str) -> tuple[list[str], dict]:
    """Same (model, speaker, lang, codec, text) dobara aaye to Sarvam call + cost zero."""
    key = cache.make_key("tts", TTS_PROVIDER, TTS_MODEL, TTS_SPEAKER, TTS_CODEC if TTS_USE_STREAM else "wav",
                         lang_code, text)
    hit, info = await cache.aget_ex(key, "tts")
    if isinstance(hit, dict) and hit.get("audios"):
        logger.info("TTS cache HIT | chars=%d took=%.1f ms saved=₹%.5f", len(text), info["ms"],
                    calculate_sarvam_tts_cost(text))
        meta = {"status": "ok", "latency_ms": info["ms"], "ttfb_ms": info["ms"], "cached": True, "tts_cost_inr": 0.0,
                "input_chars": len(text), "provider": "cache", "mime": hit.get("mime", "audio/mpeg"),
                "audio_sec": hit.get("audio_sec", 0.0), "cache_ttl_sec": info["ttl_sec"],
                "saved_cost_inr": calculate_sarvam_tts_cost(text)}
        log_metric("tts", **meta)
        return hit["audios"], meta
    audios, meta = await text_to_speech(text, lang_code)
    if audios:
        task = asyncio.create_task(cache.aset(key, {"audios": audios, "mime": meta["mime"],
                                                    "audio_sec": meta["audio_sec"]}, TTS_CACHE_TTL))
        _bg.add(task)
        task.add_done_callback(_bg.discard)
    return audios, meta
