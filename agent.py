import asyncio
import os
import re
import time

import cache
import mcp_tool
import metrics
import query_opt
import rag
import voice
from mcp_client import MCPToolClient
from logger import logger, log_metric, Timer

WORD_LIMIT = int(os.getenv("REPLY_MAX_WORDS", "15"))
CORE_TARGET_MS = float(os.getenv("CORE_TARGET_MS", "2000"))      # target: time-to-first-audio
SUPPORT_HELPLINE = os.getenv("SUPPORT_HELPLINE", "1800-123-4567")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "70"))
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
LLM_TTFT_TIMEOUT_S = float(os.getenv("LLM_TTFT_TIMEOUT_S", "2"))  # pehla token isse late => fallback (hang nahi)
PROMPT_TOKEN_BUDGET = int(os.getenv("PROMPT_TOKEN_BUDGET", "160"))
TTFT_TARGET_MS = float(os.getenv("TTFT_TARGET_MS", "500"))

HISTORY_MSGS = int(os.getenv("HISTORY_MSGS", "2"))                # sirf last 1 exchange
HISTORY_MSG_CHARS = int(os.getenv("HISTORY_MSG_CHARS", "80"))
QUERY_LLM_WORDS = int(os.getenv("QUERY_LLM_WORDS", "25"))         # isse lambi query -> LLM intent-summary (retrieval ke saath parallel)
SUMMARY_MAX_WORDS = int(os.getenv("SUMMARY_MAX_WORDS", "15"))
MAX_QUERY_WORDS = int(os.getenv("MAX_QUERY_WORDS", "60"))
ANSWER_CACHE_TTL = int(os.getenv("ANSWER_CACHE_TTL", "1800"))
FIRST_CLAUSE_MIN_CHARS = int(os.getenv("FIRST_CLAUSE_MIN_CHARS", "20"))   # pehla clause (comma pe) jaldi TTS ko
TTS_PREWARM = os.getenv("TTS_PREWARM", "1") == "1"

# ---- REAL cost config (no guesses). SerpApi Developer plan: $75 / 5,000 searches = $0.015 per billed search.
SERPAPI_PLAN_USD = float(os.getenv("SERPAPI_PLAN_USD", "75"))
SERPAPI_PLAN_SEARCHES = float(os.getenv("SERPAPI_PLAN_SEARCHES", "5000"))
# Redis / Chroma / embedder / reranker self-hosted hain => per-request marginal cost 0.
# Apna server bill / monthly requests yahan daalo (default 0 = assume nahi karte).
OTHER_INFRA_COST_INR = float(os.getenv("OTHER_INFRA_COST_INR", "0"))

USE_MCP = os.getenv("USE_MCP", "1") == "1"   # 1 = MCP server mode; 0 = in-process mode
mcp = MCPToolClient()
SCORE_RE = re.compile(r"score=([0-9.]+)")
SRC_RE = re.compile(r"^\[\d+\] score=[0-9.]+ \| (.+?) \|", re.M)
HDR_RE = re.compile(r"^\[\d+\] score=[0-9.]+ \| .+$\n?", re.M)
META_RE = re.compile(r"^#meta (.*)$", re.M)
SEARCH_RE = re.compile(r"^\s*SEARCH\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

FALLBACK = {
    "hi-IN": "क्षमा करें, मुझे इसकी सटीक चिकित्सा जानकारी नहीं मिली। कृपया हमारे हेल्थकेयर सपोर्ट से संपर्क करें।",
    "en-IN": "Sorry, I couldn't find precise medical details regarding that. Please reach out to healthcare support.",
}

_bg_tasks: set[asyncio.Task] = set()

# Groq $/1M tokens (input, output) - groq.com/pricing, Nov 2026. NOTE: llama-3.1-8b-instant ko kuch pages "enterprise" dikhate hain;
# apne console ka rate GROQ_RATE_IN_USD / GROQ_RATE_OUT_USD se override karo.
_GROQ_RATES = [("gpt-oss-120b", 0.15, 0.60), ("gpt-oss-20b", 0.075, 0.30), ("llama-3.3-70b", 0.59, 0.79),
               ("70b", 0.59, 0.79), ("8b", 0.05, 0.08), ("instant", 0.05, 0.08), ("qwen", 0.29, 0.59)]


def groq_rates(model: str) -> tuple[float, float]:
    ei, eo = os.getenv("GROQ_RATE_IN_USD"), os.getenv("GROQ_RATE_OUT_USD")
    if ei and eo:
        return float(ei), float(eo)
    m = model.lower()
    return next(((i, o) for k, i, o in _GROQ_RATES if k in m), (0.15, 0.60))


def calculate_groq_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    in_usd, out_usd = groq_rates(model)
    cost_usd = (prompt_tokens / 1_000_000) * in_usd + (completion_tokens / 1_000_000) * out_usd
    return round(cost_usd * voice.USD_INR, 6)


def serpapi_cost_per_search_inr() -> float:
    return round(SERPAPI_PLAN_USD / SERPAPI_PLAN_SEARCHES * voice.USD_INR, 6)


def rates_summary() -> dict:
    i, o = groq_rates(voice.GROQ_MODEL)
    return {**voice.rates_summary(), "llm_model": voice.GROQ_MODEL, "llm_usd_per_1m_in": i, "llm_usd_per_1m_out": o,
            "serpapi_inr_per_billed_search": serpapi_cost_per_search_inr(), "other_infra_inr_per_req": OTHER_INFRA_COST_INR}


# ------------------------------------------------------------------ language (reply language = user's language = TTS language)
_SCRIPT_RANGES = [(0x0900, 0x097F, "deva"), (0x0980, 0x09FF, "beng"), (0x0A00, 0x0A7F, "guru"), (0x0A80, 0x0AFF, "gujr"),
                  (0x0B00, 0x0B7F, "orya"), (0x0B80, 0x0BFF, "taml"), (0x0C00, 0x0C7F, "telu"), (0x0C80, 0x0CFF, "knda"),
                  (0x0D00, 0x0D7F, "mlym")]
_LANG_SCRIPT = {"hi-IN": "deva", "mr-IN": "deva", "en-IN": "latn", "bn-IN": "beng", "gu-IN": "gujr", "kn-IN": "knda",
                "ml-IN": "mlym", "od-IN": "orya", "pa-IN": "guru", "ta-IN": "taml", "te-IN": "telu"}


def dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        if ch.isascii():
            counts["latn"] = counts.get("latn", 0) + 1
            continue
        o = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else None


def reply_language_name(lang: str, transcript: str) -> str:
    """LLM ko explicit batao kis language me jawab dena hai (prompt me 4-5 tokens)."""
    if lang == "hi-IN":
        return "Hinglish (Hindi in Roman script)" if query_opt.latin_ratio(transcript) >= 0.7 else "Hindi (Devanagari script)"
    return voice.LANGUAGES.get(lang, "English")


def response_language_check(lang: str, transcript: str, reply: str) -> tuple[bool, str | None]:
    """returns (language_mismatch, reply_script). Hinglish (hi-IN + Roman reply + Roman question) mismatch nahi hai."""
    got = dominant_script(reply)
    if got is None:
        return False, None
    expected = _LANG_SCRIPT.get(lang, "latn")
    if got == expected:
        return False, got
    if expected == "deva" and got == "latn" and query_opt.latin_ratio(transcript) >= 0.7:
        return False, got
    return True, got


# ------------------------------------------------------------------ prompts (compact = fewer input tokens; target total <= 200)
def _base_rules(lang_name: str) -> str:
    return (f"Healthcare & medical e-commerce voice assistant. Reply only in {lang_name}, plain text, max {WORD_LIMIT} words. "
            "Define only the asked term, in healthcare context. "
            f"No diagnosis/dosage; serious symptoms: see doctor; emergency: call {SUPPORT_HELPLINE}. "
            "Off-topic: say you only help with healthcare e-commerce.")


def smalltalk_system_prompt(lang_name: str) -> str:
    return _base_rules(lang_name) + " Greet briefly and offer help."


def kb_system_prompt(lang_name: str) -> str:
    return _base_rules(lang_name) + " Facts only from CONTEXT. If missing, output only: SEARCH: <short English web query>"


def web_system_prompt(lang_name: str) -> str:
    return _base_rules(lang_name) + " Facts only from WEB; say it is general information. Never output SEARCH."


# ------------------------------------------------------------------ helpers
def clean_for_speech(text: str) -> str:
    text = THINK_RE.sub(" ", text)
    text = re.sub(r"[*_`#>~|]+", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)  # emoji
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _spawn(coro, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _bg_tasks.add(task)

    def _done(tk: asyncio.Task) -> None:
        _bg_tasks.discard(tk)
        if not tk.cancelled() and tk.exception():
            logger.error("BG task fail | %s: %r", name, tk.exception())

    task.add_done_callback(_done)


def _recent_history(history: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"][:HISTORY_MSG_CHARS]} for m in history[-HISTORY_MSGS:]]


def _est_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _log_prompt_budget(stage: str, msgs: list[dict], context: str = "") -> int:
    sys_t = _est_tokens(msgs[0]["content"])
    ctx_t = _est_tokens(context) if context else 0
    hist_t = sum(_est_tokens(m["content"]) for m in msgs[1:-1])
    user_t = _est_tokens(msgs[-1]["content"]) - ctx_t
    total = sum(_est_tokens(m["content"]) for m in msgs)
    (logger.info if total <= PROMPT_TOKEN_BUDGET else logger.warning)(
        "PROMPT budget | stage=%s est_total=%d (target %d) = system %d + context %d + history %d + question %d",
        stage, total, PROMPT_TOKEN_BUDGET, sys_t, ctx_t, hist_t, max(0, user_t))
    return total


# ------------------------------------------------------------------ sentence streaming
_BOUNDARY = re.compile(r"[.!?।]+\s+")
_FIRST = re.compile(r"(?:[.!?।]+|[,;:،])\s+")      # pehle chunk me comma pe bhi cut (first-audio jaldi)


class SentenceSplitter:
    MIN_CHARS = 18

    def __init__(self) -> None:
        self.buf = ""
        self.first = True

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            regex, minc = (_FIRST, FIRST_CLAUSE_MIN_CHARS) if self.first else (_BOUNDARY, self.MIN_CHARS)
            cut = None
            for m in regex.finditer(self.buf):
                if m.end() >= minc:
                    cut = m.end()
                    break
            if cut is None:
                break
            out.append(self.buf[:cut].strip())
            self.buf = self.buf[cut:]
            self.first = False
        return out

    def flush(self) -> list[str]:
        s, self.buf = self.buf.strip(), ""
        return [s] if s else []


def _split_all(text: str) -> list[str]:
    sp = SentenceSplitter()
    return sp.feed(text + " ") + sp.flush()


class _TTSStream:
    """Har sentence ka TTS turant parallel start hota hai (LLM abhi bhi generate kar raha hota hai);
    audio order me emit hota hai. TTS language = STT language (forced)."""

    def __init__(self, tts_lang: str, emit, t0: float) -> None:
        self.tts_lang, self.emit, self.t0 = tts_lang, emit, t0
        logger.info("TTS language FORCED = stt_lang | %s", tts_lang)
        self.q: asyncio.Queue = asyncio.Queue()
        self.runner: asyncio.Future | None = None
        self.tasks: list[asyncio.Future] = []
        self.audios: list[str] = []
        self.cost = 0.0
        self.ms_sum = 0.0
        self.audio_sec = 0.0
        self.chars = 0
        self.n = 0
        self.cached_n = 0
        self.errors = 0
        self.first_audio_ms: float | None = None
        self.first_ttfb_ms: float | None = None
        self.ttl_sec: int | None = None
        self.mime: str | None = None

    def submit(self, sentence: str) -> None:
        text = clean_for_speech(sentence)
        if not text:
            return
        logger.info("TTS submit | sentence=%r", text[:80])
        task = asyncio.ensure_future(voice.text_to_speech_cached(text, self.tts_lang))
        self.tasks.append(task)
        self.q.put_nowait((text, task))
        if self.runner is None:
            self.runner = asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        while True:
            item = await self.q.get()
            if item is None:
                return
            text, task = item
            try:
                audios, meta = await task
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                logger.error("TTS sentence failed (reply text still returned) | %s: %s", type(e).__name__, e)
                continue
            if self.first_audio_ms is None:
                self.first_audio_ms = round((time.perf_counter() - self.t0) * 1000, 1)
                self.first_ttfb_ms = meta.get("ttfb_ms")
                logger.info("PIPELINE first audio ready | %.1f ms since request start (tts_ttfb=%s ms cached=%s)",
                            self.first_audio_ms, self.first_ttfb_ms, meta.get("cached"))
            self.audios.extend(audios)
            self.mime = self.mime or meta.get("mime")
            self.cost += meta.get("tts_cost_inr", 0.0)
            self.ms_sum += meta.get("latency_ms", 0.0)
            self.audio_sec += meta.get("audio_sec", 0.0) or 0.0
            self.chars += meta.get("input_chars", 0)
            if meta.get("cached"):
                self.cached_n += 1
                self.ttl_sec = meta.get("cache_ttl_sec", self.ttl_sec)
            self.emit({"type": "audio", "index": self.n, "text": text, "audios": audios, "mime": meta.get("mime"),
                       "language": self.tts_lang, "elapsed_ms": self.first_audio_ms})
            self.n += 1

    async def finish(self) -> None:
        self.q.put_nowait(None)
        if self.runner is not None:
            await self.runner

    def cancel(self) -> None:
        for t in self.tasks:
            t.cancel()
        if self.runner is not None:
            self.runner.cancel()


# ------------------------------------------------------------------ LLM (streaming + TTFT)
def _llm_kwargs(model: str, messages: list[dict], max_tokens: int) -> dict:
    kw = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens)
    if "gpt-oss" in model.lower():
        kw["max_tokens"] = max(max_tokens, 500)
        kw["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
    return kw


async def _chat_once(stage: str, messages: list[dict], model: str, max_tokens: int, t_req0: float) -> tuple[str, dict]:
    with Timer() as t:
        resp = await voice.groq_client.chat.completions.create(**_llm_kwargs(model, messages, max_tokens))
    reply = THINK_RE.sub("", resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    meta = {"step": stage, "latency_ms": t.ms, "ttft_ms": t.ms, "ttft_abs_ms": round((time.perf_counter() - t_req0) * 1000, 1),
            "model": model, "finish_reason": resp.choices[0].finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) or sum(_est_tokens(m["content"]) for m in messages),
            "completion_tokens": getattr(usage, "completion_tokens", None) or _est_tokens(reply),
            "reply_words": len(reply.split())}
    log_metric("llm", **meta)
    logger.info("LLM done | stage=%s (non-stream) took=%.1f ms in=%s out=%s", stage, t.ms, meta["prompt_tokens"],
                meta["completion_tokens"])
    return reply, meta


async def _chat_stream(stage: str, messages: list[dict], on_sentence, allow_search: bool, t_req0: float,
                       model: str | None = None) -> tuple[str, dict]:
    """Stream tokens; TTFT measure (aur TTFT timeout guard); complete sentences turant on_sentence() ko do.
    'SEARCH:' reply speak nahi hota - pehle chars dekh ke decide hota hai."""
    model = model or voice.GROQ_MODEL
    logger.info("LLM start | stage=%s model=%s msgs=%d stream=1 ttft_timeout=%.1fs", stage, model, len(messages), LLM_TTFT_TIMEOUT_S)
    kw = _llm_kwargs(model, messages, LLM_MAX_TOKENS)
    kw["stream"] = True

    t_start = time.perf_counter()
    ttft, ttft_abs, full, finish, usage, error = None, None, "", None, None, None
    decided, is_search = not allow_search, False
    splitter = SentenceSplitter()
    stream = None

    try:
        stream = await asyncio.wait_for(voice.groq_client.chat.completions.create(**kw), LLM_TTFT_TIMEOUT_S)
        it = stream.__aiter__()
        while True:
            try:
                if ttft is None:
                    remaining = max(0.1, LLM_TTFT_TIMEOUT_S - (time.perf_counter() - t_start))
                    chunk = await asyncio.wait_for(it.__anext__(), remaining)
                else:
                    chunk = await it.__anext__()
            except StopAsyncIteration:
                break
            u = getattr(getattr(chunk, "x_groq", None), "usage", None) or getattr(chunk, "usage", None)
            if u:
                usage = u
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.finish_reason:
                finish = ch.finish_reason
            piece = ch.delta.content or ""
            if not piece:
                continue
            if ttft is None:
                ttft = round((time.perf_counter() - t_start) * 1000, 1)
                ttft_abs = round((time.perf_counter() - t_req0) * 1000, 1)
                (logger.info if ttft <= TTFT_TARGET_MS else logger.warning)(
                    "LLM TTFT | stage=%s %.1f ms (target %.0f) | %.1f ms since request start", stage, ttft, TTFT_TARGET_MS, ttft_abs)
            full += piece
            if not decided:
                s = full.lstrip().upper()
                if len(s) < 6 and "SEARCH"[:len(s)] == s:
                    continue                                   # abhi ambiguous
                decided = True
                is_search = s.startswith("SEARCH")
                if not is_search:
                    for sent in splitter.feed(full):
                        on_sentence(sent)
                continue
            if not is_search:
                for sent in splitter.feed(piece):
                    on_sentence(sent)
    except asyncio.TimeoutError:
        error = "ttft_timeout" if ttft is None else "stream_timeout"
        logger.error("LLM %s | stage=%s > %.1fs - fallback", error, stage, LLM_TTFT_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        logger.error("LLM stream error | stage=%s %s", stage, error)
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception:  # noqa: BLE001
                pass

    if not decided:
        is_search = full.lstrip().upper().startswith("SEARCH")
        if not is_search:
            for sent in splitter.feed(full):
                on_sentence(sent)
    if is_search and not SEARCH_RE.match(full):            # "Searching your order..." jaisa normal reply
        is_search = False
        for sent in splitter.feed(full):
            on_sentence(sent)
    if not is_search:
        for sent in splitter.flush():
            on_sentence(sent)

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    reply = full.strip()
    est = usage is None
    meta = {
        "step": stage, "latency_ms": total_ms, "ttft_ms": ttft, "ttft_abs_ms": ttft_abs, "model": model,
        "finish_reason": finish, "error": error,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) or sum(_est_tokens(m["content"]) for m in messages),
        "completion_tokens": getattr(usage, "completion_tokens", None) or _est_tokens(reply),
        "reply_words": len(reply.split()), "tokens_estimated": est,
    }
    log_metric("llm", **meta)
    logger.info("LLM done | stage=%s ttft=%s total=%.1f ms in=%s out=%s%s reply=%r", stage, ttft, total_ms,
                meta["prompt_tokens"], meta["completion_tokens"], " (est)" if est else "", reply[:120])
    voice._warn_if_slow("llm", total_ms, voice.SLOW_LLM_MS)
    if finish == "length":
        logger.warning("LLM max_tokens limit reached (finish_reason=length)")
    return reply, meta


# ------------------------------------------------------------------ pipeline pieces
async def _summarize_query(q: str, llm_calls: list[dict], t_req0: float) -> str:
    """Lambi query ka intent-only version (same language). Numbers/IDs/dawai ke naam guard se wapas jodte hain,
    taaki input tokens kam hon par answer ki accuracy na jaye."""
    try:
        msgs = [{"role": "system", "content":
                 f"Rewrite the spoken request as one short question in the same language, max {SUMMARY_MAX_WORDS} words. "
                 "Keep all medicine names, numbers, order IDs, dates. Output only the question."},
                {"role": "user", "content": q}]
        r, meta = await _chat_once("summarize", msgs, voice.GROQ_FAST_MODEL, 60, t_req0)
        llm_calls.append(meta)
        if r:
            lost = query_opt.missing_entities(q, r)
            if lost:
                logger.info("PIPELINE intent guard | entities restored=%s", lost)
                r = f"{r} {' '.join(lost)}"
            logger.info("PIPELINE intent summary | %d -> %d words | %r", len(q.split()), len(r.split()), r)
            return r
    except Exception as e:  # noqa: BLE001
        logger.warning("PIPELINE query summarise failed: %s: %s", type(e).__name__, e)
    return q


def _parse_meta(text: str) -> tuple[str, dict]:
    meta: dict = {}
    m = META_RE.search(text)
    if m:
        for kv in m.group(1).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                meta[k] = v
        text = META_RE.sub("", text)
    return text.strip(), meta


def _num(v):
    try:
        return float(v) if v not in (None, "None", "") else None
    except ValueError:
        return None


async def _retrieve(q: str, via_mcp: bool) -> dict:
    with Timer() as t:
        if via_mcp:
            try:
                kb_text = await mcp.call("kb_search", {"query": q, "top_k": 0})     # 0 = dynamic top-k
            except Exception as e:  # noqa: BLE001
                logger.error("PIPELINE MCP kb_search fail: %s: %s", type(e).__name__, e)
                kb_text = ""
            kb_text, meta = _parse_meta(kb_text)
            if kb_text.startswith(mcp_tool.NO_KB_RESULT):
                kb_text = ""
            scores = SCORE_RE.findall(kb_text)
            top = float(scores[0]) if scores else None
            sources = sorted(set(SRC_RE.findall(kb_text)))
            context = HDR_RE.sub("", kb_text).strip()
            st = {"cache_hit": meta.get("cache") == "hit", "cache_ttl_sec": _num(meta.get("cache_ttl")),
                  "cache_get_ms": _num(meta.get("cache_get_ms")), "k": meta.get("k"), "term_coverage": _num(meta.get("cov")),
                  "avg_score": _num(meta.get("avg")), "reranked": meta.get("reranked") == "True",
                  "one_hit": meta.get("one_hit") == "True", "rerank_ms": _num(meta.get("rerank_ms")),
                  "dense_ms": _num(meta.get("dense_ms"))}
        else:
            hits, cached, stats = await mcp_tool.retrieve_cached(q)
            context = rag.format_context(hits)
            top = hits[0]["score"] if hits else None
            sources = sorted({h["source"] for h in hits})
            st = {"cache_hit": cached, "cache_ttl_sec": stats.get("cache_ttl_sec"), "cache_get_ms": stats.get("cache_get_ms"),
                  "k": stats.get("k"), "term_coverage": stats.get("term_coverage"), "avg_score": stats.get("avg_score"),
                  "reranked": stats.get("reranked"), "one_hit": stats.get("one_hit"),
                  "rerank_ms": stats.get("rerank_ms"), "dense_ms": stats.get("dense_ms")}
    logger.info("PIPELINE retrieval done | via=%s cache_hit=%s top_score=%s took=%.1f ms context_chars=%d "
                "k=%s reranked=%s one_hit=%s coverage=%s avg_score=%s",
                "MCP" if via_mcp else "in-process", st["cache_hit"], top, t.ms, len(context), st["k"], st["reranked"],
                st["one_hit"], st["term_coverage"], st["avg_score"])
    return {"context": context, "top": top, "sources": sources, "ms": t.ms, **st}


async def _web(search_q: str, question: str, lang: str, via_mcp: bool) -> dict:
    if not via_mcp:
        return await mcp_tool.serpapi_search(search_q, lang)
    with Timer() as tw:
        try:
            txt = await mcp.call("web_search_save", {"question": question, "search_query": search_q})
        except Exception as e:  # noqa: BLE001
            logger.error("PIPELINE MCP web_search_save fail: %s: %s", type(e).__name__, e)
            txt = mcp_tool.NO_WEB_RESULT
    found = not txt.startswith(mcp_tool.NO_WEB_RESULT)
    head, _, tail = txt.partition("\n\nSources:")
    note = tail.splitlines()[-1] if tail else txt
    cached = "(cached)" in note
    return {"found": found, "evidence": head,
            "sources": [x.strip() for x in tail.splitlines()[0].split(",") if x.strip()] if tail else [],
            "cached": cached, "billed": ("(billed" in note) or ("(billed)" in txt), "latency_ms": tw.ms}


# ------------------------------------------------------------------ pipeline
async def run_voice(audio_bytes: bytes, history: list[dict], filename: str, content_type: str,
                    on_event=None) -> dict:
    state: dict = {"tts": None}
    try:
        return await _run_voice(audio_bytes, history, filename, content_type, on_event, state)
    except Exception:
        if state["tts"] is not None:
            state["tts"].cancel()
        raise


async def _run_voice(audio_bytes, history, filename, content_type, on_event, state) -> dict:
    t0 = time.perf_counter()
    emit = on_event or (lambda ev: None)
    llm_calls: list[dict] = []
    ops = cache.begin_request()
    logger.info("PIPELINE start | audio_kb=%.1f history_msgs=%d", len(audio_bytes) / 1024, len(history))

    # 1) STT (Groq Whisper, Sarvam fallback)
    transcript, lang, stt_meta = await voice.speech_to_text(audio_bytes, filename, content_type)
    logger.info("PIPELINE transcript=%r stt_lang=%s", transcript, lang)
    if not transcript:
        logger.warning("PIPELINE empty transcript - skipping steps")
        stt_c = stt_meta.get("stt_cost_inr", 0.0)
        daily = metrics.add_request(stt=stt_c, path="empty")
        return {"transcript": "", "reply": "", "language": lang, "audios": [], "path": "empty",
                "metadata": {"status": "empty_transcript", "stt": stt_meta, "stt_cost_inr": stt_c, "total_cost_inr": stt_c,
                             "total_ms": round((time.perf_counter() - t0) * 1000, 1), **daily}}
    emit({"type": "transcript", "transcript": transcript, "language": lang})

    # reply language = user's language (prompt me explicit), TTS language = STT language (forced)
    lang_name = reply_language_name(lang, transcript)
    tts_lang = lang
    logger.info("PIPELINE language | stt=%s tts=%s (forced equal) reply_language=%r", lang, tts_lang, lang_name)

    # 2) query optimisation: local compress (free) -> lambi query ho to LLM intent-summary ‖ retrieval (parallel)
    via_mcp = mcp.active
    local_q = query_opt.truncate_words(query_opt.compress_query(transcript), MAX_QUERY_WORDS)
    smalltalk = query_opt.is_smalltalk(local_q)
    ret_task = None
    if len(local_q.split()) > QUERY_LLM_WORDS and not smalltalk:
        logger.info("PIPELINE parallel | intent-summary ‖ retrieval(local query) | words=%d", len(local_q.split()))
        ret_task = asyncio.ensure_future(_retrieve(query_opt.truncate_words(local_q, 30), via_mcp))
        q = query_opt.truncate_words(await _summarize_query(local_q, llm_calls, t0), MAX_QUERY_WORDS)
    else:
        q = local_q
    logger.info("PIPELINE query | raw_words=%d q_words=%d smalltalk=%s q=%r", len(transcript.split()), len(q.split()), smalltalk, q)

    use_hist = bool(history) and query_opt.needs_history(q)      # standalone query = history ke tokens nahi
    recent = _recent_history(history) if use_hist else []
    logger.info("PIPELINE history | used=%s msgs=%d", use_hist, len(recent))
    ver = await cache.akb_version()
    ans_key = cache.make_key("ans", ver, lang, query_opt.norm_text(q))
    cached, ans_info = (None, {}) if use_hist else await cache.aget_ex(ans_key, "ans")

    tts = _TTSStream(tts_lang, emit, t0)
    state["tts"] = tts
    fallback_text = FALLBACK["hi-IN"] if lang == "hi-IN" else FALLBACK["en-IN"]
    path, sources, top_score = "kb", [], None
    ret = {}
    web_ms = web_cost = 0.0
    web_billed = False
    reply = ""
    answer_cache_ttl = ans_info.get("ttl_sec") if cached else ANSWER_CACHE_TTL

    if cached and cached.get("reply"):
        path, reply, sources = "cache", cached["reply"], cached.get("sources", [])
        logger.info("PIPELINE answer cache HIT | ttl_left=%ss get_ms=%.2f", ans_info.get("ttl_sec"), ans_info.get("ms", 0))
        if ret_task is not None:
            ret_task.cancel()
        for s in _split_all(reply):
            tts.submit(s)

    elif smalltalk:
        path = "smalltalk"
        msgs = [{"role": "system", "content": smalltalk_system_prompt(lang_name)}, *recent, {"role": "user", "content": q}]
        _log_prompt_budget("smalltalk", msgs)
        reply, m1 = await _chat_stream("smalltalk", msgs, tts.submit, False, t0)
        llm_calls.append(m1)
        if not reply:
            path, reply = "llm_empty", fallback_text
            tts.submit(reply)

    else:
        # 3) retrieval (dynamic top-k + rerank + compression, Redis cached)
        ret = await (ret_task if ret_task is not None else _retrieve(q, via_mcp))
        context, top_score, sources = ret["context"], ret["top"], ret["sources"]
        msgs = [{"role": "system", "content": kb_system_prompt(lang_name)}, *recent,
                {"role": "user", "content": f"CONTEXT:\n{context or '(empty)'}\nQ: {q}"}]
        _log_prompt_budget("kb", msgs, context)
        reply, m1 = await _chat_stream("kb", msgs, tts.submit, True, t0)
        llm_calls.append(m1)

        m = SEARCH_RE.match(reply)
        if not reply:
            logger.warning("PIPELINE LLM empty reply (error=%s) - fallback text used", m1.get("error"))
            path, reply = "llm_empty", fallback_text
            tts.submit(reply)
        elif m:
            # 4) web fallback (SerpAPI, cached 24h, trusted domains only)
            search_q = m.group(1).strip().splitlines()[0]
            path = "web"
            logger.info("PIPELINE KB answer missing -> web search | query=%r", search_q)
            web = await _web(search_q, q, lang, via_mcp)
            web_ms = web["latency_ms"]
            web_billed = bool(web.get("billed"))
            web_cost = serpapi_cost_per_search_inr() if web_billed else 0.0
            logger.info("PIPELINE web done | found=%s cached=%s billed=%s cost=₹%.4f took=%.1f ms", web["found"],
                        web.get("cached"), web_billed, web_cost, web_ms)
            if web["found"]:
                msgs2 = [{"role": "system", "content": web_system_prompt(lang_name)}, *recent,
                         {"role": "user", "content": f"WEB:\n{web['evidence']}\nQ: {q}"}]
                _log_prompt_budget("web", msgs2, web["evidence"])
                reply, m2 = await _chat_stream("web", msgs2, tts.submit, True, t0)
                llm_calls.append(m2)
                sources = web["sources"]
                if not reply or SEARCH_RE.match(reply):
                    logger.warning("PIPELINE web LLM reply invalid - fallback text used")
                    path, reply = "web_llm_failed", fallback_text
                    tts.submit(reply)
                elif not via_mcp and not web.get("cached"):
                    _spawn(mcp_tool.save_web_answer(q, search_q, reply, web["evidence"], web["sources"]), "save_serpapi")
            else:
                path, reply, sources = "web_failed", fallback_text, []
                tts.submit(reply)

    await tts.finish()
    reply = clean_for_speech(reply)
    mismatch, reply_script = response_language_check(lang, transcript, reply)
    (logger.warning if mismatch else logger.info)(
        "PIPELINE reply language | stt=%s tts=%s reply_script=%s language_mismatch=%s", lang, tts_lang, reply_script, mismatch)

    # history me compressed query (kam tokens) + bounded size
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": reply})
    del history[:-12]

    if path in ("kb", "web", "smalltalk") and not use_hist:
        _spawn(cache.aset(ans_key, {"reply": reply, "sources": sources}, ANSWER_CACHE_TTL), "answer_cache_set")

    # ---------------- metrics
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    stt_ms = stt_meta["latency_ms"]
    llm_ms = round(sum(c["latency_ms"] for c in llm_calls), 1)
    answer_call = next((c for c in reversed(llm_calls) if c["step"] != "summarize"), None)
    first_call = answer_call or (llm_calls[0] if llm_calls else None)
    ttft_ms = first_call["ttft_ms"] if first_call else None
    ttft_abs_ms = first_call.get("ttft_abs_ms") if first_call else None
    first_audio_ms = tts.first_audio_ms if tts.first_audio_ms is not None else total_ms

    prompt_tokens = sum(c.get("prompt_tokens") or 0 for c in llm_calls)
    completion_tokens = sum(c.get("completion_tokens") or 0 for c in llm_calls)
    llm_cost = round(sum(calculate_groq_cost(c.get("prompt_tokens") or 0, c.get("completion_tokens") or 0,
                                             c["model"]) for c in llm_calls), 6)
    stt_cost = stt_meta.get("stt_cost_inr", 0.0)
    tts_cost = round(tts.cost, 6)
    total_cost_inr = round(stt_cost + tts_cost + llm_cost + web_cost + OTHER_INFRA_COST_INR, 6)
    words = max(1, len(reply.split()))
    ops_s = cache.ops_summary(ops)
    daily = metrics.add_request(stt=stt_cost, llm=llm_cost, tts=tts_cost, web=web_cost, other=OTHER_INFRA_COST_INR,
                                words=len(reply.split()), audio_sec=tts.audio_sec, path=path)

    summary = {
        "status": "ok", "path": path, "language": lang, "top_score": top_score, "via_mcp": via_mcp,
        "stt_provider": stt_meta.get("provider"), "query_words": len(q.split()),
        "raw_query_words": len(transcript.split()),
        # ---- real cost (INR)
        "stt_cost_inr": stt_cost, "llm_cost_inr": llm_cost, "tts_cost_inr": tts_cost, "web_cost_inr": web_cost,
        "vector_db_cost_inr": 0.0, "other_infra_cost_inr": OTHER_INFRA_COST_INR, "total_cost_inr": total_cost_inr,
        "web_billed": web_billed, "tts_billed_chars": tts.chars,
        # ---- tokens
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "prompt_over_budget": bool(answer_call and answer_call["prompt_tokens"] > PROMPT_TOKEN_BUDGET),
        # ---- latency
        "stt_ms": stt_ms, "retrieval_ms": ret.get("ms", 0.0), "llm_ms": llm_ms, "web_ms": web_ms,
        "tts_ms": round(tts.ms_sum, 1), "tts_ttfb_ms": tts.first_ttfb_ms, "tts_sentences": tts.n,
        "tts_cached_sentences": tts.cached_n, "tts_errors": tts.errors,
        "ttft_ms": ttft_ms, "ttft_abs_ms": ttft_abs_ms, "answer_ttft_ms": ttft_ms,
        "first_audio_ms": first_audio_ms, "core_ms": first_audio_ms, "total_ms": total_ms,
        # ---- cache / redis
        "redis_hit": bool(cached and cached.get("reply")), "retrieval_cache_hit": ret.get("cache_hit"),
        "redis_get_ms": ops_s["get_ms_total"], "redis_get_ops": ops_s["get_ops"], "redis_set_ms": cache.avg_set_ms(),
        "cache_ttl_sec": answer_cache_ttl, "retrieval_cache_ttl_sec": ret.get("cache_ttl_sec"),
        "tts_cache_ttl_sec": tts.ttl_sec, "cache_hits_by_kind": ops_s["by_kind"], "cache_backend": cache.backend(),
        "cache_hit_rate": cache.hit_rate(),
        # ---- retrieval quality (live proxies)
        "retrieval_k": ret.get("k"), "retrieval_reranked": ret.get("reranked"), "retrieval_one_hit": ret.get("one_hit"),
        "retrieval_term_coverage": ret.get("term_coverage"), "retrieval_avg_score": ret.get("avg_score"),
        # ---- language
        "response_language": tts_lang, "stt_language": lang, "reply_script": reply_script, "language_mismatch": mismatch,
        # ---- efficiency
        "reply_words": len(reply.split()), "audio_sec": round(tts.audio_sec, 2),
        "cost_per_word": round(total_cost_inr / words, 5),
        "cost_per_second_audio": round(total_cost_inr / tts.audio_sec, 5) if tts.audio_sec else None,
        "tokens_per_word": round(completion_tokens / words, 2),
        "total_tokens_per_word": round((prompt_tokens + completion_tokens) / words, 2),
        **daily,
    }
    log_metric("pipeline", **summary)

    ok = first_audio_ms <= CORE_TARGET_MS
    (logger.info if ok else logger.warning)(
        "LATENCY SUMMARY | path=%s | STT=%.0f retrieval=%.0f web=%.0f LLM(total)=%.0f TTFT=%s (abs %s) TTS_TTFB=%s | "
        "FIRST AUDIO=%.0f ms (target %.0f) [%s] | TOTAL=%.0f ms | COST = ₹%.4f",
        path, stt_ms, ret.get("ms", 0.0), web_ms, llm_ms, ttft_ms, ttft_abs_ms, tts.first_ttfb_ms, first_audio_ms,
        CORE_TARGET_MS, "OK" if ok else "OVER TARGET", total_ms, total_cost_inr)
    logger.info("COST BREAKDOWN | STT[₹%.4f] | LLM[in=%d out=%d ₹%.4f] | TTS[₹%.4f chars=%d cached=%d/%d] | Web[₹%.4f billed=%s] | "
                "OTHER[₹%.4f] | per_word=₹%.5f per_audio_sec=%s",
                stt_cost, prompt_tokens, completion_tokens, llm_cost, tts_cost, tts.chars, tts.cached_n, tts.n, web_cost,
                web_billed, OTHER_INFRA_COST_INR, summary["cost_per_word"], summary["cost_per_second_audio"])
    logger.info("CACHE SUMMARY | answer_hit=%s retrieval_hit=%s get_ops=%d get_ms=%.2f set_avg_ms=%s ttl=%ss by_kind=%s "
                "hit_rate=%s%% backend=%s", summary["redis_hit"], ret.get("cache_hit"), ops_s["get_ops"], ops_s["get_ms_total"],
                summary["redis_set_ms"], answer_cache_ttl, ops_s["by_kind"], summary["cache_hit_rate"], cache.backend())
    logger.info("EFFICIENCY | words=%d audio=%.2fs tokens/word=%.2f total_tokens/word=%.2f prompt_tokens=%d (budget %d) daily_cost=₹%.4f",
                len(reply.split()), tts.audio_sec, summary["tokens_per_word"], summary["total_tokens_per_word"],
                prompt_tokens, PROMPT_TOKEN_BUDGET, daily["daily_cost"])

    return {"transcript": transcript, "reply": reply, "language": lang, "audios": tts.audios,
            "audio_mime": tts.mime, "path": path, "sources": sources, "metadata": summary}


async def warmup() -> None:
    logger.info("WARMUP start")

    async def w_retrieval():
        if mcp.active:
            await mcp.call("kb_search", {"query": "medicine order status", "top_k": 1})
        else:
            await asyncio.to_thread(rag.warmup_models)         # embedder + reranker load

    async def w_groq():
        await voice.groq_client.chat.completions.create(
            model=voice.GROQ_MODEL, messages=[{"role": "user", "content": "hello"}], max_tokens=1)

    async def w_sarvam():
        try:
            await voice.http_client.get("https://api.sarvam.ai")
        except Exception as e:  # noqa: BLE001
            logger.info("WARMUP sarvam connect: %s", type(e).__name__)

    async def w_cache():
        await cache.aget("warmup")
        logger.info("WARMUP redis info | %s", await cache.redis_info())

    with Timer() as t:
        results = await asyncio.gather(w_retrieval(), w_groq(), w_sarvam(), w_cache(), return_exceptions=True)
    for name, r in zip(("retrieval", "groq", "sarvam", "cache"), results):
        if isinstance(r, Exception):
            logger.warning("WARMUP %s fail: %s: %s", name, type(r).__name__, r)
    logger.info("WARMUP done (parallel) | took=%.1f ms", t.ms)

    if TTS_PREWARM:
        await prewarm_tts()


async def prewarm_tts() -> None:
    """Fixed fallback lines ka TTS ek baar banake cache me rakho => runtime par in lines ka TTS cost/latency = 0."""
    phrases = [(FALLBACK["hi-IN"], "hi-IN"), (FALLBACK["en-IN"], "en-IN")]
    logger.info("WARMUP tts prewarm start | phrases=%d", len(phrases))
    results = await asyncio.gather(*[voice.text_to_speech_cached(t, l) for t, l in phrases], return_exceptions=True)
    cost = 0.0
    for (text, lng), r in zip(phrases, results):
        if isinstance(r, Exception):
            logger.warning("WARMUP tts prewarm fail | %s: %s", type(r).__name__, r)
            continue
        cost += r[1].get("tts_cost_inr", 0.0)
    if cost:
        metrics.add_request(tts=cost, path="tts_prewarm", count_request=False)
    logger.info("WARMUP tts prewarm done | one-time cost=₹%.4f", cost)
