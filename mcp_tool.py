import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

import cache
import query_opt
import rag
from logger import logger, log_metric, Timer

load_dotenv()

SERPAPI_URL = "https://serpapi.com/search.json"
SERPAPI_TIMEOUT = float(os.getenv("SERPAPI_TIMEOUT", "5"))
SERPAPI_FILE = rag.WEB_FILE
SERPAPI_HEADER = (
    "# Web Answers (SerpAPI)\n\n"
    "This file is dynamically populated. When user queries are not resolved within local "
    "healthcare and medical documentation,medical e-commerce external product/medical information is fetched and saved.\n"
)

WEB_CACHE_TTL = int(os.getenv("WEB_CACHE_TTL", "86400"))              # found result: 24h
WEB_NEG_CACHE_TTL = int(os.getenv("WEB_NEG_CACHE_TTL", "1800"))        # not-found: 10 min
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "1800"))   # key me kb-version hai => reindex par khud invalid
EVIDENCE_MAX_CHARS = int(os.getenv("EVIDENCE_MAX_CHARS", "450"))      # web evidence = LLM input tokens
SAVE_MIN_EVIDENCE_CHARS = int(os.getenv("SAVE_MIN_EVIDENCE_CHARS", "80"))
SAVE_DETAIL_CHARS = int(os.getenv("SAVE_DETAIL_CHARS", "600"))

# ---- V2: trusted-domain allowlist + staging + TTL
TRUSTED_DOMAINS = tuple(d.strip().lower() for d in os.getenv(
    "TRUSTED_DOMAINS",
    "who.int,nih.gov,cdc.gov,fda.gov,medlineplus.gov,mayoclinic.org,nhs.uk,clevelandclinic.org,"
    "health.harvard.edu,hopkinsmedicine.org,icmr.gov.in,mohfw.gov.in,cdsco.gov.in,nhp.gov.in").split(",") if d.strip())
WEB_STRICT = os.getenv("WEB_STRICT", "1") == "1"                      # 1 = sirf trusted domains ka data use/save
SERP_NUM = int(os.getenv("SERPAPI_NUM", "3"))                         # strict filter ke baad bhi kuch bache
WEB_TTL_DAYS = int(os.getenv("WEB_TTL_DAYS", "30"))                   # promoted web entry ki life
STAGING_TTL_DAYS = int(os.getenv("WEB_STAGING_TTL_DAYS", "7"))
PROMOTE_MIN_HITS = int(os.getenv("WEB_PROMOTE_MIN_HITS", "2"))        # same sawal itni baar confirm => promote
PROMOTE_MIN_DOMAINS = int(os.getenv("WEB_PROMOTE_MIN_DOMAINS", "2"))  # ya itne alag trusted domains => promote
STAGE_JACCARD = float(os.getenv("WEB_STAGE_JACCARD", "0.8"))
STAGING_PATH = rag.VECTOR_DIR / "web_staging.json"

NO_WEB_RESULT = "NO_WEB_RESULT"
NO_KB_RESULT = "NO_KB_RESULT"
_http = httpx.AsyncClient(timeout=SERPAPI_TIMEOUT)
_bg: set = set()


def _api_key() -> str:
    return os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY", "")


def _h(s: str, n: int = 12) -> str:
    return hashlib.sha1(query_opt.norm_text(s).encode("utf-8")).hexdigest()[:n]


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def is_trusted(url: str) -> bool:
    d = _domain(url)
    return bool(d) and any(d == t or d.endswith("." + t) for t in TRUSTED_DOMAINS)


def _spawn(coro, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _bg.add(task)
    task.add_done_callback(lambda t: (_bg.discard(t),
                                      (not t.cancelled()) and t.exception() and
                                      logger.error("BG task fail | %s: %r", name, t.exception())))


# ------------------------------------------------------------------ SerpAPI
async def serpapi_search(query: str, lang_code: str = "en-IN") -> dict:
    hl = lang_code.split("-")[0] if lang_code else "en"
    ckey = cache.make_key("web", hl, query_opt.cache_key_text(query))     # rephrase = same key = no paid search
    with Timer() as tc:
        cached, cinfo = await cache.aget_ex(ckey, "web")
    if cached is not None:
        logger.info("SERPAPI cache HIT | q=%r found=%s ttl_left=%ss (paid search saved)", query, cached.get("found"), cinfo["ttl_sec"])
        log_metric("serpapi", status="cache_hit", found=cached.get("found"), latency_ms=tc.ms, cache_ttl_sec=cinfo["ttl_sec"])
        return {**cached, "latency_ms": tc.ms, "cached": True, "billed": False}

    key = _api_key()
    if not key:
        logger.error("SERPAPI | SERPAPI_API_KEY missing - web search skipped")
        return {"found": False, "answer": "", "evidence": "", "sources": [], "latency_ms": 0.0, "cached": False, "billed": False}

    params = {"engine": "google", "q": query, "api_key": key, "num": SERP_NUM, "hl": hl, "gl": "in"}
    logger.info("SERPAPI start | q=%r hl=%s", query, hl)

    with Timer() as t:
        try:
            r = await _http.get(SERPAPI_URL, params=params)
        except httpx.HTTPError as e:
            logger.error("SERPAPI network error | %s: %s", type(e).__name__, e)
            log_metric("serpapi", status="error", latency_ms=t.elapsed_ms, error=str(e)[:200])
            return {"found": False, "answer": "", "evidence": "", "sources": [], "latency_ms": t.elapsed_ms,
                    "cached": False, "billed": False}

    if r.status_code != 200:
        logger.error("SERPAPI http %s | %s", r.status_code, r.text[:200])
        log_metric("serpapi", status="error", http_status=r.status_code, latency_ms=t.ms)
        return {"found": False, "answer": "", "evidence": "", "sources": [], "latency_ms": t.ms, "cached": False, "billed": False}

    j = r.json()
    dropped = 0                                          # untrusted domain ki wajah se hataye gaye results
    answer, from_box = "", False
    ab = j.get("answer_box") or {}
    if ab and WEB_STRICT and not is_trusted(ab.get("link", "")):
        dropped += 1
        ab = {}
    for field in ("answer", "result", "snippet"):
        if ab.get(field):
            answer, from_box = str(ab[field]), True
            break
    if not answer:
        kg = j.get("knowledge_graph") or {}
        if kg.get("description") and WEB_STRICT and not is_trusted((kg.get("source") or {}).get("link", "")):
            dropped += 1
        else:
            answer = kg.get("description", "")
            from_box = bool(answer)

    lines, sources = [], []
    for item in (j.get("organic_results") or [])[:SERP_NUM]:
        link = item.get("link") or ""
        if WEB_STRICT and not is_trusted(link):
            dropped += 1
            continue
        if len(lines) >= 4:
            break
        snip = (item.get("snippet") or "").strip()
        if snip:
            lines.append(f"- {item.get('title', '')}: {snip}")
        if link:
            sources.append(link)
    if not answer and lines:
        answer = lines[0][2:]
    # answer lines[0] se aaya ho to duplicate na karo (token bachat)
    head = [answer] if (answer and from_box) else []
    evidence = "\n".join(head + lines).strip()[:EVIDENCE_MAX_CHARS]
    domains = sorted({d for d in (_domain(x) for x in sources) if d})
    if dropped:
        logger.info("SERPAPI trusted-domain filter | dropped=%d kept_sources=%d domains=%s", dropped, len(sources), domains)

    found = bool(evidence)
    meta = {"status": "ok", "found": found, "latency_ms": t.ms, "organic": len(j.get("organic_results") or []),
            "has_answer_box": bool(ab), "evidence_chars": len(evidence), "untrusted_dropped": dropped,
            "trusted_domains": domains}
    log_metric("serpapi", **meta)
    logger.info("SERPAPI done | found=%s organic=%d answer_box=%s took=%.1f ms | answer=%r",
                found, meta["organic"], meta["has_answer_box"], t.ms, answer[:100])
    if not found and dropped:
        logger.warning("SERPAPI no trusted-domain result (strict mode) | q=%r dropped=%d", query, dropped)
    result = {"found": found, "answer": answer, "evidence": evidence, "sources": sources[:3], "domains": domains}
    _spawn(cache.aset(ckey, result, WEB_CACHE_TTL if found else WEB_NEG_CACHE_TTL), "web_cache_set")
    # SerpApi: HTTP 200 wali har fresh search billed hoti hai (trusted result mile ya na mile)
    return {**result, "latency_ms": t.ms, "cached": False, "billed": True}


def _load_staging() -> dict:
    try:
        return json.loads(STAGING_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_staging(d: dict) -> None:
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STAGING_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STAGING_PATH)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _save_sync(question: str, search_query: str, answer: str, evidence: str, sources: list[str]) -> tuple[str, int]:
    """Web answer -> STAGING -> (promotion rule pass hone par) KB (serpapi.md).
    Promotion: same sawal >= PROMOTE_MIN_HITS baar confirm ya >= PROMOTE_MIN_DOMAINS alag trusted domains.
    Returns (status, new_chunks)."""
    now = time.time()
    domains = sorted({_domain(u) for u in sources if _domain(u) and (is_trusted(u) or not WEB_STRICT)})
    if WEB_STRICT and not domains:
        logger.info("WEB staging rejected | no trusted domain in sources=%s", sources)
        return "rejected_untrusted", 0
    cmp_text = f"{question.strip()}\n{answer.strip()}"
    dup, best, matched = rag.semantic_duplicate(cmp_text)
    if dup:                                  # meaning pehle se KB me hai (curated ya web) -> staging/save dono skip
        logger.info("WEB save skipped: semantic duplicate | score=%.3f matched=%s", best, matched)
        return "duplicate_semantic", 0
    sq, qn = _h(search_query), query_opt.norm_text(question)
    qt = set(qn.split())
    with rag.WRITE_LOCK:
        stg = {k: v for k, v in _load_staging().items() if v.get("expires", 0) > now}      # staging TTL
        mk = next((k for k, v in stg.items() if v.get("sq") == sq or
                   _jaccard(qt, set(v.get("qn", "").split())) >= STAGE_JACCARD), None)
        if mk:
            e = stg[mk]
            e["hits"] = e.get("hits", 1) + 1
            e["domains"] = sorted(set(e.get("domains", [])) | set(domains))
            e["links"] = sorted(set(e.get("links", [])) | set(sources))[:4]
            e["last_seen"] = now
            logger.info("WEB staging confirm | key=%s hits=%d domains=%s", mk, e["hits"], e["domains"])
        else:
            mk = _h(question)
            e = {"question": question.strip(), "qn": qn, "sq": sq, "search_query": search_query,
                 "answer": answer.strip(), "details": evidence.strip()[:SAVE_DETAIL_CHARS],
                 "domains": domains, "links": sources[:4], "hits": 1, "first_seen": now, "last_seen": now,
                 "expires": now + STAGING_TTL_DAYS * 86400}
            stg[mk] = e
            logger.info("WEB staging new | key=%s domains=%s", mk, domains)

        if e["hits"] < PROMOTE_MIN_HITS and len(e["domains"]) < PROMOTE_MIN_DOMAINS:
            _save_staging(stg)
            return "staged", 0
        stg.pop(mk, None)
        _save_staging(stg)

        expires = (datetime.now() + timedelta(days=WEB_TTL_DAYS)).strftime("%Y-%m-%d")
        entry = (
            f"## Q: {e['question']}\n\n"
            f"- Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"- Origin: web\n"
            f"- Expires: {expires}\n"
            f"- Domains: {', '.join(e['domains'])}\n"
            f"- Confirmed: {e['hits']}x\n"
            f"- Search query: {e['search_query']}\n"
            f"- Links: {', '.join(e['links']) if e['links'] else 'n/a'}\n\n"
            f"Answer: {e['answer']}\n\n"
            f"Details: {e['details']}\n"
        )
        extra = ["s-" + sq, "c-" + _h(e["answer"] + "|" + e["details"][:300])]
        n = rag.append_markdown_entry(SERPAPI_FILE, entry, _h(e["question"]), SERPAPI_HEADER, dedupe_keys=extra,
                                  compare_text=f"{e['question']}\n{e['answer']}")
        return ("promoted" if n else "duplicate_kb"), n


async def save_web_answer(question: str, search_query: str, answer: str, evidence: str, sources: list[str]) -> int:
    # sirf kaam ka data save ho: khaali/chhota evidence skip
    if not answer.strip() or len(evidence.strip()) < SAVE_MIN_EVIDENCE_CHARS:
        logger.info("SERPAPI save skipped (low-value data) | evidence_chars=%d", len(evidence.strip()))
        return 0
    with Timer() as t:
        status, n = await asyncio.to_thread(_save_sync, question, search_query, answer, evidence, sources)
    logger.info("SERPAPI save | status=%s file=%s new_chunks=%d took=%.1f ms", status, SERPAPI_FILE, n, t.ms)
    log_metric("serpapi_save", status=status, new_chunks=n, latency_ms=t.ms)
    return n


# ------------------------------------------------------------------ retrieval (cached)
async def retrieve_cached(query: str, top_k: int = 0) -> tuple[list[dict], bool, dict]:
    """Redis/memory retrieval cache. Key = kb-version + top_k + order-free content terms (rephrase bhi hit karte hain)."""
    ver = await cache.akb_version()
    key = cache.make_key("ret", ver, top_k, query_opt.cache_key_text(query))
    c, info = await cache.aget_ex(key, "ret")
    if c is not None:
        logger.info("RETRIEVAL cache HIT | q=%r ttl_left=%ss get_ms=%.2f", query[:80], info["ttl_sec"], info["ms"])
        return c["hits"], True, {**c.get("stats", {}), "cache_ttl_sec": info["ttl_sec"], "cache_get_ms": info["ms"]}
    logger.info("RETRIEVAL cache MISS | q=%r get_ms=%.2f -> rag.retrieve", query[:80], info["ms"])
    hits, stats = await asyncio.to_thread(rag.retrieve, query, top_k)
    _spawn(cache.aset(key, {"hits": hits, "stats": stats}, RETRIEVAL_CACHE_TTL), "ret_cache_set")
    return hits, False, {**stats, "cache_ttl_sec": RETRIEVAL_CACHE_TTL, "cache_get_ms": info["ms"]}


# ------------------------------------------------------------------ MCP TOOLS
async def kb_search(query: str, top_k: int = 0) -> str:
    """Healthcare & Medical E-Commerce knowledge base search (rerank + compression). top_k=0 means dynamic."""
    logger.info("TOOL kb_search | query=%r top_k=%d", query, top_k)
    hits, cached, st = await retrieve_cached(query, max(0, min(top_k, 8)))
    meta = (f"#meta cache={'hit' if cached else 'miss'} k={st.get('k')} returned={st.get('returned')} "
            f"cov={st.get('term_coverage')} avg={st.get('avg_score')} ctx_chars={st.get('ctx_chars')} "
            f"reranked={st.get('reranked')} one_hit={st.get('one_hit')} cache_ttl={st.get('cache_ttl_sec')} "
            f"cache_get_ms={st.get('cache_get_ms')} dense_ms={st.get('dense_ms')} rerank_ms={st.get('rerank_ms')}")
    if not hits:
        return f"{NO_KB_RESULT}\n{meta}"
    body = "\n\n".join(
        f"[{i}] score={h['score']} | {h['source']} | {h['section']}\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    return f"{body}\n{meta}"


async def kb_list_sources() -> str:
    """List indexed files in vector database."""
    logger.info("TOOL kb_list_sources")
    s = await asyncio.to_thread(rag.stats)
    lines = [f"data_dir={s['data_dir']} | total_chunks={s['total_chunks']} | embed_model={s['embed_model']}"]
    for name, m in s["files"].items():
        lines.append(f"- {name} ({m.get('type')}) chunks={m.get('chunks')} indexed_at={m.get('indexed_at')}")
    return "\n".join(lines)


async def kb_reindex(force: bool = False) -> str:
    """Incrementally reindex updated or new Markdown/PDF files from data folder."""
    logger.info("TOOL kb_reindex | force=%s", force)
    s = await asyncio.to_thread(rag.sync_data_dir, force)
    purge = await asyncio.to_thread(rag.purge_expired)
    logger.info("TOOL kb_reindex | purge=%s", purge)
    return (f"purged_expired={purge['expired_entries_removed']} indexed={s['indexed']} skipped={s['skipped']} removed={s['removed']} "
            f"chunks_added={s['chunks_added']} chunks_deleted={s['chunks_deleted']} "
            f"total_chunks={s['total_chunks']} took_ms={s['sync_ms']}")


async def web_search_save(question: str, search_query: str = "") -> str:
    """Web search fallback for general external medical specifications (cached; saves only non-duplicate data)."""
    q = (search_query or question).strip()
    logger.info("TOOL web_search_save | question=%r search_query=%r", question, q)
    res = await serpapi_search(q)
    if not res["found"]:
        return f"{NO_WEB_RESULT}: No search results obtained.{' (billed)' if res.get('billed') else ''}"
    if res.get("cached"):
        note = "(cached)"
    else:
        _spawn(save_web_answer(question, q, res["answer"] or res["evidence"][:300], res["evidence"],
                               res["sources"]), "save_serpapi")
        note = f"(billed; saving to {SERPAPI_FILE} in background if new)"
    return (f"Answer: {res['answer']}\n\nDetails:\n{res['evidence']}\n\nSources: {', '.join(res['sources'])}\n{note}")
