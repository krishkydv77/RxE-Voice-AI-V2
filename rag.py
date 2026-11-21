import argparse
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import cache
import query_opt
from logger import logger, log_metric, Timer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data_basic"))
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", BASE_DIR / "vectordb"))
MANIFEST_PATH = VECTOR_DIR / "manifest.json"
COLLECTION = "healthcare_knowledge_base"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
ADD_BATCH = 64

# ---- retrieval tuning
TOP_K = int(os.getenv("TOP_K", "3"))                          # max k
TOP_K_MIN = min(int(os.getenv("TOP_K_MIN", "2")), TOP_K)      # short queries ke liye k
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "6"))             # rerank pool chhota = rerank latency kam            # rerank ke liye candidate pool
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))         # isse kam cosine = irrelevant, drop
DENSE_MARGIN = float(os.getenv("RAG_DENSE_MARGIN", "0.15"))   # top score se zyada neeche wale chunks drop
CONFIDENT_SCORE = float(os.getenv("RAG_CONFIDENT_SCORE", "0.72"))
CONFIDENT_GAP = float(os.getenv("RAG_CONFIDENT_GAP", "0.08"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "420"))      # ~130 tokens: prompt <=200 token target
ONE_HIT_SCORE = float(os.getenv("RAG_ONE_HIT_SCORE", "0.80"))       # top hit itna confident + gap => sirf 1 chunk bhejo
ONE_HIT_GAP = float(os.getenv("RAG_ONE_HIT_GAP", "0.10"))
RERANK_TEXT_CHARS = int(os.getenv("RERANK_TEXT_CHARS", "600"))

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")   # small (~80MB) English cross-encoder
RERANK_LATIN_ONLY = os.getenv("RERANK_LATIN_ONLY", "1") == "1"            # English-only model -> Hindi query pe skip
RERANK_DROP_GAP = float(os.getenv("RERANK_DROP_GAP", "6.0"))

NEAR_DUP_SCORE = float(os.getenv("SAVE_NEAR_DUP_SCORE", "0.88"))      # semantic same => save nahi
ENTRY_KEYS_MAX = 5000

# ---- V2: web-sourced chunks (SerpAPI) ke liye
WEB_FILE = "serpapi.md"
WEB_PENALTY = float(os.getenv("WEB_SCORE_PENALTY", "0.90"))             # web chunk ka score isse multiply (curated > web)
CURATED_OVERRIDE = float(os.getenv("CURATED_OVERRIDE_SCORE", "0.55"))   # curated hit isse upar => web hits drop

SUPPORTED_EXT = {".md", ".markdown", ".pdf"}

WRITE_LOCK = threading.RLock()

_embeddings = None
_store = None
_reranker = None
_reranker_failed = False
_RERANK_LOCK = threading.Lock()


# ------------------------------------------------------------------ models / store
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        logger.info("RAG embed model load start | model=%s", EMBED_MODEL)
        with Timer() as t:
            _embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
        logger.info("RAG embed model ready | model=%s load_ms=%.1f", EMBED_MODEL, t.ms)
    return _embeddings


def get_reranker():
    global _reranker, _reranker_failed
    if not RERANK_ENABLED or _reranker_failed:
        return None
    if _reranker is None:
        with _RERANK_LOCK:
            if _reranker is None and not _reranker_failed:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    logger.info("RAG reranker load start | model=%s", RERANK_MODEL)
                    with Timer() as t:
                        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
                    logger.info("RAG reranker ready | load_ms=%.1f", t.ms)
                except Exception as e:  # noqa: BLE001
                    _reranker_failed = True
                    logger.warning("RAG reranker disabled (load failed): %s: %s", type(e).__name__, e)
    return _reranker


def get_store():
    global _store
    if _store is None:
        with WRITE_LOCK:
            if _store is None:
                from langchain_chroma import Chroma

                VECTOR_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("RAG vector store open | dir=%s collection=%s", VECTOR_DIR, COLLECTION)
                _store = Chroma(
                    collection_name=COLLECTION,
                    embedding_function=get_embeddings(),
                    persist_directory=str(VECTOR_DIR),
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info("RAG vector store ready | existing_chunks=%d", _store._collection.count())
    return _store


def warmup_models() -> None:
    emb = get_embeddings()
    get_store()
    emb.embed_query("warmup")
    rr = get_reranker()
    if rr is not None:
        list(rr.rerank("warmup", ["warmup document"]))


# ------------------------------------------------------------------ manifest
def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read manifest, creating new: %s", e)
    return {}


def _save_manifest(m: dict) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _chunk_id(name: str, text: str) -> str:
    """Content-hash id: same text -> same id, isliye sirf naye/badle chunks embed hote hain."""
    return f"{name}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


# ------------------------------------------------------------------ chunking
def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "। ", ". ", " ", ""],
    )


_WEB_ENTRY = re.compile(r"^## (.+?)\n(.*?)(?=^## |\Z)", re.S | re.M)
DAY_SEC = 86400


def _to_epoch(day: str) -> float | None:
    try:
        return datetime.strptime(day.strip(), "%Y-%m-%d").timestamp()
    except ValueError:
        return None


def _web_meta(text: str) -> dict[str, dict]:
    """'## Q: ...' entries jinme '- Origin: web' hai: origin / expires_at / domains metadata (TTL ke liye)."""
    out: dict[str, dict] = {}
    for m in _WEB_ENTRY.finditer(text):
        body = m.group(2)
        if not re.search(r"^- Origin:\s*web", body, re.M | re.I):
            continue
        meta: dict = {"origin": "web"}
        e = re.search(r"^- Expires:\s*(\S+)", body, re.M)
        ts = _to_epoch(e.group(1)) if e else None
        if ts:
            meta["expires_at"] = ts + DAY_SEC           # expiry din ke end tak valid
        d = re.search(r"^- Domains:\s*(.+)$", body, re.M)
        if d:
            meta["domains"] = d.group(1).strip()[:200]
        out[m.group(1).strip()] = meta
    return out


def split_markdown_text(text: str, source: str) -> list[Document]:
    webmeta = _web_meta(text)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)
    chunks = _recursive_splitter().split_documents(sections)
    out: list[Document] = []
    for c in chunks:
        keys = [k for k in ("h2", "h3") if k in c.metadata] or [k for k in ("h1",) if k in c.metadata]
        path = " > ".join(c.metadata[k] for k in keys)
        body = c.page_content.strip()
        if not body:
            continue
        if path and not body.lstrip().startswith("#"):
            body = f"[{path}] {body}"
        md = {"source": source, "section": path or source, "type": "md"}
        md.update(webmeta.get(str(c.metadata.get("h2", "")).strip(), {}))
        out.append(Document(page_content=body, metadata=md))
    return out


def load_and_chunk(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    logger.info("RAG load start | file=%s type=%s size=%.1f KB", path.name, ext, path.stat().st_size / 1024)

    if ext in (".md", ".markdown"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs = split_markdown_text(text, path.name)
        logger.info("RAG load md | file=%s chars=%d chunks=%d", path.name, len(text), len(docs))
        return docs

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(path)).load()
        with_text = [p for p in pages if p.page_content.strip()]
        logger.info("RAG load pdf | file=%s pages=%d pages_with_text=%d", path.name, len(pages), len(with_text))
        if not with_text:
            logger.warning("RAG no text in pdf | file=%s", path.name)
            return []
        chunks = _recursive_splitter().split_documents(with_text)
        out = []
        for c in chunks:
            page_no = int(c.metadata.get("page", 0)) + 1
            out.append(Document(page_content=c.page_content.strip(),
                                metadata={"source": path.name, "section": f"page {page_no}",
                                          "page": page_no, "type": "pdf"}))
        logger.info("RAG chunk pdf | file=%s chunks=%d", path.name, len(out))
        return out

    logger.warning("RAG unsupported file skipped | file=%s", path.name)
    return []


# ------------------------------------------------------------------ indexing (incremental)
def _add_docs(docs: list[Document], ids: list[str], label: str) -> float:
    store = get_store()
    with Timer() as t:
        for i in range(0, len(docs), ADD_BATCH):
            with Timer() as tb:
                store.add_documents(docs[i:i + ADD_BATCH], ids=ids[i:i + ADD_BATCH])
            logger.info("RAG embed+store batch | %s chunks %d-%d/%d took=%.1f ms",
                        label, i + 1, min(i + ADD_BATCH, len(docs)), len(docs), tb.ms)
    return t.ms


def _delete_source(name: str) -> int:
    store = get_store()
    old = store.get(where={"source": name}).get("ids", [])
    if old:
        store.delete(ids=old)
    return len(old)


def index_file(path: Path, manifest: dict, force: bool = False, sha: str | None = None) -> dict:
    """Chunk-level incremental: sirf naye chunks embed, sirf purane (hataye gaye) chunks delete."""
    st = path.stat()
    sha = sha or _file_sha(path)
    with Timer() as t_load:
        docs = load_and_chunk(path)

    new: dict[str, Document] = {}
    for d in docs:
        new.setdefault(_chunk_id(path.name, d.page_content), d)

    store = get_store()
    old_ids = set(store.get(where={"source": path.name}).get("ids", []))
    if force:
        to_del, to_add = set(old_ids), list(new)
    else:
        to_del = old_ids - set(new)
        to_add = [i for i in new if i not in old_ids]

    if to_del:
        store.delete(ids=list(to_del))
    embed_ms = _add_docs([new[i] for i in to_add], to_add, path.name) if to_add else 0.0

    prev = manifest.get(path.name, {})
    manifest[path.name] = {
        **({"entry_keys": prev["entry_keys"]} if prev.get("entry_keys") else {}),
        "sha256": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
        "chunks": len(new), "type": path.suffix.lower().lstrip("."),
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
    }
    logger.info("RAG incremental | file=%s total=%d added=%d deleted=%d kept=%d",
                path.name, len(new), len(to_add), len(to_del), len(new) - len(to_add))
    log_metric("rag_index", file=path.name, chunks=len(new), added=len(to_add), deleted=len(to_del),
               load_chunk_ms=t_load.ms, embed_store_ms=embed_ms)
    return {"chunks": len(new), "added": len(to_add), "deleted": len(to_del)}


def sync_data_dir(force: bool = False) -> dict:
    with WRITE_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
        logger.info("RAG sync start | dir=%s files_found=%d force=%s", DATA_DIR, len(files), force)
        if not files:
            logger.warning("RAG data folder empty: %s", DATA_DIR)

        manifest = _load_manifest()
        store = get_store()
        if manifest and store._collection.count() == 0:
            logger.warning("RAG manifest present but vector DB empty - forcing reindex")
            force = True

        summary = {"indexed": [], "skipped": [], "removed": [], "chunks_added": 0, "chunks_deleted": 0}
        with Timer() as t:
            for p in files:
                st = p.stat()
                m = manifest.get(p.name, {})
                # fast path: size+mtime same -> file hash bhi nahi nikalte
                if not force and m.get("size") == st.st_size and m.get("mtime_ns") == st.st_mtime_ns:
                    summary["skipped"].append(p.name)
                    continue
                sha = _file_sha(p)
                if not force and m.get("sha256") == sha:
                    m.update(size=st.st_size, mtime_ns=st.st_mtime_ns)
                    manifest[p.name] = m
                    summary["skipped"].append(p.name)
                    continue
                logger.info("RAG index needed | file=%s reason=%s", p.name,
                            "force" if force else ("new" if p.name not in manifest else "changed"))
                info = index_file(p, manifest, force, sha)
                summary["indexed"].append(p.name)
                summary["chunks_added"] += info["added"]
                summary["chunks_deleted"] += info["deleted"]
                _save_manifest(manifest)

            present = {p.name for p in files}
            for name in [n for n in manifest if n not in present]:
                removed = _delete_source(name)
                manifest.pop(name)
                summary["removed"].append(name)
                summary["chunks_deleted"] += removed
                logger.info("RAG file deleted - chunks purged | file=%s removed=%d", name, removed)
            _save_manifest(manifest)

        summary["total_chunks"] = store._collection.count()
        summary["sync_ms"] = t.ms
        if summary["indexed"] or summary["removed"]:
            cache.bump_kb_version()          # purane cached answers/retrievals invalid
        log_metric("rag_sync", **summary)
        logger.info("RAG sync done | indexed=%s skipped=%s removed=%s total_chunks=%d took=%.1f ms",
                    summary["indexed"], summary["skipped"], summary["removed"],
                    summary["total_chunks"], t.ms)
        return summary


def rebuild() -> dict:
    global _store
    with WRITE_LOCK:
        logger.warning("RAG REBUILD: collection + manifest purge")
        try:
            get_store().delete_collection()
        except Exception as e:  # noqa: BLE001
            logger.info("RAG rebuild delete_collection skipped (%s)", e)
        _store = None
        MANIFEST_PATH.unlink(missing_ok=True)
        return sync_data_dir(force=True)


def semantic_duplicate(text: str, threshold: float = NEAR_DUP_SCORE) -> tuple[bool, float, str]:
    """Poore KB (curated + web, sabhi files) me dekho ki yahi meaning pehle se hai ya nahi.
    returns (is_dup, best_score, matched_source)."""
    try:
        now = time.time()
        res = get_store().similarity_search_with_score(text, k=3)
    except Exception as e:  # noqa: BLE001
        logger.info("RAG semantic-dup check skipped: %s", e)
        return False, 0.0, ""
    best, src = 0.0, ""
    for doc, dist in res:
        exp = doc.metadata.get("expires_at")
        if exp and exp < now:
            continue
        sc = 1 - float(dist)
        if sc > best:
            best, src = sc, doc.metadata.get("source", "?")
    return best >= threshold, round(best, 3), src


def append_markdown_entry(filename: str, entry_md: str, entry_key: str, file_header: str,
                          dedupe_keys: list[str] | None = None, compare_text: str | None = None) -> int:
    """Entry sirf tab save hoti hai jab same question / same search query / same content /
    near-duplicate pehle se saved na ho."""
    with WRITE_LOCK:
        manifest = _load_manifest()
        meta = manifest.get(filename, {})
        known: list[str] = list(meta.get("entry_keys", []))
        keys = [entry_key, *(dedupe_keys or [])]

        if any(k in known for k in keys):
            logger.info("RAG entry duplicate (key) - skipped | file=%s key=%s", filename, entry_key)
            return 0
        store = get_store()
        if store.get(ids=[f"{filename}:entry-{entry_key}:0"]).get("ids"):     # legacy format
            logger.info("RAG entry duplicate (legacy id) - skipped | file=%s key=%s", filename, entry_key)
            return 0

        docs = split_markdown_text(entry_md, filename)
        dup, best, matched = (False, 0.0, "")
        if docs and NEAR_DUP_SCORE > 0:
            # metadata lines (Saved on / Expires...) hata kar sirf meaning compare karo, poore KB ke against
            dup, best, matched = semantic_duplicate(compare_text or _clean_chunk(docs[0].page_content))
        if dup:
            logger.info("RAG entry duplicate (semantic %.3f >= %.2f vs %s) - skipped | file=%s key=%s",
                        best, NEAR_DUP_SCORE, matched, filename, entry_key)
            meta["entry_keys"] = (known + [k for k in keys if k not in known])[-ENTRY_KEYS_MAX:]
            manifest[filename] = meta
            _save_manifest(manifest)
            return 0

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / filename
        if not path.exists():
            path.write_text(file_header.rstrip() + "\n", encoding="utf-8")
            logger.info("RAG created file | %s", path)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + entry_md.strip() + "\n")
        logger.info("RAG entry written | file=%s key=%s chars=%d", filename, entry_key, len(entry_md))

        uniq: dict[str, Document] = {}
        for d in docs:
            uniq.setdefault(_chunk_id(filename, d.page_content), d)
        embed_ms = _add_docs(list(uniq.values()), list(uniq), f"{filename}#{entry_key}") if uniq else 0.0

        st = path.stat()
        manifest[filename] = {
            **meta, "sha256": _file_sha(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "chunks": meta.get("chunks", 0) + len(uniq), "type": "md",
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
            "entry_keys": (known + keys)[-ENTRY_KEYS_MAX:],
        }
        _save_manifest(manifest)
        log_metric("rag_entry", file=filename, key=entry_key, chunks=len(uniq), embed_store_ms=embed_ms)
        return len(uniq)


# ------------------------------------------------------------------ retrieval
_SENT = re.compile(r"(?<=[.!?।])\s+|\n+")
_STOP = frozenset(
    "the a an is are was were of to in on for and or with what how do does can i you my me it this that be at by "
    "from as if your में की के का है हैं से को और कि क्या कैसे मुझे आप".split())


def _terms(s: str) -> set[str]:
    return {t for t in query_opt.norm_text(s).split() if len(t) > 2 and t not in _STOP}


def _is_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.7


_META_LINE = re.compile(r"^\s*-\s*(?:Saved on|Origin|Expires|Domains|Confirmed|Search query|Links)\s*:.*$", re.I | re.M)


def _clean_chunk(text: str) -> str:
    """LLM ko bekaar tokens na jaye: web-entry metadata lines, markdown # hatao."""
    t = _META_LINE.sub("", text)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"\n{2,}", "\n", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def dynamic_k(query: str) -> int:
    return TOP_K_MIN if len(query.split()) <= 6 else TOP_K


def _compress_text(qterms: set[str], text: str, max_chars: int) -> str:
    """Contextual compression (extractive): query se match karne wale sentences hi rakho."""
    if len(text) <= max_chars:
        return text
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if len(sents) <= 1:
        return text[:max_chars]
    scored = sorted(((len(qterms & _terms(s)), -i, i) for i, s in enumerate(sents)), reverse=True)
    keep, used = {0}, len(sents[0])           # first sentence = heading/context, hamesha rakho
    for sc, _, i in scored:
        if i in keep:
            continue
        if sc == 0 and len(keep) >= 2:
            break
        if used + len(sents[i]) + 1 > max_chars:
            continue
        keep.add(i)
        used += len(sents[i]) + 1
    return " ".join(sents[i] for i in sorted(keep))[:max_chars]


def compress_hits(query: str, hits: list[dict], max_chars: int) -> list[dict]:
    if not hits:
        return hits
    qt = _terms(query)
    weights = [1.5] + [1.0] * (len(hits) - 1)       # top hit ko zyada budget
    tot = sum(weights)
    for h, w in zip(hits, weights):
        h["text"] = _compress_text(qt, h["text"], max(200, int(max_chars * w / tot)))
    return hits


def _should_rerank(query: str, hits: list[dict]) -> bool:
    if get_reranker() is None:
        return False
    if RERANK_LATIN_ONLY and not _is_latin(query):
        return False
    if hits[0]["score"] >= CONFIDENT_SCORE and hits[0]["score"] - hits[1]["score"] >= CONFIDENT_GAP:
        return False                                  # dense result already confident -> rerank skip
    return True


def _rerank(query: str, hits: list[dict]) -> list[dict]:
    rr = get_reranker()
    scores = list(rr.rerank(query, [h["text"][:RERANK_TEXT_CHARS] for h in hits]))   # ek hi batch me saare candidates
    for h, s in zip(hits, scores):
        h["rerank"] = float(s)
    return sorted(hits, key=lambda h: h["rerank"], reverse=True)


def retrieve(query: str, top_k: int = 0, max_chars: int = CONTEXT_MAX_CHARS) -> tuple[list[dict], dict]:
    """dense fetch -> min-score filter -> (optional) cross-encoder rerank -> dynamic cut -> compression."""
    with Timer() as t:
        k = top_k if top_k > 0 else dynamic_k(query)
        fetch = max(CANDIDATES, k) if RERANK_ENABLED else k
        with Timer() as t_dense:
            res = get_store().similarity_search_with_score(query, k=fetch)

        hits, seen, expired, now = [], set(), 0, time.time()
        for doc, dist in res:
            md = doc.metadata
            exp = md.get("expires_at")
            if exp and exp < now:                       # TTL khatam -> web chunk ignore
                expired += 1
                continue
            origin = md.get("origin", "curated")
            sc = round(1 - float(dist), 3)
            if origin == "web":
                sc = round(sc * WEB_PENALTY, 3)         # curated-over-web ranking
            text = _clean_chunk(doc.page_content)
            key = text[:200]
            if sc < MIN_SCORE or key in seen:
                continue
            seen.add(key)
            hits.append({"text": text, "source": md.get("source", "?"),
                         "section": md.get("section", ""), "page": md.get("page"),
                         "score": sc, "origin": origin})
        hits.sort(key=lambda h: h["score"], reverse=True)
        web_dropped = 0
        if any(h["origin"] != "web" and h["score"] >= CURATED_OVERRIDE for h in hits):
            web_dropped = sum(1 for h in hits if h["origin"] == "web")
            hits = [h for h in hits if h["origin"] != "web"]        # curated KB ne answer de diya -> web noise hatao
        stats = {"k": k, "fetched": len(res), "after_filter": len(hits), "reranked": False,
                 "expired_skipped": expired, "web_dropped": web_dropped,
                 "dense_ms": t_dense.ms, "rerank_ms": 0.0, "compress_ms": 0.0, "one_hit": False}

        if hits:
            if len(hits) > 1 and _should_rerank(query, hits):
                try:
                    with Timer() as t_rr:
                        hits = _rerank(query, hits)
                    stats["reranked"], stats["rerank_ms"] = True, t_rr.ms
                except Exception as e:  # noqa: BLE001
                    logger.warning("RAG rerank failed -> dense order: %s: %s", type(e).__name__, e)
            if stats["reranked"]:
                top = hits[0]["rerank"]
                hits = [h for h in hits if h["rerank"] >= top - RERANK_DROP_GAP]
            else:
                top = hits[0]["score"]
                hits = [h for h in hits if h["score"] >= top - DENSE_MARGIN]
            if (not stats["reranked"] and len(hits) > 1 and hits[0]["score"] >= ONE_HIT_SCORE
                    and hits[0]["score"] - hits[1]["score"] >= ONE_HIT_GAP):
                hits, stats["one_hit"] = hits[:1], True          # confident top hit -> baaki chunks = sirf extra tokens
            with Timer() as t_c:
                hits = compress_hits(query, hits[:k], max_chars)
            stats["compress_ms"] = t_c.ms
        stats["returned"] = len(hits)
        ctx = " ".join(h["text"] for h in hits)
        qt = _terms(query)
        stats["ctx_chars"] = len(ctx)
        stats["avg_score"] = round(sum(h["score"] for h in hits) / len(hits), 3) if hits else None
        # recall proxy: query ke kitne terms context me mile (cross-language me meaningless -> None)
        stats["term_coverage"] = round(len(qt & _terms(ctx)) / len(qt), 2) if (qt and _is_latin(query)) else None
    stats["ms"] = t.ms
    logger.info("RAG retrieve | q=%r %s took=%.1f ms", query[:80], stats, t.ms)
    log_metric("rag_retrieve", **stats, top_score=hits[0]["score"] if hits else None)
    return hits, stats


def purge_expired() -> dict:
    """TTL khatam hui web entries: vector chunks delete + serpapi.md se entry hatao."""
    now = time.time()
    removed_chunks = removed_entries = 0
    with WRITE_LOCK:
        try:
            ids = get_store().get(where={"expires_at": {"$lt": now}}).get("ids", [])
        except Exception as e:  # noqa: BLE001
            logger.warning("RAG purge query failed: %s: %s", type(e).__name__, e)
            ids = []
        if ids:
            get_store().delete(ids=ids)
            removed_chunks = len(ids)
        path = DATA_DIR / WEB_FILE
        if path.exists():
            parts = re.split(r"(?m)^(?=## )", path.read_text(encoding="utf-8"))
            keep = [parts[0]]
            for part in parts[1:]:
                m = re.search(r"^- Expires:\s*(\S+)", part, re.M)
                ts = _to_epoch(m.group(1)) if m else None
                if ts and ts + DAY_SEC < now:
                    removed_entries += 1
                else:
                    keep.append(part)
            if removed_entries:
                path.write_text("".join(keep), encoding="utf-8")
                manifest = _load_manifest()
                if WEB_FILE in manifest:
                    st = path.stat()
                    manifest[WEB_FILE].update(sha256=_file_sha(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                                              entry_keys=[])      # expired question dobara save ho sake
                    _save_manifest(manifest)
    if removed_chunks or removed_entries:
        cache.bump_kb_version()
    res = {"expired_chunks_removed": removed_chunks, "expired_entries_removed": removed_entries}
    logger.info("RAG purge_expired | %s", res)
    log_metric("rag_purge", **res)
    return res


def search(query: str, k: int = TOP_K) -> list[dict]:
    return retrieve(query, k)[0]


def format_context(hits: list[dict], max_chars: int = CONTEXT_MAX_CHARS + 200) -> str:
    parts, used = [], 0
    for h in hits:
        piece = h["text"][: max(0, max_chars - used)]
        if piece:
            parts.append(piece)
            used += len(piece)
        if used >= max_chars:
            break
    return "\n---\n".join(parts)


def stats() -> dict:
    m = _load_manifest()
    return {"files": m, "total_chunks": get_store()._collection.count(), "data_dir": str(DATA_DIR),
            "embed_model": EMBED_MODEL}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Healthcare data_basic -> vector DB")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--purge", action="store_true", help="expired web entries hatao")
    ap.add_argument("--query")
    a = ap.parse_args()
    out = rebuild() if a.rebuild else sync_data_dir()
    if a.purge:
        out["purge"] = purge_expired()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.query:
        for h in search(a.query):
            print(f"{h['score']:.3f} | {h['source']} | {h['section']}\n   {h['text'][:160]}\n")
