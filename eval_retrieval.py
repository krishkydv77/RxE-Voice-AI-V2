"""Offline retrieval quality: precision@k, recall@k, MRR + context size + latency.

Usage:
  python eval_retrieval.py --set eval_set.json            # default k (dynamic)
  python eval_retrieval.py --set eval_set.json --k 1,2,3  # k sweep: best precision/recall/token trade-off dhundo

eval_set.json:
  [{"q": "return policy kya hai", "sources": ["returns.md"], "keywords": ["7 days", "unopened"]}, ...]
  - sources  : in files me se koi bhi chunk aaye => chunk 'relevant'
  - keywords : chunk text me koi keyword mile => chunk 'relevant'; recall = kitne keywords context me mile
"""
import argparse
import json
import statistics

import rag


def _relevant(hit: dict, case: dict) -> bool:
    if hit["source"] in case.get("sources", []):
        return True
    low = hit["text"].lower()
    return any(k.lower() in low for k in case.get("keywords", []))


def evaluate(cases: list[dict], k: int) -> dict:
    p, r, mrr, chars, ms = [], [], [], [], []
    for c in cases:
        hits, st = rag.retrieve(c["q"], top_k=k)
        rel = [_relevant(h, c) for h in hits]
        p.append(sum(rel) / len(hits) if hits else 0.0)
        kws = c.get("keywords", [])
        ctx = " ".join(h["text"] for h in hits).lower()
        if kws:
            r.append(sum(1 for w in kws if w.lower() in ctx) / len(kws))
        elif c.get("sources"):
            r.append(len({h["source"] for h in hits} & set(c["sources"])) / len(set(c["sources"])))
        first = next((i for i, x in enumerate(rel, 1) if x), None)
        mrr.append(1 / first if first else 0.0)
        chars.append(st.get("ctx_chars", 0))
        ms.append(st["ms"])
    return {"k": k or "dynamic", "n": len(cases), "precision": round(statistics.mean(p), 3),
            "recall": round(statistics.mean(r), 3) if r else None, "mrr": round(statistics.mean(mrr), 3),
            "avg_ctx_chars": round(statistics.mean(chars)), "avg_ctx_tokens_est": round(statistics.mean(chars) / 3.5),
            "avg_ms": round(statistics.mean(ms), 1), "p95_ms": round(sorted(ms)[int(len(ms) * 0.95) - 1], 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval_set.json")
    ap.add_argument("--k", default="0", help="comma list, 0 = dynamic")
    a = ap.parse_args()
    cases = json.load(open(a.set, encoding="utf-8"))
    rag.warmup_models()
    rows = [evaluate(cases, int(k)) for k in a.k.split(",")]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    json.dump(rows, open("logs/retrieval_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
