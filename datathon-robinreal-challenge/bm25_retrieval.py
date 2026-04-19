"""Multilingual BM25 retrieval pipeline.

Builds one index per language (en/de/fr/it) from listing_texts_{lang}.jsonl.
For each query, searches all 4 indexes and fuses with MAX score per listing.

Experiments available:
  --mode word       word tokenization (baseline)
  --mode ngram      character 3-gram tokenization
  --mode hybrid     word + char-ngram combined (default)
  --variant okapi   BM25Okapi  (default)
  --variant plus    BM25Plus
  --variant bm25s   bm25s native (fastest)

Usage:
  python retrieval.py "3 Zimmer Wohnung Zürich unter 2500 CHF"
  python retrieval.py "bright apartment in Lausanne with balcony" --top 10
  python retrieval.py --benchmark          # run all variants on test queries
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import os as _os
_DATA_DIR = Path(_os.environ.get("RETRIEVAL_DATA_DIR", "/workshop/retrieval_aws/data"))
OUTPUTS = Path(_os.environ.get("BM25_OUTPUTS_DIR", str(Path(__file__).parent.parent / "bm25_outputs")))
LANGS = ["en", "de", "fr", "it"]

# ── Tokenisers ────────────────────────────────────────────────────────────────

_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


def tok_word(text: str) -> list[str]:
    return [t.lower() for t in _SPLIT.split(_ascii(text)) if len(t) > 1]


def tok_ngram(text: str, n: int = 3) -> list[str]:
    t = _ascii(text).lower()
    return [t[i:i+n] for i in range(len(t) - n + 1)]


def tok_hybrid(text: str) -> list[str]:
    return tok_word(text) + tok_ngram(text, 3)


_TOKENISERS = {
    "word":   tok_word,
    "ngram":  tok_ngram,
    "hybrid": tok_hybrid,
}


# ── Index ─────────────────────────────────────────────────────────────────────

@dataclass
class Index:
    ids: list[str]
    mode: str
    variant: str
    _model: Any = field(default=None, repr=False)

    @staticmethod
    def build(records: list[dict], mode: str, variant: str) -> "Index":
        tok = _TOKENISERS[mode]
        ids = [r["id"] for r in records]
        tokenized = [tok(r["text"]) for r in records]

        if variant == "bm25s":
            import bm25s
            model = bm25s.BM25()
            model.index(tokenized)
        elif variant == "okapi":
            from rank_bm25 import BM25Okapi
            model = BM25Okapi(tokenized)
        elif variant == "plus":
            from rank_bm25 import BM25Plus
            model = BM25Plus(tokenized)
        elif variant == "l":
            from rank_bm25 import BM25L
            model = BM25L(tokenized)
        else:
            raise ValueError(variant)

        idx = Index(ids=ids, mode=mode, variant=variant)
        idx._model = model
        return idx

    def scores(self, query: str) -> np.ndarray:
        tok = _TOKENISERS[self.mode]
        q_tokens = tok(query)
        if not q_tokens:
            return np.zeros(len(self.ids))

        if self.variant == "bm25s":
            import bm25s
            result = self._model.get_scores(q_tokens)
            return np.array(result, dtype=float)
        else:
            return np.array(self._model.get_scores(q_tokens), dtype=float)


# ── Multilingual retriever ────────────────────────────────────────────────────

class MultiLingualRetriever:
    def __init__(self, mode: str = "hybrid", variant: str = "bm25s"):
        self.mode = mode
        self.variant = variant
        self.indexes: dict[str, Index] = {}
        self.id_to_pos: dict[str, dict[str, int]] = {}  # lang -> {id: pos}

    def load_and_build(self) -> None:
        all_ids: set[str] = set()
        lang_records: dict[str, list[dict]] = {}

        for lang in LANGS:
            path = OUTPUTS / f"listing_texts_{lang}.jsonl"
            if not path.exists():
                print(f"  [WARN] missing {path.name}")
                continue
            records = [json.loads(l) for l in open(path)]
            lang_records[lang] = records
            all_ids.update(r["id"] for r in records)

        self.all_ids = sorted(all_ids)
        self.n = len(self.all_ids)
        global_pos = {lid: i for i, lid in enumerate(self.all_ids)}

        for lang, records in lang_records.items():
            t0 = time.time()
            self.indexes[lang] = Index.build(records, self.mode, self.variant)
            self.id_to_pos[lang] = {r["id"]: i for i, r in enumerate(records)}
            print(f"  [{lang}] {len(records)} docs indexed in {time.time()-t0:.2f}s")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        # Collect per-language scores aligned to global_ids
        fused = np.zeros(self.n)
        per_lang: dict[str, np.ndarray] = {}
        global_pos = {lid: i for i, lid in enumerate(self.all_ids)}

        for lang, idx in self.indexes.items():
            raw = idx.scores(query)
            # Normalise to [0,1]
            mx = raw.max()
            if mx > 0:
                raw = raw / mx
            # Map to global array
            aligned = np.zeros(self.n)
            for local_i, lid in enumerate(idx.ids):
                g = global_pos.get(lid)
                if g is not None:
                    aligned[g] = raw[local_i]
            per_lang[lang] = aligned
            fused = np.maximum(fused, aligned)

        top_idx = np.argsort(fused)[::-1][:top_k]
        results = []
        for gi in top_idx:
            lid = self.all_ids[gi]
            results.append({
                "id": lid,
                "score": float(fused[gi]),
                "per_lang": {l: float(per_lang[l][gi]) for l in LANGS if l in per_lang},
            })
        return results


# ── Benchmark ─────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "3 Zimmer Wohnung Zürich unter 2500 CHF",
    "bright apartment Lausanne balcony",
    "4.5 Zimmer Haus Bern Garten ruhig",
    "appartement 2 pièces Genève pas cher",
    "studio Zürich furnished temporary",
    "Wohnung Basel Balkon Haustiere erlaubt",
    "3 room apartment near public transport Winterthur",
    "appartamento Lugano 3 locali terrazza",
]

# Pseudo-relevance: a result is "relevant" if its EN text contains ALL of these keywords
# (lowercase, partial match). Used for MRR computation.
_RELEVANCE: list[list[str]] = [
    ["zürich", "zimmer"],
    ["lausanne", "balcon"],
    ["bern", "zimmer"],
    ["genève", "genf", "geneva", "pièces", "rooms"],
    ["zürich", "furnished"],
    ["basel", "balkon", "balcony", "haustiere", "pets"],
    ["winterthur"],
    ["lugano", "terraz"],
]


def _is_relevant(text_en: str, keywords: list[str]) -> bool:
    t = text_en.lower()
    # for location alternatives, any one keyword in a group counts
    # group format: comma-separated alternatives within an item
    for kw in keywords:
        alts = [a.strip() for a in kw.split(",")]
        if not any(a in t for a in alts):
            return False
    return True

# Load records once for display
_TITLES: dict[str, str] = {}
_TEXTS_EN: dict[str, str] = {}


def _load_titles() -> None:
    if _TITLES:
        return
    path = OUTPUTS / "listing_texts_en.jsonl"
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            text = r["text"]
            _TEXTS_EN[r["id"]] = text
            title = text.split(".")[0].replace("Title: ", "")
            _TITLES[r["id"]] = title[:60]


def _mrr(hits: list[dict], keywords: list[str], k: int = 10) -> float:
    for rank, h in enumerate(hits[:k], 1):
        text = _TEXTS_EN.get(h["id"], "")
        if _is_relevant(text, keywords):
            return 1.0 / rank
    return 0.0


def benchmark() -> None:
    configs = [
        ("word",   "okapi"),
        ("word",   "plus"),
        ("word",   "bm25s"),
        ("ngram",  "bm25s"),
        ("hybrid", "bm25s"),
        ("hybrid", "okapi"),
        ("hybrid", "plus"),
    ]

    _load_titles()
    summary: list[dict] = []

    for mode, variant in configs:
        label = f"{mode}/{variant}"
        print(f"\n{'='*64}")
        print(f"  Config: {label}")
        print('='*64)
        r = MultiLingualRetriever(mode=mode, variant=variant)
        t_build = time.time()
        r.load_and_build()
        build_time = time.time() - t_build

        query_times: list[float] = []
        lang_wins: dict[str, int] = {l: 0 for l in LANGS}
        mrr_scores: list[float] = []

        for q, rel_kws in zip(TEST_QUERIES, _RELEVANCE):
            t0 = time.time()
            hits = r.search(q, top_k=10)
            query_times.append(time.time() - t0)

            mrr_val = _mrr(hits, rel_kws, k=10)
            mrr_scores.append(mrr_val)

            if hits:
                pl = hits[0]["per_lang"]
                winner = max(pl, key=lambda l: pl[l])
                lang_wins[winner] += 1
                top_title = _TITLES.get(hits[0]["id"], "?")
                hit_rank = next(
                    (i+1 for i, h in enumerate(hits[:10])
                     if _is_relevant(_TEXTS_EN.get(h["id"],""), rel_kws)), None
                )
                print(f"  [{winner}] Q: {q[:40]:<40} MRR={mrr_val:.2f}  "
                      f"(1st rel @{hit_rank or '>10'})  → {top_title[:38]}")

        avg_q = np.mean(query_times) * 1000
        mean_mrr = float(np.mean(mrr_scores))
        print(f"\n  Build: {build_time:.2f}s | Avg query: {avg_q:.1f}ms | "
              f"MRR@10: {mean_mrr:.3f} | Lang wins: {lang_wins}")
        summary.append({
            "config": label, "build_s": build_time,
            "query_ms": avg_q, "mrr": mean_mrr, "lang_wins": lang_wins,
        })

    print("\n\n=== Summary (sorted by MRR@10) ===")
    print(f"{'Config':<18} {'MRR@10':>7} {'Build':>7} {'Query':>8}  Lang wins")
    for s in sorted(summary, key=lambda x: -x["mrr"]):
        wins = " ".join(f"{l}:{s['lang_wins'][l]}" for l in LANGS)
        print(f"  {s['config']:<16} {s['mrr']:>7.3f} {s['build_s']:>6.2f}s {s['query_ms']:>7.1f}ms  {wins}")


def demo(query: str, top_k: int, mode: str, variant: str) -> None:
    _load_titles()
    print(f"\nMode={mode}  Variant={variant}  Query: {query!r}")
    r = MultiLingualRetriever(mode=mode, variant=variant)
    r.load_and_build()
    hits = r.search(query, top_k=top_k)
    print(f"\n{'Rank':<5} {'Score':<8} {'EN':>6} {'DE':>6} {'FR':>6} {'IT':>6}  Title")
    print("-" * 80)
    for i, h in enumerate(hits, 1):
        pl = h["per_lang"]
        title = _TITLES.get(h["id"], "?")
        print(f"  {i:<3} {h['score']:.4f}  "
              f"{pl.get('en',0):.3f}  {pl.get('de',0):.3f}  "
              f"{pl.get('fr',0):.3f}  {pl.get('it',0):.3f}  "
              f"{title}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--mode", choices=["word", "ngram", "hybrid"], default="hybrid")
    p.add_argument("--variant", choices=["okapi", "plus", "l", "bm25s"], default="bm25s")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()

    if args.benchmark:
        benchmark()
    elif args.query:
        demo(args.query, args.top, args.mode, args.variant)
    else:
        # default: run benchmark
        benchmark()


if __name__ == "__main__":
    main()
