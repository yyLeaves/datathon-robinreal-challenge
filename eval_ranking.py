"""
Evaluate ranking with different BM25 top-N settings on stored 56 queries.
Compares: no_bm25 vs bm25_top5 vs bm25_top10

Usage: python3.11 eval_ranking.py
"""
import json, os, sys, time
from pathlib import Path
from collections import defaultdict

# Set ANTHROPIC_API_KEY in your environment before running

sys.path.insert(0, "/workshop")
sys.path.insert(0, "/workshop/datathon-robinreal-challenge")

import search as S
import bm25_retrieval as B
import numpy as np
from app.config import get_settings
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts
from app.participant.ranking import rank_listings, _load_corpus
from app.harness.search_service import to_hard_filter_params, search_with_relaxation, _resolve_near_place

settings = get_settings()

print("Loading BGE-M3 + indexes...", flush=True)
_bge, _ort_tok, _ort_sess, _ids, _faiss, _inv, _texts = S.load_indexes()
print("Loading BM25...", flush=True)
_bm25 = B.MultiLingualRetriever(mode="hybrid", variant="bm25s")
_bm25.load_and_build()
print("Loading ranking corpus...", flush=True)
_load_corpus()
print("Ready.\n", flush=True)

stored = json.load(open("/workshop/harness_results.json"))
all_queries = [(g, r["query"]) for g, results in stored["groups"].items() for r in results]


def run_query(query: str, bm25_top_n: int) -> dict:
    t0 = time.time()
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = 5000
    hard_facts.offset = 0
    hard_facts = _resolve_near_place(hard_facts)
    soft_facts = extract_soft_facts(query)
    if hard_facts.neighborhood:
        soft_facts["neighborhoods"] = hard_facts.neighborhood

    qv, qw = S.encode_query(_bge, _ort_tok, _ort_sess, query)
    soft_facts["_query_dense"] = qv
    soft_facts["_query_sparse"] = qw
    soft_facts["_query"] = query

    if bm25_top_n > 0:
        bm25_hits = _bm25.search(query, top_k=bm25_top_n)
        soft_facts["_bm25_top"] = {h["id"]: rank + 1 for rank, h in enumerate(bm25_hits)}

    candidates, relaxations = search_with_relaxation(settings.db_path, to_hard_filter_params(hard_facts))
    candidates = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(candidates, soft_facts)
    elapsed = round((time.time() - t0) * 1000)

    top10_ids = [r.listing_id for r in ranked[:10]]
    top10_scores = [round(r.score, 6) for r in ranked[:10]]
    return {
        "n_candidates": len(candidates),
        "n_results": len(ranked),
        "latency_ms": elapsed,
        "top10_ids": top10_ids,
        "top10_scores": top10_scores,
    }


SETTINGS = [0, 5, 10]
results_by_setting = {n: [] for n in SETTINGS}

for i, (group, query) in enumerate(all_queries, 1):
    print(f"[{i:2d}/56] {group} | {query[:60]}...", flush=True)
    row = {"group": group, "query": query}
    for n in SETTINGS:
        r = run_query(query, n)
        row[f"bm25_{n}"] = r
        print(f"  bm25_top{n}: cands={r['n_candidates']} ranked={r['n_results']} "
              f"top3={r['top10_ids'][:3]} scores={r['top10_scores'][:3]}", flush=True)
    results_by_setting["all"] = results_by_setting.get("all", [])
    results_by_setting["all"].append(row)

out_path = Path("/workshop/eval_ranking_results.json")
out_path.write_text(json.dumps(results_by_setting, indent=2, ensure_ascii=False))
print(f"\nSaved → {out_path}")

# Summary stats
print("\n=== SUMMARY ===")
rows = results_by_setting["all"]
for n in SETTINGS:
    avg_cands = sum(r[f"bm25_{n}"]["n_candidates"] for r in rows) / len(rows)
    avg_lat = sum(r[f"bm25_{n}"]["latency_ms"] for r in rows) / len(rows)
    # Overlap: how often does top-1 match across settings vs bm25_0
    if n > 0:
        matches = sum(1 for r in rows if r[f"bm25_{n}"]["top10_ids"][:1] == r["bm25_0"]["top10_ids"][:1])
        print(f"bm25_top{n:2d}: avg_candidates={avg_cands:.0f} avg_latency={avg_lat:.0f}ms "
              f"top1_same_as_no_bm25={matches}/56")
    else:
        print(f"bm25_top{n:2d}: avg_candidates={avg_cands:.0f} avg_latency={avg_lat:.0f}ms (baseline)")
