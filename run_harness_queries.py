"""Run all test queries through the datathon harness pipeline and save results."""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Set ANTHROPIC_API_KEY in your environment before running

REPO = Path("/workshop/datathon-robinreal-challenge")
sys.path.insert(0, str(REPO))

from app.config import get_settings
from app.harness.bootstrap import bootstrap_database
from app.core.hard_filters import search_listings
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts
from app.participant.ranking import rank_listings
from app.harness.search_service import to_hard_filter_params, search_with_relaxation, _resolve_near_place

settings = get_settings()
print("Bootstrapping database...")
bootstrap_database(db_path=settings.db_path, raw_data_dir=settings.raw_data_dir)
print(f"DB ready at {settings.db_path}")

sys.path.insert(0, str(Path("/workshop")))
from test_queries import EN, DE, MIXED, GROUPS


def run_query(query: str) -> dict:
    t0 = time.time()
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = 5000
    hard_facts.offset = 0
    hard_facts = _resolve_near_place(hard_facts)
    soft_facts = extract_soft_facts(query)
    if hard_facts.neighborhood:
        soft_facts["neighborhoods"] = hard_facts.neighborhood
    candidates, relaxations = search_with_relaxation(settings.db_path, to_hard_filter_params(hard_facts))
    candidates = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(candidates, soft_facts)
    elapsed = round((time.time() - t0) * 1000)

    out = {
        "query": query,
        "hard_facts": hard_facts.model_dump(exclude_none=True),
        "soft_facts": {k: v for k, v in soft_facts.items() if v},
        "total_candidates": len(candidates),
        "total_results": len(ranked),
        "latency_ms": elapsed,
        "results": [
            {
                "listing_id": r.listing_id,
                "score": r.score,
                "reason": r.reason,
                "title": r.listing.title,
                "city": r.listing.city,
                "canton": r.listing.canton,
                "price_chf": r.listing.price_chf,
                "rooms": r.listing.rooms,
                "area_sqm": r.listing.living_area_sqm,
                "features": r.listing.features,
                "available_from": r.listing.available_from,
                "street": r.listing.street,
            }
            for r in ranked
        ],
    }
    if relaxations is not None:
        out["relaxations_applied"] = relaxations
    return out


results = {"run_at": datetime.utcnow().isoformat(), "groups": {}}

for group_name, queries in GROUPS:
    print(f"\n{'='*60}")
    print(f"GROUP: {group_name} ({len(queries)} queries)")
    print("="*60)
    group_results = []
    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:80]}...", end=" ", flush=True)
        try:
            result = run_query(query)
            print(f"→ {result['total_candidates']} candidates, {result['total_results']} results ({result['latency_ms']}ms)")
            group_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            group_results.append({"query": query, "error": str(e)})
    results["groups"][group_name] = group_results

out_path = Path("/workshop/harness_results.json")
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nSaved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
