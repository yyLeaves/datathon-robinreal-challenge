"""
CLI hybrid retrieval over precomputed BGE-M3 embeddings.

Default mode: weighted-sum hybrid (alpha=0.7, dense-heavy) + optional hard post-filters.

Usage:
  python search.py "Wohnung in Zürich mit Balkon, bis 3000 CHF" --top 10
  python search.py "family apartment in Basel" --mode dense --top 5
  python search.py "..." --mode hybrid_ws --alpha 0.7 --city Zürich --rooms_min 3 --price_max 3500
"""
from __future__ import annotations
import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel
import onnxruntime as ort
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DENSE_NPZ    = HERE / "embeddings_bge_dense.npz"
SPARSE_JSONL = HERE / "embeddings_sparse.jsonl"
TEXTS_JSONL  = HERE / "listing_texts.jsonl"
BGE_MODEL    = "/workshop/models/bge-m3"
ONNX_PATH    = "/workshop/models/bge-m3-onnx"

_META_CACHE: dict[str, dict] = {}


def parse_meta(text: str) -> dict:
    m_rooms = re.search(r"(\d+(?:\.\d+)?)\s*rooms", text)
    m_area  = re.search(r"Living area:\s*(\d+(?:\.\d+)?)\s*m", text)
    m_price = re.search(r"CHF\s*(\d+)\s*/month", text)
    m_city  = re.search(r"Located in\s+([^,\.]+)", text)
    return {
        "rooms": float(m_rooms.group(1)) if m_rooms else None,
        "area":  float(m_area.group(1))  if m_area else None,
        "price": int(m_price.group(1))   if m_price else None,
        "city":  m_city.group(1).strip() if m_city else None,
    }


def load_indexes():
    print("Loading BGE-M3 (sparse) + ONNX (dense)...", flush=True)
    import torch
    bge = BGEM3FlagModel(BGE_MODEL, use_fp16=torch.cuda.is_available())
    ort_tok  = AutoTokenizer.from_pretrained(ONNX_PATH)
    ort_sess = ort.InferenceSession(ONNX_PATH + "/model.onnx",
                                    providers=["CPUExecutionProvider"])

    print("Loading dense index...", flush=True)
    d = np.load(DENSE_NPZ)
    ids = list(d["ids"])
    vecs = d["vecs"].astype(np.float32)
    faiss_index = faiss.IndexFlatIP(vecs.shape[1])
    faiss_index.add(vecs)

    print("Loading sparse inverted index...", flush=True)
    inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
    sparse_ids = []
    with open(SPARSE_JSONL) as f:
        for doc_idx, line in enumerate(f):
            rec = json.loads(line)
            sparse_ids.append(rec["id"])
            for tok_id, w in rec["weights"].items():
                inverted[tok_id].append((doc_idx, float(w)))
    assert ids == sparse_ids, "Dense/sparse id orderings diverge"

    print("Loading texts...", flush=True)
    texts = {}
    with open(TEXTS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            texts[r["id"]] = r["text"]
            _META_CACHE[r["id"]] = parse_meta(r["text"])

    print(f"Ready. N={len(ids)}, dim={vecs.shape[1]}, sparse_tokens={len(inverted)}", flush=True)
    return bge, ort_tok, ort_sess, ids, faiss_index, inverted, texts


def encode_query(bge, ort_tok, ort_sess, q: str):
    # dense via ONNX (~25ms)
    enc = ort_tok([q], padding=True, truncation=True, max_length=512)
    feeds = {k: np.array(v) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    qv = ort_sess.run(None, feeds)[1].astype(np.float32)
    qv /= np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-9)
    # sparse via PyTorch BGE-M3
    out = bge.encode([q], return_dense=False, return_sparse=True, return_colbert_vecs=False)
    qw = {str(k): float(v) for k, v in out["lexical_weights"][0].items()}
    return qv, qw


def dense_search(qv, faiss_index, ids, k):
    scores, idxs = faiss_index.search(qv, k)
    return [(ids[i], float(s)) for i, s in zip(idxs[0], scores[0])]


def sparse_search(qw, inverted, ids, k):
    scores: dict[int, float] = defaultdict(float)
    for tok_id, qweight in qw.items():
        for doc_idx, dw in inverted.get(tok_id, []):
            scores[doc_idx] += qweight * dw
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [(ids[i], s) for i, s in top]


def minmax(results):
    if not results:
        return {}
    s = [sc for _, sc in results]
    lo, hi = min(s), max(s)
    rng = hi - lo if hi > lo else 1.0
    return {lid: (sc - lo) / rng for lid, sc in results}


def rrf(dres, sres, k_rrf, top_k):
    scores = defaultdict(float)
    for rank, (lid, _) in enumerate(dres, 1): scores[lid] += 1.0 / (k_rrf + rank)
    for rank, (lid, _) in enumerate(sres, 1): scores[lid] += 1.0 / (k_rrf + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def ws(dres, sres, alpha, top_k):
    dn, sn = minmax(dres), minmax(sres)
    keys = set(dn) | set(sn)
    fused = {lid: alpha * dn.get(lid, 0.0) + (1 - alpha) * sn.get(lid, 0.0) for lid in keys}
    return sorted(fused.items(), key=lambda x: -x[1])[:top_k]


def apply_filters(results, *, city=None, rooms_min=None, rooms_max=None, area_min=None, price_max=None):
    out = []
    for lid, score in results:
        m = _META_CACHE.get(lid, {})
        if city and not (m.get("city") and city.lower() in m["city"].lower()):
            continue
        if rooms_min is not None and (m.get("rooms") is None or m["rooms"] < rooms_min - 0.01):
            continue
        if rooms_max is not None and (m.get("rooms") is None or m["rooms"] > rooms_max + 0.01):
            continue
        if area_min is not None and (m.get("area") is None or m["area"] < area_min):
            continue
        if price_max is not None and (m.get("price") is None or m["price"] > price_max):
            continue
        out.append((lid, score))
    return out


def search(bge, ort_tok, ort_sess, ids, faiss_index, inverted, texts, query, *,
           mode="hybrid_ws", alpha=0.7, k_rrf=60,
           top=10, fetch_k=200,
           city=None, rooms_min=None, rooms_max=None, area_min=None, price_max=None):
    qv, qw = encode_query(bge, ort_tok, ort_sess, query)
    dres = dense_search(qv, faiss_index, ids, fetch_k)
    sres = sparse_search(qw, inverted, ids, fetch_k)

    if mode == "dense":
        fused = dres
    elif mode == "sparse":
        fused = sres
    elif mode == "hybrid_rrf":
        fused = rrf(dres, sres, k_rrf, fetch_k)
    elif mode == "hybrid_ws":
        fused = ws(dres, sres, alpha, fetch_k)
    else:
        raise ValueError(f"unknown mode: {mode}")

    filtered = apply_filters(
        fused, city=city, rooms_min=rooms_min, rooms_max=rooms_max,
        area_min=area_min, price_max=price_max,
    )
    return filtered[:top]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--mode", choices=["dense", "sparse", "hybrid_rrf", "hybrid_ws"], default="hybrid_ws")
    p.add_argument("--alpha", type=float, default=0.7, help="weighted-sum dense weight (hybrid_ws)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing (hybrid_rrf)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--fetch-k", type=int, default=200)
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--rooms-min", type=float, default=None)
    p.add_argument("--rooms-max", type=float, default=None)
    p.add_argument("--area-min", type=float, default=None)
    p.add_argument("--price-max", type=float, default=None)
    args = p.parse_args()

    bge, ort_tok, ort_sess, ids, faiss_index, inverted, texts = load_indexes()
    hits = search(
        bge, ort_tok, ort_sess, ids, faiss_index, inverted, texts, args.query,
        mode=args.mode, alpha=args.alpha, k_rrf=args.rrf_k,
        top=args.top, fetch_k=args.fetch_k,
        city=args.city, rooms_min=args.rooms_min, rooms_max=args.rooms_max,
        area_min=args.area_min, price_max=args.price_max,
    )
    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode}  (filters: city={args.city} rooms=[{args.rooms_min},{args.rooms_max}] area>={args.area_min} price<={args.price_max})")
    print(f"Results: {len(hits)}\n")
    for rank, (lid, score) in enumerate(hits, 1):
        m = _META_CACHE.get(lid, {})
        t = texts.get(lid, "")[:180].replace("\n", " ")
        print(f"[{rank:2d}] {score:.4f}  id={lid}  {m.get('city','?')}  r={m.get('rooms')}  a={m.get('area')}  p={m.get('price')}")
        print(f"     {t}")


if __name__ == "__main__":
    main()
