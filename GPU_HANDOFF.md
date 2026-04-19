# GPU Handoff — Embedding Two New CSV Files

## Context

This is a Swiss real estate retrieval pipeline. Two new CSV files need to be embedded using BGE-M3 (dense + sparse). The current CPU instance is too slow (~70 min per run). You have a GPU server — use it.

The existing pipeline has already processed `structured_data_withimages_updated.csv` + `structured_data_withoutimages-1776412361239.csv` (13,644 listings). The new CSVs follow **the same schema** (52 columns, same column names).

---

## What You Need to Produce

For the two new CSV files, generate:

| Output file | Description |
|---|---|
| `embeddings_bge_dense.npz` | BGE-M3 dense vectors `(N, 1024)` float32, L2-normalised |
| `embeddings_sparse.jsonl` | BGE-M3 sparse lexical weights, one JSON per line |
| `listing_texts.jsonl` | Plain-text representation of each listing (input to embeddings) |

---

## Step 1 — Environment Setup

```bash
pip install "transformers==4.51.3" \
    FlagEmbedding datasets peft sentencepiece \
    huggingface_hub einops accelerate \
    faiss-gpu   # or faiss-cpu if GPU FAISS not needed
```

**Critical version pin:** `transformers==4.51.3`
- Lower: doesn't support Qwen3 architecture
- Higher (5.x): breaks FlagEmbedding (missing `is_torch_fx_available`)

---

## Step 2 — Transfer Files from Current Server

```bash
# From the CPU server (54.184.212.11), run:
rsync -avz -e "ssh -i ~/gpu-embedding-key.pem" \
  /workshop/ \
  <gpu-user>@<gpu-ip>:/workspace/
```

Key files needed on GPU server:
```
listing_text.py          # text builder + column normaliser
build_text_mapping.py    # CSV → listing_texts.jsonl
build_embeddings.py      # JSONL → dense .npz + sparse .jsonl
models/bge-m3/           # ~2.3GB — BGE-M3 weights
```

Place your two new CSV files in `/workspace/` as well.

---

## Step 3 — Build Text Mapping

Edit `build_text_mapping.py` to point to your new CSVs:

```python
CSV_PATHS = [
    Path("/workspace/your_new_file_1.csv"),
    Path("/workspace/your_new_file_2.csv"),
]
OUT_PATH = Path("/workspace/listing_texts.jsonl")
```

Then run:
```bash
python3 build_text_mapping.py
```

Expected output:
```
Processing your_new_file_1.csv...
Processing your_new_file_2.csv...
Done. Written: XXXX  Duplicates skipped: 0  Empty skipped: 0
```

---

## Step 4 — Run Embeddings (GPU)

`build_embeddings.py` auto-detects GPU via `torch.cuda.is_available()`.

**Add this at the top of `build_embeddings.py`** if not already there:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

And pass to model:
```python
bge = BGEM3FlagModel(BGE_PATH, use_fp16=True)   # fp16 on GPU
qwen = AutoModel.from_pretrained(...).to(DEVICE)
```

Run:
```bash
python3 build_embeddings.py
```

On a T4 GPU: ~5 min for 13k listings. On A10/V100: ~2–3 min.

---

## Step 5 — Verify Outputs

```python
import numpy as np, json

# Dense
d = np.load("embeddings_bge_dense.npz")
print(d["ids"].shape, d["vecs"].shape, d["vecs"].dtype)
# Expected: (N,)  (N, 1024)  float32

# Sparse — check token ID keys (must be numeric strings, not decoded words)
with open("embeddings_sparse.jsonl") as f:
    rec = json.loads(f.readline())
print("sparse key sample:", list(rec["weights"].keys())[:3])
# Expected: ['12', '18799', '13']  ← numeric token IDs, NOT 'Title', 'in', etc.
```

**Important:** sparse weights must use **token ID strings** as keys (e.g. `"18799"`), NOT decoded subword strings (e.g. `"Wohn"`). This is already correct in `build_embeddings.py` line 78:
```python
weights = {str(k): round(float(v), 5) for k, v in lw.items()}
```

---

## Key Files Reference

### `listing_text.py`
Converts a raw CSV row → natural language string.
- `normalise_row(raw)` — maps CSV column names, strips NULLs, strips HTML
- `build_listing_text(row)` — builds the text
- `build_query_text(query, qwen=False)` — adds instruction prefix for Qwen3

### `build_text_mapping.py`
Reads CSVs → deduplicates by `id` → writes `listing_texts.jsonl`

### `build_embeddings.py`
Two passes over `listing_texts.jsonl`:
1. **Qwen3-0.6B dense** → `embeddings_dense.npz` *(not needed for the API, kept for reference)*
2. **BGE-M3 sparse** → `embeddings_sparse.jsonl`

For the GPU run you only need BGE-M3 dense + sparse. Skip Qwen3 by commenting out that section.

### `build_bge_dense.py`
Dedicated BGE-M3 dense encoder — produces `embeddings_bge_dense.npz` + builds a FAISS index. **This is what the API uses.** Run this instead of the Qwen3 section.

---

## CSV Schema (both new files must match)

The pipeline expects these columns (others are ignored):

| Column | Used as |
|---|---|
| `id` | unique listing ID |
| `title` | listing title |
| `object_description` | description text (HTML-stripped, first 1500 chars) |
| `object_city` | city |
| `object_state` | canton |
| `number_of_rooms` | room count |
| `area` | m² |
| `price` | CHF/month |
| `floor` | floor number |
| `year_built` | construction year |
| `distance_public_transport` | metres to PT |
| `prop_balcony` | 0/1 |
| `prop_elevator` | 0/1 |
| `prop_parking` | 0/1 |
| `prop_garage` | 0/1 |
| `prop_fireplace` | 0/1 |
| `prop_child_friendly` | 0/1 |
| `animal_allowed` | 0/1 |
| `is_new_building` | 0/1 |

NULL values stored as the string `"NULL"` are handled automatically.

---

## Running the API (after embeddings are done)

```bash
pip install fastapi uvicorn onnxruntime optimum

# Export BGE-M3 to ONNX (6x faster query encoding, ~25ms vs 160ms)
optimum-cli export onnx \
  --model /workspace/models/bge-m3 \
  --task feature-extraction \
  --opset 17 \
  /workspace/models/bge-m3-onnx/

# Start API
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:
```
GET /health
GET /search?q=<query>&top_k=5&mode=hybrid|dense|sparse
```

Latency with ONNX on CPU: ~30–45ms. On GPU: <10ms expected.
