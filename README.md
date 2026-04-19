# Swiss Real Estate Retrieval Pipeline

Hybrid retrieval system for Swiss apartment listings using dense (Qwen3-0.6B) and sparse (BGE-M3) embeddings.

---

## Data

| File | Rows | Notes |
|---|---|---|
| `structured_data_withimages_updated.csv` | 6,887 | Primary scrape; all rows have images |
| `structured_data_withoutimages-1776412361239.csv` | 6,757 | No image URLs; zero overlap with above |
| **Total unique listings** | **13,644** | |

Both files share the same 52-column schema. Key columns used for text building:

`title`, `object_description`, `object_city`, `object_state`, `number_of_rooms`, `area`, `price`, `floor`, `year_built`, `distance_public_transport`, `prop_balcony`, `prop_elevator`, `prop_parking`, `prop_garage`, `prop_fireplace`, `prop_child_friendly`, `animal_allowed`, `is_new_building`

---

## Models

| Model | Role | Local path |
|---|---|---|
| `Qwen/Qwen3-0.6B` | Dense retrieval scorer (main) | `models/Qwen3-0.6B/` |
| `BAAI/bge-m3` | Sparse scorer (replaces BM25) | `models/bge-m3/` |

Downloaded via `huggingface_hub.snapshot_download`, safetensors only (`.pt` / TF / Flax excluded).

### Dependency note

FlagEmbedding 1.3.5 requires `transformers==4.51.x` — versions 4.47 and below don't support the `qwen3` architecture; versions 5.x removed `is_torch_fx_available` which FlagEmbedding imports internally.

```
pip install transformers==4.51.3 FlagEmbedding datasets peft sentencepiece \
            huggingface_hub sentence-transformers einops accelerate
```

---

## Pipeline

### Step 1 — Text representation (`listing_text.py`)

Converts a raw CSV row into a natural-language string for embedding or BM25.

**Key functions:**

- `normalise_row(raw)` — maps CSV column names to canonical names, coerces feature flags to `int`, strips literal `"NULL"` / `"nicht verfügbar"` strings, strips HTML from description.
- `build_listing_text(row)` — concatenates scalar fields and active boolean features into a readable sentence sequence. Features with value `0` or NULL are omitted.
- `build_query_text(query, qwen=False)` — prepends the Qwen3 instruction prefix for query encoding; pass-through for BGE-M3.

**Instruction prefix used for Qwen3 queries:**
```
Instruct: Given a Swiss real estate search query, retrieve the most relevant apartment listing.
Query: <query>
```

### Step 2 — Build text mapping (`build_text_mapping.py`)

Reads both CSVs, deduplicates by `id`, and writes one JSON record per listing.

```bash
python3 build_text_mapping.py
```

Output: `listing_texts.jsonl`
```jsonl
{"id": "24544", "text": "Title: Moderne Loft-Wohnung ... Description: ..."}
{"id": "24545", "text": "Title: Bureaux à louer ..."}
```

### Step 3 — Validate mapping (`check_mapping.py`)

Spot-checks first 20 JSONL rows against the source CSV to confirm no fields or features are silently dropped.

```bash
python3 check_mapping.py
```

Prints a table with per-row status (`OK` / `-` = source was NULL / `MISS` = value present but not found in text) and lists any active features that failed to appear in the output text.

### Step 4 — Generate embeddings (`build_embeddings.py`)

Encodes all 13,644 listings in batches of 64.

```bash
python3 build_embeddings.py
```

**Dense (Qwen3-0.6B):**
- Mean-pool over last hidden states, then L2-normalise
- Saved to `embeddings_dense.npz` — keys: `ids` (str array), `vecs` (float32 matrix `[13644, 1024]`)

**Sparse (BGE-M3):**
- Lexical weights per token (replaces BM25)
- Saved to `embeddings_sparse.jsonl` — one line per listing: `{"id": "...", "weights": {"token": score, ...}}`

---

## File tree

```
/workshop/
├── README.md
├── listing_text.py              # text builder + column normaliser
├── build_text_mapping.py        # CSV → listing_texts.jsonl
├── check_mapping.py             # 20-row QA check
├── build_embeddings.py          # JSONL → dense .npz + sparse .jsonl
├── listing_texts.jsonl          # 13,644 text records
├── embeddings_dense.npz         # Qwen3 dense vectors
├── embeddings_sparse.jsonl      # BGE-M3 sparse weights
├── structured_data_withimages_updated.csv
├── structured_data_withoutimages-1776412361239.csv
└── models/
    ├── Qwen3-0.6B/
    └── bge-m3/
```
# datathon-robinreal-challenge
