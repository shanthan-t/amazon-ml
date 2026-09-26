# SageMaker Processing Job Manifest — P6+Exact Test Inference
# ==========================================================
#
# This manifest documents every artifact required to run the production
# P6+exact test inference on SageMaker Processing.
#
# Production command being reproduced:
#   python3 -m src.predict_test \
#     --index phase2_test_exact_index.sqlite3 \
#     --model models/logistic_p6_exact_candidate.json \
#     --matching output_v2/matching_results.tsv \
#     --candidate output_v2/candidate_pairs.tsv \
#     --report reports/test_prediction_v2.json \
#     --p6-index phase2_test_p6_routes.sqlite3 \
#     --name-index phase2_test_names.sqlite3

## Required Source Files (Python)

| File | Role | Size |
|------|------|------|
| `src/__init__.py` | Package marker | 54 B |
| `src/predict_test.py` | **Main entry point** — streaming scorer with P6+exact SQL | 14,290 B |
| `src/features.py` | 15-feature vector (FEATURE_NAMES, pair_features_normalized, pair_features_exact_name) | 3,905 B |
| `src/normalization.py` | Unicode NFKC casefold + punctuation normalization | 1,767 B |
| `src/data_loader.py` | Chunked TSV reader (iter_source_chunks) | 2,398 B |
| `src/blocking.py` | Only used for constants: `DEFAULT_MAX_POSTINGS_PER_KEY=500`, `SOURCE1_BATCH_SIZE=1000` | 43,988 B |

### Import chain
```
predict_test.py
├── src.blocking        (constants only: DEFAULT_MAX_POSTINGS_PER_KEY, SOURCE1_BATCH_SIZE)
├── src.data_loader     (iter_source_chunks)
├── src.features        (FEATURE_NAMES, pair_features_exact_name, pair_features_normalized)
│   └── src.normalization
└── src.normalization   (normalize_business_name, normalize_business_address, normalize_country)
```

### Standard library imports (no install needed)
argparse, csv, json, re, resource, sqlite3, time, unicodedata, pathlib

### Third-party imports
numpy, scipy.special.expit, rapidfuzz.fuzz, pandas

## Required SQLite Indexes

| File | Purpose | Size |
|------|---------|------|
| `phase2_test_exact_index.sqlite3` | Main blocking index: exact normalized name route postings, block_counts, targets (with normalized_address, normalized_country) | **2.0 GB** |
| `phase2_test_p6_routes.sqlite3` | P6 prefix-6 route postings + block_counts (ATTACHed as `p6`) | **569 MB** |
| `phase2_test_names.sqlite3` | Target normalized names (ATTACHed as `names`, table `target_names`) | **385 MB** |

> These are pre-built read-only indexes. Do NOT rebuild them.

## Required Model

| File | Format | Threshold | Features | Size |
|------|--------|-----------|----------|------|
| `models/logistic_p6_exact_candidate.json` | JSON (coefficients, means, scales, intercept, threshold) | 0.0252 | 15 (FEATURE_NAMES) | 2,005 B |

## Required Test Data

| File | Purpose | Size | Rows |
|------|---------|------|------|
| `dataset/test/test_source1.tsv` | Source-1 entities to score | **167 MB** | 1,732,544 (excl. header) |

> `test_source2.tsv` and `test_source3.tsv` are NOT needed — targets are
> already indexed in the SQLite files.

## Required Python Dependencies

| Package | Version (pinned) | Used by |
|---------|-----------------|---------|
| `pandas` | 2.3.3 | data_loader (chunked TSV reading) |
| `rapidfuzz` | 3.14.3 | features.py, predict_test.py (fuzz.ratio) |
| `scipy` | 1.17.1 | predict_test.py (scipy.special.expit) |
| `numpy` | 2.4.2 | predict_test.py (array ops, probability scoring) |

See: `sagemaker/requirements_inference.txt`

## SageMaker Channel Mapping

| Channel | S3 Path | Container Path |
|---------|---------|----------------|
| `code` | `s3://BUCKET/amazon-ml/runs/p6-test-v2/input/code/` | `/opt/ml/processing/input/code/` |
| `data` | `s3://BUCKET/amazon-ml/runs/p6-test-v2/input/data/` | `/opt/ml/processing/input/data/` |
| `indexes` | `s3://BUCKET/amazon-ml/runs/p6-test-v2/input/indexes/` | `/opt/ml/processing/input/indexes/` |
| `model` | `s3://BUCKET/amazon-ml/runs/p6-test-v2/input/model/` | `/opt/ml/processing/input/model/` |
| `results` (output) | `s3://BUCKET/amazon-ml/runs/p6-test-v2/output/results/` | `/opt/ml/processing/output/results/` |

## Expected Disk Sizes

### Input disk
| Component | Size |
|-----------|------|
| SQLite indexes (3 files) | **2.93 GB** |
| test_source1.tsv | **167 MB** |
| Model JSON | **2 KB** |
| Source code | **~66 KB** |
| **Total input** | **~3.1 GB** |

### Output disk
| Component | Estimated Size |
|-----------|---------------|
| `matching_results.tsv` | **35–50 MB** (1,732,544 rows) |
| `candidate_pairs.tsv` | **1.0–1.5 GB** (all candidate pairs) |
| `test_prediction_v2.json` | **~2 KB** |
| **Total output** | **~1.0–1.5 GB** |

### Required processing volume
| Purpose | Size |
|---------|------|
| Input staging | 3.1 GB |
| Output files | 1.5 GB |
| SQLite temp/page cache | 2–4 GB |
| OS overhead | 2 GB |
| Safety margin | 5 GB |
| **Recommended volume** | **15–20 GB** |
| **Configured default** | **100 GB** (generous margin) |

## Recommended Instance Type

| Setting | Value | Rationale |
|---------|-------|-----------|
| Instance type | `ml.m5.4xlarge` | 16 vCPU, 64 GB RAM — CPU-bound workload; SQLite I/O benefits from memory for page cache |
| Instance count | **1** | Single-node; the scorer runs single-process with streamed batches of 1,000 |
| Workers | **1** (implicit) | `predict_test.py` uses a single-process streaming approach with batched SQL queries |
| GPU | **None** | Pure CPU workload: SQLite lookups, Unicode normalization, RapidFuzz character similarity, small logistic model |

### Alternative instances
| Instance | vCPU | RAM | Use case |
|----------|------|-----|----------|
| `ml.m5.2xlarge` | 8 | 32 GB | Budget option; sufficient RAM for page cache |
| `ml.m5.4xlarge` | 16 | 64 GB | **Recommended**; comfortable overhead |
| `ml.m7i.4xlarge` | 16 | 64 GB | Newer Intel; slightly better per-core perf |

## Expected Runtime

| Metric | Value | Basis |
|--------|-------|-------|
| Local baseline (exact-only, all 1.73M entities) | ~539s (~9 min) | `reports/test_prediction.json` |
| P6+exact route (additional SQL + name similarity) | ~15–40 min estimated | More candidates per S1 entity vs exact-only |
| SageMaker overhead (data staging) | ~5–10 min | S3 download of ~3.1 GB input |
| **Estimated total wall time** | **20–50 min** | Conservative; actual depends on instance I/O |
| Max runtime configured | **4 hours** | Safety cutoff |

## Expected Outputs

The job MUST produce all three files:

1. **`matching_results.tsv`** — Tab-separated, header `source1_entity_id\tmatched_entity_ids`.
   Must contain exactly **1,732,544** data rows (one per Source-1 entity).
   Entity IDs with no matches have an empty second column.

2. **`candidate_pairs.tsv`** — Tab-separated, header `source1_entity_id\tcandidate_entity_ids`.
   Same row count as matching_results.tsv.

3. **`test_prediction_v2.json`** — JSON report with configuration, row counts,
   timing, and peak RSS.

## Verification

The runner script (`sagemaker/run_test_inference.py`) performs automatic
post-inference verification:
- Counts rows in `matching_results.tsv`
- Compares against `test_source1.tsv` entity count
- **Exits non-zero** (fails the job) if counts don't match
- This prevents silent partial submissions

## Files NOT Required (Excluded from Upload)

| Path | Reason |
|------|--------|
| `output/` | Existing known-good baseline submission — DO NOT touch |
| `output_v2/` | Existing partial P6+exact results — DO NOT touch |
| `reports/` | Local reports only |
| `phase2_index_checkpoint.sqlite3` (4.0 GB) | Training index — not needed |
| `phase2_index_resume.sqlite3` (4.2 GB) | Training index — not needed |
| `phase2_name_routes_train.sqlite3` (1.6 GB) | Training name routes — not needed |
| `dataset/test/test_source2.tsv` (486 MB) | Already indexed in SQLite |
| `dataset/test/test_source3.tsv` (483 MB) | Already indexed in SQLite |
| `dataset/train/` | Training data — not needed |
| `experiments/` | Experiment scripts |
| `benchmarks/` | Benchmark scripts |
| `tests/` | Unit tests |
| `__pycache__/` | Bytecode cache |
| `.pytest_cache/` | Test cache |
