# Amazon ML Challenge — Final V6 Entity Resolution Submission

Deterministic, precision-oriented entity resolution pipeline for matching
Source 1 business entities against Source 2/Source 3 noisy records.

## Problem Overview

Given 1,732,544 Source 1 entities (business name, address, country), identify
matching records from Source 2 and Source 3 target databases containing
~10 million records. Evaluation uses **F0.5**, which weights precision twice
as heavily as recall.

## Final Architecture

```
S1 Entity → Normalize → Blocking Key Generation → Candidate Retrieval
         → Rare-Address Overlap Retrieval → Top-25 Bounded Selection
         → 39-Feature Extraction (incl. ICU transliteration)
         → 4-Fold XGBoost Ensemble Scoring → Threshold Policy → Output
```

### Candidate Generation

Six blocking routes: `E` (exact name), `P6` (6-char prefix), `P4` (4-char prefix),
`ST` (sorted significant-token pairs), `W` (individual words ≥5 chars),
`AN` (numeric address tokens ≥3 digits). Base posting cap: 500.

**Rare-address overlap route:** Selects the two lowest-frequency eligible address
tokens (freq ≤ 5000). Candidates must share ≥2 nonnumeric address tokens AND
(token Dice ≥ 0.55 OR address fuzz ratio ≥ 70). If the rare pool exceeds 500, it
is discarded. Up to 25 qualifying rare-address additions are ranked by
address-level features and added to the candidate set. Baseline candidates are
subtracted before the top-25 cap.

### Normalization

Unicode NFKC → casefold → `&` → `and` → ASCII punctuation to space → collapse
whitespace. Non-ASCII characters retain letters/marks/numbers; everything else
becomes a space boundary.

### 39 Features

| #   | Feature                            | Type    |
|-----|------------------------------------|---------|
| 1   | name_exact                         | float32 |
| 2   | name_token_jaccard                 | float32 |
| 3   | name_fuzz_ratio                    | float32 |
| 4   | name_prefix6                       | float32 |
| 5   | name_prefix4                       | float32 |
| 6   | name_token_sort_ratio              | float32 |
| 7   | name_token_set_ratio               | float32 |
| 8   | name_partial_ratio                 | float32 |
| 9   | name_common_tokens                 | float32 |
| 10  | name_common_token_ratio            | float32 |
| 11  | name_first_word_match              | float32 |
| 12  | address_exact                      | float32 |
| 13  | address_token_jaccard              | float32 |
| 14  | address_fuzz_ratio                 | float32 |
| 15  | address_token_sort_ratio           | float32 |
| 16  | numeric_jaccard                    | float32 |
| 17  | country_equal                      | float32 |
| 18  | name_length_ratio                  | float32 |
| 19  | name_length_gap                    | float32 |
| 20  | address_length_ratio               | float32 |
| 21  | address_length_gap                 | float32 |
| 22  | s1_address_missing                 | float32 |
| 23  | target_address_missing             | float32 |
| 24  | is_india                           | float32 |
| 25  | target_is_s2                       | float32 |
| 26  | name_is_short                      | float32 |
| 27  | numeric_set_exact                  | float32 |
| 28  | numeric_conflict                   | float32 |
| 29  | rare_address_overlap               | float32 |
| 30  | original_route_count               | float32 |
| 31  | route_agreement_count              | float32 |
| 32  | name_token_containment             | float32 |
| 33  | address_token_containment          | float32 |
| 34  | name_char_trigram_jaccard           | float32 |
| 35  | address_char_trigram_jaccard        | float32 |
| 36  | name_script_mismatch               | float32 |
| 37  | transliterated_name_fuzz           | float32 |
| 38  | transliterated_token_jaccard       | float32 |
| 39  | transliterated_char_trigram_jaccard | float32 |

### ICU Transliteration

ICU 77.1 `Any-Latin; Latin-ASCII` via direct C API (`ctypes`). Converts
non-Latin scripts (Devanagari, CJK, Arabic, Cyrillic, etc.) to ASCII before
computing transliteration features. No fallback is allowed.

### Four-Model Ensemble

Four XGBoost `B_cross_script` fold models, each trained on 3 of 4 entity-disjoint
development folds. Each scores every candidate independently. The final score is
the **arithmetic mean** of the four probabilities (float32).

### Thresholds

- **Global threshold:** 0.98 (candidates with average probability ≥ 0.98)
- **Numeric-conflict threshold:** 0.99 (when `numeric_conflict > 0`)
- **Maximum matches per S1:** 11

### Windows Multiprocessing

Workers are independent subprocesses. Each opens its own read-only SQLite
connections. Disjoint S1 ranges with deterministic row assignment. Workers
produce separate candidate/match TSVs and checkpoint files. Completed shards
are SHA-256 verified and skipped on resume.

## Required Runtime Artifacts

See `RUNTIME_ARTIFACTS.md` for the full list of external artifacts not stored
in Git (retrieval indexes, target store, ICU DLLs, competition data).

## Environment Setup

```powershell
.\scripts\setup_windows.ps1
```

Requires 64-bit CPython 3.12+, creates `.venv`, installs pinned CPU
dependencies, downloads ICU4C 77.1, and verifies all models and artifacts.

## Inference Commands

```powershell
# Parity verification
.\.venv\Scripts\python.exe -m src.verify_parity

# Benchmark (optional)
.\scripts\benchmark_windows.ps1 -Source1 <path\to\test_source1.tsv>

# Full inference
.\scripts\run_windows.ps1 -Source1 <path\to\test_source1.tsv> -RunDir runs\full_v6 -Workers 24

# Resume after interruption
.\scripts\resume_windows.ps1 -Source1 <path\to\test_source1.tsv> -RunDir runs\full_v6 -Workers 24

# Monitor progress
.\scripts\progress_windows.ps1 -RunDir runs\full_v6

# Merge and validate
.\scripts\merge_windows.ps1 -Source1 <path\to\test_source1.tsv> -RunDir runs\full_v6
.\scripts\validate_windows.ps1 -RunDir runs\full_v6\final
```

## Output Files

| File                    | Description                                          |
|-------------------------|------------------------------------------------------|
| `matching_results.tsv`  | S1 entity → matched S2/S3 entity IDs                |
| `candidate_pairs.tsv`   | S1 entity → all scored candidate entity IDs          |

## Reproducibility Notes

- All feature computation uses `float32` (matching training).
- Candidate ordering is deterministic (sorted integer indexes).
- Worker S1 ranges are deterministic (sequential partition).
- Model scoring uses `nthread=1` for determinism.
- ICU version is pinned at 77.1 with version assertion.
- No randomness, no sampling, no stochastic components at inference.

## Repository layout (canonical)

- `src/` — Final V6 inference runtime (retrieval, 39 features, ICU, ensemble, policy, merge, validate)
- `training/` — Essential training, freeze, and evaluation scripts (do not retrain unless explicitly requested)
- `indexing/` — Target / numeric-address / rare-address index builders
- `models/B_cross_script/` — Four frozen 39-feature XGBoost models
- `native/icu/` — Pinned ICU 77.1 (Windows DLLs downloaded by setup; Linux `.so` vendored)
- `scripts/` — PowerShell wrappers

## Training and index build (optional reproduction)

Large indexes and training shards are local-only. See `TRAINING_ARTIFACTS.md`.

```bash
python -m indexing.build_target_index --s2 SOURCE2.tsv --s3 SOURCE3.tsv --index local_artifacts/v2_train_index.sqlite3 --numeric-index local_artifacts/phase2_train_numeric_address.sqlite3
python -m indexing.build_address_index --source local_artifacts/v2_train_index.sqlite3 --output local_artifacts/train_address_index.sqlite3
python -m training.train
```

Do not retrain to produce a new submission. The checked-in fold models are the submission models.

Obsolete DL / FAISS / SageMaker / V5 experiment trees are not part of this branch.

