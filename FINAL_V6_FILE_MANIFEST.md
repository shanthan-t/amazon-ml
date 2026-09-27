# FINAL V6 FILE MANIFEST

**Generated:** 2026-09-27T16:11+05:30
**Branch:** `final-v6-submission`
**Commit:** `14affdeba08cf85e1bfe7c0fef5243c3727deb8e`
**Total tracked files:** 46

---

## REQUIRED_RUNTIME — Files executed during inference

| File | Why required |
|------|-------------|
| `windows_inference/__init__.py` | Package marker; required for `python -m windows_inference.*` imports |
| `windows_inference/config.py` | Frozen paths, model SHA256s, thresholds (0.98/0.99), max_matches=11, TOP_RARE_ADDRESS=25 |
| `windows_inference/normalization.py` | Unicode NFKC/casefold/punctuation normalization shared by all feature computation |
| `windows_inference/base_features.py` | 26 V2 base features, blocking key generation, numeric address key generation |
| `windows_inference/features.py` | 39-feature FeatureEngine, FEATURE_NAMES tuple, `normalize()` entry, ICU feature integration |
| `windows_inference/optimized_features.py` | LRU-cached OptimizedFeatureEngine subclass used by the worker for performance |
| `windows_inference/icu.py` | Pinned ICU 77.1 C API binding for Any-Latin; Latin-ASCII transliteration via ctypes |
| `windows_inference/retrieval.py` | SQLite-based candidate retrieval: 6 blocking routes + rare-address overlap + top25 selection |
| `windows_inference/target_store.py` | Memory-mapped target record lookup (TargetStore) + MappedRetriever combining retrieval + store |
| `windows_inference/policy.py` | `choose()` threshold policy (0.98/0.99/max11) + `OutputWriter` TSV contract |
| `windows_inference/safe_io.py` | Atomic JSON publication with Windows antivirus lock retry |
| `windows_inference/worker.py` | Single-shard worker: loads models, retrieves, computes features, scores, writes outputs |
| `windows_inference/runner.py` | Multi-worker orchestrator: partitions S1, launches subprocess workers, monitors progress |
| `windows_inference/merge.py` | Shard merge: validates contiguity/manifests/checksums, concatenates, calls validate() |
| `windows_inference/validate.py` | Streaming output-contract validator (S1 order, ID format, match⊆candidates, cap≤11) |
| `windows_inference/progress.py` | Live aggregate progress monitor reading shard checkpoints |

## REQUIRED_MODEL — Trained model files loaded at inference

| File | Size | SHA-256 | Why required |
|------|------|---------|-------------|
| `models/B_cross_script/fold0.ubj` | 2,125,693 B | `7bb56d6fe0964b4c51aa1922d1ad4b8034586f2e41db5d876c85d80d625bc605` | XGBoost fold-0 model (39 features) |
| `models/B_cross_script/fold1.ubj` | 2,130,731 B | `19fea195aa2ccb3875d1e307f7cf1dd858162cc242e088759acee9f7df710ff3` | XGBoost fold-1 model (39 features) |
| `models/B_cross_script/fold2.ubj` | 2,126,376 B | `9bc18f713e5f9faa5ce3f3a0aed4160e7d1d93f3cae79169324daed08331c0e6` | XGBoost fold-2 model (39 features) |
| `models/B_cross_script/fold3.ubj` | 2,146,913 B | `8e21913e605c3d9af3eef493cce78ea4280d9756305fd427001897dad8ea6ef7` | XGBoost fold-3 model (39 features) |

## REQUIRED_SETUP — Setup and environment verification

| File | Why required |
|------|-------------|
| `requirements-windows.txt` | Pinned pip dependencies: numpy==2.4.2, xgboost==3.4.1, rapidfuzz==3.14.3, psutil==7.2.2 |
| `scripts/setup_windows.ps1` | Environment setup: venv, pip install, ICU download, model/artifact verification |
| `windows_inference/setup_check.py` | Fail-loud integrity check: packages, ICU version, model SHA256s, copied artifact sizes |

## REQUIRED_VALIDATION — Parity and benchmark verification

| File | Why required |
|------|-------------|
| `windows_inference/verify_parity.py` | Deterministic parity test: normalization, candidates, 39 features, ICU, 4 model scores, policy |
| `parity_fixture/expected_results.json` | Frozen Linux-generated fixture: 1 S1 row, 32 candidates, all 39 features, all 4 model scores |
| `parity_fixture/README.md` | Documents the parity fixture provenance and usage |
| `windows_inference/benchmark.py` | Bounded scaling benchmark using exact inference worker with time-budget control |

## REQUIRED_SCRIPTS — Operational PowerShell wrappers

| File | Why required |
|------|-------------|
| `scripts/run_windows.ps1` | Full inference launcher wrapper |
| `scripts/resume_windows.ps1` | Resume wrapper (calls run_windows.ps1 with -Resume) |
| `scripts/merge_windows.ps1` | Merge and validate wrapper |
| `scripts/validate_windows.ps1` | Standalone validation wrapper |
| `scripts/progress_windows.ps1` | Progress monitor wrapper |
| `scripts/benchmark_windows.ps1` | Benchmark launcher wrapper |

## REQUIRED_DOCUMENTATION

| File | Why required |
|------|-------------|
| `README.md` | Primary repository documentation: architecture, setup, commands, reproducibility |
| `README_WINDOWS.md` | Detailed Windows deployment guide with exact commands |
| `METHODOLOGY.md` | Technical methodology: candidate generation, features, ensemble, decision policy |
| `AUDIT_FINAL_V6.md` | Complete forensic audit of the V6 implementation with verified hashes |
| `RUNTIME_ARTIFACTS.md` | External artifacts not in Git: indexes, target store, ICU DLLs, sizes, SHA256s |
| `COPY_TO_WINDOWS.txt` | Step-by-step artifact transfer instructions from Linux to Windows |
| `WINDOWS_ARTIFACT_MANIFEST.json` | Machine-readable manifest of all external artifacts with checksums |
| `config/frozen_configuration.json` | Human-readable frozen configuration document |
| `native/icu/README.md` | Documents the ICU DLL deployment and setup_windows.ps1 auto-download |
| `native/icu/LICENSE` | ICU 77.1 license (legally required when distributing ICU binaries) |
| `.gitignore` | Git ignore rules for venv, competition data, artifacts, outputs, caches |

## REQUIRED_TRANSFER

| File | Why required |
|------|-------------|
| `windows_transfer/TRANSFER_SHA256.txt` | SHA256 checksums for verifying artifact transfers from Linux |

## OPTIONAL_DEVELOPMENT — Useful but not required for inference

| File | Why |
|------|-----|
| `PC_CLEANUP_PLAN.md` | Operational note about cleaning borrowed PC; not part of inference pipeline |

---

## OBSOLETE FILES ON THIS BRANCH

**None.** The `final-v6-submission` branch was created clean. No DL/FAISS/V5/SageMaker/training files exist on this branch.

The following obsolete material exists only in Git history (earlier commits) and on other branches:
- `src/` directory (old V2/V5 blocking, training, prediction, GPU/DL code)
- `benchmarks/` directory (old benchmark/tuning experiments)
- `sagemaker/` directory (SageMaker deployment)
- `models/xgb_v2.xgb`, `models/xgb_v3.xgb`, `models/xgb_v4.xgb` (old models)
- `models/logistic_*.json` (old logistic models)
- `requirements.txt`, `requirements_gpu.txt` (old dependencies including torch/transformers/faiss)
- `run_dl_pipeline.sh`, `run_predict_parallel.sh`, `run_train_parallel.sh`, `build_indices.sh`

These are NOT tracked on `final-v6-submission` and require no deletion.

---

## LARGE FILE AUDIT

All tracked files are under 25 MB. The four model files (~2.1 MB each) are the largest tracked files.

Large runtime artifacts (NOT tracked in Git, correctly excluded by `.gitignore`):
- `artifacts/v2_test_index.sqlite3` — 4.51 GiB
- `artifacts/test_address_index.sqlite3` — 2.76 GiB
- `artifacts/test_target_store/records.tsv` — 952 MiB
- `artifacts/phase2_test_numeric_address.sqlite3` — 303 MiB
- `artifacts/test_target_store/offsets.npy` — 76 MiB

These are documented in `RUNTIME_ARTIFACTS.md` with sizes and SHA256 values.

---

## IMPORT DEPENDENCY GRAPH

```
runner.py → config (CONFIG_ID, MODEL_PATHS, MODEL_SHA256, INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE)
worker.py → features (FEATURE_NAMES, normalize)
          → optimized_features (OptimizedFeatureEngine)
          → policy (OutputWriter, choose)
          → target_store (MappedRetriever)
          → config (INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE, MODEL_PATHS, CONFIG_ID)
          → safe_io (atomic)
features.py → base_features (V2_FEATURE_NAMES, compute_pair_features, generate_blocking_keys, generate_numeric_address_keys)
            → normalization (normalize_business_name, normalize_business_address, normalize_country)
            → icu (ICU)
optimized_features.py → features (FeatureEngine, DIGITS, keys, grams, scripts, zero_jac)
target_store.py → retrieval (Retriever)
retrieval.py → features (keys, numeric)
             → base_features (compute_pair_features)
merge.py → config (CONFIG_ID, MODEL_SHA256)
         → validate (validate)
validate.py → config (MAX_MATCHES)
setup_check.py → config (MODEL_PATHS, MODEL_SHA256)
               → icu (ICU)
verify_parity.py → config (ADDRESS_INDEX, CONFIG_ID, INDEX, MODEL_PATHS, MODEL_SHA256, NUMERIC_INDEX, TARGET_STORE)
                 → features (FEATURE_NAMES, normalize)
                 → optimized_features (OptimizedFeatureEngine)
                 → policy (choose)
                 → target_store (MappedRetriever)
benchmark.py → runner (atomic, partition, sha)
             → config (ROOT)
```

**No DL/FAISS/torch/transformers imports exist anywhere in the tracked codebase.**

## UNKNOWN_REQUIRES_REVIEW

**None.** Every tracked file has been inspected and classified.
