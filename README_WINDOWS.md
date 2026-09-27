# Amazon ML Business Entity Resolution — Windows V6 inference

This is the minimal Windows deployment for the exact V6 test-inference pipeline running on Linux. It uses deterministic V6 retrieval with bounded `rare_address_overlap` and `top_25`, the canonical 39 float32 features, ICU 77.1 `Any-Latin; Latin-ASCII`, four existing B-cross-script fold models averaged at inference, and the frozen 0.98 / 0.99 policy with an 11-match cap. It does not train a model or access sealed holdout labels.

Each fold model was trained on three of four entity-disjoint development folds. The Windows runner averages the four fold probabilities and retains the frozen OOF threshold without search or tuning.

## Setup

Clone `https://github.com/shanthan-t/amazon-ml.git`, copy the files in [`COPY_TO_WINDOWS.txt`](COPY_TO_WINDOWS.txt) to the exact relative destinations, and keep competition TSVs outside Git. In PowerShell, run:

```powershell
.\scripts\setup_windows.ps1
```

This requires 64-bit CPython 3.12, creates `.venv`, installs pinned CPU dependencies, downloads the official ICU4C 77.1 Windows x64 archive if needed, verifies its SHA-256, extracts only the required DLLs, then checks the ICU runtime, four model hashes, and copied index/store sizes and checksums. Missing external artifacts are reported explicitly.

Run parity before inference:

```powershell
.\.venv\Scripts\python.exe -m windows_inference.verify_parity
```

Parity fails on any difference in normalization, candidate IDs/order, base or rare route membership, 39 float32 features, ICU strings, four fold scores, averaged score, or final policy result.

## Benchmark and full inference

The benchmark uses a fixed, evenly spaced test subset and the exact inference worker. Its default worker sweep is capped at 12 minutes:

```powershell
.\.venv\Scripts\python.exe -m windows_inference.benchmark --source1 C:\data\test_source1.tsv --output runs\benchmark_v1 --workers 2,4,8,12,16 --budget-minutes 12
```

Run full inference with deterministic disjoint S1 ranges; choose workers based on benchmark throughput and RAM:

```powershell
.\scripts\run_windows.ps1 -Source1 C:\data\test_source1.tsv -RunDir runs\full_v6 -Workers 12 -ThreadsPerWorker 1
```

The runner prints aggregate progress, per-worker counts, S1/s, and ETA. Workers each open their own read-only SQLite connections and write separate candidate/match TSVs and checkpoints. Completed valid shards are checksum-verified and skipped on resume:

```powershell
.\scripts\resume_windows.ps1 -Source1 C:\data\test_source1.tsv -RunDir runs\full_v6 -Workers 12 -ThreadsPerWorker 1
```

Monitor from another PowerShell window:

```powershell
.\scripts\progress_windows.ps1 -RunDir runs\full_v6
```

## Merge and validate

After all shards complete, merge by canonical S1 range. The merger checks range contiguity, shard manifests and hashes, streams both TSVs, validates the temporary outputs, and publishes them under `runs\full_v6\final`:

```powershell
.\scripts\merge_windows.ps1 -Source1 C:\data\test_source1.tsv -RunDir runs\full_v6
.\scripts\validate_windows.ps1 -RunDir runs\full_v6\final
```

Validation checks exact S1 coverage/order, TSV schema, S2/S3 ID format, candidate/match uniqueness per S1, the 11-match cap, and that each predicted match is included in the exact scored candidates. Empty match lists use an empty second field. Non-finite model scores stop the worker.

## External artifacts and method

`WINDOWS_ARTIFACT_MANIFEST.json` records source Linux paths, file sizes and SHA-256 values. The multi-gigabyte retrieval indexes and target store should be copied PC-to-PC; rebuilding is slower. They are not placed in Git or Git LFS. Four fold models (~8.5 MB total) are safe in ordinary Git. Competition data, predictions, candidates, checkpoints, caches and training shards are excluded.

The runtime modules are in `windows_inference/`; frozen settings are in `config/frozen_configuration.json`. ICU uses direct `ctypes` calls into ICU 77.1, with no transliteration fallback. The package uses CPU XGBoost and does not include DL/FAISS, geocoding, or external business data.
