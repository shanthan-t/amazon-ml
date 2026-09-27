# AUDIT — Final V6 Submission Implementation

**Audit date:** 2026-09-27T15:10:00+05:30
**Auditor:** Automated forensic audit

---

## 1. Repository Identity

| Property | Value |
|----------|-------|
| Remote URL | `https://github.com/shanthan-t/amazon-ml` |
| Source branch | `main` |
| Source Git SHA | `c79e66b1b0b71dcf3eba3c1035e2e7843f6b7d21` |
| Clean branch | `final-v6-submission` |
| Configuration ID | `v6-top25-xgb39-icu-global98-20260927-v1` |

---

## 2. Machine

| Property | Value |
|----------|-------|
| OS | Microsoft Windows 11 Pro Build 26200 |
| CPU | 13th Gen Intel Core i9-13900 |
| Physical cores | 24 |
| Logical processors | 32 |
| RAM | 127.7 GiB |
| Python version | 3.13.3 (Anaconda) |
| XGBoost version | 3.4.1 |
| NumPy version | 2.4.2 |
| RapidFuzz version | 3.14.3 |
| psutil version | 7.2.2 |
| SQLite version | 3.49.1 |
| ICU version | 77.1.0.0 |
| ICU implementation | Direct ICU C API via `ctypes` (no PyICU) |

---

## 3. Final Pipeline

| Component | Module |
|-----------|--------|
| Inference entry point | `windows_inference.runner` (`runner.py:main()`) |
| Worker entry point | `windows_inference.worker` (`worker.py:worker()`) |
| Retrieval | `windows_inference.retrieval` (`retrieval.py:Retriever`) |
| Target store | `windows_inference.target_store` (`target_store.py:MappedRetriever`) |
| Normalization | `windows_inference.normalization` (`normalization.py`) |
| Base features (26) | `windows_inference.base_features` (`base_features.py:compute_pair_features()`) |
| Extended features (39) | `windows_inference.features` (`features.py:FeatureEngine`) |
| Optimized features | `windows_inference.optimized_features` (`optimized_features.py:OptimizedFeatureEngine`) |
| ICU transliteration | `windows_inference.icu` (`icu.py:ICU`) |
| Decision policy | `windows_inference.policy` (`policy.py:choose()`) |
| Output writer | `windows_inference.policy` (`policy.py:OutputWriter`) |
| Merge | `windows_inference.merge` (`merge.py:merge()`) |
| Validation | `windows_inference.validate` (`validate.py:validate()`) |
| Configuration | `windows_inference.config` (`config.py`) |
| Frozen configuration | `config/frozen_configuration.json` |
| Checkpoint I/O | `windows_inference.safe_io` (`safe_io.py:atomic()`) |
| Setup verification | `windows_inference.setup_check` (`setup_check.py:main()`) |
| Parity verification | `windows_inference.verify_parity` (`verify_parity.py:verify()`) |
| Progress monitor | `windows_inference.progress` (`progress.py:show()`) |
| Benchmark | `windows_inference.benchmark` (`benchmark.py:benchmark()`) |

---

## 4. Retrieval

### Routes actually used

| Route | Key pattern | Posting cap |
|-------|-------------|-------------|
| E | `E\|{country}\|{name}` | 500 |
| P6 | `P6\|{country}\|{prefix6}` | 500 |
| P4 | `P4\|{country}\|{prefix4}` | 500 |
| ST | `ST\|{country}\|{tok1}_{tok2}` | 500 |
| W | `W\|{country}\|{word}` | 500 |
| AN | `AN\|{country}\|{number}` | 500 |

### Rare-address behavior

- Selects 2 lowest-frequency eligible address tokens (freq ≤ 5000).
- Filter: ≥2 shared nonnumeric address tokens AND (token Dice ≥ 0.55 OR address fuzz ratio ≥ 70).
- If eligible pool > 500: entire rare route discarded for that S1.
- Top-25 additions: ranked by `(conflict, -address_token_jaccard, -address_fuzz_ratio, -numeric_exact, -name_fuzz_ratio, idx)`.
- Baseline candidates subtracted before cap.

### Candidate deduplication

Candidates are stored as `set()` of integer target indexes. Duplicate indexes
are impossible by construction. The union of baseline and rare candidates is
sorted for deterministic processing order.

### Candidate ordering/determinism

`sorted(set(baseline) | rare)` — integer sorted order. Deterministic across
runs given the same input and indexes.

---

## 5. Features

**Exact feature count:** 39

**Feature names in exact model order:**

```
 0: name_exact                          (float32)
 1: name_token_jaccard                  (float32)
 2: name_fuzz_ratio                     (float32)
 3: name_prefix6                        (float32)
 4: name_prefix4                        (float32)
 5: name_token_sort_ratio               (float32)
 6: name_token_set_ratio                (float32)
 7: name_partial_ratio                  (float32)
 8: name_common_tokens                  (float32)
 9: name_common_token_ratio             (float32)
10: name_first_word_match               (float32)
11: address_exact                       (float32)
12: address_token_jaccard               (float32)
13: address_fuzz_ratio                  (float32)
14: address_token_sort_ratio            (float32)
15: numeric_jaccard                     (float32)
16: country_equal                       (float32)
17: name_length_ratio                   (float32)
18: name_length_gap                     (float32)
19: address_length_ratio                (float32)
20: address_length_gap                  (float32)
21: s1_address_missing                  (float32)
22: target_address_missing              (float32)
23: is_india                            (float32)
24: target_is_s2                        (float32)
25: name_is_short                       (float32)
26: numeric_set_exact                   (float32)
27: numeric_conflict                    (float32)
28: rare_address_overlap                (float32)
29: original_route_count                (float32)
30: route_agreement_count               (float32)
31: name_token_containment              (float32)
32: address_token_containment           (float32)
33: name_char_trigram_jaccard            (float32)
34: address_char_trigram_jaccard         (float32)
35: name_script_mismatch                (float32)
36: transliterated_name_fuzz            (float32)
37: transliterated_token_jaccard        (float32)
38: transliterated_char_trigram_jaccard  (float32)
```

**Data type:** All features are `numpy.float32`.

**Missing-value behavior:** Missing/empty name → empty string (features compute
as 0.0 or 1.0 depending on Jaccard convention). Missing address → empty string.
`s1_address_missing` and `target_address_missing` flags indicate this. No NaN
values are produced; the worker aborts on non-finite scores.

**Numeric-conflict feature/policy:** Feature index 27 (`numeric_conflict`).
When `numeric_conflict > 0`, `policy.py:choose()` applies threshold 0.99
instead of 0.98 (line 9: `np.where(features[:,27]>0, np.float32(.99), np.float32(.98))`).

---

## 6. Transliteration

| Property | Value |
|----------|-------|
| Implementation | `windows_inference/icu.py` |
| ICU binding | Direct C API via `ctypes` (not PyICU) |
| ICU version | 77.1.0.0 (asserted at runtime) |
| Transform | `Any-Latin; Latin-ASCII` |
| Post-processing | `normalize_business_name()` applied after transliteration |
| Windows DLLs | `icudt77.dll`, `icuuc77.dll`, `icuin77.dll` |
| DLL source | Official ICU4C 77.1 Win64-MSVC2022 release |
| Fallback | None. Missing DLLs cause a hard `FileNotFoundError`. |

---

## 7. Models

### fold0.ubj
| Property | Value |
|----------|-------|
| Filename | `models/B_cross_script/fold0.ubj` |
| Size | 2,125,693 bytes |
| SHA-256 | `7bb56d6fe0964b4c51aa1922d1ad4b8034586f2e41db5d876c85d80d625bc605` |
| Feature count | 39 (asserted at load: `m.num_features() != 39`) |

### fold1.ubj
| Property | Value |
|----------|-------|
| Filename | `models/B_cross_script/fold1.ubj` |
| Size | 2,130,731 bytes |
| SHA-256 | `19fea195aa2ccb3875d1e307f7cf1dd858162cc242e088759acee9f7df710ff3` |
| Feature count | 39 |

### fold2.ubj
| Property | Value |
|----------|-------|
| Filename | `models/B_cross_script/fold2.ubj` |
| Size | 2,126,376 bytes |
| SHA-256 | `9bc18f713e5f9faa5ce3f3a0aed4160e7d1d93f3cae79169324daed08331c0e6` |
| Feature count | 39 |

### fold3.ubj
| Property | Value |
|----------|-------|
| Filename | `models/B_cross_script/fold3.ubj` |
| Size | 2,146,913 bytes |
| SHA-256 | `8e21913e605c3d9af3eef493cce78ea4280d9756305fd427001897dad8ea6ef7` |
| Feature count | 39 |

All four model SHA-256 values were independently recomputed and verified
during this audit.

---

## 8. Ensemble

**Proof from code** (`worker.py`, lines 60–62):

```python
dm = xgb.DMatrix(X, feature_names=FEATURE_NAMES, nthread=threads)
per_model = [m.predict(dm) for m in models]
probs = np.mean(np.stack(per_model), axis=0, dtype=np.float32)
```

The four model probabilities are stacked into a `(4, N)` array and averaged
with `np.mean(..., axis=0, dtype=np.float32)`. This is a simple arithmetic
mean. No weighting, no learned combination.

---

## 9. Decision Policy

**Code path** (`policy.py`, lines 5–11):

```python
def choose(candidates, features, scores):
    if len(candidates) != len(set(candidates)):
        raise ValueError('Duplicate candidate index')
    if features.shape != (len(candidates), 39) or len(scores) != len(candidates):
        raise ValueError('Scoring dimensions differ')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model score')
    threshold = np.where(features[:, 27] > 0, np.float32(.99), np.float32(.98))
    accepted = [(float(s), int(t)) for t, s, th in zip(candidates, scores, threshold) if s >= th]
    return [t for s, t in sorted(accepted, reverse=True)[:11]]
```

| Parameter | Value | Source |
|-----------|-------|--------|
| Global threshold | 0.98 | `config.py:GLOBAL_THRESHOLD` + `policy.py` line 9 |
| Numeric-conflict threshold | 0.99 | `config.py:NUMERIC_CONFLICT_THRESHOLD` + `policy.py` line 9 |
| Maximum matches | 11 | `config.py:MAX_MATCHES` + `policy.py` line 11 (`:11]`) |
| Sort order | Descending score | `sorted(accepted, reverse=True)` |

---

## 10. Successful Production Run

| Property | Value |
|----------|-------|
| Total S1 entities | 1,732,544 |
| Worker count (final successful run) | 24 logical workers |
| Run directory | `runs\full_v6\resume_32` |
| Final output directory | `runs\full_v6\resume_32\final` |
| Merge worker count | 132 (accumulated across resume sessions) |
| Candidate pairs scored | 417,061,272 |
| Predicted matches | 5,324,446 |
| Configuration ID | `v6-top25-xgb39-icu-global98-20260927-v1` |
| Resume behavior | Completed shards verified by SHA-256 and skipped |
| Checkpoint behavior | JSON checkpoint per batch in each worker output directory |

---

## 11. matching_results.tsv

| Property | Value |
|----------|-------|
| Original absolute path | `C:\Users\Indu\Desktop\amazon ml\amazon-ml\runs\full_v6\resume_32\final\matching_results.tsv` |
| Preserved copy path | `C:\Users\Indu\Desktop\AMAZON_ML_FINAL_SUBMISSION\matching_results.tsv` |
| Rows | 1,732,544 |
| Size | 91,084,774 bytes |
| SHA-256 | `16e66a121c144a5ed83b6ae9fe2bbd87d237ab1290be13f038656e399e88b0c8` |
| Predicted matches | 5,324,446 |
| Validator result | PASS (all checks passed during merge) |

SHA-256 was independently recomputed for both the original and the preserved
copy during this audit. Both matched the expected value.

---

## 12. candidate_pairs.tsv

| Property | Value |
|----------|-------|
| Original absolute path | `C:\Users\Indu\Desktop\amazon ml\amazon-ml\runs\full_v6\resume_32\final\candidate_pairs.tsv` |
| Preserved copy path | `C:\Users\Indu\Desktop\AMAZON_ML_FINAL_SUBMISSION\candidate_pairs.tsv` |
| Entity rows | 1,732,544 |
| Candidate pairs | 417,061,272 |
| Size | 5,397,809,510 bytes |
| SHA-256 | `1ca8e3bec7bf109d74848953092f3811c549f3b0b2fe4c19ce2b560f20c60d2b` |
| Validator result | PASS |

SHA-256 was independently recomputed for both the original and the preserved
copy during this audit. Both matched the expected value.

---

## 13. Consistency Validation

The merge process (`merge.py`) calls `validate()` which performs streaming
validation that **every predicted match ID appears in the corresponding
scored candidate set** for each S1 entity.

The inference manifest records:
```json
"all_predictions_subset_of_scored_candidates": true
```

All 5,324,446 predicted matches were verified to be contained in the
417,061,272 scored candidate pairs.

---

## 14. Accuracy Evidence

### Hidden test set

**UNVERIFIED FROM THIS REPOSITORY.** The test ground truth is hidden and
controlled by the competition organizers. No hidden-test precision, recall,
or F0.5 scores can be claimed.

### Development / OOF

Each of the four fold models was trained on 3 of 4 entity-disjoint folds.
OOF evaluation metrics, if computed during training, are not present in
this Windows inference deployment.

### Sealed holdout

No sealed holdout labels are present in this repository.

### Historical experiments

This repository does not contain historical experiment logs or evaluation
results.

---

## 15. Competition Compliance

### Final execution path audit

| Check | Result |
|-------|--------|
| External data | **NOT USED.** No external business databases, geocoding results, or supplementary datasets are loaded. |
| Internet augmentation | **NOT USED.** No HTTP/API calls during inference. Setup downloads only the official ICU4C package. |
| Geocoding APIs | **NOT USED.** No geocoding service calls in any imported module. |
| Commercial entity-resolution APIs | **NOT USED.** No calls to Google, AWS, or other ER services. |
| Pre-trained language models | **NOT USED.** No transformers, BERT, or embedding models. |
| GPU usage | **NOT USED.** CPU-only XGBoost. No CUDA/GPU imports in the final execution path. |

The audit covers only the final V6 execution path (`windows_inference/`).
Old unused code in the `main` branch (DL/Bi-Encoder/FAISS) is not part of
this audit.

---

## 16. Reproducibility

### Exact commands to reproduce inference on Windows

```powershell
# 1. Clone the repository
git clone https://github.com/shanthan-t/amazon-ml.git
cd amazon-ml
git checkout final-v6-submission

# 2. Copy external artifacts (see COPY_TO_WINDOWS.txt)
# - artifacts/v2_test_index.sqlite3
# - artifacts/phase2_test_numeric_address.sqlite3
# - artifacts/test_address_index.sqlite3
# - artifacts/test_target_store/ (3 files)

# 3. Setup environment
.\scripts\setup_windows.ps1

# 4. Verify parity
.\.venv\Scripts\python.exe -m windows_inference.verify_parity

# 5. Run inference
.\scripts\run_windows.ps1 -Source1 <path\to\test_source1.tsv> -RunDir runs\full_v6 -Workers 24

# 6. Resume if interrupted
.\scripts\resume_windows.ps1 -Source1 <path\to\test_source1.tsv> -RunDir runs\full_v6 -Workers 24

# 7. Merge and validate
.\scripts\merge_windows.ps1 -Source1 <path\to\test_source1.tsv> -RunDir runs\full_v6
.\scripts\validate_windows.ps1 -RunDir runs\full_v6\final
```

---

## 17. Required External Runtime Artifacts

See `RUNTIME_ARTIFACTS.md` for the complete list with sizes and SHA-256 values.

Summary:
- 3 SQLite retrieval indexes (~7.5 GiB)
- 1 target store directory (3 files, ~1.0 GiB)
- 3 ICU4C 77.1 Windows DLLs (auto-downloaded by setup)
- 1 competition test_source1.tsv (~167 MiB, user-supplied)

---

## 18. Removed Legacy Material

The following items from the `main` branch history are NOT included in the
`final-v6-submission` branch because they were never part of the V6 pipeline:

> **Note:** The `amazon-ml` repository (`main` branch at SHA `c79e66b`) already
> contained ONLY V6 inference code. No DL/FAISS/V5 code was ever tracked in
> this repository. The old implementations exist in a separate repository
> (`amazon-ml-main`).

Files removed from untracked state during branch cleanup:
- `scripts/check_local_indexes.py` — ad-hoc local index verification utility
- `scripts/finish_local_inference.py` — local finish/merge orchestrator
- `scripts/rebuild_local_address_index.py` — address index rebuild utility
- `scripts/recover_remaining.py` — recovery script for interrupted runs
- `scripts/run_local_final.ps1` — local full-run wrapper
- `scripts/run_recovered_final.ps1` — recovery run wrapper

These were operational utilities used during the production run but are not
part of the reproducible inference pipeline.

---

## 19. Known Limitations

1. **Memory requirements:** Each worker loads read-only mmap connections to
   ~8.5 GiB of indexes. With 24+ workers, total resident memory is significant
   (OS memory-maps are shared for read-only files).

2. **Disk space:** The candidate_pairs.tsv output is ~5.0 GiB. Total working
   disk usage with shards and checkpoints exceeds 20 GiB.

3. **Single-machine only:** The pipeline does not support distributed
   execution across multiple machines.

4. **No incremental updates:** The pipeline processes the entire test set.
   Adding or modifying S1 entities requires a full rerun.

5. **ICU version pinning:** The ICU 77.1 version assertion means the pipeline
   will not run with a different ICU version without code modification.

6. **No accuracy guarantees:** Without access to hidden test labels, the
   actual F0.5 score is unknown.

---

## 20. Final Audit Conclusion

The `final-v6-submission` branch faithfully represents the implementation
that produced the validated outputs:

- **matching_results.tsv** (SHA-256: `16e66a1...`) — 1,732,544 rows, 5,324,446 matches
- **candidate_pairs.tsv** (SHA-256: `1ca8e3b...`) — 1,732,544 rows, 417,061,272 pairs

The branch contains exactly the code, models, configuration, scripts, and
documentation required to understand and reproduce the V6 entity resolution
pipeline. No DL/FAISS/training/experiment code is present. No external data,
APIs, or internet resources are used during inference. All four model
checksums and both output file checksums have been independently verified.

The preserved copies at `C:\Users\Indu\Desktop\AMAZON_ML_FINAL_SUBMISSION\`
have identical SHA-256 values to the originals.
