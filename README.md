# Amazon ML Challenge — Final V6 Entity Resolution Pipeline

Canonical, minimal implementation of the final V6 entity resolution inference pipeline that produced the validated competition submission.

The pipeline performs deterministic multi-route blocking with bounded rare-address overlap retrieval, extracts 39 float32 features (including ICU transliteration across scripts), scores candidate pairs with a 4-fold XGBoost ensemble, and applies a precision-first decision policy.

---

## 1. Install Dependencies

Requires Python 3.10+ (64-bit).

```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy==2.4.2`
- `xgboost==3.4.1`
- `rapidfuzz==3.14.3`
- `psutil==7.2.2`

---

## 2. Place Dataset

Place competition test files in `dataset/` (git-ignored):

```
dataset/
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv
```

Required runtime input is `dataset/test_source1.tsv` (1,732,544 rows).

---

## 3. Place Artifacts

Place pre-built retrieval indexes and target store in `artifacts/` (git-ignored):

```
artifacts/
├── v2_test_index.sqlite3                (4,843,343,872 bytes)
├── phase2_test_numeric_address.sqlite3  (317,624,320 bytes)
├── test_address_index.sqlite3           (2,967,990,272 bytes)
└── test_target_store/
    ├── offsets.npy                      (79,756,848 bytes)
    ├── records.tsv                      (998,458,970 bytes)
    └── manifest.json                    (267 bytes)
```

The four frozen XGBoost fold models (`models/B_cross_script/fold[0-3].ubj`) and ICU native libraries (`native/icu/`) are tracked directly in the repository.

---

## 4. Run Setup Check

Verify system integrity, ICU transliteration, model SHA-256 checksums, and artifact files:

### Linux / macOS
```bash
./scripts/setup.sh
# or
python3 -m src.setup_check
```

### Windows
```powershell
.\scripts\setup_windows.ps1
```

### Parity / Smoke Test
Run the deterministic end-to-end parity test (tests candidate retrieval, 39 features, ICU, 4-fold scoring, policy, and thresholds):
```bash
python3 -m src.verify_parity
```

---

## 5. Run Full Inference

Runs parallel sharded inference across CPU cores with atomic checkpointing:

### Linux / macOS
```bash
./scripts/run.sh dataset/test_source1.tsv runs/full_v6 8 1
# or
python3 -m src.runner --source1 dataset/test_source1.tsv --run-dir runs/full_v6 --workers 8 --threads-per-worker 1
```

### Windows
```powershell
.\scripts\run_windows.ps1 -Source1 dataset\test_source1.tsv -RunDir runs\full_v6 -Workers 8
```

---

## 6. Resume Inference

Safely resumes an interrupted run. Completed shards are verified via SHA-256 and skipped; incomplete shards restart cleanly:

### Linux / macOS
```bash
./scripts/resume.sh dataset/test_source1.tsv runs/full_v6 8 1
# or
python3 -m src.runner --source1 dataset/test_source1.tsv --run-dir runs/full_v6 --workers 8 --threads-per-worker 1 --resume
```

### Windows
```powershell
.\scripts\resume_windows.ps1 -Source1 dataset\test_source1.tsv -RunDir runs\full_v6 -Workers 8
```

To monitor progress:
```bash
./scripts/progress.sh runs/full_v6
# Windows: .\scripts\progress_windows.ps1 -RunDir runs\full_v6
```

---

## 7. Merge Shard Outputs

Merges shard predictions into final submission files in canonical S1 order:

### Linux / macOS
```bash
./scripts/merge.sh dataset/test_source1.tsv runs/full_v6
# or
python3 -m src.merge --source1 dataset/test_source1.tsv --run-dir runs/full_v6
```

### Windows
```powershell
.\scripts\merge_windows.ps1 -Source1 dataset\test_source1.tsv -RunDir runs\full_v6
```

Outputs written to `runs/full_v6/final/`:
- `matching_results.tsv` — Canonical competition submission
- `candidate_pairs.tsv` — All scored candidate pairs

---

## 8. Validate Submission

Validates format, schema, entity coverage, numeric thresholds, and contract integrity:

### Linux / macOS
```bash
./scripts/validate.sh runs/full_v6/final
# or
python3 -m src.validate --run-dir runs/full_v6/final
```

### Windows
```powershell
.\scripts\validate_windows.ps1 -RunDir runs\full_v6\final
```

---

## Frozen Runtime Specifications

| Parameter | Value | Description |
|---|---|---|
| Configuration ID | `v6-top25-xgb39-icu-global98-20260927-v1` | Pinned configuration identifier |
| Blocking Routes | `E`, `P6`, `P4`, `ST`, `W`, `AN` | Exact, prefix 6/4, sorted tokens, word, numeric |
| Posting Cap | 500 | Max postings per baseline token |
| Rare-Address Route | 2 lowest freq tokens (≤5000) | Filter: ≥2 shared tokens & (Dice ≥0.55 \| fuzz ≥70) |
| Top Rare Additions | 25 | Max rare address additions per entity |
| Features | Exactly 39 float32 | 11 name, 5 address, 7 structural, 3 contextual, 2 numeric, 5 route/containment, 3 ICU |
| Cross-Script Engine | ICU 77.1 `Any-Latin; Latin-ASCII` | Direct C API binding via ctypes (no fallback) |
| Model Ensemble | 4-fold XGBoost (`B_cross_script`) | Arithmetic mean of four probabilities |
| Global Threshold | 0.98 | Matches accepted if ensemble prob ≥ 0.98 |
| Numeric Conflict Threshold | 0.99 | Matches accepted if ensemble prob ≥ 0.99 on numeric conflict |
| Max Matches per S1 | 11 | Maximum predictions retained per entity |
