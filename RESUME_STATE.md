# Resume State — Final V6 Consolidation

**Inspected:** 2026-09-27 (this session, after the 16:39 usage-limit stop)
**Canonical git repo:** `amazon_ml_clean_repo/` (now the agent workspace root)

## Current branch
`main` (clean; matches `origin/main`)

## Current HEAD
`8d0dffbbd42fb195c2cce6c11188898db681dc43`  
`8d0dffb Merge consolidate-final-v6 into main as the canonical V6 project.`

## Uncommitted files
None at inspection (working tree clean). Ignored only:
- `indexing/__pycache__/`, `src/__pycache__/`, `training/__pycache__/`
- `training_v2/` (KEEP/REVIEW local duplicate of historical `production_v2`)
- `windows_transfer/artifacts/` (large runtime indexes; gitignored)

## Staged files
None.

## Existing consolidation branch
`consolidate-final-v6` (local + `origin/consolidate-final-v6`)

## Latest consolidation commit
`0146491d7dbbd7f509b4223898f4da34300088ff`  
Message: `Fix src package launch paths and restore final V6 documentation.`  
Already pushed. Merged into `main` as `8d0dffb`.

## Main SHA
`8d0dffbbd42fb195c2cce6c11188898db681dc43` (`origin/main` identical)

## final-v6-submission SHA
`ea3bfe67d6b034afa8ab0d5d084b48c5dad62a72` (`origin/final-v6-submission`; no local branch at inspection)

## Completed phases (before this session)
1. Audit of `final-v6-submission` (`AUDIT_FINAL_V6.md` restored from `ea3bfe6` + consolidation notes)
2. Audit of local project (partial; `LOCAL_PRE_CLEANUP_MANIFEST.txt` is a parent `find` dump from 16:27)
3. Essential runtime identified (`src/` rename of `windows_inference/`)
4. Essential training identified (`training/*.py`)
5. Essential index builders identified (`indexing/build_target_index.py`, `indexing/build_address_index.py`)
6. Essential evaluation/validation identified (`src/validate.py`, `src/verify_parity.py`, `training/validate_*.py`)
7. Obsolete DL/FAISS/SageMaker not present on canonical branch (history only)
8. Docs restored (`README.md`, `AUDIT_FINAL_V6.md`, `FINAL_V6_FILE_MANIFEST.md`, `TRAINING_ARTIFACTS.md`, `README_WINDOWS.md`)
9. `requirements.txt` (runtime pins + pandas/scipy/pytest)
10. Four models + SHA-256 verified in the 16:39 session
11. Smoke imports / ICU / policy / `-m src.worker` verified in the 16:39 session
12. `consolidate-final-v6` committed (`8b9e84a`, `0146491`) and pushed
13. `main` merged (`8d0dffb`) and pushed; `origin/main` == local `main`
14. Leftover `windows_inference` launch strings fixed to `-m src.worker`

## Incomplete phases (where the previous agent stopped)
The 16:39 agent hit the usage limit **during local workspace cleanup** (listing parent caches, old `models/`, `output_v4/`, sqlite indexes). GitHub/`main` was already updated.

Still unfinished at inspection:
1. Local cleanup of caches (`__pycache__`, `.pytest_cache`)
2. Classification of parent-tree KEEP vs REVIEW vs DELETE (do not move multi-GB indexes)
3. Local `final-v6-submission` tracking branch
4. Final audit note for local cleanup + current SHAs
5. Manifest stale lines (`PC_CLEANUP_PLAN.md` still listed; tracked-file count still 46)

## Files already deleted (git, consolidation)
- Package directory name `windows_inference/` (content retained as `src/`)
- `requirements-windows.txt` (replaced by `requirements.txt`)
- `PC_CLEANUP_PLAN.md`

## Files already added (git, consolidation)
- `training/*.py` + `training/__init__.py`
- `indexing/build_address_index.py`, `indexing/build_target_index.py`, `indexing/__init__.py`
- `native/icu/libicu*.so.77` (Linux ICU 77.1)
- `TRAINING_ARTIFACTS.md`, `RESUME_STATE.md`

## Files still requiring review (local, not git)
- Parent `src/` (historical V2/V4 blockers/trainers; not canonical)
- Parent `models/` (`xgb_v2`/`v3`/`v4`, logistic JSON)
- Parent `output_v4/`
- Parent `tests/`, `reports/`, `pytest.ini`
- `v6_macro_f05/` experiment tree (KEEP: training provenance + `production_v2` / `production_run_v2`)
- `amazon_ml_clean_repo/training_v2/` (ignored duplicate + `urgent_inference.py`)
- Intermediate sqlite: `phase2_index_checkpoint.sqlite3`, `phase2_index_resume.sqlite3`, `phase2_name_routes_train.sqlite3`, `phase2_test_exact_index.sqlite3`, `phase2_test_names.sqlite3`, `phase2_test_p6_routes.sqlite3`
- `archive/pre-windows-emergency` still exists locally and on origin (AUDIT previously said it was deleted; leave it)

## Resume decision (this session)
- Do **not** rewind `main` or `consolidate-final-v6`.
- Do **not** retrain or rerun full inference.
- Continue only: local cache cleanup, KEEP/REVIEW classification, smoke re-verify, audit/manifest/resume updates, local `final-v6-submission` branch, commit+push if docs change.
