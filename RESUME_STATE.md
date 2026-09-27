# Resume State — Final V6 Consolidation

**Inspected:** 2026-09-27 (this session)
**Canonical git repo:** `amazon_ml_clean_repo/` (workspace root is not a git repository)

## Current branch
`main` (in-progress merge; not committed)

## Current HEAD
`c170d680bb7ef58f1cc675da472bc41c64498718`  
`c170d68 Merge pull request #1 from shanthan-t/final-v6-submission`  
Tag: `pre-final-consolidation`  
`origin/main` == local `main` at this SHA (before this session finishes the merge)

## Uncommitted files
- In-progress merge of `consolidate-final-v6` into `main` (`MERGE_HEAD` = `8b9e84a`)
- Working tree matches `consolidate-final-v6` (no extra unstaged edits vs that commit)
- Untracked: `training_v2/` (full historical production package copy; not staged)
- Untracked/ignored: `__pycache__/`, `windows_transfer/artifacts/`

## Staged files
Merge index (conflicts already resolved): rename `windows_inference/` → `src/`, add `training/` + `indexing/`, add Linux ICU `.so` libraries, truncated docs, `requirements-windows.txt` → `requirements.txt`, script module-path updates, delete `PC_CLEANUP_PLAN.md`

## Existing consolidation branch
`consolidate-final-v6` (local + `origin/consolidate-final-v6`)

## Latest consolidation commit
`8b9e84aebe71ce12491ce8cdfe7a78d44e9efffd`  
Message: `Consolidate final V6 runtime, training, and reproducibility tooling`  
Already pushed.

## Main SHA
`c170d680bb7ef58f1cc675da472bc41c64498718`

## final-v6-submission SHA
`ea3bfe67d6b034afa8ab0d5d084b48c5dad62a72` (`origin/final-v6-submission`; no local branch named `final-v6-submission`)

## Completed phases
1. Audit of final-v6-submission (original forensic `AUDIT_FINAL_V6.md` exists on `ea3bfe6`)
2. Identify essential runtime (package rename to `src/` of unchanged modules)
3. Identify essential training files (`training/*.py` copied from production_v2)
4. Identify essential index builders (`indexing/build_target_index.py`, `indexing/build_address_index.py`)
5. Identify essential validation (`src/validate.py`, `src/verify_parity.py`, `training/validate_*.py`)
6. Partial docs (`README.md`, `FINAL_V6_FILE_MANIFEST.md`, `TRAINING_ARTIFACTS.md`, gutted `AUDIT_FINAL_V6.md`)
7. requirements renamed/extended (`pandas`, `scipy`, `pytest`)
8. Four models present with original SHA256s
9. Consolidation branch created, committed, and pushed
10. Merge into `main` started; conflicts marked fixed; **commit never concluded**

## Incomplete phases
1. Finish/abort the uncommitted `main` merge
2. Restore comprehensive docs gutted in `8b9e84a` (AUDIT 571→23 lines; MANIFEST 179→14 lines)
3. Fix leftover `windows_inference` subprocess module names in `src/runner.py` and `src/benchmark.py` (rename incomplete)
4. Fix `scripts/setup_windows.ps1` still installing `requirements-windows.txt`
5. Update remaining docs (`README_WINDOWS.md`, `parity_fixture/README.md`)
6. Smoke-test imports / ICU / policy / model hashes
7. Decide `training_v2/` (KEEP/REVIEW local; do not commit duplicate + `urgent_inference.py`)
8. Merge consolidation to `main` only after verification
9. Push updated `main` if needed
10. Local workspace cleanup outside the git repo
11. Final audit update with complete hashes/tree counts

## Files already deleted (in consolidation commit / merge index)
- Package directory name `windows_inference/` (content retained as `src/`)
- `requirements-windows.txt` (renamed to `requirements.txt`)
- `PC_CLEANUP_PLAN.md` (present on `main`, absent on consolidation branch)

## Files already added
- `training/*.py` (10 scripts)
- `indexing/build_address_index.py`, `indexing/build_target_index.py`
- `native/icu/libicu*.so.77` (Linux ICU 77.1)
- `TRAINING_ARTIFACTS.md`
- Short replacements for README / AUDIT / MANIFEST

## Files still requiring review
- `training_v2/` (untracked duplicate of production training + extras)
- Parent workspace: `v6_macro_f05/`, large `*.sqlite3`, `src/`, `models/`, `output_v4/`, `dataset/`
- Whether Linux ICU `.so` files should remain tracked (already committed; needed for Linux ICU pin)
- `src/runner.py` / `src/benchmark.py` still launching `-m windows_inference.worker`

## Resume decision (this session)
- Do **not** discard the consolidation commit.
- Abort the unfinished merge on `main` (it is identical to `8b9e84a` and was never verified).
- Continue on existing `consolidate-final-v6`.
- Restore docs from `origin/final-v6-submission` and complete the `src/` rename.
- Do not retrain or rerun full inference.
