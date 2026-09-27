# PC Cleanup Plan — Before Returning Borrowed PC

**Date:** 2026-09-27
**Prerequisite:** Push `final-v6-submission` to GitHub before deleting anything.

---

## Current Push Status

**Push FAILED** — current Git credential (`harsh11ith`) lacks write access to
`shanthan-t/amazon-ml`. You must authenticate as `shanthan-t` and push:

```powershell
cd "C:\Users\Indu\Desktop\amazon ml\amazon-ml"
git push origin final-v6-submission
```

**DO NOT delete anything until the push is confirmed.**

---

## SAFE_TO_DELETE_AFTER_BACKUP

These items can be deleted once the GitHub push is confirmed AND the
`AMAZON_ML_FINAL_SUBMISSION` backup folder is copied to a location you control.

| Item | Path | Size | Reason |
|------|------|------|--------|
| Python virtual environment | `amazon-ml\.venv\` | ~0.25 GiB | Recreatable from `requirements-windows.txt` |
| Run shards and checkpoints | `amazon-ml\runs\` | ~13.41 GiB | Worker outputs already merged into final TSVs |
| ICU DLLs (auto-downloaded) | `amazon-ml\native\icu\*.dll` | ~0.04 GiB | Re-downloaded by `setup_windows.ps1` |
| __pycache__ directories | `amazon-ml\**\__pycache__\` | <0.01 GiB | Regenerated on import |
| Old repo (amazon-ml-main) | `amazon-ml-main\` | ~13.8 GiB | Contains old DL code, large indexes, obsolete experiments |

**Subtotal: ~27.5 GiB**

---

## KEEP_UNTIL_SUBMISSION_CONFIRMED

These should be kept until you have confirmed the competition submission is
final and accepted. Do NOT delete prematurely.

| Item | Path | Size | Reason |
|------|------|------|--------|
| Final outputs (originals) | `amazon-ml\runs\full_v6\resume_32\final\` | ~5.1 GiB | Original validated TSVs |
| Backup copies | `Desktop\AMAZON_ML_FINAL_SUBMISSION\` | ~5.1 GiB | Independent backup with SHA256SUMS |
| Retrieval indexes | `amazon-ml\artifacts\` | ~8.63 GiB | Needed to reproduce inference |
| Git clone | `amazon-ml\.git\` | <0.1 GiB | Contains unpushed branch |

**Subtotal: ~19 GiB**

---

## DO_NOT_DELETE_WITHOUT_USER_APPROVAL

| Item | Reason |
|------|--------|
| `AMAZON_ML_FINAL_SUBMISSION\matching_results.tsv` | Your submission file — keep until results are announced |
| `AMAZON_ML_FINAL_SUBMISSION\candidate_pairs.tsv` | Supports submission validation |
| `amazon-ml\models\B_cross_script\fold*.ubj` | Tracked in Git but needed if you must rerun locally |
| Any file outside `amazon ml\` directory | Not part of this project |

---

## Cleanup Order

1. **Push the branch** (fix authentication first)
2. **Verify** the remote branch exists on GitHub
3. **Copy** `AMAZON_ML_FINAL_SUBMISSION\` to external storage (USB/cloud)
4. Delete `amazon-ml\.venv\`
5. Delete `amazon-ml\runs\` (except `runs\full_v6\resume_32\final\` if desired)
6. Delete `amazon-ml\native\icu\*.dll`
7. Delete `amazon-ml-main\` entirely
8. Delete `amazon-ml\**\__pycache__\`
9. **After competition results:** Delete remaining items

---

## Disk Summary

| Category | Size |
|----------|------|
| Immediately deletable (after push) | ~27.5 GiB |
| Keep until submission confirmed | ~19 GiB |
| Total project footprint | ~46.5 GiB |
