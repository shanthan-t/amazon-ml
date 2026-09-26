#!/usr/bin/env bash
set -e

# Default settings (can be overridden via environment variables or CLI flags)
WORKERS=${WORKERS:-$(nproc)}
MODEL=${MODEL:-"models/xgb_v4.json"}
OUT_DIR=${OUT_DIR:-"output_submission"}
ADDRESS_INDEX=${ADDRESS_INDEX:-"phase2_test_numeric_address.sqlite3"}
TEST_INDEX=${TEST_INDEX:-"v2_test_index.sqlite3"}

echo "=========================================================="
echo " Amazon ML Challenge — Parallel Inference Run"
echo " Cores / Workers : ${WORKERS}"
echo " Model           : ${MODEL}"
echo " Output Directory: ${OUT_DIR}"
echo "=========================================================="

mkdir -p "${OUT_DIR}" reports

# Remove old output if user re-runs
rm -f "${OUT_DIR}/matching_results.tsv" "${OUT_DIR}/inference_report.json"

PYTHONPATH=. python3 -m src.v2_predict_parallel \
  --model "${MODEL}" \
  --index-db "${TEST_INDEX}" \
  --address-index "${ADDRESS_INDEX}" \
  --matching "${OUT_DIR}/matching_results.tsv" \
  --report "${OUT_DIR}/inference_report.json" \
  --workers "${WORKERS}"

echo "=========================================================="
echo " Inference Complete!"
echo " Submission file ready at: ${OUT_DIR}/matching_results.tsv"
echo "=========================================================="
