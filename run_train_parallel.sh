#!/usr/bin/env bash
set -e

WORKERS=${WORKERS:-$(nproc)}
SAMPLE_ROWS=${SAMPLE_ROWS:-200000}
INDEX_DB=${INDEX_DB:-"v2_train_index.sqlite3"}
ADDRESS_INDEX=${ADDRESS_INDEX:-"phase2_train_numeric_address.sqlite3"}

echo "=========================================================="
echo " Amazon ML Challenge — Parallel XGBoost Training"
echo " Workers     : ${WORKERS}"
echo " Sample Rows : ${SAMPLE_ROWS}"
echo " Index DB    : ${INDEX_DB}"
echo "=========================================================="

mkdir -p models reports

PYTHONPATH=. python3 -m src.v2_train_parallel \
  --sample-rows "${SAMPLE_ROWS}" \
  --index-db "${INDEX_DB}" \
  --address-index "${ADDRESS_INDEX}" \
  --workers "${WORKERS}"

echo "=========================================================="
echo " Training Complete! Check models/xgb_v4.json"
echo "=========================================================="
