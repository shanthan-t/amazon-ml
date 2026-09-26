#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Building Blocking & Address Sidecar Indexes"
echo " (Run this once after copying dataset/ to this directory)"
echo "=========================================================="

mkdir -p reports

echo "[1/4] Checking/Building Train Target Index (v2_train_index.sqlite3)..."
python3 -c "
from src.v2_train import build_train_target_index
from pathlib import Path
if not Path('v2_train_index.sqlite3').exists():
    build_train_target_index('dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv', 'v2_train_index.sqlite3')
else:
    print('  v2_train_index.sqlite3 already exists, skipping.')
"

echo "[2/4] Checking/Building Test Target Index (v2_test_index.sqlite3)..."
python3 -c "
from src.v2_predict import build_test_target_index
from pathlib import Path
if not Path('v2_test_index.sqlite3').exists():
    build_test_target_index('dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv', 'v2_test_index.sqlite3')
else:
    print('  v2_test_index.sqlite3 already exists, skipping.')
"

echo "[3/4] Checking/Building Test Numeric Address Index..."
if [ ! -f phase2_test_numeric_address.sqlite3 ]; then
  PYTHONPATH=. python3 -m src.test_numeric_address_index \
    --target-index v2_test_index.sqlite3 \
    --output phase2_test_numeric_address.sqlite3 \
    --report reports/test_numeric_address_index.json \
    --batch-size 100000
else
  echo "  phase2_test_numeric_address.sqlite3 already exists, skipping."
fi

echo "[4/4] Checking/Building Train Numeric Address Index..."
if [ ! -f phase2_train_numeric_address.sqlite3 ]; then
  PYTHONPATH=. python3 -m src.test_numeric_address_index \
    --target-index v2_train_index.sqlite3 \
    --output phase2_train_numeric_address.sqlite3 \
    --report reports/train_numeric_address_index.json \
    --batch-size 100000
else
  echo "  phase2_train_numeric_address.sqlite3 already exists, skipping."
fi

echo "=========================================================="
echo " All SQLite indexes successfully prepared!"
echo "=========================================================="
