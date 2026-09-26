#!/usr/bin/env python3
"""SageMaker Processing runner for V2 XGBoost test inference.

Usage inside SageMaker container:
  python3 /opt/ml/processing/input/code/sagemaker/run_v2_inference.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# SageMaker Processing directory layout
# ---------------------------------------------------------------------------
PROCESSING_ROOT = Path("/opt/ml/processing")

INPUT_CODE   = PROCESSING_ROOT / "input" / "code"
INPUT_DATA   = PROCESSING_ROOT / "input" / "data"
INPUT_INDEX  = PROCESSING_ROOT / "input" / "indexes"
INPUT_MODEL  = PROCESSING_ROOT / "input" / "model"
OUTPUT_DIR   = PROCESSING_ROOT / "output" / "results"

# Source test data
SOURCE1_PATH = INPUT_DATA / "test_source1.tsv"
SOURCE2_PATH = INPUT_DATA / "test_source2.tsv"
SOURCE3_PATH = INPUT_DATA / "test_source3.tsv"

# SQLite indexes (v2 index built on the fly or pre-built)
# If prebuilt index is uploaded, we expect it here:
PREBUILT_INDEX = INPUT_INDEX / "v2_test_index.sqlite3"
# If building on the fly, we write it to a fast local NVMe or temp storage
TEMP_INDEX = Path("/tmp/v2_test_index.sqlite3")

# Model
MODEL_JSON   = INPUT_MODEL / "xgb_v2.json"

# Outputs
MATCHING_OUTPUT   = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT  = OUTPUT_DIR / "candidate_pairs.tsv"
REPORT_OUTPUT     = OUTPUT_DIR / "v2_test_prediction.json"


def _install_dependencies() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "xgboost", "pandas", "numpy", "scikit-learn"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _set_thread_env() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def main() -> None:
    wall_start = time.time()
    print("=" * 72, flush=True)
    print("SageMaker V2 XGBoost Test Inference Runner", flush=True)
    print("=" * 72, flush=True)

    _set_thread_env()
    print("Installing dependencies...", flush=True)
    _install_dependencies()

    project_root = str(INPUT_CODE)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("\nValidating input files...", flush=True)
    if not SOURCE1_PATH.exists():
        print(f"FATAL: Missing {SOURCE1_PATH}")
        sys.exit(1)
    if not MODEL_JSON.exists():
        print(f"FATAL: Missing {MODEL_JSON}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Import logic
    from src.v2_predict import predict_test, create_index_db, build_index, finalize_index

    if PREBUILT_INDEX.exists():
        index_db_path = str(PREBUILT_INDEX)
        print(f"Using pre-built index: {index_db_path}", flush=True)
    else:
        index_db_path = str(TEMP_INDEX)
        print(f"Building test index on the fly at {index_db_path}...", flush=True)
        import sqlite3
        import gc
        t0 = time.time()
        db = create_index_db(index_db_path)
        s2_count = build_index(db, str(SOURCE2_PATH), source=2, offset=0)
        build_index(db, str(SOURCE3_PATH), source=3, offset=s2_count)
        finalize_index(db)
        db.close()
        gc.collect()
        print(f"Index built in {time.time() - t0:.0f}s", flush=True)

    print("\nStarting V2 test inference...", flush=True)
    report = predict_test(
        source1=str(SOURCE1_PATH),
        index_db=index_db_path,
        model_path=str(MODEL_JSON),
        matching_output=str(MATCHING_OUTPUT),
        report_path=str(REPORT_OUTPUT),
    )

    print("\nInference complete. Report:", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    # Post-inference verification
    if not MATCHING_OUTPUT.exists():
        print("FATAL: matching_results.tsv was not produced.", flush=True)
        sys.exit(1)

    with open(SOURCE1_PATH) as f:
        expected = sum(1 for _ in f) - 1
    with open(MATCHING_OUTPUT) as f:
        actual = sum(1 for _ in f) - 1
    
    if expected != actual:
        print(f"FATAL: Expected {expected} rows, got {actual}")
        sys.exit(1)
    
    print(f"✓ Output verified: {actual} rows")

if __name__ == "__main__":
    main()
