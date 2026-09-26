#!/usr/bin/env python3
"""SageMaker Processing runner for P6+exact test inference.

This script is the ENTRYPOINT for the SageMaker Processing job. It:
  1. Resolves SageMaker-standard /opt/ml/processing/ paths.
  2. Installs pinned dependencies.
  3. Invokes the existing src.predict_test.predict_test() function
     WITHOUT duplicating or changing any prediction logic.
  4. Runs a final row-completeness verification.
  5. Exits non-zero if the output is incomplete (fail-fast).

Usage inside SageMaker container:
  python3 /opt/ml/processing/input/code/sagemaker/run_test_inference.py

All paths are derived from the SageMaker Processing input/output channels:
  - /opt/ml/processing/input/code/     <- project source tree
  - /opt/ml/processing/input/data/     <- test_source1.tsv
  - /opt/ml/processing/input/indexes/  <- SQLite indexes
  - /opt/ml/processing/input/model/    <- model JSON
  - /opt/ml/processing/output/results/ <- matching_results.tsv,
                                          candidate_pairs.tsv,
                                          test_prediction_v2.json
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

# Source-1 test data
SOURCE1_PATH = INPUT_DATA / "test_source1.tsv"

# SQLite indexes (pre-built, read-only)
EXACT_INDEX  = INPUT_INDEX / "phase2_test_exact_index.sqlite3"
P6_INDEX     = INPUT_INDEX / "phase2_test_p6_routes.sqlite3"
NAME_INDEX   = INPUT_INDEX / "phase2_test_names.sqlite3"

# Model
MODEL_PATH   = INPUT_MODEL / "logistic_p6_exact_candidate.json"

# Outputs
MATCHING_OUTPUT   = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT  = OUTPUT_DIR / "candidate_pairs.tsv"
REPORT_OUTPUT     = OUTPUT_DIR / "test_prediction_v2.json"


def _install_dependencies() -> None:
    """Install pinned inference dependencies inside the container."""
    req_file = INPUT_CODE / "sagemaker" / "requirements_inference.txt"
    if req_file.exists():
        print(f"Installing dependencies from {req_file}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    else:
        print(f"WARNING: {req_file} not found; relying on container packages", flush=True)


def _set_thread_env() -> None:
    """Prevent native math libraries from oversubscribing CPU cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _validate_inputs() -> None:
    """Verify all required inputs are present before starting inference."""
    required = {
        "test_source1.tsv": SOURCE1_PATH,
        "phase2_test_exact_index.sqlite3": EXACT_INDEX,
        "phase2_test_p6_routes.sqlite3": P6_INDEX,
        "phase2_test_names.sqlite3": NAME_INDEX,
        "logistic_p6_exact_candidate.json": MODEL_PATH,
    }
    missing = []
    for label, path in required.items():
        if not path.exists():
            missing.append(f"  {label}: {path}")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {label}: {path} ({size_mb:.1f} MB)", flush=True)
    if missing:
        print("\nERROR: Missing required input files:", flush=True)
        for m in missing:
            print(m, flush=True)
        sys.exit(1)


def _count_source1_entities() -> int:
    """Count non-header lines in test_source1.tsv for verification."""
    count = 0
    with open(SOURCE1_PATH, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for _ in f:
            count += 1
    return count


def _verify_output(expected_s1_count: int) -> None:
    """Verify matching_results.tsv has exactly one row per Source-1 entity.

    Exits non-zero if the output is partial or missing.
    """
    if not MATCHING_OUTPUT.exists():
        print("FATAL: matching_results.tsv was not produced.", flush=True)
        sys.exit(1)

    actual = 0
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        header = next(f).strip()
        if header != "source1_entity_id\tmatched_entity_ids":
            print(f"FATAL: Unexpected header: {header}", flush=True)
            sys.exit(1)
        for _ in f:
            actual += 1

    if actual != expected_s1_count:
        print(
            f"FATAL: matching_results.tsv has {actual:,} data rows, "
            f"but test_source1.tsv has {expected_s1_count:,} entities. "
            f"Output is INCOMPLETE.",
            flush=True,
        )
        sys.exit(1)

    print(
        f"✓ Verification passed: {actual:,} rows in matching_results.tsv "
        f"== {expected_s1_count:,} Source-1 entities.",
        flush=True,
    )


def main() -> None:
    wall_start = time.time()

    print("=" * 72, flush=True)
    print("SageMaker P6+Exact Test Inference Runner", flush=True)
    print("=" * 72, flush=True)

    # 1. Set environment
    _set_thread_env()

    # 2. Install dependencies
    _install_dependencies()

    # 3. Add project root to sys.path so `src` package is importable
    project_root = str(INPUT_CODE)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"\nProject root: {project_root}", flush=True)

    # 4. Validate inputs
    print("\nValidating input files...", flush=True)
    _validate_inputs()

    # 5. Count expected S1 entities BEFORE inference
    print("\nCounting Source-1 entities...", flush=True)
    expected_s1 = _count_source1_entities()
    print(f"  Expected entities: {expected_s1:,}", flush=True)

    # 6. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Run the existing prediction function (NO logic changes)
    print("\n" + "=" * 72, flush=True)
    print("Starting P6+exact test inference...", flush=True)
    print("=" * 72, flush=True)

    # Import AFTER sys.path is set and dependencies are installed
    from src.predict_test import predict_test

    report = predict_test(
        source1=str(SOURCE1_PATH),
        index_path=str(EXACT_INDEX),
        model_path=str(MODEL_PATH),
        matching_output=str(MATCHING_OUTPUT),
        candidate_output=str(CANDIDATE_OUTPUT),
        report_path=str(REPORT_OUTPUT),
        p6_index=str(P6_INDEX),
        name_index=str(NAME_INDEX),
    )

    print("\nInference complete. Report:", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    # 8. Post-inference verification
    print("\n" + "=" * 72, flush=True)
    print("Running output verification...", flush=True)
    print("=" * 72, flush=True)
    _verify_output(expected_s1)

    # 9. Check candidate_pairs.tsv exists
    if not CANDIDATE_OUTPUT.exists():
        print("WARNING: candidate_pairs.tsv was not produced.", flush=True)
    else:
        cand_size = CANDIDATE_OUTPUT.stat().st_size / (1024 * 1024)
        print(f"✓ candidate_pairs.tsv: {cand_size:.1f} MB", flush=True)

    # 10. Check report exists
    if not REPORT_OUTPUT.exists():
        print("WARNING: test_prediction_v2.json was not produced.", flush=True)
    else:
        print(f"✓ test_prediction_v2.json written.", flush=True)

    wall_elapsed = time.time() - wall_start
    print(f"\nTotal wall time: {wall_elapsed:.1f}s ({wall_elapsed / 60:.1f} min)", flush=True)
    print("\n✓ SageMaker Processing job completed successfully.", flush=True)


if __name__ == "__main__":
    main()
