#!/usr/bin/env python3
"""Upload ONLY the required artifacts for P6+exact test inference to S3.

Usage:
    python3 sagemaker/upload_to_s3.py \
        --bucket YOUR_BUCKET \
        --prefix amazon-ml/runs/p6-test-v2 \
        [--profile YOUR_AWS_PROFILE]

This uploads:
  1. Project source code  -> s3://BUCKET/PREFIX/input/code/
  2. Test data (source1)  -> s3://BUCKET/PREFIX/input/data/
  3. SQLite indexes       -> s3://BUCKET/PREFIX/input/indexes/
  4. Model JSON           -> s3://BUCKET/PREFIX/input/model/

It does NOT upload:
  - output/ or output_v2/ (existing results)
  - training indexes (phase2_index_*.sqlite3, phase2_name_routes_train.sqlite3)
  - __pycache__, .pytest_cache, reports/, benchmarks/, experiments/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import boto3
except ImportError:
    print("ERROR: boto3 is required. Install with: pip install boto3", file=sys.stderr)
    sys.exit(1)


# Project root (parent of sagemaker/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files to upload, grouped by S3 channel
UPLOAD_MANIFEST: dict[str, list[tuple[Path, str]]] = {
    # Channel: list of (local_path, s3_key_suffix)
    "input/code": [
        # Source package
        (PROJECT_ROOT / "src" / "__init__.py",         "src/__init__.py"),
        (PROJECT_ROOT / "src" / "predict_test.py",     "src/predict_test.py"),
        (PROJECT_ROOT / "src" / "features.py",         "src/features.py"),
        (PROJECT_ROOT / "src" / "normalization.py",    "src/normalization.py"),
        (PROJECT_ROOT / "src" / "data_loader.py",      "src/data_loader.py"),
        (PROJECT_ROOT / "src" / "blocking.py",         "src/blocking.py"),
        # SageMaker runner and requirements
        (PROJECT_ROOT / "sagemaker" / "run_test_inference.py",      "sagemaker/run_test_inference.py"),
        (PROJECT_ROOT / "sagemaker" / "requirements_inference.txt", "sagemaker/requirements_inference.txt"),
    ],
    "input/data": [
        (PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv", "test_source1.tsv"),
    ],
    "input/indexes": [
        (PROJECT_ROOT / "phase2_test_exact_index.sqlite3",  "phase2_test_exact_index.sqlite3"),
        (PROJECT_ROOT / "phase2_test_p6_routes.sqlite3",    "phase2_test_p6_routes.sqlite3"),
        (PROJECT_ROOT / "phase2_test_names.sqlite3",        "phase2_test_names.sqlite3"),
    ],
    "input/model": [
        (PROJECT_ROOT / "models" / "logistic_p6_exact_candidate.json", "logistic_p6_exact_candidate.json"),
    ],
}


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / 1024**3:.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / 1024**2:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="amazon-ml/runs/p6-test-v2", help="S3 key prefix (default: amazon-ml/runs/p6-test-v2)")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    prefix = args.prefix.strip("/")

    # Validate all files exist before uploading anything
    print("Validating local files...", flush=True)
    all_files: list[tuple[Path, str]] = []
    total_bytes = 0
    for channel, files in UPLOAD_MANIFEST.items():
        for local_path, key_suffix in files:
            if not local_path.exists():
                print(f"  ERROR: Missing: {local_path}", flush=True)
                sys.exit(1)
            size = local_path.stat().st_size
            s3_key = f"{prefix}/{channel}/{key_suffix}"
            all_files.append((local_path, s3_key))
            total_bytes += size
            print(f"  ✓ {local_path.name} ({_format_size(size)}) -> s3://{args.bucket}/{s3_key}", flush=True)

    print(f"\nTotal: {len(all_files)} files, {_format_size(total_bytes)}", flush=True)

    if args.dry_run:
        print("\n[DRY RUN] No files uploaded.", flush=True)
        return

    # Upload
    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    print(f"\nUploading to s3://{args.bucket}/{prefix}/...", flush=True)
    start = time.time()
    uploaded = 0
    for local_path, s3_key in all_files:
        size = local_path.stat().st_size
        print(f"  Uploading {local_path.name} ({_format_size(size)})...", end="", flush=True)
        s3.upload_file(str(local_path), args.bucket, s3_key)
        uploaded += size
        print(f" done.", flush=True)

    elapsed = time.time() - start
    print(f"\n✓ Upload complete: {len(all_files)} files, {_format_size(uploaded)} in {elapsed:.1f}s", flush=True)
    print(f"\nS3 prefix: s3://{args.bucket}/{prefix}/", flush=True)


if __name__ == "__main__":
    main()
