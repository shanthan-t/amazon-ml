#!/usr/bin/env python3
"""Upload artifacts for V2 XGBoost test inference to S3.

Usage:
    python3 sagemaker/upload_v2_to_s3.py \
        --bucket YOUR_BUCKET \
        --prefix amazon-ml/runs/xgb-test-v2 \
        [--profile YOUR_AWS_PROFILE]
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_MANIFEST: dict[str, list[tuple[Path, str]]] = {
    "input/code": [
        (PROJECT_ROOT / "src" / "__init__.py",         "src/__init__.py"),
        (PROJECT_ROOT / "src" / "v2_predict.py",       "src/v2_predict.py"),
        (PROJECT_ROOT / "src" / "v2_common.py",        "src/v2_common.py"),
        (PROJECT_ROOT / "src" / "normalization.py",    "src/normalization.py"),
        (PROJECT_ROOT / "src" / "data_loader.py",      "src/data_loader.py"),
        (PROJECT_ROOT / "sagemaker" / "run_v2_inference.py", "sagemaker/run_v2_inference.py"),
    ],
    "input/data": [
        (PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv", "test_source1.tsv"),
        (PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv", "test_source2.tsv"),
        (PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv", "test_source3.tsv"),
    ],
    "input/indexes": [
        # (PROJECT_ROOT / "v2_test_index.sqlite3",  "v2_test_index.sqlite3"),
    ],
    "input/model": [
        (PROJECT_ROOT / "models" / "xgb_v2.json", "xgb_v2.json"),
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
    parser.add_argument("--prefix", default="amazon-ml/runs/xgb-test-v2", help="S3 key prefix")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    prefix = args.prefix.strip("/")

    # Check if we should upload pre-built index
    idx_path = PROJECT_ROOT / "v2_test_index.sqlite3"
    if idx_path.exists():
        UPLOAD_MANIFEST["input/indexes"].append((idx_path, "v2_test_index.sqlite3"))
        # We can drop source2 and source3 from data if index is pre-built, but let's keep them just in case,
        # or the run_v2_inference handles both

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
        return

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
