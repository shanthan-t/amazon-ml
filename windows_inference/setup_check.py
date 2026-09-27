"""Fail-loud local dependency/model/external-artifact integrity check."""
import hashlib
import json
import platform
import sqlite3
import sys
from pathlib import Path

import numpy
import psutil
import rapidfuzz
import xgboost

from .config import MODEL_PATHS, MODEL_SHA256
from .icu import ICU

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if platform.system() != "Windows":
        print(f"NOTE: package target is Windows; running on {platform.system()}.")
    print(f"Python {sys.version.split()[0]}; NumPy {numpy.__version__}; XGBoost {xgboost.__version__}; "
          f"RapidFuzz {rapidfuzz.__version__}; psutil {psutil.__version__}")
    connection = sqlite3.connect(":memory:")
    try:
        if connection.execute("select sqlite_version()").fetchone() is None:
            raise RuntimeError("SQLite self-test failed")
    finally:
        connection.close()
    icu = ICU()
    try:
        transformed = icu("Bengaluru")
        if transformed != "bengaluru":
            raise RuntimeError(f"Pinned ICU transliteration self-test changed: {transformed!r}")
        print("ICU 77.1 C API: loaded; Any-Latin; Latin-ASCII self-test passed")
    finally:
        icu.close()

    problems = []
    for path, expected in zip(MODEL_PATHS, MODEL_SHA256):
        if not path.is_file():
            problems.append(f"Missing model: {path}")
        elif sha(path) != expected:
            problems.append(f"Model checksum mismatch: {path}")
    manifest_path = ROOT / "WINDOWS_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["artifact_type"] != "retrieval_index_or_store":
            continue
        path = ROOT / item["expected_relative_destination"]
        if not path.exists():
            problems.append(f"Missing external artifact: {path} ({item['logical_name']}, {item['size_bytes']:,} bytes)")
            continue
        if path.stat().st_size != item["size_bytes"]:
            problems.append(f"Size mismatch: {path}; expected {item['size_bytes']:,}, got {path.stat().st_size:,}")
        if item.get("sha256") and path.is_file() and sha(path) != item["sha256"]:
            problems.append(f"SHA-256 mismatch: {path}")
        if item.get("sha256_files") and path.is_dir():
            for relative, expected in item["sha256_files"].items():
                file_path = path / relative
                if not file_path.is_file() or sha(file_path) != expected:
                    problems.append(f"SHA-256 mismatch or missing file: {file_path}")
    if problems:
        print("SETUP CHECK FAILED:", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        print("Copy the listed files from COPY_TO_WINDOWS.txt, then rerun setup.", file=sys.stderr)
        return 1
    print("SETUP CHECK PASS: all four model hashes and required copied retrieval artifacts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
