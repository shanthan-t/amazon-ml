"""Build an isolated test-side index for numeric address blocking keys."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.v2_common import generate_numeric_address_keys


def build_index(target_index: str, output: str, batch_size: int = 100_000) -> dict:
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing address index: {output_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    started = time.perf_counter()
    source = sqlite3.connect(f"file:{Path(target_index).resolve()}?mode=ro", uri=True)
    destination = sqlite3.connect(output_path)
    try:
        destination.execute("PRAGMA journal_mode=OFF")
        destination.execute("PRAGMA synchronous=OFF")
        destination.execute("PRAGMA temp_store=FILE")
        destination.execute("PRAGMA cache_size=-131072")
        destination.execute(
            "CREATE TABLE stage_postings (block_key TEXT NOT NULL, target_idx INTEGER NOT NULL)"
        )
        rows = source.execute(
            "SELECT target_idx, normalized_address, normalized_country "
            "FROM targets ORDER BY target_idx"
        )
        batch = []
        target_count = posting_count = 0
        while records := rows.fetchmany(25_000):
            for target_idx, address, country in records:
                keys = generate_numeric_address_keys(address, country)
                batch.extend((key, target_idx) for key in keys)
                target_count += 1
                if len(batch) >= batch_size:
                    destination.executemany(
                        "INSERT INTO stage_postings VALUES (?, ?)", batch
                    )
                    posting_count += len(batch)
                    batch.clear()
            if target_count and target_count % 1_000_000 == 0:
                print(f"Indexed address keys for {target_count:,} test targets", flush=True)
        if batch:
            destination.executemany("INSERT INTO stage_postings VALUES (?, ?)", batch)
            posting_count += len(batch)
        destination.commit()

        print("Sorting numeric-address postings", flush=True)
        destination.execute(
            "CREATE TABLE postings ("
            "block_key TEXT NOT NULL, target_idx INTEGER NOT NULL, "
            "PRIMARY KEY (block_key, target_idx)) WITHOUT ROWID"
        )
        destination.execute(
            "INSERT INTO postings SELECT block_key, target_idx "
            "FROM stage_postings ORDER BY block_key, target_idx"
        )
        destination.execute("DROP TABLE stage_postings")
        destination.execute(
            "CREATE TABLE key_counts (block_key TEXT PRIMARY KEY, freq INTEGER NOT NULL) WITHOUT ROWID"
        )
        destination.execute(
            "INSERT INTO key_counts SELECT block_key, COUNT(*) "
            "FROM postings GROUP BY block_key"
        )
        destination.commit()
        key_count = destination.execute("SELECT COUNT(*) FROM key_counts").fetchone()[0]
    finally:
        source.close()
        destination.close()

    return {
        "target_index": str(Path(target_index).resolve()),
        "output": str(output_path),
        "route": "AN: country + exact normalized address number, at least 3 digits",
        "targets_scanned": target_count,
        "posting_rows": posting_count,
        "distinct_keys": key_count,
        "database_bytes": output_path.stat().st_size,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", default="v2_test_index.sqlite3")
    parser.add_argument("--output", default="phase2_test_numeric_address.sqlite3")
    parser.add_argument("--report", default="reports/test_numeric_address_index.json")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    report = build_index(args.target_index, args.output, args.batch_size)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
