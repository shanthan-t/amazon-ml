"""Store normalized test target names keyed by the existing target row offsets."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.data_loader import iter_source_chunks
from src.normalization import normalize_business_name


def build_test_name_index(*, source2: str, source3: str, exact_index: str, output: str) -> dict[str, object]:
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {output_path}")
    uri = Path(exact_index).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        expected = dict(connection.execute("SELECT source,row_count FROM source_rows"))
    finally:
        connection.close()
    started = time.perf_counter()
    db = sqlite3.connect(output_path)
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE TABLE target_names(target_idx INTEGER PRIMARY KEY, normalized_name TEXT NOT NULL)")
        target_idx = 0
        counts = {}
        for source, path in ((2, source2), (3, source3)):
            count = 0
            for chunk in iter_source_chunks(path, chunk_size=50_000):
                rows = []
                for _entity_id, name, _address, _country in chunk.itertuples(index=False, name=None):
                    rows.append((target_idx, normalize_business_name(name)))
                    target_idx += 1
                    count += 1
                db.executemany("INSERT INTO target_names VALUES (?, ?)", rows)
                db.commit()
                if count and count % 500_000 < len(chunk):
                    print(f"Stored {count:,} Source-{source} normalized names", flush=True)
            if count != expected[source]:
                raise ValueError(f"Source-{source} row count {count} != indexed {expected[source]}")
            counts[source] = count
    finally:
        db.close()
    return {
        "output": str(output_path),
        "aligned_exact_index": str(Path(exact_index).resolve()),
        "targets": counts,
        "database_bytes": output_path.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--source3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--exact-index", default="phase2_test_exact_index.sqlite3")
    parser.add_argument("--output", default="phase2_test_names.sqlite3")
    parser.add_argument("--report", default="reports/test_name_index.json")
    args = parser.parse_args()
    report = build_test_name_index(source2=args.source2, source3=args.source3, exact_index=args.exact_index, output=args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
