"""Build a disk-only P6 route sidecar aligned to the existing test target index."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.data_loader import iter_source_chunks
from src.normalization import normalize_business_name, normalize_country


def build_test_p6_index(*, source2: str, source3: str, exact_index: str, output: str) -> dict[str, object]:
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {output_path}")
    exact_uri = Path(exact_index).resolve().as_uri() + "?mode=ro"
    exact = sqlite3.connect(exact_uri, uri=True)
    try:
        counts = dict(exact.execute("SELECT source,row_count FROM source_rows"))
    finally:
        exact.close()
    started = time.perf_counter()
    db = sqlite3.connect(output_path)
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-131072")
        db.executescript(
            """
            CREATE TABLE stage_postings(block_key TEXT NOT NULL, source INTEGER NOT NULL, target_idx INTEGER NOT NULL);
            CREATE TABLE source_rows(source INTEGER PRIMARY KEY, row_count INTEGER NOT NULL);
            """
        )
        target_idx = 0
        rows_by_source = {2: 0, 3: 0}
        for source, path in ((2, source2), (3, source3)):
            count = 0
            for chunk in iter_source_chunks(path, chunk_size=50_000):
                postings = []
                for _entity_id, name, _address, country in chunk.itertuples(index=False, name=None):
                    normalized_name = normalize_business_name(name)
                    normalized_country = normalize_country(country)
                    compact = "".join(normalized_name.split())
                    if normalized_country and len(compact) >= 6:
                        postings.append((f"P6|{normalized_country}|{compact[:6]}", source, target_idx))
                    target_idx += 1
                    count += 1
                if postings:
                    db.executemany("INSERT INTO stage_postings VALUES (?, ?, ?)", postings)
                db.commit()
                if count and count % 500_000 < len(chunk):
                    print(f"Scanned {count:,} Source-{source} rows for P6 keys", flush=True)
            rows_by_source[source] = count
            db.execute("INSERT INTO source_rows VALUES (?, ?)", (source, count))
            expected = counts[source]
            if count != expected:
                raise ValueError(f"Source-{source} has {count} rows, expected {expected}")
        print("Building ordered P6 postings", flush=True)
        db.executescript(
            "CREATE TABLE postings(block_key TEXT NOT NULL,source INTEGER NOT NULL,target_idx INTEGER NOT NULL,PRIMARY KEY(block_key,source,target_idx)) WITHOUT ROWID;"
        )
        db.execute("INSERT INTO postings SELECT block_key,source,target_idx FROM stage_postings ORDER BY block_key,source,target_idx")
        db.commit()
        db.execute("DROP TABLE stage_postings")
        db.executescript(
            "CREATE TABLE block_counts(block_key TEXT NOT NULL,source INTEGER NOT NULL,frequency INTEGER NOT NULL,PRIMARY KEY(block_key,source)) WITHOUT ROWID;"
        )
        db.execute("INSERT INTO block_counts SELECT block_key,source,COUNT(*) FROM postings GROUP BY block_key,source")
        db.commit()
        posting_count = db.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        key_count = db.execute("SELECT COUNT(*) FROM block_counts").fetchone()[0]
    finally:
        db.close()
    return {
        "output": str(output_path),
        "aligned_exact_index": str(Path(exact_index).resolve()),
        "route": "country + first six compact normalized-name characters",
        "max_postings_per_key_per_source": 500,
        "targets": rows_by_source,
        "posting_rows": posting_count,
        "key_source_frequencies": key_count,
        "database_bytes": output_path.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--source3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--exact-index", default="phase2_test_exact_index.sqlite3")
    parser.add_argument("--output", default="phase2_test_p6_routes.sqlite3")
    parser.add_argument("--report", default="reports/test_p6_index.json")
    args = parser.parse_args()
    report = build_test_p6_index(source2=args.source2, source3=args.source3, exact_index=args.exact_index, output=args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
