"""Build a compact exact-name target index for test-time inference."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.data_loader import iter_source_chunks
from src.normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


def build_test_exact_index(
    *, source2: str, source3: str, output: str, chunk_size: int = 50_000
) -> dict[str, object]:
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {output_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    started = time.perf_counter()
    connection = sqlite3.connect(output_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-131072")
        connection.executescript(
            """
            CREATE TABLE targets (
                target_idx INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                source INTEGER NOT NULL,
                normalized_address TEXT NOT NULL,
                normalized_country TEXT NOT NULL
            );
            CREATE TABLE stage_postings (
                block_key TEXT NOT NULL,
                source INTEGER NOT NULL,
                target_idx INTEGER NOT NULL
            );
            CREATE TABLE source_rows (
                source INTEGER PRIMARY KEY,
                row_count INTEGER NOT NULL
            );
            """
        )
        source_codes = {"S2": 2, "S3": 3}
        counts = {"S2": 0, "S3": 0}
        target_idx = 0
        for source_name, path in (("S2", source2), ("S3", source3)):
            source_code = source_codes[source_name]
            target_rows = []
            posting_rows: list[tuple[str, int, int]] = []
            for chunk in iter_source_chunks(path, chunk_size=chunk_size):
                for entity_id, name, address, country in chunk.itertuples(
                    index=False, name=None
                ):
                    normalized_name = normalize_business_name(name)
                    normalized_country = normalize_country(country)
                    normalized_address = normalize_business_address(address)
                    target_rows.append(
                        (
                            target_idx,
                            entity_id,
                            source_code,
                            normalized_address,
                            normalized_country,
                        )
                    )
                    if normalized_name and normalized_country:
                        posting_rows.append(
                            (
                                f"E|{normalized_country}|{normalized_name}",
                                source_code,
                                target_idx,
                            )
                        )
                    target_idx += 1
                    counts[source_name] += 1
                connection.executemany(
                    "INSERT INTO targets VALUES (?, ?, ?, ?, ?)", target_rows
                )
                target_rows.clear()
                if posting_rows:
                    connection.executemany(
                        "INSERT INTO stage_postings VALUES (?, ?, ?)", posting_rows
                    )
                    posting_rows.clear()
                connection.commit()
                if counts[source_name] and counts[source_name] % 500_000 < len(chunk):
                    print(
                        f"Indexed {counts[source_name]:,} {source_name} targets",
                        flush=True,
                    )
            if target_rows:
                connection.executemany(
                    "INSERT INTO targets VALUES (?, ?, ?, ?, ?)", target_rows
                )
            if posting_rows:
                connection.executemany(
                    "INSERT INTO stage_postings VALUES (?, ?, ?)", posting_rows
                )
            connection.execute(
                "INSERT INTO source_rows(source, row_count) VALUES (?, ?)",
                (source_code, counts[source_name]),
            )
            connection.commit()

        print("Building ordered exact-name postings", flush=True)
        connection.executescript(
            """
            CREATE TABLE postings (
                block_key TEXT NOT NULL,
                source INTEGER NOT NULL,
                target_idx INTEGER NOT NULL,
                PRIMARY KEY (block_key, source, target_idx)
            ) WITHOUT ROWID;
            """
        )
        connection.execute(
            "INSERT INTO postings "
            "SELECT block_key, source, target_idx FROM stage_postings "
            "ORDER BY block_key, source, target_idx"
        )
        connection.commit()
        connection.execute("DROP TABLE stage_postings")
        connection.executescript(
            """
            CREATE TABLE block_counts (
                block_key TEXT NOT NULL,
                source INTEGER NOT NULL,
                frequency INTEGER NOT NULL,
                PRIMARY KEY (block_key, source)
            ) WITHOUT ROWID;
            """
        )
        connection.execute(
            "INSERT INTO block_counts "
            "SELECT block_key, source, COUNT(*) FROM postings "
            "GROUP BY block_key, source"
        )
        connection.commit()
        posting_count = connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        key_count = connection.execute("SELECT COUNT(*) FROM block_counts").fetchone()[0]
    finally:
        connection.close()

    return {
        "output": str(output_path),
        "route": "country + exact normalized name",
        "max_postings_per_key_per_source": 500,
        "targets": counts,
        "posting_rows": posting_count,
        "key_source_frequencies": key_count,
        "database_bytes": output_path.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--source3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--output", default="phase2_test_exact_index.sqlite3")
    parser.add_argument("--report", default="reports/test_exact_index.json")
    args = parser.parse_args()
    report = build_test_exact_index(
        source2=args.source2,
        source3=args.source3,
        output=args.output,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
