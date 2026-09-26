"""Build a separate SQLite index for supplemental name blocking routes."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.blocking import POSTING_BATCH_SIZE, SOURCE_CODES, supplemental_name_route_keys
from src.data_loader import DEFAULT_CHUNK_SIZE, iter_source_chunks


def build_supplemental_route_index(
    *,
    source2: str,
    source3: str,
    output: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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

        rows_by_source = {"S2": 0, "S3": 0}
        posting_rows: list[tuple[str, int, int]] = []
        target_idx = 0
        for source_name, path in (("S2", source2), ("S3", source3)):
            source_code = SOURCE_CODES[source_name]
            for chunk in iter_source_chunks(path, chunk_size=chunk_size):
                for _entity_id, name, _address, country in chunk.itertuples(
                    index=False, name=None
                ):
                    posting_rows.extend(
                        (key, source_code, target_idx)
                        for key in supplemental_name_route_keys(name, country)
                    )
                    target_idx += 1
                    rows_by_source[source_name] += 1
                    if len(posting_rows) >= POSTING_BATCH_SIZE:
                        connection.executemany(
                            "INSERT INTO stage_postings VALUES (?, ?, ?)", posting_rows
                        )
                        connection.commit()
                        posting_rows.clear()
                if rows_by_source[source_name] and rows_by_source[source_name] % 500_000 < len(chunk):
                    print(
                        f"Indexed {rows_by_source[source_name]:,} {source_name} targets",
                        flush=True,
                    )
        if posting_rows:
            connection.executemany(
                "INSERT INTO stage_postings VALUES (?, ?, ?)", posting_rows
            )
            connection.commit()

        print("Building sorted supplemental postings", flush=True)
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
        connection.executemany(
            "INSERT INTO source_rows VALUES (?, ?)",
            [
                (SOURCE_CODES[name], count)
                for name, count in rows_by_source.items()
            ],
        )
        connection.commit()
        posting_count = connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        key_count = connection.execute("SELECT COUNT(*) FROM block_counts").fetchone()[0]
    finally:
        connection.close()

    return {
        "output": str(output_path),
        "routes": [
            "country + exact normalized name",
            "country + first six compact normalized-name characters",
        ],
        "max_postings_per_key_per_source": 500,
        "targets": rows_by_source,
        "posting_rows": posting_count,
        "key_source_frequencies": key_count,
        "database_bytes": output_path.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--source3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--output", default="phase2_name_routes_train.sqlite3")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--report", default="reports/name_route_index.json")
    args = parser.parse_args()

    report = build_supplemental_route_index(
        source2=args.source2,
        source3=args.source3,
        output=args.output,
        chunk_size=args.chunk_size,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
