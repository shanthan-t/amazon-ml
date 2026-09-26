"""Disk-backed baseline blocking and candidate-recall experiment."""

from __future__ import annotations

import argparse
from array import array
import json
import multiprocessing as mp
import os
import resource
import sqlite3
import tempfile
import time
from pathlib import Path

from src.data_loader import (
    DEFAULT_CHUNK_SIZE,
    iter_ground_truth_chunks,
    iter_source_chunks,
)
from src.normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
    normalized_tokens,
)


DEFAULT_MAX_POSTINGS_PER_KEY = 500
DEFAULT_PREFIX_LENGTH = 4
POSTING_BATCH_SIZE = 100_000
SOURCE1_BATCH_SIZE = 1_000
SOURCE_CODES = {"S2": 2, "S3": 3}
_WORKER_CONNECTION: sqlite3.Connection | None = None
_WORKER_PREFIX_LENGTH: int | None = None
_WORKER_MAX_POSTINGS_PER_KEY: int | None = None
_WORKER_SUPPLEMENTAL_ROUTES = False


def supplemental_name_route_keys(business_name: str, country: str) -> tuple[str, ...]:
    """Return selective country-aware exact-name and six-character prefix keys."""
    normalized_name = normalize_business_name(business_name)
    normalized_country = normalize_country(country)
    if not normalized_name or not normalized_country:
        return ()
    compact_name = "".join(normalized_name.split())
    keys = [f"E|{normalized_country}|{normalized_name}"]
    if len(compact_name) >= 6:
        keys.append(f"P6|{normalized_country}|{compact_name[:6]}")
    return tuple(keys)


def build_block_keys(
    business_name: str,
    business_address: str,
    country: str,
    *,
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
) -> tuple[str, ...]:
    """Return country+prefix, country+name-token, and country+address-token keys."""
    if prefix_length < 1:
        raise ValueError("prefix_length must be a positive integer")

    normalized_country = normalize_country(country)
    if not normalized_country:
        return ()

    name = normalize_business_name(business_name)
    address = normalize_business_address(business_address)
    keys: list[str] = []
    seen: set[str] = set()

    def add(route: str, token: str) -> None:
        if token:
            key = f"{route}|{normalized_country}|{token}"
            if key not in seen:
                seen.add(key)
                keys.append(key)

    add("P", "".join(name.split())[:prefix_length])
    keys.extend(_supplemental_name_keys(name, normalized_country))
    for token in normalized_tokens(name):
        if len(token) >= 2:
            add("N", token)
    for token in normalized_tokens(address):
        if len(token) >= 2:
            add("A", token)
    return tuple(keys)


def _supplemental_name_keys(normalized_name: str, normalized_country: str) -> tuple[str, ...]:
    if not normalized_name or not normalized_country:
        return ()
    compact_name = "".join(normalized_name.split())
    keys = [f"E|{normalized_country}|{normalized_name}"]
    if len(compact_name) >= 6:
        keys.append(f"P6|{normalized_country}|{compact_name[:6]}")
    return tuple(keys)


def _open_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.executescript(
        """
        CREATE TABLE s1 (
            s1_idx INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL UNIQUE,
            country TEXT NOT NULL
        );
        CREATE TABLE targets (
            target_idx INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL UNIQUE,
            source INTEGER NOT NULL,
            country TEXT NOT NULL
        );
        CREATE TABLE postings (
            block_key TEXT NOT NULL,
            source INTEGER NOT NULL,
            target_idx INTEGER NOT NULL,
            PRIMARY KEY (block_key, source, target_idx)
        ) WITHOUT ROWID;
        CREATE TABLE block_counts (
            block_key TEXT NOT NULL,
            source INTEGER NOT NULL,
            frequency INTEGER NOT NULL,
            PRIMARY KEY (block_key, source)
        ) WITHOUT ROWID;
        CREATE TABLE query_keys (
            s1_idx INTEGER NOT NULL,
            block_key TEXT NOT NULL,
            PRIMARY KEY (s1_idx, block_key)
        ) WITHOUT ROWID;
        CREATE TABLE gt_s1_keys (
            entity_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE gt_raw (
            s1_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            PRIMARY KEY (s1_id, target_id)
        ) WITHOUT ROWID;
        CREATE TABLE gt_pairs (
            s1_idx INTEGER NOT NULL,
            target_idx INTEGER NOT NULL,
            PRIMARY KEY (s1_idx, target_idx)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _load_s1(connection: sqlite3.Connection, path: str, chunk_size: int) -> int:
    row_index = 0
    for chunk in iter_source_chunks(path, chunk_size=chunk_size):
        rows = []
        for entity_id, _name, _address, country in chunk.itertuples(
            index=False, name=None
        ):
            rows.append((row_index, entity_id, normalize_country(country)))
            row_index += 1
        connection.executemany(
            "INSERT INTO s1(s1_idx, entity_id, country) VALUES (?, ?, ?)", rows
        )
        connection.commit()
        if row_index and row_index % 500_000 < len(rows):
            print(f"Loaded {row_index:,} Source-1 rows", flush=True)
    return row_index


def _load_targets(
    connection: sqlite3.Connection,
    path: str,
    source: int,
    target_index: int,
    chunk_size: int,
    prefix_length: int,
) -> int:
    for chunk in iter_source_chunks(path, chunk_size=chunk_size):
        target_rows = []
        posting_rows: list[tuple[str, int, int]] = []
        for entity_id, name, address, country in chunk.itertuples(
            index=False, name=None
        ):
            normalized_country = normalize_country(country)
            current_index = target_index
            target_index += 1
            target_rows.append((current_index, entity_id, source, normalized_country))
            keys = build_block_keys(
                name,
                address,
                normalized_country,
                prefix_length=prefix_length,
            )
            posting_rows.extend((key, source, current_index) for key in keys)
            if len(posting_rows) >= POSTING_BATCH_SIZE:
                connection.executemany(
                    "INSERT INTO postings(block_key, source, target_idx) "
                    "VALUES (?, ?, ?)",
                    posting_rows,
                )
                posting_rows.clear()

        connection.executemany(
            "INSERT INTO targets(target_idx, entity_id, source, country) "
            "VALUES (?, ?, ?, ?)",
            target_rows,
        )
        if posting_rows:
            connection.executemany(
                "INSERT INTO postings(block_key, source, target_idx) "
                "VALUES (?, ?, ?)",
                posting_rows,
            )
        connection.commit()
        if target_index and target_index % 500_000 < len(target_rows):
            print(f"Indexed {target_index:,} target rows", flush=True)
    return target_index


def _build_block_counts(connection: sqlite3.Connection) -> None:
    print("Counting block-key frequencies", flush=True)
    connection.execute(
        "INSERT INTO block_counts(block_key, source, frequency) "
        "SELECT block_key, source, COUNT(*) FROM postings "
        "GROUP BY block_key, source"
    )
    connection.commit()


def _load_ground_truth(
    connection: sqlite3.Connection, path: str, chunk_size: int
) -> tuple[int, int]:
    ground_truth_rows = 0
    reference_rows = 0
    s1_batch: list[tuple[str]] = []
    pair_batch: list[tuple[str, str]] = []

    for chunk in iter_ground_truth_chunks(path, chunk_size=chunk_size):
        for s1_id, matched_ids in chunk.itertuples(index=False, name=None):
            ground_truth_rows += 1
            s1_batch.append((s1_id,))
            if matched_ids:
                for target_id in matched_ids.split(","):
                    target_id = target_id.strip()
                    if target_id:
                        reference_rows += 1
                        pair_batch.append((s1_id, target_id))
            if len(pair_batch) >= POSTING_BATCH_SIZE:
                connection.executemany(
                    "INSERT OR IGNORE INTO gt_raw(s1_id, target_id) VALUES (?, ?)",
                    pair_batch,
                )
                pair_batch.clear()
            if len(s1_batch) >= DEFAULT_CHUNK_SIZE:
                connection.executemany(
                    "INSERT OR IGNORE INTO gt_s1_keys(entity_id) VALUES (?)",
                    s1_batch,
                )
                s1_batch.clear()
        if pair_batch:
            connection.executemany(
                "INSERT OR IGNORE INTO gt_raw(s1_id, target_id) VALUES (?, ?)",
                pair_batch,
            )
            pair_batch.clear()
        if s1_batch:
            connection.executemany(
                "INSERT OR IGNORE INTO gt_s1_keys(entity_id) VALUES (?)", s1_batch
            )
            s1_batch.clear()
        connection.commit()
    return ground_truth_rows, reference_rows


def _ground_truth_integrity(
    connection: sqlite3.Connection,
    ground_truth_rows: int,
    reference_rows: int,
) -> dict[str, int]:
    s1_count = connection.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    gt_key_count = connection.execute("SELECT COUNT(*) FROM gt_s1_keys").fetchone()[0]
    missing_keys = connection.execute(
        "SELECT COUNT(*) FROM s1 LEFT JOIN gt_s1_keys "
        "ON s1.entity_id = gt_s1_keys.entity_id "
        "WHERE gt_s1_keys.entity_id IS NULL"
    ).fetchone()[0]
    extra_keys = connection.execute(
        "SELECT COUNT(*) FROM gt_s1_keys LEFT JOIN s1 "
        "ON gt_s1_keys.entity_id = s1.entity_id "
        "WHERE s1.entity_id IS NULL"
    ).fetchone()[0]
    unique_refs = connection.execute("SELECT COUNT(*) FROM gt_raw").fetchone()[0]
    connection.execute(
        """
        INSERT INTO gt_pairs(s1_idx, target_idx)
        SELECT s.s1_idx, t.target_idx
        FROM gt_raw AS g
        JOIN s1 AS s ON s.entity_id = g.s1_id
        JOIN targets AS t ON t.entity_id = g.target_id
        """
    )
    connection.commit()
    mapped_refs = connection.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
    return {
        "ground_truth_s1_rows": ground_truth_rows,
        "unique_ground_truth_s1_keys": gt_key_count,
        "source1_entities": s1_count,
        "missing_ground_truth_s1_keys": missing_keys,
        "extra_ground_truth_s1_keys": extra_keys,
        "ground_truth_reference_rows": reference_rows,
        "unique_ground_truth_pairs": unique_refs,
        "ground_truth_pairs_found_in_sources": mapped_refs,
        "ground_truth_pairs_missing_from_sources": unique_refs - mapped_refs,
    }


_CANDIDATE_ROWS_SQL = """
SELECT q.s1_idx, p.target_idx, p.source
FROM query_keys AS q
CROSS JOIN block_counts AS b
CROSS JOIN postings AS p
WHERE b.block_key = q.block_key
  AND b.frequency <= ?
  AND p.block_key = q.block_key
  AND p.source = b.source
ORDER BY q.s1_idx
"""


_CANDIDATE_ROWS_WITH_SUPPLEMENTAL_SQL = """
SELECT s1_idx, target_idx, source
FROM (
    SELECT q.s1_idx, p.target_idx, p.source
    FROM query_keys AS q
    CROSS JOIN main.block_counts AS b
    CROSS JOIN main.postings AS p
    WHERE b.block_key = q.block_key
      AND b.frequency <= ?
      AND p.block_key = q.block_key
      AND p.source = b.source
    UNION ALL
    SELECT q.s1_idx, p.target_idx, p.source
    FROM query_keys AS q
    CROSS JOIN supplemental.block_counts AS b
    CROSS JOIN supplemental.postings AS p
    WHERE b.block_key = q.block_key
      AND b.frequency <= ?
      AND p.block_key = q.block_key
      AND p.source = b.source
)
ORDER BY s1_idx
"""


_CANDIDATE_IDS_SQL = """
SELECT q.s1_idx, p.target_idx, MIN(t.entity_id)
FROM query_keys AS q
    CROSS JOIN block_counts AS b
    CROSS JOIN postings AS p
JOIN targets AS t ON t.target_idx = p.target_idx
WHERE b.block_key = q.block_key
  AND b.frequency <= ?
  AND p.block_key = q.block_key
  AND p.source = b.source
GROUP BY q.s1_idx, p.target_idx
ORDER BY q.s1_idx, p.target_idx
"""


_CANDIDATE_IDS_WITH_SUPPLEMENTAL_SQL = """
SELECT candidate_pairs.s1_idx, candidate_pairs.target_idx, MIN(t.entity_id)
FROM (
    SELECT q.s1_idx, p.target_idx, p.source
    FROM query_keys AS q
    CROSS JOIN main.block_counts AS b
    CROSS JOIN main.postings AS p
    WHERE b.block_key = q.block_key
      AND b.frequency <= ?
      AND p.block_key = q.block_key
      AND p.source = b.source
    UNION ALL
    SELECT q.s1_idx, p.target_idx, p.source
    FROM query_keys AS q
    CROSS JOIN supplemental.block_counts AS b
    CROSS JOIN supplemental.postings AS p
    WHERE b.block_key = q.block_key
      AND b.frequency <= ?
      AND p.block_key = q.block_key
      AND p.source = b.source
) AS candidate_pairs
JOIN main.targets AS t ON t.target_idx = candidate_pairs.target_idx
GROUP BY candidate_pairs.s1_idx, candidate_pairs.target_idx
ORDER BY candidate_pairs.s1_idx, candidate_pairs.target_idx
"""


def _score_query_batch(
    connection: sqlite3.Connection,
    records: list[tuple[int, str, str]],
    key_rows: list[tuple[int, str]],
    max_postings_per_key: int,
    counts: dict[str, array],
    hits: dict[str, int],
    country_hits: dict[str, dict[str, int]],
    candidate_file,
    *,
    use_supplemental: bool = False,
) -> None:
    if key_rows:
        connection.executemany(
            "INSERT OR IGNORE INTO query_keys(s1_idx, block_key) VALUES (?, ?)",
            key_rows,
        )

    ground_truth: dict[int, set[int]] = {}
    if records:
        for s1_idx, target_idx in connection.execute(
            "SELECT s1_idx, target_idx FROM gt_pairs "
            "WHERE s1_idx BETWEEN ? AND ?",
            (records[0][0], records[-1][0]),
        ):
            ground_truth.setdefault(s1_idx, set()).add(target_idx)

    candidate_sql = (
        _CANDIDATE_ROWS_WITH_SUPPLEMENTAL_SQL
        if use_supplemental
        else _CANDIDATE_ROWS_SQL
    )
    candidate_parameters = (
        (max_postings_per_key, max_postings_per_key)
        if use_supplemental
        else (max_postings_per_key,)
    )
    candidates = iter(connection.execute(candidate_sql, candidate_parameters))
    candidate_row = next(candidates, None)
    for s1_idx, _entity_id, country in records:
        if candidate_row is not None and candidate_row[0] < s1_idx:
            raise RuntimeError("candidate counts are out of Source-1 order")
        s2_candidates: set[int] = set()
        s3_candidates: set[int] = set()
        truth = ground_truth.get(s1_idx, set())
        s2_hits = s3_hits = 0
        while candidate_row is not None and candidate_row[0] == s1_idx:
            _, target_idx, source = candidate_row
            seen = s2_candidates if source == SOURCE_CODES["S2"] else s3_candidates
            if target_idx not in seen:
                seen.add(target_idx)
                if target_idx in truth:
                    truth.remove(target_idx)
                    if source == SOURCE_CODES["S2"]:
                        s2_hits += 1
                    else:
                        s3_hits += 1
            candidate_row = next(candidates, None)

        counts["S2"].append(len(s2_candidates))
        counts["S3"].append(len(s3_candidates))
        counts["ALL"].append(len(s2_candidates) + len(s3_candidates))
        hits["S2"] += s2_hits
        hits["S3"] += s3_hits
        country_counts = country_hits.setdefault(country, {"S2": 0, "S3": 0})
        country_counts["S2"] += s2_hits
        country_counts["S3"] += s3_hits

    if candidate_row is not None:
        raise RuntimeError("candidate counts exceed the current Source-1 batch")

    if candidate_file is not None:
        candidate_ids_sql = (
            _CANDIDATE_IDS_WITH_SUPPLEMENTAL_SQL
            if use_supplemental
            else _CANDIDATE_IDS_SQL
        )
        candidate_ids_parameters = (
            (max_postings_per_key, max_postings_per_key)
            if use_supplemental
            else (max_postings_per_key,)
        )
        candidate_rows = iter(
            connection.execute(candidate_ids_sql, candidate_ids_parameters)
        )
        candidate_row = next(candidate_rows, None)
        for s1_idx, entity_id, _country in records:
            candidate_file.write(entity_id + "\t")
            first = True
            if candidate_row is not None and candidate_row[0] < s1_idx:
                raise RuntimeError("candidate IDs are out of Source-1 order")
            while candidate_row is not None and candidate_row[0] == s1_idx:
                if not first:
                    candidate_file.write(",")
                candidate_file.write(candidate_row[2])
                first = False
                candidate_row = next(candidate_rows, None)
            candidate_file.write("\n")
        if candidate_row is not None:
            raise RuntimeError("candidate IDs exceed the current Source-1 batch")

    if key_rows:
        connection.execute("DELETE FROM query_keys")
        connection.commit()


def _iter_source1_batches(source1_path: str, chunk_size: int):
    batch: list[tuple[int, str, str, str, str]] = []
    s1_index = 0
    for chunk in iter_source_chunks(source1_path, chunk_size=chunk_size):
        for entity_id, name, address, country in chunk.itertuples(
            index=False, name=None
        ):
            batch.append((s1_index, entity_id, name, address, country))
            s1_index += 1
            if len(batch) >= SOURCE1_BATCH_SIZE:
                yield batch[0][0], batch
                batch = []
    if batch:
        yield batch[0][0], batch


def _initialize_scoring_worker(
    index_path: str,
    prefix_length: int,
    max_postings_per_key: int,
    supplemental_index: str | None = None,
) -> None:
    global _WORKER_CONNECTION, _WORKER_PREFIX_LENGTH, _WORKER_MAX_POSTINGS_PER_KEY
    global _WORKER_SUPPLEMENTAL_ROUTES
    uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    if supplemental_index:
        supplemental_uri = Path(supplemental_index).resolve().as_uri() + "?mode=ro"
        connection.execute(
            "ATTACH DATABASE ? AS supplemental", (supplemental_uri,)
        )
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys ("
        "s1_idx INTEGER NOT NULL, block_key TEXT NOT NULL, "
        "PRIMARY KEY (s1_idx, block_key)) WITHOUT ROWID"
    )
    _WORKER_CONNECTION = connection
    _WORKER_PREFIX_LENGTH = prefix_length
    _WORKER_MAX_POSTINGS_PER_KEY = max_postings_per_key
    _WORKER_SUPPLEMENTAL_ROUTES = bool(supplemental_index)


def _score_worker_batch(
    task: tuple[int, list[tuple[int, str, str, str, str]]]
) -> tuple[int, dict[str, array], dict[str, int], dict[str, dict[str, int]]]:
    if (
        _WORKER_CONNECTION is None
        or _WORKER_PREFIX_LENGTH is None
        or _WORKER_MAX_POSTINGS_PER_KEY is None
    ):
        raise RuntimeError("candidate worker has no SQLite connection")

    s1_start, raw_records = task
    records: list[tuple[int, str, str]] = []
    key_rows: list[tuple[int, str]] = []
    for s1_idx, entity_id, name, address, country in raw_records:
        normalized_country = normalize_country(country)
        records.append((s1_idx, entity_id, normalized_country))
        key_rows.extend(
            (s1_idx, key)
            for key in build_block_keys(
                name,
                address,
                normalized_country,
                prefix_length=_WORKER_PREFIX_LENGTH,
            )
        )

    counts = {name: array("I") for name in ("ALL", "S2", "S3")}
    hits = {"S2": 0, "S3": 0}
    country_hits: dict[str, dict[str, int]] = {}
    _score_query_batch(
        _WORKER_CONNECTION,
        records,
        key_rows,
        _WORKER_MAX_POSTINGS_PER_KEY,
        counts,
        hits,
        country_hits,
        None,
        use_supplemental=_WORKER_SUPPLEMENTAL_ROUTES,
    )
    return s1_start, counts, hits, country_hits


def _generate_candidates_parallel(
    index_path: str,
    source1_path: str,
    chunk_size: int,
    prefix_length: int,
    max_postings_per_key: int,
    workers: int,
    supplemental_index: str | None = None,
) -> tuple[dict[str, array], dict[str, int], dict[str, dict[str, int]]]:
    print(
        f"Streaming candidate generation with {workers} read-only workers",
        flush=True,
    )
    index_uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(index_uri, uri=True)
    try:
        expected_rows = connection.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    finally:
        connection.close()

    counts = {
        name: array("I", [0]) * expected_rows for name in ("ALL", "S2", "S3")
    }
    hits = {"S2": 0, "S3": 0}
    country_hits: dict[str, dict[str, int]] = {}
    context = mp.get_context("spawn")
    task_batches = _iter_source1_batches(source1_path, chunk_size)
    processed = 0
    last_reported = 0
    with context.Pool(
        processes=workers,
        initializer=_initialize_scoring_worker,
        initargs=(index_path, prefix_length, max_postings_per_key, supplemental_index),
    ) as pool:
        for s1_start, batch_counts, batch_hits, batch_country_hits in pool.imap_unordered(
            _score_worker_batch, task_batches, chunksize=1
        ):
            row_count = len(batch_counts["ALL"])
            end = s1_start + row_count
            if end > expected_rows:
                raise RuntimeError("candidate workers exceeded Source-1 row count")
            for name in counts:
                counts[name][s1_start:end] = batch_counts[name]
            for source in hits:
                hits[source] += batch_hits[source]
            for country, values in batch_country_hits.items():
                totals = country_hits.setdefault(country, {"S2": 0, "S3": 0})
                totals["S2"] += values["S2"]
                totals["S3"] += values["S3"]

            processed += row_count
            if processed - last_reported >= 100_000:
                reported = (processed // 100_000) * 100_000
                print(f"Processed {reported:,} Source-1 rows", flush=True)
                last_reported = reported

    if processed != expected_rows:
        raise RuntimeError(
            f"candidate workers processed {processed} Source-1 rows; "
            f"expected {expected_rows}"
        )
    return counts, hits, country_hits


def _generate_candidates(
    connection: sqlite3.Connection,
    source1_path: str,
    chunk_size: int,
    prefix_length: int,
    max_postings_per_key: int,
    candidate_output: str | None = None,
    *,
    workers: int = 1,
    index_path: str | None = None,
    supplemental_index: str | None = None,
) -> tuple[dict[str, array], dict[str, int], dict[str, dict[str, int]]]:
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 1:
        if candidate_output:
            raise ValueError("--candidate-output currently requires --workers 1")
        if not index_path:
            raise ValueError("an index path is required for read-only worker mode")
        return _generate_candidates_parallel(
            index_path,
            source1_path,
            chunk_size,
            prefix_length,
            max_postings_per_key,
            workers,
            supplemental_index,
        )

    print("Streaming candidate generation and recall scoring", flush=True)
    counts = {name: array("I") for name in ("ALL", "S2", "S3")}
    hits = {"S2": 0, "S3": 0}
    country_hits: dict[str, dict[str, int]] = {}
    candidate_file = None
    if candidate_output:
        candidate_path = Path(candidate_output)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_file = candidate_path.open("w", encoding="utf-8", newline="")
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")

    s1_index = 0
    last_reported = 0
    batch_records: list[tuple[int, str, str]] = []
    key_rows: list[tuple[int, str]] = []
    try:
        for chunk in iter_source_chunks(source1_path, chunk_size=chunk_size):
            for entity_id, name, address, country in chunk.itertuples(
                index=False, name=None
            ):
                normalized_country = normalize_country(country)
                batch_records.append((s1_index, entity_id, normalized_country))
                keys = build_block_keys(
                    name,
                    address,
                    normalized_country,
                    prefix_length=prefix_length,
                )
                key_rows.extend((s1_index, key) for key in keys)
                s1_index += 1
                if len(batch_records) >= SOURCE1_BATCH_SIZE:
                    _score_query_batch(
                        connection,
                        batch_records,
                        key_rows,
                        max_postings_per_key,
                        counts,
                        hits,
                        country_hits,
                        candidate_file,
                        use_supplemental=bool(supplemental_index),
                    )
                    batch_records.clear()
                    key_rows.clear()
                if s1_index - last_reported >= 100_000:
                    reported = (s1_index // 100_000) * 100_000
                    print(f"Processed {reported:,} Source-1 rows", flush=True)
                    last_reported = reported
        if batch_records:
            _score_query_batch(
                connection,
                batch_records,
                key_rows,
                max_postings_per_key,
                counts,
                hits,
                country_hits,
                candidate_file,
                use_supplemental=bool(supplemental_index),
            )
    finally:
        if candidate_file is not None:
            candidate_file.close()
    return counts, hits, country_hits


def _distribution(values: array) -> dict[str, object]:
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        median = 0.0
    elif count % 2:
        median = float(ordered[count // 2])
    else:
        median = (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    total = sum(ordered)
    zero_count = sum(value == 0 for value in ordered)
    return {
        "candidate_pairs": total,
        "average_candidates_per_s1": total / count if count else 0.0,
        "median_candidates_per_s1": median,
        "maximum_candidates_per_s1": ordered[-1] if ordered else 0,
        "zero_candidate_s1": zero_count,
        "zero_candidate_s1_pct": zero_count * 100 / count if count else 0.0,
    }


def _candidate_statistics(counts: dict[str, array]) -> dict[str, object]:
    result = _distribution(counts["ALL"])
    result["by_source"] = {
        name: _distribution(counts[name]) for name in ("S2", "S3")
    }
    return result


def _recall_statistics(
    connection: sqlite3.Connection,
    hits: dict[str, int],
    country_hits: dict[str, dict[str, int]],
) -> dict[str, object]:
    reference_rows = connection.execute(
        "SELECT CASE substr(target_id, 1, 3) "
        "WHEN 'S2-' THEN 'S2' WHEN 'S3-' THEN 'S3' ELSE 'OTHER' END, COUNT(*) "
        "FROM gt_raw GROUP BY 1"
    ).fetchall()
    reference_count = {source: count for source, count in reference_rows}
    country_rows = connection.execute(
        """
        SELECT s.country,
               CASE substr(g.target_id, 1, 3)
                 WHEN 'S2-' THEN 'S2' WHEN 'S3-' THEN 'S3' ELSE 'OTHER'
               END AS source,
               COUNT(*)
        FROM gt_raw AS g
        JOIN s1 AS s ON s.entity_id = g.s1_id
        GROUP BY s.country, substr(g.target_id, 1, 3)
        """
    ).fetchall()
    country_references: dict[str, dict[str, int]] = {}
    for country, source, total in country_rows:
        country_references.setdefault(country, {})[source] = total

    by_source = {}
    for source in ("S2", "S3"):
        total = reference_count.get(source, 0)
        found = hits[source]
        by_source[source] = {
            "ground_truth_pairs": total,
            "candidate_hits": found,
            "candidate_recall": found / total if total else 0.0,
        }

    by_country = {}
    for country, sources in country_references.items():
        total = sum(sources.get(source, 0) for source in ("S2", "S3", "OTHER"))
        found = sum(country_hits.get(country, {}).values())
        by_country[country] = {
            "ground_truth_pairs": total,
            "candidate_hits": found,
            "candidate_recall": found / total if total else 0.0,
        }

    total_refs = sum(reference_count.values())
    total_hits = sum(hits.values())
    return {
        "ground_truth_pairs": total_refs,
        "candidate_hits": total_hits,
        "candidate_recall": total_hits / total_refs if total_refs else 0.0,
        "by_source": by_source,
        "by_s1_country": by_country,
        "other_prefix_ground_truth_pairs": reference_count.get("OTHER", 0),
    }


def _count_source(connection: sqlite3.Connection, source: int) -> int:
    return connection.execute(
        "SELECT COUNT(*) FROM targets WHERE source = ?", (source,)
    ).fetchone()[0]


def _existing_ground_truth_integrity(
    connection: sqlite3.Connection,
) -> dict[str, int]:
    s1_count = connection.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    gt_key_count = connection.execute("SELECT COUNT(*) FROM gt_s1_keys").fetchone()[0]
    unique_refs = connection.execute("SELECT COUNT(*) FROM gt_raw").fetchone()[0]
    mapped_refs = connection.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
    missing_keys = connection.execute(
        "SELECT COUNT(*) FROM s1 LEFT JOIN gt_s1_keys "
        "ON s1.entity_id = gt_s1_keys.entity_id "
        "WHERE gt_s1_keys.entity_id IS NULL"
    ).fetchone()[0]
    extra_keys = connection.execute(
        "SELECT COUNT(*) FROM gt_s1_keys LEFT JOIN s1 "
        "ON gt_s1_keys.entity_id = s1.entity_id "
        "WHERE s1.entity_id IS NULL"
    ).fetchone()[0]
    return {
        "ground_truth_s1_rows": gt_key_count,
        "unique_ground_truth_s1_keys": gt_key_count,
        "source1_entities": s1_count,
        "missing_ground_truth_s1_keys": missing_keys,
        "extra_ground_truth_s1_keys": extra_keys,
        "ground_truth_reference_rows": unique_refs,
        "unique_ground_truth_pairs": unique_refs,
        "ground_truth_pairs_found_in_sources": mapped_refs,
        "ground_truth_pairs_missing_from_sources": unique_refs - mapped_refs,
    }


def _peak_rss_mib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run_prepared_index(
    *,
    source1: str,
    prepared_index: str,
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
    max_postings_per_key: int = DEFAULT_MAX_POSTINGS_PER_KEY,
    candidate_output: str | None = None,
    workers: int = 1,
    supplemental_index: str | None = None,
) -> dict[str, object]:
    """Score candidates from an existing index, without rebuilding its postings."""
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and candidate_output:
        raise ValueError("--candidate-output currently requires --workers 1")
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    if workers > 1:
        index_uri = Path(prepared_index).resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(index_uri, uri=True)
    else:
        index_uri = Path(prepared_index).resolve().as_uri() + "?mode=rw"
        connection = sqlite3.connect(index_uri, uri=True)
    if supplemental_index:
        supplemental_path = Path(supplemental_index).resolve()
        if not supplemental_path.is_file():
            connection.close()
            raise FileNotFoundError(supplemental_path)
        supplemental_uri = supplemental_path.as_uri() + "?mode=ro"
        connection.execute(
            "ATTACH DATABASE ? AS supplemental", (supplemental_uri,)
        )
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    if workers == 1:
        connection.execute("DELETE FROM query_keys")
        connection.commit()
    try:
        s1_count = connection.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
        source2_count = _count_source(connection, SOURCE_CODES["S2"])
        source3_count = _count_source(connection, SOURCE_CODES["S3"])
        if supplemental_index:
            supplemental_rows = dict(
                connection.execute(
                    "SELECT source, row_count FROM supplemental.source_rows"
                )
            )
            expected_rows = {
                SOURCE_CODES["S2"]: source2_count,
                SOURCE_CODES["S3"]: source3_count,
            }
            if supplemental_rows != expected_rows:
                raise ValueError(
                    "supplemental route index target row counts do not match "
                    "the prepared index"
                )
        integrity = _existing_ground_truth_integrity(connection)

        phase_started = time.perf_counter()
        counts, hits, country_hits = _generate_candidates(
            connection,
            source1,
            DEFAULT_CHUNK_SIZE,
            prefix_length,
            max_postings_per_key,
            candidate_output,
            workers=workers,
            index_path=prepared_index,
            supplemental_index=supplemental_index,
        )
        timings["candidate_generation_and_scoring_seconds"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        candidates = _candidate_statistics(counts)
        recall = _recall_statistics(connection, hits, country_hits)
        timings["summary_statistics_seconds"] = time.perf_counter() - phase_started
        report: dict[str, object] = {
            "configuration": {
                "prepared_index": str(Path(prepared_index).resolve()),
                "prefix_length": prefix_length,
                "max_postings_per_key_per_source": max_postings_per_key,
                "routes": [
                    "country + normalized-name prefix",
                    "country + normalized-name token",
                    "country + normalized-address token",
                    "country + exact normalized name",
                    "country + first six compact normalized-name characters",
                ],
                "supplemental_route_index": (
                    str(Path(supplemental_index).resolve()) if supplemental_index else None
                ),
                "candidate_pairs_materialized": False,
                "workers": workers,
                "scoring_mode": "read-only-multiprocessing" if workers > 1 else "serial",
            },
            "rows": {
                "source1": s1_count,
                "source2": source2_count,
                "source3": source3_count,
                "candidate_targets": source2_count + source3_count,
            },
            "ground_truth_integrity": integrity,
            "candidates": candidates,
            "recall": recall,
            "timings_seconds": timings,
            "peak_rss_mib": _peak_rss_mib(),
        }
    finally:
        connection.close()

    report["runtime_seconds"] = time.perf_counter() - total_started
    report["peak_rss_mib"] = _peak_rss_mib()
    return report


def run_experiment(
    *,
    source1: str,
    source2: str,
    source3: str,
    ground_truth: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
    max_postings_per_key: int = DEFAULT_MAX_POSTINGS_PER_KEY,
    scratch_dir: str | None = None,
    candidate_output: str | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Build the temporary index, stream candidates, and evaluate recall."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if prefix_length < 1:
        raise ValueError("prefix_length must be a positive integer")
    if max_postings_per_key < 1:
        raise ValueError("max_postings_per_key must be a positive integer")
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and candidate_output:
        raise ValueError("--candidate-output currently requires --workers 1")

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="entity_blocking_", dir=scratch_dir) as work:
        index_path = os.path.join(work, "blocking.sqlite3")
        connection = _open_database(index_path)
        try:
            phase_started = time.perf_counter()
            print("Loading Source 1", flush=True)
            s1_count = _load_s1(connection, source1, chunk_size)
            print("Indexing Source 2", flush=True)
            target_count = _load_targets(
                connection,
                source2,
                SOURCE_CODES["S2"],
                0,
                chunk_size,
                prefix_length,
            )
            source2_count = target_count
            print("Indexing Source 3", flush=True)
            target_count = _load_targets(
                connection,
                source3,
                SOURCE_CODES["S3"],
                target_count,
                chunk_size,
                prefix_length,
            )
            source3_count = target_count - source2_count
            _build_block_counts(connection)
            timings["source_and_index_build_seconds"] = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            print("Loading ground truth", flush=True)
            gt_rows, gt_reference_rows = _load_ground_truth(
                connection, ground_truth, chunk_size
            )
            integrity = _ground_truth_integrity(
                connection, gt_rows, gt_reference_rows
            )
            timings["ground_truth_load_seconds"] = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            counts, hits, country_hits = _generate_candidates(
                connection,
                source1,
                chunk_size,
                prefix_length,
                max_postings_per_key,
                candidate_output,
                workers=workers,
                index_path=index_path,
            )
            timings["candidate_generation_and_scoring_seconds"] = (
                time.perf_counter() - phase_started
            )

            phase_started = time.perf_counter()
            candidates = _candidate_statistics(counts)
            recall = _recall_statistics(connection, hits, country_hits)
            timings["summary_statistics_seconds"] = time.perf_counter() - phase_started

            report: dict[str, object] = {
                "configuration": {
                    "chunk_size": chunk_size,
                    "prefix_length": prefix_length,
                    "max_postings_per_key_per_source": max_postings_per_key,
                    "routes": [
                        "country + normalized-name prefix",
                        "country + normalized-name token",
                        "country + normalized-address token",
                    ],
                    "candidate_pairs_materialized": False,
                    "workers": workers,
                    "scoring_mode": "read-only-multiprocessing" if workers > 1 else "serial",
                },
                "rows": {
                    "source1": s1_count,
                    "source2": source2_count,
                    "source3": source3_count,
                    "candidate_targets": target_count,
                },
                "ground_truth_integrity": integrity,
                "candidates": candidates,
                "recall": recall,
                "timings_seconds": timings,
                "peak_rss_mib": _peak_rss_mib(),
            }
        finally:
            connection.close()

    report["runtime_seconds"] = time.perf_counter() - total_started
    report["peak_rss_mib"] = _peak_rss_mib()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a disk-backed baseline candidate-recall experiment."
    )
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2")
    parser.add_argument("--source3")
    parser.add_argument("--ground-truth")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--prefix-length", type=int, default=DEFAULT_PREFIX_LENGTH)
    parser.add_argument(
        "--max-postings-per-key",
        type=int,
        default=DEFAULT_MAX_POSTINGS_PER_KEY,
        help="Skip any country+route key with more postings per source than this.",
    )
    parser.add_argument(
        "--scratch-dir",
        help="Directory for the temporary SQLite index (needs several GB free).",
    )
    parser.add_argument(
        "--candidate-output",
        help="Optional path for the generated candidate TSV; can be very large.",
    )
    parser.add_argument(
        "--prepared-index",
        help="Reuse an existing SQLite index and skip source/ground-truth indexing.",
    )
    parser.add_argument(
        "--supplemental-index",
        help="Optional read-only SQLite index containing additional route postings.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Read-only scoring worker processes (1 keeps the serial scorer).",
    )
    parser.add_argument("--report", help="Optional path to write the JSON report.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.prepared_index:
        report = run_prepared_index(
            source1=args.source1,
            prepared_index=args.prepared_index,
            prefix_length=args.prefix_length,
            max_postings_per_key=args.max_postings_per_key,
            candidate_output=args.candidate_output,
            workers=args.workers,
            supplemental_index=args.supplemental_index,
        )
    else:
        if args.supplemental_index:
            raise SystemExit("--supplemental-index requires --prepared-index")
        missing = [
            name
            for name, value in (
                ("--source2", args.source2),
                ("--source3", args.source3),
                ("--ground-truth", args.ground_truth),
            )
            if not value
        ]
        if missing:
            raise SystemExit(f"Fresh index runs require: {', '.join(missing)}")
        report = run_experiment(
            source1=args.source1,
            source2=args.source2,
            source3=args.source3,
            ground_truth=args.ground_truth,
            chunk_size=args.chunk_size,
            prefix_length=args.prefix_length,
            max_postings_per_key=args.max_postings_per_key,
            scratch_dir=args.scratch_dir,
            candidate_output=args.candidate_output,
            workers=args.workers,
        )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
