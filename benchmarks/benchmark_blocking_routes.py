"""Measure selective supplemental name routes on a fixed S1 sample."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import sqlite3
import statistics
import time

from benchmarks.benchmark_scoring import _load_sample, _run_streamed_workers
from src.blocking import (
    DEFAULT_MAX_POSTINGS_PER_KEY,
    build_block_keys,
    supplemental_name_route_keys,
)


ROUTES = ("exact_name", "name_prefix_6")
SOURCE_CODES = {"S2": 2, "S3": 3}
_FREQUENCY_QUERY_KEYS = None
_ACTIVE_LOOKUP = None
_BASE_ALLOWED_KEYS = None
_SAMPLE_TRUTH = None
_SAMPLE_START = None
_SAMPLE_SIZE = None


def _route_keys(name: str, country: str) -> dict[str, str]:
    return dict(
        zip(("exact_name", "name_prefix_6"), supplemental_name_route_keys(name, country))
    )


def _load_sample_ground_truth(index_path: str, start: int, size: int):
    connection = sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro", uri=True)
    try:
        source2_count = connection.execute(
            "SELECT COUNT(*) FROM targets WHERE source = 2"
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT s1_idx, target_idx FROM gt_pairs "
            "WHERE s1_idx BETWEEN ? AND ? ORDER BY s1_idx, target_idx",
            (start, start + size - 1),
        )
        truth = defaultdict(set)
        for s1_idx, target_idx in rows:
            truth[s1_idx].add(target_idx)

        denominators = {"S2": 0, "S3": 0}
        for targets in truth.values():
            for target_idx in targets:
                denominators["S2" if target_idx < source2_count else "S3"] += 1
        return source2_count, dict(truth), denominators
    finally:
        connection.close()


def _load_base_allowed_keys(index_path: str, sample_rows, cap: int):
    connection = sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro", uri=True)
    allowed = defaultdict(set)
    try:
        connection.execute(
            "CREATE TEMP TABLE sample_keys ("
            "s1_idx INTEGER NOT NULL, block_key TEXT NOT NULL, "
            "PRIMARY KEY (s1_idx, block_key)) WITHOUT ROWID"
        )
        key_rows = []
        for s1_idx, _entity_id, name, address, country in sample_rows:
            key_rows.extend(
                (s1_idx, key)
                for key in build_block_keys(name, address, country)
            )
            if len(key_rows) >= 100_000:
                connection.executemany(
                    "INSERT OR IGNORE INTO sample_keys VALUES (?, ?)", key_rows
                )
                key_rows.clear()
        if key_rows:
            connection.executemany(
                "INSERT OR IGNORE INTO sample_keys VALUES (?, ?)", key_rows
            )
        for s1_idx, block_key, source in connection.execute(
            "SELECT q.s1_idx, q.block_key, b.source "
            "FROM sample_keys AS q JOIN block_counts AS b "
            "ON b.block_key = q.block_key WHERE b.frequency <= ?",
            (cap,),
        ):
            allowed[(s1_idx, source)].add(block_key)
    finally:
        connection.close()
    return allowed


def _byte_tasks(source2: str, source3: str, workers: int):
    tasks = []
    for source, path in ((2, source2), (3, source3)):
        size = Path(path).stat().st_size
        for part in range(workers):
            start = size * part // workers
            end = size * (part + 1) // workers
            tasks.append((source, part, path, start, end))
    return tasks


def _rows_in_byte_range(path: str, start: int, end: int):
    with open(path, "rb") as source_file:
        if start == 0:
            source_file.readline()
        else:
            source_file.seek(start - 1)
            previous = source_file.read(1)
            source_file.seek(start)
            if previous != b"\n":
                source_file.readline()

        while source_file.tell() < end:
            line = source_file.readline()
            if not line:
                break
            parsed = next(csv.reader([line.decode("utf-8")], delimiter="\t"), None)
            if parsed is not None:
                yield parsed


def _frequency_worker_init(query_keys):
    global _FREQUENCY_QUERY_KEYS
    _FREQUENCY_QUERY_KEYS = query_keys


def _frequency_worker(task):
    source, part, path, start, end = task
    frequencies = {route: defaultdict(int) for route in ROUTES}
    row_count = 0
    for row in _rows_in_byte_range(path, start, end):
        if len(row) != 4:
            raise ValueError(f"Expected four TSV columns in {path}")
        keys = _route_keys(row[1], row[3])
        for route, key in keys.items():
            if key in _FREQUENCY_QUERY_KEYS[route]:
                frequencies[route][key] += 1
        row_count += 1
    return {
        "source": source,
        "part": part,
        "row_count": row_count,
        "frequencies": {
            route: dict(counts) for route, counts in frequencies.items()
        },
    }


def _candidate_worker_init(active_lookup, allowed, truth, start, size):
    global _ACTIVE_LOOKUP, _BASE_ALLOWED_KEYS, _SAMPLE_TRUTH
    global _SAMPLE_START, _SAMPLE_SIZE
    _ACTIVE_LOOKUP = active_lookup
    _BASE_ALLOWED_KEYS = allowed
    _SAMPLE_TRUTH = truth
    _SAMPLE_START = start
    _SAMPLE_SIZE = size


def _candidate_worker(task):
    source, part, path, byte_start, byte_end, target_start = task
    new_counts = [0] * _SAMPLE_SIZE
    new_hits = {"S2": 0, "S3": 0}
    route_candidates = {route: 0 for route in ROUTES}
    route_hits = {route: {"S2": 0, "S3": 0} for route in ROUTES}
    source_name = "S2" if source == 2 else "S3"
    target_idx = target_start
    row_count = 0

    for row in _rows_in_byte_range(path, byte_start, byte_end):
        if len(row) != 4:
            raise ValueError(f"Expected four TSV columns in {path}")
        _entity_id, name, address, country = row
        keys = _route_keys(name, country)
        matched_s1 = set()
        for route, key in keys.items():
            sample_s1s = _ACTIVE_LOOKUP[route][source].get(key)
            if sample_s1s:
                route_candidates[route] += len(sample_s1s)
                matched_s1.update(sample_s1s)
                for s1_idx in sample_s1s:
                    if target_idx in _SAMPLE_TRUTH.get(s1_idx, ()):
                        route_hits[route][source_name] += 1

        if matched_s1:
            target_keys = set(build_block_keys(name, address, country))
            for s1_idx in matched_s1:
                base_keys = _BASE_ALLOWED_KEYS.get((s1_idx, source), ())
                is_base_candidate = not target_keys.isdisjoint(base_keys)
                if not is_base_candidate:
                    new_counts[s1_idx - _SAMPLE_START] += 1
                    if target_idx in _SAMPLE_TRUTH.get(s1_idx, ()):
                        new_hits[source_name] += 1

        target_idx += 1
        row_count += 1

    return {
        "source": source,
        "part": part,
        "row_count": row_count,
        "new_counts": new_counts,
        "new_hits": new_hits,
        "route_candidates": route_candidates,
        "route_hits": route_hits,
    }


def _run_frequency_scan(tasks, query_keys, workers):
    frequency = {route: {2: defaultdict(int), 3: defaultdict(int)} for route in ROUTES}
    row_counts = {}
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_frequency_worker_init,
        initargs=(query_keys,),
    ) as pool:
        for completed, result in enumerate(pool.imap_unordered(_frequency_worker, tasks), 1):
            source = result["source"]
            part = result["part"]
            row_counts[(source, part)] = result["row_count"]
            for route in ROUTES:
                for key, count in result["frequencies"][route].items():
                    frequency[route][source][key] += count
            print(f"Route frequency partitions complete: {completed}/{len(tasks)}", flush=True)
    return frequency, row_counts


def _run_candidate_scan(tasks, active_lookup, allowed, truth, start, size, workers):
    new_counts = [0] * size
    new_hits = {"S2": 0, "S3": 0}
    route_candidates = {route: 0 for route in ROUTES}
    route_hits = {route: {"S2": 0, "S3": 0} for route in ROUTES}
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_candidate_worker_init,
        initargs=(active_lookup, allowed, truth, start, size),
    ) as pool:
        for completed, result in enumerate(pool.imap_unordered(_candidate_worker, tasks), 1):
            for idx, count in enumerate(result["new_counts"]):
                new_counts[idx] += count
            for source in new_hits:
                new_hits[source] += result["new_hits"][source]
            for route in ROUTES:
                route_candidates[route] += result["route_candidates"][route]
                for source in ("S2", "S3"):
                    route_hits[route][source] += result["route_hits"][route][source]
            print(f"Route candidate partitions complete: {completed}/{len(tasks)}", flush=True)
    return new_counts, new_hits, route_candidates, route_hits


def _distribution(values):
    ordered = sorted(values)
    total = sum(ordered)
    count = len(ordered)
    return {
        "candidate_pairs": total,
        "average_candidates_per_s1": total / count if count else 0.0,
        "median_candidates_per_s1": statistics.median(ordered) if ordered else 0.0,
        "zero_candidate_s1": sum(value == 0 for value in ordered),
    }


def benchmark_routes(
    *,
    source1: str,
    source2: str,
    source3: str,
    index_path: str,
    start: int,
    size: int,
    cap: int,
    workers: int,
):
    total_started = time.perf_counter()
    sample_rows = _load_sample(source1, start, size)
    baseline_started = time.perf_counter()
    baseline = _run_streamed_workers(index_path, sample_rows, workers, cap)
    baseline_seconds = time.perf_counter() - baseline_started

    source2_count, truth, gt_denominators = _load_sample_ground_truth(
        index_path, start, size
    )
    allowed_started = time.perf_counter()
    allowed = _load_base_allowed_keys(index_path, sample_rows, cap)
    allowed_seconds = time.perf_counter() - allowed_started

    route_lookup = {route: defaultdict(list) for route in ROUTES}
    base_keys = {}
    for s1_idx, _entity_id, name, address, country in sample_rows:
        route_keys = _route_keys(name, country)
        for route, key in route_keys.items():
            route_lookup[route][key].append(s1_idx)
        base_keys[s1_idx] = build_block_keys(name, address, country)

    query_keys = {
        route: set(values)
        for route, values in route_lookup.items()
    }
    target_tasks = _byte_tasks(source2, source3, workers)
    frequency_started = time.perf_counter()
    frequencies, row_counts = _run_frequency_scan(target_tasks, query_keys, workers)
    frequency_seconds = time.perf_counter() - frequency_started

    active_lookup = {route: {2: {}, 3: {}} for route in ROUTES}
    route_candidate_estimates = {route: 0 for route in ROUTES}
    for route in ROUTES:
        for source in (2, 3):
            active = {
                key: sample_s1s
                for key, sample_s1s in route_lookup[route].items()
                if frequencies[route][source].get(key, 0) <= cap
            }
            active_lookup[route][source] = active
            route_candidate_estimates[route] += sum(
                frequencies[route][source].get(key, 0) * len(sample_s1s)
                for key, sample_s1s in active.items()
            )

    candidate_tasks = []
    for source, part, path, byte_start, byte_end in target_tasks:
        first_part_rows = sum(
            row_counts[(source, previous_part)] for previous_part in range(part)
        )
        source_offset = 0 if source == 2 else source2_count
        candidate_tasks.append(
            (
                source,
                part,
                path,
                byte_start,
                byte_end,
                source_offset + first_part_rows,
            )
        )
    candidate_started = time.perf_counter()
    new_candidates, new_hits, route_candidates, route_hits = _run_candidate_scan(
        candidate_tasks, active_lookup, allowed, truth, start, size, workers
    )
    candidate_seconds = time.perf_counter() - candidate_started
    baseline_counts = baseline["counts"]["ALL"]
    combined_counts = [base + added for base, added in zip(baseline_counts, new_candidates)]
    base_hits = baseline["hits"]
    combined_hits = {source: base_hits[source] + new_hits[source] for source in ("S2", "S3")}
    gt_total = sum(gt_denominators.values())
    candidate_total = sum(combined_counts)
    hit_total = sum(combined_hits.values())

    return {
        "sample": {
            "source1": str(Path(source1).resolve()),
            "source2": str(Path(source2).resolve()),
            "source3": str(Path(source3).resolve()),
            "prepared_index": str(Path(index_path).resolve()),
            "start_row_zero_based": start,
            "rows": size,
            "workers": workers,
            "max_postings_per_key_per_source": cap,
        },
        "baseline": {
            "candidate_pairs": sum(baseline_counts),
            "candidate_hits": sum(base_hits.values()),
            "candidate_recall": sum(base_hits.values()) / gt_total if gt_total else 0.0,
        },
        "supplemental_routes": {
            "routes": [
                "country + exact normalized name",
                "country + first six compact normalized-name characters",
            ],
            "route_candidate_pairs_upper_bounds_before_route_union": route_candidate_estimates,
            "route_candidate_pairs_exact": route_candidates,
            "route_true_match_hits_including_baseline_overlap": route_hits,
            "new_candidate_pairs_after_union_with_baseline": sum(new_candidates),
            "new_true_match_hits": new_hits,
            "candidate_pairs_after_union": candidate_total,
            "candidates_per_s1_after_union": _distribution(combined_counts),
            "candidate_recall_after_union": {
                "ALL": hit_total / gt_total if gt_total else 0.0,
                "by_source": {
                    source: {
                        "ground_truth_pairs": gt_denominators[source],
                        "candidate_hits": combined_hits[source],
                        "candidate_recall": (
                            combined_hits[source] / gt_denominators[source]
                            if gt_denominators[source]
                            else 0.0
                        ),
                    }
                    for source in ("S2", "S3")
                },
            },
        },
        "timings_seconds": {
            "baseline_4_worker_scoring": baseline_seconds,
            "baseline_allowed_key_lookup": allowed_seconds,
            "supplemental_route_frequency_scan": frequency_seconds,
            "supplemental_candidate_union_scan": candidate_seconds,
            "total": time.perf_counter() - total_started,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--source2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--source3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--prepared-index", default="phase2_index_resume.sqlite3")
    parser.add_argument("--sample-start", type=int, default=400_000)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--max-postings-per-key", type=int, default=DEFAULT_MAX_POSTINGS_PER_KEY)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", default="reports/blocking_routes_sample.json")
    args = parser.parse_args()

    report = benchmark_routes(
        source1=args.source1,
        source2=args.source2,
        source3=args.source3,
        index_path=args.prepared_index,
        start=args.sample_start,
        size=args.sample_rows,
        cap=args.max_postings_per_key,
        workers=args.workers,
    )
    output_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
