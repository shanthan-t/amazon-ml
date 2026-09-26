"""Measure individual/cumulative blocker-route contributions on a fixed sample."""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src.blocking import (
    DEFAULT_MAX_POSTINGS_PER_KEY,
    DEFAULT_PREFIX_LENGTH,
    build_block_keys,
)
from src.train_model import DEFAULT_SAMPLE_ROWS, DEFAULT_SAMPLE_START, _read_source1_sample


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("P", "N", "A", "E", "P6")
UNION_ORDER = ROUTES


def _connect(base_index: Path, supplemental_index: Path) -> sqlite3.Connection:
    uri = base_index.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute(
        "ATTACH DATABASE ? AS supplemental",
        (supplemental_index.resolve().as_uri() + "?mode=ro",),
    )
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys ("
        "s1_idx INTEGER NOT NULL, route TEXT NOT NULL, block_key TEXT NOT NULL, "
        "PRIMARY KEY(s1_idx, route, block_key)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TEMP TABLE batch_s1 (s1_idx INTEGER PRIMARY KEY) WITHOUT ROWID"
    )
    return connection


_AGGREGATE_SQL = """
WITH route_candidates AS (
    SELECT q.s1_idx,p.target_idx,q.route
    FROM query_keys q
    CROSS JOIN main.block_counts b
    CROSS JOIN main.postings p
    WHERE b.block_key=q.block_key AND b.frequency<=?
      AND p.block_key=q.block_key AND p.source=b.source
      AND q.route IN ('P','N','A')
    UNION ALL
    SELECT q.s1_idx,p.target_idx,q.route
    FROM query_keys q
    CROSS JOIN supplemental.block_counts b
    CROSS JOIN supplemental.postings p
    WHERE b.block_key=q.block_key AND b.frequency<=?
      AND p.block_key=q.block_key AND p.source=b.source
      AND q.route IN ('E','P6')
), pair_routes AS (
    SELECT s1_idx,target_idx,
           MAX(route='P') p, MAX(route='N') n, MAX(route='A') a,
           MAX(route='E') e, MAX(route='P6') p6
    FROM route_candidates GROUP BY s1_idx,target_idx
), scored AS (
    SELECT r.*, (g.target_idx IS NOT NULL) hit
    FROM pair_routes r
    LEFT JOIN main.gt_pairs g ON g.s1_idx=r.s1_idx AND g.target_idx=r.target_idx
), totals AS (
    SELECT s1_idx,
           SUM(p) pc, SUM(p*hit) ph,
           SUM(n) nc, SUM(n*hit) nh,
           SUM(a) ac, SUM(a*hit) ah,
           SUM(e) ec, SUM(e*hit) eh,
           SUM(p6) p6c, SUM(p6*hit) p6h,
           SUM(p OR n) pnc, SUM((p OR n)*hit) pnh,
           SUM(p OR n OR a) pnac, SUM((p OR n OR a)*hit) pnah,
           SUM(p OR n OR a OR e) pnaec, SUM((p OR n OR a OR e)*hit) pnaeh,
           SUM(p OR n OR a OR e OR p6) allc,
           SUM((p OR n OR a OR e OR p6)*hit) allh
    FROM scored GROUP BY s1_idx
)
SELECT q.s1_idx,
       COALESCE(t.pc,0),COALESCE(t.ph,0),COALESCE(t.nc,0),COALESCE(t.nh,0),
       COALESCE(t.ac,0),COALESCE(t.ah,0),COALESCE(t.ec,0),COALESCE(t.eh,0),
       COALESCE(t.p6c,0),COALESCE(t.p6h,0),COALESCE(t.pnc,0),COALESCE(t.pnh,0),
       COALESCE(t.pnac,0),COALESCE(t.pnah,0),COALESCE(t.pnaec,0),COALESCE(t.pnaeh,0),
       COALESCE(t.allc,0),COALESCE(t.allh,0)
FROM batch_s1 q LEFT JOIN totals t ON t.s1_idx=q.s1_idx ORDER BY q.s1_idx
"""


def _metric(counts, hits, gt_total, rows):
    ordered = [counts[index] for index in rows]
    return {
        "candidate_pairs": sum(ordered),
        "average_candidates_per_s1": sum(ordered) / len(ordered) if ordered else 0.0,
        "median_candidates_per_s1": statistics.median(ordered) if ordered else 0.0,
        "zero_candidate_s1": sum(value == 0 for value in ordered),
        "ground_truth_hits": hits,
        "candidate_recall": hits / gt_total if gt_total else 0.0,
    }


def _candidate_only_macro_ceiling(hit_values, truth, rows):
    scores = []
    for offset, s1_idx in enumerate(rows):
        positives = len(truth.get(s1_idx, ()))
        if positives == 0:
            scores.append(1.0)
            continue
        recall = hit_values[offset] / positives
        scores.append(1.25 * recall / (0.25 + recall) if recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_audit(
    *, sample_start: int, sample_rows: int, cap: int, batch_rows: int,
    base_index: Path, supplemental_index: Path, source1_path: Path, report_path: Path,
):
    started = time.perf_counter()
    sample, _ = _read_source1_sample(str(source1_path), sample_start, sample_rows)
    by_idx = {row[0]: row for row in sample}
    ids = list(by_idx)
    connection = _connect(base_index, supplemental_index)
    try:
        s2_count = connection.execute("SELECT COUNT(*) FROM main.targets WHERE source=2").fetchone()[0]
        truth: dict[int, set[int]] = defaultdict(set)
        for s1_idx, target_idx in connection.execute(
            "SELECT s1_idx,target_idx FROM main.gt_pairs WHERE s1_idx BETWEEN ? AND ?",
            (ids[0], ids[-1]),
        ):
            truth[s1_idx].add(target_idx)
        gt_by_source = {
            "S2": sum(target < s2_count for values in truth.values() for target in values),
            "S3": sum(target >= s2_count for values in truth.values() for target in values),
        }
        gt_total = sum(gt_by_source.values())

        fields = {
            "P": (0, 1), "N": (2, 3), "A": (4, 5), "E": (6, 7), "P6": (8, 9),
            "P+N": (10, 11), "P+N+A": (12, 13),
            "P+N+A+E": (14, 15), "P+N+A+E+P6": (16, 17),
        }
        counts_by_s1 = {key: [0] * sample_rows for key in fields}
        hits_by_s1 = {key: [0] * sample_rows for key in fields}
        query_seconds = 0.0
        batch_started = 0

        for offset in range(0, sample_rows, batch_rows):
            batch = sample[offset : offset + batch_rows]
            connection.executemany(
                "INSERT INTO batch_s1 VALUES (?)", ((row[0],) for row in batch)
            )
            keyed_rows = []
            for s1_idx, _entity_id, name, address, country in batch:
                for key in build_block_keys(
                    name, address, country, prefix_length=DEFAULT_PREFIX_LENGTH
                ):
                    route = key.split("|", 1)[0]
                    keyed_rows.append((s1_idx, route, key))
            connection.executemany("INSERT INTO query_keys VALUES (?,?,?)", keyed_rows)
            query_started = time.perf_counter()
            aggregates = connection.execute(_AGGREGATE_SQL, (cap, cap)).fetchall()
            query_seconds += time.perf_counter() - query_started
            if len(aggregates) != len(batch):
                raise RuntimeError("route aggregate omitted Source-1 rows")
            for row_offset, aggregate in enumerate(aggregates, start=offset):
                for label, (count_column, hit_column) in fields.items():
                    counts_by_s1[label][row_offset] = int(aggregate[count_column + 1])
                    hits_by_s1[label][row_offset] = int(aggregate[hit_column + 1])
            connection.execute("DELETE FROM query_keys")
            connection.execute("DELETE FROM batch_s1")
            connection.commit()
            batch_started += len(batch)
            if batch_started % max(batch_rows * 10, 1) == 0 or batch_started == sample_rows:
                print(f"Audited routes for {batch_started:,}/{sample_rows:,} S1 rows", flush=True)

        routes_report = {}
        for route in ROUTES:
            routes_report[route] = _metric(
                counts_by_s1[route], sum(hits_by_s1[route]), gt_total, range(sample_rows)
            )
            routes_report[route]["oracle_macro_f0_5_ceiling_from_candidates"] = (
                _candidate_only_macro_ceiling(hits_by_s1[route], truth, ids)
            )
        union_order = ("P", "P+N", "P+N+A", "P+N+A+E", "P+N+A+E+P6")
        prior_count = prior_hits = 0
        for route in union_order:
            current = _metric(
                counts_by_s1[route], sum(hits_by_s1[route]), gt_total, range(sample_rows)
            )
            current["incremental_candidate_pairs_over_previous_union"] = current["candidate_pairs"] - prior_count
            current["incremental_ground_truth_hits_over_previous_union"] = current["ground_truth_hits"] - prior_hits
            routes_report[route] = current
            routes_report[route]["oracle_macro_f0_5_ceiling_from_candidates"] = (
                _candidate_only_macro_ceiling(hits_by_s1[route], truth, ids)
            )
            prior_count, prior_hits = current["candidate_pairs"], current["ground_truth_hits"]
        report = {
            "sample": {
                "source1": str(source1_path.resolve()),
                "prepared_index": str(base_index.resolve()),
                "supplemental_index": str(supplemental_index.resolve()),
                "start_row_zero_based": sample_start,
                "rows": sample_rows,
                "cap_per_key_per_source": cap,
                "s1_batch_rows": batch_rows,
                "candidate_storage": "SQLite aggregation for one bounded S1 batch; no candidate-pair output/table or feature file",
                "route_semantics": {
                    "P": "country + first 4 compact normalized-name characters",
                    "N": "country + each unique normalized-name token of length >=2",
                    "A": "country + each unique normalized-address token of length >=2",
                    "E": "country + exact normalized name",
                    "P6": "country + first 6 compact normalized-name characters",
                },
                "union_order": list(UNION_ORDER),
            },
            "ground_truth_pairs": {"S2": gt_by_source["S2"], "S3": gt_by_source["S3"], "ALL": gt_total},
            "routes": routes_report,
            "route_aggregation_query_seconds": query_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }
    finally:
        connection.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-row", type=int, default=DEFAULT_SAMPLE_START)
    parser.add_argument("--rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--cap", type=int, default=DEFAULT_MAX_POSTINGS_PER_KEY)
    parser.add_argument("--batch-rows", type=int, default=250)
    parser.add_argument("--base-index", type=Path, default=ROOT / "phase2_index_resume.sqlite3")
    parser.add_argument("--supplemental-index", type=Path, default=ROOT / "phase2_name_routes_train.sqlite3")
    parser.add_argument("--source1", type=Path, default=ROOT / "dataset/train/train_source1.tsv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/route_contribution_sample.json")
    args = parser.parse_args()
    if min(args.rows, args.cap, args.batch_rows) < 1 or args.start_row < 0:
        parser.error("rows, cap, and batch size must be positive; start row nonnegative")
    result = run_audit(
        sample_start=args.start_row, sample_rows=args.rows, cap=args.cap,
        batch_rows=args.batch_rows, base_index=args.base_index,
        supplemental_index=args.supplemental_index, source1_path=args.source1,
        report_path=args.report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
