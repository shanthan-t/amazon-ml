"""Classify missed ground-truth pairs in the fixed broad-blocker sample."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.blocking import DEFAULT_MAX_POSTINGS_PER_KEY, build_block_keys
from src.train_model import DEFAULT_SAMPLE_ROWS, DEFAULT_SAMPLE_START, _read_source1_sample


ROOT = Path(__file__).resolve().parents[1]

_MISS_CAUSES_SQL = """
WITH truth AS (
    SELECT g.s1_idx,g.target_idx,s.country AS s1_country,t.country AS target_country
    FROM main.gt_pairs g
    JOIN main.s1 s ON s.s1_idx=g.s1_idx
    JOIN main.targets t ON t.target_idx=g.target_idx
    WHERE g.s1_idx BETWEEN ? AND ?
), matches AS (
    SELECT g.s1_idx,g.target_idx,
           MAX(b.frequency<=?) AS within_cap,
           MAX(b.frequency>?) AS over_cap
    FROM truth g
    CROSS JOIN query_keys q
    CROSS JOIN main.block_counts b
    CROSS JOIN main.postings p
    WHERE q.s1_idx=g.s1_idx AND q.route IN ('P','N','A')
      AND b.block_key=q.block_key
      AND p.block_key=q.block_key AND p.source=b.source
      AND p.target_idx=g.target_idx
    GROUP BY g.s1_idx,g.target_idx
    UNION ALL
    SELECT g.s1_idx,g.target_idx,
           MAX(b.frequency<=?) AS within_cap,
           MAX(b.frequency>?) AS over_cap
    FROM truth g
    CROSS JOIN query_keys q
    CROSS JOIN supplemental.block_counts b
    CROSS JOIN supplemental.postings p
    WHERE q.s1_idx=g.s1_idx AND q.route IN ('E','P6')
      AND b.block_key=q.block_key
      AND p.block_key=q.block_key AND p.source=b.source
      AND p.target_idx=g.target_idx
    GROUP BY g.s1_idx,g.target_idx
), status AS (
    SELECT s1_idx,target_idx,MAX(within_cap) within_cap,MAX(over_cap) over_cap
    FROM matches GROUP BY s1_idx,target_idx
)
SELECT CASE
         WHEN COALESCE(m.within_cap,0)=1 THEN 'unexpected_candidate_hit'
         WHEN COALESCE(m.over_cap,0)=1 THEN 'cap_truncation'
         WHEN g.s1_country='' OR g.target_country='' THEN 'missing_country'
         WHEN g.s1_country<>g.target_country THEN 'country_mismatch'
         ELSE 'same_country_no_shared_key'
       END AS cause,
       COUNT(*) AS missed_pairs
FROM truth g
LEFT JOIN status m ON m.s1_idx=g.s1_idx AND m.target_idx=g.target_idx
WHERE COALESCE(m.within_cap,0)=0
GROUP BY cause
"""


def run(start_row: int, rows: int, cap: int, report_path: Path):
    started = time.perf_counter()
    sample, _ = _read_source1_sample(
        str(ROOT / "dataset/train/train_source1.tsv"), start_row, rows
    )
    connection = sqlite3.connect(
        (ROOT / "phase2_index_resume.sqlite3").resolve().as_uri() + "?mode=ro",
        uri=True,
    )
    connection.execute(
        "ATTACH DATABASE ? AS supplemental",
        ((ROOT / "phase2_name_routes_train.sqlite3").resolve().as_uri() + "?mode=ro",),
    )
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys(s1_idx INTEGER NOT NULL,route TEXT NOT NULL,block_key TEXT NOT NULL,"
        "PRIMARY KEY(s1_idx,route,block_key)) WITHOUT ROWID"
    )
    try:
        key_rows = []
        for s1_idx, _entity_id, name, address, country in sample:
            for key in build_block_keys(name, address, country):
                key_rows.append((s1_idx, key.split("|", 1)[0], key))
        connection.executemany("INSERT INTO query_keys VALUES (?,?,?)", key_rows)
        rows_out = connection.execute(
            _MISS_CAUSES_SQL,
            (sample[0][0], sample[-1][0], cap, cap, cap, cap),
        ).fetchall()
        missed = {cause: count for cause, count in rows_out}
        total_gt = connection.execute(
            "SELECT COUNT(*) FROM main.gt_pairs WHERE s1_idx BETWEEN ? AND ?",
            (sample[0][0], sample[-1][0]),
        ).fetchone()[0]
        report = {
            "sample": {"start_row_zero_based": start_row, "s1_rows": rows, "cap_per_key_per_source": cap},
            "ground_truth_pairs": total_gt,
            "missed_pairs": sum(missed.values()),
            "miss_rate": sum(missed.values()) / total_gt if total_gt else 0.0,
            "miss_categories": {
                key: {"pairs": value, "fraction_of_misses": value / sum(missed.values()) if missed else 0.0}
                for key, value in missed.items()
            },
            "definitions": {
                "cap_truncation": "No route retrieved the pair at cap 500, but at least one matching block key exists above cap.",
                "country_mismatch": "No same-key route even without cap and both normalized countries exist but differ.",
                "missing_country": "No same-key route and either normalized country is empty.",
                "same_country_no_shared_key": "No same-key route at any frequency despite equal nonempty normalized countries; requires record-text analysis to separate name/address variation subtypes.",
            },
            "runtime_seconds": time.perf_counter() - started,
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
    parser.add_argument("--report", type=Path, default=ROOT / "reports/blocking_miss_sample.json")
    args = parser.parse_args()
    print(json.dumps(run(args.start_row, args.rows, args.cap, args.report), indent=2))


if __name__ == "__main__":
    main()
