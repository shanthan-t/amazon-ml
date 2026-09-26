"""Verify the supplemental scorer against the fixed-sample route benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmarks.benchmark_scoring import (
    _ground_truth_denominators,
    _load_sample,
    _run_streamed_workers,
    _summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--prepared-index", default="phase2_index_resume.sqlite3")
    parser.add_argument("--supplemental-index", default="phase2_name_routes_train.sqlite3")
    parser.add_argument("--sample-start", type=int, default=400_000)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--benchmark-report", default="reports/blocking_routes_sample.json")
    parser.add_argument("--report", default="reports/blocking_routes_verification.json")
    args = parser.parse_args()

    rows = _load_sample(args.source1, args.sample_start, args.sample_rows)
    denominators = _ground_truth_denominators(
        args.prepared_index, args.sample_start, args.sample_rows
    )
    started = time.perf_counter()
    baseline = _summary(
        _run_streamed_workers(
            args.prepared_index, rows, args.workers, args.cap
        ),
        denominators,
    )
    enhanced = _summary(
        _run_streamed_workers(
            args.prepared_index,
            rows,
            args.workers,
            args.cap,
            args.supplemental_index,
        ),
        denominators,
    )

    benchmark = json.loads(Path(args.benchmark_report).read_text(encoding="utf-8"))
    expected_baseline = benchmark["baseline"]
    expected_enhanced = benchmark["supplemental_routes"]
    if baseline["candidate_pairs"]["ALL"] != expected_baseline["candidate_pairs"]:
        raise AssertionError("baseline candidate count differs from route benchmark")
    if baseline["true_match_hits"]["ALL"] != expected_baseline["candidate_hits"]:
        raise AssertionError("baseline true-match hits differ from route benchmark")
    if enhanced["candidate_pairs"]["ALL"] != expected_enhanced["candidate_pairs_after_union"]:
        raise AssertionError("supplemental candidate count differs from route benchmark")
    for source in ("S2", "S3"):
        expected_hits = expected_enhanced["candidate_recall_after_union"]["by_source"][source][
            "candidate_hits"
        ]
        if enhanced["true_match_hits"][source] != expected_hits:
            raise AssertionError(f"supplemental {source} hits differ from route benchmark")

    report = {
        "sample_rows": args.sample_rows,
        "workers": args.workers,
        "cap": args.cap,
        "baseline": baseline,
        "supplemental": enhanced,
        "exactly_matches_route_benchmark": True,
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
