"""Bounded deterministic scaling benchmark using the exact current inference worker."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .runner import atomic, partition, sha
from .config import ROOT


def sample_source(source, sample_path, sample_rows):
    source, sample_path = Path(source), Path(sample_path)
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        count = sum(1 for _ in reader)
    if count < sample_rows:
        sample_rows = count
    positions = set(min(count - 1, int((i + .5) * count / sample_rows)) for i in range(sample_rows))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as inp, sample_path.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(inp, delimiter="\t")
        writer = csv.DictWriter(out, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for pos, row in enumerate(reader):
            if pos in positions:
                writer.writerow(row)
    return sample_rows


def benchmark(source1, output, worker_counts, sample_rows=4096, budget_minutes=12,
              threads_per_worker=1):
    started = time.monotonic()
    deadline = started + budget_minutes * 60
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact_manifest = json.loads((ROOT / "WINDOWS_ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    expected = next(item for item in artifact_manifest["artifacts"]
                    if item["logical_name"] == "competition_test_source1")
    if Path(source1).stat().st_size != expected["size_bytes"] or sha(source1) != expected["sha256"]:
        raise ValueError("test_source1.tsv size/SHA-256 differs from the frozen test input")
    sample = output / "sample_source1.tsv"
    sample_count = sample_source(source1, sample, sample_rows)
    results = []
    for count in worker_counts:
        remaining = deadline - time.monotonic()
        if remaining < 15:
            break
        run_dir = output / f"workers_{count}"
        _, run_dir, parts = partition(sample, run_dir, count, resume=False, verify_test_source=False)
        worker_root = run_dir / "workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        procs, logs = [], []
        baseline_cpu = psutil.cpu_times()
        baseline_wall = time.monotonic()
        min_available = psutil.virtual_memory().available
        system_cpu = []
        for part in parts["workers"]:
            i = part["worker"]
            shard = run_dir / "shards" / f"worker_{i:04d}"
            out_dir = worker_root / f"worker_{i:04d}"
            out_dir.mkdir(parents=True)
            log = (run_dir / f"worker_{i:04d}.log").open("w", encoding="utf-8")
            command = [sys.executable, "-m", "windows_inference.worker", "--source",
                       str(shard / "source1.tsv"), "--output", str(out_dir),
                       "--threads", str(threads_per_worker)]
            p = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            procs.append(p)
            logs.append(log)
        timed_out = False
        while any(p.poll() is None for p in procs):
            min_available = min(min_available, psutil.virtual_memory().available)
            system_cpu.append(psutil.cpu_percent(interval=None))
            if time.monotonic() >= deadline:
                timed_out = True
                for p in procs:
                    if p.poll() is None:
                        p.terminate()
                break
            time.sleep(1)
        codes = []
        for p, log in zip(procs, logs):
            try:
                codes.append(p.wait(timeout=30))
            except subprocess.TimeoutExpired:
                p.kill()
                codes.append(p.wait())
            log.close()
        wall = time.monotonic() - baseline_wall
        manifests = []
        for i, code in enumerate(codes):
            path = worker_root / f"worker_{i:04d}" / "worker_manifest.json"
            if code == 0 and path.exists():
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
        results.append({"workers": count, "entities": sum(m["source_rows_completed"] for m in manifests),
                        "candidates": sum(m["candidate_pairs"] for m in manifests), "wall_seconds": wall,
                        "entities_per_second": sum(m["source_rows_completed"] for m in manifests) / max(wall, .001),
                        "candidates_per_second": sum(m["candidate_pairs"] for m in manifests) / max(wall, .001),
                        "worker_peak_rss_mib": [m["peak_rss_mib"] for m in manifests],
                        "min_available_ram_gib": min_available / 1024**3,
                        "mean_sampled_cpu_percent": sum(system_cpu) / len(system_cpu) if system_cpu else 0,
                        "worker_exit_codes": codes, "timed_out": timed_out,
                        "safe_complete": len(manifests) == count and all(c == 0 for c in codes)})
        print(json.dumps(results[-1], indent=2), flush=True)
        if timed_out or len(manifests) != count:
            break
    payload = {"source1": str(Path(source1).resolve()), "sample_rows": sample_count,
               "budget_minutes": budget_minutes, "configuration_id": "v6-top25-xgb39-icu-global98-20260927-v1",
               "runs": results}
    atomic(output / "benchmark_results.json", payload)
    print(f"Results: {output / 'benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", default="2,4,8,12,16")
    parser.add_argument("--sample-rows", type=int, default=4096)
    parser.add_argument("--budget-minutes", type=float, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    counts = [int(x) for x in args.workers.split(",") if x.strip()]
    benchmark(args.source1, args.output, counts, args.sample_rows,
              args.budget_minutes, args.threads_per_worker)


if __name__ == "__main__":
    main()
