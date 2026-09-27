"""Windows-safe shard manager. Workers are independent subprocesses and open their own read-only indexes."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import CONFIG_ID, MODEL_PATHS, MODEL_SHA256


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def partition(source, run_dir, workers, resume, verify_test_source=True):
    source, run_dir = Path(source).resolve(), Path(run_dir).resolve()
    source_hash = sha(source)
    source_bytes = source.stat().st_size
    if verify_test_source:
        repository = Path(__file__).resolve().parents[1]
        artifact_manifest = json.loads((repository / "WINDOWS_ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
        expected_source = next(item for item in artifact_manifest["artifacts"]
                               if item["logical_name"] == "competition_test_source1")
        if source_hash != expected_source["sha256"] or source_bytes != expected_source["size_bytes"]:
            raise ValueError("test_source1.tsv size/SHA-256 differs from the frozen test input")
    shard_root = run_dir / "shards"
    manifest_path = shard_root / "partition_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Run already has shards; use --resume: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_sha256"] != sha(source) or manifest["source_rows"] < 1:
            raise ValueError("Resume source does not match the frozen partition manifest")
        if manifest["worker_count"] != workers:
            raise ValueError("Resume worker count differs from the existing partition")
        return source, run_dir, manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        # No workers start until the manifest is atomically committed, so an
        # interrupted input-only partition is safe to rebuild on explicit resume.
        if resume and shard_root.exists() and not (run_dir / "workers").exists():
            shutil.rmtree(shard_root)
        else:
            raise FileExistsError(f"Run directory is non-empty but has no partition manifest: {run_dir}")
    shard_root.mkdir(parents=True, exist_ok=True)
    counts = [0] * workers
    first = [None] * workers
    last = [None] * workers
    handles, writers = [], []
    with source.open("r", encoding="utf-8", newline="") as inp:
        reader = csv.DictReader(inp, delimiter="\t")
        if reader.fieldnames != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError(f"Unexpected S1 TSV schema: {reader.fieldnames}")
        header = reader.fieldnames
        # First count rows without retaining any data; then shard sequentially in canonical order.
        total = sum(1 for _ in reader)
        if total == 0:
            raise ValueError("S1 input is empty")
        inp.seek(0)
        reader = csv.DictReader(inp, delimiter="\t")
        chunk = math.ceil(total / workers)
        try:
            for i in range(workers):
                directory = shard_root / f"worker_{i:04d}"
                directory.mkdir()
                path = directory / "source1.tsv"
                out = path.open("x", encoding="utf-8", newline="")
                handles.append(out)
                writer = csv.DictWriter(out, fieldnames=header, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writers.append(writer)
            for pos, row in enumerate(reader):
                idx = min(pos // chunk, workers - 1)
                if first[idx] is None:
                    first[idx] = (pos, row["entity_id"])
                last[idx] = (pos, row["entity_id"])
                counts[idx] += 1
                writers[idx].writerow(row)
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in handles:
                handle.close()
    parts = []
    for i in range(workers):
        directory = shard_root / f"worker_{i:04d}"
        source_part = directory / "source1.tsv"
        part = {"worker": i, "source_rows": counts[i], "source_row_start": first[i][0],
                "source_row_end_exclusive": last[i][0] + 1, "first_s1_id": first[i][1],
                "last_s1_id": last[i][1], "source_sha256": sha(source_part)}
        atomic(directory / "source1.json", part)
        parts.append(part)
    if sum(counts) != total:
        raise ValueError("Shard partition lost or duplicated S1 input rows")
    manifest = {"configuration_id": CONFIG_ID, "source": str(source), "source_sha256": source_hash,
                "source_rows": total, "worker_count": workers, "workers": parts}
    atomic(manifest_path, manifest)
    return source, run_dir, manifest


def completed_valid(out_dir, part):
    manifest_path = out_dir / "worker_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (m.get("complete") is True and m.get("configuration_id") == CONFIG_ID
                and m.get("source_rows_completed") == part["source_rows"]
                and m.get("source_row_start") == part["source_row_start"]
                and m.get("source_row_end_exclusive") == part["source_row_end_exclusive"]
                and m.get("source_sha256") == part["source_sha256"]
                and m.get("model_sha256") == MODEL_SHA256
                and m.get("candidate_file_sha256") == sha(out_dir / "candidate_pairs.tsv")
                and m.get("matching_file_sha256") == sha(out_dir / "matching_results.tsv"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run(source, run_dir, workers, threads, batch_size, resume):
    source, run_dir, manifest = partition(source, run_dir, workers, resume)
    for model, expected in zip(MODEL_PATHS, MODEL_SHA256):
        if not model.is_file() or sha(model) != expected:
            raise ValueError(f"Model missing or checksum mismatch: {model}")
    from .config import INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE
    for required in (INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE / "offsets.npy",
                     TARGET_STORE / "records.tsv", TARGET_STORE / "manifest.json"):
        if not required.exists():
            raise FileNotFoundError(f"Required copied artifact is missing: {required}")

    processes = []
    for part in manifest["workers"]:
        i = part["worker"]
        shard = run_dir / "shards" / f"worker_{i:04d}"
        out_dir = run_dir / "workers" / f"worker_{i:04d}"
        if completed_valid(out_dir, part):
            print(f"worker {i}: verified completed shard; skipping", flush=True)
            continue
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        restart = out_dir.exists()
        command = [sys.executable, "-m", "windows_inference.worker", "--source",
                   str(shard / "source1.tsv"), "--output", str(out_dir),
                   "--threads", str(threads), "--batch-size", str(batch_size)]
        if restart:
            command.append("--restart-partial")
        log_path = run_dir / f"worker_{i:04d}.log"
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1])
        processes.append((i, process, log))
        print(f"worker {i}: pid={process.pid}, input rows={part['source_rows']:,}", flush=True)
    failed = []
    last_display = 0.0
    while any(process.poll() is None for _, process, _ in processes):
        now = time.monotonic()
        if now - last_display >= 5:
            done = candidates = matches = elapsed = 0
            state_text = []
            for part in manifest["workers"]:
                i = part["worker"]
                out = run_dir / "workers" / f"worker_{i:04d}"
                mp = out / "worker_manifest.json"
                cp = out / "checkpoint.json"
                state = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else (
                    json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {})
                count = int(state.get("source_rows_completed", state.get("entities_completed", 0)))
                is_done = bool(state.get("complete", False))
                done += count
                candidates += int(state.get("candidate_pairs", 0))
                matches += int(state.get("predicted_matches", 0))
                elapsed = max(elapsed, float(state.get("wall_seconds", 0)))
                state_text.append(f"W{i}:{count:,}/{part['source_rows']:,}{'✓' if is_done else ''}")
            total = manifest["source_rows"]
            speed = done / elapsed if elapsed else 0
            remaining = (total - done) / speed if speed else None
            if remaining is None:
                eta = "ETA pending checkpoints"
            else:
                finish = dt.datetime.now().astimezone() + dt.timedelta(seconds=remaining)
                eta = f"ETA {remaining/3600:.1f}h (~{finish:%Y-%m-%d %H:%M %Z})"
            print(f"Progress {done:,}/{total:,} S1 ({done/total:.1%}), {speed:,.1f} S1/s, "
                  f"{candidates:,} candidates, {matches:,} matches; {eta}; {' | '.join(state_text)}",
                  flush=True)
            last_display = now
        time.sleep(1)
    for i, process, log in processes:
        code = process.returncode
        log.close()
        if code:
            failed.append((i, code))
    if failed:
        raise RuntimeError(f"Worker failure(s): {failed}. Completed shards remain reusable; rerun with --resume.")
    print("All worker shards completed. Run merge.py to validate and assemble outputs.", flush=True)


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True, help="Competition test_source1.tsv")
    parser.add_argument("--run-dir", required=True, help="New or existing output run directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("worker and thread counts must be positive")
    run(args.source1, args.run_dir, args.workers, args.threads_per_worker, args.batch_size, args.resume)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
