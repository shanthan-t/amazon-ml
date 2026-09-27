"""Live aggregate S1 progress and ETA from durable shard checkpoints."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path


def show(run_dir, refresh=5):
    run_dir = Path(run_dir)
    partition = json.loads((run_dir / "shards" / "partition_manifest.json").read_text(encoding="utf-8"))
    total = partition["source_rows"]
    worker_count = partition["worker_count"]
    width = 32
    try:
        while True:
            completed = candidates = matches = 0
            max_elapsed = 0.0
            states = []
            all_done = True
            for i, part in enumerate(partition["workers"]):
                out = run_dir / "workers" / f"worker_{i:04d}"
                manifest_path = out / "worker_manifest.json"
                checkpoint_path = out / "checkpoint.json"
                state = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
                if state is None:
                    state = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
                count = int(state.get("source_rows_completed", state.get("entities_completed", 0)))
                done = bool(state.get("complete", False))
                completed += count
                candidates += int(state.get("candidate_pairs", 0))
                matches += int(state.get("predicted_matches", 0))
                max_elapsed = max(max_elapsed, float(state.get("wall_seconds", 0)))
                all_done = all_done and done
                states.append(f"W{i}:{count:,}/{part['source_rows']:,}{'✓' if done else ''}")
            fraction = completed / total if total else 1.0
            speed = completed / max_elapsed if max_elapsed else 0.0
            seconds_left = max(0, total - completed) / speed if speed else None
            cells = int(width * fraction)
            bar = "#" * cells + "-" * (width - cells)
            if all_done:
                eta = "complete"
            elif seconds_left is None:
                eta = "ETA pending checkpoints"
            else:
                finish = dt.datetime.now().astimezone() + dt.timedelta(seconds=seconds_left)
                eta = f"ETA {seconds_left/3600:.1f}h (~{finish:%Y-%m-%d %H:%M %Z})"
            line = (f"[{bar}] {fraction*100:6.2f}%  {completed:,}/{total:,} S1  "
                    f"{speed:,.1f} S1/s  {eta}  |  {candidates:,} candidates  "
                    f"{matches:,} matches  |  {' '.join(states)}")
            print("\r" + line[:300].ljust(300), end="", flush=True)
            if all_done:
                print()
                return
            time.sleep(max(1, refresh))
    except KeyboardInterrupt:
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--refresh", type=int, default=5)
    args = parser.parse_args()
    show(args.run_dir, args.refresh)


if __name__ == "__main__":
    main()
