"""Range-checked, resumable-shard merge with validation before atomic publication."""
import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from .config import CONFIG_ID, MODEL_SHA256
from .validate import validate


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def merge(run_dir, source1):
    run_dir, source1 = Path(run_dir).resolve(), Path(source1).resolve()
    partition_path = run_dir / "shards" / "partition_manifest.json"
    if not partition_path.exists():
        raise FileNotFoundError(f"Missing partition manifest: {partition_path}")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    if partition["configuration_id"] != CONFIG_ID or partition["source_sha256"] != sha(source1):
        raise ValueError("Source or frozen configuration does not match this run")
    parts = sorted(partition["workers"], key=lambda item: item["source_row_start"])
    cursor = 0
    manifests = []
    for part in parts:
        if part["source_row_start"] != cursor or part["source_row_end_exclusive"] - part["source_row_start"] != part["source_rows"]:
            raise ValueError("Shard ranges overlap or contain a gap")
        cursor = part["source_row_end_exclusive"]
        worker_dir = run_dir / "workers" / f"worker_{part['worker']:04d}"
        worker_manifest_path = worker_dir / "worker_manifest.json"
        if not worker_manifest_path.exists():
            raise FileNotFoundError(f"Worker shard is incomplete: {worker_dir}")
        m = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
        if (not m.get("complete") or m.get("configuration_id") != CONFIG_ID
                or m.get("worker") != part["worker"] or m.get("source_rows_completed") != part["source_rows"]
                or m.get("source_row_start") != part["source_row_start"]
                or m.get("source_row_end_exclusive") != part["source_row_end_exclusive"]
                or m.get("source_sha256") != part["source_sha256"]
                or m.get("model_sha256") != MODEL_SHA256):
            raise ValueError(f"Worker manifest does not match partition: {worker_dir}")
        for filename, hash_key in (("candidate_pairs.tsv", "candidate_file_sha256"),
                                   ("matching_results.tsv", "matching_file_sha256")):
            if sha(worker_dir / filename) != m[hash_key]:
                raise ValueError(f"Shard output checksum mismatch: {worker_dir / filename}")
        manifests.append((part, m))
    if cursor != partition["source_rows"] or len(manifests) != partition["worker_count"]:
        raise ValueError("Missing shard ranges or worker manifests")

    stage = run_dir / ("merge_staging_" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        for filename in ("matching_results.tsv", "candidate_pairs.tsv"):
            dest = stage / filename
            with dest.open("xb") as out:
                expected = (b"source1_entity_id\tmatched_entity_ids\n" if filename.startswith("matching")
                            else b"source1_entity_id\tcandidate_entity_ids\n")
                out.write(expected)
                for part, _ in manifests:
                    source = run_dir / "workers" / f"worker_{part['worker']:04d}" / filename
                    with source.open("rb") as inp:
                        header = inp.readline()
                        if header != expected:
                            raise ValueError(f"Invalid shard output header: {source}")
                        shutil.copyfileobj(inp, out, 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
        candidate_total = sum(m["candidate_pairs"] for _, m in manifests)
        match_total = sum(m["predicted_matches"] for _, m in manifests)
        checked = validate(source1, stage / "candidate_pairs.tsv", stage / "matching_results.tsv",
                           candidate_total, match_total)
        output_dir = run_dir / "final"
        output_dir.mkdir(exist_ok=True)
        for filename in ("candidate_pairs.tsv", "matching_results.tsv"):
            os.replace(stage / filename, output_dir / filename)
        result = {"configuration_id": CONFIG_ID, "source1": str(source1),
                  "worker_count": len(manifests), "source_entities": checked["source_rows"],
                  "candidate_pairs": checked["candidate_pairs"], "predicted_matches": checked["matching_pairs"],
                  "valid": True,
                  "candidate_pairs_sha256": sha(output_dir / "candidate_pairs.tsv"),
                  "matching_results_sha256": sha(output_dir / "matching_results.tsv"),
                  "worker_manifest_sha256": [sha(run_dir / "workers" / f"worker_{p['worker']:04d}" / "worker_manifest.json")
                                              for p, _ in manifests],
                  "all_predictions_subset_of_scored_candidates": True}
        tmp_manifest = output_dir / "inference_manifest.json.tmp"
        tmp_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_manifest, output_dir / "inference_manifest.json")
        print(json.dumps(result, indent=2))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source1", required=True)
    args = parser.parse_args()
    merge(args.run_dir, args.source1)


if __name__ == "__main__":
    main()
