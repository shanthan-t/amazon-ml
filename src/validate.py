"""Streaming output-contract validator shared by merge and standalone validation."""
import csv
import json
import re
from pathlib import Path

from .config import MAX_MATCHES

ID_PATTERN = re.compile(r"^S[23]-.+$")


def validate(source1, candidates_path, matches_path, expected_candidates=None, expected_matches=None):
    source1, candidates_path, matches_path = map(Path, (source1, candidates_path, matches_path))
    source_count = candidate_count = match_count = 0
    with source1.open("r", encoding="utf-8", newline="") as sf, \
         candidates_path.open("r", encoding="utf-8", newline="") as cf, \
         matches_path.open("r", encoding="utf-8", newline="") as mf:
        sources = csv.DictReader(sf, delimiter="\t", strict=True)
        candidates = csv.DictReader(cf, delimiter="\t", strict=True)
        matches = csv.DictReader(mf, delimiter="\t", strict=True)
        if sources.fieldnames != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError("Invalid source1 TSV header")
        if candidates.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Invalid candidate_pairs.tsv header")
        if matches.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Invalid matching_results.tsv header")
        source_iter, cand_iter, match_iter = iter(sources), iter(candidates), iter(matches)
        while True:
            try:
                source = next(source_iter)
            except StopIteration:
                source = None
            try:
                cand = next(cand_iter)
            except StopIteration:
                cand = None
            try:
                match = next(match_iter)
            except StopIteration:
                match = None
            if source is None or cand is None or match is None:
                if source is not None or cand is not None or match is not None:
                    raise ValueError("Source, candidate, and matching row counts differ")
                break
            eid = source["entity_id"]
            if not eid or cand["source1_entity_id"] != eid or match["source1_entity_id"] != eid:
                raise ValueError(f"S1 missing, duplicated, or out of canonical order near row {source_count+1}")
            candidate_ids = cand["candidate_entity_ids"].split(",") if cand["candidate_entity_ids"] else []
            match_ids = match["matched_entity_ids"].split(",") if match["matched_entity_ids"] else []
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"Duplicate candidate ID for {eid}")
            if len(match_ids) != len(set(match_ids)):
                raise ValueError(f"Duplicate match ID for {eid}")
            if len(match_ids) > MAX_MATCHES:
                raise ValueError(f"Match cap exceeded for {eid}: {len(match_ids)}")
            if any(not ID_PATTERN.fullmatch(x) for x in candidate_ids + match_ids):
                raise ValueError(f"Invalid S2/S3 target ID for {eid}")
            if not set(match_ids).issubset(candidate_ids):
                raise ValueError(f"Predicted match not present in scored candidates for {eid}")
            source_count += 1
            candidate_count += len(candidate_ids)
            match_count += len(match_ids)
    if expected_candidates is not None and candidate_count != expected_candidates:
        raise ValueError(f"Candidate count mismatch: {candidate_count} != {expected_candidates}")
    if expected_matches is not None and match_count != expected_matches:
        raise ValueError(f"Match count mismatch: {match_count} != {expected_matches}")
    return {"valid": True, "source_rows": source_count, "candidate_pairs": candidate_count,
            "matching_pairs": match_count, "matching_file": str(matches_path),
            "candidate_file": str(candidates_path)}


def validate_from_manifest(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = validate(manifest["source1"], run_dir / "candidate_pairs.tsv",
                      run_dir / "matching_results.tsv", manifest["candidate_pairs"],
                      manifest["predicted_matches"])
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    validate_from_manifest(args.run_dir)
