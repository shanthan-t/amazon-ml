"""Stream test candidates through the validated logistic baseline."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sqlite3
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from rapidfuzz import fuzz

from src.blocking import DEFAULT_MAX_POSTINGS_PER_KEY, SOURCE1_BATCH_SIZE
from src.data_loader import iter_source_chunks
from src.features import (
    FEATURE_NAMES,
    pair_features_exact_name,
    pair_features_normalized,
)
from src.normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


_CANDIDATES_SQL = """
SELECT q.s1_idx, p.target_idx, p.source, t.entity_id,
       t.normalized_address, t.normalized_country
FROM query_keys AS q
CROSS JOIN main.block_counts AS b
CROSS JOIN main.postings AS p
JOIN main.targets AS t ON t.target_idx = p.target_idx
WHERE b.block_key = q.block_key
  AND b.frequency <= ?
  AND p.block_key = q.block_key
  AND p.source = b.source
ORDER BY q.s1_idx, p.target_idx
"""

_CANDIDATES_WITH_P6_SQL = """
SELECT c.s1_idx,c.target_idx,c.source,t.entity_id,t.normalized_address,
       t.normalized_country,n.normalized_name
FROM (
    SELECT q.s1_idx,p.target_idx,p.source
    FROM query_keys AS q
    CROSS JOIN main.block_counts AS b
    CROSS JOIN main.postings AS p
    WHERE b.block_key=q.block_key AND b.frequency<=?
      AND p.block_key=q.block_key AND p.source=b.source
      AND q.block_key LIKE 'E|%'
    UNION ALL
    SELECT q.s1_idx,p.target_idx,p.source
    FROM query_keys AS q
    CROSS JOIN p6.block_counts AS b
    CROSS JOIN p6.postings AS p
    WHERE b.block_key=q.block_key AND b.frequency<=?
      AND p.block_key=q.block_key AND p.source=b.source
      AND q.block_key LIKE 'P6|%'
) AS c
JOIN main.targets AS t ON t.target_idx=c.target_idx
JOIN names.target_names AS n ON n.target_idx=c.target_idx
ORDER BY c.s1_idx,c.target_idx
"""


def _load_model(path: str) -> dict[str, object]:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("model feature schema does not match the current implementation")
    return model


def predict_test(
    *,
    source1: str,
    index_path: str,
    model_path: str,
    matching_output: str,
    candidate_output: str,
    report_path: str,
    cap: int = DEFAULT_MAX_POSTINGS_PER_KEY,
    limit_s1_rows: int | None = None,
    p6_index: str | None = None,
    name_index: str | None = None,
    minimum_name_similarity: float = 0.0,
) -> dict[str, object]:
    started = time.perf_counter()
    model = _load_model(model_path)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    intercept = float(model["intercept"])
    threshold = float(model["threshold"])

    uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    if p6_index:
        if not name_index:
            connection.close()
            raise ValueError("--name-index is required together with --p6-index")
        connection.execute(
            "ATTACH DATABASE ? AS p6",
            (Path(p6_index).resolve().as_uri() + "?mode=ro",),
        )
        connection.execute(
            "ATTACH DATABASE ? AS names",
            (Path(name_index).resolve().as_uri() + "?mode=ro",),
        )
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys ("
        "s1_idx INTEGER NOT NULL, block_key TEXT NOT NULL, "
        "PRIMARY KEY (s1_idx, block_key)) WITHOUT ROWID"
    )

    matching_path = Path(matching_output)
    candidate_path = Path(candidate_output)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    s1_rows = candidate_pairs = raw_blocker_pairs = predicted_pairs = zero_candidate = zero_match = 0
    source_candidates = {"S2": 0, "S3": 0}
    source_predictions = {"S2": 0, "S3": 0}
    batch: list[tuple[int, str, str, str, str]] = []
    next_s1_idx = 0

    def flush_batch(records, matching_writer, candidate_writer):
        nonlocal candidate_pairs, raw_blocker_pairs, predicted_pairs, zero_candidate, zero_match
        if not records:
            return
        query_rows = []
        for s1_idx, _entity_id, name, _address, country in records:
            normalized_country = normalize_country(country)
            normalized_name = normalize_business_name(name)
            if normalized_country and normalized_name:
                query_rows.append((s1_idx, f"E|{normalized_country}|{normalized_name}"))
                if p6_index:
                    compact = "".join(normalized_name.split())
                    if len(compact) >= 6:
                        query_rows.append((s1_idx, f"P6|{normalized_country}|{compact[:6]}"))
        if query_rows:
            connection.executemany(
                "INSERT INTO query_keys VALUES (?, ?)", query_rows
            )
        candidate_sql = _CANDIDATES_WITH_P6_SQL if p6_index else _CANDIDATES_SQL
        parameters = (cap, cap) if p6_index else (cap,)
        candidates = iter(connection.execute(candidate_sql, parameters))
        current = next(candidates, None)
        flat_features: list[tuple[float, ...]] = []
        flat_sources: list[int] = []
        rows = []
        for s1_idx, entity_id, name, address, country in records:
            normalized_name = normalize_business_name(name)
            normalized_address = normalize_business_address(address)
            normalized_country = normalize_country(country)
            candidate_ids: list[str] = []
            start = len(flat_features)
            if current is not None and current[0] < s1_idx:
                raise RuntimeError("candidate rows are out of Source-1 order")
            while current is not None and current[0] == s1_idx:
                _s1_idx, target_idx, source, target_id, target_address, target_country = current[:6]
                target_name = current[6] if p6_index else normalized_name
                while current is not None and current[0] == s1_idx and current[1] == target_idx:
                    current = next(candidates, None)
                raw_blocker_pairs += 1
                name_similarity = fuzz.ratio(normalized_name, target_name) / 100.0
                if name_similarity < minimum_name_similarity:
                    continue
                candidate_ids.append(target_id)
                if p6_index:
                    flat_features.append(
                        pair_features_normalized(
                            normalized_name,
                            normalized_address,
                            normalized_country,
                            target_name,
                            target_address,
                            target_country,
                            name_similarity=name_similarity,
                        )
                    )
                else:
                    flat_features.append(
                        pair_features_exact_name(
                            normalized_name,
                            normalized_address,
                            normalized_country,
                            target_address,
                            target_country,
                        )
                    )
                flat_sources.append(source)
            end = len(flat_features)
            rows.append((entity_id, candidate_ids, start, end))
            if not candidate_ids:
                zero_candidate += 1
        if current is not None:
            raise RuntimeError("candidate rows exceed the current S1 batch")

        if flat_features:
            matrix = np.asarray(flat_features, dtype=np.float64)
            probabilities = expit(((matrix - means) / scales) @ coefficients + intercept)
        else:
            probabilities = np.empty(0, dtype=np.float64)
        for entity_id, candidate_ids, start, end in rows:
            selected_ids = [
                candidate_ids[offset]
                for offset, probability in enumerate(probabilities[start:end])
                if probability >= threshold
            ]
            candidate_writer.writerow((entity_id, ",".join(candidate_ids)))
            matching_writer.writerow((entity_id, ",".join(selected_ids)))
            candidate_pairs += len(candidate_ids)
            predicted_pairs += len(selected_ids)
            if not selected_ids:
                zero_match += 1
        for source, label in ((2, "S2"), (3, "S3")):
            source_candidates[label] += sum(
                int(value == source) for value in flat_sources
            )
            source_predictions[label] += sum(
                int(value == source and probabilities[position] >= threshold)
                for position, value in enumerate(flat_sources)
            )
        connection.execute("DELETE FROM query_keys")
        connection.commit()

    try:
        with matching_path.open("w", encoding="utf-8", newline="") as matching_file, candidate_path.open(
            "w", encoding="utf-8", newline=""
        ) as candidate_file:
            matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            matching_writer.writerow(("source1_entity_id", "matched_entity_ids"))
            candidate_writer.writerow(("source1_entity_id", "candidate_entity_ids"))
            for chunk in iter_source_chunks(source1, chunk_size=50_000):
                for entity_id, name, address, country in chunk.itertuples(
                    index=False, name=None
                ):
                    batch.append((next_s1_idx, entity_id, name, address, country))
                    next_s1_idx += 1
                    if len(batch) >= SOURCE1_BATCH_SIZE:
                        flush_batch(batch, matching_writer, candidate_writer)
                        s1_rows += len(batch)
                        batch.clear()
                        if s1_rows and s1_rows % 100_000 == 0:
                            print(f"Scored {s1_rows:,} test S1 rows", flush=True)
                    if limit_s1_rows is not None and next_s1_idx >= limit_s1_rows:
                        break
                if limit_s1_rows is not None and next_s1_idx >= limit_s1_rows:
                    break
            if batch:
                flush_batch(batch, matching_writer, candidate_writer)
                s1_rows += len(batch)
                batch.clear()
    finally:
        connection.close()

    report = {
        "configuration": {
            "index": str(Path(index_path).resolve()),
            "model": str(Path(model_path).resolve()),
            "route": (
                "country-aware P6 plus exact normalized name"
                if p6_index
                else "country + exact normalized name"
            ),
            "p6_index": str(Path(p6_index).resolve()) if p6_index else None,
            "name_index": str(Path(name_index).resolve()) if name_index else None,
            "minimum_name_character_similarity": minimum_name_similarity,
            "route_description": (
                "country-aware P6 plus exact normalized name"
                if p6_index
                else "country + exact normalized name"
            ),
            "max_postings_per_key_per_source": cap,
            "threshold": threshold,
            "candidate_pairs_materialized_in_memory": False,
            "s1_batch_size": SOURCE1_BATCH_SIZE,
            "limited_s1_rows": limit_s1_rows,
        },
        "rows": {
            "source1_scored": s1_rows,
            "candidate_pairs": candidate_pairs,
            "raw_blocker_pairs_before_similarity_prefilter": raw_blocker_pairs,
            "predicted_match_pairs": predicted_pairs,
            "zero_candidate_s1": zero_candidate,
            "zero_prediction_s1": zero_match,
            "candidate_pairs_by_source": source_candidates,
            "predicted_pairs_by_source": source_predictions,
        },
        "timing_seconds": time.perf_counter() - started,
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "matching_output": str(matching_path.resolve()),
        "candidate_output": str(candidate_path.resolve()),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--index", default="phase2_test_exact_index.sqlite3")
    parser.add_argument("--model", default="models/logistic_baseline.json")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--report", default="reports/test_prediction.json")
    parser.add_argument("--cap", type=int, default=DEFAULT_MAX_POSTINGS_PER_KEY)
    parser.add_argument("--limit-s1-rows", type=int)
    parser.add_argument("--p6-index")
    parser.add_argument("--name-index")
    parser.add_argument("--minimum-name-similarity", type=float, default=0.0)
    args = parser.parse_args()
    if args.cap < 1 or (args.limit_s1_rows is not None and args.limit_s1_rows < 1):
        parser.error("cap and row limit must be positive")
    report = predict_test(
        source1=args.source1,
        index_path=args.index,
        model_path=args.model,
        matching_output=args.matching,
        candidate_output=args.candidate,
        report_path=args.report,
        cap=args.cap,
        limit_s1_rows=args.limit_s1_rows,
        p6_index=args.p6_index,
        name_index=args.name_index,
        minimum_name_similarity=args.minimum_name_similarity,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
