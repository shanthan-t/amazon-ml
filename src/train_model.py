"""Train and entity-disjointly validate a weighted logistic matcher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.blocking import (
    DEFAULT_MAX_POSTINGS_PER_KEY,
    DEFAULT_PREFIX_LENGTH,
    _CANDIDATE_ROWS_WITH_SUPPLEMENTAL_SQL,
    build_block_keys,
)
from src.data_loader import iter_source_chunks
from src.features import FEATURE_NAMES, pair_features
from src.normalization import normalize_country


DEFAULT_INDEX = "phase2_index_resume.sqlite3"
DEFAULT_SUPPLEMENTAL_INDEX = "phase2_name_routes_train.sqlite3"
DEFAULT_SAMPLE_START = 400_000
DEFAULT_SAMPLE_ROWS = 50_000
NEGATIVE_SAMPLE_MODULUS = 100
EXPECTED_SAMPLE_CANDIDATES = 37_119_275
EXPECTED_SAMPLE_HITS = 154_399


def _entity_is_validation(entity_id: str) -> bool:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 5 == 0


def _negative_sample(s1_idx: int, target_idx: int) -> bool:
    mask = (1 << 64) - 1
    value = (s1_idx * 0x9E3779B185EBCA87 + target_idx * 0xC2B2AE3D27D4EB4F)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    return value % NEGATIVE_SAMPLE_MODULUS == 0


def _read_source1_sample(
    path: str, start: int, rows: int
) -> tuple[list[tuple[int, str, str, str, str]], int]:
    end = start + rows
    result: list[tuple[int, str, str, str, str]] = []
    row_index = 0
    for chunk in iter_source_chunks(path, chunk_size=100_000):
        chunk_end = row_index + len(chunk)
        if chunk_end > start and row_index < end:
            left = max(0, start - row_index)
            right = min(len(chunk), end - row_index)
            for offset, values in enumerate(
                chunk.iloc[left:right].itertuples(index=False, name=None), start=left
            ):
                entity_id, name, address, country = values
                result.append(
                    (row_index + offset, entity_id, name, address, country)
                )
        row_index = chunk_end
        if row_index >= end:
            break
    if len(result) != rows:
        raise ValueError(f"sample contains {len(result)} rows; expected {rows}")
    return result, row_index


def _readonly_connection(index_path: str, supplemental_path: str) -> sqlite3.Connection:
    base_uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(base_uri, uri=True)
    supplemental_uri = Path(supplemental_path).resolve().as_uri() + "?mode=ro"
    connection.execute("ATTACH DATABASE ? AS supplemental", (supplemental_uri,))
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys ("
        "s1_idx INTEGER NOT NULL, block_key TEXT NOT NULL, "
        "PRIMARY KEY (s1_idx, block_key)) WITHOUT ROWID"
    )
    return connection


def _candidate_examples(
    connection: sqlite3.Connection,
    sample: list[tuple[int, str, str, str, str]],
    *,
    candidate_db: str,
    max_postings_per_key: int,
) -> dict[str, int]:
    first_idx, last_idx = sample[0][0], sample[-1][0]
    s1_records = {
        row[0]: {
            "entity_id": row[1],
            "name": row[2],
            "address": row[3],
            "country": row[4],
            "validation": _entity_is_validation(row[1]),
        }
        for row in sample
    }
    query_rows: list[tuple[int, str]] = []
    for s1_idx, _entity_id, name, address, country in sample:
        normalized_country = normalize_country(country)
        query_rows.extend(
            (s1_idx, key)
            for key in build_block_keys(
                name, address, normalized_country, prefix_length=DEFAULT_PREFIX_LENGTH
            )
        )
    connection.executemany(
        "INSERT OR IGNORE INTO query_keys(s1_idx, block_key) VALUES (?, ?)",
        query_rows,
    )

    truth: dict[int, set[int]] = {}
    truth_counts: dict[int, int] = {}
    for s1_idx, target_idx in connection.execute(
        "SELECT s1_idx, target_idx FROM gt_pairs "
        "WHERE s1_idx BETWEEN ? AND ? ORDER BY s1_idx, target_idx",
        (first_idx, last_idx),
    ):
        truth.setdefault(s1_idx, set()).add(target_idx)
        truth_counts[s1_idx] = truth_counts.get(s1_idx, 0) + 1

    output = sqlite3.connect(candidate_db)
    output.execute("PRAGMA journal_mode=OFF")
    output.execute("PRAGMA synchronous=OFF")
    output.execute("PRAGMA temp_store=FILE")
    output.execute(
        "CREATE TABLE examples ("
        "s1_idx INTEGER NOT NULL, target_idx INTEGER NOT NULL, "
        "label INTEGER NOT NULL, weight REAL NOT NULL, source INTEGER NOT NULL, "
        "PRIMARY KEY (s1_idx, target_idx)) WITHOUT ROWID"
    )
    candidate_count = positive_count = selected_negatives = 0
    batch: list[tuple[int, int, int, float, int]] = []
    candidates = iter(
        connection.execute(
            _CANDIDATE_ROWS_WITH_SUPPLEMENTAL_SQL,
            (max_postings_per_key, max_postings_per_key),
        )
    )
    candidate = next(candidates, None)
    try:
        for s1_idx in s1_records:
            seen_targets: set[int] = set()
            while candidate is not None and candidate[0] < s1_idx:
                raise RuntimeError("candidate rows are out of sample order")
            while candidate is not None and candidate[0] == s1_idx:
                _, target_idx, source = candidate
                candidate = next(candidates, None)
                if target_idx in seen_targets:
                    continue
                seen_targets.add(target_idx)
                candidate_count += 1
                label = int(target_idx in truth.get(s1_idx, ()))
                if label:
                    positive_count += 1
                if label or _negative_sample(s1_idx, target_idx):
                    if not label:
                        selected_negatives += 1
                    batch.append(
                        (
                            s1_idx,
                            target_idx,
                            label,
                            1.0 if label else float(NEGATIVE_SAMPLE_MODULUS),
                            source,
                        )
                    )
                if len(batch) >= 50_000:
                    output.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?)", batch)
                    output.commit()
                    batch.clear()
            if s1_idx % 10_000 == 0:
                print(f"Deduplicated candidates for S1 row {s1_idx:,}", flush=True)
        if candidate is not None:
            raise RuntimeError("candidate rows extend beyond the sample")
        if batch:
            output.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?)", batch)
            output.commit()
        output.execute("CREATE INDEX examples_by_target ON examples(target_idx, s1_idx)")
        output.commit()
    finally:
        output.close()
        connection.execute("DELETE FROM query_keys")
        connection.commit()

    return {
        "candidate_pairs": candidate_count,
        "candidate_positive_pairs": positive_count,
        "ground_truth_pairs": sum(truth_counts.values()),
        "selected_negative_examples": selected_negatives,
        "sampled_examples": positive_count + selected_negatives,
    }


def _write_feature_examples(
    *,
    candidate_db: str,
    source2_path: str,
    source3_path: str,
    source2_count: int,
    s1_records: list[tuple[int, str, str, str, str]],
    truth_counts: dict[int, int],
    output_path: Path,
) -> int:
    s1_values = {
        s1_idx: (name, address, country, _entity_is_validation(entity_id))
        for s1_idx, entity_id, name, address, country in s1_records
    }
    connection = sqlite3.connect(candidate_db)
    cursor = connection.execute(
        "SELECT s1_idx, target_idx, label, weight, source "
        "FROM examples ORDER BY target_idx, s1_idx"
    )
    current = cursor.fetchone()
    written = 0
    header = ("s1_idx", "is_validation", "gt_total", "source", "label", "weight", *FEATURE_NAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            target_idx = 0
            for source, path, source_offset in (
                (2, source2_path, 0),
                (3, source3_path, source2_count),
            ):
                target_idx = source_offset
                for chunk in iter_source_chunks(path, chunk_size=50_000):
                    for _target_id, target_name, target_address, target_country in chunk.itertuples(
                        index=False, name=None
                    ):
                        while current is not None and current[1] < target_idx:
                            raise RuntimeError("sample example target index was skipped")
                        while current is not None and current[1] == target_idx:
                            s1_idx, _, label, weight, candidate_source = current
                            if candidate_source != source:
                                raise RuntimeError("candidate source and target offset disagree")
                            name, address, country, is_validation = s1_values[s1_idx]
                            features = pair_features(
                                name,
                                address,
                                country,
                                target_name,
                                target_address,
                                target_country,
                            )
                            writer.writerow(
                                (
                                    s1_idx,
                                    int(is_validation),
                                    truth_counts.get(s1_idx, 0),
                                    candidate_source,
                                    label,
                                    weight,
                                    *features,
                                )
                            )
                            written += 1
                            current = cursor.fetchone()
                        target_idx += 1
                if source == 2 and target_idx != source2_count:
                    raise RuntimeError(
                        f"Source-2 count {target_idx} does not match index count {source2_count}"
                    )
            if current is not None:
                raise RuntimeError("candidate examples refer to missing target rows")
    finally:
        connection.close()
    return written


def _fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float) -> tuple[np.ndarray, float, dict[str, object]]:
    total_weight = float(weights.sum())

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:-1]
        intercept = parameters[-1]
        logits = x @ coefficients + intercept
        loss = np.dot(weights, np.logaddexp(0.0, logits) - y * logits) / total_weight
        loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
        errors = weights * (expit(logits) - y) / total_weight
        gradient = np.empty_like(parameters)
        gradient[:-1] = x.T @ errors + l2 * coefficients
        gradient[-1] = errors.sum()
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"logistic optimizer did not converge: {result.message}")
    return result.x[:-1], float(result.x[-1]), {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
    }


def _f_score(precision: float, recall: float, beta: float = 0.5) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def _evaluate_thresholds(
    val_frame: pd.DataFrame,
    probabilities: np.ndarray,
    validation_entities: dict[int, int],
) -> tuple[dict[str, object], float]:
    indices = val_frame["s1_idx"].to_numpy(dtype=np.int64)
    labels = val_frame["label"].to_numpy(dtype=np.int8)
    weights = val_frame["weight"].to_numpy(dtype=np.float64)
    totals = val_frame["gt_total"].to_numpy(dtype=np.int32)
    threshold_values = np.unique(
        np.concatenate(([0.0], probabilities, [1.0]))
    )
    if len(threshold_values) > 300:
        threshold_values = np.unique(
            np.quantile(probabilities, np.linspace(0.0, 1.0, 301))
        )
    best_micro: tuple[float, float, dict[str, object]] | None = None
    best_macro: tuple[float, float, dict[str, object]] | None = None
    candidate_rows: list[dict[str, object]] = []
    for threshold in threshold_values:
        predicted = probabilities >= threshold
        tp_mask = predicted & (labels == 1)
        fp_mask = predicted & (labels == 0)
        tp = int(tp_mask.sum())
        fp = float(weights[fp_mask].sum())
        gt_total = sum(validation_entities.values())
        fn = gt_total - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_total if gt_total else 0.0
        micro_f = _f_score(precision, recall)

        entity_tp: dict[int, int] = {}
        entity_fp: dict[int, float] = {}
        for row_index in np.flatnonzero(tp_mask):
            entity = int(indices[row_index])
            entity_tp[entity] = entity_tp.get(entity, 0) + 1
        for row_index in np.flatnonzero(fp_mask):
            entity = int(indices[row_index])
            entity_fp[entity] = entity_fp.get(entity, 0.0) + float(weights[row_index])
        entity_scores = []
        positive_entity_scores = []
        no_match_empty_probabilities = []
        for entity, positives in validation_entities.items():
            entity_true = entity_tp.get(entity, 0)
            entity_false = entity_fp.get(entity, 0.0)
            if positives:
                entity_precision = (
                    entity_true / (entity_true + entity_false)
                    if entity_true + entity_false
                    else 0.0
                )
                entity_recall = entity_true / positives
                entity_score = _f_score(entity_precision, entity_recall)
                positive_entity_scores.append(entity_score)
            else:
                # Inverse-probability weighted sampled negatives estimate the
                # number of false predictions. A Poisson zero-event estimate
                # approximates P(no false merge), the singleton's expected score.
                entity_score = float(np.exp(-entity_false))
                no_match_empty_probabilities.append(entity_score)
            entity_scores.append(entity_score)
        macro_f = float(np.mean(entity_scores)) if entity_scores else 0.0
        no_match_entities = [entity for entity, count in validation_entities.items() if count == 0]
        predicted_no_match = sum(entity_fp.get(entity, 0.0) > 0 for entity in no_match_entities)
        no_match_fpr = predicted_no_match / len(no_match_entities) if no_match_entities else 0.0
        entry = {
            "threshold": float(threshold),
            "candidate_precision": precision,
            "overall_recall_including_blocker_misses": recall,
            "micro_f0_5": micro_f,
            "macro_f0_5": macro_f,
            "macro_positive_entity_f0_5": float(np.mean(positive_entity_scores)) if positive_entity_scores else 0.0,
            "mean_estimated_singleton_empty_score": float(np.mean(no_match_empty_probabilities)) if no_match_empty_probabilities else 0.0,
            "estimated_true_positives": tp,
            "estimated_false_positives": fp,
            "estimated_false_positives_per_validation_s1": fp / len(validation_entities) if validation_entities else 0.0,
            "false_negatives_including_blocker_misses": fn,
            "false_negatives_per_validation_s1": fn / len(validation_entities) if validation_entities else 0.0,
            "zero_match_entity_false_positive_rate": no_match_fpr,
            "estimated_empty_prediction_s1": sum(
                no_match_empty_probabilities
            ) + sum(
                1 for entity, positives in validation_entities.items()
                if positives > 0 and entity_tp.get(entity, 0) + entity_fp.get(entity, 0.0) == 0
            ),
            "predicted_match_pairs": int(round(tp + fp)),
        }
        candidate_rows.append(entry)
        if best_micro is None or micro_f > best_micro[0] or (
            micro_f == best_micro[0] and threshold > best_micro[1]
        ):
            best_micro = (micro_f, float(threshold), entry)
        if best_macro is None or macro_f > best_macro[0] or (
            macro_f == best_macro[0] and threshold > best_macro[1]
        ):
            best_macro = (macro_f, float(threshold), entry)

    assert best_micro is not None and best_macro is not None
    threshold_report = dict(best_macro[2])
    threshold_report["threshold_selection"] = (
        "maximum estimated official macro F0.5 across all held-out S1 entities, "
        "including singleton empty/empty credit; singleton empty probability uses "
        "a Poisson estimate from inverse-probability weighted sampled negatives; "
        "tie-break toward higher threshold"
    )
    threshold_report["maximum_micro_f0_5_diagnostic"] = dict(best_micro[2])

    multi_entities = {
        entity: count for entity, count in validation_entities.items() if count >= 2
    }
    selected = probabilities >= best_macro[1]
    multi_tp = sum(
        1
        for row_index in np.flatnonzero(selected & (labels == 1))
        if int(indices[row_index]) in multi_entities
    )
    multi_gt = sum(multi_entities.values())
    threshold_report["multi_match_behavior"] = {
        "validation_s1_with_two_or_more_matches": len(multi_entities),
        "validation_ground_truth_pairs_for_multi_match_s1": multi_gt,
        "predicted_true_pairs_for_multi_match_s1": multi_tp,
        "multi_match_pair_recall_including_blocker_misses": multi_tp / multi_gt if multi_gt else 0.0,
    }
    threshold_report["threshold_grid_points"] = len(candidate_rows)
    return threshold_report, best_micro[1]


def train_sample(
    *,
    source1: str,
    source2: str,
    source3: str,
    prepared_index: str,
    supplemental_index: str,
    sample_start: int,
    sample_rows: int,
    max_postings_per_key: int,
    examples_csv: str,
    model_path: str,
    report_path: str,
    reuse_examples: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    sample, _ = _read_source1_sample(source1, sample_start, sample_rows)
    connection = _readonly_connection(prepared_index, supplemental_index)
    try:
        source2_count = connection.execute(
            "SELECT COUNT(*) FROM targets WHERE source=2"
        ).fetchone()[0]
        truth_counts = dict(
            connection.execute(
                "SELECT s1_idx, COUNT(*) FROM gt_pairs "
                "WHERE s1_idx BETWEEN ? AND ? GROUP BY s1_idx",
                (sample[0][0], sample[-1][0]),
            )
        )
        validation_entities = {
            s1_idx: truth_counts.get(s1_idx, 0)
            for s1_idx, entity_id, *_ in sample
            if _entity_is_validation(entity_id)
        }
        if reuse_examples:
            previous_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            candidate_stats = previous_report["candidate_universe"]
        else:
            with tempfile.TemporaryDirectory(prefix="matcher_sample_") as temporary:
                candidate_db = str(Path(temporary) / "sample_candidates.sqlite3")
                candidate_stats = _candidate_examples(
                    connection,
                    sample,
                    candidate_db=candidate_db,
                    max_postings_per_key=max_postings_per_key,
                )
                if sample_start == DEFAULT_SAMPLE_START and sample_rows == DEFAULT_SAMPLE_ROWS:
                    if candidate_stats["candidate_pairs"] != EXPECTED_SAMPLE_CANDIDATES:
                        raise RuntimeError(
                            "sample candidate count changed: "
                            f"{candidate_stats['candidate_pairs']:,} != "
                            f"{EXPECTED_SAMPLE_CANDIDATES:,}"
                        )
                    if candidate_stats["candidate_positive_pairs"] != EXPECTED_SAMPLE_HITS:
                        raise RuntimeError(
                            "sample hit count changed: "
                            f"{candidate_stats['candidate_positive_pairs']:,} != "
                            f"{EXPECTED_SAMPLE_HITS:,}"
                        )
                examples_written = _write_feature_examples(
                    candidate_db=candidate_db,
                    source2_path=source2,
                    source3_path=source3,
                    source2_count=source2_count,
                    s1_records=sample,
                    truth_counts=truth_counts,
                    output_path=Path(examples_csv),
                )
    finally:
        connection.close()

    frame = pd.read_csv(examples_csv)
    examples_written = len(frame)
    feature_columns = list(FEATURE_NAMES)
    train_frame = frame[frame["is_validation"] == 0]
    validation_frame = frame[frame["is_validation"] == 1]
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError("entity-disjoint split produced an empty train or validation set")

    raw_train_x = train_frame[feature_columns].to_numpy(dtype=np.float64)
    train_y = train_frame["label"].to_numpy(dtype=np.float64)
    train_weights = train_frame["weight"].to_numpy(dtype=np.float64)
    means = np.average(raw_train_x, axis=0, weights=train_weights)
    variances = np.average((raw_train_x - means) ** 2, axis=0, weights=train_weights)
    scales = np.sqrt(variances)
    scales[scales < 1e-8] = 1.0
    train_x = (raw_train_x - means) / scales
    coefficients, intercept, optimizer = _fit_logistic(
        train_x, train_y, train_weights, l2=1.0
    )

    val_x = validation_frame[feature_columns].to_numpy(dtype=np.float64)
    probabilities = expit(((val_x - means) / scales) @ coefficients + intercept)
    validation_entity_counts = {
        int(s1_idx): int(count)
        for s1_idx, count in validation_entities.items()
    }
    full_validation_report, _ = _evaluate_thresholds(
        validation_frame, probabilities, validation_entity_counts
    )
    exact_route_mask = validation_frame["name_exact"].to_numpy() == 1.0
    exact_validation_report, threshold = _evaluate_thresholds(
        validation_frame[exact_route_mask],
        probabilities[exact_route_mask],
        validation_entity_counts,
    )

    model = {
        "model_type": "weighted_logistic_regression",
        "feature_names": feature_columns,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "threshold": threshold,
        "training": {
            "sample_start_row_zero_based": sample_start,
            "sample_rows": sample_rows,
            "negative_sampling_rate": f"1/{NEGATIVE_SAMPLE_MODULUS}",
            "negative_inverse_probability_weight": NEGATIVE_SAMPLE_MODULUS,
            "validation_split": "entity-id blake2b hash modulo 5 equals zero",
            "l2_regularization": 1.0,
            "deployment_candidate_route": "country-aware exact normalized name, capped at 500 per source",
        },
    }
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    report: dict[str, object] = {
        "sample": {
            "start_row_zero_based": sample_start,
            "source1_rows": sample_rows,
            "train_entities": len(sample) - len(validation_entities),
            "validation_entities": len(validation_entities),
            "validation_positive_ground_truth_entities": sum(
                count > 0 for count in validation_entities.values()
            ),
            "features": feature_columns,
        },
        "candidate_universe": candidate_stats,
        "training_examples": {
            "rows_written": examples_written,
            "training_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "training_positive_rows": int(train_frame["label"].sum()),
            "validation_positive_rows": int(validation_frame["label"].sum()),
            "sampled_negative_rows_have_inverse_probability_weight": NEGATIVE_SAMPLE_MODULUS,
        },
        "optimizer": optimizer,
        "validation": {
            "full_blocker_union": full_validation_report,
            "deployment_exact_name_route_proxy": exact_validation_report,
        },
        "model": str(model_file.resolve()),
        "feature_examples_csv": str(Path(examples_csv).resolve()),
        "runtime_scope": (
            "refit existing feature examples only"
            if reuse_examples
            else "candidate extraction, feature generation, fit, and validation"
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--source2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--source3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--prepared-index", default=DEFAULT_INDEX)
    parser.add_argument("--supplemental-index", default=DEFAULT_SUPPLEMENTAL_INDEX)
    parser.add_argument("--sample-start", type=int, default=DEFAULT_SAMPLE_START)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--cap", type=int, default=DEFAULT_MAX_POSTINGS_PER_KEY)
    parser.add_argument("--examples-csv", default="models/training_sample/features.csv")
    parser.add_argument("--model", default="models/logistic_baseline.json")
    parser.add_argument("--report", default="reports/training_model_validation.json")
    parser.add_argument("--reuse-examples", action="store_true")
    args = parser.parse_args()
    if args.sample_start < 0 or args.sample_rows < 1 or args.cap < 1:
        parser.error("sample start must be nonnegative; sample rows and cap must be positive")
    report = train_sample(
        source1=args.source1,
        source2=args.source2,
        source3=args.source3,
        prepared_index=args.prepared_index,
        supplemental_index=args.supplemental_index,
        sample_start=args.sample_start,
        sample_rows=args.sample_rows,
        max_postings_per_key=args.cap,
        examples_csv=args.examples_csv,
        model_path=args.model,
        report_path=args.report,
        reuse_examples=args.reuse_examples,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
