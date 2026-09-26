"""Compare the current exact-name model with a P6+exact model on held-out S1s."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.features import FEATURE_NAMES
from src.train_model import (
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_SAMPLE_START,
    NEGATIVE_SAMPLE_MODULUS,
    _entity_is_validation,
    _fit_logistic,
    _read_source1_sample,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "models/training_sample/features.csv"
CURRENT_MODEL = ROOT / "models/logistic_baseline.json"
PREPARED_INDEX = ROOT / "phase2_index_resume.sqlite3"
REPORT = ROOT / "reports/training_model_validation.json"
OUTPUT_REPORT = ROOT / "reports/p6_model_comparison.json"
OUTPUT_MODEL = ROOT / "models/logistic_p6_candidate.json"
S1_PATH = ROOT / "dataset/train/train_source1.tsv"


def _validation_entities() -> tuple[list[int], np.ndarray]:
    sample, _ = _read_source1_sample(
        str(S1_PATH), DEFAULT_SAMPLE_START, DEFAULT_SAMPLE_ROWS
    )
    selected = [row for row in sample if _entity_is_validation(row[1])]
    ids = [row[0] for row in selected]
    connection = sqlite3.connect(f"file:{PREPARED_INDEX}?mode=ro", uri=True)
    try:
        counts = dict(
            connection.execute(
                "SELECT s1_idx, COUNT(*) FROM gt_pairs "
                "WHERE s1_idx BETWEEN ? AND ? GROUP BY s1_idx",
                (DEFAULT_SAMPLE_START, DEFAULT_SAMPLE_START + DEFAULT_SAMPLE_ROWS - 1),
            )
        )
    finally:
        connection.close()
    return ids, np.asarray([counts.get(idx, 0) for idx in ids], dtype=np.int64)


def _score_thresholds(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    validation_ids: list[int],
    ground_truth_counts: np.ndarray,
) -> tuple[dict[str, object], float]:
    group_lookup = {entity: position for position, entity in enumerate(validation_ids)}
    row_ids = frame["s1_idx"].to_numpy(dtype=np.int64)
    groups = np.fromiter((group_lookup[idx] for idx in row_ids), dtype=np.int64, count=len(row_ids))
    labels = frame["label"].to_numpy(dtype=np.int8)
    weights = frame["weight"].to_numpy(dtype=np.float64)
    is_positive = labels == 1
    positive_candidates = np.bincount(
        groups[is_positive], minlength=len(validation_ids)
    )
    total_gt = int(ground_truth_counts.sum())
    route_hits = int(positive_candidates.sum())
    thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    if len(thresholds) > 301:
        thresholds = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 301)))

    best: tuple[float, float, dict[str, object]] | None = None
    for threshold in thresholds:
        predicted = probabilities >= threshold
        true_positive = np.bincount(
            groups[predicted & is_positive], minlength=len(validation_ids)
        )
        false_positive = np.bincount(
            groups[predicted & ~is_positive],
            weights=weights[predicted & ~is_positive],
            minlength=len(validation_ids),
        )
        predicted_total = true_positive + false_positive
        precision = np.divide(
            true_positive,
            predicted_total,
            out=np.zeros(len(validation_ids), dtype=np.float64),
            where=predicted_total > 0,
        )
        recall = np.divide(
            true_positive,
            ground_truth_counts,
            out=np.zeros(len(validation_ids), dtype=np.float64),
            where=ground_truth_counts > 0,
        )
        denominator = 0.25 * precision + recall
        per_entity_f = np.divide(
            1.25 * precision * recall,
            denominator,
            out=np.zeros(len(validation_ids), dtype=np.float64),
            where=denominator > 0,
        )
        no_match = ground_truth_counts == 0
        correct_no_match = no_match & (predicted_total == 0)
        macro_empty_is_one = float(np.mean(per_entity_f + correct_no_match))
        macro_empty_is_zero = float(np.mean(per_entity_f))

        total_tp = int(true_positive.sum())
        total_fp = float(false_positive.sum())
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        micro_recall = total_tp / total_gt if total_gt else 0.0
        micro_f = (
            1.25 * micro_precision * micro_recall / (0.25 * micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        entry = {
            "threshold": float(threshold),
            "official_macro_f0_5_empty_empty_one": macro_empty_is_one,
            "macro_f0_5_empty_empty_zero": macro_empty_is_zero,
            "positive_entity_macro_f0_5": float(np.mean(per_entity_f[~no_match])),
            "estimated_pair_precision": micro_precision,
            "pair_recall_including_blocker_misses": micro_recall,
            "micro_f0_5": micro_f,
            "estimated_true_positives": total_tp,
            "estimated_false_positives": total_fp,
            "estimated_false_negatives": total_gt - total_tp,
            "validation_s1_with_zero_ground_truth": int(no_match.sum()),
            "zero_match_s1_predicted_nonempty_estimate": int(
                np.count_nonzero(no_match & (predicted_total > 0))
            ),
            "estimated_total_predicted_pairs": int(round(float(predicted_total.sum()))),
        }
        score = macro_empty_is_one
        if best is None or score > best[0] or (score == best[0] and threshold > best[1]):
            best = (score, float(threshold), entry)

    assert best is not None
    chosen = dict(best[2])
    chosen["threshold_selection"] = (
        "maximum macro F0.5 over all held-out S1s; empty-ground-truth and empty-prediction scores 1; tie-break higher threshold"
    )
    chosen["candidate_recall"] = route_hits / total_gt if total_gt else 0.0
    chosen["candidate_true_pairs"] = route_hits
    chosen["all_ground_truth_pairs"] = total_gt
    chosen["estimated_candidate_pairs"] = int(
        route_hits
        + (len(frame) - int(frame["label"].sum())) * NEGATIVE_SAMPLE_MODULUS
    )
    return chosen, best[1]


def _prepare_model(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=np.float64)
    weights = frame["weight"].to_numpy(dtype=np.float64)
    means = np.average(x, axis=0, weights=weights)
    variance = np.average((x - means) ** 2, axis=0, weights=weights)
    scales = np.sqrt(variance)
    scales[scales < 1e-8] = 1.0
    coefficients, intercept, optimizer = _fit_logistic(
        (x - means) / scales, y, weights, l2=1.0
    )
    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "intercept": intercept,
        "optimizer": optimizer,
    }


def _predict(frame: pd.DataFrame, feature_columns: list[str], model: dict[str, object]) -> np.ndarray:
    matrix = frame[feature_columns].to_numpy(dtype=np.float64)
    return expit(
        ((matrix - model["means"]) / model["scales"]) @ model["coefficients"]
        + model["intercept"]
    )


def compare() -> dict[str, object]:
    started = time.perf_counter()
    frame = pd.read_csv(FEATURES)
    feature_columns = list(FEATURE_NAMES)
    validation_ids, gt_counts = _validation_entities()
    split_train = frame["is_validation"] == 0
    split_val = frame["is_validation"] == 1
    exact_route = frame["name_exact"] == 1.0
    p6_route = (frame["name_prefix6"] == 1.0) | exact_route

    current = json.loads(CURRENT_MODEL.read_text(encoding="utf-8"))
    current_model = {
        "means": np.asarray(current["means"], dtype=np.float64),
        "scales": np.asarray(current["scales"], dtype=np.float64),
        "coefficients": np.asarray(current["coefficients"], dtype=np.float64),
        "intercept": float(current["intercept"]),
    }
    baseline_val = frame.loc[split_val & exact_route]
    baseline_probs = _predict(baseline_val, feature_columns, current_model)
    baseline_metrics, baseline_opt_threshold = _score_thresholds(
        baseline_val, baseline_probs, validation_ids, gt_counts
    )
    current_threshold = float(current["threshold"])
    baseline_current = dict(baseline_metrics)
    baseline_current["threshold"] = current_threshold
    baseline_current = _score_thresholds_at(
        baseline_val, baseline_probs, validation_ids, gt_counts, current_threshold
    )

    p6_train = frame.loc[split_train & p6_route]
    p6_val = frame.loc[split_val & p6_route]
    p6_model = _prepare_model(p6_train, feature_columns)
    p6_probs = _predict(p6_val, feature_columns, p6_model)
    p6_metrics, p6_threshold = _score_thresholds(
        p6_val, p6_probs, validation_ids, gt_counts
    )

    model_artifact = {
        "model_type": "weighted_logistic_regression",
        "feature_names": feature_columns,
        "means": p6_model["means"].tolist(),
        "scales": p6_model["scales"].tolist(),
        "coefficients": p6_model["coefficients"].tolist(),
        "intercept": p6_model["intercept"],
        "threshold": p6_threshold,
        "training": {
            "candidate_route": "country-aware P6 prefix union exact name",
            "sample_start_row_zero_based": DEFAULT_SAMPLE_START,
            "sample_rows": DEFAULT_SAMPLE_ROWS,
            "entity_disjoint_split": "entity-id blake2b hash modulo 5 equals zero",
            "negative_sampling_rate": "1/100 per candidate pair, inverse-probability weight 100",
            "l2_regularization": 1.0,
        },
    }
    OUTPUT_MODEL.write_text(json.dumps(model_artifact, indent=2) + "\n", encoding="utf-8")
    result = {
        "official_metric": "macro F0.5 computed per held-out S1 entity, then averaged",
        "empty_set_convention": "Prompt does not specify empty/empty; primary comparison counts correct zero-match predictions as F0.5=1 and also reports empty/empty=0 sensitivity.",
        "sample": {
            "s1_rows": DEFAULT_SAMPLE_ROWS,
            "validation_s1_entities": len(validation_ids),
            "validation_ground_truth_pairs": int(gt_counts.sum()),
            "feature_example_rows": len(frame),
            "training_positive_examples": int(frame.loc[split_train, "label"].sum()),
            "validation_positive_examples": int(frame.loc[split_val, "label"].sum()),
            "negative_sample_weight": NEGATIVE_SAMPLE_MODULUS,
        },
        "baseline_current_exact_name_model_at_existing_threshold": baseline_current,
        "baseline_exact_name_model_retuned_on_official_macro_f0_5": baseline_metrics,
        "baseline_retuned_threshold": baseline_opt_threshold,
        "p6_plus_exact_model": p6_metrics,
        "p6_threshold": p6_threshold,
        "p6_training_rows": len(p6_train),
        "p6_validation_rows": len(p6_val),
        "p6_optimizer": p6_model["optimizer"],
        "model_candidate_path": str(OUTPUT_MODEL.resolve()),
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _score_thresholds_at(frame, probabilities, validation_ids, gt_counts, threshold):
    group_lookup = {entity: position for position, entity in enumerate(validation_ids)}
    groups = np.fromiter(
        (group_lookup[idx] for idx in frame["s1_idx"].to_numpy(dtype=np.int64)),
        dtype=np.int64,
        count=len(frame),
    )
    labels = frame["label"].to_numpy(dtype=np.int8)
    weights = frame["weight"].to_numpy(dtype=np.float64)
    predicted = probabilities >= threshold
    tp = np.bincount(groups[predicted & (labels == 1)], minlength=len(validation_ids))
    fp = np.bincount(
        groups[predicted & (labels == 0)],
        weights=weights[predicted & (labels == 0)],
        minlength=len(validation_ids),
    )
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0)
    recall = np.divide(tp, gt_counts, out=np.zeros(len(tp), dtype=float), where=gt_counts > 0)
    den = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, den, out=np.zeros(len(tp), dtype=float), where=den > 0)
    no_match = gt_counts == 0
    no_match_pred = no_match & (tp + fp > 0)
    total_tp, total_fp = int(tp.sum()), float(fp.sum())
    total_gt = int(gt_counts.sum())
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / total_gt if total_gt else 0.0
    micro_f = 1.25 * micro_p * micro_r / (0.25 * micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "threshold": float(threshold),
        "official_macro_f0_5_empty_empty_one": float(np.mean(f + (no_match & (tp + fp == 0)))),
        "macro_f0_5_empty_empty_zero": float(np.mean(f)),
        "positive_entity_macro_f0_5": float(np.mean(f[~no_match])),
        "estimated_pair_precision": micro_p,
        "pair_recall_including_blocker_misses": micro_r,
        "micro_f0_5": micro_f,
        "estimated_true_positives": total_tp,
        "estimated_false_positives": total_fp,
        "estimated_false_negatives": total_gt - total_tp,
        "validation_s1_with_zero_ground_truth": int(no_match.sum()),
        "zero_match_s1_predicted_nonempty_estimate": int(no_match_pred.sum()),
        "estimated_total_predicted_pairs": int(round(float((tp + fp).sum()))),
        "candidate_true_pairs": int(np.sum(labels == 1)),
        "all_ground_truth_pairs": total_gt,
        "candidate_recall": float(np.sum(labels == 1) / total_gt) if total_gt else 0.0,
        "estimated_candidate_pairs": int(
            np.sum(labels == 1)
            + int(np.sum(labels == 0)) * NEGATIVE_SAMPLE_MODULUS
        ),
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
