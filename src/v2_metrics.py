"""Validation helpers for the V2 candidate matcher."""

from __future__ import annotations

import numpy as np


def _f05(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denominator = 0.25 * precision + recall
    return np.divide(
        1.25 * precision * recall,
        denominator,
        out=np.zeros_like(precision, dtype=np.float64),
        where=denominator != 0,
    )


def tune_official_macro_f05(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    s1_positions: np.ndarray,
    validation_positions: list[int],
    ground_truth_counts: dict[int, int],
) -> tuple[float, dict]:
    """Tune over every validation S1, correcting Bernoulli-sampled negatives.

    Positive candidate pairs are retained exhaustively. Negative candidates are
    sampled at rate 1/10 and carry inverse-probability weight 10. For a true
    singleton, exp(-estimated FP count) estimates the probability of an empty
    prediction under a Poisson approximation.
    """
    if not validation_positions:
        raise ValueError("validation_positions must include held-out S1 rows")

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    s1_positions = np.asarray(s1_positions, dtype=np.int64)
    if not (len(probabilities) == len(labels) == len(weights) == len(s1_positions)):
        raise ValueError("validation arrays must have equal lengths")

    position_to_row = {position: row for row, position in enumerate(validation_positions)}
    entity_rows = np.fromiter(
        (position_to_row[int(position)] for position in s1_positions),
        dtype=np.int64,
        count=len(s1_positions),
    )
    gt = np.fromiter(
        (ground_truth_counts.get(position, 0) for position in validation_positions),
        dtype=np.float64,
        count=len(validation_positions),
    )

    if len(probabilities):
        thresholds = np.unique(np.quantile(probabilities, np.linspace(0, 1, 501)))
    else:
        thresholds = np.asarray([1.0])

    best_threshold = float(thresholds[0])
    best_macro = -1.0
    best_summary = {}
    positive = labels
    negative = ~labels

    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = np.bincount(
            entity_rows,
            weights=(predicted & positive).astype(np.float64),
            minlength=len(validation_positions),
        )
        fp = np.bincount(
            entity_rows,
            weights=(predicted & negative).astype(np.float64) * weights,
            minlength=len(validation_positions),
        )
        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp),
            where=(tp + fp) != 0,
        )
        recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt != 0)
        scores = _f05(precision, recall)
        singleton = gt == 0
        scores[singleton] = np.exp(-fp[singleton])
        macro = float(np.mean(scores))

        if macro > best_macro or (macro == best_macro and threshold > best_threshold):
            best_macro = macro
            best_threshold = float(threshold)
            best_summary = {
                "estimated_true_positives": float(tp.sum()),
                "estimated_false_positives": float(fp.sum()),
                "estimated_false_negatives": float(np.maximum(gt - tp, 0).sum()),
                "validation_singletons": int(singleton.sum()),
                "estimated_singleton_empty_prediction_rate": float(
                    np.exp(-fp[singleton]).mean() if singleton.any() else 0.0
                ),
            }

    return best_threshold, {
        "threshold": best_threshold,
        "macro_f0_5": best_macro,
        "validation_entities": len(validation_positions),
        "validation_entities_with_candidate_examples": int(
            len(np.unique(entity_rows))
        ),
        "validation_entities_without_candidate_examples": int(
            len(validation_positions) - len(np.unique(entity_rows))
        ),
        "threshold_selection": (
            "official per-S1 macro F0.5 over every held-out Source-1 row; "
            "all positive candidates retained; sampled negative false positives "
            "inverse-probability weighted; singleton empty credit estimated "
            "with exp(-weighted false positives)"
        ),
        **best_summary,
    }
