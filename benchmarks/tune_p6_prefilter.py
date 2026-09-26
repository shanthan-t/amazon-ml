"""Tune a cheap name-similarity prefilter on the P6 held-out sample."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from benchmarks.train_p6_route import FEATURES_PATH, _threshold_report
from src.features import FEATURE_NAMES
from src.train_model import (
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_SAMPLE_START,
    _entity_is_validation,
    _read_source1_sample,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/logistic_p6_exact_candidate.json"
REPORT_PATH = ROOT / "reports/p6_prefilter_validation.json"
BASE_INDEX = ROOT / "phase2_index_resume.sqlite3"
SOURCE1_PATH = ROOT / "dataset/train/train_source1.tsv"


def _validation_entities():
    sample, _ = _read_source1_sample(
        str(SOURCE1_PATH), DEFAULT_SAMPLE_START, DEFAULT_SAMPLE_ROWS
    )
    ids = [idx for idx, entity_id, *_ in sample if _entity_is_validation(entity_id)]
    connection = sqlite3.connect(f"file:{BASE_INDEX}?mode=ro", uri=True)
    try:
        counts = dict(
            connection.execute(
                "SELECT s1_idx,COUNT(*) FROM gt_pairs WHERE s1_idx BETWEEN ? AND ? GROUP BY s1_idx",
                (DEFAULT_SAMPLE_START, DEFAULT_SAMPLE_START + DEFAULT_SAMPLE_ROWS - 1),
            )
        )
    finally:
        connection.close()
    return ids, np.asarray([counts.get(idx, 0) for idx in ids], dtype=np.int64)


def main():
    frame = pd.read_csv(FEATURES_PATH)
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    validation_ids, gt_counts = _validation_entities()
    val = frame[(frame["is_validation"] == 1) & (frame["p6_weight"] > 0)].copy()
    x = val[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    probabilities = expit(
        ((x - np.asarray(model["means"])) / np.asarray(model["scales"]))
        @ np.asarray(model["coefficients"])
        + float(model["intercept"])
    )
    results = []
    for minimum_similarity in (0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        selected = val["name_char_similarity"].to_numpy() >= minimum_similarity
        filtered = val.loc[selected]
        metrics, threshold = _threshold_report(
            filtered,
            probabilities[selected],
            "p6_weight",
            validation_ids,
            gt_counts,
        )
        metrics["minimum_name_char_similarity"] = minimum_similarity
        metrics["tuned_threshold"] = threshold
        results.append(metrics)
    report = {
        "prefilter": "RapidFuzz normalized name character ratio >= threshold, applied within P6+exact candidates before full feature calculation",
        "held_out_threshold_results": results,
        "selection": "max official macro F0.5; tie-break favors fewer estimated candidate pairs",
    }
    best = max(results, key=lambda row: (row["macro_f0_5"], -row["candidate_pairs"]))
    report["best_prefilter"] = best
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
