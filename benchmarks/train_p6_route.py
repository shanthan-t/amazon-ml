"""Build true P6/exact-route sample examples and tune macro-F0.5."""

from __future__ import annotations

import csv
import json
import random
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.blocking import DEFAULT_MAX_POSTINGS_PER_KEY
from src.data_loader import iter_source_chunks
from src.features import FEATURE_NAMES, pair_features
from src.normalization import normalize_business_name, normalize_country
from src.train_model import (
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_SAMPLE_START,
    _entity_is_validation,
    _fit_logistic,
    _read_source1_sample,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_INDEX = ROOT / "phase2_index_resume.sqlite3"
ROUTE_INDEX = ROOT / "phase2_name_routes_train.sqlite3"
SOURCE1 = ROOT / "dataset/train/train_source1.tsv"
SOURCE2 = ROOT / "dataset/train/train_source2.tsv"
SOURCE3 = ROOT / "dataset/train/train_source3.tsv"
CURRENT_MODEL_PATH = ROOT / "models/logistic_baseline.json"
FEATURES_PATH = ROOT / "models/training_sample/p6_exact_route_features.csv"
MODEL_PATH = ROOT / "models/logistic_p6_exact_candidate.json"
REPORT_PATH = ROOT / "reports/p6_exact_route_validation.json"
NEGATIVES_PER_S1 = 10
SEED = 20260925

_ROUTE_CANDIDATES_SQL = """
SELECT q.s1_idx, p.target_idx, p.source, q.block_key
FROM query_keys AS q
CROSS JOIN block_counts AS b
CROSS JOIN postings AS p
WHERE b.block_key = q.block_key
  AND b.frequency <= ?
  AND p.block_key = q.block_key
  AND p.source = b.source
ORDER BY q.s1_idx, p.target_idx, q.block_key
"""


def _reservoir_offer(reservoir, seen, value, rng):
    seen += 1
    if len(reservoir) < NEGATIVES_PER_S1:
        reservoir.append(value)
    else:
        slot = rng.randrange(seen)
        if slot < NEGATIVES_PER_S1:
            reservoir[slot] = value
    return seen


def _route_keys(name: str, country: str) -> tuple[str, ...]:
    country = normalize_country(country)
    name = normalize_business_name(name)
    if not name or not country:
        return ()
    compact = "".join(name.split())
    keys = [f"E|{country}|{name}"]
    if len(compact) >= 6:
        keys.append(f"P6|{country}|{compact[:6]}")
    return tuple(keys)


def _candidate_db(sample, truths, path: str) -> dict[str, object]:
    route_uri = ROUTE_INDEX.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(route_uri, uri=True)
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys (s1_idx INTEGER NOT NULL, block_key TEXT NOT NULL, "
        "PRIMARY KEY(s1_idx, block_key)) WITHOUT ROWID"
    )
    key_rows = []
    for s1_idx, _eid, name, _address, country in sample:
        key_rows.extend((s1_idx, key) for key in _route_keys(name, country))
    connection.executemany("INSERT OR IGNORE INTO query_keys VALUES (?, ?)", key_rows)

    output = sqlite3.connect(path)
    output.execute("PRAGMA journal_mode=OFF")
    output.execute("PRAGMA synchronous=OFF")
    output.execute(
        "CREATE TABLE examples (s1_idx INTEGER NOT NULL, target_idx INTEGER NOT NULL, "
        "label INTEGER NOT NULL, p6_weight REAL NOT NULL, exact_weight REAL NOT NULL, "
        "source INTEGER NOT NULL, PRIMARY KEY(s1_idx,target_idx)) WITHOUT ROWID"
    )
    cursor = iter(connection.execute(_ROUTE_CANDIDATES_SQL, (DEFAULT_MAX_POSTINGS_PER_KEY,)))
    current = next(cursor, None)
    sample_by_idx = {row[0]: row for row in sample}
    stats = {
        "candidate_pairs_p6_plus_exact": 0,
        "candidate_pairs_p6": 0,
        "candidate_pairs_exact": 0,
        "candidate_hits_p6_plus_exact": 0,
        "candidate_hits_p6": 0,
        "candidate_hits_exact": 0,
        "ground_truth_pairs": sum(len(values) for values in truths.values()),
        "negative_examples_p6_plus_exact": 0,
        "negative_examples_exact": 0,
        "positive_examples": 0,
    }
    pending = []
    for s1_idx, entity_id, *_ in sample:
        p6_rng = random.Random(SEED + s1_idx)
        exact_rng = random.Random(SEED ^ s1_idx)
        p6_negative_reservoir = []
        exact_negative_reservoir = []
        p6_negative_count = exact_negative_count = 0
        selected = {}
        while current is not None and current[0] < s1_idx:
            raise RuntimeError("route candidate rows are out of sample order")
        while current is not None and current[0] == s1_idx:
            pair_idx, target_idx, source = current[:3]
            has_p6 = has_exact = False
            while current is not None and current[0] == pair_idx and current[1] == target_idx:
                if current[3].startswith("P6|"):
                    has_p6 = True
                elif current[3].startswith("E|"):
                    has_exact = True
                current = next(cursor, None)
            stats["candidate_pairs_p6_plus_exact"] += 1
            if has_p6:
                stats["candidate_pairs_p6"] += 1
            if has_exact:
                stats["candidate_pairs_exact"] += 1
            label = int(target_idx in truths.get(s1_idx, ()))
            if label:
                stats["candidate_hits_p6_plus_exact"] += 1
                if has_p6:
                    stats["candidate_hits_p6"] += 1
                if has_exact:
                    stats["candidate_hits_exact"] += 1
                selected[target_idx] = [1, 1.0, float(has_exact), source]
            else:
                if has_p6 or has_exact:
                    p6_negative_count = _reservoir_offer(
                        p6_negative_reservoir,
                        p6_negative_count,
                        (target_idx, source),
                        p6_rng,
                    )
                if has_exact:
                    exact_negative_count = _reservoir_offer(
                        exact_negative_reservoir,
                        exact_negative_count,
                        (target_idx, source),
                        exact_rng,
                    )

        if p6_negative_reservoir:
            weight = p6_negative_count / len(p6_negative_reservoir)
            for target_idx, source in p6_negative_reservoir:
                row = selected.setdefault(target_idx, [0, 0.0, 0.0, source])
                row[1] = weight
        if exact_negative_reservoir:
            weight = exact_negative_count / len(exact_negative_reservoir)
            for target_idx, source in exact_negative_reservoir:
                row = selected.setdefault(target_idx, [0, 0.0, 0.0, source])
                row[2] = weight
        stats["negative_examples_p6_plus_exact"] += len(p6_negative_reservoir)
        stats["negative_examples_exact"] += len(exact_negative_reservoir)
        stats["positive_examples"] += sum(row[0] for row in selected.values())
        for target_idx, (label, p6_weight, exact_weight, source) in selected.items():
            pending.append((s1_idx, target_idx, label, p6_weight, exact_weight, source))
        if len(pending) >= 50_000:
            output.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?)", pending)
            output.commit()
            pending.clear()
        if (s1_idx - sample[0][0] + 1) % 10_000 == 0:
            print(f"P6 candidates scanned for {s1_idx + 1 - sample[0][0]:,} sample S1 rows", flush=True)
    if current is not None:
        raise RuntimeError("candidate rows extend beyond the Source-1 sample")
    if pending:
        output.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?)", pending)
        output.commit()
    output.execute("CREATE INDEX examples_by_target ON examples(target_idx,s1_idx)")
    output.commit()
    output.close()
    connection.close()
    return stats


def _write_features(db_path, sample, truth_counts, source2_count):
    s1_data = {
        idx: (name, address, country, int(_entity_is_validation(eid)))
        for idx, eid, name, address, country in sample
    }
    db = sqlite3.connect(db_path)
    cursor = db.execute(
        "SELECT s1_idx,target_idx,label,p6_weight,exact_weight,source "
        "FROM examples ORDER BY target_idx,s1_idx"
    )
    current = cursor.fetchone()
    output_path = FEATURES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "s1_idx", "is_validation", "gt_total", "label", "p6_weight", "exact_weight", "source", *FEATURE_NAMES
    )
    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for source, path, offset in ((2, SOURCE2, 0), (3, SOURCE3, source2_count)):
            target_idx = offset
            for chunk in iter_source_chunks(path, chunk_size=50_000):
                for _target_id, target_name, target_address, target_country in chunk.itertuples(index=False, name=None):
                    while current is not None and current[1] < target_idx:
                        raise RuntimeError(f"target index skipped at {current[1]}")
                    while current is not None and current[1] == target_idx:
                        s1_idx, _, label, p6_weight, exact_weight, candidate_source = current
                        if candidate_source != source:
                            raise RuntimeError("candidate source/target index mismatch")
                        name, address, country, is_validation = s1_data[s1_idx]
                        values = pair_features(name, address, country, target_name, target_address, target_country)
                        writer.writerow((s1_idx, is_validation, truth_counts.get(s1_idx, 0), label, p6_weight, exact_weight, candidate_source, *values))
                        written += 1
                        current = cursor.fetchone()
                    target_idx += 1
            if source == 2 and target_idx != source2_count:
                raise RuntimeError("Source-2 target count differs from prepared index")
    db.close()
    if current is not None:
        raise RuntimeError("some sampled candidates refer to absent target rows")
    return written


def _fit_model(train_frame, feature_names, weights_column):
    x = train_frame[feature_names].to_numpy(dtype=np.float64)
    y = train_frame["label"].to_numpy(dtype=np.float64)
    weights = train_frame[weights_column].to_numpy(dtype=np.float64)
    means = np.average(x, axis=0, weights=weights)
    variances = np.average((x - means) ** 2, axis=0, weights=weights)
    scales = np.sqrt(variances)
    scales[scales < 1e-8] = 1.0
    coefficients, intercept, optimizer = _fit_logistic((x - means) / scales, y, weights, 1.0)
    return {"means": means, "scales": scales, "coefficients": coefficients, "intercept": intercept, "optimizer": optimizer}


def _threshold_report(frame, probabilities, weight_col, validation_ids, gt_counts):
    lookup = {s1: i for i, s1 in enumerate(validation_ids)}
    groups = np.fromiter((lookup[int(idx)] for idx in frame["s1_idx"]), dtype=np.int64, count=len(frame))
    labels = frame["label"].to_numpy(dtype=np.int8)
    weights = frame[weight_col].to_numpy(dtype=np.float64)
    active = weights > 0
    frame = frame.loc[active]
    groups, labels, weights, probabilities = groups[active], labels[active], weights[active], probabilities[active]
    positive = labels == 1
    candidate_hits = int(np.sum(positive))
    total_gt = int(gt_counts.sum())
    neg_n = np.bincount(groups[~positive], weights=weights[~positive], minlength=len(validation_ids))
    neg_k = np.bincount(groups[~positive], minlength=len(validation_ids))
    thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    if len(thresholds) > 301:
        thresholds = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 301)))
    best = None
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = np.bincount(groups[predicted & positive], minlength=len(validation_ids))
        fp = np.bincount(groups[predicted & ~positive], weights=weights[predicted & ~positive], minlength=len(validation_ids))
        precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0)
        recall = np.divide(tp, gt_counts, out=np.zeros(len(tp), dtype=float), where=gt_counts > 0)
        denom = .25 * precision + recall
        entity_f = np.divide(1.25 * precision * recall, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0)
        no_match = gt_counts == 0
        sampled_fp = np.bincount(groups[predicted & ~positive], minlength=len(validation_ids))
        empty_probability = np.ones(len(tp), dtype=float)
        sampled_group = (neg_k > 0) & no_match
        sample_fpr = np.divide(sampled_fp, neg_k, out=np.zeros(len(tp), dtype=float), where=neg_k > 0)
        partial = sampled_group & (neg_k < neg_n)
        empty_probability[partial] = np.power(np.maximum(0.0, 1.0 - sample_fpr[partial]), neg_n[partial])
        complete = sampled_group & (neg_k >= neg_n)
        empty_probability[complete] = (sampled_fp[complete] == 0).astype(float)
        macro = float(np.mean(entity_f + no_match * empty_probability))
        tp_total, fp_total = int(tp.sum()), float(fp.sum())
        precision_micro = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
        recall_micro = tp_total / total_gt if total_gt else 0.0
        f_micro = 1.25 * precision_micro * recall_micro / (.25 * precision_micro + recall_micro) if precision_micro + recall_micro else 0.0
        candidate_pairs = candidate_hits + int(round(float(neg_n.sum())))
        entry = {
            "threshold": float(threshold),
            "macro_f0_5": macro,
            "macro_f0_5_empty_empty_zero_sensitivity": float(np.mean(entity_f)),
            "positive_entity_macro_f0_5": float(np.mean(entity_f[~no_match])),
            "precision": precision_micro,
            "recall_including_blocker_misses": recall_micro,
            "micro_f0_5": f_micro,
            "estimated_true_positives": tp_total,
            "estimated_false_positives": fp_total,
            "estimated_false_negatives": total_gt - tp_total,
            "validation_no_match_s1": int(no_match.sum()),
            "expected_no_match_s1_predicted_nonempty": float(np.sum(no_match * (1.0 - empty_probability))),
            "estimated_predicted_pairs": int(round(tp_total + fp_total)),
            "candidate_pairs": candidate_pairs,
            "candidate_recall": candidate_hits / total_gt if total_gt else 0.0,
        }
        if best is None or macro > best[0] or (macro == best[0] and threshold > best[1]):
            best = (macro, float(threshold), entry)
    assert best
    result = dict(best[2])
    result["threshold_selection"] = "maximize official macro F0.5 across all held-out S1 entities; empty/empty=1; no-match false-positive probability estimated from stratified negatives"
    return result, best[1]


def run():
    started = time.perf_counter()
    sample, _ = _read_source1_sample(str(SOURCE1), DEFAULT_SAMPLE_START, DEFAULT_SAMPLE_ROWS)
    base = sqlite3.connect(f"file:{BASE_INDEX}?mode=ro", uri=True)
    try:
        source2_count = base.execute("SELECT COUNT(*) FROM targets WHERE source=2").fetchone()[0]
        gt_rows = base.execute("SELECT s1_idx,target_idx FROM gt_pairs WHERE s1_idx BETWEEN ? AND ?", (sample[0][0], sample[-1][0]))
        truths = {}
        truth_counts = {}
        for s1_idx, target_idx in gt_rows:
            truths.setdefault(s1_idx, set()).add(target_idx)
            truth_counts[s1_idx] = truth_counts.get(s1_idx, 0) + 1
    finally:
        base.close()
    validation = [(idx, eid) for idx, eid, *_ in sample if _entity_is_validation(eid)]
    validation_ids = [idx for idx, _ in validation]
    gt_counts = np.asarray([truth_counts.get(idx, 0) for idx in validation_ids], dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="p6_route_train_") as temporary:
        db_path = str(Path(temporary) / "examples.sqlite3")
        extraction = _candidate_db(sample, truths, db_path)
        examples_written = _write_features(db_path, sample, truth_counts, source2_count)
    frame = pd.read_csv(FEATURES_PATH)
    feature_names = list(FEATURE_NAMES)
    training = frame["is_validation"] == 0
    heldout = frame["is_validation"] == 1
    p6_train = frame.loc[training & (frame["p6_weight"] > 0)]
    p6_val = frame.loc[heldout & (frame["p6_weight"] > 0)]
    p6_model = _fit_model(p6_train, feature_names, "p6_weight")
    p6_x = p6_val[feature_names].to_numpy(dtype=np.float64)
    p6_probs = expit(((p6_x - p6_model["means"]) / p6_model["scales"]) @ p6_model["coefficients"] + p6_model["intercept"])
    p6_metrics, p6_threshold = _threshold_report(p6_val, p6_probs, "p6_weight", validation_ids, gt_counts)

    current = json.loads(CURRENT_MODEL_PATH.read_text(encoding="utf-8"))
    current_model = {key: np.asarray(current[key], dtype=np.float64) for key in ("means", "scales", "coefficients")}
    current_intercept = float(current["intercept"])
    exact_val = frame.loc[heldout & (frame["exact_weight"] > 0)]
    exact_x = exact_val[feature_names].to_numpy(dtype=np.float64)
    exact_probs = expit(((exact_x - current_model["means"]) / current_model["scales"]) @ current_model["coefficients"] + current_intercept)
    exact_retuned, exact_threshold = _threshold_report(exact_val, exact_probs, "exact_weight", validation_ids, gt_counts)
    exact_current = _threshold_report_at(exact_val, exact_probs, "exact_weight", validation_ids, gt_counts, float(current["threshold"]))

    artifact = {
        "model_type": "weighted_logistic_regression",
        "feature_names": feature_names,
        "means": p6_model["means"].tolist(),
        "scales": p6_model["scales"].tolist(),
        "coefficients": p6_model["coefficients"].tolist(),
        "intercept": p6_model["intercept"],
        "threshold": p6_threshold,
        "training": {
            "candidate_route": "country-aware P6 plus exact normalized name; cap 500 per key/source",
            "sample_start_row_zero_based": DEFAULT_SAMPLE_START,
            "sample_rows": DEFAULT_SAMPLE_ROWS,
            "split": "entity-id blake2b hash modulo 5 equals zero",
            "negative_sampling": f"up to {NEGATIVES_PER_S1} uniform negatives per S1, stratified by route; inverse-probability weights",
            "l2_regularization": 1.0,
        },
    }
    MODEL_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    report = {
        "official_metric": "macro F0.5 per S1 entity, averaged across all S1s",
        "empty_set_rule": "As specified in attached evaluation screenshot: empty ground truth and empty prediction scores 1; a prediction for a no-match entity scores 0.",
        "sample": {
            "source1_rows": DEFAULT_SAMPLE_ROWS,
            "validation_s1_entities": len(validation_ids),
            "validation_ground_truth_pairs": int(gt_counts.sum()),
            "feature_rows": examples_written,
            "negative_examples_per_s1_cap": NEGATIVES_PER_S1,
        },
        "candidate_generation": extraction,
        "baseline_current_exact_route_at_current_threshold": exact_current,
        "baseline_exact_route_retuned_macro_f0_5": exact_retuned,
        "baseline_retuned_threshold": exact_threshold,
        "p6_plus_exact_route_model": p6_metrics,
        "p6_threshold": p6_threshold,
        "p6_train_examples": len(p6_train),
        "p6_validation_examples": len(p6_val),
        "p6_optimizer": p6_model["optimizer"],
        "candidate_model": str(MODEL_PATH.resolve()),
        "features_csv": str(FEATURES_PATH.resolve()),
        "runtime_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _threshold_report_at(frame, probabilities, weight_col, validation_ids, gt_counts, threshold):
    result, _ = _threshold_report_fixed(frame, probabilities, weight_col, validation_ids, gt_counts, threshold)
    return result


def _threshold_report_fixed(frame, probabilities, weight_col, validation_ids, gt_counts, threshold):
    lookup = {s1: i for i, s1 in enumerate(validation_ids)}
    groups = np.fromiter((lookup[int(idx)] for idx in frame["s1_idx"]), dtype=np.int64, count=len(frame))
    labels = frame["label"].to_numpy(dtype=np.int8)
    weights = frame[weight_col].to_numpy(dtype=np.float64)
    active = weights > 0
    groups, labels, weights, probabilities = groups[active], labels[active], weights[active], probabilities[active]
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = np.bincount(groups[predicted & positive], minlength=len(validation_ids))
    fp = np.bincount(groups[predicted & ~positive], weights=weights[predicted & ~positive], minlength=len(validation_ids))
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0)
    recall = np.divide(tp, gt_counts, out=np.zeros(len(tp), dtype=float), where=gt_counts > 0)
    den = .25 * precision + recall
    entity_f = np.divide(1.25 * precision * recall, den, out=np.zeros(len(tp), dtype=float), where=den > 0)
    no_match = gt_counts == 0
    sampled_fp = np.bincount(groups[predicted & ~positive], minlength=len(validation_ids))
    neg_n = np.bincount(groups[~positive], weights=weights[~positive], minlength=len(validation_ids))
    neg_k = np.bincount(groups[~positive], minlength=len(validation_ids))
    empty_probability = np.ones(len(tp), dtype=float)
    active_no_match = no_match & (neg_k > 0)
    sample_fpr = np.divide(sampled_fp, neg_k, out=np.zeros(len(tp), dtype=float), where=neg_k > 0)
    partial = active_no_match & (neg_k < neg_n)
    empty_probability[partial] = np.power(np.maximum(0.0, 1.0 - sample_fpr[partial]), neg_n[partial])
    complete = active_no_match & (neg_k >= neg_n)
    empty_probability[complete] = (sampled_fp[complete] == 0).astype(float)
    total_tp, total_fp = int(tp.sum()), float(fp.sum())
    total_gt = int(gt_counts.sum())
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / total_gt if total_gt else 0.0
    return {
        "threshold": float(threshold),
        "macro_f0_5": float(np.mean(entity_f + no_match * empty_probability)),
        "macro_f0_5_empty_empty_zero_sensitivity": float(np.mean(entity_f)),
        "precision": micro_p,
        "recall_including_blocker_misses": micro_r,
        "micro_f0_5": 1.25 * micro_p * micro_r / (.25 * micro_p + micro_r) if micro_p + micro_r else 0.0,
        "estimated_true_positives": total_tp,
        "estimated_false_positives": total_fp,
        "estimated_false_negatives": total_gt - total_tp,
        "validation_no_match_s1": int(no_match.sum()),
        "expected_no_match_s1_predicted_nonempty": float(np.sum(no_match * (1.0 - empty_probability))),
        "estimated_predicted_pairs": int(round(total_tp + total_fp)),
        "candidate_pairs": int(np.sum(labels == 1) + round(float(np.bincount(groups[~positive], weights=weights[~positive], minlength=len(validation_ids)).sum()))),
        "candidate_recall": float(np.sum(labels == 1) / total_gt) if total_gt else 0.0,
    }, float(threshold)


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
