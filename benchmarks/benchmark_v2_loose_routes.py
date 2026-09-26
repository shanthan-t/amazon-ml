"""Measure supplemental phonetic and numeric-address routes without pair export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
import sqlite3
import statistics
import time
from pathlib import Path

import numpy as np

from src.data_loader import iter_ground_truth_chunks
from src.normalization import normalize_business_address, normalize_business_name, normalize_country
from src.v2_train import sample_s1
from src.v2_common import MAX_POSTINGS_PER_KEY, compute_pair_features, generate_blocking_keys
from src.v2_metrics import tune_official_macro_f05


_DIGITS = re.compile(r"\d+")
_SOUNDEX_DIGITS = str.maketrans({
    **{c: "1" for c in "bfpv"},
    **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"},
    "l": "4",
    **{c: "5" for c in "mn"},
    "r": "6",
})
_SOUNDEX_IGNORE = frozenset("aehiouwy")
_NAME_STOPWORDS = frozenset({
    "the", "and", "inc", "llc", "ltd", "corp", "co", "pvt", "limited",
    "incorporated", "corporation", "company", "private", "group", "holdings",
})


def _soundex(token: str) -> str | None:
    token = token.lower()
    if len(token) < 4 or not token.isascii() or not token.isalpha():
        return None
    code = [token[0]]
    previous = token[0].translate(_SOUNDEX_DIGITS)
    for char in token[1:]:
        digit = char.translate(_SOUNDEX_DIGITS)
        if digit != previous and digit not in _SOUNDEX_IGNORE:
            code.append(digit)
        if char not in "hw":
            previous = digit
    return ("".join(code) + "000")[:4].upper()


def _supplemental_keys(name: str, address: str, country: str) -> dict[str, set[str]]:
    if not country:
        return {route: set() for route in ("SX", "AN", "NG3")}
    soundex_keys = {
        f"SX|{country}|{code}"
        for token in set(name.split())
        if token not in _NAME_STOPWORDS
        if (code := _soundex(token)) is not None
    }
    address_keys = {
        f"AN|{country}|{number}"
        for number in _DIGITS.findall(address)
        if len(number) >= 3
    }
    compact_name = "".join(name.split())
    ngram_keys = {
        f"NG3|{country}|{compact_name[index:index + 3]}"
        for index in range(max(0, len(compact_name) - 2))
    }
    return {"SX": soundex_keys, "AN": address_keys, "NG3": ngram_keys}


def _distribution(counts: np.ndarray, ground_truth_counts: np.ndarray) -> dict:
    return {
        "candidate_pairs": int(counts.sum()),
        "average_candidates_per_s1": float(counts.mean()) if len(counts) else 0.0,
        "median_candidates_per_s1": float(statistics.median(counts.tolist())) if len(counts) else 0.0,
        "zero_candidate_s1": int(np.count_nonzero(counts == 0)),
        "candidate_ground_truth_hits": int(ground_truth_counts.sum()),
    }


def _is_validation(entity_id: str) -> bool:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 5 == 0


def _keep_negative(s1_pos: int, target_idx: int) -> bool:
    mask = (1 << 64) - 1
    value = s1_pos * 0x9E3779B185EBCA87 + target_idx * 0xC2B2AE3D27D4EB4F
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    return value % 10 == 0


def _score_sample(model_path: str, features, labels, weights, positions,
                  route_flags, sample, truth, candidate_counts, candidate_hits,
                  sample_start: int) -> dict:
    import xgboost as xgb

    model_meta = json.loads(Path(model_path).read_text(encoding="utf-8"))
    model = xgb.XGBClassifier(n_jobs=1)
    model.load_model(str(Path(model_path).parent / model_meta["xgb_model_file"]))
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.ndim != 2 or feature_array.shape[1] != len(model_meta["feature_names"]):
        raise ValueError("candidate feature schema does not match the loaded model")
    probabilities = model.predict_proba(feature_array)[:, 1] if len(features) else np.asarray([])

    validation_positions = [s1_pos for s1_pos, eid, *_ in sample if _is_validation(eid)]
    validation_gt = {s1_pos: len(truth[s1_pos]) for s1_pos in validation_positions}
    validation_position_set = set(validation_positions)
    validation_mask = np.asarray([
        s1_pos in validation_position_set
        for s1_pos in range(sample_start, sample_start + len(sample))
    ], dtype=bool)
    validation_gt_total = sum(validation_gt.values())
    reports = {}
    for name, flags in route_flags.items():
        route_mask = np.asarray(flags, dtype=bool)
        threshold, metrics = tune_official_macro_f05(
            probabilities[route_mask],
            np.asarray(labels, dtype=np.float64)[route_mask],
            np.asarray(weights, dtype=np.float64)[route_mask],
            np.asarray(positions, dtype=np.int64)[route_mask],
            validation_positions,
            validation_gt,
        )
        tp = metrics["estimated_true_positives"]
        fp = metrics["estimated_false_positives"]
        reports[name] = {
            **metrics,
            "threshold": threshold,
            "estimated_precision": tp / (tp + fp) if tp + fp else 0.0,
            "estimated_recall_including_blocker_misses": (
                tp / validation_gt_total if validation_gt_total else 0.0
            ),
            "validation_candidate_pairs": int(candidate_counts[name][validation_mask].sum()),
            "validation_candidate_recall": (
                int(candidate_hits[name][validation_mask].sum()) / validation_gt_total
                if validation_gt_total else 0.0
            ),
            "validation_zero_candidate_s1": int(
                np.count_nonzero(candidate_counts[name][validation_mask] == 0)
            ),
        }
    return {
        "model": str(Path(model_path).resolve()),
        "model_training_threshold": model_meta.get("threshold"),
        "negative_sampling_rate": 10,
        "validation_split": "BLAKE2b(Source-1 entity ID) modulo 5 equals zero",
        "feature_rows_scored": int(len(features)),
        "results": reports,
    }


def benchmark(
    *, index_path: str, source1_path: str, ground_truth_path: str,
    sample_start: int, sample_rows: int, cap: int, model_path: str | None = None,
) -> dict:
    started = time.perf_counter()
    sample = sample_s1(source1_path, sample_start, sample_rows)
    s1_by_id = {row[1]: row[0] for row in sample}
    s1_values = {}
    base_query = defaultdict(list)
    extra_query = {route: defaultdict(list) for route in ("SX", "AN", "NG3")}
    for s1_pos, _eid, raw_name, raw_address, raw_country in sample:
        name = normalize_business_name(raw_name)
        address = normalize_business_address(raw_address)
        country = normalize_country(raw_country)
        s1_values[s1_pos] = (name, address, country)
        for key in generate_blocking_keys(name, country):
            base_query[key].append(s1_pos)
        for route, keys in _supplemental_keys(name, address, country).items():
            for key in keys:
                extra_query[route][key].append(s1_pos)

    truth = {row[0]: set() for row in sample}
    for chunk in iter_ground_truth_chunks(ground_truth_path, chunk_size=100_000):
        for s1_id, matched_ids in chunk.itertuples(index=False, name=None):
            s1_pos = s1_by_id.get(s1_id)
            if s1_pos is not None and matched_ids:
                truth[s1_pos].update(value.strip() for value in matched_ids.split(",") if value.strip())
    gt_counts = np.asarray([len(truth[pos]) for pos, *_ in sample], dtype=np.int64)
    total_gt = int(gt_counts.sum())
    gt_by_source = {
        "S2": sum(value.startswith("S2-") for pairs in truth.values() for value in pairs),
        "S3": sum(value.startswith("S3-") for pairs in truth.values() for value in pairs),
    }

    db = sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro", uri=True)
    db.execute("PRAGMA cache_size=-65536")
    db.execute("CREATE TEMP TABLE base_query_keys (block_key TEXT PRIMARY KEY) WITHOUT ROWID")
    base_keys = list(base_query)
    for start in range(0, len(base_keys), 50_000):
        db.executemany(
            "INSERT OR IGNORE INTO base_query_keys VALUES (?)",
            ((key,) for key in base_keys[start:start + 50_000]),
        )
    base_frequency = {
        key: frequency
        for key, frequency in db.execute(
            "SELECT q.block_key, COALESCE(k.freq, 0) "
            "FROM base_query_keys q LEFT JOIN key_counts k USING(block_key)"
        )
    }

    target_rows = db.execute(
        "SELECT entity_id, normalized_name, normalized_address, normalized_country, source "
        "FROM targets ORDER BY target_idx"
    )
    extra_frequency = {route: Counter() for route in extra_query}
    scanned = 0
    scan_started = time.perf_counter()
    while rows := target_rows.fetchmany(20_000):
        for _target_id, name, address, country, _source in rows:
            for route, keys in _supplemental_keys(name, address, country).items():
                for key in keys:
                    if key in extra_query[route]:
                        extra_frequency[route][key] += 1
        scanned += len(rows)
        if scanned and scanned % 1_000_000 == 0:
            print(f"Frequency pass scanned {scanned:,} targets", flush=True)
    frequency_seconds = time.perf_counter() - scan_started
    target_rows.close()

    active_extra = {
        route: {key for key, count in extra_frequency[route].items() if count <= cap}
        for route in extra_query
    }
    key_probe_upper_bounds = {
        route: sum(
            extra_frequency[route][key] * len(extra_query[route][key])
            for key in active_extra[route]
        )
        for route in extra_query
    }
    base_probe_upper_bound = sum(
        base_frequency[key] * len(s1_positions)
        for key, s1_positions in base_query.items()
        if 0 < base_frequency.get(key, 0) <= cap
    )

    base_routes = {"E", "P6", "P4", "ST", "W"}
    variants = (
        "base", "base_plus_ngrams", "base_plus_soundex",
        "base_plus_address_numbers", "combined",
    )
    counts = {variant: np.zeros(sample_rows, dtype=np.int64) for variant in variants}
    hits = {variant: np.zeros(sample_rows, dtype=np.int64) for variant in variants}
    source_hits = {variant: {"S2": 0, "S3": 0} for variant in variants}
    standalone = {
        route: {"candidate_pairs": 0, "ground_truth_hits": 0}
        for route in extra_query
    }
    validation_features = []
    validation_labels = []
    validation_weights = []
    validation_positions = []
    validation_route_flags = {variant: [] for variant in variants}
    max_target_route_probes = 150_000_000
    observed_probe_upper_bound = base_probe_upper_bound + sum(key_probe_upper_bounds.values())
    candidate_seconds = 0.0

    if observed_probe_upper_bound <= max_target_route_probes:
        target_rows = db.execute(
            "SELECT target_idx, entity_id, normalized_name, normalized_address, normalized_country, source "
            "FROM targets ORDER BY target_idx"
        )
        route_started = time.perf_counter()
        scanned = 0
        while rows := target_rows.fetchmany(20_000):
            for target_idx, target_id, name, address, country, source in rows:
                route_matches = defaultdict(set)
                for key in generate_blocking_keys(name, country):
                    if base_frequency.get(key, cap + 1) <= cap:
                        for s1_pos in base_query.get(key, ()):
                            route_matches[s1_pos].add(key.split("|", 1)[0])
                for route, keys in _supplemental_keys(name, address, country).items():
                    for key in keys & active_extra[route]:
                        for s1_pos in extra_query[route].get(key, ()):
                            route_matches[s1_pos].add(route)

                for s1_pos, routes in route_matches.items():
                    offset = s1_pos - sample_start
                    base_hit = bool(routes & base_routes)
                    sx_hit = "SX" in routes
                    an_hit = "AN" in routes
                    selected = {
                        "base": base_hit,
                        "base_plus_ngrams": base_hit or "NG3" in routes,
                        "base_plus_soundex": base_hit or sx_hit,
                        "base_plus_address_numbers": base_hit or an_hit,
                        "combined": base_hit or sx_hit or an_hit,
                    }
                    truth_hit = target_id in truth[s1_pos]
                    for variant, present in selected.items():
                        if present:
                            counts[variant][offset] += 1
                            if truth_hit:
                                hits[variant][offset] += 1
                                source_hits[variant]["S2" if source == 2 else "S3"] += 1
                    for route in extra_query:
                        if route in routes:
                            standalone[route]["candidate_pairs"] += 1
                            standalone[route]["ground_truth_hits"] += int(truth_hit)
                    if (
                        model_path
                        and _is_validation(sample[offset][1])
                        and (truth_hit or _keep_negative(s1_pos, target_idx))
                    ):
                        validation_features.append(compute_pair_features(
                            *s1_values[s1_pos], name, address, country
                        ))
                        validation_labels.append(float(truth_hit))
                        validation_weights.append(1.0 if truth_hit else 10.0)
                        validation_positions.append(s1_pos)
                        for variant, present in selected.items():
                            validation_route_flags[variant].append(present)

            scanned += len(rows)
            if scanned and scanned % 1_000_000 == 0:
                print(f"Candidate pass scanned {scanned:,} targets", flush=True)
        target_rows.close()
        candidate_seconds = time.perf_counter() - route_started

    variants_report = {}
    for variant in variants:
        candidate_hits = int(hits[variant].sum())
        gt_recall_per_s1 = np.divide(
            hits[variant], gt_counts,
            out=np.zeros(sample_rows, dtype=np.float64), where=gt_counts != 0,
        )
        per_entity_oracle = np.divide(
            1.25 * gt_recall_per_s1,
            0.25 + gt_recall_per_s1,
            out=np.zeros(sample_rows, dtype=np.float64),
            where=gt_counts != 0,
        )
        per_entity_oracle[gt_counts == 0] = 1.0
        variants_report[variant] = {
            **_distribution(counts[variant], hits[variant]),
            "candidate_recall": candidate_hits / total_gt if total_gt else 0.0,
            "candidate_recall_by_source": {
                "S2": source_hits[variant]["S2"] / gt_by_source["S2"] if gt_by_source["S2"] else 0.0,
                "S3": source_hits[variant]["S3"] / gt_by_source["S3"] if gt_by_source["S3"] else 0.0,
            },
            "perfect_precision_macro_f0_5_ceiling": float(per_entity_oracle.mean()),
        }

    model_validation = None
    if model_path and candidate_seconds:
        model_validation = _score_sample(
            model_path,
            validation_features,
            validation_labels,
            validation_weights,
            validation_positions,
            validation_route_flags,
            sample,
            truth,
            counts,
            hits,
            sample_start,
        )
    db.close()
    return {
        "sample": {"start_row_zero_based": sample_start, "source1_rows": sample_rows},
        "blocking": {
            "existing_routes": ["E", "P6", "P4", "ST", "W"],
            "supplemental_routes": {
                "SX": "country + Soundex code of significant ASCII name tokens, token length >= 4",
                "AN": "country + exact numeric address token, at least 3 digits",
                "NG3": "country + unique character trigrams of compact normalized name",
            },
            "frequency_cap": cap,
            "frequency_scope": "target rows across Source 2 and Source 3 combined",
            "candidate_pair_exported": False,
        },
        "ground_truth_pairs": total_gt,
        "ground_truth_pairs_by_source": gt_by_source,
        "supplemental_key_stats": {
            route: {
                "query_keys": len(query),
                "keys_below_cap": len(active_extra[route]),
                "target_key_probe_upper_bound": key_probe_upper_bounds[route],
                **standalone[route],
            }
            for route, query in extra_query.items()
        },
        "base_route_key_probe_upper_bound": base_probe_upper_bound,
        "all_route_key_probe_upper_bound": observed_probe_upper_bound,
        "variants": variants_report if candidate_seconds else None,
        "model_validation": model_validation,
        "candidate_scan_skipped": observed_probe_upper_bound > max_target_route_probes,
        "candidate_scan_skip_reason": (
            f"new-route query-key probe upper bound {observed_probe_upper_bound:,} "
            f"exceeds safety limit {max_target_route_probes:,}"
            if observed_probe_upper_bound > max_target_route_probes else None
        ),
        "timing_seconds": {
            "frequency_scan": round(frequency_seconds, 1),
            "candidate_scan": round(candidate_seconds, 1),
            "total": round(time.perf_counter() - started, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="v2_train_index.sqlite3")
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--sample-start", type=int, default=400_000)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--cap", type=int, default=MAX_POSTINGS_PER_KEY)
    parser.add_argument("--model", default=None, help="Optional saved XGBoost model for held-out scoring")
    parser.add_argument("--report", default="reports/v2_loose_routes_50k.json")
    args = parser.parse_args()

    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    report = benchmark(
        index_path=args.index,
        source1_path=args.source1,
        ground_truth_path=args.ground_truth,
        sample_start=args.sample_start,
        sample_rows=args.sample_rows,
        cap=args.cap,
        model_path=args.model,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
