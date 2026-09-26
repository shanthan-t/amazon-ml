"""V2 Training: multi-key blocking + 23-feature XGBoost.

Usage:
    python3 -m src.v2_train [--sample-rows 100000] [--sample-start 0]

Outputs:
    models/xgb_v2.json       – XGBoost model (JSON)
    reports/v2_training.json  – validation metrics
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import iter_source_chunks
from src.normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)
from src.v2_common import (
    MAX_POSTINGS_PER_KEY,
    V2_FEATURE_NAMES,
    compute_pair_features,
    generate_blocking_keys,
)
from src.v2_metrics import tune_official_macro_f05

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_SAMPLE_START = 0
DEFAULT_SAMPLE_ROWS = 100_000
NEG_SUBSAMPLE_RATE = 10          # keep 1-in-10 negatives (massively increases precision!)
S1_QUERY_BATCH = 2_000


# ── Helpers ───────────────────────────────────────────────────────────

def _is_validation(entity_id: str) -> bool:
    h = hashlib.blake2b(entity_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % 5 == 0


def _neg_sample(s1_pos: int, target_idx: int) -> bool:
    mask = (1 << 64) - 1
    v = (s1_pos * 0x9E3779B185EBCA87 + target_idx * 0xC2B2AE3D27D4EB4F)
    v ^= v >> 30
    v = (v * 0xBF58476D1CE4E5B9) & mask
    v ^= v >> 27
    return v % NEG_SUBSAMPLE_RATE == 0


# ── Phase 1: build blocking index ────────────────────────────────────

def build_index(db: sqlite3.Connection, source_path: str, source: int,
                offset: int) -> int:
    """Stream a source TSV into the blocking index. Returns row count."""
    count = 0
    key_batch: list[tuple[str, int]] = []
    target_batch: list[tuple[int, str, str, str, str, int]] = []
    for chunk in iter_source_chunks(source_path, chunk_size=50_000):
        for entity_id, name, address, country in chunk.itertuples(
            index=False, name=None
        ):
            idx = offset + count
            nn = normalize_business_name(name)
            na = normalize_business_address(address)
            nc = normalize_country(country)
            target_batch.append((idx, entity_id, nn, na, nc, source))
            for key in generate_blocking_keys(nn, nc):
                key_batch.append((key, idx))
            count += 1
            if len(target_batch) >= 50_000:
                db.executemany(
                    "INSERT INTO targets VALUES(?,?,?,?,?,?)", target_batch)
                db.executemany(
                    "INSERT OR IGNORE INTO postings VALUES(?,?)", key_batch)
                target_batch.clear()
                key_batch.clear()
        if count % 500_000 == 0:
            print(f"  indexed {count:,} from source {source}", flush=True)
    if target_batch:
        db.executemany(
            "INSERT INTO targets VALUES(?,?,?,?,?,?)", target_batch)
        db.executemany(
            "INSERT OR IGNORE INTO postings VALUES(?,?)", key_batch)
    db.commit()
    print(f"  finished source {source}: {count:,} targets", flush=True)
    return count


def create_index_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-131072")  # 128 MB cache
    db.execute("PRAGMA temp_store=FILE")
    db.execute("""CREATE TABLE targets (
        target_idx INTEGER PRIMARY KEY,
        entity_id TEXT NOT NULL,
        normalized_name TEXT NOT NULL DEFAULT '',
        normalized_address TEXT NOT NULL DEFAULT '',
        normalized_country TEXT NOT NULL DEFAULT '',
        source INTEGER NOT NULL
    )""")
    db.execute("""CREATE TABLE postings (
        block_key TEXT NOT NULL,
        target_idx INTEGER NOT NULL,
        PRIMARY KEY (block_key, target_idx)
    ) WITHOUT ROWID""")
    return db


def finalize_index(db: sqlite3.Connection):
    """Build key_counts table and create indexes."""
    print("  building key_counts ...", flush=True)
    db.execute("""CREATE TABLE key_counts AS
        SELECT block_key, COUNT(*) AS freq
        FROM postings GROUP BY block_key""")
    db.execute("CREATE UNIQUE INDEX idx_kc ON key_counts(block_key)")
    db.commit()
    print("  index finalized", flush=True)


# ── Phase 2: generate labelled candidate pairs ───────────────────────

def load_ground_truth(path: str) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for s1_id, matched in df.itertuples(index=False, name=None):
        if matched:
            gt[s1_id] = set(matched.split(","))
        else:
            gt[s1_id] = set()
    return gt


def sample_s1(
    source1_path: str, start: int, rows: int,
) -> list[tuple[int, str, str, str, str]]:
    """Read a contiguous sample of S1 rows."""
    result: list[tuple[int, str, str, str, str]] = []
    row_idx = 0
    end = start + rows
    for chunk in iter_source_chunks(source1_path, chunk_size=50_000):
        chunk_end = row_idx + len(chunk)
        if chunk_end > start and row_idx < end:
            lo = max(0, start - row_idx)
            hi = min(len(chunk), end - row_idx)
            for off, vals in enumerate(
                chunk.iloc[lo:hi].itertuples(index=False, name=None),
                start=lo,
            ):
                eid, name, addr, country = vals
                result.append((row_idx + off, eid, name, addr, country))
        row_idx = chunk_end
        if row_idx >= end:
            break
    return result


_CANDIDATES_SQL = """
SELECT q.s1_pos, p.target_idx
FROM query_keys AS q
JOIN key_counts AS kc ON kc.block_key = q.block_key AND kc.freq <= ?
JOIN postings  AS p  ON p.block_key = q.block_key
ORDER BY q.s1_pos, p.target_idx
"""


def generate_examples(
    db: sqlite3.Connection,
    s1_sample: list[tuple[int, str, str, str, str]],
    ground_truth: dict[str, set[str]],
) -> tuple[list[tuple], dict]:
    """Yield labelled (s1_pos, target_idx, label, weight) examples."""
    db.execute("""CREATE TEMP TABLE IF NOT EXISTS query_keys (
        s1_pos INTEGER NOT NULL, block_key TEXT NOT NULL,
        PRIMARY KEY (s1_pos, block_key)) WITHOUT ROWID""")

    # Build lookup: s1_pos → (entity_id, norm_name, norm_addr, norm_country)
    s1_lookup: dict[int, tuple[str, str, str, str]] = {}
    for s1_pos, eid, name, addr, country in s1_sample:
        nn = normalize_business_name(name)
        na = normalize_business_address(addr)
        nc = normalize_country(country)
        s1_lookup[s1_pos] = (eid, nn, na, nc)

    examples: list[tuple] = []  # (s1_pos, target_idx, label, weight)
    stats = {"total_candidates": 0, "positives": 0, "negatives_kept": 0,
             "positives_found_by_route": 0}

    # Process in batches
    for batch_start in range(0, len(s1_sample), S1_QUERY_BATCH):
        batch = s1_sample[batch_start:batch_start + S1_QUERY_BATCH]
        key_rows: list[tuple[int, str]] = []
        for s1_pos, _eid, name, _addr, country in batch:
            nn = normalize_business_name(name)
            nc = normalize_country(country)
            for key in generate_blocking_keys(nn, nc):
                key_rows.append((s1_pos, key))

        db.execute("DELETE FROM query_keys")
        if key_rows:
            db.executemany(
                "INSERT OR IGNORE INTO query_keys VALUES(?,?)", key_rows)

        candidates = db.execute(
            _CANDIDATES_SQL, (MAX_POSTINGS_PER_KEY,)).fetchall()

        # Deduplicate per S1 and label
        prev_s1 = -1
        seen: set[int] = set()
        for s1_pos, target_idx in candidates:
            if s1_pos != prev_s1:
                prev_s1 = s1_pos
                seen = set()
            if target_idx in seen:
                continue
            seen.add(target_idx)
            stats["total_candidates"] += 1

            eid = s1_lookup[s1_pos][0]
            # Fetch target entity_id for ground-truth check
            t_eid = db.execute(
                "SELECT entity_id FROM targets WHERE target_idx=?",
                (target_idx,)).fetchone()[0]
            gt_set = ground_truth.get(eid, set())
            label = 1 if t_eid in gt_set else 0

            if label:
                stats["positives"] += 1
                examples.append((s1_pos, target_idx, 1, 1.0))
            elif _neg_sample(s1_pos, target_idx):
                stats["negatives_kept"] += 1
                examples.append(
                    (s1_pos, target_idx, 0, float(NEG_SUBSAMPLE_RATE)))

        if (batch_start + S1_QUERY_BATCH) % 20_000 == 0:
            print(f"  candidates for {batch_start + S1_QUERY_BATCH:,} S1 rows,"
                  f" {len(examples):,} examples so far", flush=True)

    return examples, stats


# ── Phase 3: compute features ────────────────────────────────────────

def compute_features_for_examples(
    db: sqlite3.Connection,
    examples: list[tuple],
    s1_lookup: dict[int, tuple[str, str, str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, weights) arrays."""
    n = len(examples)
    X = np.empty((n, len(V2_FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float64)

    # Pre-fetch all needed target data
    target_idxs = sorted(set(ex[1] for ex in examples))
    target_data: dict[int, tuple[str, str, str]] = {}
    for batch_start in range(0, len(target_idxs), 10_000):
        batch = target_idxs[batch_start:batch_start + 10_000]
        placeholders = ",".join("?" * len(batch))
        rows = db.execute(
            f"SELECT target_idx, normalized_name, normalized_address,"
            f" normalized_country FROM targets"
            f" WHERE target_idx IN ({placeholders})", batch
        ).fetchall()
        for tidx, tn, ta, tc in rows:
            target_data[tidx] = (tn, ta, tc)

    for i, (s1_pos, target_idx, label, weight) in enumerate(examples):
        _, s1_nn, s1_na, s1_nc = s1_lookup[s1_pos]
        t_nn, t_na, t_nc = target_data[target_idx]
        X[i] = compute_pair_features(s1_nn, s1_na, s1_nc, t_nn, t_na, t_nc)
        y[i] = label
        w[i] = weight
        if (i + 1) % 200_000 == 0:
            print(f"  computed features for {i + 1:,} / {n:,} examples",
                  flush=True)

    return X, y, w


# ── Phase 4: train XGBoost ───────────────────────────────────────────

def _f05(p: float, r: float) -> float:
    d = 0.25 * p + r
    return 1.25 * p * r / d if d else 0.0


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, w: np.ndarray,
    s1_positions: np.ndarray, is_val: np.ndarray,
    gt_counts: dict[int, int],
) -> tuple[xgb.XGBClassifier, float, dict]:
    """Train XGBoost, tune threshold on macro F0.5, return model."""
    train_mask = ~is_val
    val_mask = is_val

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]

    pos = float(w_train[y_train == 1].sum())
    neg = float(w_train[y_train == 0].sum())
    spw = 1.0  # Force 1.0 to maximize precision instead of balanced accuracy

    print(f"  training: {train_mask.sum():,} examples "
          f"({int(y_train.sum()):,} pos), "
          f"validation: {val_mask.sum():,} examples "
          f"({int(y_val.sum()):,} pos)", flush=True)
    print(f"  scale_pos_weight: {spw:.1f}", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
        verbosity=1,
    )
    model.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=50,
    )

    # Tune against every held-out S1, including rows with no candidate examples.
    val_probs = model.predict_proba(X_val)[:, 1]
    val_s1 = s1_positions[val_mask]
    validation_positions = list(gt_counts)
    best_t, report = tune_official_macro_f05(
        val_probs,
        y_val,
        w_val,
        val_s1,
        validation_positions,
        gt_counts,
    )
    validation_gt_pairs = sum(gt_counts.values())
    report.update({
        "candidate_recall": (
            float(y_val.sum()) / validation_gt_pairs if validation_gt_pairs else 0.0
        ),
        "validation_ground_truth_pairs": validation_gt_pairs,
        "best_iteration": int(model.best_iteration)
        if hasattr(model, "best_iteration") else -1,
    })
    return model, best_t, report


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source1", default="dataset/train/train_source1.tsv")
    ap.add_argument("--source2", default="dataset/train/train_source2.tsv")
    ap.add_argument("--source3", default="dataset/train/train_source3.tsv")
    ap.add_argument("--ground-truth",
                    default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--sample-start", type=int, default=DEFAULT_SAMPLE_START)
    ap.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    ap.add_argument("--index-db", default="v2_train_index.sqlite3")
    ap.add_argument("--model", default="models/xgb_v2.json")
    ap.add_argument("--report", default="reports/v2_training.json")
    args = ap.parse_args()

    t0 = time.perf_counter()

    # ── Phase 1: build blocking index ─────────────────────────────
    index_path = Path(args.index_db)
    if index_path.exists():
        print(f"Reusing existing index: {index_path}", flush=True)
        db = sqlite3.connect(str(index_path))
        db.execute("PRAGMA cache_size=-131072")
        s2_count = db.execute(
            "SELECT COUNT(*) FROM targets WHERE source=2").fetchone()[0]
    else:
        print("Phase 1: building blocking index ...", flush=True)
        db = create_index_db(str(index_path))
        s2_count = build_index(db, args.source2, source=2, offset=0)
        build_index(db, args.source3, source=3, offset=s2_count)
        finalize_index(db)
        gc.collect()
    t1 = time.perf_counter()
    print(f"Index ready ({t1 - t0:.0f}s)\n", flush=True)

    # ── Phase 2: sample S1 + generate candidates ─────────────────
    print("Phase 2: sampling S1 + generating candidates ...", flush=True)
    s1_sample = sample_s1(args.source1, args.sample_start, args.sample_rows)
    gt = load_ground_truth(args.ground_truth)

    # Build s1_lookup
    s1_lookup: dict[int, tuple[str, str, str, str]] = {}
    for s1_pos, eid, name, addr, country in s1_sample:
        nn = normalize_business_name(name)
        na = normalize_business_address(addr)
        nc = normalize_country(country)
        s1_lookup[s1_pos] = (eid, nn, na, nc)

    examples, cand_stats = generate_examples(db, s1_sample, gt)
    t2 = time.perf_counter()
    print(f"Generated {len(examples):,} examples "
          f"({cand_stats['positives']:,} pos, "
          f"{cand_stats['negatives_kept']:,} neg) "
          f"from {cand_stats['total_candidates']:,} candidates "
          f"({t2 - t1:.0f}s)\n", flush=True)

    # ── Phase 3: compute features ────────────────────────────────
    print("Phase 3: computing features ...", flush=True)
    X, y, w = compute_features_for_examples(db, examples, s1_lookup)
    t3 = time.perf_counter()
    print(f"Features computed ({t3 - t2:.0f}s)\n", flush=True)

    # ── Phase 4: train + evaluate ────────────────────────────────
    print("Phase 4: training XGBoost ...", flush=True)
    s1_positions = np.array([ex[0] for ex in examples], dtype=np.int64)
    is_val = np.array([
        _is_validation(s1_lookup[ex[0]][0]) for ex in examples
    ], dtype=bool)

    # Ground-truth pair counts per validation S1
    gt_counts: dict[int, int] = {}
    for s1_pos, eid, *_ in s1_sample:
        if _is_validation(eid):
            gt_counts[s1_pos] = len(gt.get(eid, set()))

    model, threshold, val_report = train_and_evaluate(
        X, y, w, s1_positions, is_val, gt_counts)
    t4 = time.perf_counter()
    print(f"\nThreshold: {threshold:.6f}, "
          f"Macro F0.5: {val_report['macro_f0_5']:.6f} "
          f"({t4 - t3:.0f}s)\n", flush=True)

    # ── Phase 5: save model ──────────────────────────────────────
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    xgb_path = model_path.with_suffix(".xgb")
    model.save_model(str(xgb_path))

    meta = {
        "model_type": "xgboost_v2",
        "feature_names": list(V2_FEATURE_NAMES),
        "threshold": threshold,
        "xgb_model_file": str(xgb_path.name),
        "training": {
            "sample_start": args.sample_start,
            "sample_rows": args.sample_rows,
            "neg_subsample_rate": NEG_SUBSAMPLE_RATE,
            "negative_inverse_probability_weight": NEG_SUBSAMPLE_RATE,
            "max_postings_per_key": MAX_POSTINGS_PER_KEY,
            "blocking_routes": ["E", "P6", "P4", "ST", "W"],
        },
        "validation": val_report,
    }
    model_path.write_text(json.dumps(meta, indent=2) + "\n")

    report = {
        **meta,
        "candidate_stats": cand_stats,
        "total_examples": len(examples),
        "timing": {
            "index_build_s": round(t1 - t0, 1),
            "candidate_gen_s": round(t2 - t1, 1),
            "feature_compute_s": round(t3 - t2, 1),
            "training_s": round(t4 - t3, 1),
            "total_s": round(t4 - t0, 1),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")
    print(f"Total time: {t4 - t0:.0f}s")
    db.close()


if __name__ == "__main__":
    main()
