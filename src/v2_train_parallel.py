"""V2 Parallel Training — Max Precision Architecture.

Uses multiprocess feature extraction and fast SQLite batching.
Forces weight=1.0 and scale_pos_weight=1.0 to maximize F0.5 precision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

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
    generate_numeric_address_keys,
)
from src.v2_metrics import tune_official_macro_f05

# ── Configuration ─────────────────────────────────────────────────────
NEG_SUBSAMPLE_RATE = 10          # 10x more negatives!
S1_QUERY_BATCH = 10_000

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

# ── Worker functions ──────────────────────────────────────────────────
def _worker_compute(batch: list[tuple]) -> tuple:
    """Computes features for a batch of examples."""
    feats = []
    labels = []
    weights = []
    s1_positions = []
    is_vals = []
    for s1_nn, s1_na, s1_nc, t_nn, t_na, t_nc, t_source, label, weight, s1_pos, is_val in batch:
        feats.append(compute_pair_features(s1_nn, s1_na, s1_nc, t_nn, t_na, t_nc, target_source=t_source))
        labels.append(label)
        weights.append(weight)
        s1_positions.append(s1_pos)
        is_vals.append(is_val)
    
    return (
        np.array(feats, dtype=np.float64),
        np.array(labels, dtype=np.float64),
        np.array(weights, dtype=np.float64),
        np.array(s1_positions, dtype=np.int64),
        np.array(is_vals, dtype=bool)
    )

# ── Main ─────────────────────────────────────────────────────────────

def _f05(p: float, r: float) -> float:
    d = 0.25 * p + r
    return 1.25 * p * r / d if d else 0.0

def main():
    import concurrent.futures
    mp.set_start_method('spawn', force=True)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", default="dataset/train/train_source1.tsv")
    ap.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--sample-rows", type=int, default=200_000)
    ap.add_argument("--index-db", default="v2_train_index.sqlite3")
    ap.add_argument("--address-index", default=None,
                    help="Optional numeric-address sidecar index for AN route")
    ap.add_argument("--model", default="models/xgb_v4.json")
    ap.add_argument("--report", default="reports/v4_training.json")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    t0 = time.perf_counter()

    print(f"Loading Ground Truth...", flush=True)
    gt: dict[str, set[str]] = {}
    df = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
    for s1_id, matched in df.itertuples(index=False, name=None):
        gt[s1_id] = set(matched.split(",")) if matched else set()

    print(f"Sampling {args.sample_rows:,} S1 rows...", flush=True)
    s1_sample = []
    row_idx = 0
    for chunk in iter_source_chunks(args.source1, chunk_size=50_000):
        for eid, name, addr, country in chunk.itertuples(index=False, name=None):
            nn = normalize_business_name(name)
            na = normalize_business_address(addr)
            nc = normalize_country(country)
            s1_sample.append((row_idx, eid, nn, na, nc, _is_validation(eid)))
            row_idx += 1
            if row_idx >= args.sample_rows: break
        if row_idx >= args.sample_rows: break

    # Open SQLite in main process
    db = sqlite3.connect(f"file:{args.index_db}?mode=ro", uri=True)
    db.execute("PRAGMA cache_size=-131072")
    db.execute("""CREATE TEMP TABLE query_keys (
        s1_pos INTEGER NOT NULL, block_key TEXT NOT NULL,
        PRIMARY KEY (s1_pos, block_key)) WITHOUT ROWID""")
    address_db = None
    if args.address_index:
        address_db = sqlite3.connect(f"file:{args.address_index}?mode=ro", uri=True)
        address_db.execute("PRAGMA cache_size=-32768")
        address_db.execute("""CREATE TEMP TABLE query_keys (
            s1_pos INTEGER NOT NULL, block_key TEXT NOT NULL,
            PRIMARY KEY (s1_pos, block_key)) WITHOUT ROWID""")

    print(f"Generating Candidates & Features with {args.workers} workers...", flush=True)
    
    all_X = []
    all_y = []
    all_w = []
    all_s1p = []
    all_isval = []
    
    cand_stats = {
        "total_candidates": 0,
        "positives": 0,
        "negatives_kept": 0,
        "zero_candidate_s1": 0,
        "validation_zero_candidate_s1": 0,
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pbar = tqdm(total=len(s1_sample), desc="Processing", unit=" rows")
        
        for batch_start in range(0, len(s1_sample), S1_QUERY_BATCH):
            batch = s1_sample[batch_start:batch_start + S1_QUERY_BATCH]
            
            # 1. Fetch Candidates from SQLite (base routes)
            key_rows = []
            address_key_rows = []
            for s1_pos, _, nn, na, nc, _ in batch:
                for key in generate_blocking_keys(nn, nc):
                    key_rows.append((s1_pos, key))
                if address_db is not None:
                    for key in generate_numeric_address_keys(na, nc):
                        address_key_rows.append((s1_pos, key))
                    
            db.execute("DELETE FROM query_keys")
            if key_rows:
                db.executemany("INSERT OR IGNORE INTO query_keys VALUES(?,?)", key_rows)
            
            candidates = db.execute("""
                SELECT q.s1_pos, p.target_idx
                FROM query_keys AS q
                JOIN key_counts AS kc ON kc.block_key = q.block_key AND kc.freq <= ?
                JOIN postings  AS p  ON p.block_key = q.block_key
                ORDER BY q.s1_pos, p.target_idx
            """, (MAX_POSTINGS_PER_KEY,)).fetchall()

            # 1b. Address Number candidates from sidecar
            if address_db is not None and address_key_rows:
                address_db.execute("DELETE FROM query_keys")
                address_db.executemany(
                    "INSERT OR IGNORE INTO query_keys VALUES (?, ?)",
                    address_key_rows,
                )
                address_candidates = address_db.execute("""
                    SELECT q.s1_pos, p.target_idx
                    FROM query_keys AS q
                    JOIN key_counts AS kc ON kc.block_key = q.block_key AND kc.freq <= ?
                    JOIN postings AS p ON p.block_key = q.block_key
                    ORDER BY q.s1_pos, p.target_idx
                """, (MAX_POSTINGS_PER_KEY,)).fetchall()
                candidates.extend(address_candidates)
                candidates.sort()
            
            # 2. Identify Unique Targets & Fetch Data in sorted order
            pos_tidxs = defaultdict(list)
            all_tidxs = set()
            for pos, tidx in candidates:
                if not pos_tidxs[pos] or pos_tidxs[pos][-1] != tidx:
                    pos_tidxs[pos].append(tidx)
                    all_tidxs.add(tidx)
                    
            target_data = {}   # tidx -> (name, addr, country, source)
            target_eids = {}
            sorted_tidxs = sorted(all_tidxs)
            for bs in range(0, len(sorted_tidxs), 20_000):
                chunk = sorted_tidxs[bs:bs + 20_000]
                ph = ",".join("?" * len(chunk))
                rows = db.execute(f"""
                    SELECT target_idx, entity_id, normalized_name, normalized_address, normalized_country, source
                    FROM targets WHERE target_idx IN ({ph})
                """, chunk).fetchall()
                for r in rows:
                    target_data[r[0]] = (r[2], r[3], r[4], r[5])  # name, addr, country, source
                    target_eids[r[0]] = r[1]
            
            # 3. Ground Truth Matching & Subsampling
            worker_args = []
            for s1_pos, s1_eid, s1_nn, s1_na, s1_nc, is_val in batch:
                gt_set = gt.get(s1_eid, set())
                cands = pos_tidxs.get(s1_pos, [])
                cand_stats["total_candidates"] += len(cands)
                if not cands:
                    cand_stats["zero_candidate_s1"] += 1
                    if is_val:
                        cand_stats["validation_zero_candidate_s1"] += 1
                
                valid_examples = []
                for tidx in cands:
                    t_eid = target_eids[tidx]
                    label = 1.0 if t_eid in gt_set else 0.0
                    t_nn, t_na, t_nc, t_src = target_data[tidx]
                    
                    if label == 1.0:
                        cand_stats["positives"] += 1
                        valid_examples.append((s1_nn, s1_na, s1_nc, t_nn, t_na, t_nc, t_src, label, 1.0, s1_pos, is_val))
                    elif _neg_sample(s1_pos, tidx):
                        cand_stats["negatives_kept"] += 1
                        valid_examples.append((s1_nn, s1_na, s1_nc, t_nn, t_na, t_nc, t_src, label, float(NEG_SUBSAMPLE_RATE), s1_pos, is_val))
                
                # Split into chunks of 100 for pool to balance load
                for i in range(0, len(valid_examples), 100):
                    worker_args.append(valid_examples[i:i+100])

            # 4. Multiprocess Feature Computation
            for fx, fy, fw, fs1, fval in pool.map(_worker_compute, worker_args):
                if len(fx) > 0:
                    all_X.append(fx)
                    all_y.append(fy)
                    all_w.append(fw)
                    all_s1p.append(fs1)
                    all_isval.append(fval)
                    
            pbar.update(len(batch))
            
        pbar.close()

    sampled_gt_pairs = sum(len(gt.get(s1_eid, set())) for _, s1_eid, *_ in s1_sample)
    cand_stats["ground_truth_pairs_sampled"] = sampled_gt_pairs
    cand_stats["candidate_recall"] = (
        cand_stats["positives"] / sampled_gt_pairs if sampled_gt_pairs else 0.0
    )

    # ── Concat arrays ──
    print(f"\nConcatenating Arrays...", flush=True)
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    w = np.concatenate(all_w)
    s1_positions = np.concatenate(all_s1p)
    is_val = np.concatenate(all_isval)
    
    t1 = time.perf_counter()
    print(f"Features ready! {len(X):,} examples ({t1 - t0:.1f}s)", flush=True)

    # ── Train XGBoost ──
    print("\nPhase 4: training XGBoost ...", flush=True)
    train_mask = ~is_val
    val_mask = is_val

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]

    print(f"  training: {train_mask.sum():,} examples ({int(y_train.sum()):,} pos)")
    print(f"  validation: {val_mask.sum():,} examples ({int(y_val.sum()):,} pos)")
    print(f"  scale_pos_weight: 1.0 (Forced for High Precision)")

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=1.0,
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

    # ── Tune Threshold ──
    print("\nTuning Threshold...", flush=True)
    val_probs = model.predict_proba(X_val)[:, 1]
    validation_positions = [
        s1_pos for s1_pos, _eid, *_ in s1_sample if _is_validation(_eid)
    ]
    gt_counts = {
        s1_pos: len(gt.get(s1_eid, set()))
        for s1_pos, s1_eid, *_ in s1_sample
        if _is_validation(s1_eid)
    }
    best_t, val_report = tune_official_macro_f05(
        val_probs,
        y_val,
        w_val,
        s1_positions[val_mask],
        validation_positions,
        gt_counts,
    )
    validation_gt_pairs = sum(gt_counts.values())
    val_report.update({
        "candidate_recall": (
            float(y_val.sum()) / validation_gt_pairs if validation_gt_pairs else 0.0
        ),
        "validation_ground_truth_pairs": validation_gt_pairs,
        "validation_zero_candidate_s1": cand_stats["validation_zero_candidate_s1"],
    })
    val_entities = validation_positions

    print(f"\nThreshold: {best_t:.6f}, Macro F0.5: {val_report['macro_f0_5']:.6f}")

    # ── Save ──
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    xgb_path = model_path.with_suffix(".xgb")
    model.save_model(str(xgb_path))

    val_report = {
        "threshold": best_t,
        **val_report,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else -1,
    }

    meta = {
        "model_type": "xgboost_v4",
        "feature_names": list(V2_FEATURE_NAMES),
        "threshold": best_t,
        "xgb_model_file": str(xgb_path.name),
        "training": {
            "sample_start": 0,
            "sample_rows": args.sample_rows,
            "neg_subsample_rate": NEG_SUBSAMPLE_RATE,
            "negative_inverse_probability_weight": NEG_SUBSAMPLE_RATE,
            "max_postings_per_key": MAX_POSTINGS_PER_KEY,
            "blocking_routes": ["E", "P6", "P4", "ST", "W"] + (
                ["AN"] if args.address_index else []
            ),
        },
        "validation": val_report,
    }
    model_path.write_text(json.dumps(meta, indent=2) + "\n")

    report = {
        **meta,
        "candidate_stats": cand_stats,
        "total_examples": len(X),
        "timing": {
            "feature_compute_s": round(t1 - t0, 1),
            "training_s": round(time.perf_counter() - t1, 1),
            "total_s": round(time.perf_counter() - t0, 1),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved model to {model_path}")
    print(f"Total time: {time.perf_counter() - t0:.0f}s")

if __name__ == "__main__":
    main()
