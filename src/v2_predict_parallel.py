"""V2 Parallel Test Prediction — Architecture 3.0.

Main process batches SQLite reads with SORTED disk access (lightning fast).
Worker processes handle feature computation & XGBoost (CPU heavy).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import resource
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    compute_pair_features,
    generate_blocking_keys,
    generate_numeric_address_keys,
)

# ── Worker functions (No SQLite here) ───────────────────────────

def _init_worker(model_path, threshold, max_matches):
    global _w_model, _w_threshold, _w_max_matches
    meta = json.loads(Path(model_path).read_text())
    
    _w_threshold = threshold if threshold is not None else meta.get("threshold", 0.5)
    _w_max_matches = max_matches
    
    xgb_file = Path(model_path).parent / meta["xgb_model_file"]
    _w_model = xgb.XGBClassifier(n_jobs=1)
    _w_model.load_model(str(xgb_file))


def _score_entity(args: tuple) -> tuple[str, str]:
    """Receives (s1_eid, nn, na, nc, candidates_list). Computes feats + XGB."""
    s1_eid, nn, na, nc, candidates = args
    if not candidates:
        return (s1_eid, "")

    model = _w_model
    threshold = _w_threshold
    max_matches = _w_max_matches

    feats = []
    cand_eids = []
    for teid, tn, ta, tc, tsrc in candidates:
        feats.append(compute_pair_features(nn, na, nc, tn, ta, tc, target_source=tsrc))
        cand_eids.append(teid)

    X = np.array(feats, dtype=np.float64)
    probs = model.predict_proba(X)[:, 1]

    scored = [
        (probs[i], cand_eids[i]) for i in range(len(cand_eids))
        if probs[i] >= threshold
    ]
    scored.sort(reverse=True)
    if max_matches and len(scored) > max_matches:
        scored = scored[:max_matches]

    selected = [eid_s for _, eid_s in scored]
    return (s1_eid, ",".join(selected))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import concurrent.futures
    mp.set_start_method('spawn', force=True)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", default="dataset/test/test_source1.tsv")
    ap.add_argument("--index-db", default="v2_test_index.sqlite3")
    ap.add_argument("--address-index", default=None,
                    help="Optional numeric-address index aligned to --index-db")
    ap.add_argument("--model", default="models/xgb_v4.json")
    ap.add_argument("--matching", default="output_v5/matching_results.tsv")
    ap.add_argument("--report", default="reports/v2_test_v5.json")
    ap.add_argument("--threshold", type=float, default=None, help="If unset, uses threshold tuned during training")
    ap.add_argument("--max-matches", type=int, default=11)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-s1-rows", type=int, default=None,
                    help="Optional row limit for smoke runs")
    args = ap.parse_args()

    out_path = Path(args.matching)
    report_path = Path(args.report)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite matching output: {out_path}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite inference report: {report_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Open SQLite in main process
    db = sqlite3.connect(f"file:{args.index_db}?mode=ro", uri=True)
    db.execute("PRAGMA cache_size=-131072")
    db.execute("""CREATE TEMP TABLE query_keys (
        s1_pos INTEGER NOT NULL, block_key TEXT NOT NULL,
        PRIMARY KEY (s1_pos, block_key)) WITHOUT ROWID""")
    address_db = None
    if args.address_index:
        address_db = sqlite3.connect(
            f"file:{args.address_index}?mode=ro", uri=True
        )
        address_db.execute("PRAGMA cache_size=-32768")
        address_db.execute("""CREATE TEMP TABLE query_keys (
            s1_pos INTEGER NOT NULL, block_key TEXT NOT NULL,
            PRIMARY KEY (s1_pos, block_key)) WITHOUT ROWID""")

    print(f"Scoring with {args.workers} workers, threshold={args.threshold} ...")
    
    total_predicted = 0
    total_zero = 0
    total_zero_candidates = 0
    total_candidates = 0
    total_s1_scored = 0

    with out_path.open("w", newline="") as f, \
         concurrent.futures.ProcessPoolExecutor(
             max_workers=args.workers,
             initializer=_init_worker,
             initargs=(args.model, args.threshold, args.max_matches)
         ) as pool:
        
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "matched_entity_ids"))

        pbar = tqdm(
            total=args.limit_s1_rows or 1732544,
            desc="Scoring",
            unit=" rows",
        )
        
        batch_size = 10000
        batch_meta = []  # (eid, nn, na, nc)
        key_rows = []    # (pos, key)
        address_key_rows = []
        input_rows_read = 0
        
        def process_batch():
            nonlocal total_predicted, total_zero, total_zero_candidates
            nonlocal total_candidates, total_s1_scored
            if not batch_meta: return
            
            # 1. Fetch matching target_idxs for all queries
            db.execute("DELETE FROM query_keys")
            db.executemany("INSERT OR IGNORE INTO query_keys VALUES(?,?)", key_rows)
            
            # Fetch candidates using indexed SQL
            candidates = db.execute("""
                SELECT q.s1_pos, p.target_idx
                FROM query_keys AS q
                JOIN key_counts AS kc ON kc.block_key = q.block_key AND kc.freq <= ?
                JOIN postings  AS p  ON p.block_key = q.block_key
                ORDER BY q.s1_pos, p.target_idx
            """, (MAX_POSTINGS_PER_KEY,)).fetchall()
            if address_db is not None:
                address_db.execute("DELETE FROM query_keys")
                if address_key_rows:
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
            
            # 2. Group by position and get unique targets
            pos_tidxs = defaultdict(list)
            all_tidxs = set()
            for pos, tidx in candidates:
                if not pos_tidxs[pos] or pos_tidxs[pos][-1] != tidx:
                    pos_tidxs[pos].append(tidx)
                    all_tidxs.add(tidx)
                    
            # 3. Fetch target data in sorted chunk order (CRITICAL for speed)
            target_data = {}
            sorted_tidxs = sorted(all_tidxs)
            for bs in range(0, len(sorted_tidxs), 20_000):
                chunk = sorted_tidxs[bs:bs + 20_000]
                ph = ",".join("?" * len(chunk))
                rows = db.execute(f"""
                    SELECT target_idx, entity_id, normalized_name, normalized_address, normalized_country
                    FROM targets WHERE target_idx IN ({ph})
                """, chunk).fetchall()
                for r in rows:
                    target_data[r[0]] = (r[1], r[2], r[3], r[4], 0)  # eid, name, addr, country, source(unknown for test)
                    
            # 4. Build inputs for workers
            worker_args = []
            for pos, (eid, nn, na, nc) in enumerate(batch_meta):
                target_idxs = pos_tidxs.get(pos, [])
                total_candidates += len(target_idxs)
                if not target_idxs:
                    total_zero_candidates += 1
                cands = [target_data[tidx] for tidx in target_idxs]
                worker_args.append((eid, nn, na, nc, cands))
                
            # 5. Process in parallel
            for eid, matched in pool.map(_score_entity, worker_args, chunksize=200):
                writer.writerow((eid, matched))
                if matched: total_predicted += len(matched.split(","))
                else: total_zero += 1
            
            pbar.update(len(batch_meta))
            total_s1_scored += len(batch_meta)
            f.flush()
            batch_meta.clear()
            key_rows.clear()
            address_key_rows.clear()

        # Stream rows
        for chunk in iter_source_chunks(args.source1, chunk_size=50_000):
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                if args.limit_s1_rows is not None and input_rows_read >= args.limit_s1_rows:
                    break
                input_rows_read += 1
                nn = normalize_business_name(name)
                na = normalize_business_address(addr)
                nc = normalize_country(country)
                
                pos = len(batch_meta)
                batch_meta.append((eid, nn, na, nc))
                
                for key in generate_blocking_keys(nn, nc):
                    key_rows.append((pos, key))
                if address_db is not None:
                    for key in generate_numeric_address_keys(na, nc):
                        address_key_rows.append((pos, key))
                
                if len(batch_meta) >= batch_size:
                    process_batch()
            if args.limit_s1_rows is not None and input_rows_read >= args.limit_s1_rows:
                break
                    
        if batch_meta:
            process_batch()
            
        pbar.close()
    
    elapsed = time.perf_counter() - t0
    report = {
        "model": str(Path(args.model).resolve()),
        "threshold": args.threshold,
        "max_matches_per_s1": args.max_matches,
        "blocking_routes": ["E", "P6", "P4", "ST", "W"] + (
            ["AN"] if args.address_index else []
        ),
        "max_postings_per_key": MAX_POSTINGS_PER_KEY,
        "rows": {
            "source1_scored": total_s1_scored,
            "candidate_pairs": total_candidates,
            "predicted_pairs": total_predicted,
            "zero_candidate_s1": total_zero_candidates,
            "zero_match_s1": total_zero,
        },
        "timing_seconds": round(elapsed, 1),
        "rows_per_second": round(total_s1_scored / elapsed, 1) if elapsed else 0.0,
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    if address_db is not None:
        address_db.close()
    db.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
