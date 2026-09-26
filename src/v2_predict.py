"""V2 Test Prediction: multi-key blocking + 23-feature XGBoost.

Usage:
    python3 -m src.v2_predict [--model models/xgb_v2.json]

Outputs:
    output_v3/matching_results.tsv
    output_v3/candidate_pairs.tsv
    reports/v2_test_prediction.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sqlite3
import time
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
    V2_FEATURE_NAMES,
    compute_pair_features,
    generate_blocking_keys,
)

S1_BATCH_SIZE = 1_000


# ── Index building (same logic as v2_train) ──────────────────────────

def create_index_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-131072")
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


def build_index(db: sqlite3.Connection, source_path: str, source: int,
                offset: int) -> int:
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


def finalize_index(db: sqlite3.Connection):
    print("  building key_counts ...", flush=True)
    db.execute("""CREATE TABLE key_counts AS
        SELECT block_key, COUNT(*) AS freq
        FROM postings GROUP BY block_key""")
    db.execute("CREATE UNIQUE INDEX idx_kc ON key_counts(block_key)")
    db.commit()
    print("  index finalized", flush=True)


# ── Prediction ───────────────────────────────────────────────────────

_CANDIDATES_SQL = """
SELECT q.s1_pos, p.target_idx
FROM query_keys AS q
JOIN key_counts AS kc ON kc.block_key = q.block_key AND kc.freq <= ?
JOIN postings  AS p  ON p.block_key = q.block_key
ORDER BY q.s1_pos, p.target_idx
"""


def predict_test(
    *,
    source1: str,
    index_db: str,
    model_path: str,
    matching_output: str,
    report_path: str,
    limit_s1_rows: int | None = None,
    threshold_override: float | None = None,
    max_matches_per_s1: int | None = None,
) -> dict:
    started = time.perf_counter()

    # Load model
    meta = json.loads(Path(model_path).read_text())
    threshold = threshold_override if threshold_override is not None else float(meta["threshold"])
    xgb_file = Path(model_path).parent / meta["xgb_model_file"]
    model = xgb.XGBClassifier()
    model.load_model(str(xgb_file))
    print(f"Loaded model: threshold={threshold:.6f}"
          f"{f', max_matches={max_matches_per_s1}' if max_matches_per_s1 else ''}",
          flush=True)

    # Open index
    db = sqlite3.connect(index_db)
    db.execute("PRAGMA cache_size=-131072")
    db.execute("""CREATE TEMP TABLE query_keys (
        s1_pos INTEGER NOT NULL, block_key TEXT NOT NULL,
        PRIMARY KEY (s1_pos, block_key)) WITHOUT ROWID""")

    # Open output files
    m_path = Path(matching_output)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    s1_total = 0
    candidate_pairs_total = 0
    predicted_pairs_total = 0
    zero_candidate = 0
    zero_match = 0
    batch: list[tuple[int, str, str, str, str]] = []
    next_pos = 0

    def flush(records, m_writer):
        nonlocal candidate_pairs_total, predicted_pairs_total
        nonlocal zero_candidate, zero_match
        if not records:
            return

        # Insert query keys
        key_rows = []
        for pos, _eid, name, _addr, country in records:
            nn = normalize_business_name(name)
            nc = normalize_country(country)
            for key in generate_blocking_keys(nn, nc):
                key_rows.append((pos, key))
        db.execute("DELETE FROM query_keys")
        if key_rows:
            db.executemany(
                "INSERT OR IGNORE INTO query_keys VALUES(?,?)", key_rows)

        # Fetch candidates
        candidates = db.execute(
            _CANDIDATES_SQL, (MAX_POSTINGS_PER_KEY,)).fetchall()

        # Group by S1 + deduplicate
        from collections import defaultdict
        s1_candidates: dict[int, list[int]] = defaultdict(list)
        prev_s1 = -1
        seen: set[int] = set()
        for s1_pos, tidx in candidates:
            if s1_pos != prev_s1:
                prev_s1 = s1_pos
                seen = set()
            if tidx not in seen:
                seen.add(tidx)
                s1_candidates[s1_pos].append(tidx)

        # Fetch target data for all candidates
        all_tidxs = sorted(set(
            t for tlist in s1_candidates.values() for t in tlist))
        target_data: dict[int, tuple[str, str, str, str]] = {}
        for bs in range(0, len(all_tidxs), 10_000):
            chunk_idxs = all_tidxs[bs:bs + 10_000]
            ph = ",".join("?" * len(chunk_idxs))
            rows = db.execute(
                f"SELECT target_idx, entity_id, normalized_name,"
                f" normalized_address, normalized_country"
                f" FROM targets WHERE target_idx IN ({ph})",
                chunk_idxs
            ).fetchall()
            for tidx, teid, tn, ta, tc in rows:
                target_data[tidx] = (teid, tn, ta, tc)

        # Score each S1
        for pos, eid, name, addr, country in records:
            nn = normalize_business_name(name)
            na = normalize_business_address(addr)
            nc = normalize_country(country)
            cand_tidxs = s1_candidates.get(pos, [])

            if not cand_tidxs:
                zero_candidate += 1
                m_writer.writerow((eid, ""))
                continue

            # Compute features
            feats = []
            cand_eids = []
            for tidx in cand_tidxs:
                teid, tn, ta, tc = target_data[tidx]
                feats.append(compute_pair_features(nn, na, nc, tn, ta, tc))
                cand_eids.append(teid)

            X = np.array(feats, dtype=np.float64)
            probs = model.predict_proba(X)[:, 1]

            # Select candidates above threshold, sorted by probability
            scored = [
                (probs[i], cand_eids[i]) for i in range(len(cand_eids))
                if probs[i] >= threshold
            ]
            scored.sort(reverse=True)  # highest probability first

            # Cap at max_matches_per_s1 if set
            if max_matches_per_s1 and len(scored) > max_matches_per_s1:
                scored = scored[:max_matches_per_s1]

            selected = [eid_s for _, eid_s in scored]

            candidate_pairs_total += len(cand_eids)
            predicted_pairs_total += len(selected)
            if not selected:
                zero_match += 1

            m_writer.writerow((eid, ",".join(selected)))

    try:
        mode = "a" if limit_s1_rows and limit_s1_rows < 0 else "w"
        skip_rows = -limit_s1_rows if limit_s1_rows and limit_s1_rows < 0 else 0
        
        with m_path.open(mode, newline="") as mf:
            mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
            if mode == "w":
                mw.writerow(("source1_entity_id", "matched_entity_ids"))

            pbar = tqdm(desc="Scoring", unit=" rows", initial=skip_rows)
            for chunk in iter_source_chunks(source1, chunk_size=50_000):
                for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                    if next_pos < skip_rows:
                        next_pos += 1
                        continue
                        
                    batch.append((next_pos, eid, name, addr, country))
                    next_pos += 1
                    if len(batch) >= S1_BATCH_SIZE:
                        flush(batch, mw)
                        s1_total += len(batch)
                        pbar.update(len(batch))
                        batch.clear()
                
                # If we had a positive limit to stop early
                if limit_s1_rows and limit_s1_rows > 0 and next_pos >= limit_s1_rows:
                    break
            if batch:
                flush(batch, mw)
                s1_total += len(batch)
                pbar.update(len(batch))
                batch.clear()
            pbar.close()
    finally:
        db.close()

    elapsed = time.perf_counter() - started
    report = {
        "model": str(Path(model_path).resolve()),
        "threshold": threshold,
        "blocking_routes": ["E", "P6", "P4", "ST", "W"],
        "max_postings_per_key": MAX_POSTINGS_PER_KEY,
        "rows": {
            "source1_scored": s1_total,
            "candidate_pairs": candidate_pairs_total,
            "predicted_pairs": predicted_pairs_total,
            "zero_candidate_s1": zero_candidate,
            "zero_match_s1": zero_match,
        },
        "timing_seconds": round(elapsed, 1),
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source1", default="dataset/test/test_source1.tsv")
    ap.add_argument("--source2", default="dataset/test/test_source2.tsv")
    ap.add_argument("--source3", default="dataset/test/test_source3.tsv")
    ap.add_argument("--index-db", default="v2_test_index.sqlite3")
    ap.add_argument("--model", default="models/xgb_v2.json")
    ap.add_argument("--matching", default="output_v3/matching_results.tsv")
    ap.add_argument("--report", default="reports/v2_test_prediction.json")
    ap.add_argument("--limit-s1-rows", type=int)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override model threshold (e.g. 0.95)")
    ap.add_argument("--max-matches", type=int, default=None,
                    help="Cap matches per S1 entity (e.g. 11)")
    args = ap.parse_args()

    # Build or reuse test index
    idx_path = Path(args.index_db)
    if idx_path.exists():
        print(f"Reusing existing test index: {idx_path}", flush=True)
    else:
        print("Building test blocking index ...", flush=True)
        t0 = time.perf_counter()
        db = create_index_db(str(idx_path))
        s2_count = build_index(db, args.source2, source=2, offset=0)
        build_index(db, args.source3, source=3, offset=s2_count)
        finalize_index(db)
        db.close()
        gc.collect()
        print(f"Index built in {time.perf_counter() - t0:.0f}s\n", flush=True)

    print("Running test prediction ...", flush=True)
    report = predict_test(
        source1=args.source1,
        index_db=str(idx_path),
        model_path=args.model,
        matching_output=args.matching,
        report_path=args.report,
        limit_s1_rows=args.limit_s1_rows,
        threshold_override=args.threshold,
        max_matches_per_s1=args.max_matches,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
