"""GPU FAISS Dense Matcher.

Uses a SentenceTransformer bi-encoder to embed all entities into a dense vector space (e.g. 384D)
and runs blazing-fast cosine similarity search using FAISS.

This completely replaces SQLite and string-matching heuristics.

Usage:
    PYTHONPATH=. python3 -m src.gpu_dense_matcher \
        --model models/finetuned_biencoder \
        --output output_dl/matching_results.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data_loader import iter_source_chunks


def _entity_text(name: str, address: str, country: str) -> str:
    """Combine text for the embedding model."""
    parts = []
    if name: parts.append(name.strip())
    if address: parts.append(address.strip())
    if country: parts.append(country.strip())
    return " | ".join(parts) if parts else ""


def build_faiss_index(model: SentenceTransformer, sources: list[str], batch_size: int = 1024) -> tuple[faiss.Index, list[str]]:
    """Embed all target entities and build a FAISS FlatIP (Inner Product = Cosine for normalized vectors) index."""
    # SentenceTransformers outputs normalized embeddings by default if we ask for it,
    # or we can normalize manually. FlatIP with normalized vectors = Cosine Similarity.
    
    emb_dim = model.get_sentence_embedding_dimension()
    
    # Use GPU FAISS if available, else CPU
    res = faiss.StandardGpuResources() if hasattr(faiss, "StandardGpuResources") else None
    
    cpu_index = faiss.IndexFlatIP(emb_dim)
    if res:
        print("Using FAISS GPU Index")
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print("Using FAISS CPU Index")
        index = cpu_index

    target_eids = []
    
    for src in sources:
        print(f"Encoding targets from {src}...")
        for chunk in iter_source_chunks(src, chunk_size=50_000):
            batch_texts = []
            batch_eids = []
            
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                batch_texts.append(_entity_text(name, addr, country))
                batch_eids.append(eid)
                
            # Embed
            embeddings = model.encode(
                batch_texts, 
                batch_size=batch_size, 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True # Crucial for Inner Product = Cosine Similarity
            )
            
            index.add(embeddings)
            target_eids.extend(batch_eids)
            
            print(f"  Indexed {len(target_eids):,} total targets...", end="\r")
        print()
            
    print(f"Index complete! Total targets in FAISS: {len(target_eids):,}")
    return index, target_eids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--s2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--s3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                        help="Path to fine-tuned model or HuggingFace hub name")
    parser.add_argument("--output", default="output_dl/matching_results.tsv")
    parser.add_argument("--report", default="reports/dl_inference.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10, help="Max candidates to retrieve per S1")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold (0.0 to 1.0)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model: {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    # 1. Build Index for Targets (S2 + S3)
    t0 = time.perf_counter()
    index, target_eids = build_faiss_index(model, [args.s2, args.s3], batch_size=args.batch_size)
    idx_time = time.perf_counter() - t0

    # 2. Score S1 against Index
    print(f"Scoring S1 queries against FAISS index (Threshold: {args.threshold})...")
    
    t0_score = time.perf_counter()
    total_s1 = 0
    total_matches = 0
    
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "matched_entity_ids"))
        
        for chunk in iter_source_chunks(args.s1, chunk_size=10_000):
            batch_texts = []
            batch_eids = []
            
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                batch_texts.append(_entity_text(name, addr, country))
                batch_eids.append(eid)
                
            total_s1 += len(batch_eids)
            
            # Embed queries
            query_embeddings = model.encode(
                batch_texts, 
                batch_size=args.batch_size, 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # FAISS Search
            distances, indices = index.search(query_embeddings, k=args.top_k)
            
            # Filter and Write
            for i in range(len(batch_eids)):
                s1_eid = batch_eids[i]
                matches = []
                for j in range(args.top_k):
                    score = float(distances[i][j])
                    if score >= args.threshold:
                        matched_eid = target_eids[indices[i][j]]
                        matches.append(matched_eid)
                        total_matches += 1
                
                writer.writerow((s1_eid, ",".join(matches)))
                
            print(f"  Processed {total_s1:,} queries...", end="\r")
    
    score_time = time.perf_counter() - t0_score
    print()
    print("Done!")
    
    report = {
        "model": args.model,
        "threshold": args.threshold,
        "targets_indexed": len(target_eids),
        "queries_scored": total_s1,
        "total_predicted_pairs": total_matches,
        "indexing_time_sec": round(idx_time, 1),
        "scoring_time_sec": round(score_time, 1)
    }
    
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    
    print(f"Saved submission to: {args.output}")

if __name__ == "__main__":
    main()
