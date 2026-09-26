"""Fine-tune a Sentence-Transformer bi-encoder on training ground truth pairs.

Uses MultipleNegativesRankingLoss (contrastive learning) to teach the model
that matched entity pairs should have high cosine similarity, while unmatched
entities should be pushed apart in vector space.

Usage:
    PYTHONPATH=. python3 -m src.gpu_finetune_biencoder \
        --epochs 3 --batch-size 256 --output models/finetuned_biencoder
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

from src.data_loader import iter_source_chunks, iter_ground_truth_chunks
from src.normalization import normalize_business_name, normalize_business_address


# ── Helpers ───────────────────────────────────────────────────────────

def _entity_text(name: str, address: str, country: str) -> str:
    """Build a single text string for the transformer to encode."""
    parts = []
    if name:
        parts.append(name.strip())
    if address:
        parts.append(address.strip())
    if country:
        parts.append(country.strip())
    return " | ".join(parts) if parts else ""


def load_entities(source_path: str) -> dict[str, str]:
    """Load a source TSV into {entity_id: combined_text}."""
    entities: dict[str, str] = {}
    for chunk in iter_source_chunks(source_path, chunk_size=100_000):
        for eid, name, addr, country in chunk.itertuples(index=False, name=None):
            entities[eid] = _entity_text(name, addr, country)
    return entities


def build_training_pairs(
    gt_path: str,
    s1_entities: dict[str, str],
    s2_entities: dict[str, str],
    s3_entities: dict[str, str],
    max_pairs: int = 500_000,
) -> list[InputExample]:
    """Build contrastive training pairs from ground truth.
    
    For each S1 entity, create positive pairs with each matched S2/S3 entity.
    MultipleNegativesRankingLoss uses in-batch negatives automatically.
    """
    target_entities = {**s2_entities, **s3_entities}
    pairs: list[InputExample] = []

    for chunk in iter_ground_truth_chunks(gt_path, chunk_size=100_000):
        for s1_eid, matched_ids_str in chunk.itertuples(index=False, name=None):
            if not matched_ids_str or pd.isna(matched_ids_str):
                continue
            s1_text = s1_entities.get(s1_eid, "")
            if not s1_text:
                continue

            matched_ids = str(matched_ids_str).split(",")
            for target_eid in matched_ids:
                target_eid = target_eid.strip()
                target_text = target_entities.get(target_eid, "")
                if not target_text:
                    continue
                pairs.append(InputExample(texts=[s1_text, target_text]))

                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    print(f"Built {len(pairs):,} training pairs")
    return pairs


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default="BAAI/bge-small-en-v1.5",
        help="Pre-trained sentence-transformer to fine-tune"
    )
    parser.add_argument("--s1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--output", default="models/finetuned_biencoder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=500_000,
                        help="Cap training pairs (contrastive learning)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 1. Load entities
    print("Loading S1 entities...")
    s1 = load_entities(args.s1)
    print(f"  S1: {len(s1):,} entities")

    print("Loading S2 entities...")
    s2 = load_entities(args.s2)
    print(f"  S2: {len(s2):,} entities")

    print("Loading S3 entities...")
    s3 = load_entities(args.s3)
    print(f"  S3: {len(s3):,} entities")

    # 2. Build training pairs
    print("Building training pairs from ground truth...")
    pairs = build_training_pairs(args.gt, s1, s2, s3, max_pairs=args.max_pairs)

    # Free memory
    del s1, s2, s3

    # 3. Load model
    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model, device=device)

    # 4. Set up training
    train_dataloader = DataLoader(pairs, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    print(f"Training: {len(pairs):,} pairs, {args.epochs} epochs, "
          f"batch_size={args.batch_size}, warmup={warmup_steps} steps")

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.output,
    )
    elapsed = time.perf_counter() - t0
    print(f"Fine-tuning complete in {elapsed/60:.1f} minutes")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
