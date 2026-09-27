# Amazon ML Challenge - Final V6 Pipeline

This repository contains the canonical, finalized V6 pipeline that produced our validated submission. It includes all code required to build indexes, generate features, train the final four-fold ensemble, and run highly scalable parallel inference.

## Architecture

Our approach completely replaces earlier Deep Learning/Transformer embeddings with a robust, highly optimized heuristic and Gradient Boosting architecture.

- **Candidate Generation**: SQLite-based blocking using precise routes (Prefix-6, Token Sort, Word Tokens, Numeric Address, and Rare Address Overlap).
- **Features (39D)**: 39 hand-crafted floating point features comparing strings, numbers, lengths, missing data patterns, and cross-script properties.
- **Cross-Script Handling**: We pin ICU 77.1 for deterministic `Any-Latin; Latin-ASCII` transliteration, enabling accurate matching of non-Latin business names.
- **Ensemble Model**: Four XGBoost models trained on disjoint entity folds using out-of-core memory chunking.
- **Scoring**: Averages probabilities across the 4 models, applying a global threshold of `0.98` and a stricter numeric-conflict threshold of `0.99`. Max matches capped at 11.

## Repository Structure

- `src/` - Core runtime logic (retrieval, feature extraction, normalization).
- `training/` - Scripts for out-of-core XGBoost training and evaluation.
- `indexing/` - Scripts to build the required SQLite target indexes.
- `models/` - The final 4 `.ubj` XGBoost model files.
- `native/icu/` - Pinned ICU libraries for cross-platform deterministic transliteration.
- `scripts/` - Easy-to-use PowerShell wrappers for Windows execution.

## Reproduction Steps

1. **Environment Setup**:
   `pip install -r requirements.txt`
   
2. **Build Training Indexes** (if retraining):
   `python -m indexing.build_target_index`
   `python -m indexing.build_address_index`
   
3. **Train Models**:
   `python -m training.train`
   
4. **Run Inference**:
   Use `scripts/run_windows.ps1` for local inference.

Note: Obsolete experiments (DL pipelines, V5 heuristics, FAISS vector search, SageMaker deployment) have been completely removed from this branch to ensure clarity and strict reproducibility of the winning architecture.
