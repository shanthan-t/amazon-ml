# Amazon ML Challenge — Entity Resolution & Matching Pipeline

High-throughput, parallelized entity matching pipeline designed for massive enterprise entity linkage (millions of multi-source business records).

This repository is optimized for high-performance workstations (multi-core CPUs + NVMe storage + optional NVIDIA GPUs).

---

## 🚀 Key Highlights & Architecture

* **Multi-Route Inverted Index Blocking**: Exact Name, Prefix-6, Prefix-4, Token Sort, Word Tokens, and Numeric Street Address sidecars.
* **26-Dimensional Pairwise Feature Extractor**: Levenshtein token ratios, Jaccard similarities, Soundex phonetic agreement, address digit matching, source indicators, and regional adaptation flags (`is_india`, `target_is_s2`, `name_is_short`).
* **Calibrated XGBoost Re-Ranker**: Evaluates candidates per entity, filters against official Macro $F_{0.5}$ thresholds, and outputs ranked matches formatted for submission.
* **High-Throughput Parallel Engine**: Batched SQL retrieval in the main process paired with multi-worker concurrent feature computation and inference across all CPU cores.

---

## 📁 Repository Structure

```
├── build_indices.sh          # One-click script to build SQLite blocking indices from TSVs
├── run_predict_parallel.sh   # High-throughput multi-worker test prediction script
├── run_train_parallel.sh     # Multi-worker model training script
├── requirements.txt          # Python dependencies (CPU/Standard)
├── requirements_gpu.txt      # Python dependencies with GPU support (CUDA / XGBoost / PyTorch)
├── src/                      # Core modules
│   ├── normalization.py      # Unicode, punctuation, business suffix cleaner
│   ├── v2_common.py          # 26D feature engineering & blocking key generators
│   ├── v2_train_parallel.py  # Parallel training pipeline
│   ├── v2_predict_parallel.py# Parallel inference pipeline
│   ├── test_numeric_address_index.py # Address sidecar index builder
│   └── ...
├── models/                   # Pretrained models
│   ├── xgb_v4.json           # Best tuned XGBoost model (Macro F0.5 = 0.858+)
│   └── xgb_v4.xgb            # Binary booster weights
├── benchmarks/               # Diagnostic & validation scripts
└── sagemaker/                # AWS SageMaker cloud execution scripts
```

---

## ⚡ Quickstart on a High-Spec Machine

### 1. Environment Setup

Create an isolated virtual environment and install dependencies:

```bash
# Recommended Python: 3.10, 3.11, or 3.12
python3 -m venv .venv
source .venv/bin/activate

# For CPU execution:
pip install -r requirements.txt

# OR for NVIDIA GPU / PyTorch execution:
pip install -r requirements_gpu.txt
```

### 2. Dataset Setup

Place your dataset inside `dataset/` following this structure:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

> **Note:** The `dataset/` directory and large SQLite files are automatically ignored by `.gitignore` so you never exceed GitHub file size limits.

### 3. Database Indexes

You need the SQLite databases to run blocking:

* **Fast Option (Recommended if transferring from old machine):** Copy the existing `.sqlite3` files directly via local network (`rsync`/`scp`) or USB drive:
  - `v2_test_index.sqlite3`
  - `phase2_test_numeric_address.sqlite3`
  - `v2_train_index.sqlite3` (if training)
  - `phase2_train_numeric_address.sqlite3` (if training)

* **Build from Scratch (Option B):**
  Run the automated index builder:
  ```bash
  ./build_indices.sh
  ```

---

## 🔮 Running Test Predictions (Generate Submission)

To run parallel scoring across all CPU cores on your machine:

```bash
./run_predict_parallel.sh
```

**Customizing Core Count or Paths:**
```bash
# Example: Use 16 workers on a 16-core / 32-thread machine
WORKERS=16 ./run_predict_parallel.sh
```

The submission TSV will be saved directly to:
```
output_submission/matching_results.tsv
```

---

## 🎯 Retraining the Model

To train the XGBoost model with custom sample sizes:

```bash
# Trains on 200,000 ground truth rows using all available cores
./run_train_parallel.sh
```

---

## 📤 How to Push This Repository to GitHub

From inside this folder:

```bash
# 1. Initialize git repository
git init -b main

# 2. Add all clean files (.gitignore protects datasets & databases)
git add .

# 3. Create initial commit
git commit -m "Initial commit: Amazon ML Challenge parallel matching pipeline"

# 4. Link your GitHub remote
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 5. Push to GitHub
git push -u origin main
```
