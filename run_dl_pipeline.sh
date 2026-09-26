#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Amazon ML Challenge — Deep Learning Pipeline"
echo " (Optimized for High-Spec PC with NVIDIA GPU)"
echo "=========================================================="

mkdir -p models output_dl reports

# Default models
BASE_MODEL="BAAI/bge-small-en-v1.5"
FINETUNED_MODEL="models/finetuned_biencoder"

if [ "$1" == "--finetune" ]; then
    echo "[1/2] Fine-Tuning Bi-Encoder (Contrastive Learning)..."
    PYTHONPATH=. python3 -m src.gpu_finetune_biencoder \
        --base-model "$BASE_MODEL" \
        --output "$FINETUNED_MODEL" \
        --epochs 3 --batch-size 256
    
    MODEL_TO_USE="$FINETUNED_MODEL"
else
    echo "Skipping fine-tuning. To fine-tune, run: ./run_dl_pipeline.sh --finetune"
    MODEL_TO_USE="$BASE_MODEL"
fi

echo "[2/2] Running Dense Vector Matching (FAISS)..."
PYTHONPATH=. python3 -m src.gpu_dense_matcher \
    --model "$MODEL_TO_USE" \
    --output "output_dl/matching_results.tsv" \
    --report "reports/dl_inference.json" \
    --batch-size 1024 \
    --threshold 0.85

echo "=========================================================="
echo " Deep Learning Pipeline Complete!"
echo " Submission file ready at: output_dl/matching_results.tsv"
echo "=========================================================="
