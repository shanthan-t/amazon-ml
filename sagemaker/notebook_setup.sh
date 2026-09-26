#!/bin/bash
# Run this on your LOCAL machine to generate presigned URLs,
# then paste the output into the SageMaker notebook terminal.
#
# Usage: bash sagemaker/notebook_setup.sh

set -e

BUCKET="amazonml-rex"
PREFIX="amazon-ml/runs/p6-test-v2"
EXPIRY=7200  # 2 hours

echo "============================================================"
echo "PASTE EVERYTHING BELOW INTO THE NOTEBOOK TERMINAL"
echo "============================================================"
echo ""
echo "# --- Step 1: Create directories ---"
echo "mkdir -p ~/SageMaker/input/{code/src,code/sagemaker,data,indexes,model}"
echo "mkdir -p ~/SageMaker/output"
echo ""

# Generate wget commands with presigned URLs
echo "# --- Step 2: Download files from S3 (presigned, no IAM needed) ---"

FILES=(
    "input/code/src/__init__.py"
    "input/code/src/predict_test.py"
    "input/code/src/features.py"
    "input/code/src/normalization.py"
    "input/code/src/data_loader.py"
    "input/code/src/blocking.py"
    "input/code/sagemaker/requirements_inference.txt"
    "input/code/sagemaker/run_test_inference.py"
    "input/data/test_source1.tsv"
    "input/indexes/phase2_test_exact_index.sqlite3"
    "input/indexes/phase2_test_p6_routes.sqlite3"
    "input/indexes/phase2_test_names.sqlite3"
    "input/model/logistic_p6_exact_candidate.json"
)

for f in "${FILES[@]}"; do
    URL=$(aws s3 presign "s3://${BUCKET}/${PREFIX}/${f}" --expires-in $EXPIRY 2>/dev/null)
    echo "wget -q -O ~/SageMaker/${f} '${URL}'"
done

echo ""
echo "# --- Step 3: Install compatible dependencies ---"
echo "# (notebook runs Python 3.9/3.10, need older scipy)"
echo 'pip install -q "pandas>=2.2,<2.4" "rapidfuzz>=3.0,<4" "scipy>=1.11,<1.16" "numpy>=1.26,<2.5"'
echo ""
echo "# --- Step 4: Run inference ---"
echo "cd ~/SageMaker/input/code"
echo "nohup python3 -m src.predict_test \\"
echo "    --source1 ~/SageMaker/input/data/test_source1.tsv \\"
echo "    --index ~/SageMaker/input/indexes/phase2_test_exact_index.sqlite3 \\"
echo "    --model ~/SageMaker/input/model/logistic_p6_exact_candidate.json \\"
echo "    --matching ~/SageMaker/output/matching_results.tsv \\"
echo "    --candidate ~/SageMaker/output/candidate_pairs.tsv \\"
echo "    --report ~/SageMaker/output/test_prediction_v2.json \\"
echo "    --p6-index ~/SageMaker/input/indexes/phase2_test_p6_routes.sqlite3 \\"
echo "    --name-index ~/SageMaker/input/indexes/phase2_test_names.sqlite3 \\"
echo "    > ~/SageMaker/inference.log 2>&1 &"
echo ""
echo "echo 'Job started in background. Monitor with:'"
echo "echo '  tail -f ~/SageMaker/inference.log'"
echo ""
echo "============================================================"
