#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE1="${1:-dataset/test_source1.tsv}"
RUN_DIR="${2:-runs/full_v6}"

python3 -m src.merge --source1 "$SOURCE1" --run-dir "$RUN_DIR"
