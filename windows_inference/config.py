"""Frozen V6 inference paths and model checksums."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("V6_ARTIFACTS", ROOT / "artifacts")).resolve()
INDEX = ARTIFACTS / "v2_test_index.sqlite3"
NUMERIC_INDEX = ARTIFACTS / "phase2_test_numeric_address.sqlite3"
ADDRESS_INDEX = ARTIFACTS / "test_address_index.sqlite3"
TARGET_STORE = ARTIFACTS / "test_target_store"
MODEL_DIR = ROOT / "models" / "B_cross_script"
MODEL_PATHS = [MODEL_DIR / f"fold{i}.ubj" for i in range(4)]
MODEL_SHA256 = [
    "7bb56d6fe0964b4c51aa1922d1ad4b8034586f2e41db5d876c85d80d625bc605",
    "19fea195aa2ccb3875d1e307f7cf1dd858162cc242e088759acee9f7df710ff3",
    "9bc18f713e5f9faa5ce3f3a0aed4160e7d1d93f3cae79169324daed08331c0e6",
    "8e21913e605c3d9af3eef493cce78ea4280d9756305fd427001897dad8ea6ef7",
]
CONFIG_ID = "v6-top25-xgb39-icu-global98-20260927-v1"
GLOBAL_THRESHOLD = 0.98
NUMERIC_CONFLICT_THRESHOLD = 0.99
MAX_MATCHES = 11
TOP_RARE_ADDRESS = 25
