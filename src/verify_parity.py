"""Verify Windows runtime against frozen Linux outputs; exit nonzero on any mismatch."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import xgboost as xgb

from .config import (ADDRESS_INDEX, CONFIG_ID, INDEX, MODEL_PATHS, MODEL_SHA256,
                     NUMERIC_INDEX, TARGET_STORE)
from .features import FEATURE_NAMES, normalize
from .optimized_features import OptimizedFeatureEngine
from .policy import choose
from .target_store import MappedRetriever

FIXTURE = Path(__file__).resolve().parents[1] / "parity_fixture" / "expected_results.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if fixture["configuration_id"] != CONFIG_ID or tuple(fixture["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Fixture configuration or feature order differs")
    for path, expected in zip(MODEL_PATHS, MODEL_SHA256):
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Model checksum mismatch: {path}")
    source = fixture["source1"]
    query = normalize(source["business_name"], source["business_address"], source["country"])
    if list(query) != fixture["normalized_source1"]:
        raise ValueError("Normalization parity failed")

    retriever = MappedRetriever(INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE)
    engine = OptimizedFeatureEngine(cache_size=0)
    try:
        candidates, rare, targets = retriever.retrieve(query)
        candidate_ids = [targets[idx][0] for idx in candidates]
        if candidate_ids != fixture["candidate_generation"]["candidate_ids_in_output_order"]:
            raise ValueError("Candidate generation/order parity failed")
        baseline_ids = [targets[idx][0] for idx in candidates if idx not in rare]
        rare_ids = [targets[idx][0] for idx in candidates if idx in rare]
        expected_routes = fixture["candidate_generation"]
        if baseline_ids != expected_routes["baseline_candidate_ids"]:
            raise ValueError("Base retrieval route parity failed")
        if rare_ids != expected_routes["rare_address_top25_added_ids"] or len(rare_ids) > 25:
            raise ValueError("Bounded rare_address_overlap/top_25 parity failed")

        matrix = engine.matrix(query, candidates, targets, rare)
        expected_rows = fixture["candidate_expectations"]
        expected_matrix = np.asarray([item["features_float32"] for item in expected_rows], dtype=np.float32)
        if not np.array_equal(matrix, expected_matrix):
            delta = float(np.max(np.abs(matrix - expected_matrix))) if matrix.size else 0.0
            raise ValueError(f"39-feature float32 parity failed (max abs diff {delta})")
        if len(matrix[0]) != 39 or not np.isfinite(matrix).all():
            raise ValueError("Feature schema or finite-value check failed")

        transliterations = {item["target_id"]: (item["query_transliteration"], item["target_transliteration"])
                            for item in fixture["icu_expected"]}
        for idx in candidates:
            target_id, target_name = targets[idx][0], targets[idx][1]
            observed = (engine.transliterate(query[0]), engine.transliterate(target_name))
            if observed != tuple(transliterations[target_id]):
                raise ValueError(f"ICU Any-Latin; Latin-ASCII parity failed for {target_id}")

        dmatrix = xgb.DMatrix(matrix, feature_names=FEATURE_NAMES, nthread=1)
        models = [xgb.Booster(model_file=str(path)) for path in MODEL_PATHS]
        for model in models:
            if model.num_features() != 39:
                raise ValueError("Expected exactly 39 model features")
            model.set_param({"nthread": 1})
        per_model = np.stack([model.predict(dmatrix) for model in models])
        expected_models = np.asarray([item["four_model_probabilities"] for item in expected_rows], dtype=np.float32).T
        if not np.allclose(per_model, expected_models, rtol=0, atol=1e-6):
            raise ValueError("Four-model individual score parity failed")
        ensemble = np.mean(per_model, axis=0, dtype=np.float32)
        expected_ensemble = np.asarray([item["ensemble_probability_float32"] for item in expected_rows], dtype=np.float32)
        if not np.allclose(ensemble, expected_ensemble, rtol=0, atol=1e-6):
            raise ValueError("Four-model averaged score parity failed")
        selected = choose(candidates, matrix, ensemble)
        observed_selected = [targets[idx][0] for idx in selected]
        if observed_selected != fixture["expected_selected_target_ids"]:
            raise ValueError("Frozen 0.98/0.99 policy or max_matches=11 parity failed")
        print(json.dumps({"status": "PASS", "fixture": FIXTURE.name,
                          "candidates": len(candidates), "rare_address_top25": len(rare),
                          "features": 39, "models": 4, "selected_matches": len(selected),
                          "configuration_id": CONFIG_ID}, indent=2))
    finally:
        engine.close()
        retriever.close()


if __name__ == "__main__":
    try:
        verify()
    except Exception as exc:
        print(f"PARITY FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
