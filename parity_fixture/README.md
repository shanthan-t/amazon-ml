# Linux-generated parity fixture

`expected_results.json` was produced by the current Linux V6 urgent-inference modules and checked against the active worker's candidate output. It contains one deterministic test-source row and 32 scored candidates, including 25 rare-address additions. It records normalization, route membership, all 39 float32 features, ICU outputs, each fold-model score, the averaged score, and final selected IDs. It contains no labels or holdout data.

Run `python -m src.verify_parity` after the copied indexes and target store are in place. A different ICU result, candidate set, feature, model score, or policy output fails loudly.
