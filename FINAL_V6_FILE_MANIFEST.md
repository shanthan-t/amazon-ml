# Final V6 File Manifest

| Path | Purpose | Category | Why Retained |
|---|---|---|---|
| `src/` | Final cross-platform runtime for candidate generation and scoring | Runtime | Canonical implementation of the V6 pipeline |
| `src/icu.py` | ICU library bindings for transliteration | Runtime | Required for 39th cross-script feature |
| `training/train.py` | Trains the 4-fold XGBoost models | Training | Reproducibility of final models |
| `training/prepare_training.py` | Sets up external memory matrices | Training | Reproducibility |
| `indexing/build_address_index.py` | Builds numeric address indexes | Index Build | Required to generate runtime indexes |
| `indexing/build_target_index.py` | Builds core sqlite targets | Index Build | Required to generate runtime indexes |
| `models/B_cross_script/` | Final 4 serialized XGBoost models | Model | The exact models used in submission |
| `native/icu/` | ICU 77.1 pinned libraries | Runtime | Required by `icu.py` |
| `scripts/` | PowerShell scripts for execution | Execution | Simplifies running on Windows |
| `requirements.txt` | Minimal dependencies | Setup | Reproducible environment |
