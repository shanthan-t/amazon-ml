# Runtime Artifacts — External Files Not Stored in Git

These artifacts are required for inference but are too large for Git.
They must be copied to the expected relative paths before running.

## Retrieval Indexes

### v2_test_index.sqlite3
- **Purpose:** Main target posting index (blocking keys → target indexes)
- **Expected path:** `artifacts/v2_test_index.sqlite3`
- **Size:** 4,843,343,872 bytes (4.51 GiB)
- **SHA-256:** `7a730cfb910899af13ec6ddb394b332b59672a6c90d82b67df53c9713f15d716`

### phase2_test_numeric_address.sqlite3
- **Purpose:** Numeric-address blocking route index (AN| keys)
- **Expected path:** `artifacts/phase2_test_numeric_address.sqlite3`
- **Size:** 317,624,320 bytes (303 MiB)
- **SHA-256:** `428100505ca0fc171587b12e1b31cc6a403fbae44401edb62cfc6f3469a4ed10`

### test_address_index.sqlite3
- **Purpose:** Rare-address overlap retrieval index (address tokens → targets)
- **Expected path:** `artifacts/test_address_index.sqlite3`
- **Size:** 2,950,680,576 bytes (2.75 GiB)
- **SHA-256:** `57e7b91a53d912f367d548bd446d61d52718188a6d67592d50ff737862537c8b`

## Target Store

### test_target_store/offsets.npy
- **Purpose:** Byte offset array for random-access target record lookup
- **Expected path:** `artifacts/test_target_store/offsets.npy`
- **Size:** 79,756,848 bytes (76 MiB)
- **SHA-256:** `4091275db13a455decf3bb69d125afbf3f556e5d5b31711d27060d682dff7c83`

### test_target_store/records.tsv
- **Purpose:** Memory-mapped target records (entity_id, name, address, country, source)
- **Expected path:** `artifacts/test_target_store/records.tsv`
- **Size:** 998,458,970 bytes (952 MiB)
- **SHA-256:** `8b14da4d425601ab5c35e98264b2c3c56c308a2796cf9da3b494be5062189614`

### test_target_store/manifest.json
- **Purpose:** Target store integrity manifest
- **Expected path:** `artifacts/test_target_store/manifest.json`
- **Size:** 267 bytes
- **SHA-256:** `eef938379e64a8617892d1c934ed2ea303d9f7fb52b02d09463d9a743ccad0da`

## ICU Runtime

### ICU4C 77.1 Windows DLLs
- **Purpose:** ICU transliteration engine (`Any-Latin; Latin-ASCII`)
- **Expected path:** `native/icu/icudt77.dll`, `native/icu/icuuc77.dll`, `native/icu/icuin77.dll`
- **Source:** Official ICU4C 77.1 Win64-MSVC2022 release
- **Archive SHA-256:** `6b62471ed2895959d6a85c64c58572ac677734547fe8383e6e3fd706a06dc3fa`
- **Automated:** `setup_windows.ps1` downloads and extracts these automatically

## Competition Input

### test_source1.tsv
- **Purpose:** Competition test input (Source 1 entities)
- **Expected path:** User-supplied path passed as `--source1`
- **Size:** 175,022,086 bytes (167 MiB)
- **SHA-256:** `3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5`
- **Note:** Do not store in the repository

## Models (In Git)

The four fold models (~8.5 MB total) ARE stored in Git:

| File | Size | SHA-256 |
|------|------|---------|
| `models/B_cross_script/fold0.ubj` | 2,125,693 | `7bb56d6fe0964b4c51aa1922d1ad4b8034586f2e41db5d876c85d80d625bc605` |
| `models/B_cross_script/fold1.ubj` | 2,130,731 | `19fea195aa2ccb3875d1e307f7cf1dd858162cc242e088759acee9f7df710ff3` |
| `models/B_cross_script/fold2.ubj` | 2,126,376 | `9bc18f713e5f9faa5ce3f3a0aed4160e7d1d93f3cae79169324daed08331c0e6` |
| `models/B_cross_script/fold3.ubj` | 2,146,913 | `8e21913e605c3d9af3eef493cce78ea4280d9756305fd427001897dad8ea6ef7` |

## Total External Artifact Size

~8.58 GiB (retrieval indexes + target store, excluding ICU DLLs and competition input)
