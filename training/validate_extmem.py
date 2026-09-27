"""Small, deterministic in-memory vs XGBoost external-memory training audit."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.features import FEATURE_NAMES

ROOT = Path('v6_macro_f05/production_run_v2')


class OneShard(xgb.DataIter):
    def __init__(self, x_path, y_path, cache):
        self.x_path, self.y_path, self.done = str(x_path), str(y_path), False
        super().__init__(cache_prefix=str(cache), release_data=True)

    def reset(self):
        self.done = False

    def next(self, input_data):
        if self.done:
            return False
        self.done = True
        x = np.load(self.x_path, mmap_mode='r')
        y = np.load(self.y_path, mmap_mode='r')
        input_data(data=x, label=y, feature_names=FEATURE_NAMES)
        return True


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).glob('*') if p.is_file())


def run(shard_index=0, rounds=500):
    prefix = ROOT / 'training_shards' / f'{shard_index:05d}'
    source_x = np.load(str(prefix) + '_X.npy', mmap_mode='r')
    source_y = np.load(str(prefix) + '_y.npy', mmap_mode='r')
    # Deterministic row split. Both trainers receive byte-identical train rows.
    val_mask = np.arange(len(source_y)) % 5 == 0
    train_mask = ~val_mask
    work = ROOT / 'extmem_parity_v1'
    data = work / 'data'
    cache = work / 'cache'
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    x_train_path, y_train_path = data / 'X_train.npy', data / 'y_train.npy'
    np.save(x_train_path, np.asarray(source_x[train_mask], dtype=np.float32))
    np.save(y_train_path, np.asarray(source_y[train_mask], dtype=np.uint8))
    x_train = np.load(x_train_path, mmap_mode='r')
    y_train = np.load(y_train_path, mmap_mode='r')
    x_val = np.asarray(source_x[val_mask], dtype=np.float32)
    y_val = np.asarray(source_y[val_mask], dtype=np.uint8)
    assert x_train.shape[1] == len(FEATURE_NAMES)
    assert x_train.dtype == np.float32 and y_train.dtype == np.uint8

    negatives = int((y_train == 0).sum())
    positives = int(y_train.sum())
    params = {
        'max_depth': 7, 'eta': 0.08, 'min_child_weight': 10,
        'subsample': 0.85, 'colsample_bytree': 0.9, 'lambda': 2.0,
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'tree_method': 'hist', 'max_bin': 256, 'seed': 42,
        'nthread': 4, 'scale_pos_weight': negatives / positives,
        'device': 'cpu',
    }
    val = xgb.DMatrix(x_val, label=y_val, feature_names=FEATURE_NAMES, nthread=4)
    results = {
        'xgboost_version': xgb.__version__, 'api': 'ExtMemQuantileDMatrix(DataIter)',
        'source_shard': str(prefix), 'source_shard_sha256': {
            'X': sha(str(prefix) + '_X.npy'), 'y': sha(str(prefix) + '_y.npy')},
        'source_rows': int(len(source_y)), 'train_rows': int(len(y_train)),
        'validation_rows': int(len(y_val)), 'positive_rows': positives,
        'negative_rows': negatives, 'validation_positive_rows': int(y_val.sum()),
        'feature_names': FEATURE_NAMES, 'feature_count': int(x_train.shape[1]),
        'feature_dtype': str(x_train.dtype), 'label_dtype': str(y_train.dtype),
        'parameters': params, 'rounds': rounds,
        'in_memory': {}, 'external_memory': {},
    }

    start = time.perf_counter()
    memory = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_NAMES, nthread=4)
    model_a = xgb.train(params, memory, num_boost_round=rounds)
    results['in_memory'] = {
        'wall_seconds': time.perf_counter() - start,
        'peak_rss_mib_process': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'prediction_sha256': hashlib.sha256(model_a.predict(val).tobytes()).hexdigest(),
    }
    model_a.save_model(work / 'in_memory.ubj')
    del memory, model_a

    start = time.perf_counter()
    iterator = OneShard(x_train_path, y_train_path, cache / 'train')
    external = xgb.ExtMemQuantileDMatrix(iterator, max_bin=256, nthread=4)
    results['external_memory']['matrix_rows'] = int(external.num_row())
    results['external_memory']['matrix_columns'] = int(external.num_col())
    results['external_memory']['cache_bytes_after_matrix'] = cache_bytes(cache)
    model_b = xgb.train(params, external, num_boost_round=rounds)
    pred_b = model_b.predict(val)
    pred_a = xgb.Booster({'nthread': 4}, model_file=work / 'in_memory.ubj').predict(val)
    corr = float(np.corrcoef(pred_a, pred_b)[0, 1])
    results['external_memory'].update({
        'wall_seconds_including_matrix': time.perf_counter() - start,
        'cache_bytes_after_fit': cache_bytes(cache),
        'peak_rss_mib_process': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'prediction_sha256': hashlib.sha256(pred_b.tobytes()).hexdigest(),
        'prediction_correlation': corr,
        'max_abs_prediction_difference': float(np.max(np.abs(pred_a - pred_b))),
        'mean_abs_prediction_difference': float(np.mean(np.abs(pred_a - pred_b))),
        'feature_names': model_b.feature_names,
        'tree_count': model_b.num_boosted_rounds(),
    })
    model_b.save_model(work / 'external_memory.ubj')
    results['in_memory']['feature_names'] = FEATURE_NAMES
    results['in_memory']['tree_count'] = rounds
    results['runtime_equal_parameters'] = True
    results['entity_level_prediction_comparison'] = 'not available: shard stores aggregate entity/candidate hashes, not per-row S1 grouping offsets'
    results['peak_rss_mib_cumulative_process'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (work / 'parity.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=500)
    args = parser.parse_args()
    run(args.shard_index, args.rounds)
