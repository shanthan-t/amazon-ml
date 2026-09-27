"""Bounded, monitored XGBoost external-memory resource scaling test."""
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import xgboost as xgb

from src.features import FEATURE_NAMES
from src.validate_extmem import cache_bytes

RUN = Path('v6_macro_f05/production_run_v2')


class Shards(xgb.DataIter):
    def __init__(self, prefixes, cache):
        self.prefixes, self.i = [str(x) for x in prefixes], 0
        super().__init__(cache_prefix=str(cache), release_data=True)

    def reset(self):
        self.i = 0

    def next(self, input_data):
        if self.i >= len(self.prefixes):
            return False
        prefix = self.prefixes[self.i]
        self.i += 1
        x = np.load(prefix + '_X.npy', mmap_mode='r')
        y = np.load(prefix + '_y.npy', mmap_mode='r')
        input_data(data=x, label=y, feature_names=FEATURE_NAMES)
        return True


def run(shard_count=10, rounds=100):
    prefixes = [RUN / 'training_shards' / f'{i:05d}' for i in range(shard_count)]
    for p in prefixes:
        if not Path(str(p) + '_X.npy').exists() or not Path(str(p) + '_y.npy').exists():
            raise FileNotFoundError(p)
    xs = [np.load(str(p) + '_X.npy', mmap_mode='r') for p in prefixes]
    ys = [np.load(str(p) + '_y.npy', mmap_mode='r') for p in prefixes]
    rows = sum(len(y) for y in ys)
    positives = sum(int(y.sum()) for y in ys)
    negatives = rows - positives
    del xs, ys

    work = RUN / 'extmem_scale_v1'
    cache = work / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            samples.append({
                'time': time.time(), 'rss_bytes': process.memory_info().rss,
                'system_available_bytes': vm.available, 'swap_used_bytes': sw.used,
                'cache_bytes': cache_bytes(cache),
            })
            stop.wait(.25)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    start = time.perf_counter()
    iterator = Shards(prefixes, cache / 'train')
    matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=256, nthread=4)
    matrix_seconds = time.perf_counter() - start
    assert matrix.num_row() == rows and matrix.num_col() == len(FEATURE_NAMES)
    params = {
        'max_depth': 7, 'eta': 0.08, 'min_child_weight': 10,
        'subsample': 0.85, 'colsample_bytree': 0.9, 'lambda': 2.0,
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'tree_method': 'hist', 'max_bin': 256, 'seed': 42,
        'nthread': 4, 'scale_pos_weight': negatives / positives,
        'device': 'cpu',
    }
    fit_start = time.perf_counter()
    model = xgb.train(params, matrix, num_boost_round=rounds)
    fit_seconds = time.perf_counter() - fit_start
    model.save_model(work / 'bounded_model.ubj')
    stop.set()
    monitor_thread.join()
    swap_before = samples[0]['swap_used_bytes']
    swap_after = samples[-1]['swap_used_bytes']
    result = {
        'xgboost_version': xgb.__version__,
        'api': 'ExtMemQuantileDMatrix(DataIter)', 'device': 'cpu',
        'shards': shard_count, 'rows': rows, 'features': len(FEATURE_NAMES),
        'positives': positives, 'negatives': negatives, 'rounds': rounds,
        'matrix_build_seconds': matrix_seconds, 'fit_seconds': fit_seconds,
        'seconds_per_tree': fit_seconds / rounds,
        'rows_per_second_per_tree': rows / (fit_seconds / rounds),
        'peak_rss_mib': max(s['rss_bytes'] for s in samples) / 1024**2,
        'peak_system_available_gib': min(s['system_available_bytes'] for s in samples) / 1024**3,
        'swap_used_before_gib': swap_before / 1024**3,
        'swap_used_after_gib': swap_after / 1024**3,
        'swap_growth_mib': (swap_after - swap_before) / 1024**2,
        'cache_bytes_after_fit': cache_bytes(cache),
        'free_disk_gib_after_fit': __import__('shutil').disk_usage(RUN).free / 1024**3,
        'parameters': params,
        'safety_limits': {'min_available_ram_gib': 1.5, 'max_swap_growth_mib': 512, 'min_free_disk_gib': 300},
        'limits_passed': min(s['system_available_bytes'] for s in samples) >= 1.5 * 1024**3 and (swap_after - swap_before) <= 512 * 1024**2 and __import__('shutil').disk_usage(RUN).free >= 300 * 1024**3,
        'samples': len(samples),
        'peak_rss_mib_resource': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (work / 'resource_profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()
    run(args.shards, args.rounds)
