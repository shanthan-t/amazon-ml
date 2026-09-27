"""One fixed-policy development sanity check for the four approved fold models."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.audit import RUN, load_dev
from src.features import FEATURE_NAMES
from src.optimized_features import OptimizedFeatureEngine
from src.policy import choose
from src.target_store import MappedRetriever
from v6_macro_f05.audit_baseline import entity_score, summarize

ROOT = Path('v6_macro_f05')
MODEL_DIR = ROOT / 'final_focused_v1/models/B_cross_script'
MODEL_PATHS = [MODEL_DIR / f'fold{i}.ubj' for i in range(4)]


def run(limit=None):
    start = time.perf_counter()
    dev, expected, eids, _ = load_dev()
    if limit is not None:
        eids = eids[:limit]
    ret = MappedRetriever('v2_train_index.sqlite3', 'phase2_train_numeric_address.sqlite3',
                          RUN / 'train_address_index.sqlite3', RUN / 'train_target_store')
    engine = OptimizedFeatureEngine()
    matrices, candidates, entities = [], [], []
    for i, eid in enumerate(eids):
        q = dev[eid]['query']
        cs, rare, targets = ret.retrieve(q)
        assert cs == sorted(expected[eid])
        xx = engine.matrix(q, cs, targets, rare)
        assert xx.dtype == np.float32 and xx.shape == (len(cs), 39)
        matrices.append(xx)
        candidates.extend(cs)
        entities.append((eid, len(cs), targets, dev[eid]['truth'], q[2]))
        if (i + 1) % 500 == 0:
            print('ensemble sanity', i + 1, 'S1s', 'candidates', len(candidates), flush=True)
    x = np.concatenate(matrices)
    assert x.shape == (len(candidates), len(FEATURE_NAMES))
    dmat = xgb.DMatrix(x, feature_names=FEATURE_NAMES, nthread=4)
    models = []
    for path in MODEL_PATHS:
        model = xgb.Booster(model_file=str(path)); model.set_param({'nthread': 4})
        assert model.num_features() == 39
        models.append(model)
    per_model = [m.predict(dmat) for m in models]
    scores = np.mean(np.stack(per_model), axis=0, dtype=np.float32)
    metrics, offset = [], 0
    fp = fn = candidate_count = 0
    for (eid, count, targets, truth, country), xx in zip(entities, matrices):
        cs = candidates[offset:offset + count]
        pp = scores[offset:offset + count]
        selected = choose(cs, xx, pp)
        gt = set(truth)
        predicted = {targets[i][0] for i in selected}
        candidate_ids = {targets[i][0] for i in cs}
        metrics.append({'eid': eid, 'country': country, 'gt': len(gt),
                        'tp': len(gt & predicted), 'fp': len(predicted - gt),
                        'predicted': len(predicted), 'score': entity_score(gt, predicted),
                        'candidates': len(cs), 'retrieved': len(gt & candidate_ids),
                        'ceiling': entity_score(gt, gt & candidate_ids)})
        fp += len(predicted - gt); fn += len(gt - predicted); candidate_count += len(cs)
        offset += count
    result = {
        'purpose': 'fixed 0.98/0.99 operational sanity only; no tuning; development entities are not independent because each fold model trained on three folds',
        'model_paths': [str(p) for p in MODEL_PATHS],
        'model_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL_PATHS],
        'entities': len(eids), 'candidates': candidate_count,
        'average_rule': 'arithmetic mean of the four per-model float32 positive probabilities',
        'policy': {'global_threshold': 0.98, 'numeric_conflict_threshold': 0.99, 'max_matches': 11},
        'ensemble_development_metrics': summarize(metrics),
        'macro_f05': float(np.mean([m['score'] for m in metrics])),
        'fp': fp, 'fn': fn,
        'historical_oof': {'macro_f05': 0.916170, 'precision': 0.97629,
                           'recall': 0.83665, 'singleton_accuracy': 0.92105,
                           'non_singleton_macro_f05': 0.915926, 'fp': 285, 'fn': 2291},
        'wall_seconds': time.perf_counter() - start,
        'peak_rss_mib': __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 1024,
    }
    (RUN / 'ensemble_sanity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    engine.close(); ret.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int)
    run(p.parse_args().limit)
