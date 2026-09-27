"""Spawn-safe worker for one deterministic, disjoint S1 shard."""
import csv, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import psutil
import xgboost as xgb
from .features import FEATURE_NAMES, normalize
from .optimized_features import OptimizedFeatureEngine
from .policy import OutputWriter, choose
from .target_store import MappedRetriever
from .config import INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE, MODEL_PATHS, CONFIG_ID

def atomic(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def rows(path):
    with open(path,encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames != ['entity_id','business_name','business_address','country']:
            raise ValueError(f'Unexpected S1 schema: {path}')
        yield from reader

def worker(source, output, threads=1, batch_size=64, restart_partial=False):
    source, output = Path(source), Path(output)
    if output.exists():
        old = output / 'worker_manifest.json'
        if old.exists() and json.loads(old.read_text()).get('complete'):
            raise FileExistsError(f'Completed shard already exists: {output}')
        if not restart_partial: raise FileExistsError(f'Partial output exists; pass --restart-partial for this shard only: {output}')
        shutil.rmtree(output)
    output.mkdir(parents=True)
    part_path = source.with_suffix('.json')
    if not part_path.exists(): part_path = source.parent / 'partition.json'
    part = json.loads(part_path.read_text()) if part_path.exists() else {'worker': 0}
    models = []
    for p in MODEL_PATHS:
        m = xgb.Booster(model_file=str(p)); m.set_param({'nthread': threads})
        if m.num_features() != 39: raise ValueError(f'Model does not have 39 features: {p}')
        models.append(m)
    ret = MappedRetriever(INDEX, NUMERIC_INDEX, ADDRESS_INDEX, TARGET_STORE)
    engine = OptimizedFeatureEngine()
    orig_trans = engine.icu.trans; timings = {'retrieval': 0., 'features': 0., 'ICU': 0., 'scoring': 0., 'policy_and_write': 0.}
    def traced(*args):
        t = time.perf_counter(); value = orig_trans(*args); timings['ICU'] += time.perf_counter() - t; return value
    engine.icu.trans = traced
    writer = OutputWriter(output)
    count = candidates = matches_count = 0; batch = []; start = time.perf_counter()
    last_id = None; min_candidates = None; max_candidates = 0

    def flush_batch():
        nonlocal count, candidates, matches_count, batch, last_id, min_candidates, max_candidates
        if not batch: return
        X = np.concatenate([r['X'] for r in batch]) if sum(len(r['X']) for r in batch) else np.empty((0,39), np.float32)
        t = time.perf_counter()
        if len(X):
            dm = xgb.DMatrix(X, feature_names=FEATURE_NAMES, nthread=threads)
            per_model = [m.predict(dm) for m in models]
            probs = np.mean(np.stack(per_model), axis=0, dtype=np.float32)
        else: probs = np.empty(0, np.float32)
        timings['scoring'] += time.perf_counter() - t
        offset = 0
        for r in batch:
            cs, xx = r['cs'], r['X']; pp = probs[offset:offset+len(cs)]; offset += len(cs)
            t = time.perf_counter(); selected = choose(cs, xx, pp); writer.write(r['eid'], cs, selected, r['targets']); timings['policy_and_write'] += time.perf_counter()-t
            count += 1; candidates += len(cs); matches_count += len(selected); last_id = r['eid']
            min_candidates = len(cs) if min_candidates is None else min(min_candidates, len(cs)); max_candidates = max(max_candidates, len(cs))
        writer.cf.flush(); writer.mf.flush()
        atomic(output / 'checkpoint.json', {'worker': part.get('worker'), 'entities_completed': count,
               'last_completed_s1_id': last_id, 'candidate_pairs': candidates, 'predicted_matches': matches_count,
               'wall_seconds': time.perf_counter()-start, 'complete': False})
        batch = []

    for row in rows(source):
        eid = row['entity_id']; q = normalize(row['business_name'], row['business_address'], row['country'])
        t = time.perf_counter(); cs, rare, targets = ret.retrieve(q); timings['retrieval'] += time.perf_counter()-t
        t = time.perf_counter(); X = engine.matrix(q, cs, targets, rare); timings['features'] += time.perf_counter()-t
        batch.append({'eid':eid, 'cs':cs, 'targets':targets, 'X':X})
        if len(batch) >= batch_size: flush_batch()
    flush_batch(); writer.close(); engine.close(); ret.close()
    manifest = {'configuration_id':CONFIG_ID,'complete':True,'worker':part.get('worker'),
        'source_file':str(source),'source_rows_expected':part.get('source_rows',part.get('sample_rows')),
        'source_sha256':sha(source),
        'source_rows_completed':count,'source_row_start':part.get('source_row_start',part.get('original_test_row_start')),
        'source_row_end_exclusive':part.get('source_row_end_exclusive',part.get('original_test_row_end_exclusive')),
        'first_s1_id':part.get('first_s1_id'),'last_s1_id':part.get('last_s1_id',last_id),
        'candidate_pairs':candidates,'predicted_matches':matches_count,'min_candidates_per_s1':min_candidates,
        'max_candidates_per_s1':max_candidates,'wall_seconds':time.perf_counter()-start,
        'entities_per_second':count/(time.perf_counter()-start),'candidates_per_second':candidates/(time.perf_counter()-start),
        'peak_rss_mib':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)/1024**2,
        'retrieval_seconds':dict(ret.timings),'component_seconds':timings,'threads_per_model':threads,
        'model_paths':[str(p) for p in MODEL_PATHS],'model_sha256':[sha(p) for p in MODEL_PATHS],
        'candidate_file_sha256':sha(output/'candidate_pairs.tsv'),'matching_file_sha256':sha(output/'matching_results.tsv'),
        'candidate_file_bytes':(output/'candidate_pairs.tsv').stat().st_size,
        'matching_file_bytes':(output/'matching_results.tsv').stat().st_size}
    atomic(output / 'worker_manifest.json', manifest)
    atomic(output / 'checkpoint.json', {'worker':part.get('worker'),'entities_completed':count,
           'last_completed_s1_id':last_id,'candidate_pairs':candidates,'predicted_matches':matches_count,
           'wall_seconds':manifest['wall_seconds'],'complete':True})
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    import argparse, multiprocessing
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=1); parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--restart-partial',action='store_true')
    args=parser.parse_args()
    worker(args.source,args.output,args.threads,args.batch_size,args.restart_partial)
