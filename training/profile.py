"""Development-only profiling/parity, optionally with trained production model."""
import csv,hashlib,json,resource,time
from collections import Counter
from pathlib import Path
import numpy as np
import xgboost as xgb
from src.audit import load_dev
from src.features import FeatureEngine
from src.retrieval import Retriever
from src.policy import choose,OutputWriter

RUN=Path('v6_macro_f05/production_run_v2')

def proc_io():
    try:return {a:int(b) for a,b in (s.split(':') for s in Path('/proc/self/io').read_text().splitlines())}
    except OSError:return {}

def run(optimized=False,limit=128,model=None,filtered=False):
    # A representative deterministic spread across the fixed dev order.
    dev,sets,eids,routes=load_dev();chosen=np.linspace(0,len(eids)-1,limit,dtype=int)
    ret=Retriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3')
    if optimized:
        from src.optimized_features import OptimizedFeatureEngine
        from src.target_store import MappedRetriever
        ret.close()
        ret=MappedRetriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3',RUN/'train_target_store')
        engine=OptimizedFeatureEngine()
        if filtered:
            from src.token_filter import FilteredRetriever
            ret.close();ret=FilteredRetriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3',RUN/'train_target_store')
    else:engine=FeatureEngine()
    predictor=xgb.XGBClassifier(n_jobs=4)
    model=model or 'v6_macro_f05/final_focused_v1/models/B_cross_script/fold0.ubj'
    predictor.load_model(model);predictor.set_params(n_jobs=4)
    name=('filtered' if filtered else 'optimized' if optimized else 'baseline')+('_production' if 'production_run_v2' in str(model) else '_reference')
    out=RUN/f'profile_{name}';writer=OutputWriter(out)
    wall=time.perf_counter();cpu=time.process_time();io0=proc_io();timings=Counter();matrices=[];scores=[];pair_hash=hashlib.sha256();pred_hash=hashlib.sha256();count=0
    # Instrument actual C transform calls, excluded from remaining feature time.
    orig=engine.icu.trans
    def traced(*a):
        t=time.perf_counter();r=orig(*a);timings['ICU_transliteration']+=time.perf_counter()-t;return r
    engine.icu.trans=traced
    for pos in chosen:
        eid=eids[pos];query=dev[eid]['query'];cs,rare,targets=ret.retrieve(query)
        assert cs==sorted(sets[eid])
        t=time.perf_counter();xx=engine.matrix(query,cs,targets,rare);timings['features_including_ICU']+=time.perf_counter()-t
        t=time.perf_counter();pp=predictor.predict_proba(xx)[:,1] if len(cs) else np.empty(0,np.float32);timings['XGBoost_scoring']+=time.perf_counter()-t
        t=time.perf_counter();matches=choose(cs,xx,pp);timings['policy_aggregation']+=time.perf_counter()-t
        # Validate contract via shared writer, measure its two actual row writes.
        candidate_write=writer.cw;matching_write=writer.mw
        class Timed:
            def __init__(self,inner,key):self.inner,self.key=inner,key
            def writerow(self,row):
                t=time.perf_counter();self.inner.writerow(row);timings[self.key]+=time.perf_counter()-t
        writer.cw=Timed(candidate_write,'candidate_TSV_writing');writer.mw=Timed(matching_write,'matching_TSV_writing')
        writer.write(eid,cs,matches,targets);writer.cw=candidate_write;writer.mw=matching_write
        pair_hash.update((eid+'\t'+','.join(map(str,cs))+'\n').encode());pred_hash.update((eid+'\t'+','.join(map(str,matches))+'\n').encode())
        matrices.append(xx);scores.append(pp);count+=len(cs)
    writer.close();engine.close();ret.close();timings.update(ret.timings)
    xx=np.concatenate(matrices);pp=np.concatenate(scores);np.save(out/'features.npy',xx);np.save(out/'scores.npy',pp)
    elapsed=time.perf_counter()-wall;cpu=time.process_time()-cpu;io1=proc_io()
    r={'name':name,'model':str(model),'purpose':'pipeline timing and implementation parity only, NOT accuracy estimation on fold0 predictions','source_entities':len(chosen),'candidates':count,'candidate_set_sha256':pair_hash.hexdigest(),'prediction_set_sha256':pred_hash.hexdigest(),'feature_sha256':hashlib.sha256(xx.tobytes()).hexdigest(),'score_sha256':hashlib.sha256(pp.tobytes()).hexdigest(),'wall_seconds':elapsed,'cpu_seconds':cpu,'cpu_utilization_percent_one_core':cpu/elapsed*100,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'pairs_per_second':count/elapsed,'component_seconds':dict(timings),'disk_io':{k:io1[k]-io0.get(k,0) for k in io1},'candidate_TSV_bytes':(out/'candidate_pairs.tsv').stat().st_size,'matching_TSV_bytes':(out/'matching_results.tsv').stat().st_size}
    if optimized:
        base=RUN/'profile_baseline_reference';bx=np.load(base/'features.npy');bp=np.load(base/'scores.npy');br=json.loads((base/'profile.json').read_text())
        r['parity']={'exact_features':bool(np.array_equal(xx,bx)),'max_abs_feature_diff':float(np.max(np.abs(xx-bx))),'exact_scores':bool(np.array_equal(pp,bp)),'max_abs_score_diff':float(np.max(np.abs(pp-bp))),'exact_candidates':pair_hash.hexdigest()==br['candidate_set_sha256'],'exact_predictions':pred_hash.hexdigest()==br['prediction_set_sha256']}
        assert all(r['parity'][key] for key in ['exact_features','exact_scores','exact_candidates','exact_predictions'])
    (out/'profile.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--optimized',action='store_true');a.add_argument('--limit',type=int,default=128);a.add_argument('--model');a.add_argument('--filtered',action='store_true');v=a.parse_args();run(v.optimized or v.filtered,v.limit,v.model,v.filtered)
