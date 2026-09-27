"""Read only fixed development artifacts; never reads sealed labels/examples."""
import csv,hashlib,json,pickle,resource,time
from pathlib import Path
import numpy as np
from src.features import FeatureEngine, FEATURE_NAMES, normalize
from src.retrieval import Retriever,connect

ROOT=Path('v6_macro_f05');RUN=ROOT/'production_run_v2'

def load_dev():
    with (ROOT/'retrieval_study/route_candidates.pkl').open('rb') as f:routes,dev,_=pickle.load(f)
    with (ROOT/'candidate_efficiency_v1/candidate_sets.pkl').open('rb') as f:sets=pickle.load(f)['top_25']
    rows=list(csv.DictReader((ROOT/'model_study_v1/fold_assignments.tsv').open(),delimiter='\t'))
    eids=[r['source1_entity_id'] for r in rows]
    return dev,sets,eids,routes

def dump(name,r):
    (RUN/name).write_text(json.dumps(r,indent=2)+'\n');print(name,json.dumps(r),flush=True)

def feature_audit():
    start=time.perf_counter();dev,sets,eids,routes=load_dev();M=ROOT/'model_study_v1'
    x=np.load(M/'features.npy',mmap_mode='r');ep=np.load(M/'entity_positions.npy',mmap_mode='r');tids=np.load(M/'target_indices.npy',mmap_mode='r')
    setlookup={k:set(v) for k,v in sets.items()}
    ix=np.fromiter((i for i in range(len(ep)) if int(tids[i]) in setlookup[eids[int(ep[i])]]),dtype=np.int64)
    assert len(ix)==1049495
    ext=np.load(ROOT/'matcher_recall_v1/feature_extensions.npy',mmap_mode='r');latin=np.load(ROOT/'final_focused_v1/transliteration_features.npy',mmap_mode='r')
    # Uniform stable sample plus all positives in important missing/cross-script cases.
    y=np.load(M/'labels.npy',mmap_mode='r')[ix]
    sample=np.unique(np.r_[np.arange(0,len(ix),101),np.flatnonzero((y==1)&((ext[:,4]>0)|(x[ix,22]>0)))])
    engine=FeatureEngine();db=connect('v2_train_index.sqlite3');targets={};ids=sorted(set(map(int,tids[ix[sample]])))
    for st in range(0,len(ids),8000):
        ch=ids[st:st+8000]
        for idx,*r in db.execute('SELECT * FROM targets WHERE target_idx IN ('+','.join('?'*len(ch))+')',ch):targets[idx]=r
    actual=[];expected=[]
    for j in sample:
        i=ix[j];eid=eids[int(ep[i])];idx=int(tids[i]);r=targets[idx]
        actual.append(engine.pair(dev[eid]['query'],tuple(r[1:4]),r[4],idx in routes['rare_address_overlap'][eid]))
        expected.append(np.r_[x[i],ext[j],latin[j]])
    aa=np.asarray(actual,np.float32);ee=np.asarray(expected,np.float32);diff=np.abs(aa-ee)
    # independent experiment transform vs production, twice, for all 4k development names
    from v6_macro_f05.final_focused_v1 import ICU as ExperimentICU
    old=ExperimentICU();names=[dev[e]['query'][0] for e in eids]
    transformed1=[engine.transliterate(n) for n in names];transformed2=[engine.transliterate(n) for n in names]
    reference=[old(normalize(n,'','')[0]) for n in names];old.close()
    h=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    det={'development_names':len(names),'pass':transformed1==transformed2==reference,'run1_sha256':h(transformed1),'run2_sha256':h(transformed2),'experiment_sha256':h(reference)}
    dump('icu_reproducibility.json',det)
    r={'sample_pairs':len(sample),'feature_count':39,'dtype':str(aa.dtype),'exactly_equal':bool(np.array_equal(aa,ee)),'max_abs_difference':float(diff.max()),'max_abs_difference_by_feature':dict(zip(FEATURE_NAMES,map(float,diff.max(axis=0)))),'sample_row_sha256':hashlib.sha256(sample.tobytes()).hexdigest(),'production_features_sha256':hashlib.sha256(aa.tobytes()).hexdigest(),'experiment_features_sha256':hashlib.sha256(ee.tobytes()).hexdigest(),'wall_seconds':time.perf_counter()-start}
    engine.close();db.close();dump('feature_parity.json',r)
    assert r['exactly_equal'] and det['pass']

def retrieval_audit(optimized=False,filtered=False):
    start=time.perf_counter();cpu=time.process_time();dev,sets,eids,routes=load_dev()
    ret=Retriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3')
    if optimized:
        from src.target_store import MappedRetriever
        ret.close();ret=MappedRetriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3',RUN/'train_target_store')
    if filtered:
        from src.token_filter import FilteredRetriever
        ret.close();ret=FilteredRetriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3',RUN/'train_target_store')
    missing=extra=count=0;expected_hash=hashlib.sha256();actual_hash=hashlib.sha256();rare_mismatch=0;prod={}
    for pos,eid in enumerate(eids):
        cs,rare,targets=ret.retrieve(dev[eid]['query']);expected=set(sets[eid]);actual=set(cs)
        missing+=len(expected-actual);extra+=len(actual-expected);count+=len(cs)
        rare_mismatch+=len(rare^(set(routes['rare_address_overlap'][eid])&expected))
        actual_hash.update((eid+'\t'+','.join(map(str,cs))+'\n').encode());expected_hash.update((eid+'\t'+','.join(map(str,sorted(expected)))+'\n').encode());prod[eid]=cs
        if (pos+1)%250==0:print('retrieval parity',pos+1,'pairs',count,'missing/extra',missing,extra,flush=True)
    r={'entities':len(eids),'candidate_count':count,'missing_pairs':missing,'extra_pairs':extra,'rare_metadata_differences':rare_mismatch,'candidate_set_sha256':actual_hash.hexdigest(),'experiment_candidate_set_sha256':expected_hash.hexdigest(),'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'component_seconds':dict(ret.timings)}
    ret.close();dump('retrieval_parity_filtered.json' if filtered else 'retrieval_parity_optimized.json' if optimized else 'retrieval_parity.json',r)
    assert count==1049495 and missing==extra==rare_mismatch==0 and actual_hash.hexdigest()==expected_hash.hexdigest()
    if not optimized:
        with (RUN/'production_development_candidates.pkl').open('xb') as f:pickle.dump(prod,f,protocol=5)
if __name__=='__main__':
    import sys
    if sys.argv[1]=='features':feature_audit()
    elif sys.argv[1]=='retrieval':retrieval_audit()
    elif sys.argv[1]=='retrieval_optimized':retrieval_audit(True)
    elif sys.argv[1]=='retrieval_filtered':retrieval_audit(True,True)
