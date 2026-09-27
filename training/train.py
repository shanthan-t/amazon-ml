"""One full eligible-data XGBoost model, unchanged candidates/class weighting.

Float32 feature shards and CPU external-memory quantile matrices bound RAM.
No holdout labels, early stopping, new threshold search, negative sampling or ensemble.
"""
import hashlib,json,os,resource,shutil,time
from pathlib import Path
import numpy as np
import xgboost as xgb
from src.features import FeatureEngine,FEATURE_NAMES
from src.retrieval import Retriever,connect

ROOT=Path('v6_macro_f05');RUN=ROOT/'production_run_v2';SHARDS=RUN/'training_shards'

def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def require_parity():
    f=json.loads((RUN/'feature_parity.json').read_text());r=json.loads((RUN/'retrieval_parity.json').read_text());u=json.loads((RUN/'icu_reproducibility.json').read_text())
    assert f['exactly_equal'] and f['max_abs_difference']==0 and r['missing_pairs']==r['extra_pairs']==0 and r['candidate_count']==1049495 and u['pass']
    freeze=ROOT/'accuracy_freeze_v2';m=json.loads((freeze/'manifest.json').read_text())
    for file,expected in m['files'].items():assert sha(freeze/file)==expected
    assert '41 passed' in (RUN/'tests.log').read_text() or 'passed' in (RUN/'tests.log').read_text() and 'failed' not in (RUN/'tests.log').read_text()

def fit_memory_preflight(candidate_rows):
    """Refuse CPU fitting when row-scale training buffers lack safe RAM headroom.

    The out-of-core quantile matrix pages features to disk, but CPU histogram
    training still needs labels, predictions, and gradient/Hessian pairs. Use
    16 bytes per row for those arrays plus 2 GiB for XGBoost/Python/OS slack.
    """
    meminfo={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value,*_=line.replace(':','').split()
        meminfo[key]=int(value)*1024
    available=meminfo['MemAvailable']
    required=int(candidate_rows)*16+2*1024**3
    report={'candidate_rows':int(candidate_rows),'bytes_per_row_buffer_estimate':16,
            'reserve_bytes':2*1024**3,'required_available_bytes':required,
            'available_bytes':available,'pass':available>=required}
    atomic(RUN/'fit_memory_preflight.json',report)
    if not report['pass']:
        raise RuntimeError('Unsafe full-data fit RAM: external-memory features do not bound row-scale labels/predictions/gradients; '+json.dumps(report))
    return report

def build_shards(optimized=False,chunk_entities=1000):
    require_parity();SHARDS.mkdir(exist_ok=True);start=time.perf_counter();cpu=time.process_time()
    statepath=RUN/'training_build.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'complete':False,'next_entity_position':0,'entities':0,'candidates':0,'positives':0,'negatives':0,'shards':[],'configuration_id':'v6-top25-xgb39-icu-global98-20260927-v1','active_seconds':0.}
    if state['complete']:print('training shards already complete');return
    source=connect(RUN/'eligible_training.sqlite3');total=source.execute('SELECT COUNT(*) FROM entities').fetchone()[0]
    assert total==2196821
    ret=Retriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3')
    if optimized:
        from src.optimized_features import OptimizedFeatureEngine
        from src.target_store import MappedRetriever
        ret.close()
        ret=MappedRetriever('v2_train_index.sqlite3','phase2_train_numeric_address.sqlite3',RUN/'train_address_index.sqlite3',RUN/'train_target_store')
        engine=OptimizedFeatureEngine()
        pr=json.loads((RUN/'profile_optimized_reference/profile.json').read_text())
        assert all(pr['parity'][s] for s in ('exact_features','exact_scores','exact_candidates','exact_predictions'))
    else:engine=FeatureEngine()
    prev=time.perf_counter()
    while state['next_entity_position']<total:
        # Leave space for model external-memory pages and quantile construction.
        free=shutil.disk_usage(RUN).free
        if free<80*1024**3:raise RuntimeError('Less than 80 GiB free; stopped before disk exhaustion')
        pos=state['next_entity_position'];records=source.execute('SELECT * FROM entities WHERE pos>=? ORDER BY pos LIMIT ?',(pos,chunk_entities)).fetchall()
        assert records and records[0][0]==pos
        matrices=[];labels=[];ph=hashlib.sha256();idsh=hashlib.sha256()
        for number,eid,n,a,c,truth_text in records:
            truth=set(filter(None,truth_text.split(',')));cs,rare,targets=ret.retrieve((n,a,c))
            xx=engine.matrix((n,a,c),cs,targets,rare);yy=np.asarray([targets[i][0] in truth for i in cs],np.uint8)
            matrices.append(xx);labels.append(yy);ph.update((eid+'\t'+','.join(map(str,cs))+'\n').encode());idsh.update((eid+'\n').encode())
        xx=np.concatenate(matrices);yy=np.concatenate(labels);number=len(state['shards']);prefix=SHARDS/f'{number:05d}'
        # State commits after both files. Resume never rereads sealed source files.
        np.save(str(prefix)+'_X.npy',xx);np.save(str(prefix)+'_y.npy',yy)
        sh={'prefix':str(prefix),'first_entity_position':pos,'entities':len(records),'candidates':len(yy),'positives':int(yy.sum()),'candidate_sha256':ph.hexdigest(),'entity_sha256':idsh.hexdigest(),'features_sha256':sha(str(prefix)+'_X.npy'),'labels_sha256':sha(str(prefix)+'_y.npy')}
        state['shards'].append(sh);state['entities']+=len(records);state['next_entity_position']=records[-1][0]+1;state['candidates']+=len(yy);state['positives']+=int(yy.sum());state['negatives']+=int((yy==0).sum());state['active_seconds']+=time.perf_counter()-prev;state['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;atomic(statepath,state);prev=time.perf_counter()
        print(json.dumps({k:state[k] for k in ['entities','candidates','positives','negatives','active_seconds','peak_rss_mib']}),flush=True)
    state['complete']=True;state['feature_names']=FEATURE_NAMES;state['dtype']='float32';atomic(statepath,state)
    source.close();ret.close();engine.close()

class ShardIter(xgb.DataIter):
    def __init__(self,shards,cache):
        self.shards=shards;self.i=0
        super().__init__(cache_prefix=str(cache),release_data=True)
    def reset(self):self.i=0
    def next(self,input_data):
        if self.i==len(self.shards):return False
        prefix=self.shards[self.i]['prefix'];self.i+=1
        xx=np.load(prefix+'_X.npy',mmap_mode='r');yy=np.load(prefix+'_y.npy',mmap_mode='r')
        input_data(data=xx,label=yy)
        return True

class LogProgress(xgb.callback.TrainingCallback):
    def __init__(self):self.start=time.perf_counter()
    def after_iteration(self,model,epoch,evals_log):
        if epoch%10==0:print('tree',epoch+1,'seconds',round(time.perf_counter()-self.start,1),flush=True)
        if (epoch+1)%50==0:model.save_model(RUN/'production_checkpoint.ubj')
        return False

def fit():
    require_parity();state=json.loads((RUN/'training_build.json').read_text());assert state['complete'] and state['entities']==2196821
    fit_memory_preflight(state['candidates'])
    path=RUN/'model_xgb39.ubj'
    if path.exists():raise FileExistsError(path)
    cfg=json.loads((ROOT/'accuracy_freeze_v2/configuration.json').read_text());p=cfg['full_training']['parameters'];params={'max_depth':p['max_depth'],'eta':p['learning_rate'],'min_child_weight':p['min_child_weight'],'subsample':p['subsample'],'colsample_bytree':p['colsample_bytree'],'lambda':p['reg_lambda'],'objective':p['objective'],'eval_metric':p['eval_metric'],'tree_method':p['tree_method'],'max_bin':p['max_bin'],'seed':p['random_state'],'nthread':p['n_jobs'],'scale_pos_weight':state['negatives']/state['positives'],'device':'cpu'}
    cache=RUN/'xgb_cache';cache.mkdir(exist_ok=True);t=time.perf_counter();print('Building CPU external-memory quantile matrix',flush=True)
    iterator=ShardIter(state['shards'],cache/'train');dm=xgb.ExtMemQuantileDMatrix(iterator,max_bin=256,nthread=4)
    print('External quantile matrix ready, pairs',dm.num_row(),'seconds',time.perf_counter()-t,flush=True)
    assert dm.num_row()==state['candidates'] and dm.num_col()==39
    model=xgb.train(params,dm,num_boost_round=cfg['full_training']['estimators'],callbacks=[LogProgress()]);model.save_model(path)
    manifest={'configuration_id':cfg['configuration_id'],'training_entities':state['entities'],'positives':state['positives'],'negatives':state['negatives'],'candidate_pairs':state['candidates'],'feature_names':FEATURE_NAMES,'dtype':'float32','parameters':params,'num_boost_round':cfg['full_training']['estimators'],'early_stopping':False,'ensemble':False,'model_sha256':sha(path),'training_build_manifest_sha256':sha(RUN/'training_build.json'),'eligible_manifest_sha256':sha(RUN/'eligible_training.json'),'sealed_holdout_included':False,'wall_seconds':time.perf_counter()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    atomic(RUN/'training_manifest.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['build','fit']);a.add_argument('--optimized',action='store_true');v=a.parse_args()
    if v.stage=='build':build_shards(v.optimized)
    else:fit()
