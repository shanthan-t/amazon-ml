"""Create an immutable new accuracy configuration, never replace an old freeze."""
import hashlib,json,shutil,sys,time
from pathlib import Path
import xgboost as xgb
from src.features import FEATURE_NAMES

ROOT=Path('v6_macro_f05');ID='v6-top25-xgb39-icu-global98-20260927-v1'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def create():
    dest=ROOT/'accuracy_freeze_v2';dest.mkdir(exist_ok=False);source=Path(__file__).parent
    names=['normalization.py','base_features.py','icu.py','features.py','retrieval.py','policy.py','requirements.lock.txt','environment.json']
    for name in names:shutil.copy2(source/name,dest/name)
    shutil.copytree(source/'native',dest/'native')
    best=[];foldhash={}
    for i in range(4):
        p=ROOT/f'final_focused_v1/models/B_cross_script/fold{i}.ubj';m=xgb.XGBClassifier();m.load_model(p)
        best.append(m.best_iteration+1);foldhash[str(p)]=sha(p)
    iterations=sorted(best)[len(best)//2]
    cfg={'configuration_id':ID,'approval':'User productionization request attached 2026-09-27; freeze 39-feature architecture only',
     'feature_names':FEATURE_NAMES,'feature_dtype':'float32','normalization':'snapshot normalization.py: Python Unicode 16.0.0 NFKC/casefold; before/after ICU name normalization',
     'retrieval':{'base_routes':['E','P6','P4','ST','W','AN'],'base_posting_cap':500,'rare_address_token_min_length':3,'rare_address_posting_cap':5000,'rarest_query_tokens':2,'rare_token_order':['global target frequency','token ascending'],'minimum_shared_nonnumeric_tokens':2,'minimum_address_dice':.55,'minimum_address_fuzz':70,'similarity_predicate':'dice >= .55 OR fuzz >= 70','raw_eligible_route_cap':500,'oversize_behavior':'reject entire rare route before subtracting base','subtract_baseline_before_pruning':True,'top_rare_additions':25,'ranking':['numeric_conflict asc','float32 address Jaccard desc','float32 address fuzz desc','numeric_exact desc','float32 name fuzz desc','target_idx asc'],'candidate_order':'ascending target_idx; S2 file order followed by S3 file order','metadata':'original routes = all shared key route labels including suppressed keys; rare flag only non-base additions; route agreement = original_route_count + rare_flag'},
     'ICU':{'version':'77.1','transform':'Any-Latin; Latin-ASCII','API':'ctypes utrans_openU_77 / utrans_transUChars_77','native_dependencies':'exact bundled libicui18n, libicuuc, libicudata, hashes in manifest; Fedora 44 x86_64/glibc 2.43 environment'},
     'source_feature':'target_is_s2 = float(targets.source == 2); explicit S2=2 S3=3 mandatory',
     'decision':{'global_threshold':.98,'numeric_conflict_threshold':.99,'max_matches':11,'tie_order':'score desc then target_idx desc','threshold_dtype':'float32, same as successful OOF experiment'},
     'OOF':{'entity_folds':4,'fold_assignments_sha256':sha(ROOT/'model_study_v1/fold_assignments.tsv'),'candidate_count':1049495,'pair_positive_count':13343,'pair_negative_count':1036152,'all_retrieved_negatives_retained':True,'estimators_limit':500,'early_stopping_rounds':35,'validation_metric':'weighted-training / unweighted validation binary logloss','fold_best_tree_counts':best,'fold_model_hashes':foldhash},
     'full_training':{'eligible_partitions':['train','development'],'exclude_partition':'untouched_holdout','holdout_metadata_only_for_exclusion':True,'model_count':1,'estimators':iterations,'estimator_derivation':'median of (best_iteration+1) from four approved OOF folds; no holdout','early_stopping':False,'scale_pos_weight':'all eligible retrieved training negatives / all eligible retrieved training positives; no negative subsampling','parameters':{'max_depth':7,'learning_rate':.08,'min_child_weight':10,'subsample':.85,'colsample_bytree':.9,'reg_lambda':2.,'objective':'binary:logistic','eval_metric':'logloss','tree_method':'hist','max_bin':256,'random_state':42,'n_jobs':4}},
     'prohibited':'no conditional .96 branch; no 43-feature model; no further holdout evaluation; no test inference in this phase'}
    (dest/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n');(dest/'feature_names.json').write_text(json.dumps(FEATURE_NAMES,indent=2)+'\n')
    manifest={'configuration_id':ID,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for p in dest.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in dest.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    dest.chmod(0o555);print(json.dumps({'id':ID,'path':str(dest),'trees':iterations,'manifest_sha256':sha(dest/'manifest.json')}))
if __name__=='__main__':create()
