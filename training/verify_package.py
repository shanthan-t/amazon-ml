"""Read-only verification of frozen semantics, native libs, and optional model."""
import hashlib,json
from pathlib import Path
from src.features import FEATURE_NAMES
from src.icu import ICU

ROOT=Path('v6_macro_f05')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(require_model=False):
    freeze=ROOT/'accuracy_freeze_v2';manifest=json.loads((freeze/'manifest.json').read_text())
    for file,expected in manifest['files'].items():
        if sha(freeze/file)!=expected:raise ValueError('Frozen file changed: '+file)
    # Runtime canonical definitions must continue to be the frozen definitions.
    runtime=Path(__file__).parent
    for file in ['normalization.py','base_features.py','features.py','icu.py','retrieval.py','policy.py','native/libicudata.so.77','native/libicuuc.so.77','native/libicui18n.so.77']:
        if sha(runtime/file)!=manifest['files'][file]:raise ValueError('Runtime canonical drift: '+file)
    cfg=json.loads((freeze/'configuration.json').read_text());assert cfg['feature_names']==list(FEATURE_NAMES)
    icu=ICU();assert icu('Café')=='cafe';icu.close()
    protected={'models/xgb_v4.xgb':'810af218bf6589f62bf300f591d7bd1450b2738f06d96d8dedc2543379bf8dfc','v6_macro_f05/frozen_checkpoint_v1/xgb31_top25.ubj':'9babea0d6aca8c68ff97622c90c3300999cb7f8ffc25fb7d07f769077c6df11b'}
    for path,h in protected.items():assert sha(path)==h
    model=ROOT/'production_run_v2/model_xgb39.ubj';model_ready=model.exists()
    if model_ready:
        m=json.loads((model.parent/'training_manifest.json').read_text());assert sha(model)==m['model_sha256'] and m['feature_names']==list(FEATURE_NAMES)
    if require_model and not model_ready:raise RuntimeError('Final production model has not been trained')
    return {'freeze_manifest_sha256':sha(freeze/'manifest.json'),'frozen_files':len(manifest['files']),'runtime_canonical_matches_freeze':True,'native_ICU_loads':True,'protected_models_unchanged':True,'production_model_ready':model_ready}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
