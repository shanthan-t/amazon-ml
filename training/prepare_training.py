"""Materialize only eligible S1/labels; partition metadata is exclusion-only.

Shared TSV transport bytes are necessarily streamed. The ID prefix is inspected
first; excluded payloads are discarded without decoding, CSV parsing, logging,
feature generation, or storing labels/examples. No holdout artifact is opened.
"""
import csv,hashlib,json,resource,sqlite3,time
from pathlib import Path
from src.features import normalize

RUN=Path('v6_macro_f05/production_run_v2')

def eligible_ids(path):
    out=set();excluded=0
    with open(path,encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['partition'] in ('train','development'):out.add(row['entity_id'].encode())
            elif row['partition']=='untouched_holdout':excluded+=1
            else:raise ValueError('Unknown partition')
    return out,excluded

def filtered_rows(path,allowed):
    with open(path,'rb') as f:
        header=f.readline().decode('utf-8').rstrip('\r\n').split('\t')
        for raw in f:
            prefix=raw.partition(b'\t')[0]
            if prefix not in allowed:continue
            # Decode/parse ONLY after eligibility. Shared files use one TSV record per line.
            vals=next(csv.reader([raw.decode('utf-8')],delimiter='\t'))
            if len(vals)!=len(header):raise ValueError('Malformed eligible TSV row')
            yield dict(zip(header,vals))

def prepare():
    t=time.perf_counter();p=RUN/'eligible_training.sqlite3'
    if p.exists():raise FileExistsError(p)
    allowed,excluded=eligible_ids('v6_macro_f05/splits_v1/assignments.tsv')
    assert len(allowed)==2196821 and excluded==10000
    db=sqlite3.connect(p);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-32768')
    db.execute('CREATE TABLE entities(pos INTEGER PRIMARY KEY,eid TEXT NOT NULL UNIQUE,name TEXT,address TEXT,country TEXT,truth TEXT)')
    batch=[];count=0;idh=hashlib.sha256()
    for row in filtered_rows('dataset/train/train_source1.tsv',allowed):
        n,a,c=normalize(row['business_name'],row['business_address'],row['country'])
        eid=row['entity_id'];idh.update((eid+'\n').encode());batch.append((count,eid,n,a,c));count+=1
        if len(batch)>=10000:db.executemany('INSERT INTO entities(pos,eid,name,address,country) VALUES(?,?,?,?,?)',batch);db.commit();batch.clear()
    if batch:db.executemany('INSERT INTO entities(pos,eid,name,address,country) VALUES(?,?,?,?,?)',batch)
    db.commit();assert count==len(allowed)
    print('eligible records',count,'excluded partition entries',excluded,flush=True)
    batch=[];nlabels=0
    for row in filtered_rows('dataset/train/train_ground_truth.tsv',allowed):
        batch.append((row['matched_entity_ids'],row['source1_entity_id']));nlabels+=1
        if len(batch)>=10000:db.executemany('UPDATE entities SET truth=? WHERE eid=?',batch);db.commit();batch.clear()
    if batch:db.executemany('UPDATE entities SET truth=? WHERE eid=?',batch)
    db.commit();assert nlabels==count and db.execute('SELECT COUNT(*) FROM entities WHERE truth IS NULL').fetchone()[0]==0
    r={'eligible_entities':count,'labels':nlabels,'excluded_partition_entries':excluded,'eligible_id_sha256':idh.hexdigest(),'eligibility':'all train + development IDs from frozen partition metadata','sealed_holdout_labels_or_examples_parsed':0,'exclusion_handling':'Filter ID prefix before decoding/parsing payload; no sealed holdout files or records inspected','wall_seconds':time.perf_counter()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'database_bytes':p.stat().st_size}
    db.execute('CREATE TABLE manifest(json TEXT)');db.execute('INSERT INTO manifest VALUES(?)',(json.dumps(r),));db.commit();db.close()
    (RUN/'eligible_training.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':prepare()
