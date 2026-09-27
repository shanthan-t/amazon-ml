"""Full-corpus country/address-token index. Read-only existing target table."""
import json,resource,sqlite3,time
from pathlib import Path
from src.retrieval import connect

def build(source,output):
    p=Path(output)
    if p.exists():raise FileExistsError(p)
    t=time.perf_counter();cpu=time.process_time(); db=sqlite3.connect(p)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-65536');db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE address_postings(country TEXT NOT NULL,token TEXT NOT NULL,target_idx INTEGER NOT NULL)')
    src=connect(source);batch=[];rows=0;nt=0
    for idx,a,c in src.execute('SELECT target_idx,normalized_address,normalized_country FROM targets ORDER BY target_idx'):
        batch.extend((c,token,idx) for token in set(a.split()) if len(token)>=3);nt+=1
        if len(batch)>=100000:
            db.executemany('INSERT INTO address_postings VALUES(?,?,?)',batch);rows+=len(batch);batch.clear()
        if nt%500000==0:db.commit();print(json.dumps({'targets':nt,'postings':rows,'seconds':time.perf_counter()-t}),flush=True)
    if batch:db.executemany('INSERT INTO address_postings VALUES(?,?,?)',batch);rows+=len(batch)
    db.commit();src.close(); print('building country/token index',flush=True)
    db.execute('CREATE UNIQUE INDEX address_postings_key ON address_postings(country,token,target_idx)');db.commit()
    db.execute('CREATE TABLE address_counts AS SELECT country,token,count(*) AS freq FROM address_postings GROUP BY country,token')
    db.execute('CREATE UNIQUE INDEX address_count_key ON address_counts(country,token)');db.commit()
    # Retain full postings on disk; retrieval still strictly uses count <=5000.
    r={'target_count':nt,'posting_count':rows,'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'bytes':p.stat().st_size,'complete':True}
    db.execute('CREATE TABLE build_manifest(json TEXT)');db.execute('INSERT INTO build_manifest VALUES(?)',(json.dumps(r),));db.commit();db.close()
    p.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',required=True);v=a.parse_args();build(v.source,v.output)
