"""Portable source-ordered target and numeric indexes, without country whitelist."""
import csv,sqlite3,time
from pathlib import Path
from src.features import normalize
from src.base_features import generate_blocking_keys,generate_numeric_address_keys

def build(s2,s3,index,numeric_index):
    for p in [index,numeric_index]:
        if Path(p).exists():raise FileExistsError(p)
    db=sqlite3.connect(index);ndb=sqlite3.connect(numeric_index)
    for conn in (db,ndb):
        conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF');conn.execute('PRAGMA cache_size=-65536')
        conn.execute('CREATE TABLE postings(block_key TEXT NOT NULL,target_idx INTEGER NOT NULL,PRIMARY KEY(block_key,target_idx)) WITHOUT ROWID')
    db.execute('CREATE TABLE targets(target_idx INTEGER PRIMARY KEY,entity_id TEXT NOT NULL UNIQUE,normalized_name TEXT NOT NULL,normalized_address TEXT NOT NULL,normalized_country TEXT NOT NULL,source INTEGER NOT NULL)')
    idx=0;rows=[];post=[];nums=[]
    def flush():
        db.executemany('INSERT INTO targets VALUES(?,?,?,?,?,?)',rows);db.executemany('INSERT OR IGNORE INTO postings VALUES(?,?)',post);ndb.executemany('INSERT OR IGNORE INTO postings VALUES(?,?)',nums)
        db.commit();ndb.commit();rows.clear();post.clear();nums.clear()
    for source,path in [(2,s2),(3,s3)]:
        with open(path,encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f,delimiter='\t')
            if reader.fieldnames!=['entity_id','business_name','business_address','country']:raise ValueError('Target TSV schema mismatch')
            for r in reader:
                if not r['entity_id'].startswith(f'S{source}-'):raise ValueError('Invalid target ID/source')
                n,a,c=normalize(r['business_name'],r['business_address'],r['country']);rows.append((idx,r['entity_id'],n,a,c,source));post.extend((k,idx) for k in generate_blocking_keys(n,c));nums.extend((k,idx) for k in generate_numeric_address_keys(a,c));idx+=1
                if len(rows)>=10000:flush()
    flush()
    for conn in (db,ndb):
        conn.execute('CREATE TABLE key_counts AS SELECT block_key,COUNT(*) AS freq FROM postings GROUP BY block_key');conn.execute('CREATE UNIQUE INDEX key_count_key ON key_counts(block_key)');conn.commit();conn.close()
    return idx
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--s2',required=True);a.add_argument('--s3',required=True);a.add_argument('--index',required=True);a.add_argument('--numeric-index',required=True);v=a.parse_args();print(build(v.s2,v.s3,v.index,v.numeric_index))
