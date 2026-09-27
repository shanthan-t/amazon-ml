"""Independent reproduction of approved routes and top_25 selection."""
import sqlite3
import time
from collections import Counter
import numpy as np
from rapidfuzz import fuzz
from .features import keys, numeric
from .base_features import compute_pair_features


def connect(path):
    from pathlib import Path
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    db=sqlite3.connect(uri,uri=True)
    db.execute('PRAGMA cache_size=-16384');return db


def top25(query,baseline,eligible,targets):
    # The original experiment subtracts base candidates BEFORE the top25 cap.
    added=set(eligible)-set(baseline)
    rank=[]
    for idx in added:
        _,n,a,c,s=targets[idx]
        # Float32 rounding before ranking is part of the approved experiment.
        f=np.asarray(compute_pair_features(*query,n,a,c,target_source=s),np.float32)
        exact,conflict=numeric(query[1],a)
        rank.append((conflict,-float(f[12]),-float(f[13]),-exact,-float(f[2]),idx))
    rare={r[-1] for r in sorted(rank)[:25]}
    return sorted(set(baseline)|rare),rare


class Retriever:
    def __init__(self,index,numeric_index,address_index):
        self.db=connect(index);self.ndb=connect(numeric_index);self.adb=connect(address_index)
        self.timings=Counter()
    def fetch(self,ids):
        ids=sorted(ids);out={}
        for st in range(0,len(ids),8000):
            ch=ids[st:st+8000]
            for idx,*row in self.db.execute('SELECT * FROM targets WHERE target_idx IN ('+','.join('?'*len(ch))+')',ch):out[idx]=row
        if len(out)!=len(ids):raise ValueError('Missing target')
        return out
    def retrieve(self,query):
        t=time.perf_counter();base=set()
        for key in sorted(keys(query)):
            conn=self.ndb if key.startswith('AN|') else self.db
            row=conn.execute('SELECT freq FROM key_counts WHERE block_key=?',(key,)).fetchone()
            if row and row[0]<=500:
                base.update(r[0] for r in conn.execute('SELECT target_idx FROM postings WHERE block_key=?',(key,)))
        self.timings['base_retrieval']+=time.perf_counter()-t
        t=time.perf_counter();n,a,c=query;rs=[]
        for token in sorted(set(a.split())):
            if len(token)<3:continue
            row=self.adb.execute('SELECT freq FROM address_counts WHERE country=? AND token=?',(c,token)).fetchone()
            if row and 0<row[0]<=5000:rs.append((row[0],token))
        pool=set()
        for _,token in sorted(rs)[:2]:pool.update(r[0] for r in self.adb.execute('SELECT target_idx FROM address_postings WHERE country=? AND token=?',(c,token)))
        targets=self.fetch(base|pool);at=set(a.split());eligible=set()
        for idx in pool:
            ta=targets[idx][2];tt=set(ta.split());common=at&tt
            if sum(not z.isdigit() for z in common)>=2 and (2*len(common)/max(1,len(at)+len(tt))>=.55 or fuzz.ratio(a,ta)>=70):eligible.add(idx)
        if len(eligible)>500:eligible=set()
        self.timings['rare_address_overlap_retrieval']+=time.perf_counter()-t
        t=time.perf_counter();cs,rare=top25(query,base,eligible,targets);self.timings['top25_pruning']+=time.perf_counter()-t
        return cs,rare,{i:targets[i] for i in cs}
    def close(self):
        for c in (self.db,self.ndb,self.adb):c.close()
