"""Approved policy and TSV contract: all scored candidates are exported."""
import csv
import numpy as np

def choose(candidates,features,scores):
    if len(candidates)!=len(set(candidates)): raise ValueError('Duplicate candidate index')
    if features.shape!=(len(candidates),39) or len(scores)!=len(candidates): raise ValueError('Scoring dimensions differ')
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite model score')
    threshold=np.where(features[:,27]>0,np.float32(.99),np.float32(.98))
    accepted=[(float(s),int(t)) for t,s,th in zip(candidates,scores,threshold) if s>=th]
    return [t for s,t in sorted(accepted,reverse=True)[:11]]

class OutputWriter:
    def __init__(self,directory):
        from pathlib import Path
        p=Path(directory); p.mkdir(parents=True,exist_ok=True)
        self.cf=(p/'candidate_pairs.tsv').open('x',encoding='utf-8',newline='')
        self.mf=(p/'matching_results.tsv').open('x',encoding='utf-8',newline='')
        self.cw=csv.writer(self.cf,delimiter='\t',lineterminator='\n')
        self.mw=csv.writer(self.mf,delimiter='\t',lineterminator='\n')
        self.cw.writerow(['source1_entity_id','candidate_entity_ids']);self.mw.writerow(['source1_entity_id','matched_entity_ids'])
        self.seen=set()
    def write(self,eid,candidates,matches,targets):
        if eid in self.seen:raise ValueError('Repeated S1 ID')
        if len(candidates)!=len(set(candidates)) or len(matches)!=len(set(matches)):raise ValueError('Duplicate candidate/match')
        if not set(matches)<=set(candidates) or len(matches)>11:raise ValueError('Match outside candidate set or cap')
        ids=[targets[i][0] for i in candidates]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate target entity ID')
        if any(targets[i][4] not in (2,3) or not targets[i][0].startswith(f'S{targets[i][4]}-') for i in candidates):raise ValueError('Invalid target source/ID')
        self.seen.add(eid)
        self.cw.writerow([eid,','.join(ids)]);self.mw.writerow([eid,','.join(targets[i][0] for i in matches)])
    def close(self):self.cf.close();self.mf.close()
