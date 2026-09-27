"""Streaming final-output contract validation; never performs inference."""
import csv
from itertools import zip_longest

def validate(source_ids,candidate_file,matching_file,valid_target):
    seen=set();pairs=matches=count=0
    with open(candidate_file,encoding='utf-8',newline='') as cf,open(matching_file,encoding='utf-8',newline='') as mf:
        cs=csv.DictReader(cf,delimiter='\t');ms=csv.DictReader(mf,delimiter='\t')
        if cs.fieldnames!=['source1_entity_id','candidate_entity_ids'] or ms.fieldnames!=['source1_entity_id','matched_entity_ids']:raise ValueError('TSV headers')
        for eid,c,m in zip_longest(source_ids,cs,ms):
            if eid is None or c is None or m is None:raise ValueError('Missing/extra S1 output row')
            if eid in seen or c['source1_entity_id']!=eid or m['source1_entity_id']!=eid:raise ValueError('Duplicate/order/S1 ID mismatch')
            seen.add(eid);cc=c['candidate_entity_ids'].split(',') if c['candidate_entity_ids'] else [];mm=m['matched_entity_ids'].split(',') if m['matched_entity_ids'] else []
            if len(cc)!=len(set(cc)) or len(mm)!=len(set(mm)):raise ValueError('Duplicate candidates/matches')
            if not set(mm)<=set(cc) or len(mm)>11:raise ValueError('Match subset/cap violation')
            if any(not valid_target(t) for t in cc):raise ValueError('Invalid S2/S3 target')
            pairs+=len(cc);matches+=len(mm);count+=1
    return {'source_rows':count,'candidate_pairs':pairs,'matching_pairs':matches,'valid':True}
