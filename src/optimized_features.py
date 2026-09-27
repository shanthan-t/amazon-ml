"""Cached per-record representations; arithmetic/order match frozen features.py."""
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz
from .features import FeatureEngine,DIGITS,keys,grams,scripts,zero_jac

class OptimizedFeatureEngine(FeatureEngine):
    def __init__(self,cache_size=8192):
        super().__init__(cache_size=cache_size)
        self.prepare=lru_cache(maxsize=cache_size)(self._prepare)
    def _prepare(self,q):
        n,a,c=q;nt=set(n.split());at=set(a.split());compact=''.join(n.split());nums=set(DIGITS.findall(a));latin=self.transliterate(n)
        return (nt,at,compact,nums,n.split()[0] if n else '',keys(q),grams(n),grams(a),scripts(n),latin,set(latin.split()),grams(latin))
    def pair(self,query,target,source,rare):
        if source not in (2,3):raise ValueError('Target source must explicitly be 2 or 3')
        n,a,c=query;tn,ta,tc=target
        nt,at,nc,nums,first,qkeys,ng,ag,sc,latin,lt,lg=self.prepare(tuple(query))
        tt,tat,ntc,tnums,tfirst,tkeys,tng,tag,tsc,tlatin,tlt,tlg=self.prepare(tuple(target))
        common=nt&tt;union=nt|tt;au=at|tat;nu=nums|tnums;maxn=max(len(n),len(tn),1);maxa=max(len(a),len(ta),1)
        nj=len(common)/len(union) if union else 1.0
        base=(float(bool(n) and n==tn),nj,fuzz.ratio(n,tn)/100.,float(len(nc)>=6 and len(ntc)>=6 and nc[:6]==ntc[:6]),float(len(nc)>=4 and len(ntc)>=4 and nc[:4]==ntc[:4]),fuzz.token_sort_ratio(n,tn)/100.,fuzz.token_set_ratio(n,tn)/100.,fuzz.partial_ratio(n,tn)/100.,float(len(common)),nj,float(bool(first) and first==tfirst),float(bool(a) and a==ta),len(at&tat)/len(au) if au else 1.0,fuzz.ratio(a,ta)/100.,fuzz.token_sort_ratio(a,ta)/100.,len(nums&tnums)/len(nu) if nu else 1.0,float(bool(c) and c==tc),min(len(n),len(tn))/maxn,abs(len(n)-len(tn))/100.,min(len(a),len(ta))/maxa,abs(len(a)-len(ta))/200.,float(not a),float(not ta),float('india' in c),float(source==2),float(min(len(nc),len(ntc))<=5))
        both=bool(nums and tnums);exact=float(both and nums==tnums);conflict=float(both and not nums&tnums)
        original={k.split('|',1)[0] for k in qkeys&tkeys}
        ext=(len(common)/max(1,min(len(nt),len(tt))),len(at&tat)/max(1,min(len(at),len(tat))),zero_jac(ng,tng),zero_jac(ag,tag),float(bool(sc and tsc and not sc&tsc)))
        transformed=(fuzz.ratio(latin,tlatin)/100.,zero_jac(lt,tlt),zero_jac(lg,tlg))
        return np.asarray((*base,exact,conflict,int(rare),len(original),len(original)+int(rare),*ext,*transformed),np.float32)
