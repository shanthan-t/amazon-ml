"""Canonical float32 39-feature generation shared by training and inference.

Input triples are already normalized exactly once using normalization.py.
Transliteration repeats name normalization before and after ICU, matching OOF.
"""
import re
import unicodedata
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz
from .base_features import V2_FEATURE_NAMES, compute_pair_features, generate_blocking_keys, generate_numeric_address_keys
from .normalization import normalize_business_name, normalize_business_address, normalize_country
from .icu import ICU

FEATURE_NAMES = (*V2_FEATURE_NAMES,
    'numeric_set_exact','numeric_conflict','rare_address_overlap','original_route_count','route_agreement_count',
    'name_token_containment','address_token_containment','name_char_trigram_jaccard','address_char_trigram_jaccard','name_script_mismatch',
    'transliterated_name_fuzz','transliterated_token_jaccard','transliterated_char_trigram_jaccard')
assert len(FEATURE_NAMES)==39
DIGITS=re.compile(r'\d+')

def normalize(name,address,country):
    return normalize_business_name(name),normalize_business_address(address),normalize_country(country)

def grams(s):
    s=''.join(s.split())
    return {s[i:i+3] for i in range(len(s)-2)} if len(s)>=3 else ({s} if s else set())

def scripts(s):
    out=set()
    for ch in s:
        if unicodedata.category(ch).startswith('L'):
            words=unicodedata.name(ch,'').split()
            if words: out.add(words[0])
    return out

def keys(q):
    n,a,c=q
    return set(generate_blocking_keys(n,c)+generate_numeric_address_keys(a,c))

def numeric(a,b):
    x,y=set(DIGITS.findall(a)),set(DIGITS.findall(b)); both=bool(x and y)
    return float(both and x==y),float(both and not x&y)

def zero_jac(a,b):
    return len(a&b)/max(1,len(a|b))

class FeatureEngine:
    def __init__(self,cache_size=0):
        self.icu=ICU()
        self.transliterate=lru_cache(maxsize=cache_size)(self._transliterate) if cache_size else self._transliterate
    def _transliterate(self,name):
        return self.icu(normalize_business_name(name))
    def pair(self,query,target,source,rare):
        if source not in (2,3): raise ValueError('Target source must explicitly be 2 or 3')
        n,a,c=query; tn,ta,tc=target
        base=compute_pair_features(n,a,c,tn,ta,tc,target_source=source)
        exact,conflict=numeric(a,ta)
        original={k.split('|',1)[0] for k in keys(query)&keys(target)}
        nt,tt=set(n.split()),set(tn.split()); at,tat=set(a.split()),set(ta.split())
        ns,ts=scripts(n),scripts(tn)
        ext=(len(nt&tt)/max(1,min(len(nt),len(tt))),len(at&tat)/max(1,min(len(at),len(tat))),
             zero_jac(grams(n),grams(tn)),zero_jac(grams(a),grams(ta)),float(bool(ns and ts and not ns&ts)))
        nl,tl=self.transliterate(n),self.transliterate(tn)
        latin=(fuzz.ratio(nl,tl)/100,zero_jac(set(nl.split()),set(tl.split())),zero_jac(grams(nl),grams(tl)))
        return np.asarray((*base,exact,conflict,int(rare),len(original),len(original)+int(rare),*ext,*latin),dtype=np.float32)
    def matrix(self,query,candidates,targets,rare):
        return np.asarray([self.pair(query,tuple(targets[i][1:4]),targets[i][4],i in rare) for i in candidates],dtype=np.float32).reshape(-1,39)
    def close(self): self.icu.close()
