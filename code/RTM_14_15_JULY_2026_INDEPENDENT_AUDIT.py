#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent second implementation for auditing the RTM 14→15 July 2026 package.

This file intentionally does NOT import the primary validation engine.  It reimplements
from scratch:
  * standard Hebrew gematria,
  * the locked number/date string constructors,
  * the 14 July 9/9 gate,
  * the 15 July 9/9 gate,
  * the 1900–2100 exhaustive date and consecutive-pair scan,
  * exact finite-space counts for the uniform-letter null.

The purpose is implementation cross-checking, not a new statistical model.
All tail probabilities are conditional on the frozen template and declared null.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

from pyluach import dates as pydates
from scipy.stats import norm

START = date(1900, 1, 1)
END = date(2100, 12, 31)
D1_OBS = date(2026, 7, 14)
D2_OBS = date(2026, 7, 15)

FIRST_OBS = 320
SURNAME_OBS = 292
FULL_OBS = 612
AGE_OBS = 44
BIRTH_OBS = 1982
APT_OBS = 58
RTM116 = 116

AGE_LO, AGE_HI = 18, 90
APT_LO, APT_HI = 1, 120

V = {
    'א':1,'ב':2,'ג':3,'ד':4,'ה':5,'ו':6,'ז':7,'ח':8,'ט':9,'י':10,
    'כ':20,'ך':20,'ל':30,'מ':40,'ם':40,'נ':50,'ן':50,'ס':60,'ע':70,
    'פ':80,'ף':80,'צ':90,'ץ':90,'ק':100,'ר':200,'ש':300,'ת':400,
}
BASE_VALUES = (1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400)

def g(s: str) -> int:
    return sum(V.get(c, 0) for c in s)

ONES = {0:'',1:'אחת',2:'שתיים',3:'שלוש',4:'ארבע',5:'חמש',6:'שש',7:'שבע',8:'שמונה',9:'תשע'}
TEENS = {10:'עשר',11:'אחת עשרה',12:'שתים עשרה',13:'שלוש עשרה',14:'ארבע עשרה',15:'חמש עשרה',16:'שש עשרה',17:'שבע עשרה',18:'שמונה עשרה',19:'תשע עשרה'}
TENS = {20:'עשרים',30:'שלושים',40:'ארבעים',50:'חמישים',60:'שישים',70:'שבעים',80:'שמונים',90:'תשעים'}
HUNDS = {100:'מאה',200:'מאתיים',300:'שלוש מאות',400:'ארבע מאות',500:'חמש מאות',600:'שש מאות',700:'שבע מאות',800:'שמונה מאות',900:'תשע מאות'}
DIG = {0:'אפס',1:'אחד',2:'שתיים',3:'שלוש',4:'ארבע',5:'חמש',6:'שש',7:'שבע',8:'שמונה',9:'תשע'}
MONTH_G = {1:'בינואר',2:'בפברואר',3:'במרץ',4:'באפריל',5:'במאי',6:'ביוני',7:'ביולי',8:'באוגוסט',9:'בספטמבר',10:'באוקטובר',11:'בנובמבר',12:'בדצמבר'}
MONTH_H = {1:'בניסן',2:'באייר',3:'בסיון',4:'בתמוז',5:'באב',6:'באלול',7:'בתשרי',8:'בחשון',9:'בכסלו',10:'בטבת',11:'בשבט',12:'באדר',13:'באדר ב'}

def under100(n:int)->str:
    if n < 10: return ONES[n]
    if n < 20: return TEENS[n]
    t, o = (n//10)*10, n%10
    return TENS[t] if not o else f"{TENS[t]} ו{ONES[o]}"

def under1000(n:int)->str:
    if n < 100: return under100(n)
    h, r = (n//100)*100, n%100
    if not r: return HUNDS[h]
    join = ' ו' if r < 20 else ' '
    return HUNDS[h] + join + under100(r)

def words(n:int)->str:
    if not 0 <= n <= 9999: raise ValueError(n)
    if n < 1000: return under1000(n)
    th, r = n//1000, n%1000
    if th == 1: p='אלף'
    elif th == 2: p='אלפיים'
    else:
        p={3:'שלושת אלפים',4:'ארבעת אלפים',5:'חמשת אלפים',6:'ששת אלפים',7:'שבעת אלפים',8:'שמונת אלפים',9:'תשעת אלפים'}[th]
    return p if r == 0 else p+' '+under1000(r)

def digits(n:int, one='אחד')->str:
    out=[]
    for c in str(n):
        d=int(c); out.append(one if d==1 else DIG[d])
    return ' '.join(out)

def compact(d:date)->int: return int(f"{d.day}{d.month}")
def ddmm(d:date)->int: return int(f"{d.day:02d}{d.month:02d}")

def hval(d:date, month_map: Dict[int,str] | None = None)->int:
    hm = month_map or MONTH_H
    h=pydates.GregorianDate(d.year,d.month,d.day).to_heb()
    return h.day + g(hm[h.month]) + (h.year % 1000)

def full_date(d:date)->str: return f"{words(d.day)} {MONTH_G[d.month]} {words(d.year)}"
def date_stmt(d:date, one:str)->str: return f"תאריך היום הוא {digits(compact(d),one)}"

Q = g('מה התאריך העברי של היום')
QW = g(words(Q))
MODEL = g('מציאות מתעדכנת רטרואקטיבית')
BEN = g('בן')
APT_LABEL = g('דירה')

# Small memo tables avoid repeatedly formatting strings during scans.
WG = [g(words(n)) for n in range(10000)]
DM = [g(digits(n,'אחד')) for n in range(10000)]
DF = [g(digits(n,'אחת')) for n in range(10000)]

def get(a, n):
    return a[n] if 0 <= n < len(a) else None

def d2_families(d:date, first:int, surname:int, age:int, birth:int, month_map=None)->Dict[str,bool]:
    H=hval(d, month_map); c=compact(d); z=ddmm(d); y=d.year
    r=Q-H; rw=get(WG,r); dsf=g(date_stmt(d,'אחת')); dsm=g(date_stmt(d,'אחד')); ph=g(full_date(d))
    x=H-c; xd=get(DM,x)
    A=xd is not None and xd==2*H and 2*H-x-H==c
    B=rw is not None and rw==dsf
    C=dsm-(MODEL-dsf)==H
    aw=get(WG,age)
    if aw is None: return {k:False for k in 'ABCDEGHIJ'}
    full=first+surname; phrase=full+BEN+aw
    D=rw is not None and dsf-r==phrase
    fw=get(WG,full)
    E=fw is not None and QW==ph and fw==ph
    G=ph-phrase-H==first
    Hf=H-(2*ph-birth-y)==age
    ad=get(DM,age)
    I=ad is not None and ad-(((z-H)-aw)-age)-c==first
    fw_first=get(WG,first)
    J=fw_first is not None and y-(2*fw_first-birth)==aw
    return dict(A=A,B=B,C=C,D=D,E=E,G=G,H=Hf,I=I,J=J)

def d1_families(d:date, first:int, surname:int, age:int, birth:int, apt:int, month_map=None)->Dict[str,bool]:
    H=hval(d,month_map); c=compact(d); ph=g(full_date(d)); full=first+surname
    hw=get(WG,H); aw=get(WG,apt); hd=get(DM,H); fw=get(WG,full); fdf=get(DF,full); fdm=get(DM,full)
    gap=Q-H; gw=get(WG,gap); gd=get(DM,gap)
    if None in (hw,aw,hd,fw,fdf,fdm,gw,gd): return {k:False for k in 'ABCDEFGHI'}
    apt_phrase=APT_LABEL+aw
    A=hw==apt_phrase
    B=QW==fw
    C=QW-hd-hw==apt
    D=c-gap==apt
    E=gw+gd-Q==apt_phrase
    F=fdf-Q-RTM116==2*c
    G=c-(fdm-H)==age
    Hf=(QW-Q)+(fdf-Q)+(fdm-Q)-H+c==first
    I=ph-birth-c-RTM116==apt
    return dict(A=A,B=B,C=C,D=D,E=E,F=F,G=G,H=Hf,I=I)

def dates_iter()->Iterable[date]:
    d=START
    while d<=END:
        yield d; d+=timedelta(days=1)

def sumdist(length:int)->Counter:
    c=Counter({0:1})
    for _ in range(length):
        n=Counter()
        for s,k in c.items():
            for v in BASE_VALUES: n[s+v]+=k
        c=n
    return c

C3=sumdist(3); C4=sumdist(4)


def d2_core(d:date, month_map=None)->bool:
    H=hval(d,month_map); c=compact(d); r=Q-H
    dsf=g(date_stmt(d,'אחת')); dsm=g(date_stmt(d,'אחד')); ph=g(full_date(d))
    x=H-c; xd=get(DM,x); rw=get(WG,r)
    return (xd is not None and xd==2*H and (2*H-x-H)==c
            and rw is not None and rw==dsf
            and dsm-(MODEL-dsf)==H
            and QW==ph)

def candidate_from_DG(d:date, age:int, month_map=None):
    H=hval(d,month_map); r=Q-H
    if r<0: return None
    dsf=g(date_stmt(d,'אחת')); ph=g(full_date(d)); aw=get(WG,age)
    if aw is None:return None
    phrase=dsf-r
    full=phrase-BEN-aw
    first=ph-phrase-H
    surname=full-first
    if first<0 or surname<0:return None
    return first,surname,d.year-age

def scan(month_map=None):
    ds=list(dates_iter()); h1=[];h2=[];pairs=[]
    for d in ds:
        if all(d1_families(d,FIRST_OBS,SURNAME_OBS,AGE_OBS,BIRTH_OBS,APT_OBS,month_map).values()): h1.append(d.isoformat())
        if all(d2_families(d,FIRST_OBS,SURNAME_OBS,AGE_OBS,BIRTH_OBS,month_map).values()): h2.append(d.isoformat())
    for a,b in zip(ds[:-1],ds[1:]):
        if (all(d1_families(a,FIRST_OBS,SURNAME_OBS,AGE_OBS,BIRTH_OBS,APT_OBS,month_map).values())
            and all(d2_families(b,FIRST_OBS,SURNAME_OBS,AGE_OBS,BIRTH_OBS,month_map).values())
            and g(full_date(b))==QW): pairs.append((a.isoformat(),b.isoformat()))
    return h1,h2,pairs

def exact_letter_nulls(month_map=None):
    age_n=AGE_HI-AGE_LO+1; names_n=22**7; nd=(END-START).days+1; npairs=nd-1; apt_n=APT_HI-APT_LO+1
    fixed_num=0; joint_num=0; seq_num=0; fixed_rows=[];joint_rows=[];seq_rows=[]
    # fixed day 2
    for age in range(AGE_LO,AGE_HI+1):
        cand=candidate_from_DG(D2_OBS,age,month_map)
        if not cand: continue
        f,s,birth=cand; w=C3.get(f,0)*C4.get(s,0)
        if w and all(d2_families(D2_OBS,f,s,age,birth,month_map).values()):
            fixed_num+=w; fixed_rows.append((D2_OBS.isoformat(),f,s,age,birth,w))
    # joint and sequential
    d=START
    while d<=END:
        if d2_core(d,month_map):
            for age in range(AGE_LO,AGE_HI+1):
                cand=candidate_from_DG(d,age,month_map)
                if not cand: continue
                f,s,birth=cand; w=C3.get(f,0)*C4.get(s,0)
                if not w: continue
                if all(d2_families(d,f,s,age,birth,month_map).values()):
                    joint_num+=w; joint_rows.append((d.isoformat(),f,s,age,birth,w))
                    if d>START:
                        d1=d-timedelta(days=1)
                        for apt in range(APT_LO,APT_HI+1):
                            if all(d1_families(d1,f,s,age,birth,apt,month_map).values()) and g(full_date(d))==QW:
                                seq_num+=w; seq_rows.append((d1.isoformat(),d.isoformat(),f,s,age,birth,apt,w))
        d+=timedelta(days=1)
    def pack(num,den,rows):
        p=num/den; return {'numerator':num,'denominator':den,'p':p,'z_one_sided':float(norm.isf(p)),'rows':rows}
    return {
        'fixed_letters':pack(fixed_num,age_n*names_n,fixed_rows),
        'joint_letters':pack(joint_num,nd*age_n*names_n,joint_rows),
        'sequential_letters':pack(seq_num,npairs*age_n*names_n*apt_n,seq_rows),
    }

def deterministic_ledger(ledger:Path):
    data=json.loads(ledger.read_text(encoding='utf-8')); bad=[]
    for x in data['strings']:
        if g(x['hebrew'])!=int(x['value']): bad.append(x['id'])
    allowed=set('0123456789+- ()')
    for x in data['arithmetic_checks']:
        e=x['expression']
        if any(c not in allowed for c in e): bad.append(x['id']+'-unsafe'); continue
        if int(eval(e,{'__builtins__':{}},{}))!=int(x['expected']): bad.append(x['id'])
    return {'strings':len(data['strings']),'equations':len(data['arithmetic_checks']),'failures':bad,'pass':not bad}

def main():
    base=Path(__file__).resolve().parent
    ledger=base/'RTM_15_JULY_2026_EXACT_LEDGER.json'
    out={'implementation':'independent_second_implementation','conditional_template_null':True}
    out['ledger_audit']=deterministic_ledger(ledger)
    out['observed_day1']=d1_families(D1_OBS,FIRST_OBS,SURNAME_OBS,AGE_OBS,BIRTH_OBS,APT_OBS)
    out['observed_day2']=d2_families(D2_OBS,FIRST_OBS,SURNAME_OBS,AGE_OBS,BIRTH_OBS)
    out['base_scan']={}
    h1,h2,pairs=scan()
    out['base_scan'].update(day1_hits=h1,day2_hits=h2,sequential_hits=pairs)
    # One combined plausible month-spelling sensitivity pass.
    alt=MONTH_H.copy(); alt.update({3:'בסיוון',8:'במרחשון',12:'באדר א',13:'באדר ב'})
    a1,a2,ap=scan(alt)
    out['month_spelling_sensitivity']={'mapping_changes':{3:'בסיוון',8:'במרחשון',12:'באדר א',13:'באדר ב'},'day1_hits':a1,'day2_hits':a2,'sequential_hits':ap}
    out['exact_letter_nulls']=exact_letter_nulls()
    Path('RTM_14_15_INDEPENDENT_AUDIT_RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
