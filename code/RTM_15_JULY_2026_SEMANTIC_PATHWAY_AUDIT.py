#!/usr/bin/env python3
import json
from pathlib import Path

VALUES = dict(zip('אבגדהוזחטיכלמנסעפצקרשת', [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400]))
VALUES.update({'ך':20,'ם':40,'ן':50,'ף':80,'ץ':90})

PHRASE = 'אני יודע שהמציאות מתעדכנת רטרואקטיבית'
EXPECTED = 2934

def gematria(s: str) -> int:
    return sum(VALUES.get(ch, 0) for ch in s)

def main():
    checks = []
    got = gematria(PHRASE)
    checks.append({'id':'S01','type':'string','expression':PHRASE,'expected':EXPECTED,'actual':got,'pass':got==EXPECTED})
    actual = 2934 - 792
    checks.append({'id':'E01','type':'arithmetic','expression':'2934 - 792','expected':2142,'actual':actual,'pass':actual==2142})
    actual = 2142 - 1507
    checks.append({'id':'E02','type':'arithmetic','expression':'2142 - 1507','expected':635,'actual':actual,'pass':actual==635})
    result = {
        'scope':'deterministic semantic pathway supplement; non-gating',
        'phrase': PHRASE,
        'anchors': {
            '792':'א׳ באב תשפ״ו / locked same-day Hebrew-date value',
            '1507':'15|07 / zero-padded Gregorian date encoding',
            '635':'existing core node represented by שש שלוש חמש in Closure family A',
            '157':'15|7 / terminal of the established core date-return branch'
        },
        'checks': checks,
        'pass': all(c['pass'] for c in checks)
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['pass']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
