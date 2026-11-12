"""Check every mapping locality's lat,lon against GeoNames: a populated place with the locality's name (or one of
locality_aka) must exist in that country near the coordinates.

    python -m tools.coords_check [OUT.csv]   # default _output/checks/coords_check.csv

GeoNames files in ~/geonames (GEONAMES_DIR to change): countryInfo.txt and admin1CodesASCII.txt, and populated.txt
(the populated places, feature class P, of allCountries.txt: 5.2M) or else cities500.txt (places of 500+ people):
    cd ~/geonames && curl -sSfLO https://download.geonames.org/export/dump/allCountries.zip &&
    unzip -p allCountries.zip allCountries.txt | awk -F'\\t' '$7=="P"' > populated.txt && rm allCountries.zip
Takes about 2 min and 2 GB of memory with populated.txt.

Issue per row (most serious first):
  other_country  the coordinates are nearer to places of another country and >5 km from any place of its own country
  wrong_state    the place of this name in the row's state is far away, and the coordinates lie in another state
  far            places with this name exist (in the row's state, when it has one), but the nearest is farther than
                 the radius (10 km, 20 km for 100k+ people, 35 km for 1M+)
  no_name        no place of this name in the country in GeoNames (very small place, spelling or a typo), or, for a
                 row with a state, only in other states: it is never matched to a place in another state
  ok             a place with this name within the radius
"""
import csv
import os
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

GEONAMES = os.environ.get('GEONAMES_DIR', os.path.expanduser('~/geonames'))
R_EARTH = 6371.0
ISSUES = ['other_country', 'wrong_state', 'far', 'no_name', 'ok']
CA_PROV = {'AB': 'Alberta', 'BC': 'British Columbia', 'MB': 'Manitoba', 'NB': 'New Brunswick',
           'NL': 'Newfoundland and Labrador', 'NS': 'Nova Scotia', 'NT': 'Northwest Territories', 'NU': 'Nunavut',
           'ON': 'Ontario', 'PE': 'Prince Edward Island', 'QC': 'Quebec', 'SK': 'Saskatchewan', 'YT': 'Yukon'}
ABBREV = {'st': 'saint', 'ste': 'sainte', 'mt': 'mount', 'ft': 'fort'}
SUFFIX = r' (oblast|krai|republic|province|prefecture|state|region|county|division|district|governorate)$'


def norm(s):
    """Name for matching: no accents, case or punctuation."""
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c)).casefold()
    for a, b in (('ß', 'ss'), ('ı', 'i'), ('ø', 'o'), ('ł', 'l'), ('đ', 'd'), ('æ', 'ae')):
        s = s.replace(a, b)
    s = re.sub(r"[\-'’`.,()/]", ' ', s)
    s = re.sub(r'^(city of|municipality of|town of) ', '', s.strip())
    # abbreviated words as GeoNames writes them out (St Paul / Saint Paul; Pavlo, 2026-10-25); both sides go through
    # this, so St. Gallen still meets GeoNames' alternate name St. Gallen of Sankt Gallen
    s = re.sub(r'\b(st|ste|mt|ft)\b', lambda m: ABBREV[m.group(1)], s)
    return re.sub(r'\s+', ' ', s).strip()


def fits(state, cc, adm):
    """Whether the mapping's state can be this GeoNames admin1 name (None when either is missing): the same name, a
    Canadian province code, or an abbreviation whose letters appear in order in the name from its first letter
    (QLD, CHIH, NSW; the US codes are GeoNames' own)."""
    if not state or not adm:
        return None
    if cc == 'CA' and state.upper() in CA_PROV:
        return norm(CA_PROV[state.upper()]) == norm(adm)
    st = re.sub(SUFFIX, '', norm(state)).replace('republic of ', '')
    an = re.sub(SUFFIX, '', norm(adm)).replace('republic of ', '')
    if st == an or (len(st) > 3 and (st in an or an in st)):
        return True
    if state.isupper() and 2 <= len(state) <= 5:
        letters = an.replace(' ', '')
        if not letters.startswith(st[0]):
            return False
        pos = 0
        for ch in st:
            pos = letters.find(ch, pos)
            if pos < 0:
                return False
            pos += 1
        return True
    return False


def radius(pop):
    return 35.0 if pop >= 1_000_000 else 20.0 if pop >= 100_000 else 10.0


def _xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def _km(chord):
    return 2 * R_EARTH * np.arcsin(np.clip(chord / 2, 0, 1))


def aka(cell):
    return [a.strip() for a in (cell or '').strip('[]').split(',') if a.strip()]


def check(rows, gdir=GEONAMES):
    """One result dict per mapping row (dicts of mapping.csv), with 'issue' and the GeoNames place it was matched to
    (match_*), up to 3 other places of the name (alternatives: [name, admin1, lat, lon, km]) and the nearest place."""
    iso3_to2 = {}
    for line in open(os.path.join(gdir, 'countryInfo.txt'), encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        p = line.rstrip('\n').split('\t')
        iso3_to2[p[1]] = p[0]
    iso3_to2.setdefault('XKX', 'XK')  # Kosovo
    admin1 = {}
    regions = defaultdict(list)  # country -> its first-level regions
    for line in open(os.path.join(gdir, 'admin1CodesASCII.txt'), encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        admin1[p[0]] = p[1]
        regions[p[0].split('.')[0]].append(p[1])
    wanted = {norm(n) for r in rows for n in [r['locality'], *aka(r['locality_aka'])] if n.strip()}
    countries = {iso3_to2.get(r['iso3'], '') for r in rows}
    names = defaultdict(list)  # (country, normalised name) -> places, only for names the mapping uses
    plat, plon, pcc, pname, ppop, padm = [], [], [], [], [], []
    src = 'populated.txt' if os.path.exists(os.path.join(gdir, 'populated.txt')) else 'cities500.txt'
    for line in open(os.path.join(gdir, src), encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if p[8] not in countries:
            continue
        i = len(plat)
        plat.append(float(p[4]))
        plon.append(float(p[5]))
        pcc.append(p[8])
        pname.append(p[1])
        ppop.append(int(p[14] or 0))
        padm.append(admin1.get(f'{p[8]}.{p[10]}', p[10]))
        for n in {p[1], p[2], *p[3].split(',')}:
            k = norm(n) if n else ''
            if k in wanted:
                names[(p[8], k)].append(i)
    plat, plon, ppop = np.array(plat), np.array(plon), np.array(ppop)
    pts = _xyz(plat, plon)
    tree_all = cKDTree(pts)
    by_cc = defaultdict(list)
    for i, c in enumerate(pcc):
        by_cc[c].append(i)
    trees = {c: (cKDTree(pts[ix]), np.array(ix)) for c, ix in by_cc.items()}
    out = []
    for r in rows:
        lat, lon = float(r['lat']), float(r['lon'])
        cc, state = iso3_to2.get(r['iso3'], ''), r['state'].strip()
        q = _xyz(np.array([lat]), np.array([lon]))[0]
        res = dict(id=r['id'], locality=r['locality'], locality_aka=r['locality_aka'], state=state,
                   country=r['country'], iso3=r['iso3'], population=r['population_locality'], lat=lat, lon=lon)
        d_any, i_any = tree_all.query(q)
        res.update(nearest_place=pname[i_any], nearest_country=pcc[i_any], nearest_km=round(float(_km(d_any)), 1))
        own = trees.get(cc)
        i_own, d_own = None, float('inf')
        if own:
            d, j = own[0].query(q)
            i_own, d_own = int(own[1][j]), float(_km(d))
            res.update(coord_admin1=padm[i_own], nearest_own=pname[i_own], nearest_own_km=round(d_own, 1))
        cand = sorted({i for n in [r['locality'], *aka(r['locality_aka'])] for i in names.get((cc, norm(n)), [])})
        issue = 'no_name'
        in_state = []
        if cand:
            in_state = [i for i in cand if fits(state, cc, padm[i])]
            pool = np.array(in_state or cand)
            d = _km(np.linalg.norm(pts[pool] - q, axis=1))
            order = np.argsort(d)
            b = int(pool[order[0]])
            res.update(match=pname[b], match_admin1=padm[b], match_pop=int(ppop[b]), match_lat=float(plat[b]),
                       match_lon=float(plon[b]), match_km=round(float(d[order[0]]), 1), same_name_places=len(cand),
                       alternatives=[[pname[int(pool[k])], padm[int(pool[k])], float(plat[int(pool[k])]),
                                      float(plon[int(pool[k])]), round(float(d[k]), 1)] for k in order[1:4]])
            pop = max(int(ppop[b]), int(float(r['population_locality'] or 0)))
            issue = 'ok' if d[order[0]] <= radius(pop) else 'far'
            if issue == 'far' and state and i_own is not None and padm[i_own] != padm[b]:
                fm, fc = fits(state, cc, padm[b]), fits(state, cc, padm[i_own])
                if fm and not fc:
                    issue = 'wrong_state'  # the state is the named place's; the coordinates lie in another one
                elif fc and not fm:
                    issue = 'no_name'  # the only same-name places are in other states: this one is not in GeoNames
        if issue in ('far', 'no_name') and cand and state and not in_state and \
                any(fits(state, cc, a) for a in regions.get(cc, ())):
            # the row's state is a real region of the country, and the places of this name are all in other states and
            # far away (St Paul, OR is not the one in Minnesota): no match, so no move to offer (Pavlo, 2026-10-25)
            for k in ('match', 'match_admin1', 'match_pop', 'match_lat', 'match_lon', 'match_km'):
                res.pop(k, None)
            res.update(alternatives=[], other_states=sorted({padm[i] for i in cand})[:5])
            issue = 'no_name'
        if pcc[i_any] != cc and d_own > 5:
            issue = 'other_country'
        res['issue'] = issue
        out.append(res)
    return out


KEYS = ['issue', 'id', 'locality', 'state', 'country', 'iso3', 'population', 'lat', 'lon', 'match', 'match_admin1',
        'match_pop', 'match_lat', 'match_lon', 'match_km', 'coord_admin1', 'same_name_places', 'nearest_own',
        'nearest_own_km', 'nearest_place', 'nearest_country', 'nearest_km']


def _check():
    assert fits('QLD', 'AU', 'Queensland') and fits('CHIH', 'MX', 'Chihuahua') and fits('NSW', 'AU', 'New South Wales')
    assert fits('YT', 'CA', 'Yukon') and not fits('BC', 'CA', 'Yukon')
    assert fits('OR', 'US', 'Oregon') and not fits('OR', 'US', 'Massachusetts')
    assert fits('Lviv oblast', 'UA', 'Lviv') and fits('Anambra', 'NG', 'Anambra')
    assert not fits('Anambra', 'NG', 'Enugu State')
    assert fits('', 'US', 'Texas') is None
    assert fits('OR', 'US', 'Minnesota') is False  # St Paul, OR: the Minnesota one is never its match
    assert norm('Saarbrücken') == norm('Saarbrucken') == 'saarbrucken' and norm('Ivano-Frankivsk') == 'ivano frankivsk'
    assert norm('St Paul') == norm('St. Paul') == norm('Saint Paul') and norm('Ste-Agathe') == norm('Sainte Agathe')
    assert norm('Mt Vernon') == norm('Mount Vernon') and norm('Ft. Worth') == norm('Fort Worth')
    assert norm('Stuttgart') == 'stuttgart' and norm('Westmount') == 'westmount'  # only whole words


if __name__ == '__main__':
    _check()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: mapping.csv
    path = sys.argv[1] if len(sys.argv) > 1 else '_output/checks/coords_check.csv'
    with open('mapping.csv', encoding='utf-8') as f:
        result = check(list(csv.DictReader(f)))
    result.sort(key=lambda x: (ISSUES.index(x['issue']), -float(x.get('match_km') or 0)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, KEYS, extrasaction='ignore')
        w.writeheader()
        w.writerows(result)
    counts = {k: sum(x['issue'] == k for x in result) for k in ISSUES}
    print(counts, '->', path)
