"""Localities for the globe widget: where footage is, and which localities got segments in the last day and RECENT_H.

The mapping has no dates per segment, so "recent" compares the mapping here (the review repo's mapping.csv, reviews not
pushed yet included) with GitHub's main as it was a day and RECENT_H hours ago (git history of mapping.csv).
A segment is a (video, start) pair, compared across all localities (ids get renumbered). Stdlib only, Python 3.9+.
"""
import csv
import io
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(os.environ.get('CROWD_REPO', Path(__file__).resolve().parents[2]))  # the repo this file is in
CACHE_S = 3600
RECENT_H = 72  # amber: segments new in the last 3 days (a week took in the merged mapping-NAME files: half the data)
_cache = {}


def segments(text):
    """{locality id: (lat, lon, name, hours, {(video, start), ...})} of a mapping.csv's text."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(row['lat']), float(row['lon'])
        except ValueError:
            continue
        videos = [v.strip() for v in row['videos'].strip('[]').split(',') if v.strip()]
        starts, ends = json.loads(row['start_time'] or '[]'), json.loads(row['end_time'] or '[]')
        segs, hours = set(), 0.0
        for v, ss, es in zip(videos, starts, ends):
            for s, e in zip(ss, es):
                segs.add((v, s))
                hours += max(0, e - s) / 3600
        us = row.get('state') and row['country'] == 'United States'
        name = row['locality'] + (f", {row['state']}" if us else '')
        out[row['id']] = (lat, lon, f"{name}, {row['country']}", hours, segs)
    return out


def at(ago_s):
    """mapping.csv text on GitHub's main as it was ago_s seconds ago ('' if the history does not go back that far)."""
    when = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time() - ago_s))
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-list', '-1', f'--before={when}', 'origin/main', '--',
                          'mapping.csv'],
                         capture_output=True, text=True).stdout.strip()
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{sha}:mapping.csv'], capture_output=True,
                          text=True).stdout if sha else ''


def compare(now, day, week):
    """Points [lat, lon, hours, segments new in RECENT_H, segments new today, name] and totals."""
    points, totals = [], dict(recent=0, day=0, recent_places=0, day_places=0)
    # all segments of the old mapping, not per id: the crowd repo renumbers locality ids
    day, week = ({seg for v in old.values() for seg in v[4]} for old in (day, week))
    for lat, lon, name, hours, segs in now.values():
        w = len(segs - week) if week else 0
        d = len(segs - day) if day else 0
        totals['recent'] += w
        totals['day'] += d
        totals['recent_places'] += w > 0
        totals['day_places'] += d > 0
        points.append([round(lat, 3), round(lon, 3), round(hours, 1), w, d, name])
    return points, totals


def globe():
    """The globe's data, recomputed when mapping.csv changes, else at most every CACHE_S."""
    path = REPO / 'mapping.csv'
    key = path.stat().st_mtime
    if _cache.get('key') == key and time.time() - _cache['at'] < CACHE_S:
        return _cache['data']
    subprocess.run(['git', '-C', str(REPO), 'fetch', '-q', 'origin'], capture_output=True, timeout=120)
    now = segments(path.read_text(encoding='utf-8'))
    day, week = (segments(t) if t else {} for t in (at(86400), at(RECENT_H * 3600)))
    points, totals = compare(now, day, week)
    data = dict(built=time.time(), recent_h=RECENT_H, localities=len(points),
                hours=round(sum(p[2] for p in points)), **totals,
                points=points)
    _cache.update(key=key, at=time.time(), data=data)
    return data


def _check():
    head = 'id,locality,state,country,lat,lon,videos,start_time,end_time\n'
    old = head + '1,Delft,,Netherlands,52.0,4.36,"[a,b]","[[0],[10,50]]","[[30],[20,60]]"\n'
    new = old + '2,Gouda,,Netherlands,52.0,4.7,"[c]","[[0]]","[[3600]]"\n'
    new = new.replace('"[a,b]","[[0],[10,50]]","[[30],[20,60]]"', '"[a,b]","[[0],[10,50,70]]","[[30],[20,60,80]]"')
    pts, tot = compare(segments(new), segments(old), segments(old))
    assert {p[5]: (p[3], p[4]) for p in pts} == {'Delft, Netherlands': (1, 1), 'Gouda, Netherlands': (1, 1)}, pts
    assert tot == dict(recent=2, day=2, recent_places=2, day_places=2) and pts[1][2] == 1.0, (tot, pts)
    pts, tot = compare(segments(new), {}, {})
    assert tot['recent'] == 0, tot  # no history: nothing is called new
    renumbered = old.replace('\n1,Delft', '\n9,Delft')
    assert compare(segments(old), segments(renumbered), {})[1]['day'] == 0  # same segments under a new id: not new


if __name__ == '__main__':
    _check()
    print('globe_data check ok')
