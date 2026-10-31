"""How many videos each person added to the dataset on each of the last DAYS days, for the review hub.

    python dashboards/added_videos.py [WORK] [OUT]   # nightly, by build_dataset_page.sh after analysis.py

WORK is the dataset page's copy (default ~/dataset-page/work), with analysis_data/video_contributors.csv, which
analysis.py has just brought up to date (who added each video and when: utils/analytics/contributors.py). OUT
(default _output/proposals/added_videos.json next to this repo's top folder): {built, people: [names, most videos
first], days: [{day: YYYY-MM-DD, counts: {name: videos}}, newest first, every day of the DAYS, none added too]}.
Days are this machine's local dates. A video applied in review counts on the day of its first Apply (the plans of
PLANS), not the day of the commit that pushed it: reviews of an evening were often pushed the next morning, so the
days came out one late (user, 2026-10-31). The credit stays with whoever it is.
"""
import collections
import csv
import glob
import json
import os
import sys
import time

from datetime import date, datetime, timedelta
from pathlib import Path

DAYS = 10
WORK = os.path.expanduser('~/dataset-page/work')
OUT = str(Path(__file__).resolve().parent.parent / '_output/proposals/added_videos.json')
PLANS = str(Path(__file__).resolve().parent.parent / '_output/proposals')  # the review plans, CHANNEL/plan.json


def applied_times(plans=PLANS):
    """video -> when a segment of it was first applied in review (a plan's applied is that time since 2026-11-01,
    True before). Before that, the night a video first was in the mapping backups (applied_backfill.json, made once
    on 2026-10-31 from lin5080's backups of mini: noon of the day before that night's 01:30 backup)."""
    try:
        with open(os.path.join(plans, 'applied_backfill.json')) as f:
            times = json.load(f)
    except (OSError, ValueError):
        times = {}
    for path in glob.glob(os.path.join(plans, '*', 'plan.json')):
        try:
            with open(path) as f:
                plan = json.load(f)
        except (OSError, ValueError):
            continue
        for p in plan:
            for t in (s.get('applied') for e in p.get('entries', []) for s in e['segments']):
                if isinstance(t, (int, float)) and not isinstance(t, bool):
                    times[p['video']] = min(t, times.get(p['video'], t))
    return times


def build(work=WORK, out=OUT, today=None, plans=PLANS):
    today = today or date.today()
    days = [(today - timedelta(days=i)).isoformat() for i in range(DAYS)]
    applied = applied_times(plans)
    counts = collections.defaultdict(collections.Counter)  # day -> who -> videos
    with open(os.path.join(work, 'analysis_data', 'video_contributors.csv'), newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            added = datetime.fromisoformat(r['added_utc'].replace('Z', '+00:00')).timestamp()
            # a later Apply is another segment of a video added before
            day = datetime.fromtimestamp(min(added, applied.get(r['video'], added))).date().isoformat()
            if day in days:
                counts[day][r['contributor']] += 1
    total = sum(counts.values(), collections.Counter())
    data = dict(built=time.time(), people=[who for who, _ in total.most_common()],
                days=[dict(day=d, counts=dict(counts[d])) for d in days])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'w') as f:
        json.dump(data, f)
    os.replace(out + '.tmp', out)
    return data


if __name__ == '__main__':
    d = build(*sys.argv[1:3])
    print(f"videos added in the last {len(d['days'])} days:", ', '.join(
        f"{who} {sum(x['counts'].get(who, 0) for x in d['days'])}" for who in d['people']) or 'none')
