"""How many videos each person added to the dataset on each of the last DAYS days, for the review hub.

    python dashboards/added_videos.py [WORK] [OUT]   # nightly, by build_dataset_page.sh after analysis.py

WORK is the dataset page's copy (default ~/dataset-page/work), with analysis_data/video_contributors.csv, which
analysis.py has just brought up to date (who added each video and when: utils/analytics/contributors.py). OUT
(default _output/proposals/added_videos.json next to this repo's top folder): {built, people: [names, most videos
first], days: [{day: YYYY-MM-DD, counts: {name: videos}}, newest first, every day of the DAYS, none added too]}.
Days are this machine's local dates.
"""
import collections
import csv
import json
import os
import sys
import time

from datetime import date, datetime, timedelta
from pathlib import Path

DAYS = 10
WORK = os.path.expanduser('~/dataset-page/work')
OUT = str(Path(__file__).resolve().parent.parent / '_output/proposals/added_videos.json')


def build(work=WORK, out=OUT, today=None):
    today = today or date.today()
    days = [(today - timedelta(days=i)).isoformat() for i in range(DAYS)]
    counts = collections.defaultdict(collections.Counter)  # day -> who -> videos
    with open(os.path.join(work, 'analysis_data', 'video_contributors.csv'), newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            day = datetime.fromisoformat(r['added_utc'].replace('Z', '+00:00')).astimezone().date().isoformat()
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
