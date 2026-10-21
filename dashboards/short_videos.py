"""Videos shorter than their last segment (the inventory's "short"), to check one by one on the status page (/short).

    python dashboards/short_videos.py           # (re)build _output/proposals/short_videos.json (cron, 04:30)
    python dashboards/short_videos.py --apply   # carry out the decisions not applied yet

Run from the repo's top folder.

The build takes the short files from the last inventory (tools/video_inventory.py, read over ssh), their segments
from mapping.csv here and their length on YouTube (yt-dlp, metadata only; kept from the last build), and suggests a
fix: the download was cut off (YouTube is longer than the file): download it again; else the mapping's end times run
past the end of the video: trim the segments to it. The page saves a decision per video in the same file.

--apply: "redownload" moves the file into redownload-DATE on its drive (the downloader fetches it again within the
hour; the old file is left for a person to delete), "trim" sets the end of each segment past the end of the file to
the file's end in mapping.csv (under the review pages' lock), dropping a segment that starts there or later and the
video from a row where it has no segment left, then pushes the mapping as the hub's "Push mapping" does.
"""
import collections
import contextlib
import csv
import fcntl
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sync_workers: the repo's top folder
import sync_workers as sw  # noqa: E402

FILE = '_output/proposals/short_videos.json'
INVENTORY = ('lin4080', '~/crowd-counts/_output/video_inventory.csv')
DECISIONS = {'redownload': 'download it again', 'trim': "trim the segments to the video's end", 'keep': 'keep as is'}
SLACK_S = 5  # as the inventory: shorter than the last segment's end by more than this
_lock = threading.Lock()


@contextlib.contextmanager
def locked():
    """One writer at a time: the page (decide), a build (cron, every morning) and --apply run in separate processes."""
    with _lock, open(FILE + '.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def load():
    if not os.path.exists(FILE):
        return dict(built=None, cases=[])
    with open(FILE) as f:
        return json.load(f)


def save(data):
    with open(FILE + '.tmp', 'w') as f:
        json.dump(data, f, indent=1)
    os.replace(FILE + '.tmp', FILE)


def youtube_lengths(videos):
    """{video: length in s on YouTube, or None when yt-dlp cannot read it (private, removed)}."""
    out = subprocess.run(['yt-dlp', '--js-runtimes', 'node', '--cookies', 'cookies.txt', '--skip-download',
                          '--ignore-errors', '--no-warnings', '--sleep-requests', '1',
                          '--print', '%(id)s %(duration)s']
                         + [f'https://www.youtube.com/watch?v={v}' for v in videos],
                         capture_output=True, text=True).stdout
    got = dict(line.split(' ', 1) for line in out.splitlines() if ' ' in line)
    return {v: float(got[v]) if got.get(v, 'NA') not in ('NA', 'None') else None for v in videos}


def suggest(case):
    f, y = case['file_s'], case['youtube_s']
    if y is None:
        return 'trim', 'not on YouTube any more: it cannot be downloaded again'
    if y > f + SLACK_S:
        return 'redownload', f'the download was cut off: YouTube has {y - f:.0f} s more'
    return 'trim', 'the file is complete: the end times run past the end of the video'


def carried(old_case, mtime):
    """What a build keeps of a video's case from the last one: all of it, but only its YouTube length when the case
    was applied and the file is newer (downloaded again since, and still short: a new case to decide)."""
    if old_case.get('applied') and mtime > old_case['applied']:
        return dict(youtube_s=old_case.get('youtube_s'))
    return old_case


def build():
    host, path = INVENTORY
    files = [r for r in csv.DictReader(io.StringIO(sw.ssh(host, f'cat {path}'))) if 'short' in r['problem'].split()]
    length = {r['video']: float(r['duration']) for r in files}
    with open('mapping.csv.lock', 'w') as lk:  # the review pages write mapping.csv under this lock
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open('mapping.csv', encoding='utf-8') as f:
            mapping = list(csv.DictReader(f))
    rows = {}  # video -> the mapping rows with it and its segments there that end past the file's end
    for row in mapping:
        vids = [v.strip() for v in row['videos'].strip('"[] ').split(',') if v.strip()]
        starts, ends = json.loads(row['start_time'] or '[]'), json.loads(row['end_time'] or '[]')
        for i, v in enumerate(vids):
            if v in length:
                seg = [[s, e] for s, e in zip(starts[i], ends[i])]
                rows.setdefault(v, []).append(dict(id=row['id'], locality=row['locality'], country=row['country'],
                                                   segments=seg, past=[s for s in seg if s[1] > length[v] + SLACK_S]))
    titles = {}
    if os.path.exists('mapping_metadata.csv'):
        with open('mapping_metadata.csv', encoding='utf-8') as f:
            titles = {r['video']: r['title'] for r in csv.DictReader(f) if r['video'] in length}
    old = {c['video']: c for c in load()['cases']}
    todo = [v for v in length if 'youtube_s' not in old.get(v, {})]  # looked up once
    yt = youtube_lengths(todo) if todo else {}
    cases = []
    for r in files:
        v = r['video']
        if v not in rows:  # left the mapping since the inventory
            continue
        o = carried(old.get(v, {}), int(r['mtime']))
        c = dict(video=v, drive=r['folder'], file_s=length[v], height=r['height'], title=titles.get(v, ''),
                 rows=rows[v], youtube_s=yt[v] if v in yt else o.get('youtube_s'),
                 decision=o.get('decision'), note=o.get('note', ''), applied=o.get('applied'))
        c['suggestion'], c['why'] = suggest(c)
        cases.append(c)
    cases.sort(key=lambda c: (c['suggestion'], c['video']))
    with locked():  # the YouTube lookups take minutes: keep a decision made on the page meanwhile
        latest = {c['video']: c for c in load()['cases']}
        for c in cases:
            now, then = latest.get(c['video'], {}), old.get(c['video'], {})
            if (now.get('decision'), now.get('note')) != (then.get('decision'), then.get('note')):
                c['decision'], c['note'] = now.get('decision'), now.get('note', '')
        save(dict(built=time.time(), cases=cases))
    return cases


def decide(video, decision, note):
    """Saves the decision (or None to undo it) and note for one video."""
    if decision not in DECISIONS and decision is not None:
        raise ValueError(decision)
    with locked():
        data = load()
        for c in data['cases']:
            if c['video'] == video:
                c['decision'], c['note'] = decision, str(note or '')[:500]
                save(data)
                return True
    return False


def trim_row(row, video, end):
    """row (a mapping.csv row as a dict of strings) with video's segments ending at most at end; True if changed. A
    row is never left without videos (the mapping has no such rows): then it is left as it is, for a person to
    decide."""
    vids = [v.strip() for v in row['videos'].strip('"[] ').split(',') if v.strip()]
    if video not in vids:
        return False
    i = vids.index(video)
    nested = {k: json.loads(row[k]) for k in ('time_of_day', 'start_time', 'end_time')}
    segs = list(zip(nested['time_of_day'][i], nested['start_time'][i], nested['end_time'][i]))
    keep = [(t, s, min(e, end)) for t, s, e in segs if s < end - 1]
    if keep == segs or (not keep and len(vids) == 1):
        return False
    if keep:
        for k, col in zip(('time_of_day', 'start_time', 'end_time'), zip(*keep)):
            nested[k][i] = list(col)
    else:  # no segment left: the video leaves this row, with its entry in every per-video column
        for k in nested:
            del nested[k][i]
        for k in ('videos', 'vehicle_type', 'upload_date', 'channel'):
            items = row[k].strip('"[] ').split(',')
            if len(items) == len(vids):
                row[k] = '[' + ','.join(items[:i] + items[i + 1:]) + ']'
    for k, v in nested.items():
        row[k] = json.dumps(v, separators=(',', ':'))
    return True


# run on lin4080 by apply: the YOLO CSVs ({video}_{start}_{fps}.csv in each data folder's bbox/) of the trimmed
# segments without the frames past the new end (user, 2026-10-21). frame-count is 1 at the segment's start and
# main.py detects from start to end - 1 s, so a frame is kept up to (end - 1 - start) * fps; a segment dropped whole
# leaves no CSV. The original goes to bbox-trimmed-DATE beside bbox/ (count_detections.py reads only bbox/).
TRIM_BBOX = """
import csv, glob, json, os, sys, time
day = time.strftime('%Y-%m-%d')
folders = (glob.glob('/media/salam/*/pedestrians*/data/bbox')
           + glob.glob(os.path.expanduser('~/Documents/GitHub/*/data/bbox')))
for vid, start, end in json.loads(sys.argv[1]):
    for folder in folders:
        for path in glob.glob(os.path.join(folder, f'{vid}_{start}_*.csv')):
            fps = int(path[:-4].rsplit('_', 1)[1])
            last = (end - 1 - start) * fps
            aside = os.path.join(os.path.dirname(folder), f'bbox-trimmed-{day}')
            os.makedirs(aside, exist_ok=True)
            old = os.path.join(aside, os.path.basename(path))
            os.replace(path, old)
            if last > 0:
                with open(old, newline='') as f, open(path + '.tmp', 'w', newline='') as g:
                    rows, out = csv.reader(f), csv.writer(g, lineterminator='\\n')
                    head = next(rows)
                    out.writerow(head)
                    i = head.index('frame-count')
                    out.writerows(r for r in rows if int(float(r[i])) <= last)
                os.replace(path + '.tmp', path)
            print(f'{os.path.basename(path)}: ' + (f'frames after {last} left out' if last > 0 else 'set aside'))
"""


def apply():
    import add_video  # noqa: E402: the review pages' modules, needed here only
    import review_hub
    data = load()
    todo = [c for c in data['cases'] if c.get('decision') in ('redownload', 'trim') and not c.get('applied')]
    day = time.strftime('%Y-%m-%d')
    moves = [c for c in todo if c['decision'] == 'redownload']
    if moves:
        assert all(re.fullmatch(r'[A-Za-z0-9_-]{11}', c['video']) for c in moves)
        cmd = '; '.join(f"mkdir -p {'/'.join(c['drive'].split('/')[:4])}/redownload-{day} && mv -n "
                        f"{c['drive']}/{c['video']}.mp4 {'/'.join(c['drive'].split('/')[:4])}/redownload-{day}/"
                        for c in moves)
        sw.ssh(INVENTORY[0], cmd)
    trims = {c['video']: math.floor(c['file_s']) for c in todo if c['decision'] == 'trim'}
    changed, done = 0, collections.Counter()
    if trims:
        with review_hub.ps.plan_lock(add_video.FILE_PATH):  # not halfway through an Apply
            with open(add_video.FILE_PATH, encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            done = collections.Counter()
            for row in rows:
                for v, end in trims.items():
                    if v in row['videos'] and trim_row(row, v, end):
                        done[v] += 1
            changed = sum(done.values())
            tmp = add_video.FILE_PATH + f'.{os.getpid()}.tmp'
            with open(tmp, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fields, lineterminator='\n')
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, add_video.FILE_PATH)
    left = [v for v in trims if not done[v]]  # its only video in a row: that row would be left empty
    for c in todo:
        if c['video'] in left:
            c['note'] = 'not applied: it is the only video of its row, which would be left without videos'
        else:
            c['applied'] = time.time()
    with locked():
        save(data)
    msg = review_hub.push_mapping() if trims else ''
    cuts = [(c['video'], s, trims[c['video']]) for c in todo if c['decision'] == 'trim' and c['video'] not in left
            for r in c['rows'] for s, _ in r['past']]
    yolo = sw.ssh(INVENTORY[0], f'python3 -c {shlex.quote(TRIM_BBOX)} {shlex.quote(json.dumps(cuts))}') if cuts else ''
    return (f'{len(moves)} files moved to be downloaded again, {changed} rows trimmed'
            + (f', left for a person: {", ".join(left)}' if left else '') + f'. {msg}'
            + (f' YOLO CSVs: {yolo.strip() or "none of these segments has one yet"}.' if cuts else ''))


def _check_carried():
    c = dict(youtube_s=99, decision='redownload', applied=1000)
    assert carried(c, 900) == c  # the file it was applied to
    assert carried(c, 1100) == dict(youtube_s=99)  # a newer file, still short
    assert carried(dict(decision='trim'), 5000) == dict(decision='trim')  # not applied yet


def _check():
    row = dict(videos='[a,b]', vehicle_type='[0,1]', upload_date='[1,2]', channel='[x,y]',
               time_of_day='[[0],[0,1]]', start_time='[[0],[10,500]]', end_time='[[90],[400,900]]')
    assert trim_row(row, 'b', 600) and (row['start_time'], row['end_time'], row['time_of_day']) == (
        '[[0],[10,500]]', '[[90],[400,600]]', '[[0],[0,1]]'), row
    assert trim_row(row, 'b', 300) and row['end_time'] == '[[90],[300]]' and row['time_of_day'] == '[[0],[0]]', row
    assert trim_row(row, 'b', 5) and (row['videos'], row['channel'], row['end_time']) == ('[a]', '[x]', '[[90]]'), row
    assert not trim_row(row, 'a', 90)
    assert not trim_row(row, 'a', 0)  # its only video, no segment left: the row is left as it is
    assert suggest(dict(file_s=100, youtube_s=None))[0] == 'trim'
    assert suggest(dict(file_s=100, youtube_s=200))[0] == 'redownload'
    assert suggest(dict(file_s=100, youtube_s=103))[0] == 'trim'


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Short videos</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
<text y=%22.9em%22 font-size=%2290%22>🚚</text></svg>">
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#667085;--line:#e4e7ec;--link:#2563eb;--ok:#16a34a;--warn:#d97706}
:root[data-theme=dark]{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--muted:#8b93a1;--line:#272c36;--link:#60a5fa;
 --ok:#4ade80;--warn:#fbbf24}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
 font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1100px}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}h1{font-size:20px;margin:0 0 4px}
.muted{color:var(--muted)}.small{font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0}
.card.done{opacity:.6}.head{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.facts{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0}.facts b{font-variant-numeric:tabular-nums}
.why{color:var(--warn)}.why.redownload{color:var(--link)}
label{margin-right:14px;white-space:nowrap;cursor:pointer}input[type=text]{width:min(100%,420px);padding:3px 6px;
 background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.saved{color:var(--ok);font-size:12px;margin-left:6px}
button.bulk{padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);
 cursor:pointer}
#theme{position:fixed;right:12px;bottom:12px;z-index:9999;padding:6px 10px;border:1px solid var(--line);
border-radius:16px;background:var(--bg);color:var(--fg);font-size:13px;cursor:pointer}
</style>
{{ theme|safe }}
</head><body>
<h1>🚚 Videos shorter than their last segment</h1>
<div class="muted"><a href=".">pipeline status</a> · {{ cases|length }} videos ·
 {{ cases|selectattr('decision')|list|length }} decided · built {{ hm(built) if built else 'never' }} from the
 inventory
 and mapping.csv · choices are saved as you click; applying them is a separate step</div>
<p><button class="bulk" onclick="bulk()">Take the suggestion for every undecided video</button></p>
{% for c in cases %}<div class="card {{ 'done' if c.decision }}" id="{{ c.video }}">
 <div class="head"><div><a href="https://youtu.be/{{ c.video }}" target="_blank"><b>{{ c.video }}</b></a>
  {{ c.title }}{% if c.applied %} <span class="saved">applied {{ hm(c.applied) }}</span>{% endif %}</div>
  <a href="https://youtu.be/{{ c.video }}?t={{ [(c.file_s - 20)|int, 0]|max }}" target="_blank">watch where the file
   ends ↗</a></div>
 <div class="facts"><span>file <b>{{ t(c.file_s) }}</b></span>
  <span>YouTube <b>{{ t(c.youtube_s) if c.youtube_s else 'not available' }}</b></span>
  <span class="muted">{{ c.drive.split('/')[3] if c.drive.startswith('/media/') else c.drive }} ·
  {{ c.height }}p</span></div>
 {% for r in c.rows %}<div class="small">row {{ r.id }} · {{ r.locality }}, {{ r.country }} · past the end:
  {% for s in r.past %}<a href="https://youtu.be/{{ c.video }}?t={{ s[0] }}" target="_blank">{{ t(s[0]) }}–{{ t(s[1])
  }}</a>{{ ', ' if not loop.last }}{% endfor %}
  <span class="muted">({{ r.segments|length }} segment{{ 's' if r.segments|length > 1 }} in all)</span></div>
 {% endfor %}
 <div class="small why {{ c.suggestion }}" style="margin:6px 0">suggested: {{ decisions[c.suggestion] }}:
  {{ c.why }}</div>
 <form onchange="send(this)" data-video="{{ c.video }}" data-suggestion="{{ c.suggestion }}">
  {% for k, label in decisions.items() %}<label><input type="radio" name="d" value="{{ k }}"
   {{ 'checked' if c.decision == k }}> {{ label }}</label>{% endfor %}
  <input type="text" name="note" placeholder="note (optional)" value="{{ c.note }}"><span class="saved"></span>
 </form></div>{% endfor %}
<script>
async function send(form) {
  const d = form.querySelector('input[name=d]:checked');
  const r = await fetch('short/decide', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: form.dataset.video, decision: d ? d.value : null, note: form.note.value})});
  form.querySelector('.saved').textContent = r.ok ? 'saved' : 'not saved: ' + r.status;
  if (r.ok && d) form.closest('.card').classList.add('done');
}
async function bulk() {
  const open = [...document.querySelectorAll('form[data-video]')].filter(f => !f.querySelector('input:checked'));
  if (!open.length || !confirm(`Take the suggestion for ${open.length} undecided videos?`)) return;
  for (const f of open) { f.querySelector(`input[value=${f.dataset.suggestion}]`).checked = true; await send(f); }
}
</script></body></html>"""


def t(sec):
    sec = int(sec or 0)
    return f'{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}' if sec >= 3600 else f'{sec // 60}:{sec % 60:02d}'


if __name__ == '__main__':
    _check()
    _check_carried()
    if '--apply' in sys.argv:
        print(apply())
    else:
        cases = build()
        print(len(cases), 'cases;', sum(c['suggestion'] == 'redownload' for c in cases), 'to download again')
