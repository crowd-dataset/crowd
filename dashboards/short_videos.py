"""Videos shorter than their last segment (the inventory's "short"), to check one by one on the status page (/short).

    python dashboards/short_videos.py   # from the repo's top folder: (re)build _output/proposals/short_videos.json

The build takes the short files from the last inventory (tools/video_inventory.py, read over ssh), their segments
from mapping.csv here and their length on YouTube (yt-dlp, metadata only; kept from the last build), and suggests a
fix: the download was cut off (YouTube is longer than the file): download it again; else the mapping's end times run
past the end of the video: trim the segments to it. The page saves a decision per video in the same file; applying
them (moving a file away to have it downloaded again, editing mapping.csv) is a separate step.
"""
import csv
import fcntl
import io
import json
import os
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
        o = old.get(v, {})
        c = dict(video=v, drive=r['folder'], file_s=length[v], height=r['height'], title=titles.get(v, ''),
                 rows=rows[v], youtube_s=yt[v] if v in yt else o.get('youtube_s'),
                 decision=o.get('decision'), note=o.get('note', ''))
        c['suggestion'], c['why'] = suggest(c)
        cases.append(c)
    cases.sort(key=lambda c: (c['suggestion'], c['video']))
    with _lock:
        save(dict(built=time.time(), cases=cases))
    return cases


def decide(video, decision, note):
    """Saves the decision (or None to undo it) and note for one video."""
    if decision not in DECISIONS and decision is not None:
        raise ValueError(decision)
    with _lock:
        data = load()
        for c in data['cases']:
            if c['video'] == video:
                c['decision'], c['note'] = decision, str(note or '')[:500]
                save(data)
                return True
    return False


def _check():
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
  {{ c.title }}</div>
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
    cases = build()
    print(len(cases), 'cases;', sum(c['suggestion'] == 'redownload' for c in cases), 'to download again')
