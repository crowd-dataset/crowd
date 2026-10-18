"""Pipeline status for the TU/e network: what each worker is doing (channel, progress, videos per hour, problems), how
many batches and sheet updates wait, when the workers were last synced, the refresh queue and the channels by state.

    REVIEW_HOST=0.0.0.0 python dashboards/pipeline_status.py   # port 8771, from the repo's top folder (README)

Every REFRESH_S a background thread reads the end of each worker's runner.log over ssh (sync_workers.WORKERS) and this
machine's state; the page shows that snapshot, so opening it costs the workers nothing.
"""
import collections
import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run_channels and sync_workers: the repo's top folder
import run_channels as rc  # noqa: E402
import sync_workers as sw  # noqa: E402

import short_videos  # noqa: E402
from globe import globe_data  # noqa: E402

PORT, REFRESH_S, TAIL = 8771, 120, 600
# the video downloader (tools/download_videos.py): its machine and the status file it writes there
DOWNLOADER = ('lin4080', '~/Documents/GitHub/pedestrians-in-youtube/_output/download_status.json')
# the nightly inventory of the video files (tools/video_inventory.py): its machine and summary (the rows: .csv)
INVENTORY = ('lin4080', '~/crowd-counts/_output/video_inventory.json')
PROBLEMS = {'duplicate': 'also in an earlier folder', 'unreadable': 'no video stream: broken or partial',
            'short': "over 5 s shorter than its last segment's end", 'unused': 'in no mapping'}
HOST = os.environ.get('REVIEW_HOST', '127.0.0.1')
STUCK_MIN = sw.STUCK_MIN
# each machine's YouTube cookies, {host: {email, file, dir}} (file: cookies-NAME.txt in dir, cookies.txt a link to it):
# on this machine only, never in the repo
ACCOUNTS = os.path.join(rc.ROOT, 'cookie_accounts.json')
SIGN_IN = re.compile(r'sign in|not a bot', re.I)
HISTORY = os.path.join(rc.ROOT, 'status_history.jsonl')  # hourly: refresh channels not started, batches to review
app = Flask(__name__)
snapshot = {}

T = r'^(\d\d):(\d\d) '
NEXT = re.compile(T + r'(?:row (\d+) )?(?:videos of )?(https?://\S+) \(([^)]*)\): next')
ANALYSING = re.compile(T + r'\s*\[(\d+)/(\d+)\] analysing (\S+) ?(.*)')
BATCH = re.compile(T + r'\s*(\d+) videos added or re-analysed: (\d+) proposed, (\d+) excluded')


def parse_log(lines, mtime, now):
    """What a worker's runner.log says it is doing. Its lines carry only HH:MM: they are dated back from the file's
    modification time (the last line), counting a day back whenever the clock jumps forward going up the log."""
    stamps, day, later = [], 0, None
    for line in reversed(lines):
        m = re.match(T, line)
        if m:
            minute = int(m[1]) * 60 + int(m[2])
            if later is not None and minute > later:
                day += 1
            later = minute
            stamps.append(minute - day * 1440)
        else:
            stamps.append(None)
    stamps.reverse()
    last = next((s for s in reversed(stamps) if s is not None), None)

    def when(s):
        return None if s is None or last is None else mtime - (last - s) * 60
    out = dict(state='working', channel=None, row=None, country=None, i=None, n=None, video=None, title=None,
               since=None, batch=None, rate=None, problems=[], log_age=now - mtime)
    progress = []
    for line, s in zip(lines, stamps):
        if m := NEXT.match(line):
            out.update(channel=m[4], row=m[3], country=m[5], batch=None)
        elif m := ANALYSING.match(line):
            out.update(i=int(m[3]), n=int(m[4]), video=m[5], title=m[6].strip(), since=when(s))
            progress.append(when(s))
        elif m := BATCH.match(line):
            out['batch'] = dict(added=int(m[3]), proposed=int(m[4]), excluded=int(m[5]), at=when(s))
        elif sw.PROBLEM.search(line):
            out['problems'].append(dict(text=line.strip()[:220], at=when(s)))
    recent = [t for t in progress if t and t > now - 3 * 3600]
    if len(recent) >= 3 and recent[-1] > recent[0]:
        out['rate'] = (len(recent) - 1) / ((recent[-1] - recent[0]) / 3600)
    last_progress = progress[-1] if progress else None
    out['problems'] = [p for p in out['problems'] if (p['at'] or now) > now - 6 * 3600][-3:]  # the last 6 h only
    tail = next((ln for ln in reversed(lines) if ln.strip()), '')
    if 'no channel left' in tail:
        out['state'] = 'idle'
    elif out['log_age'] > STUCK_MIN * 60:
        out['state'] = 'stalled'
    elif out['problems'] and (last_progress is None or (out['problems'][-1]['at'] or 0) > last_progress):
        out['state'] = 'blocked'
    for p in out['problems']:
        p['current'] = last_progress is None or (p['at'] or 0) >= last_progress
    return out


def worker(host):
    log = f'{sw.place(host)[1]}/_output/proposals/runner.log'
    try:
        raw = sw.ssh(host, f'tail -n {TAIL} {log}; echo @@; '
                           f'stat -c %Y {log} 2>/dev/null || stat -f %m {log}; date +%s; '
                           'echo @@Y; cat ~/.wsl_yield.json 2>/dev/null || true')  # jobs paused for Windows
    except Exception as e:
        return dict(host=host, state='offline', error=str(e)[-200:], role=sw.WORKERS[host])
    body, _, tail = raw.rpartition('@@\n')
    tail, _, paused = tail.partition('@@Y\n')
    try:
        mtime, now = map(float, tail.split()[:2])
    except ValueError:  # no runner.log there (yet): a new worker before its first line
        return dict(host=host, state='offline', error='no runner.log yet', role=sw.WORKERS[host])
    w = dict(parse_log(body.splitlines(), mtime, now), host=host, role=sw.WORKERS[host], clock=now)
    try:
        y = json.loads(paused)
    except ValueError:
        y = {}
    if y.get('paused'):  # wsl_yield.py made way for someone using the Windows PC: not stalled (and no alert)
        w.update(state='paused', paused=dict(why=y.get('why', ''), since=y.get('since')))
    return w


def downloads():
    """The downloader's status, with videos per hour over the last 3 h and the time left at that pace."""
    host, path = DOWNLOADER
    try:
        d = json.loads(sw.ssh(host, f'cat {path}'))
    except Exception as e:
        return dict(host=host, error=str(e)[-200:])
    now = time.time()
    hours = min(3.0, (now - d['started']) / 3600)
    done = sum(r['t'] > now - hours * 3600 for r in d.get('recent', []))
    d['rate'] = done / hours if hours > 0.25 else None
    d['eta'] = d['queue'] / d['rate'] * 3600 if d['rate'] else None
    # room on the download drives above their reserve, and how many videos of the recent average size fit there
    sizes = [r['mb'] / 1000 for r in d.get('recent', []) if r.get('mb')]
    d['room_gb'] = sum(max(0, r['free_gb'] - d['reserve_gb']) for r in d.get('drives', []) if r.get('mounted'))
    d['video_gb'] = sum(sizes) / len(sizes) if sizes else None
    if d['video_gb'] and d['rate']:
        d['full_in'] = d['room_gb'] / (d['rate'] * d['video_gb']) * 3600
    return dict(d, host=host, failed_24h=sum(f['t'] > now - 86400 for f in d.get('failures', [])))


def inventory():
    host, path = INVENTORY
    try:
        return dict(json.loads(sw.ssh(host, f'cat {path}')), host=host)
    except Exception as e:
        return dict(host=host, error=str(e)[-200:])


def masked(email):
    """olena.bazilinska@gmail.com -> olena.b…@gmail.com: enough to tell the accounts apart on an open page."""
    name, _, domain = (email or '').partition('@')
    return f'{name[:7]}…@{domain}' if domain else (email or 'its account')


def cookies_needed(workers, downloads):
    """[{host, account, how}] of the machines YouTube asks to sign in now: a worker blocked by the sign-in check
    (it waits for a fresh cookies.txt), the downloader paused and nothing downloaded for 2 h (its Chrome login)."""
    try:
        with open(ACCOUNTS) as f:
            accounts = json.load(f)
    except (OSError, ValueError):
        accounts = {}

    def need(host, alias):
        a = accounts.get(alias) or {}
        a = a if isinstance(a, dict) else dict(email=a)
        f, d = a.get('file', 'cookies.txt'), a.get('dir', sw.REMOTE)
        link = f' && ln -sf {f} {d}/cookies.txt' if f != 'cookies.txt' else ''
        return dict(host=host, account=masked(a.get('email')), file=f,
                    how=f"scp ~/Downloads/{f} {alias}:{d}/ && ssh {alias} 'chmod 600 {d}/{f}{link}'")
    out = []
    for w in workers:
        if w.get('state') == 'blocked' and w['problems'] and SIGN_IN.search(w['problems'][-1]['text']):
            out.append(need(w['host'], sw.place(w['host'])[0]))  # a second worker on a machine shares its cookies
    last = max((r['t'] for r in downloads.get('recent', [])), default=downloads.get('started', time.time()))
    if str(downloads.get('state', '')).startswith('paused') and time.time() - last > 2 * 3600:
        out.append(need(downloads['host'], downloads['host']))  # its cookies file, then its Chrome login
    return out


def history(entry):
    """Adds entry to HISTORY when the last one is an hour old; returns the last 8 days."""
    rows = []
    if os.path.exists(HISTORY):
        with open(HISTORY) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    if not rows or entry['t'] - rows[-1]['t'] >= 3600:
        rows = [r for r in rows if r['t'] > entry['t'] - 8 * 86400] + [entry]
        with open(HISTORY + '.tmp', 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)
        os.replace(HISTORY + '.tmp', HISTORY)
    return rows


def paces(rows, now):
    """Per day over the last 7 days: refresh channels started, batches the workers finished, batches reviewed (a
    channel entering or leaving 'to review'); None with less than 12 h of history."""
    rows = [r for r in rows if r['t'] > now - 7 * 86400]
    if len(rows) < 2 or rows[-1]['t'] - rows[0]['t'] < 12 * 3600:
        return None
    days = (rows[-1]['t'] - rows[0]['t']) / 86400
    added = sum(len(set(b['to_review']) - set(a['to_review'])) for a, b in zip(rows, rows[1:]))
    reviewed = sum(len(set(a['to_review']) - set(b['to_review'])) for a, b in zip(rows, rows[1:]))
    return dict(started=(rows[0]['left'] - rows[-1]['left']) / days, added=added / days, reviewed=reviewed / days)


def collect():
    state = rc.load_state()
    by_url = {(c.get('url') or k.split('refresh:', 1)[-1]): c for k, c in state.items()}
    with ThreadPoolExecutor(len(sw.WORKERS) + 2) as ex:
        dl, inv = ex.submit(downloads), ex.submit(inventory)
        workers = list(ex.map(worker, sw.WORKERS))
        dl, inv = dl.result(), inv.result()
    for w in workers:
        url = w.get('channel') or ''
        c = by_url.get(url)  # a channel not in the state yet: the last part of its URL (@name or channel id)
        w['name'] = c['name'] if c else next((p for p in reversed(url.split('/')) if p and p != 'videos'), url)
    batches = {}  # one per batch folder: a channel both on the sheet and in the refresh queue has two state entries
    for k, c in state.items():
        b = batches.get(c['name'])
        if b is None:
            waiting, proposed = rc.review_counts(c['name'])
            if not waiting:
                continue
            b = batches[c['name']] = dict(name=c['name'], waiting=waiting, proposed=proposed, host='')
        b['host'] = ', '.join(sorted(set(filter(None, b['host'].split(', '))) | {c.get('host', 'mini')}))
    batches = list(batches.values())
    todo = json.load(open(rc.SHEET_TODO)) if os.path.exists(rc.SHEET_TODO) else []
    queue = json.load(open(rc.REFRESH_QUEUE)) if os.path.exists(rc.REFRESH_QUEUE) else []
    left = sum(1 for q in queue if q['url'] not in by_url)  # refresh channels not started
    now = time.time()
    pace = paces(history(dict(t=now, left=left, to_review=sorted(
        k for k, c in state.items() if c.get('status') == 'to review'))), now)
    statuses = collections.Counter((c.get('host', 'mini'), c.get('status', '?')) for c in state.values())
    hosts = sorted({h for h, _ in statuses})
    order = ['processing', 'to review', 'reviewed', 'marked', 'rejected', 'duplicate', 'error']
    names = sorted({s for _, s in statuses}, key=lambda s: order.index(s) if s in order else 99)
    return dict(
        at=time.time(), workers=workers, downloads=dl, inventory=inv, cookies=cookies_needed(workers, dl),
        short_open=sum(not c.get('decision') for c in short_videos.load()['cases']),
        batches=sorted(batches, key=lambda b: -b['waiting']),
        sheet_todo=len(todo), queue=len(queue), left=left, pace=pace,
        queue_phases=sorted(collections.Counter(q.get('phase') for q in queue).items(), key=lambda x: str(x[0])),
        last_sync=os.path.getmtime(sw.HEALTH) if os.path.exists(sw.HEALTH) else None,
        hosts=hosts, statuses=[(s, [statuses.get((h, s), 0) for h in hosts]) for s in names])


def refresher():
    while True:
        try:
            snapshot.update(collect())
        except Exception as e:  # the page says so and keeps the last snapshot
            snapshot['error'] = f'{type(e).__name__}: {e}'
        time.sleep(REFRESH_S)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pipeline status</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
<text y=%22.9em%22 font-size=%2290%22>🚚</text></svg>">
<meta http-equiv="refresh" content="120">
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#667085;--line:#e4e7ec;--track:#eef0f3;--link:#2563eb;
 --ok:#16a34a;--warn:#d97706;--bad:#dc2626;--bar:#2563eb}
:root[data-theme=dark]{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--muted:#8b93a1;--line:#272c36;
 --track:#232833;--link:#60a5fa;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--bar:#60a5fa}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
 font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:20px 0 8px}
.muted{color:var(--muted)}.small{font-size:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,200px),1fr));gap:12px;margin-top:12px}
.tile,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.tile b{display:block;font-size:24px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:12px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.head h3{margin:0;font-size:16px}
.st{font-size:12px;font-weight:600;border-radius:6px;padding:1px 8px;background:var(--track)}
.st.working{color:var(--ok)}.st.idle{color:var(--muted)}
.st.blocked,.st.stalled{color:var(--warn)}.st.paused{color:var(--link)}.st.offline{color:var(--bad)}
.bar{height:6px;background:var(--track);border-radius:3px;overflow:hidden;margin:6px 0 4px}
.bar i{display:block;height:100%;background:var(--bar)}
.row{margin:3px 0}.ell{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.prob{font:12px ui-monospace,Menlo,monospace;color:var(--warn);margin:3px 0;overflow-wrap:anywhere}
.prob.old{color:var(--muted)}
.card.cookie{border-color:var(--warn);margin-top:12px}.cookie code{font-size:12px;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:4px 8px;border-top:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:500}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.dl{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.6fr);gap:12px}
.dl .scroll{position:relative;padding:0;min-height:200px}.dl .scroll>div{position:absolute;inset:0;overflow:auto}
.dl .scroll th{position:sticky;top:0;background:var(--card)}
@media (max-width:900px){.dl{grid-template-columns:1fr}.dl .scroll{height:400px}}
#theme{position:fixed;right:12px;bottom:12px;z-index:9999;padding:6px 10px;border:1px solid var(--line);
border-radius:16px;background:var(--bg);color:var(--fg);font-size:13px;cursor:pointer}
</style>
<script>
// dark unless switched to light, as on the review hub: a cookie, shared by all the pages of this host (any port)
(function () {
  const m = document.cookie.match(/(?:^|; )theme=(dark|light)/);
  document.documentElement.dataset.theme = m ? m[1] : 'dark';
  addEventListener('DOMContentLoaded', () => {
    const b = document.createElement('button');
    b.id = 'theme';
    const label = () => { b.textContent = document.documentElement.dataset.theme === 'dark' ? '☀️ Light' : '🌙 Dark'; };
    b.onclick = () => {
      const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = t;
      document.cookie = `theme=${t}; path=/; max-age=31536000; SameSite=Lax`;
      label();
    };
    label();
    document.body.appendChild(b);
  });
})();
</script>
</head><body>
<h1>🚚 Pipeline status</h1>
<div class="muted small">snapshot {{ ago(s.at) }} ago (every 2 min) ·
 <a href="#" onclick="location.port=8770;return false">review hub</a>
 · <a href="#" onclick="location.port=8090;return false">machines</a>
 {% if s.error %} · <span style="color:var(--bad)">{{ s.error }}</span>{% endif %}</div>
{% for c in s.cookies %}<div class="card cookie">🍪 <b>{{ c.host }} needs new YouTube cookies</b>: in a new
 Incognito window sign in to YouTube as <b>{{ c.account }}</b>, export the cookies as {{ c.file }}, close the window,
 then
 <code>{{ c.how }}</code></div>{% endfor %}
<div class="tiles">
 <a class="tile" href="#" onclick="location.port=8770;return false"><span class="muted small">Batches to review</span>
  <b>{{ s.batches|length }}</b>
  <span class="small muted">{{ s.batches|sum(attribute='waiting') }} videos waiting
  {%- if s.pace %} · workers add {{ '%.1f'|format(s.pace.added) }}/day, you review
  {{ '%.1f'|format(s.pace.reviewed) }}/day (7 days){% if s.pace.reviewed > s.pace.added %}: clear in about
  {{ dur(s.batches|length / (s.pace.reviewed - s.pace.added) * 86400) }}{% endif %}{% endif %}</span></a>
 <div class="tile"><span class="muted small">Sheet updates waiting</span><b>{{ s.sheet_todo }}</b>
  <span class="small muted">sheet_todo.json, entered by hand</span></div>
 <div class="tile"><span class="muted small">Last worker sync</span>
  <b>{{ ago(s.last_sync) if s.last_sync else '–' }}</b>
  <span class="small muted">ago (hourly)</span></div>
 <div class="tile"><span class="muted small">Refresh queue</span><b>{{ s.queue }}</b>
  <span class="small muted">
  {% for p, n in s.queue_phases %}phase {{ p }}: {{ n }}{{ ' · ' if not loop.last }}{% endfor %}<br>
  {{ s.left }} not started{% if s.pace %} · {{ '%.1f'|format(s.pace.started) }}/day (7 days){% if s.pace.started > 0
  %} · about {{ dur(s.left / s.pace.started * 86400) }} left{% endif %}{% endif %}</span></div>
</div>
<h2>Workers</h2><div class="grid">
{% for w in s.workers %}<div class="card">
 <div class="head"><h3>{{ w.host }}</h3><span class="st {{ w.state }}">{{ w.state }}</span></div>
 <div class="muted small">{{ w.role }}</div>
 {% if w.state == 'offline' %}<div class="prob">{{ w.error }}</div>{% else %}
 {% if w.channel %}<div class="row ell" style="margin-top:8px"><b>{{ w.name }}</b>
  <span class="muted">{% if w.row %}· row {{ w.row }} {% endif %}· {{ w.country }}</span>
  <a class="small" href="{{ w.channel }}" target="_blank">↗</a></div>{% endif %}
 {% if w.n %}<div class="bar"><i style="width:{{ (100 * w.i / w.n)|round(1) }}%"></i></div>
 <div class="small muted">video {{ w.i }} of {{ w.n }}{% if w.rate %} · {{ '%.1f'|format(w.rate) }} videos/h · about
  {{ dur((w.n - w.i) / w.rate * 3600) }} for the rest of the channel{% endif %}</div>
 <div class="row ell small" title="{{ w.title }}">now: {{ w.title or w.video }}
  <span class="muted">(since {{ ago(w.since) }})</span></div>{% endif %}
 {% if w.batch %}<div class="small muted">last batch {{ ago(w.batch.at) }} ago:
  {{ w.batch.proposed }} proposed, {{ w.batch.excluded }} excluded</div>{% endif %}
 {% if w.paused %}<div class="small" style="color:var(--link)">paused{% if w.paused.since %} since
  {{ hm(w.paused.since) }}{% endif %} to make way: {{ w.paused.why }}
  (it resumes 10 min after that stops)</div>{% endif %}
 <div class="small muted">log written {{ ago(w.clock - w.log_age) }} ago</div>
 {% for p in w.problems %}<div class="prob {{ '' if p.current else 'old' }}">{{ p.text }}</div>{% endfor %}
 {% endif %}</div>{% endfor %}
</div>
{% set d = s.downloads %}<h2>Video downloads ({{ d.host }})</h2>
{% if d.error %}<div class="card"><div class="prob">status not readable: {{ d.error }}</div></div>{% else %}
<div class="dl"><div class="card">
 <div class="head"><h3>tools/download_videos.py</h3><span class="st {{ 'working' if d.state == 'downloading' else
  'idle' if d.state == 'idle' else 'blocked' }}">{{ d.state }}</span></div>
 {% if d.until %}<div class="small muted">until {{ hm(d.until) }}</div>{% endif %}
 <div class="row" style="margin-top:8px"><a href="downloads/todo"><b>{{ d.queue }}</b> videos to download</a>
  · <a href="downloads/done">downloaded</a>{% if d.rate %} ·
  {{ '%.0f'|format(d.rate) }} videos/h (last 3 h) · about {{ dur(d.eta) }} left{% endif %}</div>
 {% if d.video_gb %}<div class="small muted">room for about {{ (d.room_gb / d.video_gb)|round|int }} more videos
  ({{ '%.1f'|format(d.room_gb / 1000) }} TB at {{ '%.1f'|format(d.video_gb) }} GB each)
  {%- if d.room_gb >= d.queue * d.video_gb %}: enough for the queue{% else %}
  <span style="color:var(--warn)">· not enough for the queue{% if d.full_in %}: drives full in
  about {{ dur(d.full_in) }}{% endif %}</span>{% endif %}</div>{% endif %}
 {% if d.current %}<div class="row ell small">now:
  <a href="https://youtu.be/{{ d.current }}" target="_blank">{{ d.current }}</a></div>{% endif %}
 <div class="small muted">{{ d.downloaded }} downloaded and {{ d.failed }} failed since it started
  {{ ago(d.started) }} ago
  ({{ d.failed_24h }} failed in 24 h) · status written {{ ago(d.updated) }} ago</div>
 {% for r in d.drives %}<div class="small" style="margin-top:6px">
  {{ r.path.split('/')[3] if r.path.startswith('/media/')
  else r.path }}{% if r.path == d.drive %} <b>· filling</b>{% endif %}{% if not r.mounted %}
  <span class="prob">not mounted</span>{% endif %} <span class="muted">{{ r.free_gb }} GB free of {{ r.total_gb }}
  (down to {{ d.reserve_gb }})</span></div>
 <div class="bar"><i style="width:{{ (100 - 100 * r.free_gb / r.total_gb)|round(1) if r.total_gb else 0
  }}%"></i></div>
 {% endfor %}
</div>
{% if d.recent %}<div class="card scroll"><div><table>
<tr><th>downloaded</th><th>video</th><th>title</th><th class="n">size</th><th>drive</th></tr>
{% for r in d.recent %}<tr><td class="muted">{{ hm(r.t) }}</td>
 <td><a href="https://youtu.be/{{ r.vid }}" target="_blank">{{ r.vid }}</a></td>
 <td class="ell" style="max-width:420px">{{ r.title or '' }}</td><td class="n">{{ '%.1f'|format(r.mb / 1000) }} GB</td>
 <td class="muted">{{ r.drive.split('/')[3] if r.drive.startswith('/media/') else r.drive }}</td></tr>{% endfor %}
</table></div></div>{% endif %}</div>{% endif %}
{% set v = s.inventory %}{% if v %}<h2>Video files ({{ v.host }})</h2><div class="card">
{% if v.error %}<div class="prob">inventory not readable: {{ v.error }}</div>{% else %}
 <div class="row"><b>{{ v.videos }}</b> videos, {{ v.tb }} TB · {{ v.missing }} in the mapping on no drive (the
  download queue) <span class="small muted">· inventory of {{ hm(v.t) }}, nightly</span></div>
 <div class="small">{% for f in v.folders %}{{ f.folder.split('/')[3] if f.folder.startswith('/media/')
  else f.folder }}:
  {{ f.files }} ({{ f.tb }} TB){{ ' · ' if not loop.last }}{% endfor %}</div>
 <table style="margin-top:8px"><tr><th>problem</th><th class="n">files</th><th class="n">GB</th><th></th></tr>
 {% for p, what in problems.items() %}{% set n = v.problems.get(p, {}) %}<tr>
  <td>{% if n.files %}<a href="inventory/{{ p }}">{{ p }}</a>{% else %}{{ p }}{% endif %}</td>
  <td class="n">{{ n.files or 0 }}</td><td class="n">{{ n.gb or 0 }}</td>
  <td class="muted">{{ what }}{% if p == 'short' and n.files %} ·
  <a href="short">check them one by one</a>{% if s.short_open %} <b>({{ s.short_open }} to decide)</b>{% endif %}
  {% endif %}</td></tr>{% endfor %}
 </table>{% endif %}</div>{% endif %}
{% if s.batches %}<h2>Batches waiting for review</h2><div class="card"><table>
<tr><th>channel</th><th>fetched on</th><th class="n">to review</th><th class="n">proposed</th></tr>
{% for b in s.batches %}<tr><td>{{ b.name }}</td><td class="muted">{{ b.host }}</td>
 <td class="n">{{ b.waiting }}</td><td class="n">{{ b.proposed }}</td></tr>{% endfor %}
</table></div>{% endif %}
<h2>Channels by state</h2><div class="card"><table>
<tr><th>state</th>{% for h in s.hosts %}<th class="n">{{ h }}</th>{% endfor %}<th class="n">all</th></tr>
{% for name, counts in s.statuses %}<tr><td>{{ name }}</td>
 {% for n in counts %}<td class="n">{{ n or '' }}</td>{% endfor %}
 <td class="n">{{ counts|sum }}</td></tr>{% endfor %}
</table></div>
</body></html>"""


def dur(sec):
    if sec < 5400:
        return f'{sec / 60:.0f} min'
    return f'{sec / 3600:.0f} h' if sec < 172800 else f'{sec / 86400:.1f} days'


@app.route('/')
def index():
    if not snapshot.get('at'):
        return '<meta http-equiv="refresh" content="5">Collecting the first snapshot…'
    return render_template_string(PAGE, s=snapshot, dur=dur, problems=PROBLEMS,
                                  hm=lambda t: time.strftime('%d %b %H:%M', time.localtime(t)),
                                  ago=lambda t: dur(time.time() - t) if t else '?')


LIST_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ title }}</title>
<style>
:root{--bg:#f6f7f9;--fg:#1b1f24;--muted:#667085;--line:#e4e7ec;--link:#2563eb}
:root[data-theme=dark]{--bg:#0f1115;--fg:#e6e8ec;--muted:#8b93a1;--line:#272c36;--link:#60a5fa}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,sans-serif}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}h1{font-size:20px;margin:0 0 4px}
.muted{color:var(--muted)}table{border-collapse:collapse;margin-top:12px;font-size:13px}
td,th{padding:3px 10px 3px 0;border-top:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:500}td.t{white-space:normal;max-width:520px}td.n{text-align:right}
#theme{position:fixed;right:12px;bottom:12px;z-index:9999;padding:6px 10px;border:1px solid var(--line);
border-radius:16px;background:var(--bg);color:var(--fg);font-size:13px;cursor:pointer}
</style>
<script>
// dark unless switched to light, as on the review hub: a cookie, shared by all the pages of this host (any port)
(function () {
  const m = document.cookie.match(/(?:^|; )theme=(dark|light)/);
  document.documentElement.dataset.theme = m ? m[1] : 'dark';
  addEventListener('DOMContentLoaded', () => {
    const b = document.createElement('button');
    b.id = 'theme';
    const label = () => { b.textContent = document.documentElement.dataset.theme === 'dark' ? '☀️ Light' : '🌙 Dark'; };
    b.onclick = () => {
      const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = t;
      document.cookie = `theme=${t}; path=/; max-age=31536000; SameSite=Lax`;
      label();
    };
    label();
    document.body.appendChild(b);
  });
})();
</script>
</head><body><h1>🚚 {{ title }}</h1>
<div class="muted"><a href="..">pipeline status</a> · {{ rows|length }} videos
{% if note %} · {{ note }}{% endif %}</div>
<table><tr>{% for c in cols %}<th>{{ c }}</th>{% endfor %}</tr>
{% for r in rows %}<tr>{% for c in cols %}{% if c == 'video' %}<td><a href="https://youtu.be/{{ r[c] }}"
 target="_blank">{{ r[c] }}</a></td>{% else %}<td class="{{ 't' if c == 'title' else 'n' if c == 'GB' else ''
 }}">{{ r[c] }}</td>{% endif %}{% endfor %}</tr>{% endfor %}</table></body></html>"""


@app.route('/downloads/todo')
def downloads_todo():
    """The videos still to download, in the order the downloader takes them (as of the last snapshot)."""
    d = snapshot.get('downloads') or {}
    rows = [{'#': i, 'video': v} for i, v in enumerate(d.get('todo', []), 1)]
    return render_template_string(LIST_PAGE, title='Videos to download', cols=['#', 'video'], rows=rows,
                                  note=d.get('error') or 'in the order they are taken; new videos join every hour')


@app.route('/downloads/done')
def downloads_done():
    """Every video the downloader has fetched, newest first (read from its machine when opened)."""
    host, path = DOWNLOADER
    try:
        text = sw.ssh(host, 'cat ' + path.replace('download_status.json', 'downloaded_videos.csv'))
        rows = list(csv.DictReader(io.StringIO(text)))[::-1]
        note = None
    except Exception as e:
        rows, note = [], str(e)[-200:]
    for r in rows:
        r['GB'] = f"{int(r['mb'] or 0) / 1000:.1f}"
        r['drive'] = r['drive'].split('/')[3] if r['drive'].startswith('/media/') else r['drive']
    return render_template_string(LIST_PAGE, title='Videos downloaded', rows=rows, note=note,
                                  cols=['time', 'video', 'title', 'GB', 'source', 'drive'])


@app.route('/inventory/<problem>')
def inventory_list(problem):
    """The video files with this problem in the last inventory (read from its machine when opened)."""
    if problem not in PROBLEMS:  # it goes into a command line
        return 'unknown problem', 404
    host, path = INVENTORY
    try:
        text = sw.ssh(host, f"awk -F, '{{sub(/\\r$/, \"\")}} NR == 1 || index(\" \" $8 \" \", \" {problem} \")' "
                            + path.replace('.json', '.csv'))
        rows, note = list(csv.DictReader(io.StringIO(text))), PROBLEMS[problem]
    except Exception as e:
        rows, note = [], str(e)[-200:]
    for r in rows:
        r['GB'] = f"{int(r['mb'] or 0) / 1000:.1f}"
        r['drive'] = r['folder'].split('/')[3] if r['folder'].startswith('/media/') else r['folder']
        r['length'] = dur(float(r['duration'])) if r['duration'] else ''
    return render_template_string(LIST_PAGE, title=f'Video files: {problem}', rows=rows, note=note,
                                  cols=['video', 'drive', 'GB', 'length', 'height', 'problem'])


@app.route('/short')
def short_page():
    """A form to check the videos shorter than their last segment (dashboards/short_videos.py)."""
    data = short_videos.load()
    return render_template_string(short_videos.PAGE, cases=data['cases'], built=data['built'], t=short_videos.t,
                                  decisions=short_videos.DECISIONS, theme=PAGE[PAGE.index('<script>'):
                                                                               PAGE.index('</script>') + 9],
                                  hm=lambda t: time.strftime('%d %b %H:%M', time.localtime(t)))


@app.route('/short/decide', methods=['POST'])
def short_decide():
    d = request.get_json(force=True)
    try:
        ok = short_videos.decide(str(d.get('video')), d.get('decision'), d.get('note'))
    except ValueError:
        return 'unknown decision', 400
    return (jsonify(ok=True), 200) if ok else ('unknown video', 404)


@app.route('/status.json')
def status_json():
    return jsonify(snapshot)


@app.route('/globe')
def globe_page():
    """The globe widget (shown by the workstations widgets' server, which passes /globe.json through too)."""
    return (Path(__file__).resolve().parent / 'globe' / 'widget_globe.html').read_text(encoding='utf-8')


@app.route('/globe.json')
def globe_json():
    return jsonify(globe_data.globe())


def _check_parse_log():
    """parse_log on a made-up log: day wrap, progress, rate, a problem after the last progress = blocked."""
    lines = ['23:50 row 7 https://www.youtube.com/@x/videos (Chile): next 15 proposals',
             '23:55   [1/10] analysing aaa First', '00:05   [2/10] analysing bbb Second',
             '00:15   [3/10] analysing ccc Third', '00:20   YouTube asks to sign in: waiting 15 min']
    now = 1_000_000.0
    w = parse_log(lines, mtime=now, now=now)
    assert (w['channel'], w['row'], w['country'], w['i'], w['n']) == (
        'https://www.youtube.com/@x/videos', '7', 'Chile', 3, 10)
    assert w['since'] == now - 5 * 60 and abs(w['rate'] - 6.0) < 1e-9, (w['since'], w['rate'])
    assert w['state'] == 'blocked' and w['problems'][0]['current']
    assert parse_log(lines[:-1] + ['00:30 no channel left to process'], now, now)['state'] == 'idle'
    assert parse_log(lines[:-1], now - 2 * 3600, now)['state'] == 'stalled'


def _check_cookies():
    w = parse_log(['08:00   Stopped early: YouTube asks to sign in (ERROR: Sign in to confirm you’re not a bot)'],
                  1_000_000.0, 1_000_000.0)
    assert w['state'] == 'blocked'
    c = cookies_needed([dict(w, host='lin5080')], {})
    assert [x['host'] for x in c] == ['lin5080'] and 'lin5080:crowd-routine/' in c[0]['how'], c
    assert masked('olena.bazilinska@gmail.com') == 'olena.b…@gmail.com'
    assert cookies_needed([dict(w, problems=[dict(text='quota exceeded')], host='x')], {}) == []


def _check_paces():
    h = [dict(t=0, left=100, to_review=['a', 'b']), dict(t=43200, left=99, to_review=['b', 'c']),
         dict(t=86400, left=96, to_review=['c', 'd', 'e'])]
    assert paces(h, 86400) == dict(started=4.0, added=3.0, reviewed=2.0), paces(h, 86400)
    assert paces(h[:2], 43200)['started'] == 2.0  # 12 h: just enough
    assert paces(h[:1], 0) is None


if __name__ == '__main__':
    _check_parse_log()
    _check_paces()
    _check_cookies()
    threading.Thread(target=refresher, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False)
