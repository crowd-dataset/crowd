"""Pipeline status for the TU/e network: what each worker is doing (channel, progress, videos per hour, problems), how
many batches and sheet updates wait, when the workers were last synced, the refresh queue and the channels by state.

    REVIEW_HOST=0.0.0.0 python pipeline_status.py   # http://131.155.233.55:8771 (started by ~/review_host.sh)

Every REFRESH_S a background thread reads the end of each worker's runner.log over ssh (sync_workers.WORKERS) and this
machine's state; the page shows that snapshot, so opening it costs the workers nothing.
"""
import collections
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, render_template_string

import run_channels as rc
import sync_workers as sw

PORT, REFRESH_S, TAIL = 8771, 120, 600
HOST = os.environ.get('REVIEW_HOST', '127.0.0.1')
STUCK_MIN = sw.STUCK_MIN
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
    when = lambda s: None if s is None or last is None else mtime - (last - s) * 60
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
    tail = next((l for l in reversed(lines) if l.strip()), '')
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
    log = f'{sw.REMOTE}/_output/proposals/runner.log'
    try:
        raw = sw.ssh(host, f'tail -n {TAIL} {log}; echo @@; stat -c %Y {log} 2>/dev/null || stat -f %m {log}; date +%s')
    except Exception as e:
        return dict(host=host, state='offline', error=str(e)[-200:], role=sw.WORKERS[host])
    body, _, tail = raw.rpartition('@@\n')
    mtime, now = map(float, tail.split()[:2])
    return dict(parse_log(body.splitlines(), mtime, now), host=host, role=sw.WORKERS[host], clock=now)


def collect():
    state = rc.load_state()
    by_url = {(c.get('url') or k.split('refresh:', 1)[-1]): c for k, c in state.items()}
    with ThreadPoolExecutor(len(sw.WORKERS)) as ex:
        workers = list(ex.map(worker, sw.WORKERS))
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
    statuses = collections.Counter((c.get('host', 'mini'), c.get('status', '?')) for c in state.values())
    hosts = sorted({h for h, _ in statuses})
    order = ['processing', 'to review', 'reviewed', 'marked', 'rejected', 'duplicate', 'error']
    names = sorted({s for _, s in statuses}, key=lambda s: order.index(s) if s in order else 99)
    return dict(
        at=time.time(), workers=workers, batches=sorted(batches, key=lambda b: -b['waiting']),
        sheet_todo=len(todo), queue=len(queue),
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
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🚚</text></svg>">
<meta http-equiv="refresh" content="120">
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#667085;--line:#e4e7ec;--track:#eef0f3;--link:#2563eb;
 --ok:#16a34a;--warn:#d97706;--bad:#dc2626;--bar:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--muted:#8b93a1;--line:#272c36;
 --track:#232833;--link:#60a5fa;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--bar:#60a5fa}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
 font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:20px;margin:0 0 4px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:20px 0 8px}
.muted{color:var(--muted)}.small{font-size:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,200px),1fr));gap:12px;margin-top:12px}
.tile,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.tile b{display:block;font-size:24px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:12px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.head h3{margin:0;font-size:16px}
.st{font-size:12px;font-weight:600;border-radius:6px;padding:1px 8px;background:var(--track)}
.st.working{color:var(--ok)}.st.idle{color:var(--muted)}.st.blocked,.st.stalled{color:var(--warn)}.st.offline{color:var(--bad)}
.bar{height:6px;background:var(--track);border-radius:3px;overflow:hidden;margin:6px 0 4px}.bar i{display:block;height:100%;background:var(--bar)}
.row{margin:3px 0}.ell{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.prob{font:12px ui-monospace,Menlo,monospace;color:var(--warn);margin:3px 0;overflow-wrap:anywhere}.prob.old{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 8px;border-top:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:500}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
</style></head><body>
<h1>Pipeline status</h1>
<div class="muted small">snapshot {{ ago(s.at) }} ago (every 2 min) · <a href="#" onclick="location.port=8770;return false">review hub</a>
 · <a href="http://131.155.233.55:8090/">machines</a>{% if s.error %} · <span style="color:var(--bad)">{{ s.error }}</span>{% endif %}</div>
<div class="tiles">
 <a class="tile" href="#" onclick="location.port=8770;return false"><span class="muted small">Batches to review</span>
  <b>{{ s.batches|length }}</b><span class="small muted">{{ s.batches|sum(attribute='waiting') }} videos waiting</span></a>
 <div class="tile"><span class="muted small">Sheet updates waiting</span><b>{{ s.sheet_todo }}</b>
  <span class="small muted">sheet_todo.json, entered by hand</span></div>
 <div class="tile"><span class="muted small">Last worker sync</span><b>{{ ago(s.last_sync) if s.last_sync else '–' }}</b>
  <span class="small muted">ago (hourly)</span></div>
 <div class="tile"><span class="muted small">Refresh queue</span><b>{{ s.queue }}</b>
  <span class="small muted">{% for p, n in s.queue_phases %}phase {{ p }}: {{ n }}{{ ' · ' if not loop.last }}{% endfor %}</span></div>
</div>
<h2>Workers</h2><div class="grid">
{% for w in s.workers %}<div class="card">
 <div class="head"><h3>{{ w.host }}</h3><span class="st {{ w.state }}">{{ w.state }}</span></div>
 <div class="muted small">{{ w.role }}</div>
 {% if w.state == 'offline' %}<div class="prob">{{ w.error }}</div>{% else %}
 {% if w.channel %}<div class="row ell" style="margin-top:8px"><b>{{ w.name }}</b>
  <span class="muted">{% if w.row %}· row {{ w.row }} {% endif %}· {{ w.country }}</span> <a class="small" href="{{ w.channel }}" target="_blank">↗</a></div>{% endif %}
 {% if w.n %}<div class="bar"><i style="width:{{ (100 * w.i / w.n)|round(1) }}%"></i></div>
 <div class="small muted">video {{ w.i }} of {{ w.n }}{% if w.rate %} · {{ '%.1f'|format(w.rate) }} videos/h · about
  {{ dur((w.n - w.i) / w.rate * 3600) }} for the rest of the channel{% endif %}</div>
 <div class="row ell small" title="{{ w.title }}">now: {{ w.title or w.video }} <span class="muted">(since {{ ago(w.since) }})</span></div>{% endif %}
 {% if w.batch %}<div class="small muted">last batch {{ ago(w.batch.at) }} ago: {{ w.batch.proposed }} proposed, {{ w.batch.excluded }} excluded</div>{% endif %}
 <div class="small muted">log written {{ ago(w.clock - w.log_age) }} ago</div>
 {% for p in w.problems %}<div class="prob {{ '' if p.current else 'old' }}">{{ p.text }}</div>{% endfor %}
 {% endif %}</div>{% endfor %}
</div>
{% if s.batches %}<h2>Batches waiting for review</h2><div class="card"><table>
<tr><th>channel</th><th>fetched on</th><th class="n">to review</th><th class="n">proposed</th></tr>
{% for b in s.batches %}<tr><td>{{ b.name }}</td><td class="muted">{{ b.host }}</td><td class="n">{{ b.waiting }}</td><td class="n">{{ b.proposed }}</td></tr>{% endfor %}
</table></div>{% endif %}
<h2>Channels by state</h2><div class="card"><table>
<tr><th>state</th>{% for h in s.hosts %}<th class="n">{{ h }}</th>{% endfor %}<th class="n">all</th></tr>
{% for name, counts in s.statuses %}<tr><td>{{ name }}</td>{% for n in counts %}<td class="n">{{ n or '' }}</td>{% endfor %}
 <td class="n">{{ counts|sum }}</td></tr>{% endfor %}
</table></div>
</body></html>"""


def dur(sec):
    return f'{sec / 60:.0f} min' if sec < 5400 else f'{sec / 3600:.0f} h' if sec < 172800 else f'{sec / 86400:.1f} days'


@app.route('/')
def index():
    if not snapshot.get('at'):
        return '<meta http-equiv="refresh" content="5">Collecting the first snapshot…'
    return render_template_string(PAGE, s=snapshot, dur=dur,
                                  ago=lambda t: dur(time.time() - t) if t else '?')


@app.route('/status.json')
def status_json():
    return jsonify(snapshot)


def _check_parse_log():
    """parse_log on a made-up log: day wrap, progress, rate, a problem after the last progress = blocked."""
    lines = ['23:50 row 7 https://www.youtube.com/@x/videos (Chile): next 15 proposals',
             '23:55   [1/10] analysing aaa First', '00:05   [2/10] analysing bbb Second',
             '00:15   [3/10] analysing ccc Third', '00:20   YouTube asks to sign in: waiting 15 min']
    now = 1_000_000.0
    w = parse_log(lines, mtime=now, now=now)
    assert (w['channel'], w['row'], w['country'], w['i'], w['n']) == ('https://www.youtube.com/@x/videos', '7', 'Chile', 3, 10)
    assert w['since'] == now - 5 * 60 and abs(w['rate'] - 6.0) < 1e-9, (w['since'], w['rate'])
    assert w['state'] == 'blocked' and w['problems'][0]['current']
    assert parse_log(lines[:-1] + ['00:30 no channel left to process'], now, now)['state'] == 'idle'
    assert parse_log(lines[:-1], now - 2 * 3600, now)['state'] == 'stalled'


if __name__ == '__main__':
    _check_parse_log()
    threading.Thread(target=refresher, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False)
