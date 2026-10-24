"""Local page listing the channel batches to review; a click opens a batch's review page.

The list is rebuilt on every load from _output/proposals/channels.json, plus the earlier batches in that folder,
so batches appear while run_channels.py adds videos and drop off once everything in them is decided.

    /Users/pavlo/opt/anaconda3/bin/python review_hub.py   # then open http://127.0.0.1:8770
    REVIEW_HOST=0.0.0.0 python review_hub.py               # on the mini: open http://131.155.233.55:8770 anywhere
                                                           # on the TU/e network (its review pages are served too)

"Push the mapping" commits mapping.csv as it is here onto GitHub's main and pushes it.
"""
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pandas as pd

from flask import Flask, jsonify, redirect, render_template_string, request

import add_video
import propose_segments as ps
import run_channels as rc
import sync_workers

PORT = 8770
HOST = os.environ.get('REVIEW_HOST', '127.0.0.1')  # 0.0.0.0: reachable from other machines (the review pages too)
app = Flask(__name__)
servers = {}  # channel name -> (review process, its port)
started = {}  # channel name -> when its review page was started

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Batches to review</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
<text y=%22.9em%22 font-size=%2290%22>🚙</text></svg>">
<meta http-equiv="refresh" content="3600">
<style>
body{font:15px system-ui,sans-serif;margin:24px;max-width:760px;color:#222}
a.b{display:flex;justify-content:space-between;padding:12px 16px;margin:8px 0;border:1px solid #ccc;
 border-radius:8px;text-decoration:none;color:inherit}
a.b:hover{background:#f3f6ff;border-color:#68f}
.n{font-weight:600}.s{color:#666}.done{color:#999}
p.ok{color:#888;margin:2px 0;font-size:13px}
table.days{border-collapse:collapse;margin:8px 0 14px;font-size:14px;width:100%;font-variant-numeric:tabular-nums}
table.days td,table.days th{padding:5px 10px;text-align:right}
table.days th{font-weight:600;color:#555;border-bottom:2px solid #ccc}
table.days td:first-child,table.days th:first-child{text-align:left;white-space:nowrap}
table.days tr:nth-child(even) td{background:#f6f7fa}table.days tr.we td:first-child{color:#888}
table.days td.z{color:#c4c4c4}table.days tr.sum td{border-top:2px solid #ccc;font-weight:600;background:none}
table.days td.bar{width:30%;text-align:left}table.days td.bar i{display:block;height:9px;border-radius:3px;
 background:#68f;min-width:1px}
p.load{color:#666;margin:2px 0;font-size:13px}
p.load b.hi{color:#1b7f3b}p.load b.mid{color:#b26a00}p.load b.lo{color:#b00020}
p.bad{color:#b00020;font-weight:600;margin:4px 0}
.acts{display:flex;gap:8px;margin-top:12px}
button.g{flex:1;padding:10px 8px;font-size:14px;border:0;border-radius:4px;background:#007bff;
 color:#fff;cursor:pointer;
 white-space:nowrap}
button.g:disabled{background:#9bb;cursor:wait}
#pop{position:fixed;left:50%;top:20px;transform:translateX(-50%);max-width:640px;padding:14px 40px 14px 18px;
 border-radius:8px;background:#1e7e34;color:#fff;box-shadow:0 4px 18px rgba(0,0,0,.25);
 font-size:15px;display:none;z-index:9}
#pop.bad{background:#b00020}
#pop button{position:absolute;right:8px;top:8px;border:0;background:none;color:#fff;font-size:18px;cursor:pointer}
html[data-theme=dark]{filter:invert(.9) hue-rotate(180deg);background:#fff}
html[data-theme=dark] img,html[data-theme=dark] iframe,html[data-theme=dark] video,
html[data-theme=dark] .leaflet-container,html[data-theme=dark] .locmap,html[data-theme=dark] .routemap
{filter:invert(1) hue-rotate(180deg)}
#theme{position:fixed;right:12px;bottom:12px;z-index:9999;padding:6px 10px;border:1px solid #bbb;border-radius:16px;
background:#fff;color:#222;font-size:13px;cursor:pointer}
</style>
<script>
// dark unless switched to light; a cookie, not localStorage: the hub and each review page run on their own port,
// and cookies are shared across the ports of a host
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
</script></head><body>
<h2>🚙 Batches to review <span class="s">({{ batches|sum(attribute="waiting") }} videos)</span></h2>
<p class="ok"><a href="#" onclick="location.port=8771;return false">Pipeline status</a>:
 what each worker does, queues, channels</p>
{% if health | selectattr('ok') | list %}
<p class="load" title="load over the last 30 min: green busy, amber medium, red low">
{%- for w in health if w.ok and not w.same %}{% if not loop.first %} · {% endif %}<span title="{{ w.what }}">
{{- w.machine if w.wide else w.host }}
{%- if w.pct is not none %} {{ w.kind }} <b class="{{ w.level }}">{{ w.pct }}%</b>
{%- else %}: {{ w.short }}{% endif %}</span>
{%- endfor %}</p>{% endif %}
{% for w in health if not w.ok %}<p class="bad">{{ w.host }}: {{ w.what }}</p>{% endfor %}
{% for c in cookies %}<p class="bad">🍪 {{ c.host }} needs new YouTube cookies: sign in as {{ c.account }} in a new
 Incognito window, export the cookies as {{ c.file or 'cookies.txt' }}, close the window, then <code>{{ c.how }}</code>
 </p>{% endfor %}
{% for b in batches %}
<a class="b" href="/open/{{ b.name }}" target="_blank"><span><span class="n">{{ b.name }}</span>
 <span class="s">{% if b.row %}· row {{ b.row }} {% endif %}· {{ b.state }}</span></span>
 <span>{{ b.waiting }} of {{ b.proposed }} to review</span></a>
{% else %}<p>Nothing to review.</p>{% endfor %}
{% if unapplied %}<h2>✅ Approved, not in the mapping yet
 <span class="s">({{ unapplied|sum(attribute="count") }} segments)</span></h2>
<p class="ok">All decided, only Apply missing: open it and press Apply (a segment without a locality needs one first)</p>
{% for u in unapplied %}<a class="b" href="/open/{{ u.name }}" target="_blank"><span class="n">{{ u.name }}</span>
 <span>{{ u.count }} approved{% if u.no_place %}, {{ u.no_place }} without a locality{% endif %}</span></a>
{% endfor %}{% endif %}
<div class="acts">
<button class="g" title="commit mapping.csv as it is now and push it to GitHub"
 onclick="act(this, '/push_mapping', 'Commit mapping.csv as it is now and push it to GitHub?')"
>Push mapping to GitHub</button>
<button class="g"
 title="add the new videos of the mapping-NAME.csv files on GitHub to mapping.csv,
 make those files copies of it and push all"
 onclick="act(this, '/merge_mapping', 'Add the new videos of the mapping-NAME.csv files on GitHub to mapping.csv, ' +
  'make those files copies of it and push all?')"
 >Merge mapping-name &amp; push</button>
<button class="g" title="bring the workers' new proposals here now, instead of waiting for the hourly sync"
 onclick="act(this, '/fetch_batches')">Fetch new batches</button>
</div>
{% if done %}<details class="done"><summary>All decided ({{ done|length }})</summary>{{ done|join(', ') }}</details>
{%- endif %}
{% if added %}<h2>📥 Videos added per day <span class="s">(last {{ added.days|length }} days)</span></h2>
<p class="ok">By who added them (the mapping file each first appeared in), as of {{ hm(added.built) }}; rebuilt nightly
 with the dataset figures</p>
<table class="days"><tr><th>day</th>{% for who in added.people %}<th>{{ who }}</th>{% endfor %}<th>all</th>
<th></th></tr>
{% for d in added.days %}<tr{% if d.weekend %} class="we"{% endif %}><td>{{ d.label }}{% if d.partial %}
 <span class="s">(so far)</span>{% endif %}</td>
{% for who in added.people %}{% set n = d.counts.get(who, 0) %}<td{% if not n %} class="z"{% endif %}>
{{- '{:,}'.format(n) if n else '–' }}</td>{% endfor %}
{% set n = d.counts.values()|sum %}<td{% if not n %} class="z"{% endif %}>
<b>{{ '{:,}'.format(n) if n else '–' }}</b></td>
<td class="bar"><i style="width:{{ (100 * n / added.most)|round(1) if added.most else 0 }}%"></i></td></tr>{% endfor %}
<tr class="sum"><td>{{ added.days|length }} days</td>
{% for who in added.people %}<td>{{ '{:,}'.format(added.total.get(who, 0)) }}</td>{% endfor %}
<td><b>{{ '{:,}'.format(added.total.values()|sum) }}</b></td><td></td></tr></table>{% endif %}
<div id="pop"><span id="popmsg"></span>
<button onclick="this.parentNode.style.display='none'" title="close">&times;</button></div>
<script>
// the buttons work in the background and say how it went in a popup here (user, 2026-10-13)
async function act(btn, url, question) {
  if (question && !confirm(question)) return;
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Working…';
  let msg, ok;
  try {
    const r = await fetch(url, {method: 'POST'});
    ({msg, ok} = await r.json());
  } catch (e) { msg = 'The hub did not answer: ' + e; ok = false; }
  btn.disabled = false; btn.textContent = label;
  const pop = document.getElementById('pop');
  document.getElementById('popmsg').textContent = msg;
  pop.className = ok ? '' : 'bad'; pop.style.display = 'block';
  clearTimeout(pop.timer);
  if (ok) pop.timer = setTimeout(() => { pop.style.display = 'none'; }, 15000);
}
</script>
<p class="s">Updates every hour (reload for the latest). <a href="/lessons">Review lessons</a>: what the nightly
check (from 04:45, after 50 new reviews) found in your reviews, with rule changes to apply by hand.</p>
</body></html>"""


def stop_finished_pages():
    """Stop the review pages of batches with nothing left to review: they stayed up for days (user, 2026-10-17);
    /open starts one again. Only an hour after the plan last changed: a page went offline right after its last
    decision (user, 2026-10-17). Under the locks a page writes mapping.csv and its plan with, so none stops
    mid-Apply (a kill there truncated mapping.csv, 2026-10-05); in a page's order, its plan's then the mapping's:
    the other way round a page applying waits for the mapping while this waits for its plan (2026-10-23)."""
    for name, (proc, _) in list(servers.items()):
        plan = os.path.join(rc.ROOT, name, 'plan.json')
        if proc.poll() is None and (rc.review_counts(name)[0] or time.time() - os.path.getmtime(plan) < 3600):
            continue
        with ps.plan_lock(plan), ps.plan_lock(add_video.FILE_PATH):
            proc.terminate()
            proc.wait()
        del servers[name]


def approved_unapplied(name):
    """(approved segments of the batch not in the mapping yet, how many of them have no locality: Apply skips
    those)."""
    path = os.path.join(rc.ROOT, name, 'plan.json')
    if not os.path.exists(path):
        return 0, 0
    with open(path) as f:
        segs = [(e, s) for p in json.load(f) for e in p.get('entries', []) for s in e['segments']
                if s.get('decision') == 'approve' and not s.get('applied')]
    return len(segs), sum(not e.get('locality') for e, _ in segs)


@app.route('/')
def index():
    stop_finished_pages()
    batches, done, seen = [], [], set()
    for url, c in sorted(rc.load_state().items(), key=lambda kv: kv[1]['status'] != 'processing'):
        if c['name'] in seen:  # a sheet row and its 'refresh:' entry share the folder (CharruaNYC twice, 2026-10-12)
            continue
        seen.add(c['name'])
        waiting, proposed = rc.review_counts(c['name'])
        if waiting:
            state = 'still being processed' if c['status'] == 'processing' else 'all videos proposed'
            batches.append(dict(name=c['name'], row=c['row'], state=state, waiting=waiting, proposed=proposed))
        elif proposed:
            done.append(c['name'])
    known = {c['name'] for c in rc.load_state().values()}
    for name in sorted(set(os.listdir(rc.ROOT)) - known):  # batches made before run_channels.py
        waiting, proposed = rc.review_counts(name)
        if waiting:
            batches.append(dict(name=name, row=None, state='earlier batch', waiting=waiting, proposed=proposed))
    try:  # the workers' last check (sync_workers.check_workers, every 15 min on the reviewing machine)
        with open(os.path.join(rc.ROOT, 'worker_health.json')) as f:
            health = json.load(f)
    except (OSError, ValueError):
        health = []
    # "GPU 34%" -> one short coloured figure (user, 2026-10-13: on one line, high and low at a glance)
    loaded = set()
    for w in health:
        m = re.match(r'(GPU|CPU) (\d+)%$', w.get('what', ''))
        w['kind'], w['pct'] = (m.group(1), int(m.group(2))) if m else (None, None)
        # a load is the machine's, and so is a pause (wsl_yield.py pauses both workers of win5090): shown once, under
        # its name, not again for its second worker (lin5080b, wsl5090b; user, 2026-10-19 and 2026-10-20)
        w['machine'] = sync_workers.place(w['host'])[0]
        w['wide'] = m is not None or w.get('what', '').startswith('paused')
        w['same'] = w['wide'] and w['machine'] in loaded
        if w['wide']:
            loaded.add(w['machine'])
        w['level'] = None if not m else 'hi' if w['pct'] >= 60 else 'mid' if w['pct'] >= 30 else 'lo'
        w['short'] = w.get('what', '').split(':')[0]
    try:  # machines YouTube asks to sign in: the pipeline status page works them out (dashboards/pipeline_status.py)
        with urllib.request.urlopen('http://127.0.0.1:8771/status.json', timeout=3) as r:
            cookies = json.load(r).get('cookies', [])
    except (OSError, ValueError):
        cookies = []
    # approved, never applied, nothing left to decide: Apply forgotten, or its answer lost in an outage (user,
    # 2026-10-20); its page open or not (DrivingShetland's open page hid it: user, 2026-10-24); a batch with videos
    # still to decide is in the list above
    unapplied = []
    for name in sorted(os.listdir(rc.ROOT)):
        if os.path.isdir(os.path.join(rc.ROOT, name)) and rc.review_counts(name)[0]:
            continue
        count, no_place = approved_unapplied(name)
        if count:
            unapplied.append(dict(name=name, count=count, no_place=no_place))
    # batches the user reviews later (review_later.json: names; user, 2026-10-21: Scaryscooby proposed again): last
    try:
        with open(os.path.join(rc.ROOT, 'review_later.json')) as f:
            later = set(json.load(f))
    except (OSError, ValueError):
        later = set()
    for b in batches:
        if b['name'] in later:
            b['state'] = 'for later · ' + b['state']
    batches.sort(key=lambda b: b['name'] in later)
    return render_template_string(PAGE, batches=batches, done=done, health=health, cookies=cookies,
                                  unapplied=unapplied, added=load_added(), hm=day_time)


ADDED = os.path.join(rc.ROOT, 'added_videos.json')  # dashboards/added_videos.py, nightly (build_dataset_page.sh)


def load_added():
    """The videos each person added on each of the last days (dashboards/added_videos.py), with each person's total;
    None before its first build."""
    try:
        with open(ADDED) as f:
            added = json.load(f)
    except (OSError, ValueError):
        return None
    added['total'] = {who: sum(d['counts'].get(who, 0) for d in added['days']) for who in added['people']}
    added['most'] = max((sum(d['counts'].values()) for d in added['days']), default=0)  # the longest bar
    built = time.strftime('%Y-%m-%d', time.localtime(added['built']))
    for d in added['days']:  # "Fri 23 Oct"; the day of the build only up to then
        t = time.strptime(d['day'], '%Y-%m-%d')
        d.update(label=time.strftime('%a %-d %b', t), weekend=t.tm_wday >= 5, partial=d['day'] == built)
    return added


def day_time(t):
    return time.strftime('%d %b %H:%M', time.localtime(t))


@app.route('/open/<name>')
def open_batch(name):
    if not os.path.exists(os.path.join(rc.ROOT, name, 'plan.json')):
        return 'unknown batch', 404
    proc, port = servers.get(name, (None, None))
    # a page started before the review code last changed starts again with it (user, 2026-10-19: the player stayed
    # small on open pages), under the locks a page writes with, so none stops mid-Apply
    if proc is not None and proc.poll() is None and started.get(name, 0) < os.path.getmtime(ps.__file__):
        with ps.plan_lock(add_video.FILE_PATH), ps.plan_lock(os.path.join(rc.ROOT, name, 'plan.json')):
            proc.terminate()
            proc.wait()
    if proc is None or proc.poll() is not None:
        with socket.socket() as s:
            s.bind((HOST, 0))
            port = s.getsockname()[1]
        plan = os.path.join(rc.ROOT, name, 'plan.json')
        # the review page opens the system browser by itself; here the click already opens it
        code = ('import webbrowser; webbrowser.open = lambda *a, **k: None; import propose_segments as ps; '
                f'ps.review({plan!r}, {port})')
        proc = subprocess.Popen([sys.executable, '-c', code])
        servers[name], started[name] = (proc, port), time.time()
        # on to the page once it answers: it takes 4 s or more to start on a busy mini, and a fixed 3 s wait sent
        # the browser to a page not there yet (user, 2026-10-16)
        return render_template_string(
            '<meta http-equiv="refresh" content="90;url=http://{{h}}:{{p}}">Starting the review page…'
            '<script>const u = "http://{{h}}:{{p}}/";'
            '(async function go() { try { await fetch(u, { mode: "no-cors" }); location.href = u; }'
            ' catch (e) { setTimeout(go, 1000); } })();</script>',
            p=port, h=here())
    return redirect(f'http://{here()}:{port}')


def here():
    """The hub's address as the browser reached it (127.0.0.1 here, the mini's address from another machine)."""
    return request.host.rsplit(':', 1)[0]


def git(*args, check_tree=False, **kw):
    """git in the mapping's checkout; check_tree: a path missing from the tree gives '' instead of an error."""
    r = subprocess.run(['git', *args], capture_output=True, text=True, check=not check_tree,
                       cwd=os.path.dirname(os.path.abspath(add_video.FILE_PATH)), **kw)
    return r.stdout.strip() if r.returncode == 0 else ''


def commit_to_github(files, message):
    """Commit these files as they are here onto GitHub's main and push: the commit is built from origin/main with only
    these files replaced, so this checkout's other files and branch are left as they are. Returns the commit, or None
    when GitHub has them already. Call with git fetched."""
    blobs = {f: git('hash-object', '-w', f) for f in files}
    there = git('rev-parse', f'origin/main:{MAPPING}', check_tree=True)
    # a mapping.csv pushed whole undid what reached GitHub meanwhile (2026-10-21: Shadab's cut-time splits of 51
    # videos, reverted by the traffic index job): only one with GitHub's changes taken in (pull_github_mapping)
    if MAPPING in blobs and there not in (merged_with, git('rev-parse', mapping_base(), check_tree=True)):
        raise RuntimeError('GitHub has mapping.csv changes not taken in here: push it with push_mapping')
    if all(git('rev-parse', f'origin/main:{f}', check_tree=True) == b for f, b in blobs.items()):
        if MAPPING in blobs:
            record_pushed(blobs[MAPPING])
        return None
    index = os.path.join(git('rev-parse', '--absolute-git-dir'), 'push-mapping.index')
    env = {**os.environ, 'GIT_INDEX_FILE': index}
    git('read-tree', 'origin/main', env=env)
    for f, b in blobs.items():
        # the mode GitHub has for it (README.md is 100755: 0fe339a9 made it 100644, 2026-10-17); a new file 100644
        mode = (git('ls-tree', 'origin/main', '--', f).split() or ['100644'])[0]
        git('update-index', '--add', '--cacheinfo', f'{mode},{b},{f}', env=env)
    tree = git('write-tree', env=env)
    os.remove(index)
    commit = git('commit-tree', tree, '-p', 'origin/main', '-m', message)
    git('push', '-q', 'origin', f'{commit}:refs/heads/main')
    if MAPPING in blobs:
        record_pushed(blobs[MAPPING])
    return commit


merged_with = None  # GitHub's mapping.csv (blob) pull_github_mapping last took the changes of


def record_pushed(blob):
    with open(os.path.join(git('rev-parse', '--absolute-git-dir'), 'mapping-pushed'), 'w') as f:
        f.write(blob)


def mapping_base():
    """The mapping.csv this machine last pushed (a git object): GitHub's changes are those made since. Before the
    first record (record_pushed), the last mapping commit from here."""
    path = os.path.join(git('rev-parse', '--absolute-git-dir'), 'mapping-pushed')
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    commit = git('log', '-1', '-E', '--format=%H', '--grep=^(Update mapping.csv|Traffic index)', 'origin/main', '--',
                 MAPPING)
    return f'{commit}:{MAPPING}' if commit else ''


MAPPING = os.path.basename(add_video.FILE_PATH)
FLAT, NESTED = ('videos', 'vehicle_type', 'upload_date', 'channel'), ('time_of_day', 'start_time', 'end_time')


def add_videos(df, have, other, source, only=None):
    """The videos of mapping table other that df lacks (or only those of only), added to the row of the same
    locality, state and country, else as a new row. Returns (df, ["video (locality, from source)"]); have grows."""
    added = []
    for _, r in other.iterrows():
        cells = {c: add_video._parse_flat_list_cell(r[c]) for c in FLAT}
        cells.update({c: add_video._safe_literal_eval(r[c], []) for c in NESTED})
        new = [i for i, v in enumerate(cells['videos']) if v not in have and (only is None or v in only)]
        if not new:
            continue
        state = '' if pd.isna(r['state']) else r['state']
        hit = df.index[(df['locality'] == r['locality']) & (df['country'] == r['country'])
                       & (df['state'].fillna('') == state)]
        if len(hit):
            i = hit[0]
            for c in FLAT:
                df.at[i, c] = add_video.compact_flat_list(add_video._parse_flat_list_cell(df.at[i, c])
                                                          + [cells[c][k] for k in new])
            for c in NESTED:
                df.at[i, c] = add_video.compact_nested_list(add_video._safe_literal_eval(df.at[i, c], [])
                                                            + [cells[c][k] for k in new])
        else:
            row = r.to_dict()
            row.update({c: add_video.compact_flat_list([cells[c][k] for k in new]) for c in FLAT})
            row.update({c: add_video.compact_nested_list([cells[c][k] for k in new]) for c in NESTED})
            row['id'] = int(pd.to_numeric(df['id'], errors='coerce').max()) + 1
            df = pd.concat([df, pd.DataFrame([row])[df.columns]], ignore_index=True)
        added += [f"{cells['videos'][k]} ({r['locality']}, from {source})" for k in new]
        have |= {cells['videos'][k] for k in new}
    return df, added


def video_segments(df):
    """{video: ((locality, state, country), (times of day, starts, ends), row index, its place in the row)}."""
    out = {}
    for i, r in df.iterrows():
        nested = [add_video._safe_literal_eval(r[c], []) for c in NESTED]
        place = (r['locality'], '' if pd.isna(r['state']) else r['state'], r['country'])
        for k, v in enumerate(add_video._parse_flat_list_cell(r['videos'])):
            out[v] = (place, tuple(tuple(x[k]) if k < len(x) else () for x in nested), i, k)
    return out


def pull_github_mapping(df, have, base=None, there='origin/main'):
    """What reached mapping.csv on GitHub since this machine last pushed it (mapping_base), taken in here before it
    is pushed again, which would otherwise undo it: the videos added there (user, 2026-10-14), and the segment times
    changed there of a video left as it was here (user, 2026-10-21: Shadab's cut times split segments). A video
    changed on both sides stays as it is here, and one taken out here since does not come back. Call with git
    fetched. Returns (df, [what was taken in, or kept as here])."""
    global merged_with
    base, blob = base or mapping_base(), git('rev-parse', f'{there}:{MAPPING}')
    if not base or git('rev-parse', base) == blob:
        merged_with = blob
        return df, []
    videos = lambda t: {v for cell in t['videos'] for v in add_video._parse_flat_list_cell(cell)}  # noqa: E731
    remote = pd.read_csv(io.StringIO(git('show', f'{there}:{MAPPING}')))
    old = pd.read_csv(io.StringIO(git('show', base)))
    df, taken = add_videos(df, have, remote, 'GitHub', only=videos(remote) - videos(old))
    was, now, here = video_segments(old), video_segments(remote), video_segments(df)
    for v, (place, segs, _, _) in now.items():
        if v not in was or v not in here or segs in (was[v][1], here[v][1]) or place != was[v][0]:
            continue
        if here[v][1] != was[v][1]:
            taken.append(f'{v} (its segments changed here and on GitHub: kept as here)')
            continue
        i, k = here[v][2], here[v][3]
        for c, val in zip(NESTED, segs):
            cell = add_video._safe_literal_eval(df.at[i, c], [])
            cell[k] = list(val)
            df.at[i, c] = add_video.compact_nested_list(cell)
        taken.append(f'{v} (segments from GitHub)')
    merged_with = blob
    return df, taken


def mapping_changes():
    """The commit description of a mapping push: the videos added, changed (times, day/night, locality) or removed
    against GitHub's mapping.csv, by channel (user, 2026-10-14). Call with git fetched."""
    def per_video(t):
        out = {}
        for r in t.itertuples():
            vids = add_video._parse_flat_list_cell(r.videos)
            chans = add_video._parse_flat_list_cell(r.channel)
            nested = {c: add_video._safe_literal_eval(getattr(r, c), []) for c in NESTED}
            for k, v in enumerate(vids):
                place = (r.locality, None if pd.isna(r.state) else r.state, r.country)
                out.setdefault(v, [chans[k] if k < len(chans) else '', set()])[1].add(
                    (place, *(str(nested[c][k]) if k < len(nested[c]) else '' for c in NESTED)))
        return out
    try:
        there = per_video(pd.read_csv(io.StringIO(git('show', f'origin/main:{MAPPING}'))))
    except (ValueError, pd.errors.ParserError):
        return ''
    here = per_video(add_video.load_csv(add_video.FILE_PATH))
    names = {cid: name for cid, name in ps._channel_folders().items()}
    by_channel, totals = {}, {'added': 0, 'changed': 0, 'removed': 0}
    for v in here.keys() | there.keys():
        kind = 'added' if v not in there else 'removed' if v not in here else \
            'changed' if here[v][1] != there[v][1] else None
        if kind:
            cid = (here.get(v) or there[v])[0]
            counts = by_channel.setdefault(names.get(cid, cid or 'unknown channel'), {})
            counts[kind] = counts.get(kind, 0) + 1
            totals[kind] += 1
    if not by_channel:
        return ''
    head = ', '.join(f'{n} video{"s" if n > 1 else ""} {k}' for k, n in totals.items() if n) \
        + f', from {len(by_channel)} channel{"s" if len(by_channel) > 1 else ""}:'
    return head + ''.join(f"\n- {name}: " + ', '.join(f'{n} {k}' for k, n in counts.items())
                          for name, counts in sorted(by_channel.items(), key=lambda x: -sum(x[1].values())))


def renumber_ids(df):
    """Each row's id its place again, 1..n, when removed rows left gaps (user, 2026-10-19: 4190 ids did not match);
    returns how many changed."""
    want = list(range(1, len(df) + 1))
    changed = sum(int(a) != b for a, b in zip(df['id'], want))
    if changed:
        df['id'] = want
    return changed


def push_mapping(title='Update mapping.csv: reviewed batches', extra=()):
    """mapping.csv as it is here, with what reached GitHub meanwhile taken in first, pushed to GitHub with the files
    extra (the traffic index job: its log). Returns a message."""
    with ps.plan_lock(add_video.FILE_PATH):  # not halfway through an Apply
        git('fetch', '-q', 'origin', 'main')
        df = add_video.load_csv(add_video.FILE_PATH)
        have = {v for cell in df['videos'] for v in add_video._parse_flat_list_cell(cell)}
        df, pulled = pull_github_mapping(df, have)
        if pulled or renumber_ids(df):
            add_video.save_csv(df, add_video.FILE_PATH)
        body = mapping_changes()
        commit = commit_to_github([MAPPING, *extra], title + (f'\n\n{body}' if body else ''))
    first = f'From GitHub first: {"; ".join(pulled[:10])}{" …" if len(pulled) > 10 else ""}. ' if pulled else ''
    return first + (f'Pushed the mapping to GitHub (commit {commit[:8]}).' if commit else
                    'Nothing to push: GitHub has this mapping already.')


def merge_mapping_files():
    """The videos of GitHub's mapping-NAME.csv files (Olena's) that mapping.csv lacks, added to it: to the row of
    the same locality, state and country, else as a new row; a video the mapping has stays as it is here. Then each
    mapping-NAME.csv becomes a copy of the merged mapping (user, 2026-10-11) and all are pushed in one commit.
    Returns a message."""
    with ps.plan_lock(add_video.FILE_PATH):
        git('fetch', '-q', 'origin', 'main')
        names = [n for n in git('ls-tree', '--name-only', 'origin/main').split()
                 if re.fullmatch(r'mapping-[^/]+\.csv', n)]
        df = add_video.load_csv(add_video.FILE_PATH)
        have = {v for cell in df['videos'] for v in add_video._parse_flat_list_cell(cell)}
        df, pulled = pull_github_mapping(df, have)
        added = []
        for name in names:
            df, more = add_videos(df, have, pd.read_csv(io.StringIO(git('show', f'origin/main:{name}'))), name)
            added += more
        if pulled:
            add_video.save_csv(df, add_video.FILE_PATH)
        if added or renumber_ids(df):
            add_video.save_csv(df, add_video.FILE_PATH)
        for name in names:
            shutil.copyfile(add_video.FILE_PATH, name)
        what = f'{len(added)} new videos from ' + ', '.join(names) if added else 'no new videos in ' + ', '.join(names)
        body = mapping_changes()
        commit = commit_to_github([MAPPING, *names], f'Update mapping.csv: {what}; ' + ', '.join(names)
                                  + ' to match it' + (f'\n\n{body}' if body else ''))
    return ((f'From GitHub first: {"; ".join(pulled[:10])}{" …" if len(pulled) > 10 else ""}. ' if pulled else '')
            + f'Merged {what}' + (': ' + ', '.join(added[:20]) + (' …' if len(added) > 20 else '') if added else '')
            + (f'. Pushed (commit {commit[:8]}).' if commit else '. GitHub had all of it already.'))


@app.route('/lessons')
def lessons():
    """The review-lessons check's findings (review_lessons_check.md, written by the scheduled Claude task on this
    machine), then the report it reads (review_lessons.md)."""
    parts = []
    for name in ('review_lessons_check.md', 'review_lessons.md'):
        try:
            with open(os.path.join(rc.ROOT, name)) as f:
                parts.append(f.read())
        except OSError:
            parts.append(f'({name}: not written yet)')
    return render_template_string('<pre style="font:13px ui-monospace,monospace;white-space:pre-wrap;margin:24px;'
                                  'max-width:1000px">{{ text }}</pre>', text='\n\n---\n\n'.join(parts))


@app.route('/fetch_batches', methods=['POST'])
def fetch_batches():
    """Start a sync of the workers' batches (sync_workers.py, as the hourly one does) and come back to the list."""
    log = open(os.path.join(rc.ROOT, 'fetch_batches.log'), 'a')
    subprocess.Popen([sys.executable, 'sync_workers.py'], stdout=log, stderr=subprocess.STDOUT,
                     cwd=os.path.dirname(os.path.abspath(__file__)))
    return jsonify(ok=True, msg='Fetching new batches from all machines: reload this page in a few minutes to see '
                                'them.')


@app.route('/push_mapping', methods=['POST'])
@app.route('/merge_mapping', methods=['POST'])
def push_mapping_route():
    try:
        return jsonify(ok=True, msg=merge_mapping_files() if request.path == '/merge_mapping' else push_mapping())
    except subprocess.CalledProcessError as e:
        return jsonify(ok=False, msg=f'Push failed: {(e.stderr or e.stdout or str(e)).strip()[-500:]}')


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)
