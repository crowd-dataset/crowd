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

import pandas as pd

from flask import Flask, jsonify, redirect, render_template_string, request

import add_video
import propose_segments as ps
import run_channels as rc

PORT = 8770
HOST = os.environ.get('REVIEW_HOST', '127.0.0.1')  # 0.0.0.0: reachable from other machines (the review pages too)
app = Flask(__name__)
servers = {}  # channel name -> (review process, its port)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Batches to review</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🚙</text></svg>">
<meta http-equiv="refresh" content="3600">
<style>
body{font:15px system-ui,sans-serif;margin:24px;max-width:760px;color:#222}
a.b{display:flex;justify-content:space-between;padding:12px 16px;margin:8px 0;border:1px solid #ccc;
 border-radius:8px;text-decoration:none;color:inherit}
a.b:hover{background:#f3f6ff;border-color:#68f}
.n{font-weight:600}.s{color:#666}.done{color:#999}
p.ok{color:#888;margin:2px 0;font-size:13px}p.bad{color:#b00020;font-weight:600;margin:4px 0}
.acts{display:flex;gap:8px;margin-top:12px}
button.g{flex:1;padding:10px 8px;font-size:14px;border:0;border-radius:4px;background:#007bff;color:#fff;cursor:pointer;
 white-space:nowrap}
button.g:disabled{background:#9bb;cursor:wait}
#pop{position:fixed;left:50%;top:20px;transform:translateX(-50%);max-width:640px;padding:14px 40px 14px 18px;
 border-radius:8px;background:#1e7e34;color:#fff;box-shadow:0 4px 18px rgba(0,0,0,.25);font-size:15px;display:none;z-index:9}
#pop.bad{background:#b00020}
#pop button{position:absolute;right:8px;top:8px;border:0;background:none;color:#fff;font-size:18px;cursor:pointer}
</style></head><body>
<h2>Batches to review <span class="s">({{ batches|sum(attribute="waiting") }} videos)</span></h2>
<p class="ok"><a href="#" onclick="location.port=8771;return false">Pipeline status</a>: what each worker does, queues, channels</p>
{% for w in health %}<p class="{{ 'ok' if w.ok else 'bad' }}">{{ w.host }}: {{ w.what }}</p>{% endfor %}
{% for b in batches %}
<a class="b" href="/open/{{ b.name }}" target="_blank"><span><span class="n">{{ b.name }}</span>
 <span class="s">{% if b.row %}· row {{ b.row }} {% endif %}· {{ b.state }}</span></span>
 <span>{{ b.waiting }} of {{ b.proposed }} to review</span></a>
{% else %}<p>Nothing to review.</p>{% endfor %}
{% if done %}<p class="done">All decided: {{ done|join(', ') }}</p>{% endif %}
<div class="acts">
<button class="g" title="commit mapping.csv as it is now and push it to GitHub"
 onclick="act(this, '/push_mapping', 'Commit mapping.csv as it is now and push it to GitHub?')">Push mapping to GitHub</button>
<button class="g" title="add the new videos of the mapping-NAME.csv files on GitHub to mapping.csv, make those files copies of it and push all"
 onclick="act(this, '/merge_mapping', 'Add the new videos of the mapping-NAME.csv files on GitHub to mapping.csv, make those files copies of it and push all?')"
 >Merge mapping-NAME &amp; push</button>
<button class="g" title="bring the workers' new proposals here now, instead of waiting for the hourly sync"
 onclick="act(this, '/fetch_batches')">Fetch new batches</button>
</div>
<div id="pop"><span id="popmsg"></span><button onclick="this.parentNode.style.display='none'" title="close">&times;</button></div>
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
<p class="s">Updates every hour (reload for the latest). <a href="/lessons">Review lessons</a>: what the twice-daily
check found in your reviews.</p>
</body></html>"""


@app.route('/')
def index():
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
    return render_template_string(PAGE, batches=batches, done=done, health=health)


@app.route('/open/<name>')
def open_batch(name):
    if not os.path.exists(os.path.join(rc.ROOT, name, 'plan.json')):
        return 'unknown batch', 404
    proc, port = servers.get(name, (None, None))
    if proc is None or proc.poll() is not None:
        with socket.socket() as s:
            s.bind((HOST, 0))
            port = s.getsockname()[1]
        plan = os.path.join(rc.ROOT, name, 'plan.json')
        # the review page opens the system browser by itself; here the click already opens it
        code = ('import webbrowser; webbrowser.open = lambda *a, **k: None; import propose_segments as ps; '
                f'ps.review({plan!r}, {port})')
        proc = subprocess.Popen([sys.executable, '-c', code])
        servers[name] = proc, port
        return render_template_string(  # give the server a moment to start
            '<meta http-equiv="refresh" content="3;url=http://{{h}}:{{p}}">Starting the review page…', p=port, h=here())
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
    if all(git('rev-parse', f'origin/main:{f}', check_tree=True) == b for f, b in blobs.items()):
        return None
    index = os.path.join(git('rev-parse', '--absolute-git-dir'), 'push-mapping.index')
    env = {**os.environ, 'GIT_INDEX_FILE': index}
    git('read-tree', 'origin/main', env=env)
    for f, b in blobs.items():
        git('update-index', '--add', '--cacheinfo', f'100644,{b},{f}', env=env)
    tree = git('write-tree', env=env)
    os.remove(index)
    commit = git('commit-tree', tree, '-p', 'origin/main', '-m', message)
    git('push', '-q', 'origin', f'{commit}:refs/heads/main')
    return commit


MAPPING = os.path.basename(add_video.FILE_PATH)
FLAT, NESTED = ('videos', 'vehicle_type', 'upload_date', 'channel'), ('time_of_day', 'start_time', 'end_time')


def push_mapping():
    """mapping.csv as it is here, pushed to GitHub. Returns a message."""
    with ps.plan_lock(add_video.FILE_PATH):  # not halfway through an Apply
        git('fetch', '-q', 'origin', 'main')
        commit = commit_to_github([MAPPING], 'Update mapping.csv: reviewed batches')
    return f'Pushed the mapping to GitHub (commit {commit[:8]}).' if commit else \
        'Nothing to push: GitHub has this mapping already.'


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
        added = []
        for name in names:
            other = pd.read_csv(io.StringIO(git('show', f'origin/main:{name}')))
            for _, r in other.iterrows():
                cells = {c: add_video._parse_flat_list_cell(r[c]) for c in FLAT}
                cells.update({c: add_video._safe_literal_eval(r[c], []) for c in NESTED})
                new = [i for i, v in enumerate(cells['videos']) if v not in have]
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
                added += [f"{cells['videos'][k]} ({r['locality']}, from {name})" for k in new]
                have |= {cells['videos'][k] for k in new}
        if added:
            add_video.save_csv(df, add_video.FILE_PATH)
        for name in names:
            shutil.copyfile(add_video.FILE_PATH, name)
        what = f'{len(added)} new videos from ' + ', '.join(names) if added else 'no new videos in ' + ', '.join(names)
        commit = commit_to_github([MAPPING, *names], f'Update mapping.csv: {what}; ' + ', '.join(names)
                                  + ' to match it')
    return (f'Merged {what}' + (': ' + ', '.join(added[:20]) + (' …' if len(added) > 20 else '') if added else '')
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
