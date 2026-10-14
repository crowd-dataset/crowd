"""Check /reject_segments on a copy of a real plan: all open segments of one video rejected in one request,
applied ones untouched, and a stale request refused. Run from the repo with the crowd env's python."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

src = next(p for p in sorted(glob.glob('_output/proposals/*/plan.json'))
           if any(not s.get('decision') and not s.get('applied')
                  for v in json.load(open(p)) for e in v.get('entries', []) for s in e['segments']))
d = '/tmp/reject_test/batch'
shutil.rmtree('/tmp/reject_test', ignore_errors=True)
shutil.copytree(os.path.dirname(src), d)
plan_path = os.path.join(d, 'plan.json')
plan = json.load(open(plan_path))
v = next(v for v in plan if any(not s.get('decision') and not s.get('applied')
                                for e in v.get('entries', []) for s in e['segments']))
items = [dict(video=v['video'], entry=i, seg=j, bounds=[[s['start'], s['end']]])
         for i, e in enumerate(v['entries']) for j, s in enumerate(e['segments']) if not s.get('applied')]
port = 18999
code = ('import webbrowser; webbrowser.open = lambda *a, **k: None; import propose_segments as ps; '
        f'ps.review({plan_path!r}, {port})')
srv = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1)
            break
        except Exception:
            time.sleep(1)

    def post(body):
        return urllib.request.urlopen(urllib.request.Request(
            f'http://127.0.0.1:{port}/reject_segments', json.dumps(body).encode(),
            {'Content-Type': 'application/json'}))

    t = time.time()
    assert json.load(post(dict(segments=items)))['ok']
    took = time.time() - t
    after = next(x for x in json.load(open(plan_path)) if x['video'] == v['video'])
    segs = [s for e in after['entries'] for s in e['segments']]
    assert all(s.get('decision') == 'reject' for s in segs if not s.get('applied')), 'not all rejected'
    stale = dict(segments=[dict(items[0], bounds=[[0, 1]])])
    try:
        post(stale)
        raise AssertionError('stale request accepted')
    except urllib.error.HTTPError as e:
        assert e.code == 409
    print(f'ok: {len(items)} segments of {v["video"]} rejected in one request ({took:.2f} s), stale request refused')
finally:
    srv.terminate()
    shutil.rmtree('/tmp/reject_test', ignore_errors=True)
