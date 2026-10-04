"""Local page listing the channel batches to review; a click opens a batch's review page.

The list is rebuilt from _output/proposals/channels.json on every load and the page reloads itself, so batches
appear while run_channels.py adds videos and drop off once everything in them is decided.

    /Users/pavlo/opt/anaconda3/bin/python review_hub.py   # then open http://127.0.0.1:8770
"""
import os
import socket
import subprocess
import sys

from flask import Flask, redirect, render_template_string

import run_channels as rc

PORT = 8770
app = Flask(__name__)
servers = {}  # channel name -> (review process, its port)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Batches to review</title>
<meta http-equiv="refresh" content="3600">
<style>
body{font:15px system-ui,sans-serif;margin:24px;max-width:760px;color:#222}
a.b{display:flex;justify-content:space-between;padding:12px 16px;margin:8px 0;border:1px solid #ccc;
 border-radius:8px;text-decoration:none;color:inherit}
a.b:hover{background:#f3f6ff;border-color:#68f}
.n{font-weight:600}.s{color:#666}.done{color:#999}
</style></head><body>
<h2>Batches to review</h2>
{% for b in batches %}
<a class="b" href="/open/{{ b.name }}" target="_blank"><span><span class="n">{{ b.name }}</span>
 <span class="s">· row {{ b.row }} · {{ b.state }}</span></span>
 <span>{{ b.waiting }} of {{ b.proposed }} to review</span></a>
{% else %}<p>Nothing to review.</p>{% endfor %}
{% if done %}<p class="done">All decided: {{ done|join(', ') }}</p>{% endif %}
<p class="s">Updates every hour (reload for the latest).</p>
</body></html>"""


@app.route('/')
def index():
    batches, done = [], []
    for url, c in rc.load_state().items():
        waiting, proposed = rc.review_counts(c['name'])
        if waiting:
            state = 'all videos proposed' if c['status'] == 'to review' else 'still being processed'
            batches.append(dict(name=c['name'], row=c['row'], state=state, waiting=waiting, proposed=proposed))
        elif proposed:
            done.append(c['name'])
    return render_template_string(PAGE, batches=batches, done=done)


@app.route('/open/<name>')
def open_batch(name):
    if name not in {c['name'] for c in rc.load_state().values()}:
        return 'unknown batch', 404
    proc, port = servers.get(name, (None, None))
    if proc is None or proc.poll() is not None:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        plan = os.path.join(rc.ROOT, name, 'plan.json')
        # the review page opens the system browser by itself; here the click already opens it
        code = ('import webbrowser; webbrowser.open = lambda *a, **k: None; import propose_segments as ps; '
                f'ps.review({plan!r}, {port})')
        proc = subprocess.Popen([sys.executable, '-c', code])
        servers[name] = proc, port
        return render_template_string(  # give the server a moment to start
            '<meta http-equiv="refresh" content="3;url=http://127.0.0.1:{{p}}">Starting the review page…', p=port)
    return redirect(f'http://127.0.0.1:{port}')


if __name__ == '__main__':
    app.run(port=PORT, debug=False)
