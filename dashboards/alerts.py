"""Post problems that need the user to Discord, from the workstations dashboard (port 8090) and the pipeline status
page (8771) on this machine. Cron runs it every 10 minutes on the reviewing machine (stdlib only, Python 3.9+):

    CROWD_HOST=<its address> python3 alerts.py   # check and post; links in the posts use CROWD_HOST
    python3 alerts.py --dry-run                   # print what it would post, change nothing

The webhook URL is read from ~/.config/crowd-alerts/discord_webhook (chmod 600, never in a repository).

An alert is posted when it starts, again every REPEAT_H hours while it lasts, and once more when it clears; the
alerts open now are kept in ~/.config/crowd-alerts/state.json.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

STATE_DIR = Path.home() / '.config' / 'crowd-alerts'
HOST = os.environ.get('CROWD_HOST', 'localhost')
MACHINES = 'http://127.0.0.1:8090/status.json'
PIPELINE = 'http://127.0.0.1:8771/status.json'
HUB = f'http://{HOST}:8770'
STATUS_PAGE = f'http://{HOST}:8771'
REPEAT_H = 6
NEW_BATCHES = False  # post "new batch to review" (off for now, user 2026-10-13: may turn it on later)
ROOT_MIN_GB, ANY_MIN_GB = 100, 50  # free space below which a disk is reported: root/system disks, any disk
ROOTS = ('/', 'C:\\')


def fetch(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def problems():
    """{key: message} for everything wrong now, plus the batches waiting for review (to spot new ones)."""
    out, batches = {}, None
    try:
        for m in fetch(MACHINES):
            if not m.get('ok'):
                out[f'machine:{m["name"]}'] = f'**{m["name"]}** is not reachable ({m.get("error", "?")})'
                continue
            for d in m.get('disks', []):
                gb = d['free'] / 1e9
                if gb < ANY_MIN_GB or (d['mount'] in ROOTS and gb < ROOT_MIN_GB):
                    out[f'disk:{m["name"]}:{d["mount"]}'] = (f'**{m["name"]}** disk `{d["mount"]}` has only '
                                                             f'{gb:.0f} GB free of {d["total"] / 1e12:.1f} TB')
    except Exception as e:
        out['page:machines'] = f'the machines dashboard does not answer ({e})'
    try:
        p = fetch(PIPELINE)
        if time.time() - p.get('at', 0) > 15 * 60:
            out['page:pipeline-stale'] = 'the pipeline status page has not refreshed for over 15 minutes'
        for w in p.get('workers', []):
            if w['state'] in ('blocked', 'stalled', 'offline'):
                detail = (w.get('problems') or [{}])[-1].get('text') or w.get('error') or (
                    f'no new log line for {w.get("log_age", 0) / 60:.0f} min')
                out[f'worker:{w["host"]}:{w["state"]}'] = f'worker **{w["host"]}** is {w["state"]}: `{detail[:300]}`'
        batches = {b['name']: b['waiting'] for b in p.get('batches', [])}
    except Exception as e:
        out['page:pipeline'] = f'the pipeline status page does not answer ({e})'
    return out, batches


def post(text, dry):
    if dry:
        print('POST', text)
        return
    url = (STATE_DIR / 'discord_webhook').read_text().strip()
    req = urllib.request.Request(url, json.dumps({'content': text[:1900], 'username': 'crowd alerts'}).encode(),
                                 {'Content-Type': 'application/json', 'User-Agent': 'crowd-alerts (mini)'})
    urllib.request.urlopen(req, timeout=20)


def main(dry=False):
    state_path = STATE_DIR / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'open': {}, 'batches': None}
    now, found = time.time(), problems()
    current, batches = found
    msgs = []
    for key, text in current.items():
        last = state['open'].get(key)
        if last is None:
            msgs.append(f'🔴 {text}')
        elif now - last['posted'] > REPEAT_H * 3600:
            msgs.append(f'🟠 still: {text} (since {time.strftime("%a %H:%M", time.localtime(last["since"]))})')
        else:
            continue
        state['open'][key] = {'since': (last or {}).get('since', now), 'posted': now}
    for key in [k for k in state['open'] if k not in current]:
        msgs.append(f'🟢 resolved: {key.replace(":", " ")}')
        del state['open'][key]
    if batches is not None:
        if NEW_BATCHES and state['batches'] is not None:  # not on the first run: everything would look new
            for name in sorted(set(batches) - set(state['batches'])):
                msgs.append(f'📥 new batch to review: **{name}** ({batches[name]} videos) {HUB}')
        state['batches'] = batches
    if msgs:
        post('\n'.join(msgs) + (f'\n<{STATUS_PAGE}>' if any(m[0] != '📥' for m in msgs) else ''), dry)
    if not dry:
        state_path.write_text(json.dumps(state))
    elif not msgs:
        print('nothing to post')


if __name__ == '__main__':
    main(dry='--dry-run' in sys.argv)
