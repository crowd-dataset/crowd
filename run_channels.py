"""Work through the channels sheet continuously with the latest propose_segments.py.

From the bottom of the sheet, every channel with no status (column B) is processed chunk by chunk, its row set to
"Processing" so that someone working on the sheet in parallel leaves it alone (and rows others set are left
alone here); each chunk
runs propose_segments.py as a new process, so changes to it apply from the next chunk on. A channel whose videos
are all proposed waits for review and the next channel starts. Progress is kept in
_output/proposals/channels.json, and the batches left to review are listed in _output/proposals/to_review.md.

The sheet cannot be written from here (it needs a Google login): lines starting with "SHEET:" say which row to
set to "Processing" (started here), "Processed" (all its videos reviewed) or "Rejected" (it edits its drives).

    python run_channels.py [--chunk 15] [--max-minutes 110]
"""
import argparse
import csv
import fcntl
import io
import json
import os
import re
import subprocess
import sys
import time

import requests

import propose_segments as ps

SHEET_CSV = ('https://docs.google.com/spreadsheets/d/18O6C0Ar-JxLoqsrbuSwKAHLOVVkZd26OdEiaIBPoCww'
             '/export?format=csv&gid=0')
ROOT = '_output/proposals'
STATE = os.path.join(ROOT, 'channels.json')
TO_REVIEW = os.path.join(ROOT, 'to_review.md')
# the sheet's short country names, as the mapping writes them
COUNTRY = {'USA': 'United States', 'US': 'United States', 'UK': 'United Kingdom', 'UAE': 'United Arab Emirates',
           'Korea': 'South Korea'}
BLOCKED_WAIT_S = 15 * 60
IDLE_WAIT_S = 10 * 60


def log(msg):
    print(f'{time.strftime("%H:%M")} {msg}', flush=True)


def sheet_rows():
    """[(row number, channel url, status, country)] of the channels sheet."""
    r = requests.get(SHEET_CSV, timeout=60)
    r.raise_for_status()
    rows = list(csv.reader(io.StringIO(r.content.decode('utf-8'))))
    return [(n, row[0].strip(), row[1].strip(), row[2].strip() if len(row) > 2 else '')
            for n, row in enumerate(rows[1:], 2) if row and row[0].strip()]


def channel_name(url):
    """The folder name propose_segments.py gives the channel."""
    return re.sub(r'\W+', '_', re.sub(r'^.*youtube\.com/(?:channel/)?@?', '', url).split('/')[0])


def country_of(url, sheet_country):
    """The channel's country as the mapping writes it: from the sheet, else the country the channel gives on
    YouTube; None when neither says."""
    name = COUNTRY.get(sheet_country, sheet_country)
    if name in ps.COUNTRIES:
        return name
    handle = re.search(r'youtube\.com/@([^/?]+)', url)
    try:
        items = ps._api('channels', part='snippet', forHandle=handle.group(1)).get('items') if handle else None
        code = items[0]['snippet'].get('country') if items else None
    except Exception:  # no key, quota, network: the channel waits for a country in the sheet
        code = None
    if code:
        for c in ps.COUNTRIES:
            if ps.common.get_iso2_country_code(ps.common.correct_country(c)) == code:
                return c
    return None


def review_counts(name):
    """(videos with something left to decide, videos proposed) in the channel's plan."""
    path = os.path.join(ROOT, name, 'plan.json')
    if not os.path.exists(path):
        return 0, 0
    with open(path) as f:
        plan = [p for p in json.load(f) if 'entries' in p]
    waiting = sum(any(not s.get('decision') and not s.get('applied') for e in p['entries'] for s in e['segments'])
                  for p in plan)
    return waiting, len(plan)


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(state):
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE)


def write_to_review(state):
    """The batches left to review, with the command that opens each."""
    lines = ['# Batches to review', '']
    for url, c in state.items():
        waiting, proposed = review_counts(c['name'])
        if waiting:
            done = 'all videos proposed' if c['status'] == 'to review' else 'still being processed'
            lines += [f"- **{c['name']}** (sheet row {c['row']}): {waiting} of {proposed} videos to review, {done}",
                      f"  `/Users/pavlo/opt/anaconda3/bin/python propose_segments.py --review "
                      f"{ROOT}/{c['name']}/plan.json`"]
    if len(lines) == 2:
        lines.append('Nothing to review.')
    with open(TO_REVIEW, 'w') as f:
        f.write('\n'.join(lines) + '\n')


OURS = ('', 'Processing')  # sheet statuses of a channel this routine works on


def check_reviews(state, sheet):
    """Ask for the sheet to show what happens here, until it does: "Processing" for channels started here,
    "Processed" for channels all proposed and all reviewed, "Rejected" for channels that edit their drives."""
    status = {url: s for _, url, s, _ in sheet}
    for url, c in state.items():
        if c['status'] == 'to review' and review_counts(c['name'])[0] == 0:
            c['status'] = 'reviewed'
        if c['status'] in ('processing', 'to review') and status.get(url) == '':
            log(f"SHEET: set row {c['row']} ({url}) to Processing")
        for done, word in (('reviewed', 'Processed'), ('rejected', 'Rejected')):
            if c['status'] == done and status.get(url, '') in OURS:
                log(f"SHEET: set row {c['row']} ({url}) to {word}")
            elif c['status'] == done:
                c['status'] = 'marked'


def on_hotspot():
    """Connected through an iPhone's Personal Hotspot (it always hands out 172.20.10.x; macOS hides the Wi-Fi
    name)."""
    r = subprocess.run(['route', '-n', 'get', 'default'], capture_output=True, text=True)
    return 'gateway: 172.20.10.' in r.stdout


def run_chunk(url, country, chunk):
    """One batch of the channel with the latest code; its output, streamed."""
    env = dict(os.environ)
    if os.path.exists('cookies.txt'):
        env.setdefault('YT_DLP_COOKIES', 'cookies.txt')
    cmd = [sys.executable, '-u', 'propose_segments.py', '--channel', url, '--country', country,
           '--limit', str(chunk)]
    out = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as proc:
        for line in proc.stdout:
            if on_hotspot():  # stop downloading at once; the batch continues once off the hotspot
                proc.terminate()
                out.append('switched to the hotspot\n')
                break
            if 'Warning' not in line:
                print('  ' + line.rstrip()[:200], flush=True)
                out.append(line)
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--chunk', type=int, default=15, help='new proposals per run of propose_segments.py')
    ap.add_argument('--max-minutes', type=float, help='stop after this long (between chunks)')
    args = ap.parse_args()
    lock = open(os.path.join(ROOT, 'runner.lock'), 'w')
    try:  # one routine at a time: the hourly task may find the last one still running
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log('another run_channels.py is running; leaving it to that one')
        return
    t0 = time.time()
    state = load_state()
    while not args.max_minutes or time.time() - t0 < args.max_minutes * 60:
        if on_hotspot():
            log('on the iPhone hotspot: waiting until another network is used')
            while on_hotspot():
                time.sleep(60)
            log('off the hotspot: continuing')
        try:
            sheet = sheet_rows()
        except requests.RequestException as e:
            log(f'could not read the sheet ({e}); trying again in 10 min')
            time.sleep(IDLE_WAIT_S)
            continue
        check_reviews(state, sheet)
        save_state(state)
        write_to_review(state)
        # the next channel from the bottom: one started here and still "Processing", else a new one with no
        # status; a channel someone else marked (even "Processing") is theirs
        todo = [(n, url, country) for n, url, status, country in reversed(sheet)
                if (status == '' and url not in state or status in OURS and url in state)
                and state.get(url, {}).get('status', 'processing') == 'processing']
        if not todo:
            log('no channel left to process; waiting for reviews and the sheet')
            time.sleep(IDLE_WAIT_S)
            continue
        n, url, sheet_country = todo[0]
        if url not in state:
            log(f'SHEET: set row {n} ({url}) to Processing')
        c = state.setdefault(url, {'row': n, 'name': channel_name(url), 'status': 'processing'})
        c['row'] = n
        country = c.get('country') or country_of(url, sheet_country)
        if not country:
            log(f'row {n} {url}: no country in the sheet or on YouTube; skipped, fill in column C')
            c['status'] = 'needs country'
            save_state(state)
            continue
        c['country'] = country
        save_state(state)
        log(f'row {n} {url} ({country}): next {args.chunk} proposals')
        out = run_chunk(url, country, args.chunk)
        if 'Could not list the channel' not in out:
            c.pop('list_failures', None)
        if 'asks to sign in' in out:
            log('YouTube asks to sign in: waiting 15 min (a fresh cookies.txt helps)')
            time.sleep(BLOCKED_WAIT_S)
        elif 'three videos in a row could not be analysed' in out:
            log('three videos in a row failed (YouTube blocking or no connection?): waiting 15 min')
            time.sleep(BLOCKED_WAIT_S)
        elif 'so it is rejected' in out:
            c['status'] = 'rejected'
        elif 'Could not list the channel' in out:
            # usually a block or a dropped connection: wait; give up on the channel only after 3 tries in a row
            c['list_failures'] = c.get('list_failures', 0) + 1
            if c['list_failures'] >= 3:
                c['status'] = 'error'
            else:
                log('could not list the channel (blocked or offline?): waiting 15 min')
                time.sleep(BLOCKED_WAIT_S)
        elif 'still to do' not in out:
            c['status'] = 'to review'
            log(f"row {n} {url}: all videos proposed; on to the next channel")
        save_state(state)
        write_to_review(state)
    log('time limit reached')


if __name__ == '__main__':
    main()
