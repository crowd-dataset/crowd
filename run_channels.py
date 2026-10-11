"""Work through the channels sheet continuously with the latest propose_segments.py.

From the bottom of the sheet, every channel with no status (column B) is processed chunk by chunk, its row set to
"Processing" so that someone working on the sheet in parallel leaves it alone (and rows others set are left
alone here); each chunk
runs propose_segments.py as a new process, so changes to it apply from the next chunk on. A channel whose videos
are all proposed waits for review and the next channel starts. Progress is kept in
_output/proposals/channels.json, and the batches left to review are listed in _output/proposals/to_review.md.

The sheet cannot be written from here (it needs a Google login): lines starting with "SHEET:" say which row to
set to "Processing" (started here), "Processed" (all its videos proposed) or "Rejected" (it edits its drives),
with "Claude+Pavlo" in column E (Processed by).

    python run_channels.py [--chunk 15] [--max-minutes 110]
"""
import argparse
import csv
import fcntl
import io
import json
import os
import re
import select
import subprocess
import sys
import threading
import time

import requests

import propose_segments as ps

SHEET_CSV = ('https://docs.google.com/spreadsheets/d/18O6C0Ar-JxLoqsrbuSwKAHLOVVkZd26OdEiaIBPoCww'
             '/export?format=csv&gid=0')
ROOT = '_output/proposals'
STATE = os.path.join(ROOT, 'channels.json')
TO_REVIEW = os.path.join(ROOT, 'to_review.md')
PAUSE = os.path.join(ROOT, 'pause')  # created while the user travels (Travel calendar)
# the sheet changes still to make, [{row, url, status, by}], for whoever edits the sheet (the hourly check)
SHEET_TODO = os.path.join(ROOT, 'sheet_todo.json')
# channels done before whose new uploads come once the sheet's are all done ([{url, country, name, why}],
# refresh_queue.py; user, 2026-10-10)
REFRESH_QUEUE = os.path.join(ROOT, 'refresh_queue.json')
# the sheet's short country names, as the mapping writes them
COUNTRY = {'USA': 'United States', 'US': 'United States', 'UK': 'United Kingdom', 'UAE': 'United Arab Emirates',
           'Korea': 'South Korea'}
BLOCKED_WAIT_S = 15 * 60
SILENT_BATCH_S = 45 * 60  # a batch silent this long is stuck (the longest analysis so far: 7 min)
IDLE_WAIT_S = 10 * 60
QUOTA_WAIT_S = 60 * 60  # the API quota resets at midnight Pacific time


def log(msg):
    print(f'{time.strftime("%H:%M")} {msg}', flush=True)


PROCESSED_BY = {}  # channel url -> column E (Processed by), as last read


def sheet_rows():
    """[(row number, channel url, status, country)] of the channels sheet."""
    r = requests.get(SHEET_CSV, timeout=60)
    r.raise_for_status()
    rows = list(csv.reader(io.StringIO(r.content.decode('utf-8'))))
    PROCESSED_BY.update({row[0].strip(): row[4].strip() if len(row) > 4 else '' for row in rows[1:] if row})
    return [(n, row[0].strip(), row[1].strip(), row[2].strip() if len(row) > 2 else '')
            for n, row in enumerate(rows[1:], 2) if row and row[0].strip()]


def channel_name(url):
    """The folder name propose_segments.py gives the channel."""
    return ps.channel_folder(url)


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
    # channels added meanwhile by sync_workers.py (the workers' channels) stay
    state = {**load_state(), **state}
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
WHO = 'Claude+Pavlo'  # column E (Processed by) of the channels this routine works on
BY = f', column E (Processed by) to {WHO}'


def ours(url, status):
    """Whether this routine may set the sheet row of url: blank or "Processing", or marked by it before (a channel
    found sped up after it was "Processed")."""
    return status.get(url, '') in OURS or PROCESSED_BY.get(url) == WHO


def write_sheet_todo(state, sheet):
    """The status (column B) and Processed by (column E) the sheet should show for the channels worked on here
    and does not yet, in SHEET_TODO."""
    status = {url: s for _, url, s, _ in sheet}
    # Processed once all its videos are proposed, before the review (user, 2026-10-09)
    want = {'processing': 'Processing', 'to review': 'Processed', 'reviewed': 'Processed', 'rejected': 'Rejected'}
    todo = [{'row': c['row'], 'url': url, 'status': want[c['status']], 'by': WHO, 'comment': c.get('comment', '')}
            for url, c in state.items()
            if c['status'] in want and not c.get('refresh') and ours(url, status)
            and (status.get(url, '') != want[c['status']] or PROCESSED_BY.get(url, '') != WHO)]
    with open(SHEET_TODO + '.tmp', 'w') as f:
        json.dump(todo, f, indent=1)
    os.replace(SHEET_TODO + '.tmp', SHEET_TODO)


def check_reviews(state, sheet):
    """Ask for the sheet to show what happens here, until it does: "Processing" for channels started here,
    "Processed" for channels all proposed and all reviewed, "Rejected" for channels that edit their drives."""
    status = {url: s for _, url, s, _ in sheet}
    for url, c in state.items():
        if c['status'] == 'to review' and review_counts(c['name'])[0] == 0:
            c['status'] = 'reviewed'
        if c.get('refresh'):  # new uploads of a channel done before: the sheet stays as it is
            continue
        if c['status'] == 'processing' and status.get(url) == '':
            log(f"SHEET: set row {c['row']} ({url}) to Processing{BY}")
        for done, word in (('to review', 'Processed'), ('reviewed', 'Processed'), ('rejected', 'Rejected')):
            if c['status'] == done and ours(url, status) and status.get(url, '') != word:
                log(f"SHEET: set row {c['row']} ({url}) to {word}{BY}")
            elif c['status'] == done:
                c['status'] = 'marked'


def on_hotspot():
    """Connected through an iPhone's Personal Hotspot (it always hands out 172.20.10.x; macOS hides the Wi-Fi
    name)."""
    if sys.platform != 'darwin':  # a worker machine on the campus network has no hotspot
        return False
    r = subprocess.run(['route', '-n', 'get', 'default'], capture_output=True, text=True)
    return 'gateway: 172.20.10.' in r.stdout


def run_chunk(url, country, chunk, single_city=False, new_only=False):
    """One batch of the channel with the latest code; its output, streamed. single_city: only drives within one
    city (set "single_city": true for the channel in channels.json)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('MallocStackLogging')}  # macOS: "can't turn off
    # malloc stack logging because it was not enabled" from every python the app's environment starts
    if os.path.exists('cookies.txt'):
        env.setdefault('YT_DLP_COOKIES', 'cookies.txt')
    cmd = [sys.executable, '-u', 'propose_segments.py', '--channel', url, '--country', country,
           '--limit', str(chunk)] + (['--single-city'] if single_city else []) + (['--new-only'] if new_only else [])
    out = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env) as proc:
        fd, pending, last, done = proc.stdout.fileno(), b'', time.time(), False
        while not done:
            # a read that never returns (a connection dropped mid-request, 2026-10-09) must not hold the routine;
            # raw reads, so no line waits in a buffer select cannot see
            ready, _, _ = select.select([fd], [], [], min(60, SILENT_BATCH_S))
            if not ready:
                if time.time() - last > SILENT_BATCH_S:
                    proc.terminate()
                    log(f'the batch printed nothing for {SILENT_BATCH_S // 60} min (stuck?): stopped it')
                    break
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            last = time.time()
            *lines, pending = (pending + chunk).split(b'\n')
            for line in (raw.decode(errors='replace') + '\n' for raw in lines):
                if on_hotspot() or os.path.exists(PAUSE):  # stop downloading at once; the batch continues later
                    proc.terminate()
                    out.append('switched to the hotspot or paused\n')
                    done = True
                    break
                if 'Warning' not in line:
                    print(f'{time.strftime("%H:%M")}   ' + line.rstrip()[:200], flush=True)
                    out.append(line)
    return ''.join(out)


CHANNEL_IDS = '_output/subscriptions/channel_ids.json'  # sheet URL -> channel id (refresh_queue.py, sync_workers.py)
TAKEN = os.path.join(ROOT, 'taken.json')  # the channels any machine works or worked on (sync_workers.py)


def done_before(url, sheet, state):
    """Another sheet row (or link) of the same channel that someone did or is doing, else None: @Drivesights1 (row
    823) is UCjaZ_Lko0XB49cCs-bnlz4A (row 836). Compared: the sheet's rows with any status, the channels in state and
    those all machines are on (TAKEN, from sync_workers.py)."""
    import refresh_queue  # imports this module
    try:
        with open(CHANNEL_IDS) as f:
            known = json.load(f)
    except (OSError, ValueError):
        known = {}
    try:
        with open(TAKEN) as f:
            taken = set(json.load(f))
    except (OSError, ValueError):
        taken = set()
    rows = {u: n for n, u, status, _ in sheet if status}
    others = (set(rows) | taken | {c.get('url') or k for k, c in state.items()}) - {url}
    try:
        cid = refresh_queue.channel_id(url, known)
        for u in others:  # each looked up once (one API unit apiece), then kept
            if u not in known and not u.startswith('refresh:'):
                refresh_queue.channel_id(u, known)
    except RuntimeError:  # no API key or quota: start it
        return None
    finally:
        os.makedirs(os.path.dirname(CHANNEL_IDS), exist_ok=True)
        with open(CHANNEL_IDS + '.tmp', 'w') as f:
            json.dump(known, f, indent=1)
        os.replace(CHANNEL_IDS + '.tmp', CHANNEL_IDS)
    same = [u for u in others if cid and known.get(u) == cid]
    return (f'row {rows[same[0]]}' if same[0] in rows else same[0]) if same else None


def refresh_todo(state, args):
    """[(None, state key, url, country, new uploads only)] of the channels done before this machine fetches again:
    every m-th of the refresh queue (--rows k/m), one started here first."""
    try:
        with open(REFRESH_QUEUE) as f:
            queue = json.load(f)
    except (OSError, ValueError):
        return []
    todo, share = [], args.queue_rows or args.rows
    try:  # started by another machine (sync_workers.py): left to it
        with open(TAKEN) as f:
            taken = set(json.load(f)) - set(state)
    except (OSError, ValueError):
        taken = set()
    for i, q in enumerate(queue):
        if 'refresh:' + q['url'] in taken:
            continue
        key, c = 'refresh:' + q['url'], state.get('refresh:' + q['url'], {})
        if (c.get('status', 'processing') == 'processing' and not c.get('host')
                and (key in state or share is None or i % share[1] == share[0])
                and (q.get('country') or '').strip().lower() != 'russia'):
            todo.append((key in state, None, key, q['url'], q.get('country') or '', q.get('new_only', True)))
    todo.sort(key=lambda t: not t[0])  # one started here first, else the queue's order (its phases)
    return [t[1:] for t in todo]


def push_code_on_change(every_s=60):
    """Forever: when a .py file here changes, send the code to the worker machines (user, 2026-10-10: every update
    reaches them at once, not with the hourly sync), and every hour bring their batches here (sync_workers.py), also
    while a long batch runs here (a night of 25-minute videos kept them away from 22:42 to 07:44). A thread of the
    reviewing machine's runner (--sync)."""
    import sync_workers  # the laptop's: it names the workers
    seen, synced = None, 0
    while True:
        if time.time() - synced > 3600 and not on_hotspot():
            log('syncing the worker machines')
            try:
                subprocess.run([sys.executable, 'sync_workers.py'], timeout=1800)
            except subprocess.TimeoutExpired:
                log('the worker sync took over 30 min: stopped it, next try in an hour')
            synced = time.time()
        now = {f: os.path.getmtime(f) for f in os.listdir('.') if f.endswith('.py')}
        if seen is not None and now != seen:
            for host in sync_workers.WORKERS:
                try:
                    pushed = sync_workers.push_code(host)
                    if pushed:
                        log(f'code sent to {host}: {", ".join(pushed)}')
                except (subprocess.SubprocessError, OSError) as e:
                    log(f'code not sent to {host} ({type(e).__name__}): the hourly sync tries again')
        seen = now
        time.sleep(every_s)


def rows_rule(text):
    """--rows: (k, m) for "k/m", (0, 2) for even, (1, 2) for odd."""
    if text in ('even', 'odd'):
        return (0, 2) if text == 'even' else (1, 2)
    k, m = (int(x) for x in text.split('/'))
    if not 0 <= k < m:
        raise argparse.ArgumentTypeError('k/m with 0 <= k < m')
    return k, m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--chunk', type=int, default=15, help='new proposals per run of propose_segments.py')
    ap.add_argument('--max-minutes', type=float, help='stop after this long (between chunks)')
    # several machines on one sheet: each takes only its own rows, so two never work on the same channel
    ap.add_argument('--rows', type=rows_rule, help='only new channels on these sheet rows: even, odd, or k/m '
                    '(the row number divided by m leaves k); a channel started here is finished either way')
    ap.add_argument('--queue-rows', type=rows_rule, help='the share of the refresh queue (default: --rows; '
                    'with --refresh-only and neither: any channel no other machine has started)')
    ap.add_argument('--refresh-only', action='store_true',
                    help='no new sheet channel, straight to the refresh queue (user, 2026-10-10: w5090)')
    ap.add_argument('--sync', action='store_true', help='bring the worker machines\' batches here every hour '
                    '(sync_workers.py): the machine you review on')
    ap.add_argument('--no-new', action='store_true',
                    help='finish the channels started here, start no new one (the worker machines take those)')
    args = ap.parse_args()
    lock = open(os.path.join(ROOT, 'runner.lock'), 'w')
    try:  # one routine at a time: the hourly task may find the last one still running
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log('another run_channels.py is running; leaving it to that one')
        return
    t0 = time.time()
    state = load_state()
    code = os.path.getmtime(__file__)
    if args.sync:
        threading.Thread(target=push_code_on_change, daemon=True).start()
    while not args.max_minutes or time.time() - t0 < args.max_minutes * 60:
        if on_hotspot():
            log('on the iPhone hotspot: waiting until another network is used')
            while on_hotspot():
                time.sleep(60)
            log('off the hotspot: continuing')
        if os.path.exists(PAUSE):
            log('paused (travelling): waiting until the pause file is removed')
            while os.path.exists(PAUSE):
                time.sleep(60)
            log('pause removed: continuing')
        if os.path.getmtime(__file__) != code:  # updated (on a worker: by sync_workers.py): run the new code
            log('run_channels.py changed: restarting with the new code')
            os.execv(sys.executable, [sys.executable, '-u'] + sys.argv)
        state = load_state()  # with the workers' progress, as synced by sync_workers.py
        for key, c in state.items():  # names as the folders are now called (no ?query: UCjaZ_Lko0XB49cCs-bnlz4A)
            c['name'] = channel_name(c.get('url') or key.removeprefix('refresh:'))
        try:
            sheet = sheet_rows()
        except requests.RequestException as e:
            log(f'could not read the sheet ({e}); trying again in 10 min')
            time.sleep(IDLE_WAIT_S)
            continue
        check_reviews(state, sheet)
        write_sheet_todo(state, sheet)
        save_state(state)
        write_to_review(state)
        # the next channel from the bottom: one started here and still "Processing", else a new one with no
        # status; a channel someone else marked (even "Processing") is theirs
        todo = [(n, url, country) for n, url, status, country in reversed(sheet)
                if (status == '' and url not in state or status in OURS and url in state)
                and state.get(url, {}).get('status', 'processing') == 'processing'
                and not state.get(url, {}).get('host')  # a worker machine's channel
                and (url in state or args.rows is None or n % args.rows[1] == args.rows[0])
                and not (url not in state and country.strip().lower() == 'russia')  # Olena's (user, 2026-10-10)
                and not ((args.no_new or args.refresh_only) and url not in state)]
        started = list(state)  # one channel at a time: the earliest started one first, a new one after them
        todo.sort(key=lambda t: started.index(t[1]) if t[1] in state else len(started))
        todo = [(n, url, url, country, False) for n, url, country in todo] or refresh_todo(state, args)
        if not todo:
            log('no channel left to process; waiting for reviews and the sheet')
            time.sleep(IDLE_WAIT_S)
            continue
        n, key, url, sheet_country, new_only = todo[0]  # n is None for a channel done before (refresh_todo)
        what = f'row {n} {url}' if n is not None else f"{'new uploads' if new_only else 'videos'} of {url}"
        if key not in state and n is not None and (other := done_before(url, sheet, state)):
            log(f'row {n} {url} is the channel of {other}, done there: skipped (the sheet lists it twice)')
            state[key] = {'row': n, 'name': channel_name(url), 'status': 'duplicate', 'comment': f'same as {other}'}
            save_state(state)
            continue
        if key not in state and n is not None:
            log(f'SHEET: set row {n} ({url}) to Processing{BY}')
        c = state.setdefault(key, {'row': n, 'name': channel_name(url), 'status': 'processing',
                                   **({} if n is not None else {'refresh': True, 'url': url, 'new_only': new_only})})
        c['row'] = n
        write_sheet_todo(state, sheet)
        country = c.get('country') or country_of(url, sheet_country)
        if not country:
            log(f'{what}: no country in the sheet or on YouTube; skipped, fill in column C')
            c['status'] = 'needs country'
            save_state(state)
            continue
        c['country'] = country
        save_state(state)
        log(f'{what} ({country}): next {args.chunk} proposals')
        out = run_chunk(url, country, args.chunk, c.get('single_city', False), c.get('new_only', False))
        if 'Could not list the channel' not in out:
            c.pop('list_failures', None)
        if 'Traceback (most recent call last)' in out:
            # a bug, not a finished channel: keep it in progress and wait for a fix
            log('propose_segments.py crashed (see above): waiting 15 min, the channel stays in progress')
            time.sleep(BLOCKED_WAIT_S)
        elif 'asks to sign in' in out:
            log('YouTube asks to sign in: waiting 15 min (a fresh cookies.txt helps)')
            time.sleep(BLOCKED_WAIT_S)
        elif 'three videos in a row could not be analysed' in out:
            log('three videos in a row failed (YouTube blocking or no connection?): waiting 15 min')
            time.sleep(BLOCKED_WAIT_S)
        elif 'exceeded your' in out and 'quota' in out:
            # not the channel's fault: stay on it until the quota resets instead of counting a failure
            log('YouTube Data API quota exceeded: waiting 60 min, the channel stays in progress')
            time.sleep(QUOTA_WAIT_S)
        elif 'so it is rejected' in out:
            c['status'] = 'rejected'
            # why, for the sheet's Comments column ("Ghosts detected: ...")
            c['comment'] = next((ln.strip().removeprefix('Stopped early: ').split(', so it is rejected')[0]
                                 for ln in out.splitlines() if 'so it is rejected' in ln), '')
        elif 'Could not list the channel' in out:
            # usually a block or a dropped connection: wait; give up on the channel only after 3 tries in a row
            c['list_failures'] = c.get('list_failures', 0) + 1
            if c['list_failures'] >= 3:
                c['status'] = 'error'
            else:
                log('could not list the channel (blocked or offline?): waiting 15 min')
                time.sleep(BLOCKED_WAIT_S)
        elif ps.CHANNEL_DONE in out:
            c['status'] = 'to review'
            log(f"{what}: all videos proposed; on to the next channel")
        elif 'still to do' not in out:
            # cut off (hotspot, killed, stopped early) without saying the channel is done: stay on it
            log(f'{what}: the batch ended without finishing the channel; staying on it')
            time.sleep(60)
        save_state(state)
        write_to_review(state)
    log('time limit reached')


if __name__ == '__main__':
    main()
