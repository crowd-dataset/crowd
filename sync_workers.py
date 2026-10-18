"""Bring the batches of the worker machines to this laptop, and the mapping to them.

    python sync_workers.py

Each worker runs run_channels.py on its own sheet rows (WORKERS) in ~/crowd-routine. For every channel a worker
has started, this copies its analyses (signals, contact sheets, metadata) here and adds the videos it proposed
that the plan here does not have yet: decisions made here are never overwritten. The worker's channel progress
goes into channels.json here, which the review hub and the sheet updates read; then the mapping as reviewed here
goes to each worker, so its locality guesses and its skipping of videos already in the mapping are up to date.
"""
import hashlib
import json
import os
import re
import subprocess

import propose_segments as ps
import run_channels as rc

# ssh alias: the sheet rows and refresh-queue share it takes (run_channels.py --rows, --queue-rows); the laptop
# takes rows 2/3 and queue 2/4
# mini ('sheet rows 1/2, no queue') is out until its new account is set up (2026-10-11)
WORKERS = {'lin5080': 'queue phase 5 (highway videos to redo), then 2 (new uploads of processed channels)',
           'wsl5090': 'sheet rows 0/2, then queue phase 4 (UCxbv63aE7sg8_HzYz5KFPeA highway videos), 1 and 3',
           # account pavlo since 2026-10-12 (shadab before): ~/crowd-routine, started by cron with start_worker.sh
           'mini': 'sheet rows 1/2 (weirdlifenow, row 861)',
           # a second worker on lin5080 (user, 2026-10-18: its GPU had room), in its own folder
           'lin5080b': 'queue phase 6 (new uploads of the channels added from the mapping)'}
REMOTE = 'crowd-routine'       # the repository on each worker, relative to its home
PLACES = {'lin5080b': ('lin5080', 'crowd-routine-2')}  # a worker that is not REMOTE on the machine of its name


def place(host):
    """(ssh alias, folder) of a worker."""
    return PLACES.get(host, (host, REMOTE))


def ssh(host, command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', place(host)[0], command],
                          capture_output=True, text=True, check=True).stdout


def untouched(p):
    """Whether nobody has worked on a proposal in review: no segment decided or applied, no edit to undo, no mark."""
    return (not p.get('undo') and not any(p.get(k) for k in ps.REVIEW_MARKS)
            and not any(s.get('decision') or s.get('applied') for e in p.get('entries', []) for s in e['segments']))


def fingerprint(p):
    return hashlib.sha1(json.dumps({k: v for k, v in p.items() if k not in ('undo', 'redo')},
                                   sort_keys=True).encode()).hexdigest()


def merge_plan(name, remote_plan):
    """Add the worker's videos that this plan lacks (or has only as failed downloads). Returns how many."""
    path = os.path.join(rc.ROOT, name, 'plan.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # the worker's entries as at the last sync: a worker copy that is no different is no newer proposal, and must not
    # undo one made here since (user, 2026-10-15: fUlJsPo_04c back to Los Angeles, 79 videos proposed again here
    # reverted to the workers' older copies); none seen yet (the first sync): only remembered
    seen_path = os.path.join(rc.ROOT, name, 'remote_seen.json')
    try:
        with open(seen_path) as f:
            seen = json.load(f)
    except (OSError, ValueError):
        seen = None
    added = 0
    with ps.plan_lock(path):
        local = json.load(open(path)) if os.path.exists(path) else []
        index = {p['video']: i for i, p in enumerate(local)}
        for p in remote_plan:
            i = index.get(p['video'])
            changed_there = seen is not None and seen.get(p['video']) != fingerprint(p)
            if i is None:
                local.append(p)
                added += 1
            elif ((str(local[i].get('exclude', '')).startswith('could not analyse') or 'entries' not in local[i])
                  and 'entries' in p):
                local[i] = p  # failed or left out here, proposed there (a rule made less strict since)
                added += 1
            elif p != local[i] and 'entries' in p and untouched(local[i]) and untouched(p) and changed_there:
                # proposed again there (P3PHxtIMZEo: its GPS split came after the first copy); never a left-out copy
                # over a proposal here (3QeTN5BygQE and others proposed here again were left out by the sync)
                local[i] = p
                added += 1
        if os.path.exists(os.path.join(rc.ROOT, name, ps.CHANNEL_REJECTED)):
            # rejected in review here: what the worker proposed before it heard is rejected too (user, 2026-10-17:
            # tonechkanz offered again an hour after its rejection)
            ps.reject_undecided(local)
        with open(path + '.tmp', 'w') as f:
            json.dump(local, f, indent=1)
        os.replace(path + '.tmp', path)
        with open(seen_path + '.tmp', 'w') as f:
            json.dump({p['video']: fingerprint(p) for p in remote_plan}, f)
        os.replace(seen_path + '.tmp', seen_path)
    return added


DASHBOARD = 'http://131.155.233.55:8090/status.json'  # the compute dashboard on the mini: 3 h of load per minute
HEALTH = os.path.join(rc.ROOT, 'worker_health.json')  # the latest check_workers, shown on the review hub
LOW_GPU, LOW_CPU, LOW_FOR = 5, 15, 30  # % mean load (GPU, else CPU) over this many minutes: look why it is low
STUCK_MIN = 60  # no new runner.log line for this long while it should be working
PROBLEM = re.compile(r'sign in|bot check|quota|refused|Traceback|Error|could not|failed', re.I)
IDLE = re.compile(r'no channel left|waiting for|nothing (?:left|to do)|paused', re.I)


def check_workers():
    """[{host, ok, what}] for each worker whose load on the dashboard stayed low for LOW_FOR minutes (user,
    2026-10-12): a worker downloading uses no GPU, so low load alone is no fault; its runner.log and processes say
    what it does. Written to HEALTH; a problem new since the last check is logged."""
    import requests
    import time
    try:
        machines = {m.get('name'): m for m in requests.get(DASHBOARD, timeout=20).json()}
    except (requests.RequestException, ValueError) as e:
        machines, report = {}, [{'host': 'dashboard', 'ok': False, 'what': f'not reached ({e.__class__.__name__})'}]
    else:
        report = []
    for host in WORKERS if machines else []:
        alias, rem = place(host)
        try:  # paused by wsl_yield.py while someone uses the Windows PC (wsl5090): no fault
            y = json.loads(ssh(host, 'cat ~/.wsl_yield.json 2>/dev/null || echo "{}"'))
        except (subprocess.CalledProcessError, ValueError):
            y = {}
        if y.get('paused'):
            report.append({'host': host, 'ok': True, 'what': f'paused: {y.get("why", "")[:120]}'})
            continue
        m = machines.get(alias) or {}
        recent = [x for x in m.get('hist', []) if x[0] >= time.time() - LOW_FOR * 60]
        gpu = bool(m.get('gpus'))
        load = [x[2] if gpu else x[1] for x in recent if (x[2] if gpu else x[1]) is not None]
        mean = sum(load) / len(load) if load else None
        if mean is not None and mean >= (LOW_GPU if gpu else LOW_CPU):
            report.append({'host': host, 'ok': True, 'what': f'{"GPU" if gpu else "CPU"} {mean:.0f}%'})
            continue
        try:
            last, runners, downloads, now = ssh(host, f'tail -1 {rem}/_output/proposals/runner.log; '
                                                      'pgrep -f run_channels.py | wc -l; pgrep -f yt-dlp | wc -l; '
                                                      'date +%H:%M').strip().rsplit('\n', 3)
        except (subprocess.CalledProcessError, ValueError) as e:
            report.append({'host': host, 'ok': False, 'what': f'not reached ({str(e)[-120:]})'})
            continue
        low = f'{"GPU" if gpu else "CPU"} {mean or 0:.0f}% for {LOW_FOR} min'
        hm = re.match(r'(\d\d):(\d\d)', last)
        nh, nm = (int(x) for x in now.strip().split(':'))
        age = (nh * 60 + nm - int(hm[1]) * 60 - int(hm[2])) % 1440 if hm else None
        if int(runners) == 0:
            what, ok = 'run_channels.py is not running', False
        elif PROBLEM.search(last):
            what, ok = f'{low}: {last.strip()[:160]}', False
        elif IDLE.search(last):
            what, ok = f'idle: {last.strip()[:160]}', True
        elif age is not None and age >= STUCK_MIN:
            what, ok = (f'{low}, no progress for {age} min{" (a download running)" if int(downloads) else ""}: '
                        f'{last.strip()[:120]}'), False
        else:
            what, ok = f'{low}, {"downloading" if int(downloads) else "working"}: {last.strip()[:120]}', True
        report.append({'host': host, 'ok': ok, 'what': what})
    try:
        with open(HEALTH) as f:
            before = {r['host']: r for r in json.load(f)}
    except (OSError, ValueError):
        before = {}
    for r in report:
        if not r['ok'] and before.get(r['host'], {}).get('ok', True):
            rc.log(f"WORKERS: {r['host']}: {r['what']}")
    with open(HEALTH + '.tmp', 'w') as f:
        json.dump(report, f, indent=1)
    os.replace(HEALTH + '.tmp', HEALTH)
    return report


def push_code(host):
    """The code as it is here, to a worker: [files that changed there]. Each batch runs propose_segments.py
    afresh and run_channels.py restarts by itself, so a worker takes it up with its next batch."""
    alias, rem = place(host)
    code = sorted(f for f in os.listdir('.') if f.endswith('.py'))
    return subprocess.run(['rsync', '-c', '--timeout=60', '--out-format=%n', *code, f'{alias}:{rem}/'],
                          check=True, capture_output=True, text=True, timeout=300).stdout.split()


def sync(host):
    alias, rem = place(host)
    pushed = push_code(host)
    if pushed:
        print(f'{host}: updated {", ".join(pushed)}')
    remote_state = json.loads(ssh(host, f'cat {rem}/_output/proposals/channels.json 2>/dev/null || echo "{{}}"'))
    state = rc.load_state()
    for url, c in remote_state.items():
        name = c['name']
        r = subprocess.run(['rsync', '-az', '--exclude', 'plan.json', '--exclude', '*.tmp',
                            f'{alias}:{rem}/_output/proposals/{name}/', os.path.join(rc.ROOT, name) + '/'],
                           capture_output=True)
        # 23/24: a file the worker was writing went away while copied (its batch running): the next sync has it
        if r.returncode not in (0, 23, 24):
            raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
        plan = json.loads(ssh(host, f'cat {rem}/_output/proposals/{name}/plan.json 2>/dev/null || echo "[]"'))
        added = merge_plan(name, plan)
        mine_plan = os.path.join(rc.ROOT, name, 'plan.json')
        local = json.load(open(mine_plan)) if os.path.exists(mine_plan) else []
        # the whole channel rejected in review here: the worker stops it too
        if os.path.exists(os.path.join(rc.ROOT, name, ps.CHANNEL_REJECTED)):
            ssh(host, f'touch {rem}/_output/proposals/{name}/{ps.CHANNEL_REJECTED}')
        # the reviews here say the channel opens with highlights: the worker starts its videos after them
        if local and ps.montage_channel({q['video']: q for q in local}, os.path.join(rc.ROOT, name)):
            ssh(host, f'touch {rem}/_output/proposals/{name}/{ps.MONTAGE_MARK}')
        # videos marked sped up or with ghosts in review here go back to the worker: two reject the channel there
        for key in ps.REVIEW_MARKS:
            marked = sorted(q['video'] for q in local if q.get(key))
            if marked and not set(marked) <= {q['video'] for q in plan if q.get(key)}:
                ssh(host, f"cd {rem} && .venv/bin/python -c \"import json, propose_segments as ps\n"
                          f"path = '_output/proposals/{name}/plan.json'\n"
                          f"with ps.plan_lock(path):\n"
                          f"    plan = json.load(open(path))\n"
                          f"    for q in plan:\n"
                          f"        q[{key!r}] = q.get({key!r}) or q['video'] in {marked!r}\n"
                          f"    json.dump(plan, open(path + '.tmp', 'w'), indent=1)\n"
                          f"    __import__('os').replace(path + '.tmp', path)\"")
                print(f'{host}: {name}: {len(marked)} videos marked {ps.REVIEW_MARKS[key][0]} sent')
        mine = state.get(url, {})
        # the review state lives here: a channel reviewed (or marked on the sheet) here stays so
        keep = (mine.get('status') in ('reviewed', 'marked') and c.get('status') == 'to review'
                or mine.get('status') == 'rejected')  # rejected here (videos marked sped up or with ghosts in review)
        state[url] = {**mine, **c, 'host': host, **({'status': mine['status']} if keep else {})}
        print(f'{host}: {name} ({c.get("status")}): {added} new or updated videos')
    rc.save_state(state)
    # the mapping as reviewed here, swapped in whole so a running batch never reads half a file
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', 'mapping.csv', f'{alias}:{rem}/mapping.csv.tmp'],
                   check=True, capture_output=True)
    ssh(host, f'mv {rem}/mapping.csv.tmp {rem}/mapping.csv')
    with open(rc.TAKEN + '.tmp', 'w') as f:  # what every machine works on: a worker skips a channel listed twice
        json.dump(sorted({x for k, c in rc.load_state().items() for x in (k, c.get('url') or k)}), f)
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', rc.TAKEN + '.tmp', f'{alias}:{rem}/{rc.TAKEN}'],
                   check=True, capture_output=True)
    if os.path.exists(rc.CHANNEL_IDS):  # the sheet's channels by id: a worker skips a channel listed twice
        ssh(host, f'mkdir -p {rem}/{os.path.dirname(rc.CHANNEL_IDS)}')
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', rc.CHANNEL_IDS, f'{alias}:{rem}/{rc.CHANNEL_IDS}'],
                       check=True, capture_output=True)
    if os.path.exists(ps.MERGES):  # towns the reviewer counts in a bigger city nearby: the worker suggests that city
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', ps.MERGES, f'{alias}:{rem}/{ps.MERGES}.tmp'],
                       check=True, capture_output=True)
        ssh(host, f'mv {rem}/{ps.MERGES}.tmp {rem}/{ps.MERGES}')
    if os.path.exists(rc.REFRESH_QUEUE):  # the channels done before whose new uploads come next
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', rc.REFRESH_QUEUE,
                        f'{alias}:{rem}/{rc.REFRESH_QUEUE}.tmp'],
                       check=True, capture_output=True)
        ssh(host, f'mv {rem}/{rc.REFRESH_QUEUE}.tmp {rem}/{rc.REFRESH_QUEUE}')


SYNC_LOCK = '/tmp/crowd-sync.lock'  # one sync at a time: the hourly one and the review hub's button


def main():
    import fcntl
    lock = open(SYNC_LOCK, 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)  # a sync already running: wait for it, then fetch what came since
    for host in WORKERS:
        try:
            sync(host)
        except subprocess.CalledProcessError as e:
            print(f'{host}: not reached ({(e.stderr or "").strip()[-200:]})')
    state, sheet = rc.load_state(), rc.sheet_rows()
    rc.check_reviews(state, sheet)
    rc.write_sheet_todo(state, sheet)
    rc.save_state(state)
    rc.write_to_review(state)


if __name__ == '__main__':
    main()
