"""Bring the batches of the worker machines to this laptop, and the mapping to them.

    python sync_workers.py

Each worker runs run_channels.py on its own sheet rows (WORKERS) in ~/crowd-routine. For every channel a worker
has started, this copies its analyses (signals, contact sheets, metadata) here and adds the videos it proposed
that the plan here does not have yet: decisions made here are never overwritten. The worker's channel progress
goes into channels.json here, which the review hub and the sheet updates read; then the mapping as reviewed here
goes to each worker, so its locality guesses and its skipping of videos already in the mapping are up to date.
"""
import json
import os
import subprocess

import propose_segments as ps
import run_channels as rc

WORKERS = {'lin5080': 'even', 'mini': 'odd'}  # ssh alias: the sheet rows it takes (run_channels.py --rows)
REMOTE = 'crowd-routine'       # the repository on each worker, relative to its home


def ssh(host, command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                          capture_output=True, text=True, check=True).stdout


def merge_plan(name, remote_plan):
    """Add the worker's videos that this plan lacks (or has only as failed downloads). Returns how many."""
    path = os.path.join(rc.ROOT, name, 'plan.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    added = 0
    with ps.plan_lock(path):
        local = json.load(open(path)) if os.path.exists(path) else []
        index = {p['video']: i for i, p in enumerate(local)}
        for p in remote_plan:
            i = index.get(p['video'])
            if i is None:
                local.append(p)
                added += 1
            elif str(local[i].get('exclude', '')).startswith('could not analyse') and 'entries' in p:
                local[i] = p  # failed here, done there
                added += 1
        with open(path + '.tmp', 'w') as f:
            json.dump(local, f, indent=1)
        os.replace(path + '.tmp', path)
    return added


def sync(host):
    # the code as it is here; each batch runs propose_segments.py afresh, run_channels.py restarts by itself
    code = sorted(f for f in os.listdir('.') if f.endswith('.py'))
    pushed = subprocess.run(['rsync', '-c', '--out-format=%n', *code, f'{host}:{REMOTE}/'],
                            check=True, capture_output=True, text=True).stdout.split()
    if pushed:
        print(f'{host}: updated {", ".join(pushed)}')
    remote_state = json.loads(ssh(host, f'cat {REMOTE}/_output/proposals/channels.json 2>/dev/null || echo "{{}}"'))
    state = rc.load_state()
    for url, c in remote_state.items():
        name = c['name']
        subprocess.run(['rsync', '-az', '--exclude', 'plan.json', '--exclude', '*.tmp',
                        f'{host}:{REMOTE}/_output/proposals/{name}/', os.path.join(rc.ROOT, name) + '/'],
                       check=True, capture_output=True)
        plan = json.loads(ssh(host, f'cat {REMOTE}/_output/proposals/{name}/plan.json 2>/dev/null || echo "[]"'))
        added = merge_plan(name, plan)
        mine = state.get(url, {})
        # the review state lives here: a channel reviewed (or marked on the sheet) here stays so
        keep = mine.get('status') in ('reviewed', 'marked') and c.get('status') == 'to review'
        state[url] = {**mine, **c, 'host': host, **({'status': mine['status']} if keep else {})}
        print(f'{host}: {name} ({c.get("status")}): {added} new or updated videos')
    rc.save_state(state)
    # the mapping as reviewed here, swapped in whole so a running batch never reads half a file
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', 'mapping.csv', f'{host}:{REMOTE}/mapping.csv.tmp'],
                   check=True, capture_output=True)
    ssh(host, f'mv {REMOTE}/mapping.csv.tmp {REMOTE}/mapping.csv')


def main():
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
