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

# ssh alias: the sheet rows and refresh-queue share it takes (run_channels.py --rows, --queue-rows); the laptop
# takes rows 2/3 and queue 2/4
# mini ('sheet rows 1/2, no queue') is out until its new account is set up (2026-10-11)
WORKERS = {'lin5080': 'queue phase 5 (highway videos to redo), then 2 (new uploads of processed channels)',
           'wsl5090': 'sheet rows 0/2, then queue phase 4 (UCxbv63aE7sg8_HzYz5KFPeA highway videos), 1 and 3'}
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
            elif ((str(local[i].get('exclude', '')).startswith('could not analyse') or 'entries' not in local[i])
                  and 'entries' in p):
                local[i] = p  # failed or left out here, proposed there (a rule made less strict since)
                added += 1
        with open(path + '.tmp', 'w') as f:
            json.dump(local, f, indent=1)
        os.replace(path + '.tmp', path)
    return added


def push_code(host):
    """The code as it is here, to a worker: [files that changed there]. Each batch runs propose_segments.py
    afresh and run_channels.py restarts by itself, so a worker takes it up with its next batch."""
    code = sorted(f for f in os.listdir('.') if f.endswith('.py'))
    return subprocess.run(['rsync', '-c', '--timeout=60', '--out-format=%n', *code, f'{host}:{REMOTE}/'],
                          check=True, capture_output=True, text=True, timeout=300).stdout.split()


def sync(host):
    pushed = push_code(host)
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
        mine_plan = os.path.join(rc.ROOT, name, 'plan.json')
        local = json.load(open(mine_plan)) if os.path.exists(mine_plan) else []
        # the whole channel rejected in review here: the worker stops it too
        if os.path.exists(os.path.join(rc.ROOT, name, ps.CHANNEL_REJECTED)):
            ssh(host, f'touch {REMOTE}/_output/proposals/{name}/{ps.CHANNEL_REJECTED}')
        # the reviews here say the channel opens with highlights: the worker starts its videos after them
        if local and ps.montage_channel({q['video']: q for q in local}, os.path.join(rc.ROOT, name)):
            ssh(host, f'touch {REMOTE}/_output/proposals/{name}/{ps.MONTAGE_MARK}')
        # videos marked sped up or with ghosts in review here go back to the worker: two reject the channel there
        for key in ps.REVIEW_MARKS:
            marked = sorted(q['video'] for q in local if q.get(key))
            if marked and not set(marked) <= {q['video'] for q in plan if q.get(key)}:
                ssh(host, f"cd {REMOTE} && .venv/bin/python -c \"import json, propose_segments as ps\n"
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
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', 'mapping.csv', f'{host}:{REMOTE}/mapping.csv.tmp'],
                   check=True, capture_output=True)
    ssh(host, f'mv {REMOTE}/mapping.csv.tmp {REMOTE}/mapping.csv')
    with open(rc.TAKEN + '.tmp', 'w') as f:  # what every machine works on: a worker skips a channel listed twice
        json.dump(sorted({x for k, c in rc.load_state().items() for x in (k, c.get('url') or k)}), f)
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', rc.TAKEN + '.tmp', f'{host}:{REMOTE}/{rc.TAKEN}'],
                   check=True, capture_output=True)
    if os.path.exists(rc.CHANNEL_IDS):  # the sheet's channels by id: a worker skips a channel listed twice
        ssh(host, f'mkdir -p {REMOTE}/{os.path.dirname(rc.CHANNEL_IDS)}')
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', rc.CHANNEL_IDS, f'{host}:{REMOTE}/{rc.CHANNEL_IDS}'],
                       check=True, capture_output=True)
    if os.path.exists(rc.REFRESH_QUEUE):  # the channels done before whose new uploads come next
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', rc.REFRESH_QUEUE,
                        f'{host}:{REMOTE}/{rc.REFRESH_QUEUE}.tmp'],
                       check=True, capture_output=True)
        ssh(host, f'mv {REMOTE}/{rc.REFRESH_QUEUE}.tmp {REMOTE}/{rc.REFRESH_QUEUE}')


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
