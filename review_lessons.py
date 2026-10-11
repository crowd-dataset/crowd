"""What the reviews say about the proposals: where the reviewer keeps what propose() cuts, cuts what it keeps,
or moves a video's start or end, per rule and per channel.

    python review_lessons.py            # writes _output/proposals/review_lessons.md

Each reviewed video's saved signals are proposed again with the current code and compared with the segments
approved for it, so a rule changed since shows its effect at once. The note of a proposed cut names the rule that
made it ("stationary 120-300s", "camera shaking 163-223s"); a stretch the reviewer approved inside such a cut
overrules that rule, a stretch approved nowhere that the proposal kept is the reviewer's own cut.
"""
import collections
import glob
import json
import os
import re

import numpy as np

import propose_segments as ps

OUT = '_output/proposals/review_lessons.md'
RULE = re.compile(r'([a-z][a-z ]+?) (\d+)-(\d+)s')  # "camera shaking 163-223s", "stationary 0-53s (edge)"


def covered(t, segs):
    return any(s <= t < e for s, e in segs)


def main():
    overruled, kept = collections.Counter(), collections.Counter()  # rule -> seconds approved inside / cut stayed
    own_cuts = collections.Counter()  # channel -> seconds the reviewer cut that the proposal kept
    starts, ends = collections.defaultdict(list), collections.defaultdict(list)
    whole = collections.Counter()  # channel -> videos rejected whole though proposed
    videos = collections.Counter()
    for path in sorted(glob.glob('_output/proposals/*/plan.json')):
        d = os.path.dirname(path)
        ch = os.path.basename(d)
        for p in json.load(open(path)):
            segs = [s for e in p.get('entries', []) for s in e['segments']]
            if not segs or not all(s.get('decision') or s.get('applied') for s in segs):
                continue  # not reviewed (yet)
            sig_path = os.path.join(d, p['video'] + '_signals.npz')
            if not os.path.exists(sig_path):
                continue
            sig = dict(np.load(sig_path))
            # as the fetcher proposed it: a drive within one town gets no highway or rural cuts (locate_now)
            towns = {tuple(e['locality']) for e in p['entries'] if e.get('locality')}
            proposed, notes = ps.propose(sig, ps.ONE_PLACE if len(towns) == 1 and not p.get('route') else ps.PARAMS)
            mine = [(s['start'], s['end']) for s in proposed]
            final = [(s['start'], s['end']) for s in segs if s.get('decision') == 'approve' or s.get('applied')]
            videos[ch] += 1
            if not final:
                whole[ch] += 1
                continue
            if mine:
                starts[ch].append(min(s for s, _ in final) - min(s for s, _ in mine))
                ends[ch].append(max(e for _, e in mine) - max(e for _, e in final))
            # second by second inside the proposal's cuts: approved there overrules the rule that cut it
            n = int(len(sig['motion']) / ps.FPS)
            for m in RULE.finditer('; '.join(notes)):
                rule, a, b = m.group(1).strip(), int(m.group(2)), min(int(m.group(3)), n)
                inside = sum(covered(t, final) and not covered(t, mine) for t in range(a, b))
                overruled[rule] += inside
                kept[rule] += sum(not covered(t, final) and not covered(t, mine) for t in range(a, b))
            own_cuts[ch] += sum(covered(t, mine) and not covered(t, final) for t in range(n))
    lines = ['# What the reviews say', '', f'{sum(videos.values())} reviewed videos with signals, proposed again '
             'with the current code.', '', '## Rules: seconds the reviewer approved inside their cuts', '',
             '| rule | approved anyway (s) | cut and left cut (s) | overruled share |', '|---|---|---|---|']
    for rule in sorted(overruled, key=lambda r: -overruled[r]):
        total = overruled[rule] + kept[rule]
        if total:
            lines.append(f'| {rule} | {overruled[rule]} | {kept[rule]} | {overruled[rule] / total:.0%} |')
    lines += ['', '## Per channel: start and end the reviewer chose vs the proposal (median s; + = later start, '
              'earlier end)', '', '| channel | videos | rejected whole | start | end | own cuts (s per video) |',
              '|---|---|---|---|---|---|']
    for ch in sorted(videos, key=lambda c: -videos[c]):
        st = f'{np.median(starts[ch]):+.0f}' if starts[ch] else ''
        en = f'{np.median(ends[ch]):+.0f}' if ends[ch] else ''
        lines.append(f'| {ch} | {videos[ch]} | {whole[ch]} | {st} | {en} | {own_cuts[ch] / videos[ch]:.0f} |')
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
