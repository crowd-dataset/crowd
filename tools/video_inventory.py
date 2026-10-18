"""Inventory of the downloaded videos: every .mp4 in the video folders with its size, length, height and fps (ffprobe),
and what is wrong with it:

    duplicate   the same video in another folder too (the copy in the earlier folder of the list is kept)
    unreadable  ffprobe finds no video stream in it (a broken or partial file)
    short       over 5 s shorter than the end of its last segment (cut off while downloading, or the mapping's
                end time is past the end of the video)
    unused      in no mapping (mapping.csv or a mapping-NAME.csv): removed from the dataset, or never added

    python -m tools.video_inventory --config ~/Documents/GitHub/pedestrians-in-youtube/config

The folders are the "videos" of that config (the downloader's) but the last one, or given as arguments: with
"external_ssd" main.py copies each video into the last folder while it processes it and deletes the copy
afterwards.

Writes _output/video_inventory.csv (one row per file) and _output/video_inventory.json (counts per folder and
problem). Only new or changed files are probed again (size and time against the last inventory), so a nightly run
takes minutes.
Standard library only, and ffprobe. Nightly on lin4080, in the checkout ~/crowd-counts:

    0 3 * * * cd $HOME/crowd-counts && /usr/bin/flock -n /tmp/crowd-counts.lock bash -c 'git pull -q --ff-only;
        nice -n 10 ionice -c3 python3 -m tools.video_inventory
        --config ~/Documents/GitHub/pedestrians-in-youtube/config' >> $HOME/crowd-counts/inventory.log 2>&1
"""
import argparse
import ast
import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

OUT_CSV, OUT_JSON = '_output/video_inventory.csv', '_output/video_inventory.json'
VIDEO = re.compile(r'[A-Za-z0-9_-]{11}\.mp4')  # a YouTube id: not main.py's working clips (yt_ID_..._mod_....mp4)
FIELDS = ['video', 'folder', 'mb', 'mtime', 'duration', 'height', 'fps', 'problem']


def probe(path):
    """(duration s, height, fps) of a video file, or None when ffprobe finds no video stream."""
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                              'stream=height,avg_frame_rate:format=duration', '-of', 'json', path],
                             capture_output=True, text=True, timeout=60).stdout
        info = json.loads(out)
        s = info['streams'][0]
        num, _, den = s.get('avg_frame_rate', '0/1').partition('/')
        return float(info['format']['duration']), int(s['height']), round(float(num) / float(den or 1), 2)
    except Exception:
        return None


def segment_ends(mapping_files):
    """{video: end of its last segment in s} over these mappings. A video of the first (mapping.csv) takes its times
    from there only: a mapping-NAME.csv is an older copy of the mapping plus its new videos, and merging it keeps the
    mapping's times for the videos the mapping has (review_hub.merge_mapping_files)."""
    ends, first = {}, None
    for path in mapping_files:
        file_ends = {}
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                vids = [v.strip() for v in row['videos'].strip('"[] ').split(',') if v.strip()]
                try:
                    end_times = ast.literal_eval(row['end_time'] or '[]')
                except (ValueError, SyntaxError):
                    end_times = []
                for i, v in enumerate(vids):
                    seg = end_times[i] if i < len(end_times) and isinstance(end_times[i], list) else []
                    file_ends[v] = max([file_ends.get(v, 0), *[t for t in seg if isinstance(t, (int, float))]])
        if first is None:
            first = set(file_ends)
        ends.update({v: e for v, e in file_ends.items() if v not in first or path == mapping_files[0]})
    return ends


def problems(files, ends):
    """Sets row['problem'] of each row of files (in folder order)."""
    seen = set()
    for r in files:
        p = []
        if r['video'] in seen:
            p.append('duplicate')
        seen.add(r['video'])
        if r['duration'] == '':
            p.append('unreadable')
        elif r['video'] in ends and float(r['duration']) < ends[r['video']] - 5:
            p.append('short')
        if r['video'] not in ends:
            p.append('unused')
        r['problem'] = ' '.join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('folders', nargs='*', help='video folders (default: "videos" of --config but the last)')
    ap.add_argument('--config', default='config', help='a config with "videos" (default: ./config)')
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    folders = args.folders or json.load(open(os.path.expanduser(args.config)))['videos'][:-1]
    old = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            old = {(r['folder'], r['video']): r for r in csv.DictReader(f)}
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f'no such folder (yet): {folder}', file=sys.stderr)
            continue
        for e in sorted(os.scandir(folder), key=lambda e: e.name):
            if VIDEO.fullmatch(e.name) and e.is_file():
                st = e.stat()
                files.append(dict(video=e.name[:-4], folder=folder, mb=round(st.st_size / 1e6),
                                  mtime=int(st.st_mtime)))

    def fill(r):
        o = old.get((r['folder'], r['video']))
        if o and int(o['mb']) == r['mb'] and int(o['mtime']) == r['mtime']:
            r.update(duration=o['duration'], height=o['height'], fps=o['fps'])
            return False
        got = probe(os.path.join(r['folder'], r['video'] + '.mp4'))
        r.update(zip(('duration', 'height', 'fps'), got or ('', '', '')))
        return True

    with ThreadPoolExecutor(args.threads) as ex:
        probed = sum(ex.map(fill, files))
    ends = segment_ends(['mapping.csv'] + sorted(glob.glob('mapping-*.csv')))
    problems(files, ends)
    os.makedirs('_output', exist_ok=True)
    with open(OUT_CSV + '.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(files)
    os.replace(OUT_CSV + '.tmp', OUT_CSV)
    have = {r['video'] for r in files}
    count = collections.Counter(p for r in files for p in r['problem'].split())
    gb = collections.Counter()
    for r in files:
        for p in r['problem'].split():
            gb[p] += r['mb'] / 1000
    summary = dict(
        t=time.time(), files=len(files), videos=len(have), tb=round(sum(r['mb'] for r in files) / 1e6, 2),
        probed=probed, missing=sum(v not in have for v in ends),
        problems={p: dict(files=n, gb=round(gb[p])) for p, n in count.items()},
        folders=[dict(folder=d, files=sum(r['folder'] == d for r in files),
                      tb=round(sum(r['mb'] for r in files if r['folder'] == d) / 1e6, 2)) for d in folders])
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'folders'}))


def _check_segment_ends(tmp):
    head = 'videos,end_time\n'
    open(f'{tmp}/m.csv', 'w').write(head + '"[a,b]","[[50],[90]]"\n')
    open(f'{tmp}/m-x.csv', 'w').write(head + '"[a,c]","[[80],[30]]"\n')  # a: an old end time; c: new
    assert segment_ends([f'{tmp}/m.csv', f'{tmp}/m-x.csv']) == {'a': 50, 'b': 90, 'c': 30}


def _check():
    rows = [dict(video='a', duration='100.0'), dict(video='b', duration=''), dict(video='c', duration='50.0'),
            dict(video='a', duration='100.0'), dict(video='z', duration='9.0')]
    problems(rows, {'a': 90, 'b': 10, 'c': 80})
    assert [r['problem'] for r in rows] == ['', 'unreadable', 'short', 'duplicate', 'unused'], rows


if __name__ == '__main__':
    _check()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _check_segment_ends(tmp)
    main()
