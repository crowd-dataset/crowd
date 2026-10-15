"""Keep the mapping's videos downloaded: each video that no folder of "videos" has yet is fetched as main.py fetches
it (the file server first, then YouTube at 720p) onto the folders of "download_drives", filled one at a time: a drive
is used until less than "download_reserve_gb" is free on it, then the next.

    python -m tools.download_videos            # keeps going; the mapping on GitHub's main is checked every hour
    python -m tools.download_videos --once     # one pass over what is missing now
    python -m tools.download_videos --dry-run  # say what would be downloaded where, download nothing

Run from the repository's top folder. The videos come from mapping.csv and the contributors' mapping-NAME.csv
files (so their videos are fetched before they are merged), read from origin/main after a git fetch: new videos
arrive within CHECK_S without a pull, and the checkout others run from is left as it is (the files here when git
fails). Optional settings in config: "download_drives" (default: the folders of "videos" but the last),
"download_reserve_gb" (default 100), "download_ftp" (default true; false where the file server shares the same drives).
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import common
from helper_script import Youtube_Helper

CHECK_S = 3600            # how often the mapping on GitHub is read again
RETRY_S = 86400           # a video that failed is tried again after this long
PAUSE_S = 30              # between two downloads (shorter got YouTube's bot check more often)
BLOCKED_AFTER, BLOCKED_S = 5, 1800  # this many failures in a row (YouTube asking to sign in): wait this long,
BLOCKED_MAX_S = 14400               # twice as long each time it happens again, up to this, until a success
NEED_GB = 5               # a drive must have this much above its reserve for the next video


def setting(name, default):
    try:
        return common.get_configs(name)
    except KeyError:
        return default


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


MAPPING_FILE = re.compile(r'mapping(-[^/]+)?\.csv')  # mapping.csv and the contributors' mapping-NAME.csv


def mapping_files(names):
    """The mapping files among a folder's file names: mapping.csv first, then the mapping-NAME.csv files."""
    return sorted((n for n in names if MAPPING_FILE.fullmatch(n)), key=lambda n: (n != 'mapping.csv', n))


def mapping_videos():
    """The video ids of mapping.csv and the mapping-NAME.csv files (videos added there before they are merged), in
    order: GitHub's main after a fetch, else the files here."""
    try:
        subprocess.run(['git', 'fetch', '-q', 'origin', 'main'], check=True, capture_output=True, timeout=300)
        names = subprocess.run(['git', 'ls-tree', '--name-only', 'origin/main'], check=True, capture_output=True,
                               text=True).stdout.split()
        texts = [subprocess.run(['git', 'show', f'origin/main:{n}'], check=True, capture_output=True,
                                text=True).stdout for n in mapping_files(names)]
    except Exception as e:
        log(f'mapping from GitHub failed ({e}); using the files here')
        texts = []
        for n in mapping_files(os.listdir('.')):
            with open(n, encoding='utf-8') as f:
                texts.append(f.read())
    return list(dict.fromkeys(v.strip() for text in texts for row in csv.DictReader(io.StringIO(text))
                              for v in (row.get('videos') or '').strip().strip('"[]').split(',') if v.strip()))


def present(folders):
    """The video ids that are in any of these folders."""
    have = set()
    for d in folders:
        if os.path.isdir(d):
            have.update(f[:-4] for f in os.listdir(d) if f.endswith('.mp4'))
    return have


def mounted(folder):
    """False for a folder under /media whose drive is not mounted (it would fill the system disk instead)."""
    parts = os.path.abspath(folder).split(os.sep)
    return not (len(parts) > 3 and parts[1] == 'media') or os.path.ismount(os.sep.join(parts[:4]))


def target(drives, reserve_gb):
    """The first drive with room for the next video above its reserve, or None."""
    for d in drives:
        root = d
        while not os.path.isdir(root):
            root = os.path.dirname(root)
        if mounted(d) and shutil.disk_usage(root).free > (reserve_gb + NEED_GB) * 1e9:
            return d
    return None


def fetch(h, vid, folder, use_ftp):
    """Download one video into folder as main.py would; 'ftp', 'youtube' or None."""
    os.makedirs(folder, exist_ok=True)
    if use_ftp:
        tmp = os.path.join(folder, '.ftp_tmp', f'{vid}_{uuid.uuid4().hex}')
        try:
            got = h.download_videos_from_ftp(vid, base_url=common.get_configs('ftp_server'), out_dir=tmp,
                                             username=secret('ftp_username'), password=secret('ftp_password'),
                                             token=secret('ftp_token'), debug=False)
            if got:
                os.replace(got[0], os.path.join(folder, f'{vid}.mp4'))
                return 'ftp'
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 'youtube' if h.download_video_with_resolution(vid, output_path=folder) else None


def secret(name):
    try:
        return common.get_secrets(name) or None
    except (KeyError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--once', action='store_true', help='one pass over what is missing now')
    ap.add_argument('--dry-run', action='store_true', help='say what would be downloaded where')
    args = ap.parse_args()
    folders = common.get_configs('videos')
    drives = setting('download_drives', folders[:-1])
    reserve, use_ftp = setting('download_reserve_gb', 100), setting('download_ftp', True)
    log(f'drives in order: {drives} (each until {reserve} GB is left); file server first: {use_ftp}')
    h = None
    failed, checked, todo, streak, wait = {}, 0.0, [], 0, BLOCKED_S
    while True:
        if time.time() - checked >= CHECK_S or not todo:
            checked, have = time.time(), present(list(folders) + list(drives))
            todo = [v for v in mapping_videos() if v not in have and time.time() - failed.get(v, 0) > RETRY_S]
            log(f'{len(todo)} videos to download')
            if args.dry_run:
                for v in todo[:20]:
                    log(f'would download {v} to {target(drives, reserve)}')
                return
        if not todo:
            if args.once:
                return
            time.sleep(CHECK_S)
            continue
        vid = todo.pop(0)
        if vid in present(list(folders) + list(drives)):  # main.py or someone else fetched it meanwhile
            continue
        folder = target(drives, reserve)
        if folder is None:
            log(f'no drive has more than {reserve + NEED_GB} GB free: waiting an hour')
            todo.insert(0, vid)
            time.sleep(CHECK_S)
            continue
        h = h or Youtube_Helper()
        source = fetch(h, vid, folder, use_ftp)
        if source:
            streak, wait = 0, BLOCKED_S
            log(f'{vid}: from {source} to {folder} ({shutil.disk_usage(folder).free / 1e9:.0f} GB free there)')
        else:
            failed[vid], streak = time.time(), streak + 1
            log(f'{vid}: failed ({streak} in a row)')
            if streak >= BLOCKED_AFTER:
                log(f'{streak} failures in a row (YouTube asking to sign in?): waiting {wait // 60} min')
                time.sleep(wait)
                streak, wait = 0, min(2 * wait, BLOCKED_MAX_S)
        if args.once and not todo:
            return
        time.sleep(PAUSE_S)


if __name__ == '__main__':
    sys.exit(main())
