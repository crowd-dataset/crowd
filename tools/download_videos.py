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
"download_reserve_gb" (default 100), "download_ftp" (default true; false where the file server shares the same drives),
"download_per_hour" (default 30: more got the login flagged). yt-dlp is upgraded once a day (a new version restarts
the downloader).
What it is doing goes to _output/download_status.json (with the videos still to download) and every download to
_output/downloaded_videos.csv; the pipeline status page shows both (dashboards/).
"""
import argparse
import csv
import importlib
import importlib.metadata
import io
import json
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
BLOCKED_AFTER, BLOCKED_S = 5, 1800  # this many blocked downloads in a row: wait this long,
BLOCKED_MAX_S = 14400               # twice as long each time it happens again, up to this, until a success
# YouTube refusing this machine (bot check, forbidden, too many requests), unlike a video that is gone or private:
# only these count towards a pause (five deleted videos in a row paused it for nothing)
BLOCKED = re.compile(r'Sign in to confirm|HTTP Error 403|HTTP Error 429|Too Many Requests')
NEED_GB = 5               # a drive must have this much above its reserve for the next video
STATUS = os.path.join('_output', 'download_status.json')  # what it is doing, for dashboards/pipeline_status.py
RECENT = 50               # downloads and failures kept in STATUS
DONE = os.path.join('_output', 'downloaded_videos.csv')  # every download: time, video, title, MB, source, drive
FAILED = os.path.join('_output', 'download_failed.json')  # when each failed video last failed, kept over restarts


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


def usage(folder):
    """shutil.disk_usage of the drive a folder is on (the folder need not exist yet)."""
    while not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    return shutil.disk_usage(folder)


def target(drives, reserve_gb):
    """The first drive with room for the next video above its reserve, or None."""
    return next((d for d in drives if mounted(d) and usage(d).free > (reserve_gb + NEED_GB) * 1e9), None)


def report(status, drives, **changes):
    """Update the status and write it to STATUS (whole, so a reader never sees half a file)."""
    status.update(changes, updated=time.time(),
                  drives=[dict(path=d, mounted=mounted(d), free_gb=round(usage(d).free / 1e9),
                               total_gb=round(usage(d).total / 1e9)) for d in drives])
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS + '.tmp', 'w') as f:
        json.dump(status, f)
    os.replace(STATUS + '.tmp', STATUS)


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


def upgrade_ytdlp(h):
    """yt-dlp at its newest, checked once a day (YouTube's changes break older versions): a new version restarts this
    process, which has the old one loaded."""
    before = importlib.metadata.version('yt-dlp')
    h.upgrade_package_if_needed('yt-dlp')
    importlib.invalidate_caches()
    if (after := importlib.metadata.version('yt-dlp')) != before:
        log(f'yt-dlp {before} -> {after}: restarting')
        os.execv(sys.executable, [sys.executable, '-m', 'tools.download_videos', *sys.argv[1:]])


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
    per_hour, done_at = setting('download_per_hour', 30), []
    log(f'drives in order: {drives} (each until {reserve} GB is left); file server first: {use_ftp}')
    h = None
    checked, todo, streak, wait = 0.0, [], 0, BLOCKED_S
    try:
        with open(FAILED) as f:
            failed = {v: t for v, t in json.load(f).items() if time.time() - t < RETRY_S}
    except (OSError, ValueError):
        failed = {}
    status = dict(started=time.time(), state='starting', queue=0, downloaded=0, failed=0, current=None, drive=None,
                  until=None, reserve_gb=reserve, recent=[], failures=[], todo=[])
    try:  # the downloads and failures logged before a restart stay
        with open(STATUS) as f:
            before = json.load(f)
        status.update(recent=before.get("recent", [])[:RECENT], failures=before.get("failures", [])[:RECENT])
    except (OSError, ValueError):
        pass
    while True:
        if time.time() - checked >= CHECK_S or not todo:
            checked, have = time.time(), present(list(folders) + list(drives))
            todo = [v for v in mapping_videos() if v not in have and time.time() - failed.get(v, 0) > RETRY_S]
            log(f'{len(todo)} videos to download')
            if not args.dry_run:
                status['todo'] = todo  # the same list: the status file always has what is left
                report(status, drives, queue=len(todo), state='downloading' if todo else 'idle', until=None)
            if args.dry_run:
                for v in todo[:20]:
                    log(f'would download {v} to {target(drives, reserve)}')
                return
        if not todo:
            if args.once:
                return
            report(status, drives, state='idle', current=None, until=time.time() + CHECK_S)
            time.sleep(CHECK_S)
            continue
        vid = todo.pop(0)
        if vid in present(list(folders) + list(drives)):  # main.py or someone else fetched it meanwhile
            continue
        folder = target(drives, reserve)
        if folder is None:
            log(f'no drive has more than {reserve + NEED_GB} GB free: waiting an hour')
            report(status, drives, state='no room on any drive', current=None, until=time.time() + CHECK_S)
            todo.insert(0, vid)
            time.sleep(CHECK_S)
            continue
        done_at = [t for t in done_at if time.time() - t < 3600]
        if len(done_at) >= per_hour:
            log(f'{per_hour} downloads in the last hour: waiting {(done_at[0] + 3600 - time.time()) / 60:.0f} min')
            report(status, drives, state=f'{per_hour} an hour at most', current=None, until=done_at[0] + 3600)
            time.sleep(max(0, done_at[0] + 3600 - time.time()))
        h = h or Youtube_Helper()
        if not args.dry_run:
            upgrade_ytdlp(h)
        report(status, drives, state='downloading', current=vid, drive=folder, queue=len(todo) + 1, until=None)
        h.video_title, h.download_error = None, None
        source = fetch(h, vid, folder, use_ftp)
        if source:
            streak, wait = 0, BLOCKED_S
            done_at.append(time.time())
            log(f'{vid}: from {source} to {folder} ({usage(folder).free / 1e9:.0f} GB free there)')
            size = os.path.getsize(os.path.join(folder, f'{vid}.mp4'))
            status['recent'] = [dict(t=time.time(), vid=vid, title=h.video_title, source=source, drive=folder,
                                     mb=round(size / 1e6))] + status['recent'][:RECENT - 1]
            new = not os.path.exists(DONE)
            with open(DONE, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['time', 'video', 'title', 'mb', 'source', 'drive'])
                w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), vid, h.video_title or '', round(size / 1e6), source,
                            folder])
            report(status, drives, downloaded=status['downloaded'] + 1, current=None, queue=len(todo))
        else:
            failed[vid] = time.time()
            with open(FAILED + '.tmp', 'w') as f:
                json.dump(failed, f)
            os.replace(FAILED + '.tmp', FAILED)
            if BLOCKED.search(h.download_error or ''):
                streak += 1
                log(f'{vid}: failed, blocked ({streak} in a row)')
            else:
                log(f'{vid}: failed ({(h.download_error or "no reason given").splitlines()[0][:120]})')
            status['failures'] = [dict(t=time.time(), vid=vid)] + status['failures'][:RECENT - 1]
            report(status, drives, failed=status['failed'] + 1, current=None, queue=len(todo))
            if streak >= BLOCKED_AFTER:
                log(f'{streak} downloads blocked in a row: waiting {wait // 60} min')
                report(status, drives, state=f'paused after {streak} blocked in a row', until=time.time() + wait)
                time.sleep(wait)
                streak, wait = 0, min(2 * wait, BLOCKED_MAX_S)
        if args.once and not todo:
            return
        time.sleep(PAUSE_S)


if __name__ == '__main__':
    sys.exit(main())
