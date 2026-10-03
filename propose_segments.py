"""Propose CROWD segments for a YouTube video from its pixels, for a curator to review.

Signals come from a low resolution copy decoded at a few frames per second:
camera motion (to trim stationary stretches), hard cuts and blank frames
(edited discontinuities), and sky brightness (day/night, split at transitions).
Vehicle type comes from the channel's history in the mapping file.
Nothing here writes to the mapping file.
"""
import argparse
import contextlib
import fcntl
import json
import os
import random
import re
import subprocess
import tempfile
import time
import webbrowser
from collections import Counter
from threading import Timer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import requests

import add_video
import common

FPS = 2                 # analysis frames per second
W, H = 160, 90          # analysis frame size

PARAMS = {
    'moving_flow': 0.25,      # median flow (px per analysis frame) above which the camera is moving
    'smooth_s': 5,            # rolling median window for motion
    'edge_stop_s': 15,        # stationary stretch at the start/end longer than this is trimmed
    'mid_stop_s': 180,        # stationary stretch inside a video longer than this is cut out
    'cut_ncc': 0.3,           # a frame this uncorrelated with the previous one is a hard cut...
    'cut_neighbour_ncc': 0.6,  # ...when the frames either side of it are this coherent
    'bridge_dip': 0.5,        # a cut whose sky darkens below this share of its surroundings is a bridge
    'cut_pan_ncc': None,      # a cut that matches this well once shifted sideways is a corner (not yet calibrated)
    'blank_std': 6.0,         # frame this flat is blank or a title card
    'night_sky': 95.0,        # sky brightness below this is night (calibrated on labelled segments)
    'night_window_s': 60,     # night/day must hold this long to count as a transition
    'min_segment_s': 30,      # drop proposed segments shorter than this
    'driver_window_s': 60,    # a big face in more than driver_share of the frames over this window means the
    'driver_share': 0.25,     # camera faces the driver...
    'driver_min_s': 10,       # ...and such a stretch this long is cut out
}


# pytubefix downloads now fail YouTube's PoToken check; a current yt-dlp (with node installed) does not
YT_DLP = os.environ.get('YT_DLP', 'yt-dlp')


class BotCheck(RuntimeError):
    """YouTube wants a signed-in session; further requests will fail until it lifts."""


def _yt_dlp(*args):
    # opt-in for when YouTube asks to sign in: YT_DLP_COOKIES=chrome (or firefox, ...) uses that browser's login,
    # or a path to a cookies.txt exported from a logged-in browser
    jar = os.environ.get('YT_DLP_COOKIES')
    cookies = (['--cookies', jar] if os.path.isfile(jar) else ['--cookies-from-browser', jar]) if jar else []
    r = subprocess.run([YT_DLP, '--no-warnings', *cookies, *args], capture_output=True, text=True)
    if r.returncode:
        msg = (r.stderr.strip().splitlines() or [f'yt-dlp exited with {r.returncode}'])[-1]
        raise (BotCheck if 'not a bot' in r.stderr else RuntimeError)(msg)
    return r.stdout


def _url(video_id):
    return f'https://www.youtube.com/watch?v={video_id}'


def fetch_metadata(video_id):
    m = json.loads(_yt_dlp('-J', _url(video_id)))
    d = m.get('upload_date')  # YYYYMMDD
    return {
        'id': video_id,
        'title': m.get('title'),
        'duration': m.get('duration'),
        'upload_date': f'{d[6:8]}{d[4:6]}{d[:4]}' if d else None,
        'channel_id': m.get('channel_id'),
        'live_status': m.get('live_status'),
        'width': m.get('width'),
        'height': m.get('height'),
        'description': m.get('description') or '',
        'chapters': [(c.get('title'), c.get('start_time')) for c in (m.get('chapters') or [])],
    }


def download_low_res(video_id, out_dir):
    template = os.path.join(out_dir, f'{video_id}_low.%(ext)s')
    _yt_dlp('-q', '-f', 'bv*[height<=240][ext=mp4]/bv*[height<=360]/wv*', '-o', template, _url(video_id))
    for name in os.listdir(out_dir):
        if name.startswith(f'{video_id}_low.'):
            return os.path.join(out_dir, name)
    raise RuntimeError(f'download produced no file for {video_id}')


def frames(path):
    """Yield grey analysis frames at FPS."""
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', f'fps={FPS},scale={W}:{H},format=gray',
           '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W)
    finally:
        proc.stdout.close()
        proc.wait()


def _normalised_thumb(f):
    t = cv2.resize(f, (40, 22), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def _best_shifted_ncc(a, b, max_shift=8):
    """Best correlation of two thumbnails over horizontal shifts: a turn shifts the view, a cut replaces it."""
    best = -1.0
    for s in range(-max_shift, max_shift + 1, 2):
        x, y = (a[:, s:], b[:, :b.shape[1] - s]) if s >= 0 else (a[:, :s], b[:, -s:])
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)
        best = max(best, float((x * y).mean()))
    return best


FACES = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
PROFILES = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_profileface.xml'))
DRIVER_SHEET_FRAMES = 4  # a big face in this many of a contact sheet's 12 frames: the camera faces the driver


def _big_face(f):
    """Share of the frame covered by the largest face (frontal, or in profile either way) at least a fifth of
    the frame high, else 0. A forward dashcam only sees small faces of pedestrians; a camera turned to the
    driver sees a big one, often in profile while he watches the road."""
    side = int(H * 0.2)
    found = list(FACES.detectMultiScale(f, scaleFactor=1.15, minNeighbors=5, minSize=(side, side)))
    if not found:
        for img in (f, cv2.flip(f, 1)):  # the profile cascade finds faces turned one way; mirror for the other
            found += list(PROFILES.detectMultiScale(img, scaleFactor=1.15, minNeighbors=5, minSize=(side, side)))
    return max((w * h for _, _, w, h in found), default=0) / (W * H)


def driver_frames(frames):
    """How many of the frames show a big face."""
    return sum(_big_face(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)) > 0 for f in frames)


def _longest_run(mask):
    """(length, start, end) of the longest run of True values."""
    best, start = (0, 0, 0), None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i)
            start = None
    return best


def inset_rectangle(frames):
    """(x0, y0, x1, y1) in a 320x180 frame of a picture-in-picture inset (another camera shown in a corner),
    else None. Over frames spread through a video, scenery edges average away; an inset's inner borders stay
    put and meet in an L, enclosing a corner rectangle whose content keeps changing (unlike a static box)."""
    w, h = 320, 180
    f = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) for x in frames])
    gx = np.abs(np.diff(f, axis=2)).mean(0)  # vertical edges
    gy = np.abs(np.diff(f, axis=1)).mean(0)  # horizontal edges
    gx[:28, :75] = 0  # the time stamps printed on frame sheets
    gy[:28, :75] = 0
    vertical = [(x, *_longest_run(gx[:, x] > 20)) for x in range(int(.05 * w), int(.95 * w))]
    horizontal = [(y, *_longest_run(gy[y, :] > 20)) for y in range(int(.05 * h), int(.95 * h))]
    for x, lv, y0, y1 in vertical:
        if not .2 * h <= lv <= .7 * h:
            continue
        for y, lh, x0, x1 in horizontal:
            if not .2 * w <= lh <= .7 * w:
                continue
            top, bottom = abs(y - y1) <= 6 and y0 <= .2 * h, abs(y - y0) <= 6 and y1 >= .8 * h
            left, right = abs(x - x1) <= 6 and x0 <= .2 * w, abs(x - x0) <= 6 and x1 >= .8 * w
            if (top or bottom) and (left or right):
                rx0, rx1 = (0, x) if left else (x, w)
                ry0, ry1 = (0, y) if top else (y, h)
                if f[:, ry0:ry1, rx0:rx1].std(axis=0).mean() > 12:
                    return rx0, ry0, rx1, ry1
    return None


def sheet_frames(path):
    """The grey frames of a contact sheet written by contact_sheet (4 columns of 320x180)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    return [img[r:r + 180, c:c + 320] for r in range(0, img.shape[0], 180) for c in range(0, img.shape[1], 320)]


def thumbnail_face(video_id):
    """True when the video's YouTube thumbnail shows a big face. Thumbnails are often made-up artwork,
    so this only flags a video for checking; it never cuts anything."""
    try:
        data = requests.get(f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg', timeout=15).content
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    except requests.RequestException:
        return False
    if img is None:
        return False
    img = img[45:315] if img.shape[:2] == (360, 480) else img  # hqdefault pads 16:9 pictures with black bars
    return _big_face(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)) > 0


def signals(path):
    """Per analysis frame: camera motion, similarity to previous frame (plain and allowing for a turn),
    flatness, sky brightness, how much of the frame a big face covers, and the picture's overall sideways and
    vertical shift (a forward camera's flow spreads out from the middle, so both stay near 0; a sideways camera
    slides one way, a shaking one jumps up and down)."""
    band = slice(int(H * 0.25), int(H * 0.70))   # road and buildings: no sky, no bonnet
    motion, ncc, pan, std, sky, face, shift_x, shift_y, spread = [], [], [], [], [], [], [], [], []
    third = W // 3
    prev = prev_thumb = None
    for f in frames(path):
        face.append(_big_face(f))
        thumb = _normalised_thumb(f)
        if prev is None:
            motion.append(0.0)
            ncc.append(1.0)
            pan.append(1.0)
            shift_x.append(0.0)
            shift_y.append(0.0)
            spread.append(0.0)
        else:
            pan.append(_best_shifted_ncc(prev_thumb, thumb))
            flow = cv2.calcOpticalFlowFarneback(prev, f, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mag = np.hypot(flow[band, :, 0], flow[band, :, 1])
            motion.append(float(np.median(mag)))  # median ignores other traffic moving past
            shift_x.append(float(np.median(flow[band, :, 0])))
            shift_y.append(float(np.median(flow[band, :, 1])))
            # driving forward, the right of the picture flows right and the left flows left; aimed sideways,
            # both sides slide the same way
            spread.append(float(np.median(flow[band, -third:, 0]) - np.median(flow[band, :third, 0])))
            # correlation of brightness-normalised frames: lamps and exposure changes keep it high, cuts do not
            ncc.append(float((thumb * prev_thumb).mean()))
        std.append(float(f.std()))
        sky.append(float(f[: H // 3].mean()))
        prev, prev_thumb = f, thumb
    return {k: np.array(v) for k, v in
            (('motion', motion), ('ncc', ncc), ('pan', pan), ('std', std), ('sky', sky), ('face', face),
             ('shift_x', shift_x), ('shift_y', shift_y), ('spread', spread))}


def contact_sheet(path, duration, out_png, n=12, cols=4):
    """Grid of n colour frames spread over the video, each stamped with its time, for review."""
    tiles = []
    for t in np.linspace(duration * 0.02, duration * 0.98, n):
        out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.1f}', '-i', path, '-frames:v', '1',
                              '-vf', 'scale=320:180', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                             capture_output=True)
        img = (np.frombuffer(out.stdout, np.uint8).reshape(180, 320, 3).copy()
               if len(out.stdout) == 320 * 180 * 3 else np.zeros((180, 320, 3), np.uint8))
        label = f'{int(t) // 60}:{int(t) % 60:02d}'
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out_png, np.vstack(rows))


def _rolling_median(x, n):
    if n <= 1 or len(x) == 0:
        return x
    pad = n // 2
    xp = np.pad(x, pad, mode='edge')
    return np.array([np.median(xp[i:i + n]) for i in range(len(x))])


def _drop_short_runs(mask, n):
    """Give runs shorter than n samples the value of the run before them (the next one at the start)."""
    out = mask.copy()
    edges = [0] + [i for i in range(1, len(mask)) if mask[i] != mask[i - 1]] + [len(mask)]
    for a, b in zip(edges, edges[1:]):
        if b - a < n and len(edges) > 2:
            out[a:b] = out[a - 1] if a > 0 else mask[b]
    return out


def _runs(mask):
    """(start, end) index pairs of consecutive True values, end exclusive."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def propose(sig, p=PARAMS):
    """Return proposed segments [{start, end, night}] in seconds, plus the reasons for each cut."""
    n = len(sig['motion'])
    if n == 0:
        return [], []
    moving = _rolling_median(sig['motion'], p['smooth_s'] * FPS) > p['moving_flow']
    blank = sig['std'] < p['blank_std']
    ncc = sig['ncc']
    # a cut is one sharp drop between coherent frames; turns and dark scenes drop over several frames
    cut = np.zeros(n, bool)
    cut[1:-1] = (ncc[1:-1] < p['cut_ncc']) & (ncc[:-2] > p['cut_neighbour_ncc']) & (ncc[2:] > p['cut_neighbour_ncc'])
    cut &= ~blank
    # in a corner the view slides sideways, so it still matches once shifted; a cut matches at no shift.
    # Off (None) until calibrated on labelled corners: missing a real cut is worse than an extra split.
    if p.get('cut_pan_ncc') is not None and 'pan' in sig:
        cut &= sig['pan'] < p['cut_pan_ncc']
    # under a bridge or in a short tunnel the sky goes dark for a moment and comes back; a cut does not
    sky = sig['sky']
    for i in np.flatnonzero(cut):
        before, after = sky[max(0, i - 20):max(0, i - 6)], sky[i + 6:i + 20]
        if not len(before) or not len(after):
            continue
        if sky[max(0, i - 8):i + 9].min() < p['bridge_dip'] * min(before.mean(), after.mean()):
            cut[i] = False

    keep = ~blank
    notes = []
    for a, b in _runs(~moving):
        at_edge = a == 0 or b == n
        limit = p['edge_stop_s'] if at_edge else p['mid_stop_s']
        if (b - a) / FPS > limit:
            keep[a:b] = False
            notes.append(f"stationary {a / FPS:.0f}-{b / FPS:.0f}s{' (edge)' if at_edge else ''}")
    for i in np.flatnonzero(cut):
        notes.append(f'cut at {i / FPS:.0f}s')
    if 'face' in sig:  # signals saved before this check have no face trace
        # a camera on the driver keeps seeing him, but turning his head hides the face for a while, so judge
        # the share of frames with a big face over a longer window rather than frame by frame
        window = p['driver_window_s'] * FPS
        share = np.convolve((sig['face'] > 0).astype(float), np.ones(window) / window, mode='same')
        driver = share > p['driver_share']
        for a, b in _runs(driver):
            if (b - a) / FPS >= p['driver_min_s']:
                keep[a:b] = False
                notes.append(f'camera on the driver {a / FPS:.0f}-{b / FPS:.0f}s')

    window = p['night_window_s'] * FPS
    night = _rolling_median((sig['sky'] < p['night_sky']).astype(float), window) > 0.5
    # brief dark or bright spells (underpasses, a flickering sky near the threshold) must not split a segment
    night = _drop_short_runs(night, window)

    segments = []
    for a, b in _runs(keep):
        bounds = [a] + [i for i in range(a + 1, b) if cut[i] or night[i] != night[i - 1]] + [b]
        for s, e in zip(bounds, bounds[1:]):
            if (e - s) / FPS >= p['min_segment_s']:
                segments.append({'start': int(np.ceil(s / FPS)), 'end': int(e // FPS),
                                 'night': int(round(night[s:e].mean()))})
    return segments, notes


def channel_vehicle_type(channel_id, df=None):
    """Most common vehicle type among the channel's videos already in the mapping, or None."""
    if not channel_id:
        return None, 0
    df = add_video.load_csv(add_video.FILE_PATH) if df is None else df
    counts = Counter()
    for _, row in df.iterrows():
        channels = add_video._parse_flat_list_cell(row.get('channel', ''))
        types = add_video._parse_flat_list_cell(row.get('vehicle_type', ''))
        for ch, vt in zip(channels, types):
            if add_video._normalize_optional_text(ch) == channel_id:
                try:
                    counts[int(vt)] += 1
                except (TypeError, ValueError):
                    continue
    if not counts:
        return None, 0
    vt, k = counts.most_common(1)[0]
    return vt, k / sum(counts.values())


def analyse_video(video_id, out_dir, meta=None):
    """Download a low resolution copy, propose segments, write a contact sheet; the copy is deleted."""
    meta = meta or fetch_metadata(video_id)
    work_dir = tempfile.mkdtemp(prefix='crowd_')
    path = download_low_res(video_id, work_dir)
    try:
        sig = signals(path)
        contact_sheet(path, meta['duration'] or len(sig['motion']) / FPS,
                      os.path.join(out_dir, f'{video_id}.jpg'))
    finally:
        os.remove(path)
        os.rmdir(work_dir)
    segments, notes = propose(sig)
    np.savez_compressed(os.path.join(out_dir, f'{video_id}_signals.npz'), **sig)
    return {'meta': meta, 'segments': segments, 'notes': notes}


MIN_UPLOAD_S = 300  # the paper requires source uploads of at least five minutes
EXCLUDE_TITLE = re.compile(r'walking tour|walk tour|city walk|time[- ]?lapse|hyperlapse|compilation'
                           r'|\bcrash|\baccident', re.I)
# highway driving and road trips; "via I-5" only passes along it on an otherwise urban drive
HIGHWAY_TITLE = re.compile(r'(?i:road ?trip|\bhighway\b|\bfreeway\b|\bhwy\b|\bmotorway\b|\bautobahn\b)'
                           r'|(?<![Vv]ia )(?:\b[Ii]nterstate[- ]?|\bI[- ]?|\bi[- ])\d{1,3}\b')


def list_channel(url):
    """Video ids of a channel URL as written in the channels sheet (bare channel, /videos or /search?query=)."""
    bare = re.match(r'(https?://(?:www\.)?youtube\.com/(?:@[^/?]+|channel/[^/?]+))/?(?:\?.*)?$', url.strip())
    if bare:
        url = bare.group(1) + '/videos'
    entries = json.loads(_yt_dlp('--flat-playlist', '-J', url.strip())).get('entries') or []
    return [e['id'] for e in entries if e and len(e.get('id') or '') == 11]


YT_API = 'https://www.googleapis.com/youtube/v3/'
NIGHT_TITLE = re.compile(r'\bnight\b|\bnocturn|\bat dusk\b|\bafter dark\b', re.I)


def _api_key():
    try:
        return common.get_secrets('youtube_api_key')
    except (KeyError, OSError):
        return ''


def _api(endpoint, **params):
    """GET one YouTube Data API endpoint; errors never include the request URL, which carries the key."""
    key = _api_key()
    if not key:
        raise RuntimeError('no YouTube Data API key: add it to the secret file as "youtube_api_key"')
    try:
        r = requests.get(YT_API + endpoint, params={**params, 'key': key}, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f'YouTube Data API {endpoint}: {type(e).__name__}') from None
    if r.status_code != 200:
        raise RuntimeError(f"YouTube Data API {endpoint}: {r.json().get('error', {}).get('message', r.status_code)}")
    return r.json()


def _api_pages(endpoint, items_key, **params):
    out, token = [], None
    while True:
        d = _api(endpoint, **params, **({'pageToken': token} if token else {}))
        out += d.get(items_key, [])
        token = d.get('nextPageToken')
        if not token:
            return out


def list_channel_api(url):
    """Video ids of a channel URL from the YouTube Data API: its uploads, or its search results for ?query=."""
    m = re.search(r'/channel/(UC[\w-]{22})', url)
    if m:
        channel = m.group(1)
    else:
        handle = re.search(r'/@([^/?]+)', url)
        items = _api('channels', part='id', forHandle='@' + unquote(handle.group(1))).get('items') if handle else None
        if not items:
            raise RuntimeError(f'no channel found for {url}')
        channel = items[0]['id']
    query = parse_qs(urlparse(url).query).get('query', [None])[0]
    if query:  # search costs 100 quota units per page of 50 (10,000 a day are free)
        found = _api_pages('search', 'items', part='id', channelId=channel, q=query, type='video', maxResults=50)
        ids = [i['id']['videoId'] for i in found]
    else:
        found = _api_pages('playlistItems', 'items', part='contentDetails', playlistId='UU' + channel[2:],
                           maxResults=50)
        ids = [i['contentDetails']['videoId'] for i in found]
    return list(dict.fromkeys(ids))


def _iso_seconds(duration):
    """'PT1H2M3S' (also with days) -> 3723."""
    parts = re.fullmatch(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?', duration or '')
    if not parts:
        return None
    d, h, m, s = (int(x or 0) for x in parts.groups())
    return ((d * 24 + h) * 60 + m) * 60 + s


def fetch_metadata_api(video_ids):
    """Metadata for many videos, 50 per request (1 quota unit each), in the same shape as fetch_metadata."""
    out = {}
    for i in range(0, len(video_ids), 50):
        # the player part's embed size follows the video's own proportions, which reveals portrait videos
        for v in _api('videos', part='snippet,contentDetails,player', id=','.join(video_ids[i:i + 50]),
                      maxWidth=1000).get('items', []):
            sn, pl = v['snippet'], v.get('player', {})
            day = sn.get('publishedAt', '')[:10]  # YYYY-MM-DD
            out[v['id']] = {
                'id': v['id'], 'title': sn.get('title'), 'duration': _iso_seconds(v['contentDetails'].get('duration')),
                'upload_date': f'{day[8:10]}{day[5:7]}{day[:4]}' if day else None, 'channel_id': sn.get('channelId'),
                'live_status': {'live': 'is_live', 'upcoming': 'is_upcoming'}.get(sn.get('liveBroadcastContent')),
                'description': sn.get('description') or '', 'chapters': [],
                'width': int(pl['embedWidth']) if pl.get('embedWidth') else None,
                'height': int(pl['embedHeight']) if pl.get('embedHeight') else None,
            }
    return out


def exclusion_reason(meta):
    """Why the protocol rules a video out from its metadata alone, or None."""
    if (meta.get('duration') or 0) < MIN_UPLOAD_S:
        return f"{(meta.get('duration') or 0) / 60:.1f} min, under the 5 min minimum"
    if meta.get('live_status') in ('is_live', 'is_upcoming'):
        return 'live or upcoming stream'
    if meta.get('width') and meta.get('height') and meta['height'] > meta['width']:
        return 'portrait (phone) video'
    hit = EXCLUDE_TITLE.search(meta.get('title') or '')
    if hit:
        return f"title mentions '{hit.group(0)}'"
    hit = HIGHWAY_TITLE.search(meta.get('title') or '')
    return f"highway driving or a road trip (title mentions '{hit.group(0)}')" if hit else None


def guess_locality(title, country, df):
    """[locality, state, country] when the title names exactly one mapping locality of that country, else None."""
    found = set()
    for _, row in df[df['country'] == country].iterrows():
        names = [row['locality']]
        aka = row.get('locality_aka')
        if isinstance(aka, str) and aka.startswith('['):
            names += [a.strip() for a in aka[1:-1].split(',') if a.strip()]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            # short aliases such as "LA" only count in capitals, so "la" in a Spanish title does not match
            flags = 0 if len(name) <= 3 else re.I
            if re.search(rf'(?<!\w){re.escape(name)}(?!\w)', title or '', flags):
                state = row['state'] if isinstance(row['state'], str) else None
                found.add((row['locality'], state, country))
    return list(found.pop()) if len(found) == 1 else None


def _km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def _place_near(name, iso2, lat, lon, km=60):
    """True when GeoNames knows a populated place called name within km of (lat, lon)."""
    try:
        r = requests.get('http://api.geonames.org/searchJSON', timeout=15, params={
            'name_equals': name, 'country': iso2, 'featureClass': 'P', 'maxRows': 50,
            'username': common.get_secrets('geonames_username')})
        places = r.json().get('geonames', [])
    except (requests.RequestException, ValueError, KeyError):
        return False
    return any(_km(lat, lon, float(p['lat']), float(p['lng'])) <= km for p in places)


def channel_home(channel_id, df, plan):
    """(locality, state, country, lat, lon) where a channel mostly films: from its videos already in the mapping
    and localities set by hand in the plan, else from title guesses; None if nothing points anywhere."""
    counts = Counter()
    for _, row in df.iterrows():
        n = sum(add_video._normalize_optional_text(c) == channel_id
                for c in add_video._parse_flat_list_cell(row.get('channel', '')))
        if n:
            counts[(row['locality'], row['state'] if isinstance(row['state'], str) else None, row['country'])] += n
    for p in plan.values():
        for e in p.get('entries', []):
            if e.get('locality') and not e.get('guessed'):
                counts[tuple(e['locality'])] += 1
    if not counts:
        counts = Counter(tuple(e['locality']) for p in plan.values()
                         for e in p.get('entries', []) if e.get('locality'))
    for (locality, state, country), _ in counts.most_common():
        _, row = add_video.get_existing_locality_row(df, locality, state, country)
        if row is not None:
            return locality, state, country, float(row['lat']), float(row['lon'])
    return None


def find_reuploads(sigs, upload_dates):
    """{later video: earlier video} for uploads whose footage is the same (near-identical brightness traces)."""
    def day(vid):
        d = upload_dates.get(vid) or '01011900'
        return d[4:], d[2:4], d[:2]

    ids = sorted(sigs, key=day)
    dupes = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            x, y = sigs[a]['sky'], sigs[b]['sky']
            n = min(len(x), len(y))
            if b not in dupes and abs(len(x) - len(y)) < 20 * FPS and n > 1 and np.corrcoef(x[:n], y[:n])[0, 1] > 0.98:
                dupes[b] = a
    return dupes


@contextlib.contextmanager
def plan_lock(plan_path):
    """Exclusive lock for reading-and-writing a plan; the review page and channel runs both take it."""
    with open(plan_path + '.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def process_channel(url, country, out_dir, pause_s=15, limit=None, download=True):
    """Propose segments for every new video of a channel and write plan.json for the review page.

    Re-running resumes: videos already in the plan keep their entry and decisions, finished analyses are
    reused, and a run stopped by YouTube's sign-in check continues where it stopped. With limit, a run stops
    after that many new proposals, so the next run is the next batch. Without download, nothing is
    downloaded: each video is proposed whole from its metadata, and a later run with download analyses
    those not decided yet.
    """
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, 'plan.json')
    plan = {}  # video -> plan entry, in plan order
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            plan = {p['video']: p for p in json.load(f)
                    if not str(p.get('exclude', '')).startswith('could not analyse')}

    def undecided(p):
        return not any(s.get('decision') or s.get('applied') for e in p.get('entries', []) for s in e['segments'])

    def as_proposed(p):
        """A whole-video proposal nobody has touched: one segment over the whole video, day/night from the
        title, no locality set by hand, nothing decided."""
        if p.get('analysed') is not False or len(p.get('entries', [])) != 1 or not undecided(p):
            return False
        e = p['entries'][0]
        meta_path = os.path.join(out_dir, f"{p['video']}_meta.json")
        if (e.get('locality') and not e.get('guessed')) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            duration = json.load(f).get('duration')
        night = int(bool(NIGHT_TITLE.search(p.get('title') or '')))
        return e['segments'] == [{'start': 0, 'end': int(duration or -1), 'night': night}]

    # whole-video proposals made without downloading get analysed now, unless someone has started on them;
    # they stay in the plan until their analysis succeeds
    reanalyse = {v for v, p in plan.items() if download and as_proposed(p)}
    committed = {}  # video -> what this run last wrote for it

    def commit(videos):
        """Write these videos into the plan file, merging with what is there now. The review page may be open:
        a video edited there since this run started keeps the edits, and this run's version is dropped."""
        with plan_lock(plan_path):
            disk = {}
            if os.path.exists(plan_path):
                with open(plan_path) as f:
                    disk = {p['video']: p for p in json.load(f)}
            for v in videos:
                cur = disk.get(v)
                ours = cur is not None and json.dumps(cur, sort_keys=True) == committed.get(v)
                if cur is None or ours or as_proposed(cur) or str(cur.get('exclude', '')).startswith('could not'):
                    disk[v] = plan[v]
                    committed[v] = json.dumps(plan[v], sort_keys=True)
            tmp = plan_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(list(disk.values()), f, indent=1)
            os.replace(tmp, plan_path)
    df = add_video.load_csv(add_video.FILE_PATH)
    in_mapping = {v for cell in df['videos'] for v in add_video._parse_videos_cell(cell)}
    vehicle_cache = {}
    use_api = bool(_api_key())
    if not download and not use_api:
        print('Without downloading, video details come from the YouTube Data API: '
              'add a key to the secret file as "youtube_api_key".')
        return None

    stopped = None
    touched = []  # videos added or re-analysed in this run
    failures_in_a_row = 0
    try:
        ids = list_channel_api(url) if use_api else list_channel(url)
        todo = [v for v in ids if (v not in plan and v not in in_mapping) or v in reanalyse]
        api_meta = fetch_metadata_api(todo) if use_api else {}
    except BotCheck as e:
        print(f'YouTube asks to sign in ({e}); try again later.')
        return None
    except RuntimeError as e:
        print(f'Could not list the channel: {e}')
        return None
    for n, vid in enumerate(ids, 1):
        if vid not in todo:
            continue
        if limit and sum('entries' in plan[v] for v in touched) >= limit:
            break
        if failures_in_a_row >= 3:
            stopped = 'three videos in a row could not be analysed'
            break
        meta_path = os.path.join(out_dir, f'{vid}_meta.json')
        sig_path = os.path.join(out_dir, f'{vid}_signals.npz')
        try:
            if vid in api_meta:
                meta = api_meta[vid]
            elif os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = fetch_metadata(vid)
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            reason = exclusion_reason(meta)
            if reason:
                plan[vid] = {'video': vid, 'title': meta['title'], 'exclude': reason}
                touched.append(vid)
                commit([vid])
                continue
            if download and not os.path.exists(sig_path):
                print(f'[{n}/{len(ids)}] analysing {vid} {meta["title"][:60]}')
                analyse_video(vid, out_dir, meta)
                time.sleep(pause_s)  # spacing requests keeps YouTube's bot check away for longer
            failures_in_a_row = 0
        except BotCheck as e:
            stopped = f'YouTube asks to sign in ({e})'
            break
        except Exception as e:
            failures_in_a_row += 1
            if vid not in reanalyse:  # a whole-video proposal stays as it was
                plan[vid] = {'video': vid, 'title': '', 'exclude': f'could not analyse: {e}'}
                touched.append(vid)
                commit([vid])
            continue
        analysed = os.path.exists(sig_path)
        reason = None
        if analysed:
            frames = sheet_frames(os.path.join(out_dir, f'{vid}.jpg'))
            if inset_rectangle(frames):
                reason = 'another camera is shown in a corner of the picture (picture-in-picture)'
            elif (n_driver := driver_frames(frames)) >= DRIVER_SHEET_FRAMES:
                reason = f'the camera faces the driver (a big face in {n_driver} of {len(frames)} sampled frames)'
        if reason:
            plan[vid] = {'video': vid, 'title': meta['title'], 'exclude': reason}
            touched.append(vid)
            commit([vid])
            continue
        if analysed:
            segments, notes = propose(dict(np.load(sig_path)))
        else:
            night = bool(NIGHT_TITLE.search(meta['title'] or ''))
            segments = [{'start': 0, 'end': int(meta['duration']), 'night': int(night)}]
            notes = ['not analysed: whole video proposed, check where driving starts and ends'
                     + ('; night from the title' if night else '')]
            if thumbnail_face(vid):
                notes.append('the thumbnail shows a big face: the camera may face the driver, check')
        ch = meta.get('channel_id')
        if ch not in vehicle_cache:
            vehicle_cache[ch] = channel_vehicle_type(ch, df)
        vt, share = vehicle_cache[ch]
        vehicle_note = ('no channel history: vehicle assumed car, check the frames' if vt is None
                        else f'vehicle from channel history ({share:.0%} of its videos)')
        if vid in reanalyse:  # keep a vehicle type picked in the review page
            vt, vehicle_note = plan[vid].get('vehicle_type', vt), 'vehicle as in the earlier proposal'
        locality = guess_locality(meta['title'], country, df)
        touched.append(vid)
        plan[vid] = ({
            'video': vid, 'title': meta['title'], 'analysed': analysed,
            'entries': [{'locality': locality, 'guessed': locality is not None, 'segments': segments}],
            'vehicle_type': 0 if vt is None else vt, 'upload_date': meta['upload_date'], 'channel_id': ch,
            'note': '; '.join(filter(None, ['; '.join(notes), vehicle_note])),
        })
        commit([vid])  # appears in an open review page straight away

    new = [plan[v] for v in touched]
    # a channel films around one area: a title place whose mapping match is far from the channel's home is a
    # namesake when a place of that name exists near home ("Hollywood" from an LA channel is not Hollywood, FL),
    # and a genuine trip otherwise ("Seattle" has no namesake near LA)
    channels = Counter(p.get('channel_id') for p in plan.values() if p.get('channel_id'))
    home = channel_home(channels.most_common(1)[0][0], df, plan) if channels else None
    if home:
        iso2 = common.get_iso2_country_code(common.correct_country(home[2]))
        for p in new:
            for e in p.get('entries', []):
                loc = e.get('locality')
                if not (loc and e.get('guessed')) or tuple(loc) == home[:3]:
                    continue
                _, row = add_video.get_existing_locality_row(df, *loc)
                if row is None or _km(home[3], home[4], float(row['lat']), float(row['lon'])) < 100:
                    continue
                if _place_near(loc[0], iso2, home[3], home[4]):
                    p['note'] = f"{p['note']}; the title's {loc[0]} is the place of that name near {home[0]}, " \
                                f"not {', '.join(filter(None, loc))}".lstrip('; ')
                    e['locality'] = list(home[:3])

    def has_signals(v):
        return os.path.exists(os.path.join(out_dir, f'{v}_signals.npz'))

    fresh = {p['video'] for p in new if 'entries' in p and has_signals(p['video'])}
    sigs = {v: dict(np.load(os.path.join(out_dir, f'{v}_signals.npz')))
            for v, p in plan.items() if 'entries' in p and has_signals(v)}
    dates = {v: p.get('upload_date') for v, p in plan.items()}
    for later, earlier in find_reuploads(sigs, dates).items():
        if later in fresh:  # never overrule entries already being reviewed
            plan[later].pop('entries')
            plan[later]['exclude'] = f're-upload of {earlier} (same footage)'

    commit(touched)
    kept = sum('entries' in p for p in new)
    print(f'{len(new)} videos added or re-analysed: {kept} proposed, {len(new) - kept} excluded -> {plan_path}')
    if stopped:
        print(f'Stopped early: {stopped}. Run the same command later to continue.')
    left = sum((v not in plan and v not in in_mapping) or (v in reanalyse and v not in touched) for v in ids)
    if left:
        print(f'{left} videos of this channel are still to do: run the same command again for the next batch.')
    return plan_path


# the codes the add-video form accepts
VEHICLE_TYPES = {0: 'car', 1: 'bus', 2: 'truck', 3: 'two-wheeler', 4: 'bicycle/e-bicycle', 5: 'automated car',
                 6: 'electric scooter', 7: 'monowheel/unicycle', 8: 'emergency vehicle', 9: 'automated bus',
                 10: 'automated truck', 11: 'automated two-wheeler', 12: 'non-electric scooter'}

# the country names the add-video form offers, read from its template so both stay the same list
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'add_video.html'),
          encoding='utf-8') as _f:
    COUNTRIES = re.findall(r'<option value="([^"]+)" \{\{ "selected" if country ==', _f.read())

MAX_MERGE_GAP_S = 2  # wider gaps are excluded footage (stops, freeway), not suggested splits

LOCALITY_FIELDS = ('lat', 'lon', 'gmp', 'population_locality', 'population_country', 'traffic_mortality',
                   'continent', 'locality_aka', 'literacy_rate', 'avg_height', 'med_age', 'gini', 'traffic_index')


def apply_segments(video_id, locality, state, country, segments, vehicle_type, upload_date, channel_id):
    """Add reviewed segments to a locality through the add-video form's own submit path, so its overlap and
    value checks apply. A locality not in the mapping yet is created from the same lookups the form makes.
    Writes the mapping file."""
    client = add_video.app.test_client()
    new_row = None
    for seg in segments:
        _, row = add_video.get_existing_locality_row(add_video.load_csv(add_video.FILE_PATH), locality, state, country)
        if row is None:
            new_row = new_row or add_video.new_locality_row(locality, state or None, country)
            row = new_row
        # the form re-saves the locality fields it shows, so send them back exactly as it would render them
        form = {k: str(row[k]) for k in LOCALITY_FIELDS}
        form.update({
            'locality': row['locality'], 'state': '' if state is None else state, 'country': row['country'],
            'video_url': _url(video_id), 'time_of_day': str(seg['night']), 'vehicle_type': str(vehicle_type),
            'start_time': str(seg['start']), 'end_time': str(seg['end']),
            'upload_date_video': upload_date or '', 'channel_video': channel_id or '', 'submit_data': '1',
        })
        html = client.post('/', data=form).get_data(as_text=True)
        if 'Video added or updated successfully' not in html:
            msg = html.split('<h3>', 1)[-1].split('</h3>', 1)[0].strip()
            raise RuntimeError(f'{video_id} {seg}: {msg}')


REVIEW_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Review proposals</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f6f6f6; color: #222; }
header { position: sticky; top: 0; z-index: 1; background: #fff; border-bottom: 1px solid #ddd;
         display: flex; gap: 20px; padding: 10px 20px; align-items: center; }
#player { width: 480px; height: 270px; border: 0; background: #000; }
#status { font-size: 15px; line-height: 1.6; }
#apply { padding: 10px 18px; font-size: 15px; border: 0; border-radius: 4px;
         background: #007bff; color: #fff; cursor: pointer; }
#apply:disabled { background: #9bb; cursor: not-allowed; }
#log { font-size: 13px; white-space: pre-wrap; max-width: 520px; }
main { padding: 10px 20px 40px; max-width: 1400px; }
.video { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px 16px; margin: 12px 0; }
.video, tr.seg { scroll-margin-top: 300px; }  /* land below the sticky player */
.video { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 380px); gap: 16px; }
.desc { font-size: 13px; color: #333; max-height: 520px; overflow-y: auto; border-left: 1px solid #eee;
        padding-left: 12px; }
.desctext { white-space: pre-wrap; overflow-wrap: anywhere; }
.chapters { margin-bottom: 8px; }
input.at { width: 56px; font-size: 13px; padding: 2px 4px; }
@media (max-width: 900px) { .video { grid-template-columns: 1fr; } .desc { border-left: 0; padding-left: 0; } }
.video h3 { margin: 0 0 4px; font-size: 16px; }
.note { color: #8a5300; font-size: 13px; margin: 2px 0 8px; }
.sheet { width: 640px; max-width: 100%; display: block; margin: 8px 0; }
.thumb { width: 320px; max-width: 100%; display: block; margin: 8px 0; }
table { border-collapse: collapse; }
td { padding: 4px 8px; font-size: 14px; border-top: 1px solid #eee; }
button.tod.night { background: #2a3f8f; color: #fff; border-color: #2a3f8f; }
label.vehicle { margin-left: 16px; font-size: 14px; }
tr.entryhead td { border-top: 0; padding-top: 8px; }
.locbox { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.locbox input, .locbox select { font-size: 14px; padding: 3px 6px; }
.locwrap { position: relative; }
.loc-suggestions { display: none; position: absolute; top: 100%; left: 0; min-width: 320px; z-index: 5;
                   background: #fff; border: 1px solid #ccc; max-height: 260px; overflow-y: auto;
                   box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); }
.loc-suggestions.open { display: block; }
.loc-suggestions div { padding: 6px 10px; cursor: pointer; font-size: 14px; }
.loc-suggestions div:hover { background: #f0f8ff; }
input.t-start, input.t-end { width: 62px; font-size: 13px; padding: 2px 4px; }
button.small { padding: 3px 8px; font-size: 13px; cursor: pointer;
               border: 1px solid #bbb; border-radius: 3px; background: #fafafa; }
.seg.approve td { background: #e8f6ea; }
.seg.reject td { background: #fbe9e9; color: #888; }
.seg.approve .yes, .seg.reject .no { font-weight: bold; border-color: #333; }
.seg.applied td { background: #e3ecf8; }
tr.split td { font-size: 12px; color: #777; border-top: 0; padding-top: 0; padding-bottom: 0; }
.excluded { color: #777; font-size: 14px; }
.night { color: #2a3f8f; font-weight: bold; }
.guess { color: #b36b00; font-size: 12px; }
.clock { font-size: 18px; margin-bottom: 4px; }
#clock { color: #c00; cursor: pointer; text-decoration: underline; font-variant-numeric: tabular-nums; }
#copied { color: green; font-size: 13px; }
.keys { font-size: 12px; color: #555; margin-bottom: 6px; }
tr.seg.active td:first-child { box-shadow: inset 4px 0 #007bff; }
</style></head><body>
<header>
  <div id="player"></div>
  <div>
    <div class="clock">Time: <span id="clock" title="click to copy">0:00</span> <span id="copied"></span></div>
    <div class="keys">Keys for the outlined segment: <b>A</b> start here · <b>S</b> end here ·
      <b>D</b> whole video · <b>Q</b> day · <b>W</b> night · <b>F</b> split time here</div>
    <div id="status"></div>
    <button id="apply" disabled>Apply approved segments to the mapping</button>
    <div id="log"></div>
  </div>
</header>
<main>
{% for p in plan if p.entries %}
<div class="video"><div class="main">
  <h3>{{ p.title }}</h3>
  <a href="https://www.youtube.com/watch?v={{ p.video }}" target="_blank">{{ p.video }}</a>
  <label class="vehicle">Vehicle (whole video):
    <select onchange="setVehicle('{{ p.video }}', this)">
    {% for code, name in vehicle_types.items() %}
      <option value="{{ code }}" {{ 'selected' if code == p.vehicle_type }}>{{ code }} ({{ name }})</option>
    {% endfor %}
    </select></label>
  {% if p.note %}<div class="note">{{ p.note }}</div>{% endif %}
  {% if p.analysed is false %}
  <img class="thumb" src="https://i.ytimg.com/vi/{{ p.video }}/hqdefault.jpg" loading="lazy"
       alt="YouTube thumbnail of {{ p.video }} (not analysed)">
  {% else %}
  <img class="sheet" src="sheet/{{ p.video }}.jpg" loading="lazy" alt="frames from {{ p.video }}">
  {% endif %}
  <table>
  {% for e in p.entries %}{% set ei = loop.index0 %}
    <tr class="entryhead"><td colspan="5">
      {% if e.segments | selectattr('applied') | list %}Locality: {{ e.locality | select | join(', ') }}
      {% else %}
      <div class="locbox" data-video="{{ p.video }}" data-entry="{{ ei }}">
        <span class="locwrap"><input class="loc-name" placeholder="Locality" autocomplete="off" size="22"
            value="{{ e.locality[0] if e.locality else '' }}" oninput="suggest(this)"
            onblur="hideSuggestions(this)" onchange="saveLocality(this)">
          <div class="loc-suggestions"></div></span>
        <input class="loc-state" placeholder="State (optional)" size="14" autocomplete="off"
            value="{{ e.locality[1] if e.locality and e.locality[1] else '' }}" onchange="saveLocality(this)">
        <select class="loc-country" onfocus="fillCountries(this)" onmousedown="fillCountries(this)"
            onchange="saveLocality(this)"><option value="">Country</option>
          {% if e.locality %}<option selected>{{ e.locality[2] }}</option>{% endif %}</select>
        <span class="locstatus guess">{% if e.guessed %}guessed from title{% elif not e.locality %}
          ⚠ not set: approved segments cannot be added{% endif %}</span>
      </div>
      {% endif %}
    </td></tr>
  {% for s in e.segments %}
    <tr class="seg {{ s.decision or '' }} {{ 'applied' if s.applied else '' }}"
        data-video="{{ p.video }}" data-entry="{{ ei }}" data-seg="{{ loop.index0 }}"
        data-start="{{ s.start }}" data-end="{{ s.end }}">
      <td class="times">{% set t0 = '%d:%02d' % (s.start // 60, s.start % 60) %}
          {%- set t1 = '%d:%02d' % (s.end // 60, s.end % 60) %}
          {%- if s.applied %}{{ t0 }} – {{ t1 }}{% else %}
          <input class="t-start" value="{{ t0 }}" title="start (m:ss)" onchange="editTimes(this)"> –
          <input class="t-end" value="{{ t1 }}" title="end (m:ss)" onchange="editTimes(this)">{% endif %}
          <span class="dur">({{ '%d:%02d' % ((s.end - s.start) // 60, (s.end - s.start) % 60) }})</span></td>
      <td>{% if s.applied %}{{ 'night' if s.night else 'day' }}{% else %}
          <button class="small tod {{ 'night' if s.night else '' }}" title="switch day/night"
                  onclick="toggleNight(this)">{{ 'night' if s.night else 'day' }}</button>{% endif %}</td>
      <td><button class="small" onclick="playRow(this, 'start')">▶ start</button>
          <button class="small" onclick="playRow(this, 'end')">▶ end</button></td>
      <td>{% if s.applied %}added{% else %}
          <button class="small yes" onclick="decide(this, 'approve', true)">Approve</button>
          <button class="small no" onclick="decide(this, 'reject')">Reject</button>{% endif %}</td>
      <td>{% if not s.applied %}
          <input class="at" placeholder="m:ss" title="time to split at, e.g. where the drive enters another locality">
          <button class="small" onclick="splitAt(this)">Split</button>
          <button class="small" onclick="moveSegment(this)" title="put this segment under another locality">
          → other locality</button>{% endif %}</td>
    </tr>
    {% if not loop.last %}{% set n = e.segments[loop.index] %}
    {% if n.start - s.end <= max_merge_gap and n.night == s.night and not s.applied and not n.applied %}
    <tr class="split">
      <td colspan="2">suggested split at {{ '%d:%02d' % (s.end // 60, s.end % 60) }}</td>
      <td><button class="small" onclick="play('{{ p.video }}', {{ [s.end - 5, 0] | max }})">▶ split</button></td>
      <td><button class="small" onclick="mergeSplit('{{ p.video }}', {{ ei }}, {{ loop.index0 }},
          [[{{ s.start }}, {{ s.end }}], [{{ n.start }}, {{ n.end }}]])">Remove split</button></td>
    </tr>
    {% endif %}{% endif %}
  {% endfor %}
    <tr class="addrow"><td colspan="5">
      <button class="small" data-video="{{ p.video }}" data-entry="{{ ei }}" onclick="newSegment(this)"
              title="from the current video time (or after the last segment) to the next segment or the end">
        + new segment{{ ' in ' ~ e.locality[0] if e.locality }}</button></td></tr>
  {% endfor %}
  </table>
  <button class="small" onclick="approveRest(this)">Approve the rest of this video</button>
  <button class="small" onclick="rejectVideo(this)">Reject whole video</button>
  </div>
  <div class="desc">{% set d = descriptions.get(p.video) %}
    {% if d %}
      {% if d.chapters %}<div class="chapters"><b>Chapters</b>{% for t, title in d.chapters %}
        <div><a href="#" onclick="play('{{ p.video }}', {{ t }}); return false;">
          {{ '%d:%02d' % (t // 60, t % 60) }}</a> {{ title }}</div>{% endfor %}</div>{% endif %}
      <div class="desctext">{{ d.html }}</div>
    {% else %}<span class="excluded">no description saved for this video</span>{% endif %}
  </div>
</div>
{% endfor %}
<h3>Excluded videos</h3>
{% for p in plan if p.exclude %}
<div class="excluded"><a href="https://www.youtube.com/watch?v={{ p.video }}" target="_blank">{{ p.video }}</a>
  {{ p.title }} — {{ p.exclude }}</div>
{% endfor %}
</main>
<script>
let ytPlayer = null, ytReady = false, pending = null;
function onYouTubeIframeAPIReady() {
  ytPlayer = new YT.Player('player', { width: 480, height: 270, playerVars: { autoplay: 1, playsinline: 1, rel: 0 },
    events: { onReady: () => { ytReady = true; if (pending) play(...pending); } } });
}
function play(video, t) {
  if (!ytReady) { pending = [video, t]; return; }
  pending = null;
  ytPlayer.mute();
  ytPlayer.loadVideoById({ videoId: video, startSeconds: Math.max(0, Math.floor(t)) });
}
function playRow(btn, which) {
  const r = btn.closest('tr');
  if (which === 'start') play(r.dataset.video, +r.dataset.start);
  else play(r.dataset.video, Math.max(+r.dataset.end - 3, +r.dataset.start));
}
const pad = n => String(n).padStart(2, '0');
const fmt = t => `${Math.floor(t / 60)}:${pad(Math.floor(t) % 60)}`;
function playingVideo() {
  try { return new URL(ytPlayer.getVideoUrl()).searchParams.get('v'); } catch (e) { return null; }
}
function activeRow() {
  // the segment of the playing video that the current time is in, or the nearest one
  const video = ytReady && playingVideo();
  if (!video) return null;
  const t = ytPlayer.getCurrentTime();
  let best = null, bestDistance = Infinity;
  for (const r of document.querySelectorAll(`tr.seg:not(.applied)[data-video="${video}"]`)) {
    const distance = Math.max(0, +r.dataset.start - t, t - +r.dataset.end);
    if (distance < bestDistance) { best = r; bestDistance = distance; }
  }
  return best;
}
setInterval(() => {
  if (!ytReady) return;
  document.getElementById('clock').textContent = fmt(ytPlayer.getCurrentTime() || 0);
  const row = activeRow();
  document.querySelectorAll('tr.seg.active').forEach(r => { if (r !== row) r.classList.remove('active'); });
  if (row) row.classList.add('active');
}, 250);
document.getElementById('clock').onclick = async () => {
  await navigator.clipboard.writeText(document.getElementById('clock').textContent);
  document.getElementById('copied').textContent = 'copied';
  setTimeout(() => { document.getElementById('copied').textContent = ''; }, 1000);
};
function showBounds(row) {
  const start = +row.dataset.start, end = +row.dataset.end;
  row.querySelector('.t-start').value = fmt(start);
  row.querySelector('.t-end').value = fmt(end);
  row.querySelector('.dur').textContent = `(${fmt(end - start)})`;
}
async function setBounds(row, start, end) {
  if (!(start < end)) { alert('The start must be before the end.'); showBounds(row); return; }
  const res = await fetch('bounds', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: row.dataset.video, entry: +row.dataset.entry, seg: +row.dataset.seg,
                           bounds: [[+row.dataset.start, +row.dataset.end]], start, end }) });
  if (!res.ok) { alert((await res.json()).error); showBounds(row); return; }
  row.dataset.start = start;
  row.dataset.end = end;
  showBounds(row);
}
function editTimes(input) {
  const row = input.closest('tr');
  const start = parseTime(row.querySelector('.t-start').value), end = parseTime(row.querySelector('.t-end').value);
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    alert('Times are m:ss or h:mm:ss, e.g. 2:05 or 1:02:05');
    showBounds(row);
    return;
  }
  setBounds(row, start, end);
}
document.addEventListener('keydown', e => {
  // same keys as the add-video form; never while typing in a box or with a modifier held
  if (['input', 'textarea', 'select'].includes(document.activeElement.tagName.toLowerCase())) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  let key = e.key.toLowerCase();
  if (!/^[a-z]$/.test(key) && /^Key[A-Z]$/.test(e.code)) key = e.code.slice(3).toLowerCase();  // Cyrillic layouts
  if (!['a', 's', 'd', 'q', 'w', 'f'].includes(key)) return;
  const row = activeRow();
  if (!row) return;
  e.preventDefault();
  const t = Math.floor(ytPlayer.getCurrentTime());
  if (key === 'f') {  // current time into the Split box of the segment it falls in, and to the clipboard
    row.querySelector('input.at').value = fmt(t);
    navigator.clipboard.writeText(fmt(t)).catch(() => {});
    document.getElementById('copied').textContent = `${fmt(t)} in the Split box`;
    setTimeout(() => { document.getElementById('copied').textContent = ''; }, 1500);
  }
  if (key === 'a') setBounds(row, t, +row.dataset.end);
  if (key === 's') setBounds(row, +row.dataset.start, t);
  if (key === 'd') setBounds(row, 0, Math.floor(ytPlayer.getDuration()) - 1);
  const tod = row.querySelector('button.tod');
  if (tod && (key === 'q' || key === 'w') && tod.classList.contains('night') !== (key === 'w')) toggleNight(tod);
});
async function decide(btn, decision, advance = false) {
  const row = btn.closest('tr');
  const res = await fetch('decide', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: row.dataset.video, entry: +row.dataset.entry, seg: +row.dataset.seg, decision,
                           bounds: [[+row.dataset.start, +row.dataset.end]] }) });
  if (!res.ok) { alert((await res.json()).error); return false; }
  row.classList.remove('approve', 'reject');
  row.classList.add(decision);
  update();
  if (advance) goToNextOpen(row);
  return true;
}
function goToNextOpen(row) {
  // next undecided segment: the rest of this video first, then the following videos, then from the top
  const open = r => !r.classList.contains('approve') && !r.classList.contains('reject');
  const rows = [...document.querySelectorAll('tr.seg:not(.applied)')];
  const i = rows.indexOf(row);
  const next = rows.slice(i + 1).find(open) || rows.slice(0, Math.max(i, 0)).find(open);
  if (!next) return;
  const target = next.closest('.video') === row.closest('.video') ? next : next.closest('.video');
  target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  play(next.dataset.video, +next.dataset.start);
}
const COUNTRIES = {{ countries | tojson }};
function fillCountries(select) {
  // the full list is added on first use, so 100 videos do not each carry 248 options
  if (select.dataset.filled) return;
  const current = select.value;
  select.innerHTML = '<option value="">Country</option>';
  for (const c of COUNTRIES) select.appendChild(new Option(c, c, false, c === current));
  select.dataset.filled = '1';
}
async function suggest(input) {
  const box = input.parentElement.querySelector('.loc-suggestions');
  const q = input.value.trim();
  const mine = input._request = (input._request || 0) + 1;  // only the newest response is shown
  box.classList.remove('open');
  if (q.length < 2) return;
  const list = await (await fetch('localities?q=' + encodeURIComponent(q))).json();
  if (mine !== input._request) return;
  box.innerHTML = '';
  for (const item of list) {
    const div = document.createElement('div');
    div.textContent = item.label;
    div.onmousedown = e => { e.preventDefault(); choose(input, item.value); };
    box.appendChild(div);
  }
  box.classList.toggle('open', list.length > 0);
}
function hideSuggestions(input) {
  setTimeout(() => input.parentElement.querySelector('.loc-suggestions').classList.remove('open'), 150);
}
function choose(input, [locality, state, country]) {
  const box = input.closest('.locbox');
  box.querySelector('.loc-name').value = locality;
  box.querySelector('.loc-state').value = state || '';
  const select = box.querySelector('.loc-country');
  fillCountries(select);
  select.value = country;
  box.querySelector('.loc-suggestions').classList.remove('open');
  saveLocality(input);
}
async function saveLocality(el) {
  const box = el.closest('.locbox');
  const locality = box.querySelector('.loc-name').value.trim();
  const state = box.querySelector('.loc-state').value.trim();
  const country = box.querySelector('.loc-country').value;
  const status = box.querySelector('.locstatus');
  if (!locality || !country) { status.textContent = '⚠ not set: give a locality and a country'; return; }
  const res = await fetch('locality', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: box.dataset.video, entry: +box.dataset.entry,
                           locality: [locality, state, country] }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  const out = await res.json();
  if (out.merged) { reloadKeepingPlace(); return; }  // joined another group of this video with that locality
  status.textContent = out.new ? 'new locality: created on Apply, with the lookups the add-video form makes'
                               : '✓ in the mapping';
}
function parseTime(text) {
  const parts = text.trim().split(':').map(Number);
  return parts.length && parts.every(n => Number.isFinite(n)) ? parts.reduce((a, b) => a * 60 + b, 0) : NaN;
}
function reloadKeepingPlace(video, t) {
  // after a reload, come back to the same video and moment instead of an empty player
  if (!video && ytReady && playingVideo()) { video = playingVideo(); t = ytPlayer.getCurrentTime(); }
  if (video) sessionStorage.setItem('resume', JSON.stringify({ video, t: t || 0 }));
  location.reload();
}
async function newSegment(btn) {
  const video = btn.dataset.video;
  const here = ytReady && playingVideo() === video;
  const res = await fetch('add', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video, entry: +btn.dataset.entry,
                           at: here ? Math.floor(ytPlayer.getCurrentTime()) : null,
                           duration: here ? Math.floor(ytPlayer.getDuration()) : null }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  reloadKeepingPlace(video, (await res.json()).start);
}
async function postSegment(row, url, extra) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: row.dataset.video, entry: +row.dataset.entry, seg: +row.dataset.seg,
                           bounds: [[+row.dataset.start, +row.dataset.end]], ...extra }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  reloadKeepingPlace();
}
async function splitAt(btn) {
  const row = btn.closest('tr');
  const text = row.querySelector('input.at').value.trim();
  const at = parseTime(text);
  if (!Number.isFinite(at)) { alert(`"${text}" is not a time: use m:ss or h:mm:ss, e.g. 17:29`); return; }
  if (!(at > +row.dataset.start && at < +row.dataset.end)) {
    alert(`${text} is outside this segment (${fmt(+row.dataset.start)}–${fmt(+row.dataset.end)}): ` +
          'use the Split box of the segment that contains it, or press F while the video plays there');
    return;
  }
  await postSegment(row, 'split', { at });
}
async function moveSegment(btn) {
  await postSegment(btn.closest('tr'), 'move', {});
}
async function toggleNight(btn) {
  const row = btn.closest('tr');
  const res = await fetch('night', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: row.dataset.video, entry: +row.dataset.entry, seg: +row.dataset.seg,
                           bounds: [[+row.dataset.start, +row.dataset.end]] }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  const { night } = await res.json();
  btn.textContent = night ? 'night' : 'day';
  btn.classList.toggle('night', !!night);
}
async function setVehicle(video, select) {
  const res = await fetch('vehicle', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video, vehicle_type: +select.value }) });
  if (!res.ok) { alert((await res.json()).error); location.reload(); }
}
async function mergeSplit(video, entry, seg, bounds) {
  const res = await fetch('merge', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video, entry, seg, bounds }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  reloadKeepingPlace();
}
async function approveRest(btn) {
  const video = btn.closest('.video');
  for (const row of video.querySelectorAll('tr.seg:not(.approve):not(.reject):not(.applied)')) {
    if (!await decide(row.querySelector('.yes'), 'approve')) return;
  }
  const last = [...video.querySelectorAll('tr.seg')].pop();
  if (last) goToNextOpen(last);
}
async function rejectVideo(btn) {
  const video = btn.closest('.video');
  for (const row of video.querySelectorAll('tr.seg:not(.reject):not(.applied)')) {
    if (!await decide(row.querySelector('.no'), 'reject')) return;
  }
  const last = [...video.querySelectorAll('tr.seg')].pop();
  if (last) goToNextOpen(last);
}
function update() {
  const rows = [...document.querySelectorAll('tr.seg:not(.applied)')];
  const yes = rows.filter(r => r.classList.contains('approve')).length;
  const no = rows.filter(r => r.classList.contains('reject')).length;
  const open = rows.length - yes - no;
  document.getElementById('status').textContent =
    `${rows.length} segments to review: ${yes} approved, ${no} rejected, ${open} undecided`;
  document.getElementById('apply').disabled = yes === 0;
}
document.getElementById('apply').onclick = async () => {
  const rows = [...document.querySelectorAll('tr.seg:not(.applied)')];
  const yes = rows.filter(r => r.classList.contains('approve')).length;
  const open = rows.filter(r => !r.classList.contains('approve') && !r.classList.contains('reject')).length;
  if (!confirm(`Write ${yes} approved segment(s) to the mapping file?` +
               (open ? `\n${open} undecided segment(s) stay here for later.` : ''))) return;
  const btn = document.getElementById('apply');
  btn.disabled = true; btn.textContent = 'Applying…';
  const res = await fetch('apply', { method: 'POST' });
  const out = await res.json();
  document.getElementById('log').textContent = out.error || out.log.join('\\n');
  btn.textContent = 'Apply approved segments to the mapping';
  if (res.ok) setTimeout(() => location.reload(), 4000);
};
update();
const resume = JSON.parse(sessionStorage.getItem('resume') || 'null');
if (resume) {
  sessionStorage.removeItem('resume');
  history.scrollRestoration = 'manual';  // otherwise the browser's own scroll restore wins after loading
  const row = document.querySelector(`tr.seg[data-video="${resume.video}"]`);
  const scroll = () => row && row.closest('.video').scrollIntoView({ block: 'start' });
  scroll();
  window.addEventListener('load', () => setTimeout(scroll, 50));
  play(resume.video, resume.t);
}
</script>
<script src="https://www.youtube.com/iframe_api"></script>
</body></html>"""


def review(plan_path, port=None):
    """Local page to approve or reject every proposed segment, then apply only the approved ones."""
    from flask import Flask, jsonify, render_template_string, request, send_from_directory
    from markupsafe import Markup, escape

    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    app = Flask(__name__)

    def load():
        with open(plan_path) as f:
            return json.load(f)

    def save(plan):
        tmp = plan_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(plan, f, indent=1)
        os.replace(tmp, plan_path)

    # every change holds the plan lock from reading to saving, so a channel run writing the same plan
    # (adding analysed videos while you review) never interleaves with it
    held = {}

    @app.before_request
    def take_lock():
        if request.method == 'POST':
            held[id(request._get_current_object())] = lock = plan_lock(plan_path)
            lock.__enter__()

    @app.teardown_request
    def release_lock(exc):
        lock = held.pop(id(request._get_current_object()), None)
        if lock:
            lock.__exit__(None, None, None)

    @app.route('/')
    def index():
        plan = load()
        descriptions = {p['video']: describe(p['video']) for p in plan if 'entries' in p}
        return render_template_string(REVIEW_HTML, plan=plan, max_merge_gap=MAX_MERGE_GAP_S,
                                      descriptions={v: d for v, d in descriptions.items() if d}, countries=COUNTRIES,
                                      vehicle_types=VEHICLE_TYPES)

    def lookup(plan, d):
        """The video and segment list the page refers to, or (None, None) when the page shows an older plan."""
        p = next((p for p in plan if p['video'] == d['video']), None)
        try:
            segs = p['entries'][d['entry']]['segments']
            shown = [[s['start'], s['end']] for s in segs[d['seg']:d['seg'] + len(d['bounds'])]]
        except (TypeError, IndexError, KeyError):
            return None, None
        return (p, segs) if shown == d['bounds'] else (None, None)

    stale = 'The plan changed since this page was loaded. Reload the page and try again.'

    def describe(video_id):
        """Description (escaped, with its timestamps as play links) and chapters from the saved metadata."""
        path = os.path.join(plan_dir, f'{video_id}_meta.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            meta = json.load(f)
        text, parts, last = meta.get('description') or '', [], 0
        for m in re.finditer(r'(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])', text):
            h, mins, secs = m.groups()
            parts += [escape(text[last:m.start()]),
                      Markup('<a href="#" onclick="play(\'{}\', {}); return false;">{}</a>').format(
                          video_id, int(h or 0) * 3600 + int(mins) * 60 + int(secs), m.group(0))]
            last = m.end()
        parts.append(escape(text[last:]))
        chapters = [(int(start or 0), title) for title, start in meta.get('chapters') or []]
        return {'html': Markup('').join(parts), 'chapters': chapters} if text or chapters else None

    @app.route('/bounds', methods=['POST'])
    def bounds():
        """New start and end for a segment (the A, S and D keys); it may not overlap the video's other segments."""
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None:
            return jsonify(error=stale), 409
        s = segs[d['seg']]
        if s.get('applied'):
            return jsonify(error='this segment is already in the mapping'), 409
        start, end = int(d['start']), int(d['end'])
        if not 0 <= start < end:
            return jsonify(error='the start must be before the end'), 400
        for e in p['entries']:
            for o in e['segments']:
                if o is not s and start < o['end'] and o['start'] < end:
                    return jsonify(error=f"that would overlap this video's segment {o['start'] // 60}:"
                                         f"{o['start'] % 60:02d}–{o['end'] // 60}:{o['end'] % 60:02d}"), 400
        s['start'], s['end'] = start, end
        save(plan)
        return jsonify(start=start, end=end)

    @app.route('/add', methods=['POST'])
    def add_segment():
        """A new segment under a locality group, from the given time (or after the video's last segment)
        up to the video's next segment or its end."""
        d = request.get_json()
        plan = load()
        p = next((p for p in plan if p['video'] == d['video'] and 'entries' in p), None)
        if p is None or d['entry'] >= len(p['entries']):
            return jsonify(error=stale), 409
        e = p['entries'][d['entry']]
        taken = sorted((s['start'], s['end']) for x in p['entries'] for s in x['segments'])
        start = int(d['at']) if d.get('at') is not None else max((b for _, b in taken), default=0)
        if any(a <= start < b for a, b in taken):
            return jsonify(error=f'{start // 60}:{start % 60:02d} is inside an existing segment: split that '
                                 'segment instead, or play a part of the video no segment covers'), 400
        duration = d.get('duration')
        meta_path = os.path.join(plan_dir, f"{p['video']}_meta.json")
        if not duration and os.path.exists(meta_path):
            with open(meta_path) as f:
                duration = json.load(f).get('duration')
        ends = [a for a, _ in taken if a > start] + ([int(duration)] if duration else [])
        if not ends:
            return jsonify(error="the video's length is unknown: play it first, then add the segment"), 400
        end = min(ends)
        if end - start < 1:
            return jsonify(error='no room for a new segment there'), 400
        before = [s for s in e['segments'] if s['end'] <= start]
        night = before[-1]['night'] if before else (e['segments'][0]['night'] if e['segments'] else 0)
        e['segments'] = sorted(e['segments'] + [{'start': start, 'end': end, 'night': night}],
                               key=lambda s: s['start'])
        save(plan)
        return jsonify(start=start, end=end)

    @app.route('/split', methods=['POST'])
    def split():
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None:
            return jsonify(error=stale), 409
        s = segs[d['seg']]
        if s.get('applied'):
            return jsonify(error='this segment is already in the mapping'), 409
        at = int(d['at'])
        if not s['start'] < at < s['end']:
            return jsonify(error='the split time must be inside the segment'), 400
        segs[d['seg']:d['seg'] + 1] = [{'start': s['start'], 'end': at, 'night': s['night']},
                                       {'start': at, 'end': s['end'], 'night': s['night']}]
        save(plan)
        return jsonify(ok=True)

    @app.route('/move', methods=['POST'])
    def move():
        """Put a segment under a new locality group of the same video, whose locality is then picked."""
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None:
            return jsonify(error=stale), 409
        s = segs[d['seg']]
        if s.get('applied'):
            return jsonify(error='this segment is already in the mapping'), 409
        s.pop('decision', None)
        segs.pop(d['seg'])
        if not segs:
            p['entries'].pop(d['entry'])
        p['entries'].append({'locality': None, 'guessed': False, 'segments': [s]})
        save(plan)
        return jsonify(ok=True)

    def label(locality, state, country):
        return ', '.join(x for x in (locality, state, country) if x)

    @app.route('/localities')
    def localities():
        """Mapping localities whose name or alternative name contains the query, as canonical rows."""
        q = request.args.get('q', '').strip().lower()
        if len(q) < 2:
            return jsonify([])
        df = add_video.load_csv(add_video.FILE_PATH)
        hits = []
        for _, row in df.iterrows():
            names = [row['locality']] + ([a.strip() for a in row['locality_aka'][1:-1].split(',')]
                                         if isinstance(row['locality_aka'], str) else [])
            lower = [n.lower() for n in names if isinstance(n, str)]
            if any(q in n for n in lower):
                # exact name or alias first ("la" -> Los Angeles), then names starting with it, then the rest
                rank = 0 if q in lower else 1 if any(n.startswith(q) for n in lower) else 2
                state = row['state'] if isinstance(row['state'], str) else None
                hits.append((rank, row['locality'], state, row['country'], False))
        # localities typed in for other videos of this plan that Apply has not created yet
        in_mapping = {(h[1].lower(), h[3]) for h in hits}
        pending = {tuple(e['locality']) for p in load() for e in p.get('entries', []) if e.get('locality')}
        for locality, state, country in pending:
            low = locality.lower()
            if q in low and (low, country) not in in_mapping and add_video.get_existing_locality_row(
                    df, locality, state, country)[0] is None:
                hits.append((0 if low == q else 1 if low.startswith(q) else 2, locality, state, country, True))
        hits.sort(key=lambda h: (h[0], h[1], h[2] or '', h[3]))  # some localities have no state
        return jsonify([{'label': label(*h[1:4]) + (' (new: created on Apply)' if h[4] else ''),
                         'value': list(h[1:4])} for h in hits[:30]])

    @app.route('/locality', methods=['POST'])
    def set_locality():
        d = request.get_json()
        plan = load()
        p = next((p for p in plan if p['video'] == d['video'] and 'entries' in p), None)
        if p is None or d['entry'] >= len(p['entries']):
            return jsonify(error=stale), 409
        e = p['entries'][d['entry']]
        if any(s.get('applied') for s in e['segments']):
            return jsonify(error='these segments are already in the mapping'), 409
        locality, state, country = (str(x or '').strip() for x in d['locality'])
        state = state or None
        if not locality or country not in COUNTRIES:
            return jsonify(error='give a locality and pick a country'), 400
        same = [o for o in p['entries'] if o is not e and o.get('locality') == [locality, state, country]]
        if same:  # this video already has a group for that locality: join it
            same[0]['segments'] = sorted(same[0]['segments'] + e['segments'], key=lambda s: s['start'])
            p['entries'].remove(e)
            save(plan)
            return jsonify(ok=True, merged=True)
        e['locality'], e['guessed'] = [locality, state, country], False
        save(plan)
        idx, _ = add_video.get_existing_locality_row(add_video.load_csv(add_video.FILE_PATH), locality, state, country)
        return jsonify(ok=True, new=idx is None)

    @app.route('/night', methods=['POST'])
    def night():
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None:
            return jsonify(error=stale), 409
        seg = segs[d['seg']]
        if seg.get('applied'):
            return jsonify(error='this segment is already in the mapping'), 409
        seg['night'] = 1 - int(seg['night'])
        save(plan)
        return jsonify(night=seg['night'])

    @app.route('/vehicle', methods=['POST'])
    def vehicle():
        d = request.get_json()
        plan = load()
        p = next((p for p in plan if p['video'] == d['video'] and 'entries' in p), None)
        if p is None:
            return jsonify(error=stale), 409
        if any(s.get('applied') for e in p['entries'] for s in e['segments']):
            return jsonify(error='this video is already in the mapping; change its vehicle in the mapping file'), 409
        if d['vehicle_type'] not in VEHICLE_TYPES:
            return jsonify(error='unknown vehicle type'), 400
        p['vehicle_type'] = d['vehicle_type']
        save(plan)
        return jsonify(ok=True)

    @app.route('/merge', methods=['POST'])
    def merge():
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None:
            return jsonify(error=stale), 409
        i = d['seg']
        a, b = segs[i], segs[i + 1]
        if a.get('applied') or b.get('applied'):
            return jsonify(error='one of these segments is already in the mapping'), 409
        if a['night'] != b['night'] or b['start'] - a['end'] > MAX_MERGE_GAP_S:
            return jsonify(error='only splits between neighbouring segments with the same time of day '
                                 'can be removed'), 400
        segs[i:i + 2] = [{'start': a['start'], 'end': b['end'], 'night': a['night']}]
        p.setdefault('removed_splits', []).append(a['end'])  # false cuts, kept for tuning the detector
        save(plan)
        return jsonify(ok=True)

    @app.route('/sheet/<video_id>.jpg')
    def sheet(video_id):
        return send_from_directory(plan_dir, f'{video_id}.jpg')

    @app.route('/decide', methods=['POST'])
    def decide():
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None:
            return jsonify(error=stale), 409
        seg = segs[d['seg']]
        if seg.get('applied'):
            return jsonify(error='this segment is already in the mapping'), 409
        seg['decision'] = d['decision']
        save(plan)
        return jsonify(ok=True)

    @app.route('/apply', methods=['POST'])
    def apply():
        plan = load()
        log = []
        for p in plan:
            for e in p.get('entries', []):
                if not e.get('locality'):
                    if any(s.get('decision') == 'approve' and not s.get('applied') for s in e['segments']):
                        log.append(f"{p['video']}: locality not set, approved segments not added")
                    continue
                done = 0
                for s in e['segments']:
                    if s.get('decision') != 'approve' or s.get('applied'):
                        continue
                    try:
                        apply_segments(p['video'], *e['locality'], [s], p['vehicle_type'],
                                       p['upload_date'], p['channel_id'])
                    except Exception as ex:
                        log.append(f"{p['video']} {s['start']}-{s['end']}: not added: {ex}")
                        continue
                    s['applied'] = True
                    done += 1
                    save(plan)  # after every segment, so a failure part way never re-applies one
                if done:
                    log.append(f"{p['video']}: {done} segment(s) added to {e['locality'][0]}")
        return jsonify(log=log or ['nothing to add'])

    port = port or 5000 + random.randint(0, 999)
    Timer(1.25, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    app.run(port=port, debug=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--channel', metavar='URL', help='propose segments for every new video of a channel')
    ap.add_argument('--country', help='country of the channel as written in the mapping, for locality guesses')
    ap.add_argument('--out', help='output folder (default: _output/proposals/<channel>)')
    ap.add_argument('--pause', type=float, default=15, help='seconds between downloads')
    ap.add_argument('--limit', type=int, help='stop after this many new proposals (one batch to review)')
    ap.add_argument('--no-download', action='store_true',
                    help='propose whole videos from YouTube Data API metadata, without downloading or analysing')
    ap.add_argument('--review', metavar='PLAN_JSON', help='open the review page for a plan file')
    args = ap.parse_args()
    if args.review:
        review(args.review)
    elif args.channel:
        name = re.sub(r'\W+', '_', re.sub(r'^.*youtube\.com/(?:channel/)?@?', '', args.channel).split('/')[0])
        out = process_channel(args.channel, args.country, args.out or os.path.join('_output/proposals', name),
                              args.pause, args.limit, download=not args.no_download)
        if out:
            with open(out) as f:
                if any('entries' in p for p in json.load(f)):
                    print(f'Review with: python propose_segments.py --review {out}')
    else:
        ap.print_help()
