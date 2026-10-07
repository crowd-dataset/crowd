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
import shutil
import subprocess
import tempfile
import time
import unicodedata
import webbrowser
from collections import Counter
from functools import lru_cache, wraps
from threading import Timer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import pycountry
import numpy as np
import pandas as pd
import requests
from geopy.exc import (GeocoderAuthenticationFailure, GeocoderInsufficientPrivileges, GeocoderQuotaExceeded,
                       GeocoderRateLimited, GeopyError)
from geopy.geocoders import Nominatim

import add_video
import common

FPS = 2                 # analysis frames per second
W, H = 160, 90          # analysis frame size

PARAMS = {
    'moving_flow': 0.25,      # median flow (px per analysis frame) above which the camera is moving
    'smooth_s': 5,            # rolling median window for motion
    'edge_stop_s': 0,         # stationary stretch at the start/end longer than this is trimmed: parked before
                              # setting off or after arriving (labelled videos keep none of it)
    'mid_stop_s': 180,        # stationary stretch inside a video longer than this is cut out; shorter ones
                              # are waits at traffic lights and in traffic, and stay
    'edge_burst_s': 15,       # moving for less than this between stops at the start/end is shuffling in the
                              # parking spot or fixing the camera: the parked stretch goes on through it...
    'edge_parked_s': 5,       # ...when the video opens (or closes) standing still at least this long
    'cut_ncc': 0.3,           # a frame this uncorrelated with the previous one is a hard cut...
    'cut_neighbour_ncc': 0.6,  # ...when the frames either side of it are this coherent
    'bridge_dip': 0.5,        # a cut whose sky darkens below this share of its surroundings is a bridge
    'cut_confirm_ratio': 15,  # a cut stands only with a full-frame-rate jump this strong within a second
    'cut_pan_ncc': None,      # a cut that matches this well once shifted sideways is a corner (not yet calibrated)
    'blank_std': 6.0,         # frame this flat is blank or a title card
    # night: the sky darker than this, or the street darker than night_ground. On 19 night and 20 day segments
    # labelled in CharruaNYC (2026-10-07) this finds 12 of the night ones (the sky alone at 95: 10) and calls no day
    # segment night (darkest day sky 112, street 75); 7 labelled night look like daylight in sky and street
    'night_sky': 100.0,
    'night_ground': 65.0,
    'night_window_s': 60,     # night/day must hold this long to count as a transition
    # dark this long or shorter with daylight before and after is a bridge, tunnel or elevated railway (10 min under
    # the 7 train on Roosevelt Ave): night falls once and does not lift again within a drive
    'night_inside_day_s': 3 * 3600,
    'night_edge_s': 300,
    # possibly dusk (a hint in the note, never a label): saturation above this or lit orange lamps above this % of
    # the picture; on 8 bright segments labelled night and 20 day ones (CharruaNYC) it flags 5 and no day one
    'dusk_saturation': 125,
    'dusk_lamps': 0.015,      # dark this long or shorter at the start or end, daylight after or before: a structure
    'min_segment_s': 30,      # drop proposed segments shorter than this
    'driver_window_s': 60,    # a big face in more than driver_share of the frames over this window means the
    'driver_share': 0.1,      # camera faces the driver (found in only 10-40% of its frames: sunglasses, small
                              # picture; road footage the user kept: 0% in the median minute)...
    'driver_min_s': 10,       # ...and such a stretch this long is cut out
    # camera angle: a moving frame whose right and left sides flow apart by less than angle_spread times the
    # overall motion looks sideways (forward cameras: median 1.1-1.5; a side-facing camera: -0.1)
    'angle_spread': 0.3,
    'angle_share': 0.6,       # sideways in this share of the moving frames over a minute: cut that stretch
    'shake_jitter': 1.0,      # spread of the vertical shift over 10 s (normal: 0.02-0.46; shaking: 1.8)
    # rural driving: YOLO sees fewer than this many cars, people, traffic lights, stop signs and hydrants per frame
    # over a minute (labelled towns: 1.0-7 per frame, a rural drive through the Poconos: 0-0.4); day only
    'urban_objects': 0.6,
    # highway driving: fewer than this many people, bikes, traffic lights, stop signs and hydrants per frame over a
    # minute while cars still pass (Kensington, Philadelphia: 0.33-3.5; I-81 through Scranton: 0-0.08); day only
    # ponytail: calibrated on two videos; quiet suburbs with no one about may need it lower
    'street_objects': 0.15,
    # rural and highway stretches are long (I-81: 37 min); a quiet town street goes by in 1-2 (Kutztown, Bellefonte)
    'quiet_min_s': 180,
    'problem_min_s': 60,      # angle and shaking stretches shorter than this are kept (turns, bumps: ~30 s)
}
ONE_PLACE = {**PARAMS, 'quiet_min_s': float('inf')}  # no rural or highway check: a drive within one town
VIDEO_SIDEWAYS_SHARE = 0.5   # sideways in more than this share of all moving frames: the video is excluded


# pytubefix downloads now fail YouTube's PoToken check; a current yt-dlp (with node installed) does not
YT_DLP = os.environ.get('YT_DLP', 'yt-dlp')


class BotCheck(RuntimeError):
    """YouTube wants a signed-in session; further requests will fail until it lifts."""


def _yt_dlp(*args):
    # opt-in for when YouTube asks to sign in: YT_DLP_COOKIES=chrome (or firefox, ...) uses that browser's login,
    # or a path to a cookies.txt exported from a logged-in browser
    jar = os.environ.get('YT_DLP_COOKIES')
    cookies = (['--cookies', jar] if os.path.isfile(jar) else ['--cookies-from-browser', jar]) if jar else []
    # yt-dlp solves YouTube's JavaScript challenge only with a JS runtime and uses just deno unless told,
    # so offer node: without it YouTube hands out thumbnails instead of the video
    js = ['--js-runtimes', 'node'] if shutil.which('node') else []
    r = subprocess.run([YT_DLP, '--no-warnings', *js, *cookies, *args], capture_output=True, text=True)
    if r.returncode:
        msg = (r.stderr.strip().splitlines() or [f'yt-dlp exited with {r.returncode}'])[-1]
        raise (BotCheck if 'not a bot' in r.stderr else RuntimeError)(msg)
    return r.stdout


def _url(video_id):
    return f'https://www.youtube.com/watch?v={video_id}'


def fetch_metadata(video_id):
    m = json.loads(_yt_dlp('-J', _url(video_id)))
    d = m.get('upload_date')  # YYYYMMDD, in UTC
    if m.get('timestamp'):
        d = add_video.youtube_day(m['timestamp']).strftime('%Y%m%d')
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
    ground = []  # brightness of the road and buildings: dark streets under a still-light sky are night too
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
        ground.append(float(f[band].mean()))
        prev, prev_thumb = f, thumb
    return {k: np.array(v) for k, v in
            (('motion', motion), ('ncc', ncc), ('pan', pan), ('std', std), ('sky', sky), ('ground', ground),
             ('face', face),
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


def _rolling_mean(x, n):
    """Mean over a centred window of n frames; near the ends, over the frames that exist rather than padding
    with zeros, which would make the first and last half-window look emptier than they are."""
    k = np.ones(n)
    return np.convolve(x, k, mode='same') / np.convolve(np.ones(len(x)), k, mode='same')


def _jitter(sig):
    """How much the picture's vertical shift varies over 10 s: a steady camera barely, a shaking one a lot."""
    y = sig['shift_y']
    n = 10 * FPS
    return np.sqrt(np.maximum(_rolling_mean(y * y, n) - _rolling_mean(y, n) ** 2, 0))


def _sideways(sig, p):
    """Moving frames, and moving frames whose flow does not spread out from the middle (camera aimed sideways)."""
    moving = sig['motion'] > 2 * p['moving_flow']
    return moving, moving & (sig['spread'] / (sig['motion'] + 0.05) < p['angle_spread'])


def footage_problem(sig, p=PARAMS):
    """Why a whole video is unusable from its camera, or None (signals saved before these checks: None)."""
    if 'shift_y' in sig and np.median(_jitter(sig)) > p['shake_jitter']:
        return 'the camera shakes too much'
    if 'spread' in sig:
        moving, sideways = _sideways(sig, p)
        if moving.sum() >= 60 * FPS and sideways.sum() / moving.sum() > VIDEO_SIDEWAYS_SHARE:
            return f'the camera does not face forward ({sideways.sum() / moving.sum():.0%} of the driving)'
    return None


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
    # half a second apart, frames under an elevated railway, in a corner or past flickering light differ like a
    # cut; at the full frame rate a real cut is a jump between two frames (see skip_events). Videos analysed
    # with it keep only the cuts such a jump confirms within a second.
    if sig.get('skips') is not None:
        jumps = np.asarray(sig['skips']).reshape(-1, 3)
        jumps = jumps[jumps[:, 2] >= p['cut_confirm_ratio'], 0]
        for i in np.flatnonzero(cut):
            if not (np.abs(jumps - i / FPS) <= 1).any():
                cut[i] = False

    keep = ~blank
    notes = []
    stopped = ~moving
    for order in (slice(None), slice(None, None, -1)):  # the start, then the end read backwards
        runs = _runs(stopped[order])
        if runs and runs[0][0] == 0 and (runs[0][1] - runs[0][0]) / FPS >= p['edge_parked_s']:
            i = 0
            while i + 1 < len(runs) and (runs[i + 1][0] - runs[i][1]) / FPS < p['edge_burst_s']:
                i += 1
            stopped[order][:runs[i][1]] = True
    for a, b in _runs(stopped):
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
        driver = _rolling_mean((sig['face'] > 0).astype(float), p['driver_window_s'] * FPS) > p['driver_share']
        for a, b in _runs(driver):
            if (b - a) / FPS >= p['driver_min_s']:
                keep[a:b] = False
                notes.append(f'camera on the driver {a / FPS:.0f}-{b / FPS:.0f}s')
    problems = []
    if 'spread' in sig:
        moving, sideways = _sideways(sig, p)
        moving_share = _rolling_mean(moving.astype(float), 60 * FPS)
        share = _rolling_mean(sideways.astype(float), 60 * FPS) / np.maximum(moving_share, 1e-6)
        # judged only over minutes with enough driving; a stop says nothing about where the camera points
        problems.append(('camera not facing forward', (share > p['angle_share']) & (moving_share > 0.3)))
    if 'urban' in sig:
        per_frame = np.repeat(sig['urban'], URBAN_EVERY_S * FPS)[:n]
        per_frame = np.pad(per_frame, (0, n - len(per_frame)), mode='edge')
        day = _rolling_median(sig['sky'], 60 * FPS) >= p['night_sky']  # YOLO sees little at night anywhere
        problems.append(('rural driving', (_rolling_mean(per_frame, 60 * FPS) < p['urban_objects']) & day))
        if 'street' in sig:  # signals saved before the highway check have no street trace
            street = np.repeat(sig['street'], URBAN_EVERY_S * FPS)[:n]
            street = np.pad(street, (0, n - len(street)), mode='edge')
            problems.append(('highway driving', (_rolling_mean(street, 60 * FPS) < p['street_objects']) & day))
    if 'shift_y' in sig:
        problems.append(('camera shaking', _rolling_median(_jitter(sig), 30 * FPS) > p['shake_jitter']))
    for label, bad in problems:
        min_s = p['quiet_min_s'] if label in ('highway driving', 'rural driving') else p['problem_min_s']
        for a, b in _runs(bad):
            if (b - a) / FPS >= min_s:
                keep[a:b] = False
                notes.append(f'{label} {a / FPS:.0f}-{b / FPS:.0f}s')

    window = p['night_window_s'] * FPS
    dark = sig['sky'] < p['night_sky']
    if 'ground' in sig:  # signals saved before the street brightness have the sky only
        dark = dark | (sig['ground'][:len(dark)] < p['night_ground'])
    night = _rolling_median(dark.astype(float), window) > 0.5
    # brief dark or bright spells (underpasses, a flickering sky near the threshold) must not split a segment
    night = _drop_short_runs(night, window)
    # a switch at an edit cut is footage stitched together (a night drive after a day drive): it stays night; under a
    # bridge or the El the dark comes and goes without one
    edits = np.r_[np.flatnonzero(cut), np.array(skip_times(sig)) * FPS]
    for a, b in _runs(night):
        stitched = any(abs(e - x) <= window for e in edits for x in (a, b))
        inside = a > 0 and b < n and (b - a) / FPS <= p['night_inside_day_s']
        # a short dark start or end with daylight on the other side is the El or a bridge too (Marble Hill under the
        # 1 train, Roosevelt Ave under the 7); night falls over many minutes
        edge = (a == 0) != (b == n) and (b - a) / FPS <= p['night_edge_s']
        if (inside or edge) and not stitched:
            night[a:b] = False

    segments = []
    for a, b in _runs(keep):
        bounds = [a] + [i for i in range(a + 1, b) if cut[i] or night[i] != night[i - 1]] + [b]
        for s, e in zip(bounds, bounds[1:]):
            if (e - s) / FPS >= p['min_segment_s']:
                segments.append({'start': int(np.ceil(s / FPS)), 'end': int(e // FPS),
                                 'night': int(round(night[s:e].mean()))})
    # dusk with street lights on looks like day to the brightness checks (the camera brightens it): only a hint
    if 'saturation' in sig:
        for seg in segments:
            first = seg['start'] // DUSK_EVERY_S
            span = slice(first, max(seg['end'] // DUSK_EVERY_S, first + 1))
            sat, lamps = sig['saturation'][span], sig['lamps'][span]
            if not seg['night'] and len(sat) and (np.median(sat) > p['dusk_saturation']
                                                  or np.median(lamps) > p['dusk_lamps']):
                notes.append(f"possibly dusk {seg['start']}-{seg['end']}s: street lights may be on, check night")
    return segments, notes


DUSK_EVERY_S = 10  # one colour frame this often for the dusk hint


def dusk_colours(path):
    """(saturation, % of lit orange lamp pixels) of a colour frame every DUSK_EVERY_S seconds."""
    cap = cv2.VideoCapture(path)
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) * DUSK_EVERY_S))
    sat, lamps = [], []
    for f in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (W, H))
        sat.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()))
        b, g, r = (img[..., k].astype(int) for k in range(3))
        lamps.append(float(((r > 200) & (g > 120) & (b < 110)).mean() * 100))
    return np.array(sat), np.array(lamps)


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


URBAN_EVERY_S = 5  # one YOLO frame this often
HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_PYTHON = os.path.join(HERE, '.venv', 'bin', 'python')
URBAN_CODE = """
import sys, cv2, torch
from ultralytics import YOLO
model, cap = YOLO('yolo11x.pt'), cv2.VideoCapture(sys.argv[1])
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # the Mac's GPU: 3x faster, far less battery
step = max(1, round(cap.get(cv2.CAP_PROP_FPS) * float(sys.argv[2])))
for f in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        break
    cls = model(img, imgsz=448, conf=0.25, device=device, verbose=False)[0].boxes.cls.tolist()
    print(sum(c in (0, 1, 2, 3, 5, 9, 10, 11) for c in cls), sum(c in (0, 1, 9, 10, 11) for c in cls), flush=True)
"""


def urban_objects(path):
    """[(all, street)] objects YOLO finds in a frame every URBAN_EVERY_S seconds: all are people, bikes, cars, buses,
    traffic lights, hydrants and stop signs (few for minutes: rural driving), street those without the cars and
    buses (few while cars pass: highway driving). YOLO runs in the project's .venv (torch); None without it."""
    if not os.path.exists(YOLO_PYTHON):
        return None
    r = subprocess.run([YOLO_PYTHON, '-c', URBAN_CODE, path, str(URBAN_EVERY_S)], capture_output=True, text=True,
                       cwd=HERE)
    counts = [[int(x) for x in line.split()] for line in r.stdout.splitlines() if line.strip()]
    if r.returncode or not counts:
        print(f'  YOLO failed, no rural or highway check: {r.stderr.strip()[-300:]}', flush=True)
        return None
    return np.array(counts, float)


def skip_events(path):
    """(time s, scene score, ratio to the local median) of each frame that changes far more from the frame before
    than the frames around it do: an edit cut. At the full frame rate consecutive frames of a drive are nearly
    alike, so even a cut that only skips a wait at the same intersection stands out (cars jump, the light turns
    green between two frames), which half-second analysis frames cannot show. Candidates only: ratio >= 8."""
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-an', '-vf',
           "scale=160:-2,select='gte(scene,0)',metadata=print:file=-", '-f', 'null', '-']
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    t = np.array([float(x) for x in re.findall(r'pts_time:([\d.]+)', out)])
    score = np.array([float(x) for x in re.findall(r'lavfi\.scene_score=([\d.]+)', out)])
    n = min(len(t), len(score))
    if n < 3:
        return np.zeros((0, 3))
    t, score = t[:n], score[:n]
    # median over a second either side, from a sorted sliding window
    k = 15
    padded = np.pad(score, k, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1)
    ratio = score / np.maximum(np.median(windows, axis=1), 0.002)
    keep = (ratio >= 8) & (score >= 0.02)
    return np.column_stack([t[keep], score[keep], ratio[keep]])


# an edit cut that skips footage: a frame changing SKIP_RATIO times more than the frames around it and by at
# least SKIP_SCORE, alone (livestream stutter and intro montages jump many times in a row), and not in the first
# SKIP_INTRO_S (title cards) or last SKIP_OUTRO_S (end screens). Checked on whole approved videos: these keep
# the waits cut out at red lights and the cuts to other streets, and none of the stutter or fades.
SKIP_RATIO, SKIP_SCORE, SKIP_ALONE_S, SKIP_INTRO_S, SKIP_OUTRO_S = 25, 0.06, 10, 30, 10
SKIP_STREAK = 10       # this many analysed videos of a channel in a row with skips: the channel edits its drives


def skip_times(sig):
    """Seconds at which the video skips footage (edit cuts), from the skip candidates that are strong and alone."""
    ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
    out = []
    for t, score, ratio in ev:
        if ratio < SKIP_RATIO or score < SKIP_SCORE:
            continue
        gap = np.abs(ev[:, 0] - t)
        if not ((gap > 0.5) & (gap <= SKIP_ALONE_S)).any():  # frames right at a cut can both jump
            out.append(float(t))
    return out


def footage_skip(sig):
    """Where the video skips footage (an edit cut), as m:ss, else None. Waiting at intersections is what gets
    cut most and what the dataset needs, so such a video is left out whole."""
    end = len(sig['motion']) / FPS
    t = next((t for t in skip_times(sig) if SKIP_INTRO_S < t < end - SKIP_OUTRO_S), None)
    return None if t is None else f'{int(t) // 60}:{int(t) % 60:02d}'


GPS_PROBE_S = (10, 45, 90)  # where the first frames are read for a GPS overlay; without one there, none is read
GPS_EVERY_S = 30            # then one frame this often
# "N40.12345 W75.12345", "40.12345N 75.12345W", "40.123456, -75.123456" as dashcams print them
GPS_HEMI = re.compile(r'([NS])\s*(\d{1,2}[.,]\d{3,})\D{0,4}?([EW])\s*(\d{1,3}[.,]\d{3,})'
                      r'|(\d{1,2}[.,]\d{3,})\s*°?\s*([NS])\W{0,4}(\d{1,3}[.,]\d{3,})\s*°?\s*([EW])')
GPS_SIGNED = re.compile(r'(?<![\d.])(-?\d{1,2}\.\d{4,})\s*[,;]?\s+(-?\d{1,3}\.\d{4,})(?![\d.])')


def parse_gps(text):
    """(lat, lon) of the first coordinates in a frame's text, else None."""
    if m := GPS_HEMI.search(text):
        g = m.groups()
        ns, la, ew, lo = g[:4] if g[0] else (g[5], g[4], g[7], g[6])
        lat, lon = float(la.replace(',', '.')), float(lo.replace(',', '.'))
        lat, lon = -lat if ns == 'S' else lat, -lon if ew == 'W' else lon
    elif m := GPS_SIGNED.search(text):
        lat, lon = float(m.group(1)), float(m.group(2))
    else:
        return None
    return (lat, lon) if abs(lat) <= 90 and abs(lon) <= 180 and (lat, lon) != (0, 0) else None


def _frame_text(stream, t):
    """What tesseract reads in the frame at t seconds of a stream (url, HTTP headers) ('' when nothing)."""
    url, headers = stream
    png = subprocess.run(['ffmpeg', '-v', 'error', '-headers', headers, '-ss', str(t), '-i', url, '-frames:v', '1',
                          '-vf', 'format=gray', '-f', 'image2pipe', '-vcodec', 'png', '-'],
                         capture_output=True, timeout=120).stdout
    if not png:
        return ''
    return subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '11'], input=png, capture_output=True,
                          timeout=120).stdout.decode(errors='ignore')


def gps_track(video_id, duration):
    """[(t, lat, lon)] read from a GPS overlay in the picture, or None when the first frames show none. Frames
    come from the 720p stream (the 240p copy is too small to read), so only a few are fetched."""
    if not shutil.which('tesseract') or not duration:
        return None
    try:
        # YouTube refuses the stream (403) without the headers yt-dlp asked for it with
        m = json.loads(_yt_dlp('-j', '-f', 'bv*[height<=720][ext=mp4]/bv*[height<=720]', _url(video_id)))
        stream = m['url'], ''.join(f'{k}: {v}\r\n' for k, v in (m.get('http_headers') or {}).items())
        probe = [(t, parse_gps(_frame_text(stream, t))) for t in GPS_PROBE_S if t < duration]
        if not any(xy for _, xy in probe):
            return None
        track = [(t, *xy) for t, xy in probe if xy]
        track += [(t, *xy) for t in range(GPS_EVERY_S * 4, int(duration), GPS_EVERY_S)
                  if (xy := parse_gps(_frame_text(stream, t)))]
    except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        print(f'  GPS overlay not read: {e}', flush=True)
        return None
    track.sort()
    # a misread digit puts a point far off: keep points a car could have reached from the one before
    kept = track[:1]
    for t, lat, lon in track[1:]:
        if _km(kept[-1][1], kept[-1][2], lat, lon) <= 1 + (t - kept[-1][0]) / 3600 * MAX_KMH:
            kept.append((t, lat, lon))
    return kept if len(kept) >= 2 else None


def split_gps(p, df, out_dir):
    """Split an untouched one-group proposal into the localities its GPS overlay passes through. Returns whether
    it was split."""
    path = os.path.join(out_dir, f"{p['video']}_signals.npz")
    if len(p.get('entries', [])) != 1 or not os.path.exists(path):
        return False
    e, sig = p['entries'][0], np.load(path)
    if 'gps' not in sig or not e['segments'] or (e.get('locality') and not e.get('guessed')) \
            or any(s.get('decision') or s.get('applied') for s in e['segments']):
        return False
    countries = {common.get_iso2_country_code(common.correct_country(c)): c for c in df['country'].dropna().unique()}
    countries.pop(None, None)
    track = sig['gps'].tolist()
    stretches = []
    for i, (t, lat, lon) in enumerate(track):
        h = _osm_reverse(round(lat, 3), round(lon, 3))
        loc = (_hit_locality(h, df, countries, (lat, lon)) if h else None) if _urban(lat, lon) else None
        a = 0 if i == 0 else (track[i - 1][0] + t) / 2  # a border half way between two readings
        if stretches and stretches[-1][2] == loc:
            continue
        if stretches:
            stretches[-1] = (stretches[-1][0], a, stretches[-1][2])
        stretches.append((a, float('inf'), loc))
    entries = split_by_stretches(e['segments'], stretches)
    if not entries:
        return False
    p['entries'] = entries
    p['note'] = (f"{p.get('note') or ''}; localities from the GPS overlay: "
                 + ', '.join(f"{loc[0] if loc else 'rural (left out)'} from {int(a) // 60}:{int(a) % 60:02d}"
                             for a, _, loc in stretches)).lstrip('; ')
    return True


def analyse_video(video_id, out_dir, meta=None):
    """Download a low resolution copy, propose segments, write a contact sheet; the copy is deleted."""
    meta = meta or fetch_metadata(video_id)
    work_dir = tempfile.mkdtemp(prefix='crowd_')
    path = download_low_res(video_id, work_dir)
    try:
        sig = signals(path)
        sig['skips'] = skip_events(path)
        sig['saturation'], sig['lamps'] = dusk_colours(path)
        urban = urban_objects(path)
        if urban is not None:
            sig['urban'], sig['street'] = urban[:, 0], urban[:, 1]
        gps = gps_track(video_id, meta['duration'])
        if gps:
            sig['gps'] = np.array(gps, float)
            print(f'  GPS overlay: {len(gps)} positions', flush=True)
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
                           r'|sped[- ]?up|speed(?:ed)?[- ]?up|fast[- ]?forward|\b\d(?:\.\d)?x speed\b|\bx\d speed\b'
                           r'|\bcrash|\baccident', re.I)
# highway driving and road trips; "via I-5" only passes along it on an otherwise urban drive
HIGHWAY_TITLE = re.compile(r'(?i:road ?trip|\bhighway\b|\bfreeway\b|\bhwy\b|\bmotorway\b|\bautobahn\b)'
                           r'|(?<![Vv]ia )(?:\b[Ii]nterstate[- ]?|\bI[- ]?|\bi[- ])\d{1,3}\b')
# walks and rides on boats are no vehicle type the dataset takes; a title that also says drive keeps the video
NOT_DRIVING_TITLE = re.compile(r'\b(?:walk(?:s|ing)?|stroll(?:ing)?|hik(?:e|ing)|on foot|(?:boat|ferry) ride)\b', re.I)
DRIVING_TITLE = re.compile(r'\bdriv(?:e|es|ing)\b|\bdashcam\b', re.I)


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
            day = add_video.youtube_day(sn['publishedAt']).strftime('%Y-%m-%d') if sn.get('publishedAt') else ''
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
    if hit:
        return f"highway driving or a road trip (title mentions '{hit.group(0)}')"
    hit = NOT_DRIVING_TITLE.search(meta.get('title') or '')
    if hit and not DRIVING_TITLE.search(meta.get('title') or ''):
        return f"not a drive (title mentions '{hit.group(0)}')"
    return None


def title_countries(title):
    """The countries of the mapping a title names, not inside a US state's name: "New Jersey" is not Jersey,
    "Atlanta, Georgia" not the country."""
    states = '|'.join(re.escape(s) for s in add_video.US_STATE_CODES)
    # a country's name before a US state is a town of that name: "Lebanon, Pennsylvania", "Peru, Indiana"
    title = re.sub(rf"\b[A-Z][\w'’ ]*?\s*,\s*(?:{states}|[A-Z]{{2}})\b", ' ', title or '', flags=re.I)
    title = re.sub(rf'\b(?:{states})\b', ' ', title, flags=re.I)
    return [c for c in COUNTRIES if re.search(rf'\b{re.escape(c)}\b', title)]


def guess_locality(title, country, df):
    """[locality, state, country] when the title names exactly one mapping locality of that country, else None.
    A drive "from A to B in C" with C a locality is in C: neighbourhoods of a big city ("from Hollywood to Venice
    in Los Angeles") are often namesakes of other localities."""
    df = df[df['country'] == country]
    area = DRIVE_AREA.search(title or '')
    return (area and _title_locality(area.group(1), df)) or _title_locality(title, df)


def _common_words():
    """Lower-case English words (the system word list), for place names that are also ordinary words."""
    try:
        with open('/usr/share/dict/words') as f:
            return {w.strip() for w in f if w[:1].islower()}
    except OSError:
        return set()


COMMON_WORDS = _common_words()
ADDRESS_LINE = re.compile(r',\s*[A-Z]{2}\s+\d{5}\b|https?://|www\.|\b(?:P\.?\s?O\.? Box|Ste\.?|Suite)\b', re.I)


def channel_boilerplate(descriptions):
    """Lines a channel repeats under many of its videos (sign-offs, links, a mailing address): they say nothing
    about where one video was filmed."""
    counts = Counter(line for d in descriptions for line in {x.strip() for x in (d or '').splitlines() if x.strip()})
    return {line for line, n in counts.items() if n >= max(3, 0.2 * len(descriptions))}


def description_locality(description, df, boilerplate):
    """[locality, state, country] when a video description, without the channel's boilerplate, links and
    addresses, names exactly one locality of df, else None. A place name that is also an ordinary word
    ("Beach", "Canon" the camera, "Phoenix") only counts written with its state, as in "Phoenix, AZ"."""
    text = '\n'.join(line for line in (description or '').splitlines()
                     if line.strip() not in boilerplate and not ADDRESS_LINE.search(line))
    return _title_locality(text, df, exact_case=True, word_needs_state=True)


CHAPTER_START = re.compile(r'^\W*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)\W*(.+)$')
CHAPTER_END = re.compile(r'^\W*(.+?)\W*(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\W*$')


def video_chapters(meta):
    """[(start in s, name)] of a video's chapters: YouTube's own, else the description's timestamped lines
    ("12:30 Santa Monica" or "Santa Monica - 12:30"), when at least two run in increasing order."""
    chapters = [(int(start or 0), name) for name, start in meta.get('chapters') or [] if name]
    if not chapters:
        for line in (meta.get('description') or '').splitlines():
            if m := CHAPTER_START.match(line):
                h, mins, secs, name = m.groups()
            elif m := CHAPTER_END.match(line):
                name, h, mins, secs = m.groups()
            else:
                continue
            chapters.append((int(h or 0) * 3600 + int(mins) * 60 + int(secs), name.strip()))
    starts = [t for t, _ in chapters]
    return chapters if len(chapters) >= 2 and starts == sorted(set(starts)) else []


# a chapter named after a road ("Venice Blvd", "Rodeo Drive") is not in the place the road is named after
ROAD = re.compile(r'\b(?:blvd|boulevard|st|street|ave|avenue|rd|road|dr|drive|fwy|freeway|hwy|highway|expy|'
                  r'expressway|pkwy|parkway|ln|lane|way|terrace|strip|bridge|tunnel|pier|walk of fame)\b', re.I)
CHAPTER_PREFIX = re.compile(r'^(?:start(?:ing)? (?:in|at|from)|entering|arriv(?:e|ing|al) (?:in|at)|welcome to|'
                            r'leaving|into|in|to|from|downtown)\s+', re.I)


def chapter_localities(chapters, df, near_home=None):
    """[(start, end, locality)] stretches of a video by the localities its chapters name, or [] when none does.
    A chapter naming none (a road, a sight) is still in the locality before it; leading ones in the first named.
    near_home(name) gives a locality not in the mapping yet for a chapter that is just a place name."""
    locs = []
    for _, name in chapters:
        if ROAD.search(name):
            locs.append(None)
            continue
        loc = _title_locality(name, df, exact_case=True, word_needs_state=True)
        place = CHAPTER_PREFIX.sub('', name.strip(' .!-'))
        if not loc and near_home and re.fullmatch(r"(?:[A-Z][\w'.-]*\s?){1,4}", place):
            loc = near_home(place.strip())
        locs.append(loc)
    named = [loc for loc in locs if loc]
    if not named:
        return []
    out, current = [], named[0]
    for (start, _), loc, nxt in zip(chapters, locs, [t for t, _ in chapters[1:]] + [float('inf')]):
        current = loc or current
        if out and out[-1][2] == current:
            out[-1] = (out[-1][0], nxt, current)
        else:
            out.append((start, nxt, current))
    return out


def _title_locality(title, df, exact_case=False, word_needs_state=False):
    found = set()
    for _, row in df.iterrows():
        names = [row['locality']]
        aka = row.get('locality_aka')
        if isinstance(aka, str) and aka.startswith('['):
            names += [a.strip() for a in aka[1:-1].split(',') if a.strip()]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            # short aliases such as "LA" only count in capitals, so "la" in a Spanish title does not match
            flags = 0 if exact_case or len(name) <= 3 else re.I
            # "New Ringgold" is not Ringgold: a "New" before a name makes it another place
            pattern = rf'(?<!\w)(?<![Nn]ew\s){re.escape(name)}(?!\w)'
            if word_needs_state and name.lower() in COMMON_WORDS:
                if not isinstance(row['state'], str):
                    continue
                pattern += rf',\s*{re.escape(row["state"])}\b'
            state = row['state'] if isinstance(row['state'], str) else ''
            spans = [m.span() for m in re.finditer(pattern, title or '', flags)
                     if not (word_needs_state and (other := re.match(r',\s*([A-Z]{2})\b', (title or '')[m.end():]))
                             and other.group(1) != state)]
            if spans:
                state = row['state'] if isinstance(row['state'], str) else None
                found.update(((row['locality'], state, row['country']), s) for s in spans)
    # a name inside a longer matched name is no place of its own: "York" in "New York" is not York, PA
    # and the state after a town's name is its state, no place of its own: "New Rochelle, New York" is not New York
    title = title or ''
    found = {loc for loc, (a, b) in found
             if not any(x <= a and b <= y and (x, y) != (a, b) for other, (x, y) in found if other != loc)
             and not any(other != loc and re.fullmatch(r',\s*', title[y:a])
                         and add_video.US_STATE_CODES.get(title[a:b].lower()) == other[1]
                         for other, (x, y) in found)
             # a state's name after a comma is the state, also when the town before it is new: "Johnson City, New
             # York" is not New York City
             and not (title[a:b].lower() in add_video.US_STATE_CODES and re.search(r'\w\s*,\s*$', title[:a]))}
    # a title naming a US state is in it: Davenport, FL is a namesake of the one in "Davenport, Iowa", and
    # Philadelphia, MS of the one in "Philadelphia, Pennsylvania"
    states = {code for name, code in add_video.US_STATE_CODES.items()
              if re.search(rf'\b{re.escape(name)}\b', title, re.I) or re.search(rf',\s*{code}\b', title)}
    if states:
        found = {loc for loc in found if loc[2] != 'United States' or not loc[1] or loc[1] in states}
    return list(found.pop()) if len(found) == 1 else None


# words of titles that are no place ("Beautiful Evening Drive", "4K ASMR"); runs of capitalised words without
# them are looked up as places ("Al Khan Beach", "Jebel Jais Mountain")
NOT_PLACE = {'drive', 'drives', 'driving', 'beautiful', 'relax', 'relaxing', 'chilled', 'morning', 'afternoon',
             'evening', 'night', 'midnight', 'sunrise', 'sun', 'rise', 'sunset', 'unedited', 'sounds', 'sound',
             'asmr', 'vlog', 'travel', 'video', 'episode', 'part', 'visit', 'visiting', 'tour', 'tours', 'trip',
             'uhd', 'hd', 'fps', 'live', 'stream', 'subscribe', 'thanks', 'watching', 'enjoy', 'please', 'new',
             'route', 'journey', 'heavy', 'windy', 'cloudy', 'rainy', 'foggy', 'nice', 'view', 'best', 'my', 'i',
             'friday', 'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'january', 'february',
             'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
PLACE_JOINERS = {'al', 'el', 'de', 'del', 'la', 'le', 'of', 'the', 'bin', 'bu', 'van', 'von', 'da', 'do', 'dos'}


# "...from A to B in the Bronx, New York (2026)": the place after the last "in" holds the whole drive, when no
# other name follows it but its state or country
# and in Spanish: "Desde Hato Rey hasta Rio Piedras en San Juan, Puerto Rico" (area_city checks the start is in it)
DRIVE_AREA = re.compile(r"(?:\bfrom\b.+\bto\b|\b(?:[Dd]esde|de)\b.+\b(?:hasta|a|al)\b|^.+\b(?:hasta|al?)\b).+"
                        r"\b(?:in|en) (?:the )?([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})"
                        r"(?:,\s*[A-Z][\w .'-]*)?\s*(?:\bin \d{4}\b.*|[|(].*)?$")


def place_phrases(text, limit=4):
    """Runs of capitalised words in a title or description that may name a place, longest first: "Al Khan
    Beach" from "Beautiful Sun Rise at Al Khan Beach, unedited sounds". Hashtags and links are skipped."""
    text = re.sub(r'#\w+|https?://\S+|www\.\S+', ' ', text or '')
    phrases, run = [], []
    for token in re.findall(r"[^\W\d_][\w'’-]*|[^\w\s]+|\d\w*", text) + ['.']:
        word = token.lower()
        # a two-letter word that joins nothing ends a name: "Orosi,Ca Cutler,Ca" names Orosi and Cutler
        if (token[:1].isupper() and word not in NOT_PLACE and (len(token) > 2 or word in PLACE_JOINERS)
                or run and word in PLACE_JOINERS):
            run.append(token)
            continue
        while run and run[-1].lower() in PLACE_JOINERS:
            run.pop()
        if len(run) > 1 or run and len(run[0]) > 3 and run[0].lower() not in COMMON_WORDS:
            phrases.append(' '.join(run))
        run = []
    # a name written with its state ("Cutler,Ca", "Dorado,Puerto Rico") is a place even when it is also a word,
    # and goes first
    word = r"[A-ZÀ-Þ](?:[^\W\d_]|['’-])+"  # accents too: San Sebastián, Peñuelas
    with_state = re.findall(rf"\b({word}(?: {word}){{0,3}}),\s*[A-Z][A-Za-z]\b", text)
    # the state or country named that way comes along: it says where to look for the name ("New York" itself
    # is no run of the words above, "New" being no place word)
    # "Coram in Suffolk, New York": the town before " in " is named that way too, ahead of the county after it
    with_state += [x for x, y in re.findall(rf"\b({word}(?: {word}){{0,3}}) in ({word}(?: {word}){{0,3}})\s*,\s*"
                                            rf"[A-Z][\w'’]*(?: [A-Z][\w'’]*){{0,2}}", text)
                   if not set(x.lower().split()) & NOT_PLACE]  # not "Driving" in "Driving in Brooklyn, New York"
    with_state += [z for x, y in re.findall(rf"\b({word}(?: {word}){{0,3}})\s*,\s*"
                                            r"([A-Z][\w'’]*(?: [A-Z][\w'’]*){0,2})", text) if y in _region_names()
                   for z in (x, y)]
    unique = list(dict.fromkeys(p for p in phrases if len(p.split()) <= 5))
    return list(dict.fromkeys(with_state + sorted(unique, key=lambda p: -len(p.split()))))[:limit]


@lru_cache(maxsize=1)
def _region_names():
    """Names of the countries of the mapping and of every state, province or region (pycountry)."""
    return set(COUNTRIES) | {sub.name for sub in pycountry.subdivisions}


NOMINATIM_SLOT = os.path.join(tempfile.gettempdir(), 'crowd_nominatim.slot')


class _LocationIQ(Nominatim):
    """LocationIQ: Nominatim on OpenStreetMap data with a free key (5,000 requests a day, 2 a second)."""
    geocode_path, reverse_path = '/v1/search', '/v1/reverse'

    def __init__(self, key):
        super().__init__(user_agent='crowd-dataset-propose-segments', domain='us1.locationiq.com', scheme='https')
        self.key = key

    def _construct_url(self, base_api, params):
        return super()._construct_url(base_api, {**params, 'key': self.key})


def _locationiq_key():
    try:
        return common.get_secrets('locationiq_api_key')
    except (KeyError, OSError, ValueError):
        return None


def _nominatim(method, *args, **kwargs):
    """A geocode or reverse call: through LocationIQ when the secret file has "locationiq_api_key" (2 a second),
    else, or when LocationIQ refuses (its daily 5,000 are used up), through the public Nominatim at most once a
    second across every process on this computer (its usage policy; a channel run and a review page may ask at
    the same time). When Nominatim answers "too many requests" it is asked again after 1, 2 and 4 minutes."""
    if key := _locationiq_key():
        time.sleep(0.5)
        try:
            return getattr(_LocationIQ(key), method)(*args, **kwargs)
        except (GeocoderRateLimited, GeocoderQuotaExceeded, GeocoderAuthenticationFailure,
                GeocoderInsufficientPrivileges):
            pass
    for wait in (60, 120, 240, None):
        with open(NOMINATIM_SLOT, 'a+') as slot:
            fcntl.flock(slot, fcntl.LOCK_EX)
            slot.seek(0)
            last = float(slot.read() or 0)
            time.sleep(max(0.0, last + 1.1 - time.time()))
            try:
                return getattr(Nominatim(user_agent='crowd-dataset-propose-segments'), method)(*args, **kwargs)
            except GeocoderRateLimited:
                if wait is None:
                    raise
            finally:
                slot.seek(0)
                slot.truncate()
                slot.write(str(time.time()))
        time.sleep(wait)


# OpenStreetMap files these US territories under the US ("US-PR"); the mapping has them as countries
US_TERRITORIES = {'PR', 'GU', 'VI', 'AS', 'MP'}


def _country_code(address):
    """The ISO country code of an address, a US territory as its own: Adjuntas is in PR, not the US."""
    code, region = (address.get('country_code') or '').upper(), _region_code(address)
    return region[3:] if code == 'US' and region[3:] in US_TERRITORIES else code


def _region_code(address):
    """The ISO 3166-2 code of an address's state ("US-PA"): Nominatim gives it, LocationIQ only the state's name."""
    if address.get('ISO3166-2-lvl4'):
        return address['ISO3166-2-lvl4']
    try:
        subdivisions = pycountry.subdivisions.get(country_code=(address.get('country_code') or '').upper()) or []
    except (KeyError, LookupError):
        return ''
    state = _name_key(address.get('state') or '')
    return next((d.code for d in subdivisions if _name_key(d.name) == state), '') if state else ''


def _osm_name(raw):
    """A result's own name (LocationIQ may give only the full display name, whose first part it is)."""
    return raw.get('name') or (raw.get('display_name') or '').split(',')[0]


def _osm_type(raw):
    return raw.get('addresstype') or raw.get('type')


LOOKUP_FAILURES = [0]  # lookups that failed in this process: a video left without a locality then is tried again


def _remembered(fn):
    """Like lru_cache, but a lookup that failed (no connection, a refusing server) is not remembered as an
    answer: it returns None this time and is asked again next time."""
    cache = {}

    @wraps(fn)
    def wrapper(*args):
        if args not in cache:
            try:
                cache[args] = fn(*args)
            except (GeopyError, requests.RequestException, ValueError, KeyError, IndexError):
                LOOKUP_FAILURES[0] += 1
                return None
        return cache[args]
    return wrapper


@_remembered
def _osm_place(phrase, country_codes, near=None, reach_km=None):
    """(town, region, country code, region code, lat, lon) of an OpenStreetMap feature called phrase in those
    countries (a beach, a road, a district or a town): town is the one it lies in, region its state or emirate.
    near=(lat, lon): the nearest of the matches within reach of there, as the place a channel from there means
    ("Longwood in the Bronx" for a New York channel, not Longwood in Boston; Cutler, CA not Cutler, IL)."""
    box = {}
    if near:
        deg = (reach_km or ABROAD_KM) / 111
        box = {'viewbox': [(near[0] - deg, near[1] - deg), (near[0] + deg, near[1] + deg)], 'bounded': True}
    codes = set(country_codes) | ({'US'} if US_TERRITORIES & set(country_codes) else set())
    found = _nominatim('geocode', phrase, exactly_one=False, limit=10, country_codes=sorted(codes),
                       addressdetails=True, language='en', timeout=15, **box) or []
    wanted = _name_key(phrase)

    def fits(r):
        # a feature of that name that is a place (a town, district, beach, park, mountain) or a main road:
        # shops, offices and side streets named alike are everywhere ("Colombo" jewellers, a "Grand Avenue")
        name = _name_key(_osm_name(r.raw))
        return (_osm_type(r.raw) not in ('country', 'state') and name and (name in wanted or wanted in name)
                and (r.raw.get('class') in ('place', 'boundary', 'tourism', 'natural', 'leisure', 'landuse')
                     or r.raw.get('class') == 'highway' and r.raw.get('type') in ('motorway', 'trunk', 'primary')))
    found = [r for r in found if fits(r)]
    if not found:
        return None
    # the exact name first ("Plains" is Plains, PA, not White Plains, NY), in a town ("Lancaster" is the city,
    # not Lancaster County), then the nearest

    def in_town(r):
        return any(r.raw.get('address', {}).get(k) for k in ('city', 'town', 'village', 'municipality'))
    r = min(found, key=lambda r: (_name_key(_osm_name(r.raw)) != wanted, not in_town(r),
                                  _km(near[0], near[1], r.latitude, r.longitude) if near else 0))
    a = r.raw.get('address', {})
    kind = next((k for k in ('city', 'town', 'village', 'municipality', 'hamlet') if a.get(k)), None)
    town = a.get(kind) if kind else None
    if (not town and r.raw.get('class') == 'boundary'
            and _osm_type(r.raw) not in ('county', 'state', 'country', 'region')):
        town = _osm_name(r.raw)  # a municipality's own boundary has no town in its address (Little Silver, NJ)
    if town and re.search(r'\bCounty$', town):  # a county is no locality: "Suffolk" in "Coram in Suffolk"
        town = None
    if not town and _country_code(a) in US_TERRITORIES:  # Puerto Rico's municipios (Guayama) are its counties
        town = re.sub(r' Municipio$', '', a.get('county') or '') or None
    if town:  # OpenStreetMap calls some towns "City of Mount Vernon", "Town of Hempstead"
        town = re.sub(r'^(?:City|Town|Village|Borough|Township) of ', '', town)
    return (town, a.get('state'), _country_code(a), _region_code(a),
            r.latitude, r.longitude, float(r.raw.get('importance') or 0), kind)


def abroad_is_nearest(locality, df, home, country):
    """False when a place of that name lies nearer home in the channel's own country: Richmond Hill in a title
    of a Pennsylvania channel is the one in Queens, not Richmond Hill, Ontario."""
    _, row = add_video.get_existing_locality_row(df, *locality)
    if row is None:
        return True
    hit = _osm_place(locality[0], (common.get_iso2_country_code(common.correct_country(country)),), home[3:5])
    return hit is None or (_km(home[3], home[4], hit[4], hit[5])
                           > _km(home[3], home[4], float(row['lat']), float(row['lon'])))


def _name_key(name):
    """Name compared without accents, case, punctuation or a trailing "Emirate", "City", "Province"."""
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    name = ' '.join(re.sub(r'\W+', ' ', name).split())
    name = re.sub(r'^(?:city|town|village|borough|township) of ', '', name)  # OpenStreetMap's "City of Newburgh"
    # "St. Clair" (OpenStreetMap) is "Saint Clair" (GeoNames); Mt. and Ft. alike
    name = re.sub(r'\b(?:st|saint) ', 'saint ', re.sub(r'\bmt ', 'mount ', re.sub(r'\bft ', 'fort ', name)))
    return re.sub(r' (?:emirate|city|province|prefecture|governorate|municipality|county)$', '', name)


def place_locality(phrases, df, home=None):
    """([locality, state, country], phrase) for the first phrase OpenStreetMap finds in df's countries (and near
    home when known), None when it finds none; see place_localities."""
    return next(place_localities(phrases, df, home), None)


def _geonames_town(phrase, state):
    """A hit for a US town of exactly that name in the state, from GeoNames, else None: OpenStreetMap names some
    "Village of Johnson City" and finds hamlets called Johnson for "Johnson City, New York"."""
    name = phrase.split(',')[0].strip()
    data = add_video.get_locality_data(name, 'US', state) or {}
    g = next((g for g in data.get('geonames') or [] if _name_key(g.get('name', '')) == _name_key(name)), None)
    if not g:
        return None
    return (g['name'], g.get('adminName1'), 'US', f"US-{g.get('adminCode1')}", float(g['lat']), float(g['lng']), 0.5,
            'town' if int(g.get('population') or 0) >= SMALL_POP else 'village')


def place_localities(phrases, df, home=None):
    """([locality, state, country], phrase) for each phrase OpenStreetMap finds in df's countries (and near
    home when known): the mapping locality of the town it lies in ("Al Khan Beach" is in Sharjah), else of its
    region when that is a locality ("Jebel Jais" in the Ras Al Khaimah emirate), else that town as a new
    locality."""
    countries = {common.get_iso2_country_code(common.correct_country(c)): c for c in df['country'].dropna().unique()}
    countries.pop(None, None)
    codes = tuple(sorted(countries))
    states = {_name_key(sub.name): sub.name for code in codes
              for sub in pycountry.subdivisions.get(country_code=code) or []
              if not (code == 'US' and sub.code[3:] in US_TERRITORIES)}  # Puerto Rico is a country here
    # a state of the countries searched is the state ("Atlanta, Georgia"), else "GEORGIA" in a title is the country
    country_names = {_name_key(c) for c in COUNTRIES} - set(states)
    named = {states[_name_key(p)] for p in phrases if _name_key(p) in states}

    def usable(hit):
        # the reach of home is for places abroad: Elkhart, Indiana is far from New York but in the same country
        return (hit and hit[2] in countries
                and not (home and countries[hit[2]] != home[2] and _km(home[3], home[4], hit[4], hit[5]) > ABROAD_KM)
                and (len(named) != 1 or hit[1] in named))
    # "from Longwood to West Farms in the Bronx, New York": the best-known place a text names (the Bronx) is the
    # anchor, and each other name is the place of that name nearest it, not the one nearest the channel's home
    if home and {_name_key(p) for p in phrases} & (country_names - {_name_key(home[2])}):
        home = None  # the text names another country ("…en Adjuntas, Puerto Rico"): it is there, however far
    # a country's name with a state is a town ("Lebanon, Pennsylvania"), else the country ("GEORGIA" in a travel title)
    phrases = [p for p in phrases if named or _name_key(p) not in country_names]
    # a state only qualifies the town named with it ("Palmerton, Pennsylvania"): as a place of its own it would
    # be the anchor, or match some "Pennsylvania" road in another town; the town is looked up in it
    phrases = [p for p in phrases if _name_key(p) not in states]  # a state alone is no locality
    if len(named) == 1 and any(_name_key(p) not in states for p in phrases):
        phrases = [f'{p}, {next(iter(named))}' for p in phrases]
    # looked for nearest home, and with a state named anywhere in the country, not only within reach of home
    hits = [(p, _osm_place(p, codes, (home[3], home[4]) if home else None, 5000 if named else None)) for p in phrases]
    if len(named) == 1 and 'US' in codes:
        # GeoNames only where OpenStreetMap found the name somewhere else: Brooklyn is both, in New York City
        hits = [(p, g if (g := _geonames_town(p, next(iter(named)))) and not (h and _km(h[4], h[5], g[4], g[5]) <= 5)
                 else h) for p, h in hits]
    hits = [(p, h) for p, h in hits if usable(h)]
    if len(hits) > 1:
        anchor = max(hits, key=lambda ph: ph[1][6])
        # a name OpenStreetMap does not know near the anchor (a small neighbourhood) says nothing: a far
        # namesake (Longwood, PA for Longwood in the Bronx) would only look like a second town
        hits = [(p, h if p == anchor[0] else _osm_place(p, codes, anchor[1][4:6], 50)) for p, h in hits]
    for phrase, hit in hits:
        # the town the phrase itself names keeps its name ("Hazlet"); a sight in a town counts as usual
        itself = hit and hit[0] and _name_key(phrase.split(',')[0]) == _name_key(re.sub(r' Township$', '', hit[0]))
        if usable(hit) and (loc := _hit_locality(hit, df, countries, named=bool(itself))):
            yield loc, phrase


NEARBY_PLACES = '_output/proposals/nearby_places.json'  # GeoNames answers kept between runs


def _nearby_places(lat, lon):
    """[(name, population, lat, lon)] of the GeoNames places of 1,000+ inhabitants within MAJOR_TOWN_KM of a point
    (and some more); asked per ~5 km square, so a drive costs few of GeoNames' hourly credits, and kept. None when
    GeoNames does not answer."""
    clat, clon = round(lat * 20) / 20, round(lon * 20) / 20
    key = f'c5:{clat:.2f},{clon:.2f}'
    with contextlib.suppress(OSError, ValueError), open(NEARBY_PLACES) as f:
        known = json.load(f)
        if key in known:
            return known[key]
    try:
        r = requests.get('http://api.geonames.org/findNearbyPlaceNameJSON', timeout=30, params={
            'lat': clat, 'lng': clon, 'radius': MAJOR_TOWN_KM + 4, 'cities': 'cities1000',
            'maxRows': 500, 'username': common.get_secrets('geonames_username')}).json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(r.get('geonames'), list):  # an error message (credits used up for the hour)
        return None
    places = [(g['name'], int(g.get('population') or 0), float(g['lat']), float(g['lng'])) for g in r['geonames']]
    with plan_lock(NEARBY_PLACES):
        try:
            with open(NEARBY_PLACES) as f:
                known = json.load(f)
        except (OSError, ValueError):
            known = {}
        known[key] = places
        with open(NEARBY_PLACES + '.tmp', 'w') as f:
            json.dump(known, f)
        os.replace(NEARBY_PLACES + '.tmp', NEARBY_PLACES)
    return places


def _urban(lat, lon):
    """Whether a point is in or next to a town: within 1 km + 0.6 km x sqrt(population in thousands) of a GeoNames
    place of 1,000+ inhabitants (1.6 km for a village, 4 km for Hazleton, 5 km for Wilkes-Barre). True when
    GeoNames does not answer, so nothing is left out for lack of data."""
    places = _nearby_places(lat, lon)
    return places is None or any(_km(lat, lon, la, lo) <= _town_km(pop) for _, pop, la, lo in places)


def _town_km(population):
    """How far a town reaches from its centre."""
    # ponytail: town size as a disc around its centre; a land-use or population-density map would follow real edges
    return 1 + 0.6 * (population / 1000) ** 0.5


def _major_town(lat, lon, town, kind=None):
    """The bigger place a drive through a small one counts for, else None (the town stays). Small: a township, a
    place of fewer than SMALL_POP inhabitants, or (not in GeoNames) an OpenStreetMap village or hamlet. Its drive
    counts for the nearest place of SMALL_POP or more within MAJOR_TOWN_KM: Jenkins Township is Wyoming, PA,
    Hughestown is Pittston."""
    places = _nearby_places(lat, lon)
    if not places:
        return None
    township = bool(re.search(r'\bTownship$', town))
    # a township is not the town it is named after: Pittston Township counts for Pittston
    own = [p for p in places if not township and _name_key(p[0]) == _name_key(town)]
    own = min(own, key=lambda p: _km(lat, lon, p[2], p[3])) if own else None
    small = township or (own[1] < SMALL_POP if own else kind in ('village', 'municipality', 'hamlet'))
    bigger = [p for p in places if p[1] >= SMALL_POP and p is not own
              and _km(lat, lon, p[2], p[3]) <= MAJOR_TOWN_KM]  # places come from a wider radius
    if own:  # a town of its own counts for a bigger one only when they touch (Hughestown and Pittston, 1.4 km),
        # not across fields (Coopersburg and Hellertown, 8.6 km)
        bigger = [p for p in bigger if _km(own[2], own[3], p[2], p[3]) <= _town_km(own[1]) + _town_km(p[1])]
    if township and not bigger:  # a township around a small borough counts for it: Bridgewater Township is Montrose
        bigger = [p for p in places if _km(lat, lon, p[2], p[3]) <= MAJOR_TOWN_KM]
    if not small or not bigger:
        return None
    return min(bigger, key=lambda p: _km(lat, lon, p[2], p[3]))[0]


def _hit_locality(hit, df, countries, at=None, named=False):
    """[locality, state, country] of an OpenStreetMap hit: the mapping locality of the town it lies in, else of its
    region when that is a locality (a mountain park in an emirate), else that town as a new locality. A small
    place counts for the bigger one nearest the hit (or at=(lat, lon)), in the mapping or new (_major_town).
    Neighbourhoods come as their city already (Brooklyn is New York). named: a title names the town itself, so it
    stays (Hazlet, not the Union Beach a "Hazlet Township" would count for)."""
    town, region, code, region_code = hit[:4]
    kind = hit[-1] if len(hit) in (5, 8) else None
    at = hit[4:6] if len(hit) > 5 else at
    if town and named:
        town = re.sub(r' Township$', '', town)
    elif town and at:
        town = _major_town(at[0], at[1], town, kind) or town
    if code not in countries:
        return None
    rows = df[df['country'] == countries[code]]
    # where the country's localities in the mapping carry a state (US-NY -> NY), the town must be in it:
    # Mount Vernon, NY is not the Mount Vernon, WA of the mapping
    with_state = rows['state'].notna().mean() > 0.5
    state = region_code.split('-')[-1] if with_state and region_code else None
    if state:
        rows = rows[rows['state'] == state]
    known = {_name_key(n): r for n, r in zip(rows['locality'], rows.itertuples())}
    # alternative names too: a place merged into a locality (West Pittston into Pittston) counts for it
    known.update({_name_key(a): r for aka, r in zip(rows['locality_aka'], rows.itertuples()) if isinstance(aka, str)
                  for a in aka.strip('[]').split(',') if a.strip() and _name_key(a) not in known})
    # a region is the locality only where localities have no state (an emirate): New York State is not New York City
    for name in ((town,) if town else () if with_state else (region,)):
        if name and _name_key(name) in known:
            r = known[_name_key(name)]
            return [r.locality, r.state if isinstance(r.state, str) else None, r.country]
    if town and at and (metro := _metro_city(town, at, rows, state)):
        return metro
    # a new locality is spelled one way whichever source named it, as most of the mapping does: St. Clair
    return [re.sub(r'^Saint ', 'St. ', town), state, countries[code]] if town else None


METRO_POP = 1_000_000  # a city this big has a metropolitan area: towns around it count as the city
METRO_KM = 30          # within this many km of its centre (Yonkers 25 km from Manhattan, Bellflower 20 km from LA)
METRO_TOWN_POP = 250_000  # a town this big stays a city of its own (Newark, Jersey City)


def _metro_city(town, at, rows, state=None):
    """[locality, state, country] of the big city (METRO_POP or more, in rows: the mapping's localities of the
    country) whose metropolitan area a new town at (lat, lon) in that state lies in, else None. Another state is
    another city: Jersey City, NJ is not New York City. Towns already in the mapping keep their own entry; this is
    only for new ones."""
    pop = pd.to_numeric(rows['population_locality'], errors='coerce')
    big = [(_km(at[0], at[1], float(r.lat), float(r.lon)), r) for r in rows[pop >= METRO_POP].itertuples()
           if pd.notna(r.lat) and pd.notna(r.lon) and not (state and isinstance(r.state, str) and r.state != state)]
    big = [(d, r) for d, r in big if d <= METRO_KM and _name_key(r.locality) != _name_key(town)]
    if not big:
        return None
    own = [p[1] for p in _nearby_places(at[0], at[1]) or [] if _name_key(p[0]) == _name_key(town)]
    if own and max(own) >= METRO_TOWN_POP:
        return None
    r = min(big, key=lambda dr: dr[0])[1]
    return [r.locality, r.state if isinstance(r.state, str) else None, r.country]


# "Driving from Hazle Township to White Haven, Pennsylvania": where a drive starts and where it ends
# "Desde Sabana Grande a Ponce", "Ponce a Juana Diaz,Puerto Rico": in Spanish too, tried in this order
_END = r"\s*(?:,|\||\(|\s-\s|\s–\s|\bin\b|\ben\b|\bvia\b|\bpor\b|$)"
FROM_TO = (re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)" + _END, re.I),
           re.compile(r"\b(?:[Dd]esde|de)\s+(?:el |la )?(.+?)\s+(?:hasta|a|al)\s+(?:el |la )?(.+?)" + _END),
           re.compile(r"^(?:el |la )?(.+?)\s+(?:hasta|a|al)\s+(?:el |la )?(.+?)" + _END))
MAJOR_TOWN_KM = 10    # a small place counts for the nearest bigger one this close
SMALL_POP = 3000      # places of fewer inhabitants (and townships) count for a bigger one nearby
MAX_KMH = 120         # no drive in a video covers more than this many km an hour as the crow flies
ROUTE_SAMPLES = 20     # points along a route whose town is looked up (one OpenStreetMap request a second each)


@_remembered
def _osm_reverse(lat, lon):
    """(town, region, country code, region code) at a point; town is the city, town, village or township."""
    r = _nominatim('reverse', (lat, lon), zoom=14, addressdetails=True, language='en', timeout=15)
    a = (r.raw.get('address') or {}) if r else {}
    kind = next((k for k in ('city', 'town', 'village', 'municipality', 'hamlet') if a.get(k)), None)
    town = a.get(kind) if kind else None
    if town:
        town = re.sub(r'^(?:City|Town|Village|Borough|Township) of ', '', town)
    return (town, a.get('state'), _country_code(a), _region_code(a), kind) if a else None


@_remembered
def _route(start, end):
    """([(lat, lon) every 1/ROUTE_SAMPLES of the driving time], the route's path thinned to at most 300 points)
    of the OSRM car route between two points."""
    r = requests.get(f'https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}',
                     params={'overview': 'full', 'geometries': 'geojson', 'annotations': 'duration'}, timeout=30)
    r.raise_for_status()
    route = r.json()['routes'][0]
    coords = route['geometry']['coordinates']
    durations = np.array([d for leg in route['legs'] for d in leg['annotation']['duration']])
    if len(coords) < 2 or durations.sum() <= 0:
        return None
    done = np.r_[0, np.cumsum(durations)] / durations.sum()
    samples = [tuple(reversed(coords[min(int(np.searchsorted(done, f)), len(coords) - 1)]))
               for f in np.linspace(0, 1, ROUTE_SAMPLES + 1)]
    step = max(1, len(coords) // 300)
    path = [[round(lat, 5), round(lon, 5)] for lon, lat in coords[::step] + [coords[-1]]]
    return samples, path


def drive_ends(title, df, home=None):
    """(OpenStreetMap hit, locality) of where a drive "from A to B" starts and of where it ends (either may be
    None when not found), or None when the title names no such drive."""
    m = next((m for r in FROM_TO if (m := r.search(title or ''))), None)
    if not m:
        return None
    if home and any(c != home[2] for c in title_countries(title)):
        home = None  # the title names another country ("Ponce a Juana Diaz,Puerto Rico"): it is there, however far
    countries = {common.get_iso2_country_code(common.correct_country(c)): c for c in df['country'].dropna().unique()}
    countries.pop(None, None)
    codes = tuple(sorted(countries))
    first, last = m.group(1).strip(), m.group(2).strip()
    # "from Chinatown in Manhattan to …": the start is in that borough or town, which is found where the
    # neighbourhood is not
    first = re.sub(r"^.+?\s+(?:in|en)\s+(?:the\s+)?(?=[A-Z])", '', first)
    # "el Puerto de San Juan", "Centro de Adjuntas": a sight or part of a town is looked for as that town
    first, last = (re.sub(r"^(?:el |la )?\S.*?\s+(?:de|del)\s+(?=[A-ZÀ-Þ])", '', x) for x in (first, last))
    # "from Newburgh to Brewster, New York": both ends in the states the title names (Maine has both too); with
    # one state named, each end is looked up in it ("Newburgh, New York")
    named = {sub.name for code in codes for sub in pycountry.subdivisions.get(country_code=code) or []
             if sub.name not in COUNTRIES  # Puerto Rico is a US subdivision but a country of the mapping
             and re.search(rf'\b{re.escape(sub.name)}\b', title)}
    if len(named) == 1:
        state = next(iter(named))
        first, last = (x if state in x else f'{x}, {state}' for x in (first, last))
    # a named state says where, however far from home: no box around home (San Jose is 48 degrees from New York)
    near = (home[3], home[4]) if home and not named else None
    # a country the title names holds both ends: "Carolina a … San Juan, Puerto Rico" is not a Carolina in the US
    only = {code for code, name in countries.items() if name in title_countries(title)}
    if only and not named:  # looked up in it, as with a state ("Carolina, Puerto Rico")
        country = countries[next(iter(only))]
        first, last = (x if country in x else f'{x}, {country}' for x in (first, last))

    def place(phrase, near=None, reach=None):
        hit = _osm_place(phrase, codes, near, reach)
        return hit if hit and (not only or hit[2] in only) else None
    a = place(first, near)
    b = place(last, a[4:6] if a else near, 300 if a else None)
    if not b and a and home:  # A was a far namesake (Longwood upstate for the one in the Bronx): B near home
        b = place(last, (home[3], home[4]))
    # a drive between two towns is short: each end is also looked for near the other, and the closest pair is
    # meant ("from Plains to Hughestown": Plains Township next to Hughestown, not a Plains in New Jersey)
    pairs = [(x, y) for x in (a, b and place(first, b[4:6], 60))
             for y in (b, a and place(last, a[4:6], 60)) if x and y]
    if named:
        pairs = [(x, y) for x, y in pairs if x[1] in named and y[1] in named]
    if pairs:
        a, b = min(pairs, key=lambda xy: _km(xy[0][4], xy[0][5], xy[1][4], xy[1][5]))
    elif named:  # one end only: it must be in the state named
        a, b = (x if x and x[1] in named else None for x in (a, b))
    end = b and _hit_locality(b, df, countries)
    # "…to West Farms in the Bronx", "…hasta el Barrio Portugues en Adjuntas": the drive ends in that place, and a
    # namesake of the end elsewhere (another Barrio Portugués) is not it
    if area := re.match(r"\s*(?:in|en)\s+(?:the\s+)?([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})", title[m.end(2):]):
        # near the drive: "in Nassau" is the county by Queens, not the village of Nassau near Albany
        c = _osm_place(area.group(1) + (f', {next(iter(named))}' if len(named) == 1 else ''), codes,
                       a[4:6] if a else b[4:6] if b else (home[3], home[4]) if home else None, 60)
        if c and (in_c := _hit_locality(c, df, countries)) and end != in_c:
            b, end = c, in_c
    return (a, a and _hit_locality(a, df, countries)), (b, end)


def area_city(title, df, home=None, near=None):
    """[locality, state, country] of the city C of a drive "from A to B in C" when C is a city ("in the Bronx" is
    New York), else None (no such title, or C a county: "in Nassau")."""
    area = DRIVE_AREA.search(title or '')
    if not area:
        return None
    countries = {common.get_iso2_country_code(common.correct_country(c)): c for c in df['country'].dropna().unique()}
    countries.pop(None, None)
    codes = tuple(sorted(countries))
    state = [n.title() for n in add_video.US_STATE_CODES if re.search(rf'\b{re.escape(n)}\b', title, re.I)]
    q = area.group(1) + (f', {state[0]}' if len(state) == 1 else '')
    near = near or ((home[3], home[4]) if home else None)
    hit = _osm_place(q, codes, near, 60) or _osm_place('the ' + q, codes, near, 60)
    if hit and hit[0] and not re.search(r'\bCounty$', hit[0]):
        city = _hit_locality(hit, df, countries)
        # the start is a town of its own next to C ("from Alexandria to the Pentagon in Arlington"): not all in C;
        # a far start is a namesake of a place in C (Longwood upstate for the one in the Bronx)
        a, start = (drive_ends(title, df, home) or ((None, None),))[0]
        if a and start and start != city and _km(a[4], a[5], hit[4], hit[5]) <= METRO_KM:
            return None
        return city
    return None


def between_towns(title, df, home=None):
    """Whether a title is a drive from one locality to another: "from A to B" with A and B found in different
    localities. A drive whose end is not found is kept for the reviewer (left out, it would never come back)."""
    ends = drive_ends(title, df, home)
    if not ends:
        return False
    (a, start), (b, end) = ends
    # "from Longwood to West Farms in the Bronx": all in that city, unless it is a county ("in Nassau")
    if area_city(title, df, home, (b or a)[4:6] if (b or a) else None):
        return False
    return bool(start and end and start != end)


def route_localities(title, df, home=None, max_km=None):
    """([(share of the drive where it starts, where it ends, locality)], the route for a map) along the car route
    of a drive "from A to B" through two different towns, else ([], None). The towns on the way are looked up on
    OpenStreetMap every 1/ROUTE_SAMPLES of the driving time, so the borders are an estimate that assumes the video
    follows that route at an even pace. max_km: A and B are at most this far apart (as the crow flies); farther
    means one of them is a namesake (Bear Creek near Baltimore for Bear Creek Village, PA)."""
    ends = drive_ends(title, df, home)
    if not ends:
        return [], None
    (a, start), (b, end) = ends
    if city := area_city(title, df, home, (b or a)[4:6] if (b or a) else None):
        return [(0.0, 1.0, city)], None  # "… in the Bronx": all in New York, no route (upstate has a Longwood too)
    countries = {common.get_iso2_country_code(common.correct_country(c)): c for c in df['country'].dropna().unique()}
    countries.pop(None, None)
    if not (start and end) or max_km and _km(a[4], a[5], b[4], b[5]) > max_km:
        return [], None
    if start == end:  # a drive within one locality (between its neighbourhoods): one stretch, no route needed
        return [(0.0, 1.0, start)], None
    points, path = _route(a[4:6], b[4:6]) or ([], None)
    locs = [start] + [_hit_locality(h, df, countries, (lat, lon))
                      if (h := _osm_reverse(round(lat, 4), round(lon, 4))) else None
                      for lat, lon in points[1:-1]] + [end]
    points = [a[4:6]] + list(points[1:-1]) + [b[4:6]]
    out, stops, town = [], [], start
    for i, loc in enumerate(locs):
        town = loc or town  # a point in no town is still in the one before
        point = points[min(i, len(points) - 1)]
        here = town if _urban(*point) else None  # None: rural driving between towns, left out
        if out and out[-1][2] == here:
            out[-1] = (out[-1][0], (i + 1) / len(locs), here)
        else:
            out.append((i / len(locs), (i + 1) / len(locs), here))
            stops.append([round(point[0], 5), round(point[1], 5), here[0] if here else 'rural'])
    if len(out) < 2:  # every point in one locality (or all rural): a continuous drive
        return [(0.0, 1.0, out[0][2])] if out else [], None
    return out, {'path': path or [p[:2] for p in stops], 'stops': stops}


def title_reach(df, in_reach, title):
    """in_reach and the countries a title names: "Cayey a Gurabo, Puerto Rico" from a New York channel."""
    return in_reach | df['country'].isin(title_countries(title))


def split_routes(plan_path):
    """split_route for every proposal in a plan nobody has decided anything in yet; returns how many were split.
    The plan is locked per video, so a review page and a channel run can stay open meanwhile."""
    df = add_video.load_csv(add_video.FILE_PATH)
    with open(plan_path) as f:
        plan = {p['video']: p for p in json.load(f)}
    channels = Counter(p.get('channel_id') for p in plan.values() if p.get('channel_id'))
    home = channel_home(channels.most_common(1)[0][0], df, plan) if channels else None
    country = home[2] if home else None
    in_reach = (df['country'] == country) | (np.array([_km(home[3], home[4], lat, lon) <= ABROAD_KM
                                                       for lat, lon in zip(df['lat'], df['lon'])]) if home else False)
    in_reach |= df['country'].isin(filmed_countries(channels, df))
    done = 0
    for vid in list(plan):
        p = json.loads(json.dumps(plan[vid]))
        if not split_route(p, df[title_reach(df, in_reach, p.get('title'))], home):
            continue
        with plan_lock(plan_path):  # write only if the video is still untouched in the plan on disk
            with open(plan_path) as f:
                current = json.load(f)
            i = next((i for i, q in enumerate(current) if q['video'] == vid), None)
            if i is None or current[i].get('entries') != plan[vid].get('entries'):
                continue
            current[i] = p
            tmp = plan_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(current, f, indent=1)
            os.replace(tmp, plan_path)
        done += 1
        print(f"{vid} {p['title'][:60]}: "
              f"{p.get('exclude') or ', '.join(e['locality'][0] for e in p['entries'])}", flush=True)
    return done


def split_by_stretches(segments, stretches):
    """Locality groups {locality: segments} of segments cut at the (start s, end s, locality) stretches."""
    groups = {}
    for seg in segments:
        for a, b, loc in stretches:
            if loc is None:  # rural driving: not proposed
                continue
            lo, hi = max(seg['start'], a), min(seg['end'], b)
            if hi - lo >= 5:
                groups.setdefault(tuple(loc), []).append({'start': int(lo), 'end': int(hi), 'night': seg['night']})
    return [{'locality': list(loc), 'guessed': True, 'segments': segs} for loc, segs in groups.items()]


def split_route(p, df, home=None):
    """Split an untouched one-group proposal of a drive "from A to B" into the localities along its route, the
    driving time spread over them as on the route. Returns whether it was split."""
    if len(p.get('entries', [])) != 1:
        return False
    segs = p['entries'][0]['segments']
    if not segs or any(s.get('decision') or s.get('applied') for s in segs):
        return False
    t1 = max(s['end'] for s in segs)
    shares, route = route_localities(p.get('title'), df, home, max_km=t1 / 3600 * MAX_KMH)
    if not shares:
        return False
    if all(loc is None for _, _, loc in shares):
        p.pop('entries')
        p['exclude'] = 'rural driving: no town along the route'
        return True
    if len(shares) == 1:  # the whole drive is in one locality: set it unless one is known already
        e = p['entries'][0]
        if e.get('locality'):
            return False
        e['locality'], e['guessed'] = shares[0][2], True
        p['note'] = f"{p.get('note') or ''}; locality {shares[0][2][0]}: the drive starts and ends in it".lstrip('; ')
        return True
    # the video is the drive from A: footage left out at its start (a dark or parked opening) is on the way too
    t0 = 0
    stretches = [(t0 + a * (t1 - t0), t0 + b * (t1 - t0), loc) for a, b, loc in shares]
    p['entries'] = split_by_stretches(segs, stretches)
    if not p['entries']:
        p.pop('entries')
        p['exclude'] = 'rural driving: the footage is between towns'
        return True
    p['route'] = route
    p['note'] = (f"{p.get('note') or ''}; localities along the route, borders estimated from driving time "
                 "(check them): " + ', '.join(f"{loc[0] if loc else 'rural (left out)'} from "
                                              f'{int(a) // 60}:{int(a) % 60:02d}'
                                              for a, _, loc in stretches)).lstrip('; ')
    return True


@_remembered
def town_near(name, home):
    """[locality, state, country] of the town a place of that name near the channel's home is: itself, or for
    a neighbourhood the city it is part of (Echo Park is Los Angeles, Beverly Hills its own city). OpenStreetMap
    tells them apart; GeoNames calls both a section of a populated place."""
    r = _nominatim('geocode', f'{name}, {home[2]}', featuretype='settlement', addressdetails=True, timeout=15,
                   viewbox=[(home[3] - 1, home[4] - 1), (home[3] + 1, home[4] + 1)], bounded=True)
    if r is None or _km(home[3], home[4], r.latitude, r.longitude) > 60:
        return None
    address = r.raw.get('address', {})
    if _osm_type(r.raw) in ('city', 'town', 'village', 'municipality'):
        town = name
    else:
        town = address.get('city') or address.get('town') or address.get('village')
    return [town, home[1], home[2]] if town else None


def _km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


ABROAD_KM = 1000  # a title place in another country counts only this close to the channel's home


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


def filmed_countries(channels, df):
    """The countries the plan's channel already has localities in, in the mapping: a travel channel's trips."""
    if not channels:
        return set()
    own = df['channel'].astype(str).str.contains(channels.most_common(1)[0][0], regex=False)
    return set(df.loc[own, 'country'].dropna())


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


def process_channel(url, country, out_dir, pause_s=15, limit=None, download=True, single_city=False):
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

    def mapping_videos(df):
        return {v for cell in df['videos'] for v in add_video._parse_videos_cell(cell)}
    df = add_video.load_csv(add_video.FILE_PATH)
    in_mapping = mapping_videos(df)
    mapping_mtime = os.path.getmtime(add_video.FILE_PATH)
    vehicle_cache = {}
    use_api = bool(_api_key())
    if not download and not use_api:
        print('Without downloading, video details come from the YouTube Data API: '
              'add a key to the secret file as "youtube_api_key".')
        return None

    stopped = None
    touched = []  # videos added or re-analysed in this run
    single_home = None  # the channel's home, looked up once for single_city
    failures_in_a_row = 0
    skips_in_a_row = 0
    try:
        # the channel's video list is kept for a day: listing 8,000 videos costs ~170 of the API's 10,000 daily
        # units, and a batch runs every few minutes
        listed = os.path.join(out_dir, 'channel_list.json')
        today = time.strftime('%Y-%m-%d')
        try:
            with open(listed) as f:
                cached = json.load(f)
        except (OSError, ValueError):
            cached = {}
        if cached.get('date') == today and cached.get('url') == url:
            ids = cached['ids']
        else:
            ids = list_channel_api(url) if use_api else list_channel(url)
            with open(listed + '.tmp', 'w') as f:
                json.dump({'date': today, 'url': url, 'ids': ids}, f)
            os.replace(listed + '.tmp', listed)
        todo = [v for v in ids if v not in in_mapping and (v not in plan or v in reanalyse)]
        # details only for the videos this batch can reach (one unit per 50); others come with a later batch
        api_meta = fetch_metadata_api(todo[:limit * 10] if limit else todo) if use_api else {}
        saved = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('_meta.json')]
        boilerplate = channel_boilerplate([m.get('description') for m in api_meta.values()]
                                          + [json.load(open(f)).get('description') for f in saved])
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
        # the mapping changes while this runs (the review page applies segments): skip videos now in it
        if os.path.getmtime(add_video.FILE_PATH) != mapping_mtime:
            mapping_mtime = os.path.getmtime(add_video.FILE_PATH)
            df = add_video.load_csv(add_video.FILE_PATH)
            in_mapping = mapping_videos(df)
        if vid in in_mapping:
            continue
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
            if not reason and single_city:
                if single_home is None:
                    single_home = channel_home(meta.get('channel_id'), df, plan) or False
                named = title_countries(meta['title'])
                if between_towns(meta['title'], df[df['country'].isin([country] + named)], single_home or None):
                    reason = 'a drive from one town to another (this channel: only drives within one city)'
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
            elif at := footage_skip(dict(np.load(sig_path))):
                reason = f'the video skips footage at {at} (an edit cut; waits at intersections may be missing)'
            else:
                reason = footage_problem(dict(np.load(sig_path)))
            if 'skips' in np.load(sig_path):  # only videos analysed for skips count towards the streak
                skips_in_a_row = skips_in_a_row + 1 if reason and reason.startswith('the video skips') else 0
            if skips_in_a_row >= SKIP_STREAK:
                plan[vid] = {'video': vid, 'title': meta['title'], 'exclude': reason}
                touched.append(vid)
                commit([vid])
                stopped = (f'{SKIP_STREAK} analysed videos in a row skip footage: this channel edits its drives, '
                           'so it is rejected; mark it "Rejected" in the channels sheet')
                break
        if reason:
            plan[vid] = {'video': vid, 'title': meta['title'], 'exclude': reason}
            touched.append(vid)
            commit([vid])
            continue
        if analysed:
            segments, notes = propose(dict(np.load(sig_path)))
            if not segments:
                plan[vid] = {'video': vid, 'title': meta['title'],
                             'exclude': 'no usable footage left: ' + ('; '.join(notes) or 'too short')}
                touched.append(vid)
                commit([vid])
                continue
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

    # a run cut off (hotspot, pause) saved its videos but never got to their localities: this run finishes them
    for v, p in plan.items():
        if (v not in touched and 'entries' in p and not p.get('located')
                and not any(e.get('locality') for e in p['entries'])
                and not any(s.get('decision') or s.get('applied') for e in p['entries'] for s in e['segments'])):
            touched.append(v)
            committed[v] = json.dumps(p, sort_keys=True)  # saved below only if nobody changed it meanwhile
    new = [plan[v] for v in touched]
    failures_before = LOOKUP_FAILURES[0]
    # a channel films around one area: a title place whose mapping match is far from the channel's home is a
    # namesake when a place of that name exists near home ("Hollywood" from an LA channel is not Hollywood, FL),
    # and a genuine trip otherwise ("Seattle" has no namesake near LA)
    metas = {}
    for p in new:
        path = os.path.join(out_dir, f"{p['video']}_meta.json")
        if os.path.exists(path):
            with open(path) as f:
                metas[p['video']] = json.load(f)
    channels = Counter(p.get('channel_id') for p in plan.values() if p.get('channel_id'))
    home = channel_home(channels.most_common(1)[0][0], df, plan) if channels else None
    # places a video may be in: the channel's country, abroad within reach of its home, and the countries it has
    # filmed in before (a travel channel's trips; Puerto Rico for a New York channel), not anywhere: a title's
    # "Campo Alegre" is not the one in Brazil
    in_reach = (df['country'] == country) | df['country'].isin(filmed_countries(channels, df))
    if home:
        in_reach |= np.array([_km(home[3], home[4], lat, lon) <= ABROAD_KM for lat, lon in zip(df['lat'], df['lon'])])
    for p in new:
        for e in p.get('entries', []):
            if e.get('locality'):
                continue
            # a capitalised title name of one locality abroad ("Entering Tijuana" from an LA channel); "New Mexico"
            # or "Little Tokyo" are not trips abroad. Then the description, where "from Los Angeles to Ensenada"
            # names two places and so decides nothing
            # a country the title names is in reach however far ("…en Adjuntas, Puerto Rico" from New York)
            reach = in_reach | df['country'].isin(title_countries(p['title']))
            found = _title_locality(p['title'], df[reach & (df['country'] != country)], exact_case=True)
            if found and home and not abroad_is_nearest(found, df, home, country):
                found = None
            if not found:
                found = description_locality(metas.get(p['video'], {}).get('description'), df[in_reach], boilerplate)
                if found:
                    p['note'] = f"{p['note']}; locality {found[0]} from the description".lstrip('; ')
            if found:
                e['locality'], e['guessed'] = found, True

    # a drive through several localities often names them as chapters: split its segments at those chapters
    for p in new:
        if len(p.get('entries', [])) != 1 or p['video'] not in metas:
            continue
        stretches = chapter_localities(video_chapters(metas[p['video']]), df[in_reach],
                                       (lambda name: town_near(name, home)) if home else None)
        if not stretches:
            continue
        e = p['entries'][0]
        if len(stretches) == 1:
            if not e.get('locality'):
                e['locality'], e['guessed'] = stretches[0][2], True
                p['note'] = f"{p['note']}; locality {stretches[0][2][0]} from the chapters".lstrip('; ')
            continue
        p['entries'] = split_by_stretches(e['segments'], stretches)
        p['note'] = (f"{p['note']}; localities from the chapters: "
                     + ', '.join(f'{loc[0]} from {a // 60}:{a % 60:02d}' for a, _, loc in stretches)).lstrip('; ')
    # a named place ("Sun Rise at Al Khan Beach") is in some town: look it up on OpenStreetMap
    for p in new:
        for e in p.get('entries', []):
            if e.get('locality'):
                continue
            description = '\n'.join(line for line in (metas.get(p['video'], {}).get('description') or '').splitlines()
                                    if line.strip() not in boilerplate and not ADDRESS_LINE.search(line))
            # every place the title names, then the description's first one; two towns in a title (a drive
            # from one to the other) decide nothing
            in_title = {}
            reach = in_reach | df['country'].isin(title_countries(p['title']))
            for loc, phrase in place_localities(place_phrases(p['title']), df[reach], home):
                in_title.setdefault(tuple(loc), phrase)
            area = DRIVE_AREA.search(p['title'])
            if len(in_title) > 1 and area:
                # "Driving from Longwood to West Farms in the Bronx, New York": the drive is in the Bronx
                inside = place_locality([area.group(1)] + [p_ for p_ in place_phrases(p['title'])
                                                           if p_.lower() in add_video.US_STATE_CODES], df[reach], home)
                if inside:
                    in_title = {tuple(inside[0]): inside[1]}
            if len(in_title) > 1:
                p['note'] = (f"{p['note']}; the title names " + ' and '.join(loc[0] for loc in in_title)
                             + ': set the locality').lstrip('; ')
                continue
            found = next(iter(in_title.items()), None)
            found = found or place_locality(place_phrases(description, limit=3), df[in_reach], home)
            if found:
                e['locality'], e['guessed'] = list(found[0]), True
                p['note'] = f"{p['note']}; locality {found[0][0]} from where '{found[1]}' is".lstrip('; ')
    # a drive "from A to B" without chapters passes through A, B and the towns between: split it along its route
    for p in new:
        split_gps(p, df, out_dir) or split_route(p, df[title_reach(df, in_reach, p['title'])], home)
    if home:
        iso2 = common.get_iso2_country_code(common.correct_country(home[2]))
        for p in new:
            for e in p.get('entries', []):
                loc = e.get('locality')
                if not loc or not e.get('guessed') or tuple(loc) == home[:3]:
                    continue
                _, row = add_video.get_existing_locality_row(df, *loc)
                if row is None or _km(home[3], home[4], float(row['lat']), float(row['lon'])) < 100:
                    continue
                # a title naming a state means that state: "Milton, Pennsylvania" is not the Milton near New York,
                # and "… in Arlington, Virginia" is no place near New York at all
                named = {code for name, code in add_video.US_STATE_CODES.items()
                         if re.search(rf'\b{re.escape(name)}\b|,\s*{code}\b', p['title'] or '', re.I)}
                if named and (loc[1] in named or home[1] not in named):
                    continue
                if _place_near(loc[0], iso2, home[3], home[4]):
                    p['note'] = f"{p['note']}; {loc[0]} here is the place of that name near {home[0]}, " \
                                f"not {', '.join(filter(None, loc))}".lstrip('; ')
                    e['locality'] = list(home[:3])
    # a video the title puts in one place ("Driving by Milton, Pennsylvania") is all in it: a quiet street there is no
    # rural or highway driving to leave out (Kutztown, Bellefonte, Milton); those checks are for drives between towns
    for p in new:
        es, sig_path = p.get('entries', []), os.path.join(out_dir, f"{p['video']}_signals.npz")
        if (len(es) == 1 and es[0].get('locality') and not p.get('route') and os.path.exists(sig_path)
                and not any(s.get('decision') or s.get('applied') for s in es[0]['segments'])):
            es[0]['segments'] = propose(dict(np.load(sig_path)), ONE_PLACE)[0]
            p['note'] = re.sub(r'; (?:rural|highway) driving \d+-\d+s', '', p.get('note') or '')
    for p in new:  # chapters whose places turned out the same ("Hollywood", "Los Angeles") are one group
        if len(p.get('entries', [])) > 1:
            merged = {}
            for e in p['entries']:
                key = tuple(e['locality']) if e.get('locality') else None
                if key in merged:
                    merged[key]['segments'] = sorted(merged[key]['segments'] + e['segments'], key=lambda s: s['start'])
                else:
                    merged[key] = e
            p['entries'] = list(merged.values())

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

    for p in new:
        # localities looked for; a cut-off run leaves this unset, and so does a lookup that failed (no connection, a
        # refusing server) for a video that got none: the next run tries it again
        if failures_before == LOOKUP_FAILURES[0] or any(e.get('locality') for e in p.get('entries', [])):
            p['located'] = True
    commit(touched)
    kept = sum('entries' in p for p in new)
    print(f'{len(new)} videos added or re-analysed: {kept} proposed, {len(new) - kept} excluded -> {plan_path}')
    if stopped:
        print(f'Stopped early: {stopped}. Run the same command later to continue.')
    left = sum(v not in in_mapping and (v not in plan or (v in reanalyse and v not in touched)) for v in ids)
    if left:
        print(f'{left} videos of this channel are still to do: run the same command again for the next batch.')
    elif not stopped:
        print(CHANNEL_DONE)  # run_channels.py moves on to the next channel only on this line
    return plan_path


# the codes the add-video form accepts
VEHICLE_TYPES = {0: 'car', 1: 'bus', 2: 'truck', 3: 'two-wheeler', 4: 'bicycle/e-bicycle', 5: 'automated car',
                 6: 'electric scooter', 7: 'monowheel/unicycle', 8: 'emergency vehicle', 9: 'automated bus',
                 10: 'automated truck', 11: 'automated two-wheeler', 12: 'non-electric scooter'}

# the country names the add-video form offers, read from its template so both stay the same list
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'add_video.html'),
          encoding='utf-8') as _f:
    COUNTRIES = re.findall(r'<option value="([^"]+)" \{\{ "selected" if country ==', _f.read())

REVIEW_PAGE = 50  # videos with something left to decide shown at once in the review page
MAX_MERGE_GAP_S = 2  # wider gaps are excluded footage (stops, freeway), not suggested splits
CHANNEL_DONE = 'Every video of this channel is proposed or in the mapping.'

LOCALITY_FIELDS = ('lat', 'lon', 'gmp', 'population_locality', 'population_country', 'traffic_mortality',
                   'continent', 'locality_aka', 'literacy_rate', 'avg_height', 'med_age', 'gini', 'traffic_index')


def apply_segments(video_id, locality, state, country, segments, vehicle_type, upload_date, channel_id, near=None,
                   at=None):
    """Add reviewed segments to a locality through the add-video form's own submit path, so its overlap and
    value checks apply. A locality not in the mapping yet is created from the same lookups the form makes.
    at=(lat, lon): where the reviewer put a new locality on the map. Writes the mapping file."""
    client = add_video.app.test_client()
    new_row = None
    for seg in segments:
        _, row = add_video.get_existing_locality_row(add_video.load_csv(add_video.FILE_PATH), locality, state, country)
        if row is None:
            new_row = new_row or add_video.new_locality_row(locality, state or None, country, near=near, at=at)
            row = new_row
        # the form re-saves the locality fields it shows, so send them back exactly as it would render them
        form = {k: str(row[k]) for k in LOCALITY_FIELDS}
        form.update({
            'locality': row['locality'], 'state': '' if state is None else state, 'country': row['country'],
            'video_url': _url(video_id), 'time_of_day': str(seg['night']), 'vehicle_type': str(vehicle_type),
            'start_time': str(seg['start']), 'end_time': str(seg['end']),
            'upload_date_video': upload_date or '', 'channel_video': channel_id or '', 'submit_data': '1',
        })
        # one change to the mapping at a time across processes: two review pages applying at once would each save
        # the mapping they read, and the later save would drop the other's rows
        with plan_lock(add_video.FILE_PATH):
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
.locmap { flex-basis: 100%; font-size: 12px; color: #666; }
.locmap .leafmap { width: 420px; height: 240px; border: 1px solid #ccc; margin-top: 4px; }
/* Leaflet's own layers reach z-index 1000: keep them under the sticky player and the locality suggestions */
.routemap, .locmap .leafmap { position: relative; z-index: 0; }
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
/* its own stacking context: Leaflet's layers (z-index 400+) would otherwise scroll over the sticky player */
.routemap { width: 640px; max-width: 100%; height: 300px; margin: 8px 0; border: 1px solid #ccc;
            position: relative; z-index: 0; }
.routemap .leaflet-tooltip { font-size: 11px; padding: 1px 4px; }
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head><body>
<header>
  <div id="player"></div>
  <div>
    <div class="clock">Time: <span id="clock" title="click to copy">0:00</span> <span id="copied"></span></div>
    <div class="keys">Keys for the outlined segment: <b>A</b> start here · <b>S</b> end here ·
      <b>D</b> whole video · <b>Q</b> day · <b>W</b> night · <b>F</b> split time here</div>
    <div id="status"></div>
    {% if waiting > open_videos | length %}<div class="keys">Showing {{ open_videos | length }} of {{ waiting }}
      videos left to decide; the next appear as you finish these.</div>{% endif %}
    <button id="apply" disabled>Apply approved segments to the mapping</button>
    <div id="log"></div>
  </div>
</header>
<main>
{% for p in plan if p.entries and p.video in open_videos %}
<div class="video" data-duration="{{ durations.get(p.video) or '' }}"><div class="main">
  <h3>{{ p.title }}</h3>
  <a href="https://www.youtube.com/watch?v={{ p.video }}" target="_blank">{{ p.video }}</a>
  <label class="vehicle">Vehicle (whole video):
    <select autocomplete="off" onchange="setVehicle('{{ p.video }}', this)">
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
  {% if p.route %}{% set times = {} %}{% for e in p.entries if e.locality %}{% set _ = times.setdefault(
       e.locality[0], []).extend(e.segments | rejectattr('decision', 'equalto', 'reject')) %}{% endfor %}
  <div class="routemap" data-route='{{ p.route | tojson }}' data-times='{{ times | tojson }}'
       title="the car route the localities were estimated from"></div>{% endif %}
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
        <select class="loc-country" autocomplete="off" onfocus="fillCountries(this)" onmousedown="fillCountries(this)"
            onchange="saveLocality(this)"><option value="">Country</option>
          {% set known = p.entries | selectattr('locality') | map(attribute='locality') | list %}
          {%- set c = e.locality[2] if e.locality else known[0][2] if known else video_country[p.video] %}
          {%- if c %}<option selected>{{ c }}</option>{% endif %}</select>
        <span class="locstatus guess">{% if e.guessed %}guessed from title{% elif not e.locality %}
          ⚠ not set: approved segments cannot be added{% endif %}</span>
        <div class="locmap"{% if e.locality and e.locality | tojson in new_localities %}
             data-new="1"{% endif %}></div>
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
          <input class="t-start" value="{{ t0 }}" title="start (m:ss)" autocomplete="off"
                 onchange="editTimes(this)"> –
          <input class="t-end" value="{{ t1 }}" title="end (m:ss)" autocomplete="off"
                 onchange="editTimes(this)">{% endif %}
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
          <input class="at" placeholder="m:ss"
                 autocomplete="off" title="time to split at, e.g. where the drive enters another locality">
          <button class="small" onclick="splitAt(this)">Split</button>
          <button class="small" onclick="moveSegment(this)" title="put this segment under another locality">
          → other locality</button>{% endif %}</td>
    </tr>
    {% if not loop.last %}{% set n = e.segments[loop.index] %}
    {% if not s.applied and not n.applied %}
    {% set split = n.start - s.end <= max_merge_gap and n.night == s.night %}
    <tr class="split">
      <td colspan="2">{% if split %}suggested split at {{ '%d:%02d' % (s.end // 60, s.end % 60) }}
        {% elif n.start > s.end %}left out {{ '%d:%02d' % (s.end // 60, s.end % 60) }}–{{
          '%d:%02d' % (n.start // 60, n.start % 60) }}{% else %}{% endif %}</td>
      <td><button class="small" onclick="play('{{ p.video }}', {{ [s.end - 5, 0] | max }})">▶ {{
          'split' if split else 'gap' }}</button></td>
      <td><button class="small"
          onclick="mergeSplit(this, '{{ p.video }}', {{ ei }}, {{ loop.index0 }})"{% if not split %}
          title="one segment from {{ '%d:%02d' % (s.start // 60, s.start % 60) }} to {{
          '%d:%02d' % (n.end // 60, n.end % 60) }}, the left-out part and the first one's day/night included"
          {% endif %}>{{ 'Remove split' if split else 'Join' }}</button></td>
    </tr>
    {% endif %}{% endif %}
  {% endfor %}
    <tr class="addrow"><td colspan="5">
      <button class="small" data-video="{{ p.video }}" data-entry="{{ ei }}" onclick="newSegment(this)"
              title="from the current video time (or after the last segment) to the next segment or the end">
        + new segment{{ ' in ' ~ e.locality[0] if e.locality }}</button></td></tr>
  {% endfor %}
  </table>
  <button class="small" data-video="{{ p.video }}" onclick="addLocality(this)"
          title="an empty locality group: pick its locality, then add segments to it with + new segment">
    + add a locality</button>
  <button class="small" onclick="approveRest(this)">Approve the rest of this video</button>
  <button class="small" onclick="rejectVideo(this)">Reject whole video</button>
  </div>
  <div class="desc">{% set d = descriptions.get(p.video) %}
    {% if d %}
      <div class="desctext">{{ d.html }}</div>
      <div class="chapters" data-video="{{ p.video }}"{% if not d.fetched %} data-pending="1"{% endif %}>
        {% if d.chapters %}<b>Chapters</b>{% for t, title in d.chapters %}
        <div><a href="#" onclick="play('{{ p.video }}', {{ t }}); return false;">
          {{ '%d:%02d' % (t // 60, t % 60) }}</a> {{ title }}</div>{% endfor %}{% endif %}</div>
    {% else %}<span class="excluded">no description saved for this video</span>{% endif %}
  </div>
</div>
{% endfor %}
{% if not open_videos %}<p>Nothing left to decide in this plan.</p>
{% elif waiting > open_videos | length %}<p class="more">{{ waiting - open_videos | length }} more videos to decide:
  they appear here as you finish these {{ open_videos | length }} (reload the page).</p>{% endif %}
<details><summary>Excluded videos ({{ plan | selectattr('exclude') | list | length }})</summary>
{% for p in plan if p.exclude %}
<div class="excluded"><a href="https://www.youtube.com/watch?v={{ p.video }}" target="_blank">{{ p.video }}</a>
  {{ p.title }} — {{ p.exclude }}</div>
{% endfor %}
</details>
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
  // the video already in the player: jump there (reloading it can start it at 0:00 instead)
  if (playingVideo() === video) { ytPlayer.seekTo(Math.max(0, Math.floor(t)), true); ytPlayer.playVideo(); return; }
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
  // the undecided segment of the playing video that the current time is in, or the nearest one; a decided
  // segment only when the video has no undecided one left
  const video = ytReady && playingVideo();
  if (!video) return null;
  const t = ytPlayer.getCurrentTime();
  const rows = [...document.querySelectorAll(`tr.seg:not(.applied)[data-video="${video}"]`)];
  const open = rows.filter(r => !r.classList.contains('approve') && !r.classList.contains('reject'));
  let best = null, bestDistance = Infinity;
  for (const r of open.length ? open : rows) {
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
let stepSave;
document.addEventListener('keydown', e => {
  // up/down arrows in a time box step it by a second, show that moment, and save once the stepping pauses
  const input = e.target;
  if (!['ArrowUp', 'ArrowDown'].includes(e.key) || !input.matches('input.t-start, input.t-end, input.at')) return;
  const t = parseTime(input.value);
  if (!Number.isFinite(t)) return;
  e.preventDefault();
  const next = Math.max(0, t + (e.key === 'ArrowUp' ? 1 : -1));
  input.value = fmt(next);
  const row = input.closest('tr');
  if (ytReady && playingVideo() === row.dataset.video) ytPlayer.seekTo(next, true);
  if (input.matches('.at')) return;
  clearTimeout(stepSave);
  stepSave = setTimeout(() => editTimes(input), 600);
});
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
  // the saved length: right after a switch the player still reports the previous video's
  const length = +row.closest('.video').dataset.duration || Math.floor(ytPlayer.getDuration());
  if (key === 'd') setBounds(row, 0, length - 1);
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
const STATE_COUNTRIES = {{ state_countries | tojson }};
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
  // the mapping writes a state for these countries: wait for it instead of refusing the town typed first
  if (!state && STATE_COUNTRIES.includes(country)) {
    status.textContent = '⚠ add the state (e.g. PA) to save';
    return;
  }
  const res = await fetch('locality', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: box.dataset.video, entry: +box.dataset.entry,
                           locality: [locality, state, country] }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  const out = await res.json();
  if (out.merged) { reloadKeepingPlace(); return; }  // joined another group of this video with that locality
  status.textContent = out.new ? 'new locality: created on Apply, with the lookups the add-video form makes'
                               : '✓ in the mapping';
  showMap(box, out.new);
}
async function showMap(box, isNew) {
  // a locality not in the mapping yet: where the add-video lookups put it, to check before Apply creates it
  const map = box.querySelector('.locmap');
  map.innerHTML = '';
  if (!isNew) return;
  const q = new URLSearchParams({ locality: box.querySelector('.loc-name').value.trim(),
    state: box.querySelector('.loc-state').value.trim(), country: box.querySelector('.loc-country').value,
    video: box.dataset.video, entry: box.dataset.entry });
  map.textContent = 'finding it on the map…';
  const res = await fetch('where?' + q);
  const out = await res.json();
  if (!res.ok) { map.textContent = '⚠ ' + out.error; return; }
  map.innerHTML = '<span class="where"></span><div class="leafmap"></div>';
  const where = map.querySelector('.where');
  const say = (lat, lon, moved) => where.textContent =
    `new locality at ${lat.toFixed(4)}, ${lon.toFixed(4)}` +
    (moved ? ' (moved by hand)' : ': drag the marker if it is off');
  say(out.lat, out.lon, out.moved);
  if (!window.L) { where.textContent += ' (the map needs internet: Leaflet from unpkg.com)'; return; }
  const leaf = L.map(map.querySelector('.leafmap'), { scrollWheelZoom: false }).setView([out.lat, out.lon], 12);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              { maxZoom: 18, attribution: '&copy; OpenStreetMap contributors' }).addTo(leaf);
  const marker = L.marker([out.lat, out.lon], { draggable: true }).addTo(leaf);
  marker.on('dragend', async () => {
    const { lat, lng } = marker.getLatLng();
    const res = await fetch('at', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video: box.dataset.video, entry: +box.dataset.entry, lat, lon: lng }) });
    if (!res.ok) { alert((await res.json()).error); return; }
    say(lat, lng, true);
  });
}
document.querySelectorAll('.locmap[data-new]').forEach(m => showMap(m.closest('.locbox'), true));
// YouTube's chapters, fetched once per video and one video at a time so the page stays quick
(async () => {
  for (const box of document.querySelectorAll('.chapters[data-pending]')) {
    try {
      const list = await (await fetch('chapters/' + box.dataset.video)).json();
      if (!list.length) continue;
      box.innerHTML = '<b>Chapters</b>' + list.map(([t, title]) =>
        `<div><a href="#" onclick="play('${box.dataset.video}', ${t}); return false;">${fmt(t)}</a> ` +
        `${title.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c])}</div>`).join('');
    } catch (e) { /* no connection: the next page load tries again */ }
  }
})();
// after a reload (a join, a split) the browser refills inputs by position, so rows that moved would show another
// row's times, which a later edit would then save: always show the saved ones
window.addEventListener('pageshow', () => {
  document.querySelectorAll('tr.seg').forEach(r => { if (r.querySelector('.t-start')) showBounds(r); });
  document.querySelectorAll('input.at').forEach(i => { i.value = ''; });
  // and the saved country and vehicle, not what the browser restored by position ("Country" after a reload)
  document.querySelectorAll('select').forEach(s => {
    const saved = [...s.options].find(o => o.defaultSelected);
    if (saved) s.value = saved.value;
  });
});
function parseTime(text) {
  const parts = text.trim().split(':').map(Number);
  return parts.length && parts.every(n => Number.isFinite(n)) ? parts.reduce((a, b) => a * 60 + b, 0) : NaN;
}
function reloadKeepingPlace(video, t) {
  // after a reload, come back to the same video and moment instead of an empty player
  if (!video && ytReady && playingVideo()) { video = playingVideo(); t = ytPlayer.getCurrentTime(); }
  // nothing playing: come back to the same scroll position
  sessionStorage.setItem('resume', JSON.stringify(video ? { video, t: t || 0 } : { y: window.scrollY }));
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
async function mergeSplit(btn, video, entry, seg) {
  // the segments' times as they are now: A, S, D and the time boxes change them without a reload
  const row = i => document.querySelector(`tr.seg[data-video="${video}"][data-entry="${entry}"][data-seg="${i}"]`);
  const bounds = [row(seg), row(seg + 1)].map(r => [+r.dataset.start, +r.dataset.end]);
  btn.disabled = true;  // a second click would send the segments that are already joined
  const res = await fetch('merge', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video, entry, seg, bounds }) });
  if (!res.ok) { btn.disabled = false; alert((await res.json()).error); return; }
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
async function addLocality(btn) {
  const res = await fetch('add_locality', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: btn.dataset.video }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  reloadKeepingPlace();
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
    `${rows.length} segments to review: ${yes} approved, ${no} rejected, ${open} undecided` +
    (APPROVED_HIDDEN ? ` (+${APPROVED_HIDDEN} approved in finished videos, waiting for Apply)` : '');
  document.getElementById('apply').disabled = yes + APPROVED_HIDDEN === 0;
}
const APPROVED_HIDDEN = {{ approved_hidden }};  // approved segments of finished videos, which are not shown
document.getElementById('apply').onclick = async () => {
  const rows = [...document.querySelectorAll('tr.seg:not(.applied)')];
  const yes = rows.filter(r => r.classList.contains('approve')).length + APPROVED_HIDDEN;
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
  const row = resume.video && document.querySelector(`tr.seg[data-video="${resume.video}"]`);
  const scroll = () => row ? row.closest('.video').scrollIntoView({ block: 'start' })
                           : window.scrollTo(0, resume.y || 0);
  scroll();
  window.addEventListener('load', () => setTimeout(scroll, 50));
  if (resume.video) play(resume.video, resume.t);
}
</script>
<script>
// a drive "from A to B": its car route, with where each locality on the way starts and its segments' times
document.querySelectorAll('.routemap').forEach(el => {
  if (!window.L) { el.textContent = 'the map needs internet (Leaflet from unpkg.com)'; return; }
  const route = JSON.parse(el.dataset.route), times = JSON.parse(el.dataset.times);
  const mss = t => `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
  const map = L.map(el, { scrollWheelZoom: false });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              { maxZoom: 18, attribution: '&copy; OpenStreetMap contributors' }).addTo(map);
  const line = L.polyline(route.path, { color: '#2a6fdb', weight: 4 }).addTo(map);
  route.stops.forEach(([lat, lon, name], i) => {
    L.circleMarker([lat, lon], { radius: 5, color: i ? '#c0392b' : '#27ae60', fillOpacity: 1 })
      .bindTooltip(`${i + 1}. ${name}` +
                   (times[name] || []).map((s, j) => `${j ? ',' : ''} ${mss(s.start)}–${mss(s.end)}`)
                    .join(''), { permanent: true, direction: 'right' }).addTo(map);
  });
  map.fitBounds(line.getBounds(), { padding: [20, 20] });
});
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
        if not os.path.exists(plan_path):  # a channel run that has not finished its first video yet
            return []
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
        # the first REVIEW_PAGE videos with something left to decide; finished ones stay in the plan but off the
        # page, and make room for the next on reload
        plan = load()
        segs = {p['video']: [s for e in p['entries'] for s in e['segments']] for p in plan if 'entries' in p}
        waiting = [v for v, ss in segs.items() if any(not s.get('decision') and not s.get('applied') for s in ss)]
        open_videos = set(waiting[:REVIEW_PAGE])
        approved_hidden = sum(s.get('decision') == 'approve' and not s.get('applied')
                              for v, ss in segs.items() if v not in open_videos for s in ss)
        descriptions = {v: describe(v) for v in open_videos}
        mapping = add_video.load_csv(add_video.FILE_PATH)
        shown = {tuple(e['locality']) for p in plan if p['video'] in open_videos
                 for e in p['entries'] if e.get('locality')}
        new_localities = {json.dumps(list(loc)) for loc in shown
                          if add_video.get_existing_locality_row(mapping, *loc)[0] is None}
        # a new locality group starts in the country of the video's other groups, else the one its title names
        # ("…,Puerto Rico"), else the batch's usual one
        usual = (Counter(e['locality'][2] for p in plan for e in p.get('entries', []) if e.get('locality'))
                 .most_common(1) or [(None,)])[0][0]
        video_country = {p['video']: next(iter(title_countries(p.get('title'))), usual)
                         for p in plan if p['video'] in open_videos}
        return render_template_string(REVIEW_HTML, plan=plan, open_videos=open_videos, approved_hidden=approved_hidden,
                                      video_country=video_country,
                                      state_countries=sorted(c for c, has in mapping.groupby('country')['state']
                                                             .agg(lambda x: x.notna().mean() > 0.5).items() if has),
                                      new_localities=new_localities, waiting=len(waiting),
                                      durations={v: video_duration(v) for v in open_videos},
                                      max_merge_gap=MAX_MERGE_GAP_S, countries=COUNTRIES, vehicle_types=VEHICLE_TYPES,
                                      descriptions={v: d for v, d in descriptions.items() if d})

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

    def overlap(p, seg, start, end):
        """Error text when start-end would overlap another segment of the video, else None. Rejected segments
        never reach the mapping, so a segment may take over their time."""
        for e in p['entries']:
            for o in e['segments']:
                if o is not seg and o.get('decision') != 'reject' and start < o['end'] and o['start'] < end:
                    return (f"that would overlap this video's segment {o['start'] // 60}:{o['start'] % 60:02d}"
                            f"–{o['end'] // 60}:{o['end'] % 60:02d}")
        return None

    def video_duration(video_id):
        """The video's length in s from its saved metadata, or None."""
        path = os.path.join(plan_dir, f'{video_id}_meta.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            duration = json.load(f).get('duration')
        return int(duration) if duration else None

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
        chapters = video_chapters(meta)  # YouTube's, else the description's timestamped lines
        return {'html': Markup('').join(parts), 'chapters': chapters, 'fetched': 'yt_chapters' in meta}

    @app.route('/chapters/<video_id>')
    def chapters(video_id):
        """The video's chapters, fetched from YouTube once (the Data API that listed the channel has none)."""
        path = os.path.join(plan_dir, f'{video_id}_meta.json')
        if not re.fullmatch(r'[\w-]{11}', video_id) or not os.path.exists(path):
            return jsonify([])
        with open(path) as f:
            meta = json.load(f)
        if 'yt_chapters' not in meta:
            try:
                meta['chapters'] = fetch_metadata(video_id)['chapters']
            except Exception:  # a bot check or no connection: try again on the next page load
                return jsonify(video_chapters(meta))
            meta['yt_chapters'] = True
            with open(path + '.tmp', 'w') as f:
                json.dump(meta, f)
            os.replace(path + '.tmp', path)
        return jsonify(video_chapters(meta))

    @app.route('/bounds', methods=['POST'])
    def bounds():
        """New start and end for a segment (the A, S and D keys); it may not overlap the video's other segments
        that are not rejected."""
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
        length = video_duration(p['video'])
        if length and end > length:
            return jsonify(error=f'the video is only {length // 60}:{length % 60:02d} long'), 400
        if s.get('decision') != 'reject' and (error := overlap(p, s, start, end)):
            return jsonify(error=error), 400
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
        taken = sorted((s['start'], s['end']) for x in p['entries'] for s in x['segments']
                       if s.get('decision') != 'reject')
        start = int(d['at']) if d.get('at') is not None else max((b for _, b in taken), default=0)
        if any(a <= start < b for a, b in taken):
            return jsonify(error=f'{start // 60}:{start % 60:02d} is inside an existing segment: split that '
                                 'segment instead, or play a part of the video no segment covers'), 400
        # the saved length first: right after a switch the player still reports the previous video's
        duration = video_duration(p['video']) or d.get('duration')
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

    @app.route('/add_locality', methods=['POST'])
    def add_locality():
        """An empty locality group for a video, for a locality its proposal missed; segments come with + new
        segment, or move in with → other locality."""
        d = request.get_json()
        plan = load()
        p = next((p for p in plan if p['video'] == d['video'] and 'entries' in p), None)
        if p is None:
            return jsonify(error=stale), 409
        if any(not e['segments'] and not e.get('locality') for e in p['entries']):
            return jsonify(error='this video already has an empty group: set its locality first'), 400
        p['entries'].append({'locality': None, 'guessed': False, 'segments': []})
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

    @lru_cache(maxsize=None)
    def coordinates(locality, state, country):
        country = common.correct_country(country)
        data = add_video.get_locality_data(locality, common.get_iso2_country_code(country), state)
        return add_video.get_coordinates(locality, state, country, data)

    @app.route('/where')
    def where():
        """Coordinates of a locality not in the mapping yet: where the marker was dragged to, else found as the
        add-video form will find them."""
        locality, country = request.args.get('locality', ''), request.args.get('country', '')
        p = next((p for p in load() if p['video'] == request.args.get('video') and 'entries' in p), None)
        entry = int(request.args.get('entry', -1))
        if p and 0 <= entry < len(p['entries']) and p['entries'][entry].get('at'):
            lat, lon = p['entries'][entry]['at']
            return jsonify(lat=lat, lon=lon, moved=True)
        # where the route put it (a drive through several towns): Apply takes it when the lookup by name lands
        # over 15 km away, at a namesake
        stops = ((p or {}).get('route') or {}).get('stops', [])
        near = next(((la, lo) for la, lo, name in stops if name == locality), None)
        try:
            lat, lon = coordinates(locality, request.args.get('state') or None, country)
        except Exception as e:  # the lookups raise anything from network errors to missing fields
            if not near:
                return jsonify(error=f'could not find {locality} ({e})'), 404
            lat = lon = None
        if near and (lat is None or lon is None or _km(float(lat), float(lon), *near) > 15):
            lat, lon = near
        if lat is None or lon is None:
            return jsonify(error=f'could not find {locality}, {country} on the map'), 404
        return jsonify(lat=float(lat), lon=float(lon))

    @app.route('/at', methods=['POST'])
    def set_at():
        """Where a new locality is, as dragged on its map: for every group of the plan with that locality."""
        d = request.get_json()
        plan = load()
        p = next((p for p in plan if p['video'] == d['video'] and 'entries' in p), None)
        if p is None or d['entry'] >= len(p['entries']) or not p['entries'][d['entry']].get('locality'):
            return jsonify(error=stale), 409
        loc = p['entries'][d['entry']]['locality']
        for e in (e for q in plan for e in q.get('entries', []) if e.get('locality') == loc):
            e['at'] = [round(float(d['lat']), 6), round(float(d['lon']), 6)]
        save(plan)
        return jsonify(ok=True)

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
        rows = add_video.load_csv(add_video.FILE_PATH)
        rows = rows[rows['country'] == country]
        if not state and len(rows) and rows['state'].notna().mean() > 0.5:
            # without it Apply would make a second Northampton next to Northampton, MA
            return jsonify(error=f'give the state too: the mapping writes one for {country} (e.g. PA)'), 400
        same = [o for o in p['entries'] if o is not e and o.get('locality') == [locality, state, country]]
        if same:  # this video already has a group for that locality: join it
            same[0]['segments'] = sorted(same[0]['segments'] + e['segments'], key=lambda s: s['start'])
            p['entries'].remove(e)
            save(plan)
            return jsonify(ok=True, merged=True)
        e['locality'], e['guessed'] = [locality, state, country], False
        # a marker dragged for this locality elsewhere in the plan holds here too; one for the old locality not
        at = next((o['at'] for q in plan for o in q.get('entries', []) if o is not e and o.get('at')
                   and o.get('locality') == e['locality']), None)
        e.pop('at', None)
        if at:
            e['at'] = at
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
        # joining over a gap or a day/night change takes in the left-out footage and the first one's time of day
        segs[i:i + 2] = [{'start': min(a['start'], b['start']), 'end': max(a['end'], b['end']), 'night': a['night']}]
        if 0 <= b['start'] - a['end'] <= MAX_MERGE_GAP_S and a['night'] == b['night']:
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
        if d['decision'] != 'reject' and (error := overlap(p, seg, seg['start'], seg['end'])):
            return jsonify(error=f'{error}: shorten one of them first'), 400
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
                        # where the route put this locality: a new locality is located there, not at a namesake
                        near = next(((lat, lon) for lat, lon, name in (p.get('route') or {}).get('stops', [])
                                     if name == e['locality'][0]), None)
                        apply_segments(p['video'], *e['locality'], [s], p['vehicle_type'],
                                       p['upload_date'], p['channel_id'], near, e.get('at'))
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
    ap.add_argument('--single-city', action='store_true',
                    help='leave out drives from one town to another ("from A to B"): only drives within one city')
    ap.add_argument('--review', metavar='PLAN_JSON', help='open the review page for a plan file')
    ap.add_argument('--split-routes', metavar='PLAN_JSON',
                    help='split untouched drives "from A to B" in a plan into the localities along their route')
    args = ap.parse_args()
    if args.review:
        review(args.review)
    elif args.split_routes:
        print(f'{split_routes(args.split_routes)} drives split along their route in {args.split_routes}')
    elif args.channel:
        name = re.sub(r'\W+', '_', re.sub(r'^.*youtube\.com/(?:channel/)?@?', '', args.channel).split('/')[0])
        out = process_channel(args.channel, args.country, args.out or os.path.join('_output/proposals', name),
                              args.pause, args.limit, download=not args.no_download, single_city=args.single_city)
        if out:
            with open(out) as f:
                if any('entries' in p for p in json.load(f)):
                    print(f'Review with: python propose_segments.py --review {out}')
    else:
        ap.print_help()
