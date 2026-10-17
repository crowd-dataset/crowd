"""Propose CROWD segments for a YouTube video from its pixels, for a curator to review.

Signals come from a low resolution copy decoded at a few frames per second:
camera motion (to trim stationary stretches), hard cuts and blank frames
(edited discontinuities), and sky brightness (day/night, split at transitions).
Vehicle type comes from the channel's history in the mapping file.
Nothing here writes to the mapping file.
"""
import argparse
import ast
import contextlib
import datetime
import fcntl
import glob
import difflib
import functools
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
                       GeocoderRateLimited, GeocoderServiceError, GeopyError)
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
    'glare_spike': 1.25,       # a cut whose sky flashes this much brighter that briefly is glare
    'short_dip': 0.8, 'short_dip_s': 1.5,  # ... or under this for at most this long (see propose)
    'cut_confirm_ratio': 15,  # a cut stands only with a full-frame-rate jump this strong within a second
    # a map shown before the drive (brittrips: 4mGZZ9BGunA, aERXkCaV9BA, xkOEm0jFTcE) is far more colourful than
    # road footage: a first colour sample above intro_saturation and intro_saturation_ratio times the video's
    # median (maps 182-232 vs 45-100; 592 videos: 95% of first samples under 117) is an intro, which ends at the
    # last edit cut before forward driving starts, within intro_max_s
    'intro_saturation': 150,
    'intro_saturation_ratio': 2,
    'intro_max_s': 60,
    'cut_pan_ncc': None,      # a cut that matches this well once shifted sideways is a corner (not yet calibrated)
    'blank_std': 6.0,         # frame this flat is blank or a title card
    # night: the sky darker than this and the street darker than night_ground. On the 1,850 approved segments of a
    # minute or more (250 night; 2026-10-11) this mislabels 4.4% of the driving time (13 day segments called night,
    # 49 night ones day) against 6.7% for "sky under 100 or street under 65" (50 and 37): bright HDR or hazy daylight
    # under a slightly dim sky is no night (Va9g8hoDg0g, Blue Mountains: sky 86-100, street 77-111)
    'night_sky': 85.0,
    'day_inside_night_s': 1800,  # a bright spell this short between night stretches stays night (lit streets)
    'night_ground': 90.0,
    'yolo_day_sky': 100.0,    # the rural and highway checks (YOLO counts) only where the sky is this bright
    'night_window_s': 60,     # night/day must hold this long to count as a transition
    # dark this long or shorter with daylight before and after is a bridge, tunnel or elevated railway (10 min under
    # the 7 train on Roosevelt Ave): night falls once and does not lift again within a drive
    'night_inside_day_s': 3 * 3600,
    'night_edge_s': 300,
    # dark that CLIP's NIGHT_SCENES read at least this much as night is no bridge: on approved segments, night
    # called day 12378 s without, 7073 s with 0.5, day called night 0 s either way (user, 2026-10-16)
    'bridge_clip_night': 0.5,
    'stitch_jumps': False,    # strong frame jumps count as stitches too (see propose)
    'night_majority': False,  # a mostly dark video: lit streets before bridges (see propose)
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
    'angle_share': 0.6,
    'angle_relative': None,   # ... and under this times the video's own median (see _sideways)
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
    'tilt_up': 0.85,          # median height of road users (0 top, 1 bottom) past which the camera looks up
    'tilt_down': 0.3,         # ... and under which it looks down at the dashboard
    'stop_churn': 0.4,        # share of stopped frames changing fast (ncc < 0.9) for the sped-up hint
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
    # H.264 first: OpenCV (YOLO, CLIP frames) cannot decode AV1 ("Missing Sequence Header", AbaoAmbience)
    _yt_dlp('-q', '-f', 'bv*[height<=240][vcodec^=avc1]/bv*[height<=360][vcodec^=avc1]/bv*[height<=240][ext=mp4]'
            '/bv*[height<=360]/wv*', '-o', template, _url(video_id))
    for name in os.listdir(out_dir):
        if name.startswith(f'{video_id}_low.'):
            return os.path.join(out_dir, name)
    raise RuntimeError(f'download produced no file for {video_id}')


def letterbox(path, samples=30, black=12):
    """(top, bottom) shares of the picture height that are black bars (a letterboxed video: XqNV7dDs5hs), else
    (0, 0). Bars are rows black in every sampled frame, about as high at the top as at the bottom: a dark night
    sky is only at the top, so it is never taken for one."""
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', 'thumbnail=300,scale=160:90,format=gray',
           '-frames:v', str(samples), '-f', 'rawvideo', '-']
    g = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.uint8)
    if g.size < 160 * 90:
        return 0.0, 0.0
    rows = g[:g.size // (160 * 90) * 160 * 90].reshape(-1, 90, 160).mean(axis=2).max(axis=0)  # brightest per row
    dark = rows < black
    top = int(np.argmin(dark)) if not dark.all() else 0
    bottom = int(np.argmin(dark[::-1])) if not dark.all() else 0
    if min(top, bottom) < 2 or abs(top - bottom) > 3:
        return 0.0, 0.0
    return top / 90, bottom / 90


def frames(path, bars=(0.0, 0.0)):
    """Yield grey analysis frames at FPS, without letterbox bars (bars: top and bottom shares, see letterbox)."""
    top, bottom = bars
    crop = f'crop=iw:ih*{1 - top - bottom:.4f}:0:ih*{top:.4f},' if top or bottom else ''
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', f'{crop}fps={FPS},scale={W}:{H},format=gray',
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


WALK_SCENES = (['a photo taken while walking on a sidewalk', 'a photo of a pedestrian path in a park',
                'a photo taken inside a shop or building', 'a photo of a hiking trail',
                'a photo of a road taken through the windshield of a car', 'a dashcam photo of a street',
                'a dashcam photo of a highway'], 4)
WALK_SHEET_FRAMES = 5  # this many of a contact sheet's 12 frames on foot: a walk (approved drives: 3 at most)


def sheet_scores(sheet_path, scenes, prefix='', bottom=False):
    """CLIP's score for scenes' first ones on each frame of a contact sheet (bottom: its lower half only), [] without
    CLIP."""
    sheet = cv2.imread(sheet_path)
    if sheet is None or not os.path.exists(YOLO_PYTHON):
        return []
    work_dir = tempfile.mkdtemp(prefix='crowd_sheet_')
    try:
        tiles = []
        for r in range(0, sheet.shape[0], 180):
            for c in range(0, sheet.shape[1], 320):
                tiles.append(os.path.join(work_dir, f'tile{len(tiles)}.jpg'))
                cv2.imwrite(tiles[-1], sheet[r + (90 if bottom else 0):r + 180, c:c + 320])
        ask = json.dumps({'scenes': scenes[0], 'k': scenes[1], 'prefix': prefix})
        r = subprocess.run([YOLO_PYTHON, '-c', SCENE_CODE, ask, *tiles], capture_output=True, text=True, cwd=HERE)
        scores = [float(x) for x in r.stdout.split()]
        if r.returncode or len(scores) != len(tiles):
            print(f'  CLIP failed, no contact sheet check: {r.stderr.strip()[-300:]}', flush=True)
            return []
        return scores
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# step_bob: left out as a walk (not offered for review: user, 2026-10-16; of 649 reviewed videos, 0.65+ in 16
# rejected and no approved one, whose highest is 0.62) / a note to check
STEP_BOB_WALK, STEP_BOB_HINT = 0.65, 0.6


BOB_EVERY_S = 45  # step_bob_track: one 15-s reading this often through the video
# a town's streets by sight (user, 2026-10-13: y0VgKTLKsp4's quiet Washington neighbourhood cut as highway): on 44
# stretches the highway and rural checks cut, CLIP's middle score on 3 frames was 0.95 or more for 10 of the 21 the
# reviewer approved anyway and 2 of the 23 they left cut
TOWN_SCENES = (['a residential street with houses', 'a quiet suburban street lined with houses and trees',
                'a street in a town with shops and buildings', 'a highway with several lanes', 'a freeway',
                'a country road through fields', 'a road through a forest', 'a mountain road',
                'a road through the desert'], 3)
TOWN_EVERY_S, TOWN_SCORE = 30, 0.95


def town_scene_track(path, duration, work_dir):
    """[(second, CLIP score of TOWN_SCENES)] every TOWN_EVERY_S through the video ([] without CLIP)."""
    times = [float(t) for t in np.arange(TOWN_EVERY_S / 2, duration or 0, TOWN_EVERY_S)]
    scores = _clip_scores(path, times, work_dir, TOWN_SCENES) if times else {}
    return sorted(scores.items())


# night by sight (user, 2026-10-13: eq-5HRmOiN4's first 14 min under the El by day, as dark as night): on frames of
# the dark approved segments (4 each), CLIP's middle score for the first three was 0.12 or more on all 56 the
# reviewer marked night and 0.01 or less on 13 of the 16 they marked day; the orange-lamp measure was 0 on 100 of
# 232 night ones, no help
NIGHT_SCENES = (['a city street at night with the street lights on', 'a dark road at night lit by headlights',
                 'a road at night', 'a street in daylight under an elevated railway', 'a city street on a sunny day',
                 'a road on an overcast day', 'a road at dusk', 'a road tunnel', 'the inside of a parking garage'], 3)
NIGHT_EVERY_S, NIGHT_SCENE_DAY = 30, 0.1
# drone and aerial footage (tam7IqTDBTU, EoWg9w8DpsM; user, 2026-10-15): CLIP on AERIAL_FRAMES frames spread over the
# video; a video is drone footage when AERIAL_SHARE of them or more look aerial. On 69 drone-titled videos and 152
# approved drives (8 frames each): 64 drone videos caught (the misses: drone product reviews), no approved drive
# with even 2 of 8 frames aerial
AERIAL_SCENES = (['an aerial view from a drone', "a bird's-eye view of a city", 'a drone shot of the coast',
                  'a road seen through a car windshield', 'a city street seen from a driving car',
                  'a highway seen from a car', 'a residential street seen from a car',
                  'a person walking on a sidewalk',
                  'the inside of a car', 'a person talking to the camera', 'a parking lot'], 3, 'a photo of ')
AERIAL_FRAMES, AERIAL_SHARE = 8, 0.5


def aerial_track(path, duration, work_dir):
    """[(second, CLIP score of AERIAL_SCENES)] of AERIAL_FRAMES frames from 5% to 95% of the video."""
    if not duration:
        return []
    times = [float(duration * q) for q in np.linspace(0.05, 0.95, AERIAL_FRAMES)]
    return sorted(_clip_scores(path, times, work_dir, AERIAL_SCENES).items())


# a camera facing the driver, seen by CLIP (user, 2026-10-16: jykQR2dKANc, AlCduyvTAxI, 1MDz3XJ82Oc; the Haar face
# detector misses a face turned away or in sunglasses): a frame every DRIVER_CLIP_EVERY_S within DRIVER_NEAR_S of a
# frame with a face. On those three videos 200 of 210 frames of the driver scored 0.9+, 4 of 524 of the road, never
# two in a row: DRIVER_CLIP_RUN readings in a row at DRIVER_CLIP_SURE or more are the driver
DRIVER_SCENES = (['a person driving a car, filmed from the dashboard', 'a selfie of a driver inside a car',
                  'the face of a man sitting in a car', 'a person talking to the camera inside a car',
                  'a road seen through a car windshield', 'a city street seen from a driving car',
                  'a highway seen from a car', 'a residential street seen from a car', 'traffic ahead on a road',
                  'a parking lot'], 4, 'a photo of ')
DRIVER_CLIP_EVERY_S, DRIVER_NEAR_S, DRIVER_CLIP_SURE, DRIVER_CLIP_RUN = 10, 300, 0.9, 2


def driver_clip_track(path, face, work_dir):
    """[(second, CLIP score of DRIVER_SCENES)] every DRIVER_CLIP_EVERY_S within DRIVER_NEAR_S of a face frame."""
    seen = np.flatnonzero(np.asarray(face) > 0) / FPS
    if not len(seen):
        return []
    times = [float(t) for t in np.arange(DRIVER_CLIP_EVERY_S / 2, len(face) / FPS, DRIVER_CLIP_EVERY_S)
             if np.abs(seen - t).min() <= DRIVER_NEAR_S]
    return sorted(_clip_scores(path, times, work_dir, DRIVER_SCENES).items())


def driver_clip_runs(sig):
    """[(first frame, end frame)] of DRIVER_CLIP_RUN or more driver_clip readings in a row that are sure."""
    dc = np.asarray(sig.get('driver_clip', np.zeros((0, 2)))).reshape(-1, 2)
    out, run = [], []
    for t, score in dc:
        if score >= DRIVER_CLIP_SURE and (not run or t - run[-1] <= 1.5 * DRIVER_CLIP_EVERY_S):
            run.append(t)
            continue
        if len(run) >= DRIVER_CLIP_RUN:
            out.append(run)
        run = [t] if score >= DRIVER_CLIP_SURE else []
    if len(run) >= DRIVER_CLIP_RUN:
        out.append(run)
    half = DRIVER_CLIP_EVERY_S / 2
    return [(int(max(0, r[0] - half) * FPS), int((r[-1] + half) * FPS)) for r in out]


# night both ways, over the whole video: dusk with the street lights on counts as night, as the reviews label it
# (lmALZy7EF-g 27:47-37:47; user, 2026-10-16). On 374 approved segments (104 the brightness rules got wrong, 270
# right) taking CLIP's label where it is sure (0.2/0.8) fixed 54 and broke 3; CLIP alone was right on 334
NIGHT_CLIP = (['a city street at night with the street lights on', 'a dark road at night lit by headlights',
               'a road at night', 'a road at dusk with the street lights on',
               'a street in daylight under an elevated railway', 'a city street on a sunny day',
               'a road on an overcast day', 'a road tunnel', 'the inside of a parking garage',
               'a road at sunset in daylight'], 4)
NIGHT_CLIP_SURE = 0.2  # a piece's middle reading this or less is day, 1 minus it or more night


def night_clip_track(path, duration, work_dir):
    """[(second, CLIP night score of NIGHT_CLIP)] every NIGHT_EVERY_S over the whole video."""
    if not duration:
        return []
    times = [float(t) for t in np.arange(NIGHT_EVERY_S / 2, duration, NIGHT_EVERY_S)]
    return sorted(_clip_scores(path, times, work_dir, NIGHT_CLIP).items())


# lit lamps as small bright spots on a darker surround (sFaIgPDlHa4; user, 2026-10-16): per frame (320 px wide) the
# connected bright pixels (value 230+, the 15 px around at most 110) of 2-150 px. Alone noisy (glints by day), but
# where CLIP is unsure SPOTS_NIGHT or more mean night: on the night test 49 of 81 wrong segments fixed with CLIP and
# them, 43 with CLIP alone, 3 of 197 right ones broken either way
SPOTS_NIGHT = 6
# ... or LAMPS_NIGHT or more lit orange lamps (light_spots), also at dusk: on the same test 54 fixed, 3 broken
LAMPS_NIGHT = 4


def light_spots(png):
    """(small bright spots on a dark surround, small lit orange lamps anywhere) in a frame (PNG bytes). The lamps:
    pixels of hue 8-32 (OpenCV), saturation 140+, value 220+, as at dusk when the sky is still light (user,
    2026-10-16)."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (320, 180))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, sat, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    out = []
    for mask in ((g >= 230) & (cv2.blur(g, (15, 15)) <= 110), (h >= 8) & (h <= 32) & (sat >= 140) & (v >= 220)):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        out.append(int(((stats[1:, cv2.CC_STAT_AREA] >= 2) & (stats[1:, cv2.CC_STAT_AREA] <= 150)).sum()))
    return tuple(out)


def light_spots_track(path, duration):
    """[(second, spots, lamps)] of light_spots every NIGHT_EVERY_S over the whole video, as night_clip_track."""
    out = []
    for t in np.arange(NIGHT_EVERY_S / 2, duration or 0, NIGHT_EVERY_S):
        png = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.1f}', '-i', path, '-frames:v', '1', '-f',
                              'image2pipe', '-vcodec', 'png', '-'], capture_output=True, timeout=120).stdout
        if png and (n := light_spots(png)) is not None:
            out.append((float(t), *n))
    return out


def night_scene_track(path, sig, work_dir):
    """[(second, CLIP score of NIGHT_SCENES)] every NIGHT_EVERY_S where the picture is as dark as night."""
    dark = sig['sky'] < PARAMS['night_sky']
    if 'ground' in sig:
        dark = dark & (sig['ground'][:len(dark)] < PARAMS['night_ground'])
    times = [float(t) for t in np.arange(NIGHT_EVERY_S / 2, len(dark) / FPS, NIGHT_EVERY_S) if dark[int(t * FPS)]]
    return sorted(_clip_scores(path, times, work_dir, NIGHT_SCENES).items()) if times else []


def step_bob_track(path, duration, every=BOB_EVERY_S, length=15, fps=10):
    """[(start s, share)] of step_bob's measure on a 15-s reading every `every` seconds: a walk inside a drive
    (jhXgQ-mMrnw 9:01-9:52, the one segment its highway drive kept: 0.56-0.81 against 0.33 driving at 5:00; user,
    2026-10-13). A reading of 15 s is noisy, so a hint for the segments, not a cut."""
    w, h = 160, 90
    out = []
    for t in np.arange(0, max(0, (duration or 0) - length), every):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.1f}', '-t', str(length), '-i', path, '-an', '-vf',
                              f'fps={fps},scale={w}:{h},format=gray', '-f', 'rawvideo', '-'],
                             capture_output=True, timeout=300).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
        if len(frames) < 50:
            continue
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        dy = np.array([cv2.phaseCorrelate(a, b, win)[0][1] for a, b in zip(frames, frames[1:])])
        dy = dy - np.convolve(dy, np.ones(9) / 9, 'same')
        power = np.abs(np.fft.rfft(dy * np.hanning(len(dy)))) ** 2
        hz = np.fft.rfftfreq(len(dy), 1 / fps)
        total = power[(hz >= 0.3) & (hz <= 4.5)].sum()
        if total:
            out.append((float(t), float(power[(hz >= 1.3) & (hz <= 2.6)].sum() / total)))
    return out


def walk_hints(segments, track):
    """[(start s, end s)] of the segments whose step_bob_track readings look like walking: one of
    STEP_BOB_WALK or more, or their middle STEP_BOB_HINT or more."""
    out = []
    for seg in segments:
        inside = [x for t, x in track if seg['start'] <= t + 7.5 <= seg['end']]
        if inside and (max(inside) >= STEP_BOB_WALK or np.median(inside) >= STEP_BOB_HINT):
            out.append((seg['start'], seg['end']))
    return out


def step_bob(path, duration, windows=6, length=15, fps=10):
    """Share of the camera's vertical shake at a walker's step rate (1.3-2.6 Hz), the median over `windows`
    stretches of `length` s read at `fps` (the analysis frames, 2 a second, cannot show steps): 15 walks the
    reviewer named scored 0.54-0.92 (median 0.83), 13 approved drives 0.28-0.59 (user, 2026-10-13: "such videos have
    shaking camera from steps", 9F8zSPj77zI). 0 when too little is read."""
    w, h = 160, 90
    shares = []
    for t in np.linspace(0.15, 0.85, windows) * (duration or 0):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.1f}', '-t', str(length), '-i', path, '-an', '-vf',
                              f'fps={fps},scale={w}:{h},format=gray', '-f', 'rawvideo', '-'],
                             capture_output=True, timeout=300).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
        if len(frames) < 50:
            continue
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        dy = np.array([cv2.phaseCorrelate(a, b, win)[0][1] for a, b in zip(frames, frames[1:])])
        dy = dy - np.convolve(dy, np.ones(9) / 9, 'same')  # no slow drift (turns, hills)
        power = np.abs(np.fft.rfft(dy * np.hanning(len(dy)))) ** 2
        hz = np.fft.rfftfreq(len(dy), 1 / fps)
        total = power[(hz >= 0.3) & (hz <= 4.5)].sum()
        if total:
            shares.append(power[(hz >= 1.3) & (hz <= 2.6)].sum() / total)
    return float(np.median(shares)) if len(shares) >= 3 else 0.0


def walking_frames(sheet_path):
    """How many of a contact sheet's frames CLIP sees as taken on foot (a sidewalk, a park path, a shop, a trail)
    rather than from a car; 0 without CLIP. Of 2,380 approved videos one had 5 or more (ifgeBqX6HXI, mostly a
    walk); of 416 rejected, 14 (8ScuEwMmCno, 8xybXtQ0LgA: walks, malls, airports, theme parks)."""
    return sum(x > 0.5 for x in sheet_scores(sheet_path, WALK_SCENES))


DASH_SCENES = (['a photo of a car dashboard and steering wheel', 'a photo of a road', 'a photo of a street',
                'a photo of a highway'], 1)
DASH_BOTTOM_SCENES = (['a car dashboard with a steering wheel', 'a road', 'a car hood', 'a street', 'a highway'], 1)
DASH_SHEET, DASH_BOTTOM = 0.25, 0.35  # mean scores over a sheet's frames, whole and lower halves


def dashboard_view(sheet_path):
    """Whether the camera films from inside the car with the dashboard in view, not the road through the windshield
    (qtODPui4408: dashboard, wheel and a phone fill the lower half): CLIP's mean score on a contact sheet's frames
    of DASH_SHEET or more, or on their lower halves of DASH_BOTTOM or more. Of 600 approved videos none, of 521
    rejected 11 (qtODPui4408 0.38 on lower halves; heyKevTV's interior views 0.29-0.75 on whole frames); the
    highest approved scored 0.20 and 0.32."""
    whole = sheet_scores(sheet_path, DASH_SCENES)
    if whole and np.mean(whole) >= DASH_SHEET:
        return True
    low = sheet_scores(sheet_path, DASH_BOTTOM_SCENES, 'a photo of ', bottom=True)
    return bool(low) and np.mean(low) >= DASH_BOTTOM


WALK_EDGE_S = 300  # how far into a video's start and end to look for a walk to or from the car


def walk_edges(path, duration, work_dir, step=10):
    """(start walk end s, end walk start s) of a walk at the video's start and end, else 0 and duration: CLIP on a
    frame every step s over the first and last WALK_EDGE_S; the walk reaches to the walking frame (WALK_SCENES
    above 0.5) farthest from the edge with at least half the frames up to it, and 5 or more, walking (mYglt3SnDHU
    walks Katoomba's streets between a montage and a sideways stretch: walking frames 0:40-3:30, the drive at 4:25)."""
    if not duration or duration < 2 * WALK_EDGE_S:
        return 0, duration or 0
    first = [t + 0.5 for t in range(0, WALK_EDGE_S, step)]
    last = [duration - WALK_EDGE_S + t + 0.5 for t in range(0, WALK_EDGE_S, step)]
    scores = _clip_scores(path, first + last, work_dir, WALK_SCENES + ('',))

    def reach(times):  # from the edge inwards
        walking = [scores.get(t, 0) > 0.5 for t in times]
        hits = [i for i, w in enumerate(walking) if w and sum(walking[:i + 1]) >= max(5, (i + 1) / 2)]
        return times[hits[-1]] if hits else None
    start, end = reach(first), reach(last[::-1])
    return (int(start + step) if start is not None else 0,
            int(end - step) if end is not None else int(duration))


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
    for f in frames(path, letterbox(path)):  # the sky is the top of the picture, not a black bar above it
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


def _sideways(sig, p, relative=False):
    """Moving frames, and moving frames whose flow does not spread out from the middle (camera aimed sideways).
    relative: also well under the video's own spread while moving (angle_relative times its median): a forward
    camera that spreads little all along (9coEBrZmkWc, CuseVegas: median 0.61, forward cameras 1.1-1.5) is cut only
    where it turns (user, 2026-10-16)."""
    moving = sig['motion'] > 2 * p['moving_flow']
    ratio = sig['spread'] / (sig['motion'] + 0.05)
    limit = p['angle_spread']
    if relative and p.get('angle_relative') and moving.sum() >= 60 * FPS:
        limit = min(limit, p['angle_relative'] * float(np.median(ratio[moving])))
    return moving, moving & (ratio < limit)


def footage_problem(sig, p=PARAMS):
    """Why a whole video is unusable from its camera, or None (signals saved before these checks: None)."""
    if 'shift_y' in sig and np.median(_jitter(sig)) > p['shake_jitter']:
        return 'the camera shakes too much'
    aerial = np.asarray(sig.get('aerial', np.zeros((0, 2)))).reshape(-1, 2)
    if len(aerial) >= AERIAL_FRAMES // 2 and (up := (aerial[:, 1] >= 0.5).sum()) >= AERIAL_SHARE * len(aerial):
        return f'drone footage (an aerial view in {up} of {len(aerial)} frames)'
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


def _label_runs(mask):
    """Each frame's run number (1, 2, ...) within mask, 0 outside it."""
    starts = mask & ~np.r_[False, mask[:-1]]
    return np.where(mask, np.cumsum(starts), 0)


def _stopped(moving, p):
    """Frames the car stands still, a stop at a video's start or end taking in the bursts of creeping around it."""
    stopped = ~moving
    for order in (slice(None), slice(None, None, -1)):  # the start, then the end read backwards
        runs = _runs(stopped[order])
        if runs and runs[0][0] == 0 and (runs[0][1] - runs[0][0]) / FPS >= p['edge_parked_s']:
            i = 0
            while i + 1 < len(runs) and (runs[i + 1][0] - runs[i][1]) / FPS < p['edge_burst_s']:
                i += 1
            stopped[order][:runs[i][1]] = True
    return stopped


def edge_stops(sig, p=PARAMS):
    """[(start s, end s)] of the stops at a video's start or end that propose would cut."""
    n = len(sig['motion'])
    moving = _rolling_median(sig['motion'], p['smooth_s'] * FPS) > p['moving_flow']
    return [(a / FPS, b / FPS) for a, b in _runs(_stopped(moving, p))
            if (a == 0 or b == n) and (b - a) / FPS > max(p['edge_stop_s'], 3)]


def camera_tilt(sig, n, p=PARAMS):
    """(tilted up, tilted down) per analysis frame: the median height of the road users YOLO sees over a minute
    is past tilt_up (only at the bottom edge: the camera looks at the sky) or under tilt_down (only at the top: the
    dashboard). Looking ahead it is 0.62-0.66; KgEL7td3JK0 tilted up from 16:42 to the end: 0.95."""
    y = np.repeat(sig['object_y'], URBAN_EVERY_S * FPS)[:n]
    y = np.pad(y, (0, n - len(y)), mode='edge') if len(y) else np.full(n, np.nan)
    half = 30 * FPS
    med = np.array([np.nanmedian(w) if np.isfinite(w := y[max(0, i - half):i + half]).sum() >= 4 else np.nan
                    for i in range(0, n, FPS)])
    med = np.repeat(med, FPS)[:n]
    with np.errstate(invalid='ignore'):
        return med > p['tilt_up'], med < p['tilt_down']


SIDEWAYS_DARK = 0.5  # a sideways stretch this much in the dark is only a hint (see propose)


REPEAT_MIN_S = 300  # a stretch at least this long matching footage earlier in the video is a repeat


def repeated_footage(sig):
    """[(start s, end s, lag s)] of footage that repeats earlier footage of the video (UCLgFDrgDENPf7a59tsj8d_A's
    hour-long videos are often the same 30 min twice: YJAsLoJ_pBM, 30 more; user, 2026-10-16): per second, sky,
    road and motion within 1.5, 1.5 and 0.3 of the same measures `lag` seconds earlier, for REPEAT_MIN_S or more
    while the car moves. The first lag with such a run is taken."""
    n = len(sig['sky']) // FPS
    if n < 2 * REPEAT_MIN_S or 'ground' not in sig:
        return []
    sec = lambda x: np.asarray(x[:n * FPS], float).reshape(n, FPS).mean(1)  # noqa: E731
    sky, gr, mo = sec(sig['sky']), sec(sig['ground'][:len(sig['sky'])]), sec(sig['motion'])
    for lag in range(REPEAT_MIN_S, n - REPEAT_MIN_S + 1):
        same = (np.abs(sky[lag:] - sky[:-lag]) < 1.5) & (np.abs(gr[lag:] - gr[:-lag]) < 1.5) \
            & (np.abs(mo[lag:] - mo[:-lag]) < 0.3)
        if same.sum() < REPEAT_MIN_S:
            continue
        out = [(lag + a, lag + b, lag) for a, b in _runs(same)
               if b - a >= REPEAT_MIN_S and (mo[lag + a:lag + b] > 0.5).mean() > 0.5]
        if out:
            return out
    return []


def rear_camera_from(sig, p=PARAMS):
    """Seconds from which the rest of the video is the rear camera, else None: the moving frames' flow converges
    there to the end (median spread ratio under REAR_SPREAD) after spreading out before it (over FRONT_SPREAD).
    UCLgFDrgDENPf7a59tsj8d_A switches at 5:00 (l-sh484PHAw, mtmfpTVCFWA, n8nPtJAEtmI, lYvXd8eky6E, kMxiZs3hPqw;
    user, 2026-10-16); of 4,328 approved segments of 2 min or more one has a median under 0 (forward: 1.1-1.5)."""
    if 'spread' not in sig:
        return None
    moving = sig['motion'] > 2 * p['moving_flow']
    r = np.where(moving, sig['spread'] / (sig['motion'] + 0.05), np.nan)
    ok = np.isfinite(r)
    if ok.sum() < 120 * FPS:
        return None
    # the switch: the time that best parts frames spreading out before it from frames converging after it, on the
    # spread's 10-s median: a turn's converging moment before the switch pulled it a minute early (dGn9jpFKb2M 4:13,
    # IIKIf6nEJQQ 4:12 for 4:59; user, 2026-10-16)
    half = 5 * FPS
    sm = np.array([np.nanmedian(w) if np.isfinite(w := r[max(0, i - half):i + half]).any() else np.nan
                   for i in range(len(r))])
    okm = np.isfinite(sm)
    ahead = np.cumsum(okm & (sm > 0))                     # forward frames up to each frame
    behind = np.cumsum((okm & (sm < 0))[::-1])[::-1]      # converging frames from each frame on
    score = ahead[:-1] + behind[1:]
    # a wait around the switch (no moving frames: dGn9jpFKb2M stands 4:15-5:00, l-sh484PHAw after 5:00) leaves a
    # stretch of equal bests: the switch is the strongest frame jump in it (the picture changes at once), else
    # its first moment
    best = np.flatnonzero(score == score.max()) + 1
    ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
    near = ev[(ev[:, 0] * FPS >= best[0] - FPS) & (ev[:, 0] * FPS <= best[-1] + FPS)]
    t = int(round(near[np.argmax(near[:, 2]), 0] * FPS)) if len(near) else int(best[0])
    head, tail = r[:t][ok[:t]], r[t:][ok[t:]]
    # four minutes of front camera first, and a rear part of a fifth of the video or more: a converging end of a
    # drive (AaHFLxo7h4A, the last 7 of 65 min at -0.34) is no camera switch, nor one converging stretch among
    # others (0vfUg5xioIE after 34:00: -1.5, then 0.3 and 0.1): 70% of the rest converges
    if t >= 240 * FPS and len(r) - t >= len(r) / 5 and len(tail) >= 60 * FPS and np.median(tail) < REAR_SPREAD \
            and (tail < 0).mean() >= 0.7 and np.median(head) > FRONT_SPREAD:
        return t / FPS
    return None


REAR_SPREAD, FRONT_SPREAD = -0.2, 0.5


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
            continue
        ref = np.median(sky[max(0, i - 10 * FPS):max(1, i - FPS)])
        # ... and sun glare: a brief bright spike with the light as before after it (px5Fob6umxw 44:24, _maBKxVgQI4,
        # XZcQC5inyBc; user, 2026-10-16). Of 699 cuts shown in reviewed stretches: 26 false and 7 real are such
        high = sky[max(0, i - 8):i + 9].max() >= p['glare_spike'] * max(before.mean(), after.mean())
        if (i + 10 * FPS <= len(sky) and high
                and 0 < (sky[max(0, i - 2 * FPS):i + 4 * FPS] > 1.15 * ref).sum() / FPS <= p['short_dip_s']
                and abs(np.median(sky[i + 3 * FPS:i + 10 * FPS]) - ref) <= 0.15 * ref):
            cut[i] = False
            continue
        # ... and a shallower dip that lasts no longer than a bridge takes to pass (a second or so; long overpasses
        # and the El are the night rules' business), with the light back after it (user, 2026-10-16). Of 828 cuts
        # shown in reviewed stretches (539 false, 289 real): 176 false and 47 real are such
        low = sky[max(0, i - 8):i + 9].min() < p['short_dip'] * min(before.mean(), after.mean())
        if (i + 10 * FPS <= len(sky) and low
                and 0 < (sky[max(0, i - 2 * FPS):i + 4 * FPS] < 0.8 * ref).sum() / FPS <= p['short_dip_s']
                and np.median(sky[i + 3 * FPS:i + 10 * FPS]) >= 0.8 * ref):
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

    stamps = stamp_cuts(sig)
    for t in stamps:  # footage of another drive: a cut whatever the picture says
        if 0 < int(t * FPS) < n:
            cut[int(t * FPS)] = True
    keep = ~blank
    notes = []
    for a, b, lag in repeated_footage(sig):  # the same footage again: once is enough
        keep[a * FPS:b * FPS] = False
        notes.append(f'{a // 60}:{a % 60:02d}-{b // 60}:{b % 60:02d} repeats the footage from '
                     f'{(a - lag) // 60}:{(a - lag) % 60:02d}')
    if (rear := rear_camera_from(sig, p)) is not None:  # the rest of the video is the rear camera
        keep[int(rear * FPS):] = False
        notes.append(f'rear camera from {int(rear) // 60}:{int(rear) % 60:02d}')
    if stamps:
        notes.append('the time stamp in the picture jumps at '
                     + ', '.join(f'{int(t) // 60}:{int(t) % 60:02d}' for t in stamps) + ': footage of another drive')
    for a, b in np.asarray(sig.get('garage', np.zeros((0, 2)))).reshape(-1, 2):  # see garage_edges
        keep[int(a * FPS):int(b * FPS)] = False
        notes.append(f'in a garage {int(a) // 60}:{int(a) % 60:02d}-{int(b) // 60}:{int(b) % 60:02d}')
    if float(sig.get('intro_end', 0)) > 0:  # see clip_intro
        end = int(float(sig['intro_end']))
        keep[:end * FPS] = False
        notes.append(f'intro 0-{end}s (a map, title or logo)')
    for a, b in np.asarray(sig.get('lots', np.zeros((0, 2)))).reshape(-1, 2):  # see parking_lots
        keep[int(a * FPS):int(b * FPS)] = False
        notes.append(f'in a parking lot {int(a) // 60}:{int(a) % 60:02d}-{int(b) // 60}:{int(b) % 60:02d}')
    for a, b in np.asarray(sig.get('kerb', np.zeros((0, 2)))).reshape(-1, 2):  # see kerb_stops
        notes.append(f'stopped at the kerb? {int(a) // 60}:{int(a) % 60:02d}-{int(b) // 60}:{int(b) % 60:02d}: cut it '
                     'if parked, keep a wait in traffic')
    for a, b in np.asarray(sig.get('chapter_cuts', np.zeros((0, 2)))).reshape(-1, 2):  # see chapter_cuts
        keep[int(a * FPS):int(b * FPS)] = False
        notes.append(f'not the drive by its chapter {int(a) // 60}:{int(a) % 60:02d}-{int(b) // 60}:{int(b) % 60:02d}'
                     ' (highlights or a preview)')
    if float(sig.get('outro_start', 0)) > 0:  # see clip_outro
        start = int(float(sig['outro_start']))
        keep[start * FPS:] = False
        notes.append(f'outro from {start // 60}:{start % 60:02d} (a slideshow, map, title or end screen)')
    if float(sig.get('labelled_montage', 0)) > 0:  # see montage_label
        end = int(float(sig['labelled_montage']))
        keep[:end * FPS] = False
        notes.append(f'montage of highlights 0-{end}s (its caption says so)')
    elif (end := montage_end(sig)) is not None:  # a hint: in most channels early cuts are something else
        notes.append(f'possible montage of highlights 0-{end}s: check where the drive starts')
    sat = sig.get('saturation')
    if (sat is not None and len(sat) and sig.get('skips') is not None and 'spread' in sig
            and sat[0] > max(p['intro_saturation'], p['intro_saturation_ratio'] * np.median(sat))):
        moving_now = sig['motion'] > 2 * p['moving_flow']
        forward = moving_now & (sig['spread'] / (sig['motion'] + 0.05) >= p['angle_spread'])
        drive = np.flatnonzero(_rolling_mean(forward.astype(float), 5 * FPS) > 0.5)
        start = drive[0] / FPS if len(drive) else p['intro_max_s']
        cuts = [t for t in np.asarray(sig['skips']).reshape(-1, 3)[:, 0] if 0 < t <= min(start + 1, p['intro_max_s'])]
        end = int((max(cuts) if cuts else min(start, p['intro_max_s'])) * FPS)
        keep[:end] = False
        notes.append(f'intro 0-{end / FPS:.0f}s (a map or title before the drive)')
    parked = np.asarray(sig['parked']).reshape(-1, 2) if 'parked' in sig else None  # see parked_edges
    for a, b in _runs(_stopped(moving, p)):
        at_edge = a == 0 or b == n
        limit = p['edge_stop_s'] if at_edge else p['mid_stop_s']
        if (b - a) / FPS <= limit:
            continue
        # a wait in traffic, unless parked; no wait lasts longer than a stop mid-video may (ifgeBqX6HXI: 17 min
        # filming a beach at sunset)
        if (at_edge and parked is not None and 3 < (b - a) / FPS <= p['mid_stop_s']
                and not any(x < b / FPS and a / FPS < y for x, y in parked)):
            notes.append(f'stationary {a / FPS:.0f}-{b / FPS:.0f}s (edge, kept: waiting in traffic?)')
            continue
        keep[a:b] = False
        notes.append(f"stationary {a / FPS:.0f}-{b / FPS:.0f}s{' (edge)' if at_edge else ''}")
    for i in np.flatnonzero(cut):
        notes.append(f'cut at {i / FPS:.0f}s' + (' (the light dims before it and is back within 10 s: an overpass?)'
                                                 if overpass_like(sky, i) else ''))
    if 'face' in sig:  # signals saved before this check have no face trace
        # a camera on the driver keeps seeing him, but turning his head hides the face for a while, so judge
        # the share of frames with a big face over a longer window rather than frame by frame
        share = _rolling_mean((sig['face'] > 0).astype(float), p['driver_window_s'] * FPS)
        runs = [(a, b) for a, b in _runs(share > p['driver_share']) if (b - a) / FPS >= p['driver_min_s']]
        # a few close faces (passers-by) are no camera on the driver: a hint under DRIVER_CUT_SHARE (user,
        # 2026-10-14: 248nL9hGz8s split at 10:52-11:44 for 13%; 8 of the 12 cuts approved across were under 0.2)
        sure = [share[a:b].max() >= DRIVER_CUT_SHARE for a, b in runs]
        # ... and where CLIP sees the driver, face or not (driver_clip_runs)
        for run in driver_clip_runs(sig):
            k = int(np.searchsorted([a for a, _ in runs], run[0]))
            runs.insert(k, run)
            sure.insert(k, True)
        # ... unless next to a cut one, and then the footage between them is the driver too, the face turned away for
        # a while (1MDz3XJ82Oc 0:57-1:57; user, 2026-10-16): in reviewed videos, of 67 hint runs next to a cut one
        # within DRIVER_GAP_S none was approved, of 92 stretches between two cut ones 3
        pairs = [k for k in range(len(runs) - 1)
                 if (sure[k] or sure[k + 1]) and (runs[k + 1][0] - runs[k][1]) / FPS <= DRIVER_GAP_S]
        cut_run = [c or k in pairs or k - 1 in pairs for k, c in enumerate(sure)]
        for k in pairs:
            a, b = runs[k][1], runs[k + 1][0]
            if b > a:
                keep[a:b] = False
                notes.append(f'camera on the driver {a / FPS:.0f}-{b / FPS:.0f}s (between two such stretches)')
        for (a, b), cut_it in zip(runs, cut_run):
            if not cut_it:
                t0, t1 = int(a / FPS), int(b / FPS)
                notes.append(f'faces in {share[a:b].max():.0%} of the frames at {t0 // 60}:{t0 % 60:02d}-'
                             f'{t1 // 60}:{t1 % 60:02d}: camera on the driver? check')
                continue
            keep[a:b] = False
            notes.append(f'camera on the driver {a / FPS:.0f}-{b / FPS:.0f}s')
    problems, side_runs = [], None
    if 'spread' in sig:
        moving, sideways = _sideways(sig, p, relative=True)
        moving_share = _rolling_mean(moving.astype(float), 60 * FPS)
        share = _rolling_mean(sideways.astype(float), 60 * FPS) / np.maximum(moving_share, 1e-6)
        # judged only over minutes with enough driving; a stop says nothing about where the camera points
        # a minute blurs where a side view starts and ends (ZuA0i7CXu6A: cut 14:12-15:26 of a side view 13:50-15:30),
        # so each stretch cut (long enough by itself: winding roads look sideways for a while, 1PpJ31bBDUk) takes
        # in the side views over 10 s that overlap it
        ten_moving = _rolling_mean(moving.astype(float), 10 * FPS)
        ten = (_rolling_mean(sideways.astype(float), 10 * FPS) / np.maximum(ten_moving, 1e-6) > 0.5) \
            & (ten_moving > 0.3)
        for a, b in _runs(~ten):  # a moment looking ahead does not end a side view (ZuA0i7CXu6A at 10:44)
            if 0 < a and b < n and b - a < 3 * FPS:
                ten[a:b] = True
        side_runs = _label_runs(ten)
        problems.append(('camera not facing forward', (share > p['angle_share']) & (moving_share > 0.3)))
        # a video that switches between a forward and a side camera (5k8awa6rpmA, "Multi Angle"): its side views
        # are 20-60 s, too short for the minute above, so they are judged over 20 s once a video has many of them
        # (15+ an hour, 4+ in all: 2 of 2,335 approved videos, 10 of 434 others)
        short_moving = _rolling_mean(moving.astype(float), 20 * FPS)
        short = (_rolling_mean(sideways.astype(float), 20 * FPS) / np.maximum(short_moving, 1e-6) > 0.6) \
            & (short_moving > 0.3)
        views = [(a, b) for a, b in _runs(short) if (b - a) / FPS >= 25]
        if len(views) >= 4 and len(views) / max(n / FPS / 3600, 0.1) >= 15:
            switching = np.zeros(n, bool)
            for a, b in views:
                switching[a:b] = True
            problems.append(('camera turned to the side (the video switches angles)', switching))
    if 'urban' in sig:
        per_frame = np.repeat(sig['urban'], URBAN_EVERY_S * FPS)[:n]
        per_frame = np.pad(per_frame, (0, n - len(per_frame)), mode='edge')
        day = _rolling_median(sig['sky'], 60 * FPS) >= p['yolo_day_sky']  # YOLO sees little at night anywhere
        problems.append(('rural driving', (_rolling_mean(per_frame, 60 * FPS) < p['urban_objects']) & day))
        if 'street' in sig:  # signals saved before the highway check have no street trace
            street = np.repeat(sig['street'], URBAN_EVERY_S * FPS)[:n]
            street = np.pad(street, (0, n - len(street)), mode='edge')
            problems.append(('highway driving', (_rolling_mean(street, 60 * FPS) < p['street_objects']) & day))
    if 'shift_y' in sig:
        problems.append(('camera shaking', _rolling_median(_jitter(sig), 30 * FPS) > p['shake_jitter']))
    if 'object_y' in sig:
        up, down = camera_tilt(sig, n, p)
        problems += [('camera tilted up', up), ('camera tilted down', down)]
    if len(town := np.asarray(sig.get('town_scene', np.zeros((0, 2)))).reshape(-1, 2)):
        # where CLIP sees a town's streets (houses, shops), no highway or rural cut (TOWN_SCENES): the middle of three
        # readings in a row, so one odd frame decides nothing
        score = np.array([np.median(town[max(0, i - 1):i + 2, 1]) for i in range(len(town))])
        at = np.clip(np.searchsorted(town[:, 0], np.arange(n) / FPS), 0, len(town) - 1)
        in_town = score[at] >= TOWN_SCORE
        problems = [(label, bad & ~in_town if label in ('highway driving', 'rural driving') else bad)
                    for label, bad in problems]
    for label, bad in problems:
        min_s = p['quiet_min_s'] if label in ('highway driving', 'rural driving') else \
            25 if label.startswith('camera turned to the side') else p['problem_min_s']
        for a, b in _runs(bad):
            if (b - a) / FPS < min_s:
                continue
            if label == 'camera shaking' and a > 0 and b < n:
                # potholes and bumps mid-drive stay (QVdDzY3JaS0, vpaB7ijEuDw: the picture still flows out from
                # the middle, spread/motion 1.3-1.4); a walk with the camera in hand does not (ifgeBqX6HXI: 0.04)
                forward = np.median(sig['spread'][a:b] / (sig['motion'][a:b] + 0.05)) >= p['angle_spread'] \
                    if 'spread' in sig else True
                if forward:
                    notes.append(f'camera shaking {a / FPS:.0f}-{b / FPS:.0f}s (kept: a bumpy road?)')
                    continue
                label = 'walking or a handheld camera'
            if side_runs is not None and label.startswith(('camera not facing', 'camera turned')):  # to its edges
                ids = np.unique(side_runs[a:b][side_runs[a:b] > 0])
                if len(ids):
                    at = np.flatnonzero(np.isin(side_runs, ids))
                    a, b = min(a, at[0]), max(b, at[-1] + 1)
                # in the dark the flow has little to follow but lamps and headlights: a hint, not a cut (night
                # drives on quiet roads taken for a side camera: 8O0zlwHapmM, 8ZhmysYj30c, gqPB9b90p54,
                # j_2KkJuONF4, IHz_IuQb0DE; user, 2026-10-16)
                dark = sig['sky'][a:b] < p['night_sky']
                if 'ground' in sig:
                    dark = dark & (sig['ground'][a:b] < p['night_ground'])
                if dark.mean() >= SIDEWAYS_DARK:
                    notes.append(f'{label}? {a / FPS:.0f}-{b / FPS:.0f}s in the dark: check, cut it if so')
                    continue
            keep[a:b] = False
            notes.append(f'{label} {a / FPS:.0f}-{b / FPS:.0f}s')
    for a, b, r in clock_speedups(sig):
        keep[int(a * FPS):int(b * FPS)] = False
        notes.append(f'sped up {a:.0f}-{b:.0f}s: the time stamp in the picture runs {r:.1f}x faster than the video')
    if 'std' in sig:
        for a, b in photo_inserts(sig):
            keep[a:b] = False
            notes.append(f'photos inserted {a / FPS:.0f}-{b / FPS:.0f}s')

    fades = np.asarray(sig.get('fades', np.zeros((0, 4))))
    fades = fades.reshape(len(fades), -1) if fades.size else np.zeros((0, 4))  # 4 or 5 columns: start, end first
    ghosts = [a for a, b, *_ in fades if b - a >= FADE_MIN_S and SKIP_INTRO_S < a < n / FPS - SKIP_OUTRO_S
              and not under_bridge(sig, a, b)]
    if ghosts:  # a hint only: tunnels and slow night traffic look alike (see fade_events)
        notes.append('possible fades between scenes (ghosts: an edited drive?) at '
                     + ', '.join(f'{int(t) // 60}:{int(t) % 60:02d}' for t in ghosts[:5])
                     + (f' and {len(ghosts) - 5} more' if len(ghosts) > 5 else ''))
    # a hint (the reviewer's own check, 2026-10-11): in a city drive the car waits at red lights; when it never
    # stands still for 25 s and hardly at all, the waits were cut out, often with dissolves (ghosts): in the
    # ghost channel UCPv35e3jeQeCL6CWrS3njkw 7 of 13 videos, of 1,406 approved ones 29
    if 'urban' in sig and n / FPS >= 600 and np.median(sig['urban']) >= 3:
        moving_inner = (_rolling_median(sig['motion'], p['smooth_s'] * FPS) > p['moving_flow'])[30 * FPS:-30 * FPS]
        waits = [(b - a) / FPS for a, b in _runs(~moving_inner)]
        if 1 - moving_inner.mean() < 0.05 and max(waits, default=0) < 25:
            notes.append(f'no wait over {max(waits, default=0):.0f}s in this city drive (standing '
                         f'{1 - moving_inner.mean():.0%} of the time): waits at lights cut out (ghosts)? check an '
                         'intersection')
    vanished = [t for t in np.asarray(sig.get('ghost', [])).ravel() if SKIP_INTRO_S < t < n / FPS - SKIP_OUTRO_S]
    if vanished:  # a hint only, see ghost_vanishes
        notes.append('possible ghost (a person or car vanished mid-picture: an edit?) at '
                     + ', '.join(f'{int(t) // 60}:{int(t) % 60:02d}' for t in vanished[:5])
                     + (f' and {len(vanished) - 5} more' if len(vanished) > 5 else ''))
    # a hint: at a stop the picture stands still; changing fast all the same is traffic sped up (ofoaWwzK3_Y from
    # 5:06: similarity 0.83-0.92 against ~1.0), but cross traffic, rain and buses do it too (1.6% of approved
    # videos pass 0.4, 4.4% of rejected ones)
    stopped = _rolling_median(sig['motion'], 6) < 0.6
    if stopped.sum() >= 40 and (churn := float(np.mean(sig['ncc'][stopped] < 0.9))) >= p['stop_churn']:
        at = int(np.argmax(_rolling_mean((stopped & (sig['ncc'] < 0.9)).astype(float), 20 * FPS)) / FPS)  # densest
        notes.append(f'possibly sped up while stopped ({churn:.0%} of the time stopped the picture changes fast, most '
                     f'around {at // 60}:{at % 60:02d}): check, and reject the video if so')
    # a hint: a long drive through a city that hardly ever waits at a light had its waits cut out (user, 2026-10-13:
    # pkek7okGpS0, 78 min of Los Angeles with 2 stops of 20 s or more). Videos of 20+ min reviewed in full: under
    # WAITS_PER_HOUR stops of WAIT_S or more per hour on 64 of 696 approved (9%), 91 of 276 rejected (33%)
    waits = sum((b - a) / FPS >= WAIT_S for a, b in _runs(stopped))
    if n / FPS >= 1200 and waits < WAITS_PER_HOUR * n / FPS / 3600:
        notes.append(f'few waits at lights ({waits} stops of {WAIT_S} s or more in {n / FPS / 60:.0f} min): the waits '
                     'may be cut out of an edited drive, drag through it and check')
    window = p['night_window_s'] * FPS
    dark = sig['sky'] < p['night_sky']
    if 'ground' in sig:  # signals saved before the street brightness have the sky only
        dark = dark & (sig['ground'][:len(dark)] < p['night_ground'])
    night = _rolling_median(dark.astype(float), window) > 0.5
    # brief dark or bright spells (underpasses, a flickering sky near the threshold) must not split a segment
    night = _drop_short_runs(night, window)
    # a switch at an edit cut is footage stitched together (a night drive after a day drive): it stays night; under a
    # bridge or the El the dark comes and goes without one
    edits = np.r_[np.flatnonzero(cut), np.array(skip_times(sig)) * FPS]
    # only a cut where the light jumps: night footage after day changes the sky at the cut; a cut while under a
    # bridge does not (user, 2026-10-14: XDBi8ukETEA 11:56-14:47 night for the cut at 14:20, sky ~80 both sides)

    def jump(e, w=10 * FPS):
        e = min(max(int(e), 1), len(sig['sky']) - 1)
        return abs(np.median(sig['sky'][max(0, e - w):e]) - np.median(sig['sky'][e:e + w]))
    if p.get('stitch_jumps'):  # ... and strong frame jumps, alone or not: vRSvuv5bXV4's night clip ends at 5:00 with
        # a jump 382 times its neighbours, followed by another at 5:02, so skip_times leaves it out (user, 2026-10-16)
        ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
        edits = np.r_[edits, ev[(ev[:, 2] >= SKIP_RATIO) & (ev[:, 1] >= SKIP_SCORE), 0] * FPS]
    edits = [e for e in edits if jump(e) >= STITCH_SKY_JUMP]

    seen = np.asarray(sig.get('night_scene', np.zeros((0, 2)))).reshape(-1, 2)

    def clip_night(a, b):  # CLIP sure the dark a-b (frames) is night
        inside = seen[(seen[:, 0] * FPS >= a) & (seen[:, 0] * FPS < b), 1]
        return p.get('bridge_clip_night') is not None and len(inside) >= 4 \
            and np.median(inside) >= p['bridge_clip_night']

    def bridges():
        for a, b in _runs(night):
            if clip_night(a, b):  # no bridge: an hour of night inside a day drive (GAjYNAIn5hs; user, 2026-10-16)
                continue
            stitched = any(abs(e - x) <= window for e in edits for x in (a, b))
            inside = a > 0 and b < n and (b - a) / FPS <= p['night_inside_day_s']
            # a short dark start or end with daylight on the other side is the El or a bridge too (Marble Hill under
            # the 1 train, Roosevelt Ave under the 7); night falls over many minutes
            edge = (a == 0) != (b == n) and (b - a) / FPS <= p['night_edge_s']
            if (inside or edge) and not stitched:
                night[a:b] = False

    # ... and a short bright spell inside a night drive is lit streets, not day: day does not come and go within
    # minutes (eRaTzkbbC34: Manhattan's brightest blocks at 22:48-28:14, sky 88-102, between night on both sides)
    def lit_streets():
        for a, b in _runs(~night):
            stitched = any(abs(e - x) <= window for e in edits for x in (a, b))
            if a > 0 and b < n and (b - a) / FPS <= p['day_inside_night_s'] and not stitched:
                night[a:b] = True
    # in a mostly dark video the lamps make the bright spells, so lit streets come first: downtown at night
    # alternates between dark and lamp-lit minutes, and the bridges rule turned each dark minute into day
    # (yifqWxs2bIk, Indianapolis at night: 64% of its frames dark, all of it proposed as day; user, 2026-10-15)
    for rule in ((lit_streets, bridges) if p['night_majority'] and dark.mean() > 0.5 else (bridges, lit_streets)):
        rule()
    # ... and dark that CLIP sees as daylight is day, however long (NIGHT_SCENES: under the El, eq-5HRmOiN4)
    if len(seen := np.asarray(sig.get('night_scene', np.zeros((0, 2)))).reshape(-1, 2)):
        for a, b in _runs(night):
            inside = seen[(seen[:, 0] * FPS >= a) & (seen[:, 0] * FPS < b), 1]
            if len(inside) >= 2 and np.median(inside) < NIGHT_SCENE_DAY:
                night[a:b] = False
    # ... and CLIP's label holds against the brightness rules for each piece of footage it is sure of: the pieces
    # lie between the rules' day/night changes and strong frame jumps (an edit cut to other footage), and the
    # middle of a piece's readings (NIGHT_CLIP, every NIGHT_EVERY_S) decides when NIGHT_CLIP_SURE or more sure,
    # as in the test on approved segments (lmALZy7EF-g: 27:47-37:47, dusk with lamps, between two cuts)
    clip = np.asarray(sig.get('night_clip', np.zeros((0, 2)))).reshape(-1, 2)
    if len(clip):
        ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
        jumps = ev[(ev[:, 2] >= SKIP_RATIO) & (ev[:, 1] >= SKIP_SCORE), 0] * FPS
        bounds = sorted({0, n, *np.flatnonzero(np.diff(night.astype(int))) + 1, *jumps.astype(int)})
        spots = np.asarray(sig.get('spots', np.zeros((0, 2)))).reshape(-1, 2)
        orange = lamp_spots(sig)
        pieces = []
        for a, b in zip(bounds, bounds[1:]):
            inside = clip[(clip[:, 0] * FPS >= a) & (clip[:, 0] * FPS < b), 1]
            m = np.median(inside) if len(inside) >= 2 else 0.5
            lit = spots[(spots[:, 0] * FPS >= a) & (spots[:, 0] * FPS < b), 1]
            lamps = len(lit) >= 2 and np.median(lit) >= SPOTS_NIGHT  # unsure CLIP, lit lamps: night
            lit = orange[(orange[:, 0] * FPS >= a) & (orange[:, 0] * FPS < b), 1]
            lamps = lamps or (len(lit) >= 2 and np.median(lit) >= LAMPS_NIGHT)
            pieces.append((a, b, True if m >= 1 - NIGHT_CLIP_SURE else False if m <= NIGHT_CLIP_SURE
                           else True if lamps and len(inside) >= 2 else None))
        # unsure pieces between two sure ones of the same label take it (lmALZy7EF-g 29:50-32:47: dusk CLIP
        # reads 0.2-0.5 between night on both sides)
        known = [k for k, (_, _, x) in enumerate(pieces) if x is not None]
        for k, j in zip(known, known[1:]):  # ... also a run of unsure pieces
            if pieces[k][2] == pieces[j][2]:
                pieces[k + 1:j] = [(a, b, pieces[k][2]) for a, b, _ in pieces[k + 1:j]]
        for a, b, x in pieces:
            if x is not None:
                night[a:b] = x

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
            sat, lamps = sig['saturation'][span], dusk_lamps(sig)[span]
            if not seg['night'] and len(sat) and (np.median(sat) > p['dusk_saturation']
                                                  or np.median(lamps) > p['dusk_lamps']):
                notes.append(f"possibly dusk {seg['start']}-{seg['end']}s: street lights may be on, check night")
    return segments, notes


DUSK_EVERY_S = 10  # one colour frame this often for the dusk hint


def lamp_spots(sig):
    """[(second, lit orange lamps)] of light_spots_track. Saved as 'lamps' from 2026-10-16 16:04 to 23:00, over
    dusk_colours' measure of that name (88 videos): those have 'spots' and a 'lamps' of two columns."""
    if 'lamp_spots' in sig:
        return np.asarray(sig['lamp_spots']).reshape(-1, 2)
    old = np.asarray(sig.get('lamps', np.zeros(0)))
    return old if 'spots' in sig and old.ndim == 2 and old.shape[1] == 2 else np.zeros((0, 2))


def dusk_lamps(sig):
    """dusk_colours' % of lit orange lamp pixels per DUSK_EVERY_S, or zeros where lamp_spots overwrote it."""
    old = np.asarray(sig.get('lamps', np.zeros(0)))
    return old if old.ndim == 1 else np.zeros(len(sig['saturation']))


def dusk_colours(path):
    """(saturation, % of lit orange lamp pixels) of a colour frame every DUSK_EVERY_S seconds."""
    # ffmpeg reads the frames in one pass: OpenCV seeking in an AV1 download can hang (MilesofAmbience, 2026-10-08)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-an', '-vf', f'fps=1/{DUSK_EVERY_S},scale={W}:{H}',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True, timeout=1800).stdout
    sat, lamps = [], []
    for img in np.frombuffer(raw, np.uint8)[:len(raw) // (W * H * 3) * W * H * 3].reshape(-1, H, W, 3):
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
    boxes = model(img, imgsz=448, conf=0.25, device=device, verbose=False)[0].boxes
    cls = boxes.cls.tolist()
    # where road users sit in the picture (per mille of its height, -1 for none): a camera tilted up sees them
    # only at the bottom edge (KgEL7td3JK0 from 16:42: 950 against 620-660 looking ahead)
    road = [(y1 + y2) / 2 for c, (_, y1, _, y2) in zip(cls, boxes.xyxy.tolist()) if c in (0, 1, 2, 3, 5, 7)]
    height = int(1000 * sorted(road)[len(road) // 2] / img.shape[0]) if road else -1
    print(sum(c in (0, 1, 2, 3, 5, 9, 10, 11) for c in cls), sum(c in (0, 1, 9, 10, 11) for c in cls), height,
          flush=True)
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


CROSS_MIN_STOP_S = 15     # waits at least this long are checked for sped-up cross traffic
CROSS_MAX_S = 30          # at most this much of each wait
CROSS_BUDGET_S = {'gpu': 90, 'cpu': 15}  # seconds of waits tracked per video (yolo11x at 15 frames a second of video)
CROSS_CODE = """
import sys, cv2, numpy as np, torch
from ultralytics import YOLO
model, cap = YOLO('yolo11x.pt'), cv2.VideoCapture(sys.argv[1])
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
budget = float(sys.argv[2 if device != 'cpu' else 3])
fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, round(fps / 15))  # 15 frames a second: at 10 ByteTrack loses cars crossing at 150 km/h
for w in sys.argv[4].split(','):
    a, b = (float(x) for x in w.split('-'))
    b = min(b, a + budget)
    budget -= b - a
    if b <= a:
        break
    if model.predictor is not None and getattr(model.predictor, 'trackers', None):
        model.predictor.trackers[0].reset()  # a new wait: no track carries over
    tracks, f = {}, round(a * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    while f < b * fps:
        ok, img = cap.read()
        if not ok:
            break
        if f % step == 0:
            r = model.track(img, persist=True, imgsz=640, conf=0.3, classes=[0, 2], device=device, verbose=False,
                            tracker='bytetrack.yaml')[0]
            if r.boxes.id is not None:
                for (x1, y1, x2, y2), c, i in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.id.tolist()):
                    tracks.setdefault(int(i), []).append((f / fps, (x1 + x2) / 2, x2 - x1, y2 - y1, int(c)))
        f += 1
    for tr in tracks.values():
        tr = np.array(tr)
        if tr[-1, 0] - tr[0, 0] < 0.7:
            continue
        # across the picture in its own sizes a second: a car's length (4.5 m) side on, a person's height (1.7 m)
        size = np.median(tr[:, 2] if tr[0, 4] == 2 else tr[:, 3])
        per_s = abs(np.polyfit(tr[:, 0], tr[:, 1], 1)[0]) / size
        if per_s >= 0.2:
            print(round(tr[0, 0], 1), round(per_s * (4.5 if tr[0, 4] == 2 else 1.7) * 3.6), int(tr[0, 4]), flush=True)
"""


def cross_traffic(path, sig, p=PARAMS):
    """[(time s, km/h, YOLO class)] of the cars and people that move across the picture while the car waits (the
    longest waits first, CROSS_BUDGET_S in all). A waiting camera stands still, so their speed in their own sizes
    a second needs no calibration: 6-54 km/h for cars and 4-7 km/h for people at normal waits, 107-178 km/h for
    cars in YJH7sq2bCNc's wait at 6:12, sped up. Empty without the .venv or a wait."""
    moving = _rolling_median(sig['motion'], p['smooth_s'] * FPS) > p['moving_flow']
    waits = sorted([(a / FPS, min(b / FPS, a / FPS + CROSS_MAX_S)) for a, b in _runs(~moving)
                    if (b - a) / FPS >= CROSS_MIN_STOP_S], key=lambda w: w[0] - w[1])
    if not waits or not os.path.exists(YOLO_PYTHON):
        return np.zeros((0, 3))
    r = subprocess.run([YOLO_PYTHON, '-c', CROSS_CODE, path, str(CROSS_BUDGET_S['gpu']), str(CROSS_BUDGET_S['cpu']),
                        ','.join(f'{a:.1f}-{b:.1f}' for a, b in waits)], capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        print(f'  YOLO tracking failed, no cross traffic check: {r.stderr.strip()[-300:]}', flush=True)
    rows = [[float(x) for x in line.split()] for line in r.stdout.splitlines() if line.strip()]
    return np.array(rows, float).reshape(-1, 3)


GHOST_CODE = """
import sys, cv2, numpy as np, torch
from ultralytics import YOLO
if not torch.cuda.is_available():  # 15 frames a second of a whole video: only a CUDA GPU does it in minutes
    sys.exit(0)
model, cap = YOLO('yolo11x.pt'), cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
step = max(1, round(fps / 15))  # at 5 a second ByteTrack loses people between frames: misses and false alarms
tracks, f, end = {}, 0, 0
while f < fps * 2700:  # the first 45 min: a whole 3-hour road trip would print nothing for half an hour
    ok, img = cap.read()
    if not ok:
        break
    if f % step == 0:
        r = model.track(img, persist=True, imgsz=640, conf=0.3, classes=[0, 1, 2, 3, 5, 7], device='cuda',
                        verbose=False, tracker='bytetrack.yaml')[0]
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), i, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.conf.tolist()):
                tracks.setdefault(int(i), []).append((f / fps, x1 / W, y1 / H, x2 / W, y2 / H, c))
        end = f / fps
    f += 1
starts = [(tr[0][0], (tr[0][1] + tr[0][3]) / 2, (tr[0][2] + tr[0][4]) / 2, i) for i, tr in tracks.items()]
by_time = sorted((b[0], b, i) for i, tr in tracks.items() for b in tr)
times = [x[0] for x in by_time]
for i, tr in tracks.items():
    t, x1, y1, x2, y2, _ = b = tr[-1]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if (tr[-1][0] - tr[0][0] < 0.5 or np.median([x[5] for x in tr]) < 0.5  # a sure detection for a while
            or y2 - y1 < 0.08 or x1 <= 0.05 or x2 >= 0.95 or y2 >= 0.97 or t > end - 0.5):  # big, well inside
        continue
    lo, hi = np.searchsorted(times, t), np.searchsorted(times, t + 0.4, 'right')
    if any(j != i and o[1] <= cx <= o[3] and o[2] <= cy <= o[4] for _, o, j in by_time[lo:hi]):
        continue  # something passed in front: hidden, not gone
    if any(j != i and t < s <= t + 1 and abs(sx - cx) < 0.1 and abs(sy - cy) < 0.1 for s, sx, sy, j in starts):
        continue  # picked up again nearby under a new id
    print(round(t, 1), flush=True)
"""


def ghost_vanishes(path):
    """Times (s) a road user vanished mid-picture: a sure detection, big and well inside the frame, nothing passing
    in front of it and not found again nearby. A dissolve between two scenes fades people out where they stand
    (Q8iFS6Szcfc at 8:08: a woman on a crossing); one false alarm in 6 minutes of approved drives and far ones
    missed (5ki0IEvsSQI), so a hint only (user, 2026-10-11). On a CUDA GPU only (wsl5090, lin5080); empty
    elsewhere."""
    if not os.path.exists(YOLO_PYTHON):
        return np.zeros(0)
    r = subprocess.run([YOLO_PYTHON, '-c', GHOST_CODE, path], capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        print(f'  YOLO tracking failed, no ghost check: {r.stderr.strip()[-300:]}', flush=True)
    return np.array([float(x) for x in r.stdout.split()], float)


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


def fade_events(path, k=3, ratio=0.42, change=0.5):
    """[(start s, end s, lowest blend ratio, largest change)] of stretches where the picture may fade from one scene
    into another ("ghosts": both scenes show at once). Frames at 10 a second, brightness and contrast evened out
    (tunnels, headlights); a frame in a fade is close to the average of the frames k before and after it, which
    differ by at least change. A cut detector misses fades: no two frames differ much. Only a hint: slow, smooth
    stretches look alike (brittrips: 1-6 in 5-7 min; normal drives 0-3 in 11-23 min; a tunnel drive: 28 in 10)."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-an', '-vf', 'fps=10,scale=96:54,format=gray',
                          '-f', 'rawvideo', '-'], capture_output=True).stdout
    g = np.frombuffer(raw, np.uint8).reshape(-1, 54, 96).astype(np.float32)
    if len(g) <= 2 * k:
        return np.zeros((0, 4))
    g = (g - g.mean(axis=(1, 2), keepdims=True)) / (g.std(axis=(1, 2), keepdims=True) + 1)
    out = []
    # and frames 0.2 s apart too: a half-second dissolve averages out over 0.3 s (user, 2026-10-14: lKtZ5xqatFo at
    # 6:40, 0.44 against 0.35); on 40 approved videos (15.9 h) that adds 13 fades to the 98 found, 0.8 an hour
    for kk in sorted({k, 2}):
        before, mid, after = g[:-2 * kk], g[kk:-kk], g[2 * kk:]
        d = np.abs(before - after).mean(axis=(1, 2))
        r = np.abs(mid - (before + after) / 2).mean(axis=(1, 2)) / np.maximum(d, 1e-3)
        out += [((a + kk) / 10, (b + kk) / 10, r[a:b].min(), d[a:b].max())
                for a, b in _runs((r < ratio) & (d > change)) if b - a >= FADE_MIN_S * 10]
    merged = []  # one fade found at both spacings is one fade
    for x in sorted(out):
        if merged and x[0] <= merged[-1][1] + 0.5:
            m = merged[-1]
            merged[-1] = (m[0], max(m[1], x[1]), min(m[2], x[2]), max(m[3], x[3]))
        else:
            merged.append(x)
    return np.array(merged, float).reshape(-1, 4)


# an edit cut that skips footage: a frame changing SKIP_RATIO times more than the frames around it and by at
# least SKIP_SCORE, alone (livestream stutter and intro montages jump many times in a row), and not in the first
# SKIP_INTRO_S (title cards) or last SKIP_OUTRO_S (end screens). Checked on whole approved videos: these keep
# the waits cut out at red lights and the cuts to other streets, and none of the stutter or fades.
SKIP_RATIO, SKIP_SCORE, SKIP_ALONE_S, SKIP_INTRO_S, SKIP_OUTRO_S = 25, 0.06, 10, 30, 10
SKIP_STREAK = 10       # this many analysed videos of a channel in a row with skips: the channel edits its drives
FADE_MIN_S = 0.4       # a fade between scenes lasts at least this; shorter dips are flicker


# a jump that brightness and contrast evened out take away is the camera adjusting to the light, not a cut (user,
# 2026-10-13: XEtj22qw_sU out of an underpass at 35:22, BXizY_nii94 at 4:53 at night): the frame change at the
# jump against the median of the 2 s around it, each frame normalised, was 3.7-4.1 there and 1.2-3.3 on 4 of 17
# random cuts of approved videos (3 under bridges, the dashcam clock running on), 11.8-68 on the other 13 (8U8iqg_C27M
# 16:53: the cars at the light vanish) and 19-80 on R3hofhG0UYI's
SKIP_NORM_RATIO = 8
SKIP_STOP_S, SKIP_STOP_MOTION, SKIP_SAME_NCC = 2, 0.2, 0.9  # a car pulling away is no cut (pulls_away)
# sky change across an edit cut that makes it a splice of night and day: on the approved segments 5 gives +363 s
# right and none worse (XDBi8ukETEA, -TY9uuHDcP8, 7oAdCb9eINM); 30 lost 9eI2niyrIwo, -2,718 s
STITCH_SKY_JUMP = 5
HIGHWAY_GAP_S = 600  # a piece up to this long between rural/highway stretches of a highway title's video: no town
DRIVER_GAP_S = 600  # footage between two camera-on-the-driver stretches up to this long is the driver too
DRIVER_CUT_SHARE = 0.2  # face frames in a minute from which a camera on the driver is cut, not only noted
WAIT_S, WAITS_PER_HOUR = 20, 3  # few waits at lights: see propose


def skip_norm(path, sig):
    """[(time s, normalised change ratio)] of the strong skip candidates (see SKIP_NORM_RATIO)."""
    out = []
    for t, score, ratio in np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3):
        if ratio < SKIP_RATIO or score < SKIP_SCORE:
            continue
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{max(0, t - 1):.3f}', '-i', path, '-t', '2', '-an',
                              '-vf', 'scale=160:90,format=gray', '-f', 'rawvideo', '-'],
                             capture_output=True, timeout=120).stdout
        g = np.frombuffer(raw, np.uint8)[:len(raw) // 14400 * 14400].reshape(-1, 90, 160).astype(np.float32)
        if len(g) < 5:
            continue
        g = (g - g.mean(axis=(1, 2), keepdims=True)) / (g.std(axis=(1, 2), keepdims=True) + 1)
        d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
        out.append((float(t), float(d.max() / max(np.median(d), 1e-3))))
    return out


def photo_inserts(sig, max_s=15, share=0.5):
    """[(start, end)] analysis frames of photos shown between pieces of the drive (user, 2026-10-13:
    fxdZiHoT8ns's statue at 47:47, WQyCLXKVF4U's shop at 12:18): between two hard jumps at most max_s apart,
    mostly frozen frames, of two or more different stills, unlike the street before. A wait with cars passing in
    front (_e8uM-XAhH4) or between edit cuts (6iepzXRnnI4) shows one picture, that street. In approved footage
    40 s in 5 segments (WQyCLXKVF4U's real photos among them), in rejected 66 s in 12."""
    ncc, motion, sky, std = (np.asarray(sig[k]) for k in ('ncc', 'motion', 'sky', 'std'))
    frozen = (ncc >= 0.999) & (motion <= 0.02)
    jumps = np.flatnonzero(ncc < 0.6)
    out = []
    for i, a in enumerate(jumps):
        for b in jumps[i + 1:]:
            if (b - a) / FPS > max_s:
                break
            if b - a < 4 or frozen[a + 1:b].mean() < share:
                continue
            inside, before = np.flatnonzero(frozen[a + 1:b]) + a + 1, slice(max(0, a - 4), a)
            looks = [(np.median(sky[a + 1 + x:a + 1 + y]), np.median(std[a + 1 + x:a + 1 + y]))
                     for x, y in _runs(frozen[a + 1:b])]
            pictures = 1 + sum(abs(p[0] - q[0]) > 8 or abs(p[1] - q[1]) > 5 for p, q in zip(looks, looks[1:]))
            if pictures >= 2 and (abs(np.median(sky[inside]) - np.median(sky[before])) > 15
                                  or abs(np.median(std[inside]) - np.median(std[before])) > 5):
                out.append((int(a), int(b) + 1))
    merged = []
    for a, b in sorted(out):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def pulls_away(sig, t):
    """Whether a skip candidate at t is the car pulling away from a standstill: still before it (motion under
    SKIP_STOP_MOTION over SKIP_STOP_S) and the same scene across it (frame correlation SKIP_SAME_NCC or more).
    Against stillness the first moving frames jump like a cut (QnPFc7WrsQk 1:45; user, 2026-10-15). Of 439 cuts in
    reviewed stretches, 55 are such: 54 inside an approved segment (false), 1 where the reviewer cut."""
    i = int(t * FPS)
    if i < 1 or i + 2 > min(len(sig['motion']), len(sig['ncc'])):
        return False
    return (np.median(sig['motion'][max(0, i - SKIP_STOP_S * FPS):i]) < SKIP_STOP_MOTION
            and np.min(sig['ncc'][i - 1:i + 2]) >= SKIP_SAME_NCC)


def under_bridge(sig, a, b):
    """Whether the light around a-b (s) is a pass under a bridge: the sky nearly black within it (under BRIDGE_DARK of
    the 10 s before) and back within BRIDGE_BACK_S (BRIDGE_LIGHT of it or more; the camera's exposure often
    overshoots). A fade between scenes does not drop to black and come back in seconds (trvKU3hzes8 4:27, 9:34,
    10:57, 14:04; CokQn5byUS8 4:05-4:06: sky 167, 3, 234; user, 2026-10-16)."""
    sky = sig['sky']
    i, j = int(a * FPS), int(b * FPS)
    before = sky[max(0, i - 10 * FPS):max(0, i - FPS)]
    if not len(before) or j + BRIDGE_BACK_S * FPS > len(sky):
        return False
    ref = np.median(before)
    low = sky[max(0, i - FPS):j + FPS].min()
    back = np.median(sky[j + FPS:j + BRIDGE_BACK_S * FPS]) if BRIDGE_BACK_S > 1 else sky[j + FPS]
    return bool(low < BRIDGE_DARK * ref and back >= BRIDGE_LIGHT * ref)


BRIDGE_DARK, BRIDGE_LIGHT, BRIDGE_BACK_S = 0.3, 0.8, 5


def overpass_like(sky, i):
    """Whether the light at a picture cut (frame i) looks like a long overpass: dimming already in the 3 s before
    (under 0.8 of the 7 s before that) and back to 0.85 of it within 10 s, holding for 3 s (ch47LHcVB8U 14:40: sky
    155, 119 just before, about 100 for 9 s, 145; user, 2026-10-16). A hint, not a reason to drop the cut: of the
    326 picture cuts shown in reviewed stretches, 9 false and 12 real ones are such."""
    ref = np.median(sky[max(0, i - 10 * FPS):max(1, i - 3 * FPS)])
    if not ref or sky[max(0, i - 3 * FPS):max(1, i - 1)].min() >= 0.8 * ref:
        return False
    return any(sky[k] >= 0.85 * ref and np.median(sky[k:k + 3 * FPS]) >= 0.85 * ref
               for k in range(i + 2 * FPS, min(len(sky) - 3 * FPS, i + 10 * FPS + 1)))


def skip_times(sig):
    """Seconds at which the video skips footage (edit cuts), from the skip candidates that are strong and alone."""
    ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
    light = [t for t, r in np.asarray(sig.get('skip_norm', np.zeros((0, 2)))).reshape(-1, 2) if r < SKIP_NORM_RATIO]
    out = []
    for t, score, ratio in ev:
        if ratio < SKIP_RATIO or score < SKIP_SCORE or any(abs(t - x) < 0.01 for x in light) or pulls_away(sig, t):
            continue
        gap = np.abs(ev[:, 0] - t)
        if not ((gap > 0.5) & (gap <= SKIP_ALONE_S)).any():  # frames right at a cut can both jump
            out.append(float(t))
    return out


GARAGE_NIGHT_SKY = 60  # a video's median sky under this: at night (approved night videos: 20-50)
GARAGE_PAD_S = 6  # seconds of ramp around a garage cut with it (stretches end a few seconds inside the garage)


def dark_edges(sig, max_s=240, min_s=8):
    """[(start s, end s, 'start'|'end'|'middle')] garage candidates at a video's start or end: within max_s of the
    edge, the sharpest rise in sky brightness (10 s before vs after, at least 30: driving out), with the stretch
    before it 35 darker than the rest of the video, the minute after it 30 brighter than the stretch (a street
    shaded by buildings may lead to it: tf1JPTcdgA0), and its sky no brighter than its road (a ceiling). Against
    the rest, not the whole video: half of g9AWq8zuUvU is in a garage. Tunnels, the El and dusk pass this too,
    hence garage_edges."""
    sky = _rolling_median(sig['sky'][::FPS], 5)  # per second
    ground = sig['ground'][:len(sig['sky'])][::FPS]
    out = []
    # at night a lit garage is brighter than the street (tdMOjN3D9vQ: 100 under its lamps until 0:17, then 20): the
    # same test on the sky turned upside down
    signs = (1, -1) if np.median(sky) < GARAGE_NIGHT_SKY else (1,)
    for edge, x, y in (('start', sky, ground), ('end', sky[::-1], ground[::-1])):
        last = min(max_s, len(x) - 60)  # leave a minute of drive after the stretch
        if last <= min_s:
            continue
        for z in (sign * x for sign in signs):
            jump = [np.median(z[t:t + 10]) - np.median(z[max(0, t - 10):t]) for t in range(min_s, last)]
            t = min_s + int(np.argmax(jump))
            rest = np.median(z[t:])
            if (max(jump) >= 30 and np.median(z[:t]) <= rest - 35 and np.median(z[t:t + 60]) >= np.median(z[:t]) + 30
                    and np.median(x[:t] - y[:t]) < 20):
                t = min(t + GARAGE_PAD_S, len(x))
                out.append((0, t, edge) if edge == 'start' else (len(x) - t, len(x), edge))
                break
    # in the middle: parked in a garage and out again (wt1UHqNXQ1E 5:19-6:43): 20 s to 15 min 50 darker than the
    # video under a ceiling, dropping into it by 30 within 20 s and out of it by 30 within half a minute (a dusk
    # does neither); G4lf48BMAXw's 9 minutes in a garage (11:11-20:43) have a lit stretch of 23 s and a slow ramp out
    dark = (sky <= np.median(sky) - 50) & (sky - ground[:len(sky)] < 20)
    runs = []
    for a, b in _runs(dark):  # bridge a lamp or a car's lights for a few seconds
        if runs and a - runs[-1][1] <= 30:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    for a, b in runs:
        if (20 <= b - a <= 900 and a >= 60 and b <= len(sky) - 60 and not any(x < b and a < y for x, y, _ in out)
                and np.median(sky[a - 20:a]) - np.median(sky[a:a + 20]) >= 30
                and np.median(sky[b + 10:b + 30]) - np.median(sky[b - 20:b]) >= 30):
            out.append((max(0, a - GARAGE_PAD_S), min(len(sky), b + GARAGE_PAD_S), 'middle'))
    return out


SCENE_CODE = """
import json, sys, open_clip, torch
from PIL import Image
ask = json.loads(sys.argv[1])
model, _, prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tok = open_clip.get_tokenizer('ViT-B-32')
with torch.no_grad():
    text = model.encode_text(tok([ask.get('prefix', 'a dashcam photo of ') + s for s in ask['scenes']]))
    text /= text.norm(dim=-1, keepdim=True)
    for path in sys.argv[2:]:
        img = model.encode_image(prep(Image.open(path)).unsqueeze(0))
        img /= img.norm(dim=-1, keepdim=True)
        print(float((100 * img @ text.T).softmax(-1)[0, :ask['k']].sum()), flush=True)
"""
# (scenes CLIP chooses between, how many of the first ones count as a yes)
GARAGE_SCENES = (['the inside of a parking garage', 'a road tunnel', 'a street under an elevated railway',
                  'a road at dusk', 'a road at night', 'a city street', 'a road through a forest', 'a highway'], 1)
PARKED_SCENES = (['a parking lot', 'the inside of a parking garage',
                  'a car parked at the side of a residential street', 'a car waiting at a traffic light',
                  'a car stopped in traffic behind another car', 'a busy city intersection', 'a road'], 3)


def garage_edges(path, sig, work_dir):
    """The dark_edges that CLIP (ViT-B-32, in the project's .venv) sees as the inside of a parking garage: the
    middle of its scores on five frames at least 0.9. On the 63 dark edges of 818 reviewed videos it found all 18
    garages (Cur__d1g6j4, UkO_wSNa-zY, ...) and none of the tunnels, El, dusk drives, a street beside a car park,
    a stop facing a shopfront (0.72 and 0.83 there)."""
    edges = dark_edges(sig) if 'ground' in sig else []
    if not edges or not os.path.exists(YOLO_PYTHON):
        return []
    return _clip_stretches(path, edges, work_dir, GARAGE_SCENES, 5)


INTRO_SCENES = (['a map', 'a map with a route drawn on it', 'a title card with text', 'a logo on a plain background',
                 'a dashcam view of a road', 'a view of a street from inside a car'], 4)


def clip_intro(path, duration, work_dir, max_s=60):
    """Seconds where an opening map, title card or logo ends (CLIP, one frame a second over the first max_s), else
    None: ofoaWwzK3_Y shows a map with the route until 0:10 (1.00 a second, then 0.00), 4mGZZ9BGunA until 0:09;
    videos that open on the road score 0.00. Its colours are no help: that map is no brighter than the drive."""
    scores = _clip_scores(path, [t + 0.5 for t in range(int(min(max_s, duration or max_s)))], work_dir, INTRO_SCENES)
    times = sorted(scores)
    if len(times) < 3 or scores[times[0]] < 0.9 or scores[times[1]] < 0.9:
        return None
    end = next((t for t in times if scores[t] < 0.5), None)
    return None if end is None else int(end - 0.5)


# (scenes, how many first ones count as a yes, no "a dashcam photo of" before them)
OUTRO_SCENES = (['a still photograph framed with black borders', 'a slideshow of photos', 'a map',
                 'a title card with text', 'a logo on a plain background', 'an end screen with video thumbnails',
                 'a dashcam view of a road', 'a view of a street from inside a car',
                 'the back of the car in front seen from a car', 'a car parked in a parking lot or garage',
                 'a building seen from a car window'], 6, '')


def clip_outro(path, duration, work_dir, max_s=120):
    """Seconds where a closing slideshow, map, title or end screen starts (CLIP, a frame every 2 s over the last
    max_s), else None: the earliest frame of 0.5 or more from which 80% of the frames to the end are, at least 10 s
    of them. PSh0r4-U6qw's photos of sculptures from 34:36 score 0.37-1.00 (most over 0.9), the drive before 0.00;
    of 7,200 contact-sheet frames of 600 approved drives 11 pass 0.5."""
    if not duration:
        return None
    times = [t + 0.5 for t in range(max(0, int(duration) - max_s), int(duration) - 1, 2)]
    scores = _clip_scores(path, times, work_dir, OUTRO_SCENES)
    s = [scores[t] for t in sorted(scores)]
    start = next((i for i in range(len(s)) if s[i] >= 0.5 and np.mean(np.array(s[i:]) >= 0.5) >= 0.8), None)
    return None if start is None or len(s) - start < 5 else int(sorted(scores)[start])


def parking_lots(path, sig, work_dir, p=PARAMS):
    """[(start s, end s)] of the stretches of 30 s or more where the car moves less than 40% of the time (over a
    minute) and CLIP sees a parking lot, a garage or a parked car, not a wait in traffic (PARKED_SCENES, 3 frames).
    Waits at intersections are what the dataset wants, so standing still alone cuts nothing: of 32 mostly
    standing segments the reviewer approved CLIP put 1 in a lot, of 75 rejected 20 (MKQqzpQbZQ8 at 86:39)."""
    moving = _rolling_median(sig['motion'], p['smooth_s'] * FPS) > p['moving_flow']
    slow = _rolling_mean(moving.astype(float), 60 * FPS) < 0.4
    runs = [(a / FPS, b / FPS) for a, b in _runs(slow) if (b - a) / FPS >= 30]
    return _clip_stretches(path, runs, work_dir, PARKED_SCENES, 3) if runs else []


KERB_SCENES = (['a car pulled over at the side of the road', 'a car parked by the curb next to a sidewalk',
                'a car stopped on the shoulder of the road', 'a car waiting at a red traffic light at an intersection',
                'a car waiting in traffic behind another car', 'a car driving on a road'], 3)


def kerb_stops(path, sig, work_dir, p=PARAMS, min_s=20):
    """[(start s, end s)] of the stops of min_s or more inside a video that CLIP sees as pulled over at the kerb
    (KERB_SCENES, 3 frames, middle score 0.6): a note only. SayBzo1EeJ0's two kerb stops (2:30, 11:40) score
    0.61-0.84 and its red light 0.16-0.21, but its queue behind a car 0.50-0.61, and waits are what the dataset
    wants."""
    moving = _rolling_median(sig['motion'], p['smooth_s'] * FPS) > p['moving_flow']
    n = len(moving)
    runs = [(a / FPS, b / FPS) for a, b in _runs(~moving) if (b - a) / FPS >= min_s and a > 0 and b < n]
    return _clip_stretches(path, runs, work_dir, KERB_SCENES, 3, at=0.6) if runs else []


def parked_edges(path, sig, work_dir):
    """The edge_stops where CLIP sees a parked car (a parking lot, a garage, the kerb of a residential street): the
    middle of its scores on three frames at least 0.9. Most stops at a video's start or end are waits in traffic
    (GPCqhS6hQ40: a dashcam's files start wherever the last one ended): on the contact sheets of 156 such stops it
    gave 97 under 0.1, all queues and lights, and those of 0.9 and more were parking lots, garages and kerbs."""
    stops = edge_stops(sig)
    return _clip_stretches(path, stops, work_dir, PARKED_SCENES, 3) if stops else []


def _clip_scores(path, times, work_dir, scenes):
    """{time: CLIP score} of the frames at times (s) for scenes (list, how many first ones count as a yes); a frame
    ffmpeg could not take is left out; {} when CLIP is missing or fails."""
    if not times or not os.path.exists(YOLO_PYTHON):
        return {}
    shots = []
    for i, t in enumerate(times):
        shot = os.path.join(work_dir, f'clip{i}.jpg')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.1f}', '-i', path, '-frames:v', '1', shot],
                       capture_output=True, timeout=120)
        if os.path.exists(shot):
            shots.append((t, shot))
    if not shots:
        return {}
    ask = json.dumps({'scenes': scenes[0], 'k': scenes[1],
                      'prefix': scenes[2] if len(scenes) > 2 else 'a dashcam photo of '})
    r = subprocess.run([YOLO_PYTHON, '-c', SCENE_CODE, ask, *(f for _, f in shots)], capture_output=True, text=True,
                       cwd=HERE)
    for _, shot in shots:
        os.remove(shot)
    scores = [float(x) for x in r.stdout.split()]
    if r.returncode or len(scores) != len(shots):
        print(f'  CLIP failed, no scene check: {r.stderr.strip()[-300:]}', flush=True)
        return {}
    return {t: score for (t, _), score in zip(shots, scores)}


def _clip_stretches(path, stretches, work_dir, scenes, frames, at=0.9):
    """The stretches (start s, end s, ...) whose frames CLIP sees as one of scenes' first ones: the middle of the
    scores of at least half the frames (a frame ffmpeg could not take is left out) is `at` or more."""
    if not stretches:
        return []
    times = [(k, a + (b - a) * q) for k, (a, b, *_) in enumerate(stretches)
             for q in np.arange(1, frames + 1) / (frames + 1)]
    got = _clip_scores(path, [t for _, t in times], work_dir, scenes)
    shots, scores = [(k, t) for k, t in times if t in got], [got[t] for k, t in times if t in got]
    by = {}
    for (k, _), score in zip(shots, scores):
        by.setdefault(k, []).append(score)
    return [x for k, x in enumerate(stretches)
            if len(by.get(k, [])) >= (frames + 1) // 2 and np.median(by[k]) >= at]


def montage_end(sig, within=45, gap=20, tail=15):
    """Seconds where a montage of highlights at the start of a video ends and the drive begins, else None:
    at least two strong cuts (scene score 0.1, 50 times the frames around) in the first `within` seconds, at most
    `gap` apart; it ends at the last cut (15 times) up to `tail` after the last strong one. On five MilesofAmbience
    videos it puts the start within 0-5 s of the reviewer's (0:17, 0:35, 0:42 for 0:46, 0:19 for 0:15, 0:22 for
    0:27); other channels cut early for other reasons, hence montage_channel."""
    ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
    strong = sorted({round(float(t), 1) for t, score, ratio in ev if t <= within and score >= 0.1 and ratio >= 50})
    run = []
    for t in strong:
        if run and t - run[-1] > gap:
            break
        run.append(t)
    if len(run) < 2:
        return None
    later = [float(t) for t, _, ratio in ev if run[-1] < t <= run[-1] + tail and ratio >= 15]
    return round(max(later or [run[-1]]) + 1)


def montage_channel(plan, out_dir):
    """Whether this channel opens its videos with highlight montages: in the videos reviewed so far that have one
    by montage_end, the reviewer started the drive after it at least 3 times and 3 times as often as not."""
    after = before = 0
    for p in plan.values():
        approved = [s for e in p.get('entries', []) for s in e['segments']
                    if s.get('decision') == 'approve' or s.get('applied')]
        path = os.path.join(out_dir, f"{p['video']}_signals.npz")
        if not approved or not os.path.exists(path):
            continue
        end = montage_end(dict(np.load(path)))
        if end is not None:
            if min(s['start'] for s in approved) >= end - 5:
                after += 1
            else:
                before += 1
    return after >= 3 and after >= 3 * before


MONTAGE_MARK = 'opens_with_montage'  # a file in a channel's folder: its reviews say it opens with highlights


def trim_montage(p, sig):
    """Start a just-proposed video after its opening montage of highlights."""
    end = montage_end(sig)
    if end is None:
        return
    for e in p.get('entries', []):
        e['segments'] = [dict(s, start=max(s['start'], end)) for s in e['segments'] if s['end'] > end + 30]
    p['entries'] = [e for e in p['entries'] if e['segments']]
    p['note'] = (f"{p.get('note', '')}; intro 0-{end}s: a montage of highlights (this channel opens with one)"
                 .lstrip('; '))


def _ymd(ddmmyyyy):
    """'03052021' (the mapping's upload dates) -> '20210503'; '' for none."""
    d = str(ddmmyyyy or '')
    return d[4:] + d[2:4] + d[:2] if len(d) == 8 else ''


_newest_upload = {}


def newest_upload(df):
    """{channel id: YYYYMMDD of its newest video in the mapping}."""
    key = id(df), len(df)
    if key not in _newest_upload:
        newest = {}
        for chans, dates in zip(df['channel'], df['upload_date']):
            for c, d in zip(str(chans).strip('[]').split(','), str(dates).strip('[]').split(',')):
                c, d = c.strip(), _ymd(d.strip())
                if c.startswith('UC') and d > newest.get(c, ''):
                    newest[c] = d
        _newest_upload.clear()
        _newest_upload[key] = newest
    return _newest_upload[key]


_lead_ins = {}


def channel_lead_in(channel_id, df, min_videos=10):
    """Seconds this channel's drives never start before, from the mapping: 90% of its videos (at least min_videos)
    begin their first segment this late or later (an intro, highlights, a title: Wanderizm, AroundSydney), else
    None. It applies to new uploads of a channel done before."""
    if not channel_id:
        return None
    if channel_id not in _lead_ins:
        starts = []
        for chans, cell in zip(df['channel'], df['start_time']):
            if channel_id not in str(chans):
                continue
            try:
                firsts = ast.literal_eval(cell)
            except (ValueError, SyntaxError):
                continue
            starts += [min(f) for c, f in zip(str(chans).strip('[]').split(','), firsts)
                       if c.strip() == channel_id and f]
        lead = float(np.percentile(starts, 10)) if len(starts) >= min_videos else 0
        _lead_ins[channel_id] = int(lead) if lead >= 5 else None
    return _lead_ins[channel_id]


def trim_lead_in(p, lead):
    """Start a just-proposed video no earlier than its channel's drives start (channel_lead_in)."""
    if not any(s['start'] < lead for e in p.get('entries', []) for s in e['segments']):
        return
    for e in p['entries']:
        e['segments'] = [dict(s, start=max(s['start'], lead)) for s in e['segments'] if s['end'] > lead + 30]
    p['entries'] = [e for e in p['entries'] if e['segments']]
    if not p['entries']:
        del p['entries']
        p['exclude'] = f'nothing left after the first {lead}s, where this channel\'s drives start'
    p['note'] = (f"{p.get('note', '')}; intro 0-{lead}s: this channel's drives start at {lead}s or later "
                 '(90% of its videos in the mapping)').lstrip('; ')


# the note's items that may mean rejecting the whole video, shown red on the review page (user, 2026-10-16)
REJECT_HINT = re.compile(r'reject|sped up|slowed|few waits|the title mentions|camera on the driver\?|edited drive|'
                         r'vanished mid-picture|skips footage|time stamp in the picture jumps|walking or a handheld|'
                         r'drone|set the locality|in the dark: check', re.I)


def note_html(note):
    """The proposal's note as HTML, its REJECT_HINT items in red."""
    from markupsafe import Markup, escape
    return Markup('; ').join(Markup('<span class="warn">%s</span>') % x if warns(x) else escape(x)
                             for x in (note or '').split('; '))


def warns(item):
    """Whether a note item is a REJECT_HINT: fades only from FADES_RED of them (reviewed videos with 1-2 fades: 24%
    rejected, 6 or more: 39%, none: 15%; 2026-10-16)."""
    if item.startswith('possible fades'):
        more = re.search(r'and (\d+) more', item)
        return len(re.findall(r'\d+:\d\d', item)) + (int(more.group(1)) if more else 0) >= FADES_RED
    return bool(REJECT_HINT.search(item))


FADES_RED = 6


def gap_reason(note, a, b, night_change=False):
    """Why two segments of a video are apart, in a few words, from the proposal's note (user, 2026-10-14): the cuts
    whose stretch meets the gap a-b (s), a cut in the picture or an edit cut at it, a switch between day and night."""
    out = []
    for item in (note or '').split('; '):
        if ((item.startswith('possibly') and not item.startswith('possibly dusk'))
                or '(kept' in item or 'kept:' in item):
            continue  # a hint that left the footage in; dusk stays: a split at it outlives a changed day/night
            # (3ivZS04LX78: the reason was gone once both sides were set to night; user, 2026-10-16)
        if (m := re.match(r'(.+?) (\d+)-(\d+)s\b', item)) and int(m.group(2)) <= b + 2 and int(m.group(3)) >= a - 2:
            # without its explanation in brackets: a long reason widened the column and squeezed the buttons
            # (rgnCR9pb1aM: "camera turned to the side (the video switches angles)"; user, 2026-10-15)
            out.append(re.sub(r' \([^)]*\)$', '', m.group(1)))
        elif (m := re.match(r'cut at (\d+)s', item)) and a - 3 <= int(m.group(1)) <= b + 3:
            out.append('a cut in the picture, or an overpass?' if 'an overpass?' in item else 'a cut in the picture')
        elif item.startswith('the video skips footage at') and any(
                a - 3 <= int(x) * 60 + int(y) <= b + 3 for x, y in re.findall(r'(\d+):(\d\d)', item.split(' (')[0])):
            out.append('an edit cut')
    if night_change:
        out.append('day / night')
    return ', '.join(dict.fromkeys(out))


def footage_skip(sig):
    """Where the video skips footage (edit cuts), as "m:ss, m:ss and m:ss", else None: every cut, not only the
    first (R3hofhG0UYI: 10:53, 13:28, 14:25 and 19:32; user, 2026-10-12), so each can be checked for a missing
    wait at a light."""
    end = len(sig['motion']) / FPS
    ts = [f'{int(t) // 60}:{int(t) % 60:02d}' for t in skip_times(sig) if SKIP_INTRO_S < t < end - SKIP_OUTRO_S]
    ts = list(dict.fromkeys(ts))  # frames of one cut fall in the same second (xhW4Hz6Uazo: 7:52 three times)
    return None if not ts else ts[0] if len(ts) == 1 else ', '.join(ts[:-1]) + ' and ' + ts[-1]


# where the first frames are read for a GPS overlay; without one there, none is read (stream frames misread
# often: 6 tries)
GPS_PROBE_S = (10, 45, 90, 150, 210, 270)
GPS_EVERY_S = 30            # then one frame this often
# "N40.12345 W75.12345", "40.12345N 75.12345W", "40.123456, -75.123456" as dashcams print them
GPS_HEMI = re.compile(r'([NS])\s*(\d{1,2}[.,]\d{3,})\D{0,4}?([EW])\s*(\d{1,3}[.,]\d{3,})'
                      r'|(\d{1,2}[.,]\d{3,})\s*°?\s*([NS])\W{0,4}(\d{1,3}[.,]\d{3,})\s*°?\s*([EW])')
GPS_SIGNED = re.compile(r'(?<![\d.])(-?\d{1,2}\.\d{4,})(?:\s*[,;]?\s+|\s*(?=-))(-?\d{1,3}\.\d{4,})(?!\d|\.\d)')


def parse_gps(text):
    """(lat, lon) of the first coordinates in a frame's text, else None."""
    text = re.sub(r"(\d)['’`/](\d)", r'\1.\2', text or '').replace("'", ' ')  # 49'29065'-123.13081
    text = re.sub(r'(?<![\d.])(-?\d{1,3}) (\d{5,6})(?![\d.])', r'\1.\2', text)  # 360p: 49 25780 -123.17192
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
    text = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '11'], input=png, capture_output=True,
                          timeout=120).stdout.decode(errors='ignore')
    # dashcams print a line along the bottom or top edge ("GARMIN 29-05-2021 12:05:10 49.29065 -123.13081"), too
    # small for a read of the whole frame: those strips, enlarged, read as one line each
    # (white text on the picture: enlarged 3 times and turned black on white, 3p9Q_K6pkjo at 720p); some sit
    # higher, above the bonnet (4FkZW-vMSE8's "ROAV 01/22/2017 02:28:24": the bottom tenth)
    for crop in ('iw:ih*0.045:0:ih*0.955', 'iw:ih*0.045:0:0', 'iw:ih*0.1:0:ih*0.9'):
        strip = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-vf',
                                f"crop={crop},scale=-2:100,format=gray,lut=y='if(gt(val\\,190)\\,0\\,255)'",
                                '-f', 'image2pipe', '-vcodec', 'png', '-'], input=png, capture_output=True,
                               timeout=60).stdout
        if strip:
            line = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '7', '-c',
                                   'tessedit_char_whitelist=0123456789.,-:/NSEW '], input=strip,
                                  capture_output=True, timeout=120).stdout.decode(errors='ignore')
            text += '\n' + line
    # a phone app's time and date in a corner, two lines in letters and digits ("12:58 PM" over "Feb 6, 2025" at
    # the bottom left of CuseVegas's drives, REnfw4fpZ5A): read without the digit whitelist
    for crop in ('iw*0.35:ih*0.17:0:ih*0.82',):
        strip = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-vf',
                                f"crop={crop},scale=-2:200,format=gray,lut=y='if(gt(val\\,190)\\,0\\,255)'",
                                '-f', 'image2pipe', '-vcodec', 'png', '-'], input=png, capture_output=True,
                               timeout=60).stdout
        if strip:
            read = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '6'], input=strip, capture_output=True,
                                  timeout=120)
            text += '\n' + read.stdout.decode(errors='ignore')
    return text


CLOCK = re.compile(r'(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)[:.]([0-5]\d)(?!\d)')  # 12:05:10 in a dashcam stamp


def parse_clock(text):
    """Seconds since midnight of the first clock time in a frame's text, else None."""
    m = CLOCK.search(text or '')
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


STAMP_TIME = re.compile(r'(?<!\d)(1[0-2]|0?[1-9])[:.]([0-5]\d)\s*([AP])\.?\s*M\b', re.I)
STAMP_DATE = re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(20\d\d)\b',
                        re.I)
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def parse_stamp(text):
    """Seconds since 2000 of a time and date stamp to the minute in a frame's text ("12:58 PM" and "Feb 6, 2025"),
    else None."""
    tm, dm = STAMP_TIME.search(text or ''), STAMP_DATE.search(text or '')
    if not tm or not dm:
        return None
    hour = int(tm.group(1)) % 12 + (12 if tm.group(3).upper() == 'P' else 0)
    try:
        day = datetime.date(int(dm.group(3)), MONTHS.index(dm.group(1).lower()[:3]) + 1, int(dm.group(2)))
    except ValueError:  # a misread day (Feb 31)
        return None
    return (day - datetime.date(2000, 1, 1)).days * 86400 + hour * 3600 + int(tm.group(2)) * 60


STAMP_JUMP_S = 150  # a time and date stamp (to the minute) off the video's own time by this much more: a cut


def stamp_cuts(sig):
    """Seconds of the video where its time and date stamp jumps (the footage of another drive follows): its offset
    from the video's time changes by STAMP_JUMP_S or more and the next reading keeps the new offset (a misread
    does not). The cut is the strongest frame jump between the two readings, else their middle (REnfw4fpZ5A, a
    "MEGACUT" of CuseVegas's drives; user, 2026-10-16)."""
    st = np.asarray(sig.get('stamp', np.zeros((0, 2)))).reshape(-1, 2)
    off = st[:, 1] - st[:, 0]
    ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
    cuts = []
    for i in range(len(st) - 2):
        steady = i == 0 or abs(off[i] - off[i - 1]) < STAMP_JUMP_S  # not the way back from a misread
        if steady and abs(off[i + 1] - off[i]) >= STAMP_JUMP_S and abs(off[i + 2] - off[i + 1]) < STAMP_JUMP_S:
            t0, t1 = st[i, 0], st[i + 1, 0]
            inside = ev[(ev[:, 0] > t0) & (ev[:, 0] < t1)]
            cuts.append(float(inside[np.argmax(inside[:, 2]), 0]) if len(inside) else float((t0 + t1) / 2))
    return cuts


def gps_track(video_id, duration):
    """([(t, lat, lon)] read from a GPS overlay in the picture or None, [(t, clock seconds)] read from a time stamp
    in it, [(t, seconds since 2000)] from a time and date stamp to the minute: parse_stamp), all empty when the
    first frames show none. Frames come from the 720p stream (the 240p copy is too
    small to read), so only a few are fetched."""
    if not shutil.which('tesseract') or not duration:
        return None, [], []
    try:
        stream = _stream(video_id)
        texts = [(t, _frame_text(stream, t)) for t in GPS_PROBE_S if t < duration]
        if not any(parse_gps(x) or parse_clock(x) or parse_stamp(x) for _, x in texts):
            return None, [], []
        # at most ~200 frames: a 3-hour road trip read every 30 s printed nothing for 45 min and was stopped as
        # stuck (1D9hw6ob380); a line every 50 frames shows it is going
        every = max(GPS_EVERY_S, int(duration / 200))
        times = [t for t in range(GPS_EVERY_S * 4, int(duration), every) if t not in GPS_PROBE_S]
        for k, t in enumerate(times, 1):
            texts.append((t, _frame_text(stream, t)))
            if k % 50 == 0:
                print(f'  reading the overlay: {k} of {len(times)} frames', flush=True)
    except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        print(f'  GPS overlay not read: {e}', flush=True)
        return None, [], []
    texts.sort()
    clock = [(t, c) for t, x in texts if (c := parse_clock(x)) is not None]
    stamp = [(t, c) for t, x in texts if (c := parse_stamp(x)) is not None]
    track = [(t, *xy) for t, x in texts if (xy := parse_gps(x))]
    # a misread digit puts a point far off: keep the points a car could have driven to from the point most others
    # agree with (not the first one: JyHYb3g7wxs began with 29.26695 for 49.26695)
    reach = lambda a, b: _km(a[1], a[2], b[1], b[2]) <= 1 + abs(b[0] - a[0]) / 3600 * MAX_KMH  # noqa: E731
    anchor = max(track, key=lambda a: sum(reach(a, b) for b in track), default=None)
    kept = [x for x in track if reach(anchor, x)] if anchor else []
    return (kept if len(kept) >= 2 else None), clock, stamp


def _stream(video_id):
    """(url, HTTP headers) of the video's 720p stream: YouTube refuses it (403) without the headers yt-dlp asked
    for it with."""
    m = json.loads(_yt_dlp('-j', '-f', 'bv*[height<=720][ext=mp4]/bv*[height<=720]', _url(video_id)))
    return m['url'], ''.join(f'{k}: {v}\r\n' for k, v in (m.get('http_headers') or {}).items())


# a caption on an opening montage that says what it is (hv7ZXPUOeZ0: "Video Highlights" over 0-22 s; OCR reads it
# rough, "Higheg ht", so a part of the word does)
# ... or "Scenes from the Drive" over MilesofAmbience's openings (TzRtc0j1sz8 until 0:30)
MONTAGE_LABEL = re.compile(r'high\w*ght|highl|in this video|coming up|preview|teaser|trailer|recap|best moments|'
                           r'scenes? from|sneak peek|\bin ?s?th?is\b|this ?vide', re.I)  # OCR's "In tis", "insthis"


def montage_label(video_id, end, frames=4):
    """Whether OCR reads a highlights caption in the montage that montage_end puts before `end` seconds."""
    try:
        stream = _stream(video_id)
        return any(MONTAGE_LABEL.search(_frame_text(stream, float(t)))
                   for t in np.linspace(1.5, max(end - 1.5, 2), frames))
    except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        print(f'  montage caption not read: {e}', flush=True)
        return False


CAPTION_STEP_S, CAPTION_MAX_S = 5, 240  # the opening read every this many seconds, at most this far in
CAPTION_FIRST_S = 30  # no caption read by then: none (PF094oAMKvU's misses its first frames, read at 0:33)
# the corners where a caption sits, small (_DpauJdKxV0: "In this video..." in italics at the bottom left, which a
# read of the whole frame misses over a busy street): each read alone, enlarged, its white letters black on white
CAPTION_CORNERS = ('iw*0.25:ih*0.12:0:ih*0.86', 'iw*0.25:ih*0.12:iw*0.75:ih*0.86', 'iw*0.25:ih*0.12:0:ih*0.02',
                   'iw*0.25:ih*0.12:iw*0.75:ih*0.02')


def _caption_text(stream, t):
    """_frame_text of the frame at t plus what tesseract reads in each of CAPTION_CORNERS."""
    url, headers = stream
    png = subprocess.run(['ffmpeg', '-v', 'error', '-headers', headers, '-ss', str(t), '-i', url, '-frames:v', '1',
                          '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True, timeout=120).stdout
    if not png:
        return ''
    text = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '11'], input=png, capture_output=True,
                          timeout=120).stdout.decode(errors='ignore')
    for crop in CAPTION_CORNERS:
        img = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-vf',
                              f"crop={crop},scale=iw*4:-2,format=gray,lut=y='if(gt(val\\,200)\\,0\\,255)'",
                              '-f', 'image2pipe', '-vcodec', 'png', '-'], input=png, capture_output=True,
                             timeout=60).stdout
        if img:
            text += '\n' + subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '7'], input=img,
                                          capture_output=True, timeout=60).stdout.decode(errors='ignore')
    return text


# EasyOCR, in an environment of its own (its packages clash with this one's): text over video, which tesseract
# reads badly ("CAPITAL OF CALIFORNIA / Sacramento" in QcYf2VZqS-Y: tesseract nothing, EasyOCR both at 0:26)
OCR_PYTHON = next((x for x in (os.path.expanduser('~/ocr-venv/bin/python'),
                               os.path.expanduser('~/micromamba/envs/ocr/bin/python')) if os.path.exists(x)), None)
OCR_CODE = """
import json, sys, easyocr, torch
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
for path in sys.argv[1:]:
    print(json.dumps([text for _, text, conf in reader.readtext(path) if conf >= 0.4]), flush=True)
"""


def opening_texts(video_id, start=1, upto=61, step=None):
    """{second: [texts]} EasyOCR reads in the frames of the video's 720p stream every `step` (CAPTION_STEP_S)
    seconds from `start` to `upto`; None without the OCR environment or the stream."""
    if not OCR_PYTHON:
        return None
    step = step or CAPTION_STEP_S
    work_dir = tempfile.mkdtemp(prefix='crowd_ocr_')
    try:
        url, headers = _stream(video_id)
        shots = []
        for t in np.arange(start, upto, step):
            shot = os.path.join(work_dir, f'{t:.0f}.png')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-headers', headers, '-ss', f'{t:.1f}', '-i', url,
                            '-frames:v', '1', shot], capture_output=True, timeout=120)
            if os.path.exists(shot):
                shots.append((float(t), shot))
        if not shots:
            return None
        r = subprocess.run([OCR_PYTHON, '-c', OCR_CODE, *(x for _, x in shots)], capture_output=True, text=True,
                           timeout=1800)
        lines = [json.loads(x) for x in r.stdout.splitlines() if x.startswith('[')]
        if r.returncode or len(lines) != len(shots):
            print(f'  EasyOCR failed: {r.stderr.strip()[-300:]}', flush=True)
            return None
        return {t: texts for (t, _), texts in zip(shots, lines)}
    except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        print(f'  opening not read: {e}', flush=True)
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def screen_locality(texts, df):
    """([locality, state, country], second) of the mapping locality a label in the opening names most often
    (QcYf2VZqS-Y: "Sacramento" at 0:26; _DpauJdKxV0: "LOS ANGELES 4K"), else None."""
    seen = {}
    for t in sorted(texts or {}):
        if (loc := _title_locality(' '.join(texts[t]), df)):
            seen.setdefault(tuple(loc), []).append(t)
    if not seen:
        return None
    loc, times = max(seen.items(), key=lambda kv: len(kv[1]))
    return list(loc), times[0]


def caption_end(video_id, sig, texts=None):
    """Seconds where an opening montage captioned as one ("In this video", "Highlights", "Scenes from the Drive":
    MONTAGE_LABEL) ends, else 0: OCR on a frame every CAPTION_STEP_S from 3 s while the caption shows, then the
    first strong jump (scene score 0.1) after its last frame (_caption_text; three misses in a row end it).
    PF094oAMKvU joins its highlights with dissolves, so
    montage_end finds no cuts, but "In this video" shows until the drive starts at 1:30.5 (user, 2026-10-13). No
    caption in the first CAPTION_FIRST_S: no montage (six reads per video)."""
    if texts:  # EasyOCR's reads (opening_texts), read on while the caption still shows at their end
        hits = [t for t in sorted(texts) if MONTAGE_LABEL.search(' '.join(texts[t]))]
        if not hits or hits[0] > CAPTION_FIRST_S:
            return 0
        last = hits[0]
        for t in hits[1:]:
            if t - last > 3 * CAPTION_STEP_S:
                break
            last = t
        if last >= max(texts) - CAPTION_STEP_S and (more := opening_texts(video_id, max(texts) + CAPTION_STEP_S,
                                                                          CAPTION_MAX_S)):
            for t in sorted(more):
                if MONTAGE_LABEL.search(' '.join(more[t])):
                    last = t
                elif t - last > 3 * CAPTION_STEP_S:
                    break
        ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
        jumps = [t for t, score, _ in ev if last < t <= last + CAPTION_STEP_S + 5 and score >= 0.1]
        return int(np.ceil(jumps[0])) if jumps else int(last + CAPTION_STEP_S / 2)
    try:
        stream = _stream(video_id)
        # OCR reads the caption on some frames only (PF094oAMKvU: "In tis wu ye" at 0:03, nothing at 0:13,
        # "In this video..." at 0:33 and 1:25): it may miss two frames in a row
        last, misses = None, 0
        for t in np.arange(1, CAPTION_MAX_S, CAPTION_STEP_S):
            if MONTAGE_LABEL.search(_caption_text(stream, float(t))):
                last, misses = float(t), 0
                continue
            misses += 1
            if (last is None and t >= CAPTION_FIRST_S) or (last is not None and misses >= 3):
                break
    except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        print(f'  opening caption not read: {e}', flush=True)
        return 0
    if last is None:
        return 0
    ev = np.asarray(sig.get('skips', np.zeros((0, 3)))).reshape(-1, 3)
    jumps = [t for t, score, _ in ev if last < t <= last + CAPTION_STEP_S + 5 and score >= 0.1]
    return int(np.ceil(jumps[0])) if jumps else int(last + CAPTION_STEP_S / 2)


CLOCK_JUMP_S = 6  # the camera's clock jumping this far ahead of the video for good: footage was cut out


def overlay_speedup(sig):
    """Why the overlay says the video is sped up, else None: its time stamp runs faster than the video (at least 3
    readings 20 s or more apart at a plausible 0.5-10x, the middle rate 1.4 or more and 70% of them over 1.3), or
    its GPS positions imply over 140 km/h in the middle. A misread clock or a cut gives wild pairs (1,437x, then 1.0
    again), never a steady rate; of 558 videos with a clock all run at 1.0, of 656 with GPS the fastest is a highway
    at 128."""
    clock = np.asarray(sig.get('clock', np.zeros((0, 2)))).reshape(-1, 2)
    rates = [((c2 - c1) % 86400) / (t2 - t1) for (t1, c1), (t2, c2) in zip(clock, clock[1:]) if t2 - t1 >= 20]
    sane = [r for r in rates if 0.5 <= r <= 10]  # 245x or 0.01x is a misread digit, not a speed-up
    if (len(sane) >= 3 and len(sane) >= len(rates) / 2 and np.median(sane) >= 1.4
            and np.mean(np.array(sane) > 1.3) >= 0.7):
        return f'sped up: the time stamp in the picture runs {np.median(sane):.1f}x as fast as the video'
    gps = np.asarray(sig.get('gps', np.zeros((0, 3)))).reshape(-1, 3)
    kmh = [_km(a[1], a[2], b[1], b[2]) / ((b[0] - a[0]) / 3600) for a, b in zip(gps, gps[1:]) if b[0] - a[0] >= 20]
    if len(kmh) >= 3 and np.median(kmh) > 140:
        return f'sped up: the GPS overlay implies {np.median(kmh):.0f} km/h'
    return None


CROSS_WAIT_KMH = 50  # ... and the wait's cars at a median this fast (approved waits: 19-73)
CROSS_FAST_KMH = 90  # a car crossing in front of a waiting car this fast: the wait is sped up (normal: 54 at most)


def cross_speedup(sig):
    """Why the cross traffic at a wait says the video is sped up, else None: two cars or more within half a minute
    crossing at CROSS_FAST_KMH or faster (one could be a tracking slip), in a wait whose cars cross at a median of
    CROSS_WAIT_KMH or more: a car seen at an angle looks short and so fast (1C9l-GTFse0: 103 and 90 km/h among
    cars at a median of 25, a cyclist at a natural pace; of 8 videos the first rule left out, 4 were like that,
    the others crossed at medians of 72-85)."""
    cross = np.asarray(sig.get('cross', np.zeros((0, 3)))).reshape(-1, 3)
    fast = np.sort(cross[(cross[:, 2] == 2) & (cross[:, 1] >= CROSS_FAST_KMH), 0])
    for i in range(len(fast) - 1):
        wait = cross[(cross[:, 0] >= fast[i] - 30) & (cross[:, 0] <= fast[i + 1] + 30) & (cross[:, 2] == 2), 1]
        if fast[i + 1] - fast[i] <= 30 and np.median(wait) >= CROSS_WAIT_KMH:
            kmh = cross[(cross[:, 0] >= fast[i]) & (cross[:, 0] <= fast[i + 1]) & (cross[:, 2] == 2), 1].max()
            at = int(fast[i])
            return (f'sped up: cars cross in front at up to {kmh:.0f} km/h while the car waits at '
                    f'{at // 60}:{at % 60:02d}'
                    f' (normal waits: 54 km/h at most)')
    return None


def clock_jump(sig):
    """Where the time stamp in the picture jumps (edits that cut footage out), as "m:ss-m:ss, … and m:ss-m:ss",
    else None: the clock runs ahead of the clock so far by more than CLOCK_JUMP_S, the next two readings agree
    and it never falls back. A cut only ever moves the clock forward for good (4FkZW-vMSE8: 16 s at 1:33, 8 s at
    5:15); a misread digit comes back (vancouverdrive: -10 s, then +10 s, in 24 approved videos), and some
    cameras' readings wobble by up to 6 s. A single reading further from both neighbours than they are from each
    other is a misread and left out first (user, 2026-10-13: B0XUv86Y2Uc's crossfade at 2:42 skips a minute, and
    one reading 4 min behind at 13:00 hid that jump and two more), and every jump is given, not only the first.
    On 413 approved videos with a clock: 8 with a jump (5 before), on 31 rejected 19 (17)."""
    clock = np.asarray(sig.get('clock', np.zeros((0, 2)))).reshape(-1, 2)
    off = [(t, (c - t) % 86400) for t, c in clock]
    ahead = lambda a, b: (b - a + 43200) % 86400 - 43200  # noqa: E731 (past midnight)
    gap = lambda i, j: abs(ahead(off[i][1], off[j][1]))  # noqa: E731
    misread = {k for k in range(1, len(off) - 1) if min(gap(k, k - 1), gap(k, k + 1)) > 3
               and gap(k - 1, k + 1) < min(gap(k, k - 1), gap(k, k + 1))}
    if len(off) > 1 and ahead(off[-2][1], off[-1][1]) < -3:  # the last reading behind: the clock never falls back
        misread.add(len(off) - 1)
    off = [x for k, x in enumerate(off) if k not in misread]
    jumps, start = [], 0
    for i in range(len(off) - 2):  # the next two readings agree (one more near the end: lRagw9VbrO0 at 138:02)
        (t0, _), (t1, b), (_, c) = off[i:i + 3]
        a = off[start][1] + np.median([ahead(off[start][1], x) for _, x in off[start:i + 1]])  # the clock so far
        if (ahead(a, b) > CLOCK_JUMP_S and abs(ahead(b, c)) <= 3
                and all(ahead(a, x) > 3 for _, x in off[i + 1:])):
            jumps.append(f'{int(t0) // 60}:{int(t0) % 60:02d}-{int(t1) // 60}:{int(t1) % 60:02d}')
            start = i + 1
    return ', '.join(jumps[:-1]) + ' and ' + jumps[-1] if len(jumps) > 1 else jumps[0] if jumps else None


def clock_speedups(sig, rate=1.3, steps=3):
    """[(start s, end s, rate)] where the time stamp in the picture runs `rate` to 20 times faster than the video
    for `steps` readings in a row: sped-up footage (user, 2026-10-13: the clock as a source, LoxLj1kTIXo). Each step
    by itself: a jump (a cut) is one step, not several (kddS8sGShKA: +174 s at 107:03, at video speed around it).
    In 356 reviewed approved segments with a clock none, in 110 rejected 2 (-PfvI-MAr1c 1.4x, V24DyViW-bs 3.8x)."""
    c = np.asarray(sig.get('clock', np.zeros((0, 2)))).reshape(-1, 2)
    if len(c) < steps + 1:
        return []
    dt = np.maximum(np.diff(c[:, 0]), 1e-6)
    dc = (np.diff(c[:, 1]) + 43200) % 86400 - 43200
    fast = (dc / dt >= rate) & (dc / dt <= 20)
    return [(float(c[a, 0]), float(c[b, 0]), float(np.median(dc[a:b] / dt[a:b])))
            for a, b in _runs(fast) if b - a >= steps]


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
    if all(loc is None for _, _, loc in stretches):  # 1ei5X-4UvIo: country roads by Port Townsend, WA
        p.pop('entries')
        p['exclude'] = 'rural driving: the GPS overlay passes no town'
        return True
    entries = split_by_stretches(e['segments'], stretches)
    if not entries:
        return False
    for x in entries:  # ground truth: no later guess may move it (vgAi62wCO6w: Richmond, BC is no namesake)
        x['gps'] = True
    p['entries'] = entries
    p['note'] = (f"{p.get('note') or ''}; localities from the GPS overlay: "
                 + ', '.join(f"{loc[0] if loc else 'rural (left out)'} from {int(a) // 60}:{int(a) % 60:02d}"
                             for a, _, loc in stretches)).lstrip('; ')
    return True


def night_from_title(segments, notes, sig, title):
    """(segments, notes) all at night when the title says night, unless the picture is plainly daylight: a camera's
    exposure brightens a lit city at night (I5I9fOnHrPg, "Night Time Driving In The Bronx": sky 110); 71 of 73 such
    reviewed videos were night throughout. Also for videos proposed again from their saved signals, which skipped
    it (4nHdCtFmTuM, "Nashville After Dark": its lit streets day; user, 2026-10-16)."""
    if NIGHT_TITLE.search(title_text(title)) and not DAY_TO_NIGHT_TITLE.search(title_text(title)) \
            and np.median(sig['sky']) < NIGHT_TITLE_MAX_SKY and any(not x['night'] for x in segments):
        return [dict(x, night=1) for x in segments], notes + ['night from the title']
    return segments, notes


def analyse_video(video_id, out_dir, meta=None):
    """Download a low resolution copy, propose segments, write a contact sheet; the copy is deleted."""
    meta = meta or fetch_metadata(video_id)
    work_dir = tempfile.mkdtemp(prefix='crowd_')
    path = download_low_res(video_id, work_dir)
    try:
        sig = signals(path)
        sig['skips'] = skip_events(path)
        sig['skip_norm'] = np.array(skip_norm(path, sig), float).reshape(-1, 2)
        sig['fades'] = fade_events(path)
        sig['saturation'], sig['lamps'] = dusk_colours(path)
        urban = urban_objects(path)
        if urban is not None:
            sig['urban'], sig['street'] = urban[:, 0], urban[:, 1]
            if urban.shape[1] > 2:  # where road users sit (0 top, 1 bottom; NaN with none): see camera_tilt
                sig['object_y'] = np.where(urban[:, 2] >= 0, urban[:, 2] / 1000, np.nan)
        sig['garage'] = np.array([(a, b) for a, b, _ in garage_edges(path, sig, work_dir)], float).reshape(-1, 2)
        sig['parked'] = np.array(parked_edges(path, sig, work_dir), float).reshape(-1, 2)
        sig['lots'] = np.array(parking_lots(path, sig, work_dir), float).reshape(-1, 2)
        sig['kerb'] = np.array(kerb_stops(path, sig, work_dir), float).reshape(-1, 2)
        sig['cross'] = cross_traffic(path, sig)
        sig['ghost'] = ghost_vanishes(path)
        sig['intro_end'] = np.array(clip_intro(path, meta.get('duration'), work_dir) or 0, float)
        sig['outro_start'] = np.array(clip_outro(path, meta.get('duration'), work_dir) or 0, float)
        sig['chapter_cuts'] = np.array(chapter_cuts(meta), float).reshape(-1, 2)
        sig['town_scene'] = np.array(town_scene_track(path, meta.get('duration'), work_dir), float).reshape(-1, 2)
        sig['night_scene'] = np.array(night_scene_track(path, sig, work_dir), float).reshape(-1, 2)
        sig['night_clip'] = np.array(night_clip_track(path, meta.get('duration'), work_dir), float).reshape(-1, 2)
        lit = np.array(light_spots_track(path, meta.get('duration')), float).reshape(-1, 3)
        sig['spots'], sig['lamp_spots'] = lit[:, :2], lit[:, [0, 2]]
        sig['aerial'] = np.array(aerial_track(path, meta.get('duration'), work_dir), float).reshape(-1, 2)
        driver = driver_clip_track(path, sig.get('face', np.zeros(0)), work_dir)
        sig['driver_clip'] = np.array(driver, float).reshape(-1, 2)
        track = step_bob_track(path, meta.get('duration'))  # the whole video's measure is its middle
        sig['bob_track'] = np.array(track, float).reshape(-1, 2)
        sig['step_bob'] = np.array(np.median([x for _, x in track]) if len(track) >= 3 else 0, float)
        end = montage_end(sig)
        texts = opening_texts(video_id)  # read once: highlight captions and town labels
        sig['screen_text'] = np.array(json.dumps({f'{t:.0f}': x for t, x in (texts or {}).items()}))
        sig['labelled_montage'] = np.array(caption_end(video_id, sig, texts) or (
            end if end is not None and montage_label(video_id, end) else 0), float)
        gps, clock, stamp = gps_track(video_id, meta['duration'])
        if gps:
            sig['gps'] = np.array(gps, float)
            print(f'  GPS overlay: {len(gps)} positions', flush=True)
        if clock:
            sig['clock'] = np.array(clock, float)
        if stamp:
            sig['stamp'] = np.array(stamp, float)
        contact_sheet(path, meta['duration'] or len(sig['motion']) / FPS,
                      os.path.join(out_dir, f'{video_id}.jpg'))
    finally:
        os.remove(path)
        os.rmdir(work_dir)
    segments, notes = night_from_title(*propose(sig), sig, meta.get('title'))
    np.savez_compressed(os.path.join(out_dir, f'{video_id}_signals.npz'), **sig)
    return {'meta': meta, 'segments': segments, 'notes': notes}


MIN_UPLOAD_S = 300  # the paper requires source uploads of at least five minutes
EXCLUDE_TITLE = re.compile(r'walking tour|walk tour|city walk|time[- ]?lapse|hyperlapse|compilation'
                           r'|sped[- ]?up|speed(?:ed)?[- ]?up|fast[- ]?forward|\b\d(?:\.\d)?x speed\b|\bx\d speed\b'
                           r'|\bcrash|\baccident|multi[- ]?angle', re.I)  # multi-angle: 5k8awa6rpmA
# highway driving and road trips ("highways 160/191" too: nor-caldashcams); "via I-5" only passes along it on
# an otherwise urban drive
HIGHWAY_TITLE = re.compile(r'(?i:road ?trip|\bhighways?\b|\bfreeways?\b|\bhwy\b|\bmotorways?\b|\bautobahn\b)'
                           r'|(?<![Vv]ia )(?:\b[Ii]nterstate[- ]?|\bI[- ]?|\bi[- ])\d{1,3}\b'
                           r'|\bU\.?S\.?(?:\s?Route|\s?Rte\.?)?[- ]?\d{1,3}\b')  # US Route 9 (KD6ylENdMzA), US-101
# walks and rides on boats are no vehicle type the dataset takes; a title that also says drive keeps the video.
# "Inside" is a walk through a building (8xybXtQ0LgA, "Inside Planet Hollywood Shop": none of 23 such titles kept)
# and "Waking Tour", "Waling Tour" are typos of a walk (1pNtxLv45JI, AW7T4eSiVRk)
NOT_DRIVING_TITLE = re.compile(r'\b(?:walk(?:s|ing|around|about)?|wa(?:king|ling) tour|walk-?through|inside'
                               r'|stroll(?:ing)?|hik(?:e|ing)|on foot'
                               r'|(?:boat|ferry) ride'
                               # "Walkaround" (n_fmYiLgcaI; user, 2026-10-17: none of 96 walk-around titles approved)
                               # boats (mcaow4mKo54: "Cruising on a Yatch in Dubai"; user, 2026-10-16); not ferries
                               # (Little Ferry, Dobbs Ferry are towns, and a drive onto a BC ferry was approved) nor a
                               # bare "boat" ("Studley boat park")
                               r'|yachts?|yatch|sail(?:ing|boat)|catamaran|speedboat|kayak\w*|canoe\w*|gondolas?'
                               r'|water taxi|cruise ship|(?:on|from) (?:a|the) boat|boat (?:trip|tour)'
                               # a passenger on a bus, high up and often sideways
                               # (qEhvYWiNsX0: "Sight Seeing In Toronto
                               # ... On A Bus"; of 9 such titles reviewed or left out, none approved; user, 2026-10-13)
                               r'|sight ?-?seeing|tour bus|double[ -]?decker|hop[ -]on|open[ -]top'
                               r'|on (?:a|the) (?:city )?bus'
                               r'|bus (?:tour|ride)'
                               # trains and the like (VL7X5yOpuk0: "Riding Philadelphia's Market-Frankford Line";
                               # 2ZujrrW_EzY: "The Slow Train Through Rural Japan"; no approved title says train);
                               # "Metro" alone is mostly a metro area ("Omaha Metro Tour")
                               r'|subway|underground train|tram|streetcar|trolley|light rail|monorail|cable car'
                               r'|funicular|(?:train|cab|railway|rail) (?:ride|view|journey)|trains?|railways?'
                               r'|metro (?:ride|journey|line|train|exp)'
                               r'|(?:on|from|by) (?:the |\w+ )?metro(?! area)|riding\b[^|]*\bline'
                               # drones and flights (tam7IqTDBTU: "Downtown Long Beach from above"; of 71 such
                               # titles none approved; user, 2026-10-15)
                               r"|drones?|aerials?|from above|bird'?s?[- ]eye|fly(?:ing)? over|fpv)\b", re.I)
# national parks and forests have no urban driving (yHEDro64z0Y, Zion; none of 28 such titles was approved)
NATURE_TITLE = re.compile(r'\b(?:national|state|provincial|regional) (?:park|forest|monument|seashore|recreation area'
                          r'|preserve)s?\b|\bnational parks?\b|\bwilderness\b|\bscenic byway\b', re.I)
DRIVING_TITLE = re.compile(r'\bdriv(?:e|es|ing)\b|\bdashcam\b', re.I)


def usual_locality(plan, share=0.5, min_videos=5):
    """[locality, state, country] of more than half the channel's reviewed videos (at least min_videos of them),
    else None: CuseVegas drives mostly in Las Vegas, so a drive to "T-Mobile Arena" is there (gkGtNPzzeJ8)."""
    counts = Counter()
    for p in plan.values():
        locs = {tuple(e['locality']) for e in p.get('entries', []) if e.get('locality')
                and any(s.get('applied') or s.get('decision') == 'approve' for s in e['segments'])}
        counts.update(locs)
    total = sum(counts.values())
    if total < min_videos or not counts:
        return None
    loc, n = counts.most_common(1)[0]
    return loc if n > share * total else None


def channel_folder(url):
    """The channel's folder name under _output/proposals: its handle or id, without the path or a query
    (UCjaZ_Lko0XB49cCs-bnlz4A?source_ve_path=... from a link copied off an embed). A channel that already has a
    folder under another name keeps it, whichever way the link names it (user, 2026-10-13: ChillDrive09, done by
    its handle on 2026-10-03, came back from the subscriptions as /channel/UCWoo_I-Q2X9HmTwmkMTtGOg and was
    proposed again from scratch)."""
    name = re.sub(r'^.*youtube\.com/(?:channel/|c/|user/)?@?', '', url.strip())
    name = re.sub(r'\W+', '_', re.split(r'[/?#]', name)[0])
    if os.path.isdir(os.path.join(HERE, '_output', 'proposals', name)):
        return name
    cid = re.search(r'/channel/(UC[\w-]+)', url)
    if not cid:
        try:
            with open(os.path.join(HERE, '_output', 'subscriptions', 'channel_ids.json')) as f:
                cid = json.load(f).get(url.strip())  # sheet URL -> channel id (refresh_queue.py)
        except (OSError, ValueError):
            cid = None
    else:
        cid = cid.group(1)
    return _channel_folders().get(cid, name) if cid else name


@functools.cache
def _channel_folders():
    """{channel id: folder name} of the channel folders, from a metadata file of each."""
    root, out = os.path.join(HERE, '_output', 'proposals'), {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for meta in sorted(glob.glob(os.path.join(root, name, '*_meta.json')))[:3]:
            try:
                with open(meta) as f:
                    cid = json.load(f).get('channel_id')
            except (OSError, ValueError):
                continue
            if cid:
                out.setdefault(cid, name)
                break
    return out


def list_channel(url):
    """Video ids of a channel URL as written in the channels sheet (bare channel, /videos or /search?query=)."""
    bare = re.match(r'(https?://(?:www\.)?youtube\.com/(?:@[^/?]+|channel/[^/?]+))/?(?:\?.*)?$', url.strip())
    if bare:
        url = bare.group(1) + '/videos'
    entries = json.loads(_yt_dlp('--flat-playlist', '-J', url.strip())).get('entries') or []
    return [e['id'] for e in entries if e and len(e.get('id') or '') == 11]


YT_API = 'https://www.googleapis.com/youtube/v3/'
NIGHT_TITLE = re.compile(r'\bnight\b|\bnocturn|\bat dusk\b|\bafter dark\b', re.I)
# ... not a drive from day into night (GfIK2UBw51A: "Sunset Time Drive … Golden Hour to Night"; 2026-10-16)
DAY_TO_NIGHT_TITLE = re.compile(r'\bsunset\b|\bgolden hour\b|\binto (?:the )?night\b|\bto night\b'
                                r'|\bday (?:in)?to night\b', re.I)
NIGHT_TITLE_MAX_SKY = 140  # median sky brightness of a video plainly in daylight (night titles reviewed: 7-73)


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


TRANSLATIONS = os.path.join('_output', 'proposals', 'translations.json')  # text -> English, kept per machine
ENGLISH_HINT = re.compile(r'\b(?:the|in|to|of|and|from|with|at|drive|driving|walk|walking|night|tour|city|road)\b',
                          re.I)


@functools.lru_cache(maxsize=None)
def english(text):
    """text in English (Google Translate's free endpoint, no key; user 2026-10-11: BonBonB Phương's Vietnamese
    titles), '' when it already reads as English or the endpoint does not answer. Kept in TRANSLATIONS."""
    text = (text or '').strip()
    if not text or ENGLISH_HINT.search(text):
        return ''
    try:
        with open(TRANSLATIONS) as f:
            known = json.load(f)
    except (OSError, ValueError):
        known = {}
    if text in known:
        return known[text]
    try:
        r = requests.get('https://translate.googleapis.com/translate_a/single', timeout=15,
                         params={'client': 'gtx', 'sl': 'auto', 'tl': 'en', 'dt': 't', 'q': text[:1000]})
        out = ''.join(part[0] for part in r.json()[0] if part and part[0]) if r.ok else ''
    except (requests.RequestException, ValueError, IndexError, TypeError):
        return ''
    known[text] = out = '' if out.strip().lower() == text.lower() else out
    with contextlib.suppress(OSError):
        with open(TRANSLATIONS + '.tmp', 'w') as f:
            json.dump(known, f, ensure_ascii=False)
        os.replace(TRANSLATIONS + '.tmp', TRANSLATIONS)
    return out


def title_text(title):
    """A title for the title checks: as written plus its English translation when it is in another language."""
    title = title or ''
    return f'{title} | {translated}' if (translated := english(title)) else title


def exclusion_reason(meta):
    """Why the protocol rules a video out from its metadata alone, or None."""
    if (meta.get('duration') or 0) < MIN_UPLOAD_S:
        return f"{(meta.get('duration') or 0) / 60:.1f} min, under the 5 min minimum"
    if meta.get('live_status') in ('is_live', 'is_upcoming'):
        return 'live or upcoming stream'
    if meta.get('width') and meta.get('height') and meta['height'] > meta['width']:
        return 'portrait (phone) video'
    title = title_text(meta.get('title'))
    hit = EXCLUDE_TITLE.search(title)
    if hit:
        return f"title mentions '{hit.group(0)}'"
    hit = NOT_DRIVING_TITLE.search(title)
    if hit and not DRIVING_TITLE.search(title):
        return f"not a drive (title mentions '{hit.group(0)}')"
    hit = NATURE_TITLE.search(title)
    if hit:
        return f"a drive through a park or nature area, no urban driving (title mentions '{hit.group(0)}')"
    return None


US_NAMES = {'united states', 'united states of america', 'usa', 'us'}


def us_title(title):
    """A title naming a place in the US: the country, a state (not Georgia, also a country: "Tbilisi, Georgia") or
    a state's code after a comma ("Miami, FL"). Such videos of channels elsewhere wait for the US step (user,
    2026-10-16)."""
    t = english_states(title or '')
    states = '|'.join(re.escape(x) for x in add_video.US_STATE_CODES if x != 'georgia')
    codes = '|'.join(sorted(set(add_video.US_STATE_CODES.values())))
    return bool('United States' in title_countries(title) or re.search(r'\bU\.?S\.?A\b', t)
                or re.search(rf'\b(?:{states})\b', t, flags=re.I) or re.search(rf',\s*(?:{codes})\b', t))


def title_countries(title):
    """The countries of the mapping a title names, not inside a US state's name: "New Jersey" is not Jersey,
    "Atlanta, Georgia" not the country."""
    title = english_states(title)  # "Nueva Jersey" is no Jersey
    states = '|'.join(re.escape(s) for s in add_video.US_STATE_CODES)
    # "Rio Piedra, San Juan, P.R.", "Ponce, PR": Puerto Rico by its postal abbreviation
    pr = ['Puerto Rico'] if re.search(r'\bP\.\s?R\.|,\s*PR\b', title or '') else []
    # a country's name before a US state is a town of that name: "Lebanon, Pennsylvania", "Peru, Indiana"
    title = re.sub(rf"\b[A-Z][\w'’ ]*?\s*,\s*(?:{states}|[A-Z]{{2}})\b", ' ', title or '', flags=re.I)
    # ... and without the comma before a state's code: "Panama City Beach FL" (9F47RLjPHdM; user, 2026-10-13)
    codes = '|'.join(sorted(set(add_video.US_STATE_CODES.values())))
    title = re.sub(rf"\b(?:[A-Z][\w'’.]*\s+){{1,4}}(?:{codes})\b", ' ', title)
    title = re.sub(rf'\b(?:{states})\b', ' ', title, flags=re.I)
    found = [c for c in COUNTRIES if re.search(rf'\b{re.escape(c)}\b', title)]
    # a flag emoji names its country (OoOwyF8-T24: "Warsaw's Most Beautiful Bike Route 4K 🇵🇱" is no Warsaw, NY)
    flags = [chr(ord(a) - 0x1F1E6 + 65) + chr(ord(b) - 0x1F1E6 + 65)
             for a, b in re.findall('([\U0001F1E6-\U0001F1FF])([\U0001F1E6-\U0001F1FF])', title)]
    found += [c for c in COUNTRIES if c not in found and _iso2(c) in flags]
    return found + [c for c in pr if c not in found]


@functools.lru_cache(maxsize=None)
def _iso2(country):
    return common.get_iso2_country_code(common.correct_country(country))


# a title saying how the camera travels beats the channel's usual vehicle (OoOwyF8-T24: "Bike Route" from a channel
# that mostly drives); "motorbike" is no bike
VEHICLE_TITLE = [(4, re.compile(r'\b(?:bike (?:route|ride|tour|path|lane)s?|cycling|cycle (?:ride|tour)|bicycle'
                                r'|by bike|on (?:a|my) bike|e-?bike)\b', re.I)),
                 (3, re.compile(r'\b(?:motorcycle|motorbike|motor bike|moped)s?\b', re.I))]


def title_vehicle(title):
    """The VEHICLE_TYPES code a title names (a bicycle, a motorcycle), else None."""
    return next((vt for vt, r in VEHICLE_TITLE if r.search(title or '')), None)


WORD_STATE_CODES = {'IN', 'OR', 'ME', 'OK', 'HI', 'DE', 'CO', 'MA', 'AL', 'ID', 'OH', 'LA'}  # LA: Los Angeles


def title_states(title):
    """The US states (postal codes) a title names, by name or by code: "Bay Lake FL", "Ponce, PR"; codes that are
    also words or names (IN, OR, ME, PA, ...) only after a comma, and none in an all-capitals title."""
    title = english_states(title or '')
    codes = set(add_video.US_STATE_CODES.values())
    # "Kansas City", "Oklahoma City" name a city, not the state; "Washington DC" is not Washington State
    # "New York" alone is the city (a drive "from Newark to New York" is no Newark, NY): the state only as
    # "New York State" or by its code
    found = {code for name, code in add_video.US_STATE_CODES.items()
             if re.search(rf'\b{re.escape(name)}\b(?! City)(?!,? ?D\.? ?C\b)'
                          + (r'(?= State)' if name == 'new york' else ''), title, re.I)}
    if sum(c.isupper() for c in title) <= 0.6 * sum(c.isalpha() for c in title):
        found |= {m.group(2) for m in re.finditer(r'(,\s*)?\b([A-Z]{2})\b', title)
                  if m.group(2) in codes and (m.group(1) or m.group(2) not in WORD_STATE_CODES)}
    return found


def title_places(title, df, home):
    """place_localities of a title's places, within the US states it names and without the pull of the channel's
    home (hnldNuzdEUU: "Kissimmee … Bay Lake FL" from a New York channel is no Kissimmee near New York); with one
    state named, each place is looked up in it (FDuwYqiyOHc: "Ashland to Waterloo … Nebraska" is no Waterloo, IA)."""
    phrases, states = place_phrases(title), title_states(title)
    if not states:
        return place_localities(phrases, df, home)
    names = {code: name.title() for name, code in add_video.US_STATE_CODES.items() if len(name) > 2}
    if len(states) == 1:
        state = names.get(next(iter(states)))
        phrases = [ph if not state or state.lower() in ph.lower() else f'{ph}, {state}' for ph in phrases
                   if ph.lower() not in add_video.US_STATE_CODES]  # "Nebraska" itself is no town
    else:  # each place in the state written after it (1c1bqA68254: "Glenwood, Iowa to Gretna, Nebraska", no
        # Glenwood, IL)
        state_of = lambda ph: names.get(add_video.US_STATE_CODES.get(ph.lower()))  # noqa: E731
        phrases = [f'{ph}, {state_of(nxt)}' if state_of(nxt) else ph
                   for ph, nxt in zip(phrases, phrases[1:] + ['']) if not state_of(ph)]
    found = place_localities(phrases, df[(df['country'] != 'United States') | df['state'].isin(states)], None)
    return found or misspelt_places(phrases, df, states)


def misspelt_places(phrases, df, states):
    """[(locality, phrase)] of the phrases that name no place as written but are a misspelling of one mapping
    locality of the one US state the title names: same first letter, spelled at least MISSPELT_RATIO alike
    (IPgD51yxkWM: "Driving thru Bethlahem,Pennsylvania" is Bethlehem, PA; user, 2026-10-15)."""
    if len(states) != 1:
        return []
    rows = df[(df['country'] == 'United States') & df['state'].isin(states)]
    names = list(dict.fromkeys(rows['locality']))
    out = []
    for ph in phrases:
        town = ph.split(',')[0].strip()
        if not town or town.lower() in add_video.US_STATE_CODES:
            continue
        close = [n for n in difflib.get_close_matches(town, names, n=2, cutoff=MISSPELT_RATIO)
                 if n[:1].lower() == town[:1].lower()]
        if len(close) == 1 or (close and difflib.SequenceMatcher(None, town, close[0]).ratio() == 1.0):
            r = rows[rows['locality'] == close[0]].iloc[0]
            out.append(([r['locality'], r['state'], r['country']], ph))
    return out


MISSPELT_RATIO = 0.85


def guess_locality(title, country, df):
    """[locality, state, country] when the title names exactly one mapping locality of that country, else None.
    A drive "from A to B in C" with C a locality is in C: neighbourhoods of a big city ("from Hollywood to Venice
    in Los Angeles") are often namesakes of other localities."""
    df = df[df['country'] == country]
    area = DRIVE_AREA.search(title or '')
    if area and (loc := _title_locality(area.group(1), df)):
        return loc
    # a big city heading the title is where the drive is, its route a way through its neighbourhoods (EHZ3z7SvP3I:
    # "Chicago Dashcam Drive: Marquette Park to Midway, ..." is no Midway, Ohio)
    head = re.split(r'\s*(?::|\||\s[-–—]\s)\s*', title or '', maxsplit=1)
    # not a drive from one town to another at the head ("Lilongwe to Mponela - M1 …"): its route says where
    if len(head) == 2 and not any(r.search(head[0]) for r in FROM_TO) and (loc := _title_locality(head[0], df)) \
            and (at := add_video.get_existing_locality_row(
            df, *loc)[1]) is not None and float(pd.to_numeric(at.get('population_locality'), errors='coerce')
                                                or 0) >= TITLE_CITY_POP:
        return loc
    # "by Red Hook in Brooklyn, New York": only what comes after "in" says where (the Red Hook in the mapping is
    # upstate); a place there that is no mapping locality (Brooklyn) is looked up later
    inside = IN_PLACE.search(title or '')
    loc = _title_locality(title[inside.start(1):] if inside else title, df)
    # ... in a state the whole title names: "Broadway Virginia Scenic Drive | Hidden Gem in the Shenandoah Valley" is
    # no Shenandoah, PA (30bJBAaRl8M)
    if inside and loc and loc[2] == 'United States' and (states := title_states(title)) and loc[1] not in states:
        return None
    return loc


BIG_CITY_POP = 500_000  # a city this big named in a title is meant even outside the channel's country


def big_city_locality(title, df):
    """[locality, state, country] of the one big city (BIG_CITY_POP) a title names in any country, else None: a
    channel listed in the United States drives in Seoul too (wPKj2rSUcic, "Seoul Gangnam Skyline Sunset")."""
    big = df[pd.to_numeric(df['population_locality'], errors='coerce') >= BIG_CITY_POP]
    return _title_locality(title, big)


def title_city(title, loc, df):
    """loc, or the big city the title names when loc is none or a far smaller namesake in another state than the
    title names: a channel listed in the United States drives abroad too ("Seoul 7AM Skyline Sunrise" is no Sunrise,
    FL: ElXqMWS9_DQ; "Paris Rainy Morning Drive" no Paris, TX), "Moscow Idaho" stays Moscow, ID."""
    big = big_city_locality(title, df)
    # only a city the title starts with: inside a title it is mostly a neighbourhood or a namesake ("Van Nest in the
    # Bronx" is no Van, Türkiye; "Driving from Moscow to Moosic, Pennsylvania" no Moscow, Russia)
    if not big or big == loc or not re.match(rf'\W*{re.escape(big[0])}\b', title or '', re.I):
        return loc
    if loc is None:
        return big
    if loc[2] == 'United States' and loc[1] in title_states(title):
        return loc
    # a longer name that starts with the city's ("Trujillo Alto,Puerto Rico") or the namesake with its state after a
    # comma ("Birmingham, Al.") is the local place
    with_state = re.search(rf'{re.escape(big[0])}\s*,\s*[A-Za-z]{{2}}\b', title, re.I)
    if _name_key(loc[0]).startswith(_name_key(big[0])) and (_name_key(loc[0]) != _name_key(big[0]) or with_state):
        return loc

    def pop(place):
        row = add_video.get_existing_locality_row(df, *place)[1]
        return float(pd.to_numeric(row['population_locality'], errors='coerce') or 0) if row is not None else 0
    return big if pop(big) >= NAMESAKE_RATIO * pop(loc) else loc


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


# chapters that are no part of the drive by their name (JVvw7XtJzf0: "Highlights (15 Min Preview)" 0:39-15:36, then
# "Drive Begins"); none of the approved footage lies in one, 897 rejected seconds do. "Intro" and "Outro" chapters
# are often driving (MqE-w-gEbM8: a 14-minute "Intro"), so they stay
NOT_DRIVE_CHAPTER = re.compile(r'\b(?:highlights?|preview|teaser|trailer|recap|best of)\b', re.I)
# an opening chapter of scenes from the drive ahead (MilesofAmbience's "Scenes from the Drive": 10 of 11 at 0:00 last
# 13-43 s with 12 approved seconds inside; UCWoo's "Sneak Peek" 0-152 s, none approved); a longer one runs on into
# the drive (TzRtc0j1sz8: 0-13:57, the scenes only until 0:30), so only one of PREVIEW_CHAPTER_S or less is cut.
# UCWoo's "In This Video" / "In Today's Video" / "Coming Up" (user, 2026-10-13: XJeW6Nna3bA): 42 openings of
# 101-418 s, the caption on every frame of IiP1qRK5-4Y's until the drive starts after it (3:10)
PREVIEW_CHAPTER = re.compile(r"\b(?:scenes? from the (?:drive|ride|trip)|sneak peek|in (?:today'?s|this) "
                             r"(?:video|drive|episode)|coming up)\b", re.I)
PREVIEW_CHAPTER_S = 420
# a walk in a chapter's name: "Walk Begins" walks on until a chapter says drive again (5-9vyLU_tDo, AbaoAmbience's
# night walks: 12 videos, none approved); "Walking through the Farmers Market" is that chapter (s_iGsy8C4O8)
WALK_CHAPTER = re.compile(r'\bwalk(?:ing)?\b(?! of fame)', re.I)
WALK_START = re.compile(r'\b(?:walk (?:begins|starts|start)|main walk|start(?:ing)? (?:the |our )?walk)\b', re.I)


def chapter_cuts(meta):
    """[(start s, end s)] of the chapters NOT_DRIVE_CHAPTER names, a short opening PREVIEW_CHAPTER, and of walks by
    WALK_CHAPTER."""
    chapters = video_chapters(meta)
    ends = [t for t, _ in chapters[1:]] + [meta.get('duration') or 0]
    cuts = [(t, end) for (t, name), end in zip(chapters, ends) if NOT_DRIVE_CHAPTER.search(name) and end > t]
    cuts += [(t, end) for (t, name), end in zip(chapters, ends)
             if t == 0 and PREVIEW_CHAPTER.search(name) and 0 < end <= PREVIEW_CHAPTER_S]
    for i, ((t, name), end) in enumerate(zip(chapters, ends)):
        if WALK_START.search(name):  # until a chapter says drive
            end = next((u for u, n in chapters[i + 1:] if DRIVING_TITLE.search(n)), meta.get('duration') or end)
        if WALK_CHAPTER.search(name) and end > t:
            cuts.append((t, end))
    return cuts


def video_chapters(meta):
    """[(start in s, name)] of a video's chapters: YouTube's own, else the description's timestamped lines
    ("12:30 Santa Monica" or "Santa Monica - 12:30"), when at least two run in increasing order; a line out of
    order is a typo and left out (XJeW6Nna3bA: "01:07 Zuma Beach" between 51:53 and 1:21:53)."""
    chapters = [(int(start or 0), name) for name, start in meta.get('chapters') or [] if name]
    if not chapters:
        for line in (meta.get('description') or '').splitlines():
            if m := CHAPTER_START.match(line):
                h, mins, secs, name = m.groups()
            elif m := CHAPTER_END.match(line):
                name, h, mins, secs = m.groups()
            else:
                continue
            t = int(h or 0) * 3600 + int(mins) * 60 + int(secs)
            if not chapters or t > chapters[-1][0]:
                chapters.append((t, name.strip()))
    starts = [t for t, _ in chapters]
    return chapters if len(chapters) >= 2 and starts == sorted(set(starts)) else []


# a chapter named after a road ("Venice Blvd", "Rodeo Drive") is not in the place the road is named after
ROAD = re.compile(r'\b(?:blvd|boulevard|st|street|ave|avenue|rd|road|dr|drive|fwy|freeway|hwy|highway|expy|'
                  r'expressway|pkwy|parkway|ln|lane|way|terrace|strip|bridge|tunnel|pier|walk of fame|pl|place|'
                  # a city's Broadway is a street: "38:58 Broadway" in Downtown Los Angeles is no Broadway, VA
                  # (UCWoo_I_Q2X9HmTwmkMTtGOg: 12 one-city drives held back as multi-city; 2026-10-16)
                  r'broadway|'
                  # and the like of Japanese expressway chapters (zvCKHw9rKPg: "Tsukamoto Exit" is no Kaizuka)
                  r'junction|exit|interchange|ramp|route|loop|line|'
                  # Japanese for avenue and road: "Wakamiya Oji" is no town of Oji (-VqR0H2f-JY)
                  r'oji|ōji|dori|dōri|kaido|kaidō)\b', re.I)
CHAPTER_PREFIX = re.compile(r'^(?:start(?:ing)? (?:in|at|from)|entering|arriv(?:e|ing|al) (?:in|at)|welcome to|'
                            r'(?:driving|heading|going|cruising) (?:to|into|through|in|around)|'
                            r'leaving|into|in|to|from|downtown)\s+', re.I)


def chapter_localities(chapters, df, near_home=None, near=None, km=150, first=None):
    """[(start, end, locality)] stretches of a video by the localities its chapters name, or [] when none does.
    A chapter naming none (a road, a sight) is still in the locality before it; leading ones in the first named.
    near_home(name) gives a locality not in the mapping yet for a chapter that is just a place name. near=(lat,
    lon): only localities within km of it (Nqej7zmNUzI, a Yokohama drive: its "Kanazawa" is a ward of Yokohama,
    not Kanazawa, Ishikawa). first: the locality of the chapters before the first named one (the title's)."""
    locs = []
    for _, name in chapters:
        if ROAD.search(name):
            locs.append(None)
            continue
        loc = _title_locality(name, df, exact_case=True, word_needs_state=True)
        if loc and near and (at := _locality_anchor(loc, df)) and _km(near[0], near[1], at[3], at[4]) > km:
            loc = None
        if loc and part_of_name(name, loc, df):
            loc = None
        place = CHAPTER_PREFIX.sub('', name.strip(' .!-'))
        # the place written another way: "Driving to Mountaintop" is Mountain Top, PA (user, 2026-10-14: MNpSJ-gIduQ)
        squash = lambda x: re.sub(r'[\W_]+', '', _name_key(str(x)))  # noqa: E731 (spaces and hyphens aside)
        if not loc and len(key := squash(place)) > 3 and \
                len(same := df[[squash(x) == key for x in df['locality']]]) == 1:
            r = same.iloc[0]
            loc = [r['locality'], r['state'] if isinstance(r['state'], str) else None, r['country']]
            if near and (at := _locality_anchor(loc, df)) and _km(near[0], near[1], at[3], at[4]) > km:
                loc = None
        if not loc and near_home and re.fullmatch(r"(?:[A-Z][\w'.-]*\s?){1,4}", place):
            loc = near_home(place.strip())
        locs.append(loc)
    named = [loc for loc in locs if loc]
    if not named:
        return []
    out, current = [], first or named[0]
    for (start, _), loc, nxt in zip(chapters, locs, [t for t, _ in chapters[1:]] + [float('inf')]):
        current = loc or current
        if out and out[-1][2] == current:
            out[-1] = (out[-1][0], nxt, current)
        else:
            out.append((start, nxt, current))
    return out


NAME_LEAD = {'downtown', 'uptown', 'midtown', 'central', 'greater', 'old', 'historic', 'city', 'north', 'south',
             'east', 'west', 'welcome', 'entering', 'leaving', 'driving', 'drive', 'arriving', 'back'}


def part_of_name(name, loc, df):
    """Whether the locality named in a chapter follows another capitalised word, so that it is part of a longer
    name, a street or a neighbourhood (PvQnDdF145Q's batch: "Paseo Del Mar" and "Vista Del Mar" in Los Angeles and
    "Corona Del Mar" in Newport Beach are no Del Mar); "Downtown Pasadena" or "City of Malibu" are the place."""
    row = add_video.get_existing_locality_row(df, *loc)[1]
    names = [loc[0]] + (add_video._parse_flat_list_cell(row['locality_aka']) if row is not None else [])
    for n in names:
        if (m := re.search(rf'(?<!\w){re.escape(str(n))}(?!\w)', name, re.I)):
            before = re.findall(r'[\w\'.-]+', name[:m.start()])
            return bool(before) and before[-1][:1].isupper() and before[-1].lower() not in NAME_LEAD
    return False


# a title driving to a place ("Road Trip on SC 11 to Landrum, SC") or leaving it is there only at its end or start
# (user, 2026-10-13: 7Zx1IvuHiPY was offered as Landrum all through its 58 min, mostly highway)
TOWARDS = re.compile(r"\b(?:to|into|towards?|arriv\w* (?:in|at)|heading (?:to|for))\s+(?:downtown\s+)?$", re.I)
LEAVING = re.compile(r"\b(?:leaving|departing)\s+(?:downtown\s+)?$", re.I)
ROLE_GAP_S = 120  # footage left out between two stretches this long: a drive between towns


def title_role(title, loc):
    """'end' when the title drives to the locality, 'start' when it leaves it, else None."""
    m = re.search(rf'(?<!\w){re.escape(loc[0])}(?!\w)', title or '', re.I)
    if not m:
        return None
    before = title[:m.start()]
    return 'end' if TOWARDS.search(before) else 'start' if LEAVING.search(before) else None


def by_role(segments, loc, role):
    """Locality groups: loc for the segments after the last gap of ROLE_GAP_S (role 'end') or before the first
    ('start'), the rest in a group whose locality the reviewer sets."""
    segs = sorted(segments, key=lambda s: s['start'])
    gaps = [i for i in range(1, len(segs)) if segs[i]['start'] - segs[i - 1]['end'] >= ROLE_GAP_S]
    cut = (gaps[-1] if gaps else 0) if role == 'end' else (gaps[0] if gaps else len(segs))
    there, rest = (segs[cut:], segs[:cut]) if role == 'end' else (segs[:cut], segs[cut:])
    return [{'locality': list(loc), 'guessed': True, 'segments': there}] + \
        ([{'locality': None, 'guessed': False, 'segments': rest}] if rest else [])


NAMESAKE_RATIO = 4  # a city this many times bigger than every namesake a title could mean is the one meant


# towns named after a landform: right after a capitalised name they are the end of that name ("Red Rock Canyon"
# is no Canyon, TX: KlILKnTeVag; "Myrtle Beach", "Great Plains"; user, 2026-10-16), unless that word only leads in
# to the place ("Driving Through Summit, NJ")
LANDFORM_TOWNS = {'canyon', 'beach', 'plains', 'summit', 'mesa', 'gap', 'bay'}
LEAD_WORDS = {'in', 'into', 'to', 'from', 'through', 'thru', 'around', 'across', 'near', 'via', 'of', 'and', 'the',
              'downtown', 'drive', 'driving', 'drives', 'visit', 'visiting', 'exploring', 'explore', 'tour', 'night',
              'morning', 'evening', 'sunset', 'sunrise', 'rainy', 'snowy', 'winter', 'summer', 'spring', 'fall',
              'autumn', 'at', 'on', 'by', 'for', 'with'}


def _ends_a_name(title, at):
    """Whether the word before position `at` of the title is a capitalised name word, not a lead-in."""
    m = re.search(r"([A-Za-z][\w'’.]*)\s+$", title[:at])
    return bool(m and m.group(1)[0].isupper() and m.group(1).lower() not in LEAD_WORDS)


def _title_locality(title, df, exact_case=False, word_needs_state=False):
    title = english_states(title)
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
            # and "Jerome Ave" is a street, not Jerome
            # likewise "Mt. Lebanon" is not Lebanon, nor "Fort Lee" Lee or "Port Jefferson" Jefferson
            # and "Hampton Inn" a hotel, not Hampton (4FkZW-vMSE8), "Morris Park" a park or a neighbourhood, not
            # Morris (i5kaJ7ePcpQ), and "Upstate New York" the state (VhIUTVhXa2U)
            pattern = (rf'(?<!\w)(?<![Nn]ew\s)(?<!Mt\.\s)(?<!Mt\s)(?<!Mount\s)(?<!Ft\.\s)(?<!Fort\s)(?<!St\.\s)'
                       rf'(?<!Saint\s)(?<!Port\s)(?<![Uu]pstate\s){re.escape(name)}(?!\w)'
                       rf'(?!\.?\s+(?:{STREETS}|St|Inn|Park)\b)')
            if (word_needs_state and name.lower() in COMMON_WORDS) or name.lower() in DEMONYMS:
                if not isinstance(row['state'], str):
                    continue
                pattern += rf',\s*{re.escape(row["state"])}\b'
            state = row['state'] if isinstance(row['state'], str) else ''
            spans = [m.span() for m in re.finditer(pattern, title or '', flags)
                     if not (word_needs_state and (other := re.match(r',\s*([A-Z]{2})\b', (title or '')[m.end():]))
                             and other.group(1) != state)
                     and not (name.lower() in LANDFORM_TOWNS and _ends_a_name(title or '', m.start()))]
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
             # a state's name or code after a comma is the state, also when the town before it is new: "Johnson City,
             # New York" and "Scarsdale, Westchester, NY" are not New York City
             and not ((title[a:b].lower() in add_video.US_STATE_CODES
                       or title[a:b] in add_video.US_STATE_CODES.values()) and re.search(r'\w\s*,\s*$', title[:a]))}
    # a title naming a US state is in it: Davenport, FL is a namesake of the one in "Davenport, Iowa", and
    # Philadelphia, MS of the one in "Philadelphia, Pennsylvania"
    states = {code for name, code in add_video.US_STATE_CODES.items()
              if re.search(rf'\b{re.escape(name)}\b', title, re.I) or re.search(rf',\s*{code}\b', title)}
    if states:
        found = {loc for loc in found if loc[2] != 'United States' or not loc[1] or loc[1] in states}
    if len(found) > 1 and len({loc[0] for loc in found}) == 1:  # namesakes: a big city beats small towns
        pop = {loc: float(pd.to_numeric(df.loc[(df['locality'] == loc[0]) & (df['country'] == loc[2])
                                               & ((df['state'] == loc[1]) if loc[1] else df['state'].isna()),
                                               'population_locality'], errors='coerce').max() or 0)
               for loc in found}
        big = max(pop, key=pop.get)
        # "Never Expected Cleveland Downtown…" is Cleveland, OH (FbIYNiwpBaw), not Cleveland, TN
        if pop[big] >= TITLE_CITY_POP and all(pop[big] >= NAMESAKE_RATIO * v for k, v in pop.items() if k != big):
            return list(big)
    return list(found.pop()) if len(found) == 1 else None


# nationality words are towns only written with their state: "USA and Canadian Border" is no Canadian, TX
# (uG7ZUl7t48M)
DEMONYMS = {'american', 'canadian', 'mexican', 'russian', 'german', 'french', 'italian', 'chinese', 'japanese',
            'korean', 'indian', 'irish', 'english', 'dutch', 'polish', 'spanish', 'cuban', 'brazilian'}
# words of titles that are no place ("Beautiful Evening Drive", "4K ASMR"); runs of capitalised words without
# them are looked up as places ("Al Khan Beach", "Jebel Jais Mountain")
NOT_PLACE = {'around', 'across', 'through', 'along', 'near', 'inside',
             'drive', 'drives', 'driving', 'beautiful', 'relax', 'relaxing', 'chilled', 'morning', 'afternoon',
             'evening', 'night', 'midnight', 'sunrise', 'sun', 'rise', 'sunset', 'unedited', 'sounds', 'sound',
             'asmr', 'vlog', 'travel', 'video', 'episode', 'part', 'visit', 'visiting', 'tour', 'tours', 'trip',
             'uhd', 'hd', 'fps', 'live', 'stream', 'subscribe', 'thanks', 'watching', 'enjoy', 'please', 'new',
             'route', 'journey', 'heavy', 'windy', 'cloudy', 'rainy', 'foggy', 'nice', 'view', 'best', 'my', 'i',
             'friday', 'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'january', 'february',
             'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
             # travel-vlog titles: "Exploring Worthington, Ohio | Historic Village, Beautiful Downtown"
             'exploring', 'explore', 'discover', 'discovering', 'welcome', 'touring', 'cruising', 'inside',
             'historic', 'charming', 'scenic', 'downtown', 'village', 'communities', 'community', 'one', 'pov',
             # the weather and the like: "Driving In The Rain" is no Washington, DC (3VrSWe479-g)
             'rain', 'raining', 'storm', 'thunderstorm', 'snow', 'snowfall', 'hail', 'flood', 'flooding', 'flash',
             'warning', 'issued', 'fireworks', 'never', 'expected', 'changed', 'border',
             # "Let's" in a description is no San Diego (CGwuQYmG0JY)
             'let', 'lets', "let's", 'let’s', 'join', 'watch', 'enjoy', 'come',
             # "| No Music" ends every title of some channels: "Music, California" was San Diego (0iaHhzw0VhQ)
             'music', 'commentary', 'narration', 'talking'} | DEMONYMS
PLACE_JOINERS = {'al', 'el', 'de', 'del', 'la', 'le', 'of', 'the', 'bin', 'bu', 'van', 'von', 'da', 'do', 'dos'}


# "...from A to B in the Bronx, New York (2026)": the place after the last "in" holds the whole drive, when no
# other name follows it but its state or country
# and in Spanish: "Desde Hato Rey hasta Rio Piedras en San Juan, Puerto Rico" (area_city checks the start is in it)
DRIVE_AREA = re.compile(r"(?:\bfrom\b.+\bto\b|\b(?:[Dd]esde|de)\b.+\b(?:hasta|hacia|a|al)\b"
                        r"|^.+\b(?:hasta|hacia|al?)\b).+"
                        r"\b(?:in|en) (?:the )?([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})"
                        r"(?:,\s*([A-Z][\w .'-]*))?\s*(?:\bin \d{4}\b.*|[|(].*)?$")


# US states as Spanish titles write them ("Allentow, Pensilvania"), read as the English name
SPANISH_STATES = {'Pensilvania': 'Pennsylvania', 'Nueva York': 'New York', 'Nueva Jersey': 'New Jersey',
                  'Carolina del Norte': 'North Carolina', 'Carolina del Sur': 'South Carolina',
                  'Dakota del Norte': 'North Dakota', 'Dakota del Sur': 'South Dakota', 'Nuevo México': 'New Mexico',
                  'Nuevo Mexico': 'New Mexico', 'Virginia Occidental': 'West Virginia', 'Luisiana': 'Louisiana',
                  'Misisipi': 'Mississippi', 'Misuri': 'Missouri', 'Míchigan': 'Michigan', 'Oregón': 'Oregon',
                  'Hawái': 'Hawaii', 'Tennesse': 'Tennessee'}
_SPANISH_STATE = re.compile(r'\b(' + '|'.join(map(re.escape, SPANISH_STATES)) + r')\b')


def english_states(title):
    return _SPANISH_STATE.sub(lambda m: SPANISH_STATES[m.group(1)], title or '')


STREET_WORDS = {'ave', 'avenue', 'blvd', 'boulevard', 'rd', 'road', 'street', 'hwy', 'highway', 'pkwy', 'parkway',
                'expy', 'expressway'}


STREETS = '|'.join(w.capitalize() for w in STREET_WORDS)


def place_phrases(text, limit=4):
    """Runs of capitalised words in a title or description that may name a place, longest first: "Al Khan
    Beach" from "Beautiful Sun Rise at Al Khan Beach, unedited sounds". Hashtags and links are skipped."""
    # in English when written in another language: every capitalised Vietnamese word is no town ("THÁNG", the
    # month, was Thắng: dFeQ32cS3WM; "Chung Cư", apartments, was Chung), and "Hà Nội" reads as Hanoi
    # ponytail: english() translates the first 1000 characters only; enough for titles and a description's top lines
    text = english(text) or text
    text = english_states(text)
    text = re.sub(r'#\w+|https?://\S+|www\.\S+', ' ', text or '')
    text = re.sub(r'\b(New York|Washington) State\b', r'\1', text)  # no "York State" (Albany, New York State)

    # "Mt. Lebanon" is one name (Mount Lebanon), not Lebanon; "Main St. Hackensack" keeps its street
    def saint(m):
        street = m.group(1) == 'St' and re.search(r"\b[A-Z][\w'’]*\s*$", text[:m.start()])  # "Main St."
        return m.group(0) if street else {'Mt': 'Mount ', 'Ft': 'Fort ', 'St': 'Saint '}[m.group(1)]
    text = re.sub(r"\b(Mt|Ft|St)\.?\s+(?=[A-Z])", saint, text)
    phrases, run, before = [], [], ''
    # a hyphen inside a word only ("Wilkes-Barre"): "Brooklyn Downtown- 15 mins" names Brooklyn (pj_63OC2rSM)
    for token in re.findall(r"[^\W\d_](?:[\w'’]|-(?=\w))*|[^\w\s]+|\d\w*", text) + ['.']:
        word = token.lower()
        if not run:
            start_after = before  # the word before a run: "Downtown Buffalo"
        before = word
        # a street is no place: "3 Ave Bronx" names the Bronx, "Jerome Ave" no town Jerome ("St" only after a
        # name: "St. Clair" is Saint Clair)
        if word in STREET_WORDS or word == 'st' and run:
            run = []
            continue
        # a two-letter word that joins nothing ends a name: "Orosi,Ca Cutler,Ca" names Orosi and Cutler
        if (token[:1].isupper() and word not in NOT_PLACE and (len(token) > 2 or word in PLACE_JOINERS)
                or run and word in PLACE_JOINERS):
            run.append(token)
            continue
        while run and run[-1].lower() in PLACE_JOINERS:
            run.pop()
        # a word that is also a common one names a place next to "downtown" ("Buffalo Downtown", PhIhzm_NmlQ)
        near_downtown = bool({word, start_after} & {'downtown', 'uptown', 'midtown'})
        if len(run) > 1 or run and len(run[0]) > 3 and (run[0].lower() not in COMMON_WORDS or near_downtown):
            phrases.append(' '.join(run))
        run = []
    # a name written with its state ("Cutler,Ca", "Dorado,Puerto Rico") is a place even when it is also a word,
    # and goes first
    word = r"[A-ZÀ-Þ](?:[^\W\d_]|['’-])+"  # accents too: San Sebastián, Peñuelas
    with_state = re.findall(rf"\b({word}(?: {word}){{0,3}}),\s*[A-Z][A-Za-z]\b", text)
    # or with a US state's code and no comma: "Buffalo NY", "Atlanta GA"
    codes = '|'.join(sorted(set(add_video.US_STATE_CODES.values())))
    with_state += [x for x in re.findall(rf"\b({word}(?: {word}){{0,3}}) (?:{codes})\b", text)
                   if not set(x.lower().split()) & NOT_PLACE]
    # the state or country named that way comes along: it says where to look for the name ("New York" itself
    # is no run of the words above, "New" being no place word)
    # "Coram in Suffolk, New York": the town before " in " is named that way too, ahead of the county after it
    with_state += [x for x, y in re.findall(rf"\b({word}(?: {word}){{0,3}}) in ({word}(?: {word}){{0,3}})\s*,\s*"
                                            rf"[A-Z][\w'’]*(?: [A-Z][\w'’]*){{0,2}}", text)
                   if not set(x.lower().split()) & NOT_PLACE]  # not "Driving" in "Driving in Brooklyn, New York"
    # the region may run on into the title ("Independence, Missouri Driving Tour"): its longest start that is one
    for x, y in re.findall(rf"\b({word}(?: {word}){{0,3}})\s*,\s*([A-Z][\w'’]*(?: [A-Z][\w'’]*){{0,2}})", text):
        region = next((r for k in range(len(y.split()), 0, -1)
                       if (r := ' '.join(y.split()[:k])) in _region_names()), None)
        if region:
            with_state += [x, region]
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
        except GeocoderServiceError as e:  # 404: LocationIQ found nothing, Nominatim may (OJm2GB-6-1E's
            if '404' not in str(e):         # "Henry Hudson Highway, New York" with all the mapping's countries)
                raise
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


LANDMARK_TYPES = {'stadium', 'arena', 'casino', 'university', 'college', 'museum', 'theatre', 'cathedral',
                  'conference_centre', 'events_venue', 'zoo', 'theme_park'}


ROAD_KIND = re.compile(r'\b(?:highway|hwy|parkway|pkwy|expressway|expy|freeway|fwy|turnpike|tpke)\b', re.I)


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
        cls = r.raw.get('class') or r.raw.get('category')  # Nominatim's jsonv2 says "category"
        main_road = cls == 'highway' and r.raw.get('type') in ('motorway', 'trunk', 'primary')
        # a main road by its name whatever kind of way a title calls it (user, 2026-10-14: OJm2GB-6-1E's "Henry
        # Hudson Highway, New York" is the Henry Hudson Parkway in New York, not York)
        # (LocationIQ gives a road no class: its address names the road, and the title called it a highway)
        is_road = main_road or not cls and r.raw.get('address', {}).get('road') == _osm_name(r.raw) \
            and bool(ROAD_KIND.search(phrase))
        road = is_road and (core := _name_key(ROAD_KIND.sub(' ', _osm_name(r.raw)))) \
            and core in _name_key(ROAD_KIND.sub(' ', phrase))
        return (_osm_type(r.raw) not in ('country', 'state') and name and (name in wanted or wanted in name or road)
                and (cls in ('place', 'boundary', 'tourism', 'natural', 'leisure', 'landuse') or main_road or road
                     # a landmark that names its city: T-Mobile Arena is a "building", a stadium (gkGtNPzzeJ8)
                     or r.raw.get('type') in LANDMARK_TYPES))
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
    name = re.sub(r'^the ', '', name)  # "The Bronx" is the Bronx
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
        # a misspelt title ("Pougkeepsie"): the closest name, if it is spelt nearly the same
        data = add_video.get_locality_data(name, 'US', state, fuzzy=0.8) or {}
        g = next((g for g in data.get('geonames') or []
                  if difflib.SequenceMatcher(None, _name_key(g.get('name', '')), _name_key(name)).ratio() >= 0.85),
                 None)
    if not g:
        return None
    return (g['name'], g.get('adminName1'), 'US', f"US-{g.get('adminCode1')}", float(g['lat']), float(g['lng']), 0.5,
            'town' if int(g.get('population') or 0) >= SMALL_POP else 'village')


OWN_NAME_PLACE = 0.4  # ... and of a town found by its name alone at the other end of such a drive
MAJOR_PLACE = 0.65  # OpenStreetMap importance of a city known by its name alone: Atlanta 0.75, Brooklyn 0.74


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
    if named:  # looked up in the countries of the states named only: LocationIQ finds "Henry Hudson Highway, New
        # York" (the parkway, OJm2GB-6-1E) in the US, and nothing when asked in every country of the mapping
        codes = tuple(sorted({code for code in codes for sub in pycountry.subdivisions.get(country_code=code) or []
                              if sub.name in named})) or codes

    def usable(hit):
        # the reach of home is for places abroad: Elkhart, Indiana is far from New York but in the same country
        # a US state is no town ("Driving In Florida USA" is no "Florida, FL", izutQEcon0w); New York City is
        return (hit and hit[2] in countries and (not only or hit[2] in only)
                and not (hit[2] == 'US' and hit[0] and hit[0] == hit[1] and hit[0] != 'New York')
                and not (home and countries[hit[2]] != home[2] and _km(home[3], home[4], hit[4], hit[5]) > ABROAD_KM)
                and (len(named) != 1 or hit[1] in named))
    # "from Longwood to West Farms in the Bronx, New York": the best-known place a text names (the Bronx) is the
    # anchor, and each other name is the place of that name nearest it, not the one nearest the channel's home
    if home and {_name_key(p) for p in phrases} & (country_names - {_name_key(home[2])}):
        home = None  # the text names another country ("…en Adjuntas, Puerto Rico"): it is there, however far
    # and only there: "San Lorenzo, Puerto Rico" is not San Lorenzo, California
    only = {code for code, name in countries.items() if _name_key(name) in {_name_key(p) for p in phrases}}
    # a country outside the ones searched: the place is there, not a namesake here ("Java, Indonesia" in the
    # description is no Java in Hong Kong, 9zrNsh4xh9A)
    if not only and {_name_key(p) for p in phrases} & (country_names - {_name_key(c) for c in countries.values()}):
        return []
    if only and not named:
        country = countries[next(iter(only))]
        phrases = [p if _name_key(p) in country_names else f'{p}, {country}' for p in phrases]
    # a country's name with a state is a town ("Lebanon, Pennsylvania"), else the country ("GEORGIA" in a travel title)
    phrases = [p for p in phrases if named or _name_key(p) not in country_names]
    # a state only qualifies the town named with it ("Palmerton, Pennsylvania"): as a place of its own it would
    # be the anchor, or match some "Pennsylvania" road in another town; the town is looked up in it
    phrases = [p for p in phrases if _name_key(p) not in states]  # a state alone is no locality
    if len(named) == 1 and any(_name_key(p) not in states for p in phrases):
        phrases = [f'{p}, {next(iter(named))}' for p in phrases]
    # looked for nearest home, and with a state named anywhere in the country, not only within reach of home
    hits = [(p, _osm_place(p, codes, (home[3], home[4]) if home else None, 5000 if named else None)) for p in phrases]
    if home:  # a major city of the name beats a namesake near home (J7LDm95K-hM: Atlanta, GA for a Los Angeles
        # channel, not Atlanta, NV); Springfield (0.61) and Waterloo (0.56) are too many places to say
        hits = [(p, big if (big := _osm_place(p, codes)) and big[6] >= MAJOR_PLACE and (not h or h[6] < big[6] - 0.2)
                 else h) for p, h in hits]
    if len(named) == 1 and 'US' in codes:
        # GeoNames only where OpenStreetMap found the name somewhere else (Johnson City: a hamlet 100 km away);
        # a borough's two centres can lie 5-10 km apart (the Bronx, Brooklyn: both in New York City)
        hits = [(p, g if (g := _geonames_town(p, next(iter(named)))) and not (h and _km(h[4], h[5], g[4], g[5]) <= 15)
                 else h) for p, h in hits]
    hits = [(p, h) for p, h in hits if usable(h)]
    if len(hits) > 1:
        anchor = max(hits, key=lambda ph: ph[1][6])
        # a name OpenStreetMap does not know near the anchor (a small neighbourhood) says nothing: a far
        # namesake (Longwood, PA for Longwood in the Bronx) would only look like a second town
        hits = [(p, h if p == anchor[0] else _osm_place(p, codes, anchor[1][4:6], 50)) for p, h in hits]
    for phrase, hit in hits:
        # the town the phrase itself names keeps its name ("Hazlet"); a sight in a town counts as usual
        # ... also with its state written after it without a comma ("Philadelphia Tennessee" is Philadelphia, not
        # the Loudon it would count for as a small town: bL6hYeFo5ks)
        bare = re.sub(rf'\s+(?:{"|".join(re.escape(n) for n in add_video.US_STATE_CODES)})$', '',
                      phrase.split(',')[0], flags=re.I)
        itself = hit and hit[0] and _name_key(bare) == _name_key(re.sub(r' Township$', '', hit[0]))
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
    elif not township:  # a village GeoNames does not list touches too, as a town of no size: Schaghticoke is not
        # Mechanicville, 8.2 km off (H2vhezkUmFE; user, 2026-10-15). On 1,032 reviewed drive titles 11 ends change,
        # each to the town the title names (New Ringgold, not Orwigsburg; Franktown, not The Pinery)
        bigger = [p for p in bigger if _km(lat, lon, p[2], p[3]) <= _town_km(0) + _town_km(p[1])]
    if township and not bigger:  # a township around a small borough counts for it: Bridgewater Township is Montrose
        bigger = [p for p in places if _km(lat, lon, p[2], p[3]) <= MAJOR_TOWN_KM]
    if not small or not bigger:
        return None
    return min(bigger, key=lambda p: _km(lat, lon, p[2], p[3]))[0]


# areas OpenStreetMap names as if towns that are none: a BC regional district's electoral areas and the University
# Endowment Lands around UBC (TAmnollWXHg, 3_862G2nsDo)
UNINCORPORATED = re.compile(r'^(?:Electoral Area\b|University Endowment Lands$)')


def _electoral_area_town(at):
    """The town a point in a BC regional district's electoral area (which OpenStreetMap files as a municipality:
    TAmnollWXHg, Electoral Area A is UBC) counts for: the city OpenStreetMap puts the nearest place of SMALL_POP or
    more within MAJOR_TOWN_KM outside the area is in (Vancouver, not its neighbourhood West Point Grey), else None:
    countryside."""
    places = [p for p in (_nearby_places(*at) or []) if p[1] >= SMALL_POP and _km(*at, p[2], p[3]) <= MAJOR_TOWN_KM] \
        if at else []
    for _, _, lat, lon in sorted(places, key=lambda p: _km(*at, p[2], p[3]))[:5]:  # UBC is in the area itself
        h = _osm_reverse(round(lat, 3), round(lon, 3))
        if h and h[0] and not UNINCORPORATED.match(h[0]):
            return h[0]
    return None


def _hit_locality(hit, df, countries, at=None, named=False):
    """[locality, state, country] of an OpenStreetMap hit: the mapping locality of the town it lies in, else of its
    region when that is a locality (a mountain park in an emirate), else that town as a new locality. A small
    place counts for the bigger one nearest the hit (or at=(lat, lon)), in the mapping or new (_major_town).
    Neighbourhoods come as their city already (Brooklyn is New York). named: a title names the town itself, so it
    stays (Hazlet, not the Union Beach a "Hazlet Township" would count for)."""
    town, region, code, region_code = hit[:4]
    kind = hit[-1] if len(hit) in (5, 8) else None
    at = hit[4:6] if len(hit) > 5 else at
    if town and UNINCORPORATED.match(town):
        town = _electoral_area_town(at)
    if town:  # OpenStreetMap's "Sugar Notch Borough" is the mapping's Sugar Notch (no locality ends in Borough)
        town = re.sub(r' Borough$', '', town)
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
    # a US census place (unincorporated, no city of its own) next to a big city is that city: Paradise and Spring
    # Valley, NV are Las Vegas (LXKg43TMXek); a census place that is a big city itself (Honolulu) stays
    if town and at and state and countries[code] == 'United States' and _census_place(town, region) \
            and (city := _city_around(at, rows, town)):
        return city
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


@_remembered
def _census_place(town, region):
    """Whether OpenStreetMap has the town as a US census place (boundary=census: unincorporated)."""
    r = _nominatim('geocode', f'{town}, {region}, USA', timeout=15)
    return bool(r and r.raw.get('class') == 'boundary' and r.raw.get('type') == 'census')


def _city_around(at, rows, town):
    """[locality, state, country] of the mapping city of TITLE_CITY_POP or more within METRO_KM of at (rows: one
    state's), nearest first, unless the town is such a city itself; else None."""
    pop = pd.to_numeric(rows['population_locality'], errors='coerce')
    if (pop[rows['locality'].map(_name_key) == _name_key(town)] >= TITLE_CITY_POP).any():
        return None
    near = [(_km(at[0], at[1], float(r.lat), float(r.lon)), r) for r in rows[pop >= TITLE_CITY_POP].itertuples()
            if pd.notna(r.lat) and pd.notna(r.lon)]
    near = [(d, r) for d, r in near if d <= METRO_KM]
    if not near:
        return None
    r = min(near, key=lambda dr: dr[0])[1]
    return [r.locality, r.state if isinstance(r.state, str) else None, r.country]


METRO_POP = 1_000_000  # a city this big has a metropolitan area: towns around it count as the city
METRO_KM = 30          # within this many km of its centre (Yonkers 25 km from Manhattan, Bellflower 20 km from LA)
METRO_TOWN_POP = 250_000  # a town this big stays a city of its own (Newark, Jersey City)


TITLE_CITY_POP = 200_000  # a city a title names with towns around it is the drive's locality (Las Vegas: 650,000)


def metro_of_all(locs, df):
    """The city among a title's localities when all the others are towns around it (within METRO_KM, same state, at
    most half its population) and it has TITLE_CITY_POP or more, else None: "Driving in Las Vegas | North Town ➡️
    Henderson" is Las Vegas (H_8m8pKoG2c), as chapters through its towns are; "Ashland to Waterloo" stays open."""
    rows = {}
    for loc in set(map(tuple, locs)):
        r = df[(df['locality'] == loc[0]) & (df['country'] == loc[2])
               & ((df['state'] == loc[1]) if loc[1] else df['state'].isna())]
        if len(r):
            rows[loc] = r.iloc[0]
    if len(rows) < 2 or len(rows) != len(set(map(tuple, locs))):
        return None
    pop = {loc: float(pd.to_numeric(r['population_locality'], errors='coerce') or 0) for loc, r in rows.items()}
    city = max(pop, key=pop.get)
    c = rows[city]
    around = all(loc == city or (loc[1] == city[1] and pop[loc] <= pop[city] / 2
                                 and _km(float(r.lat), float(r.lon), float(c.lat), float(c.lon)) <= METRO_KM)
                 for loc, r in rows.items())
    return list(city) if pop[city] >= TITLE_CITY_POP and around else None


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
# "Desde Sabana Grande a Ponce", "Ponce a Juana Diaz,Puerto Rico", "de San Juan hacia Carolina": in Spanish too,
# tried in this order
_END = r"\s*(?:,|\||\(|\s-\s|\s–\s|\bin\b|\ben\b|\bvia\b|\bpor\b|$)"
FROM_TO = (re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)" + _END, re.I),
           # without "from": "Driving Lincoln to Hickman, Nebraska" (yQbYiSKgB-I), "Ashland to Waterloo Drive Tour"
           # ... also "4K POV Drive: Fremont to Downtown Omaha, Nebraska" (iSMg2dH2uuA: a colon after the word)
           re.compile(r"\b(?:driv(?:e|ing)|cruis(?:e|ing)|road ?trip)\s*[:|–-]?\s+"
                      r"(?!(?:to|back|home|down|up|through|thru|"
                      r"around|in|into|along|across|past|by|over|tour|on|at|the|my|our|a|out|with|all|"
                      r"welcome|return|dusk|dawn|day|night|sunset|sunrise|morning|evening)\b)"
                      # ... with a note on the start in brackets: "Kansas City (Legends Outlets) to Lawrence, KS"
                      r"([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})(?:\s*\([^)]*\))?\s+to\s+(.+?)" + _END, re.I),
           re.compile(r"^(?!(?:Welcome|Dusk|Dawn|Day|Night|Sunset|Sunrise|Morning|Evening|Sea)\b)"
                      r"([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})\s+[Tt]o\s+([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})"
                      r"\s+(?i:drive|driving|road ?trip|cruise)\b"),
           # or a title made of the drive and what follows: "Lilongwe to Mponela - M1 Upgrade Status" (iWbS0H_Prdc)
           # ... "To" too: "Lake George To New York City ( Full Video)" (6cAuDtDOx3Q; user, 2026-10-16)
           re.compile(r"^(?!(?:Welcome|Back|Return|Road|Guide|Dusk|Dawn|Day|Night|"
                      r"Sunset|Sunrise|Morning|Evening|Sea)\b)"
                      r"([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})\s+[Tt]o\s+([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})"
                      r"\s*(?:\s[-–—|]\s|\||:|\(|$)"),
           # ... or such a piece after a dash, bar or colon: "Driving Germany 4K - Werder to Berlin" (XfTNDV_nagE)
           re.compile(r"(?:\s[-–—|]\s|\|\s*|:\s+)(?!(?:Welcome|Back|Return|Road|Guide|Dusk|Dawn|Day|Night|"
                      r"Sunset|Sunrise|Morning|Evening|Sea)\b)"
                      r"([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})\s+[Tt]o\s+([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})"
                      r"\s*(?:\s[-–—|]\s|\||:|\(|$)"),
           # an arrow between two places ("Ingleburn → Surry Hills | Sydney POV", AroundSydney), or a dash right after
           # a drive word ("Drive Wąchock – Starachowice", 3Xg0vN2LqKo; user, 2026-10-16); a dash elsewhere
           # separates a title's parts ("TeslaXDrive — What This Channel Is")
           re.compile(r"(?:^|\|\s*|:\s+)([A-ZÀ-Þ][\w'’.-]*(?: [A-ZÀ-Þ][\w'’.-]*){0,3})\s*(?:→|->|⇒)\s*"
                      r"([A-ZÀ-Þ][\w'’.-]*(?: [A-ZÀ-Þ][\w'’.-]*){0,3})"),
           re.compile(r"\b(?:[Dd]rive|[Dd]riving|[Rr]oute)\s+([A-ZÀ-Þ][\w'’.-]*(?: [A-ZÀ-Þ][\w'’.-]*){0,3})\s+[–—]\s+"
                      r"([A-ZÀ-Þ][\w'’.-]*(?: [A-ZÀ-Þ][\w'’.-]*){0,3})"),
           re.compile(r"\b(?:[Dd]esde|de)\s+(?:el |la )?(.+?)\s+(?:hasta|hacia|a|al)\s+(?:el |la )?(.+?)" + _END),
           re.compile(r"^(?:el |la )?(.+?)\s+(?:hasta|hacia|a|al)\s+(?:el |la )?(.+?)" + _END))
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
    if a.get('state') == 'Puerto Rico' and a.get('county'):  # its localities are the municipios (Río Piedras is in
        town, kind = re.sub(r'\s+Municipio$', '', a['county']), 'city'  # San Juan), which OSM keeps as counties
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


GENERIC_ENDS = {'downtown', 'uptown', 'midtown', 'airport', 'theairport', 'home', 'thebeach', 'beach', 'citycenter',
                'citycentre', 'thecity', 'city', 'center', 'centre', 'town', 'work', 'school', 'hotel', 'thehotel'}


def _without_longer(text, name, states):
    """text without the US states whose names hold name as a word: the York of "New York City" is no region of
    York (ikS5Ru3CV8s: "Toronto, York" was looked up)."""
    for longer in states:
        if longer != name and re.search(rf'\b{re.escape(name)}\b', longer):
            text = re.sub(rf'\b{re.escape(longer)}\b', ' ', text)
    return text


def drive_ends(title, df, home=None):
    """(OpenStreetMap hit, locality) of where a drive "from A to B" starts and of where it ends (either may be
    None when not found), or None when the title names no such drive."""
    title = english_states(title)
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
    # a note in brackets is no part of the name ("Kansas City (Legends Outlets)": wAseuJ91-S4), nor "!" ("To Toronto!")
    first, last = (re.sub(r'\s*\([^)]*\)', '', x).strip().rstrip('!?. ') for x in (first, last))
    # nor the picture's quality: "Hollywood to Ventura 4K" ends in Ventura (lKtZ5xqatFo)
    first, last = (re.sub(r'(?:\s+(?:[2-8]K|HDR|UHD|\d+\s?fps|\d{3,4}p))+$', '', x, flags=re.I) for x in (first, last))
    # a US state's code after a town, without a comma, is its state: "Mobile AL to Panama City Beach FL" (9F47RLjPHdM);
    # not "LA", which is Los Angeles ("Downtown LA to Koreatown")
    names = {sub.code[3:]: sub.name for sub in pycountry.subdivisions.get(country_code='US') or []}
    own = [names[c.group(1)] if (c := re.search(r'\s+([A-Z]{2})$', x)) and c.group(1) in names and c.group(1) != 'LA'
           else None for x in (first, last)]
    first, last = (re.sub(r'\s+[A-Z]{2}$', f', {o}', x) if o else x for x, o in zip((first, last), own))
    # "to Downtown Omaha" is to Omaha: OpenStreetMap has no place of that name (iSMg2dH2uuA)
    first, last = (re.sub(r'^(?:downtown|uptown|midtown|central|historic|old town)\s+(?=[A-Z])', '', x, flags=re.I)
                   for x in (first, last))
    # a country the title names holds both ends: "Carolina a … San Juan, Puerto Rico" is not a Carolina in the US
    only = {code for code, name in countries.items() if name in title_countries(title)}
    # "from Newburgh to Brewster, New York": both ends in the states the title names (Maine has both too); with
    # one state named, each end is looked up in it ("Newburgh, New York"); only states of a country the title
    # names ("de San Juan hacia Carolina, Puerto Rico" names no San Juan province of Argentina)
    named = {sub.name for code in (only or codes) for sub in pycountry.subdivisions.get(country_code=code) or []
             if sub.name not in COUNTRIES  # Puerto Rico is a US subdivision but a country of the mapping
             and re.search(rf'\b{re.escape(sub.name)}\b', title)
             # as a qualifier of a place: after a comma or in an end's own words, not a road's name elsewhere in
             # the title ("Hollywood to Ventura 4K | Highland & Cahuenga Route" is in no Highland, Scotland)
             and (re.search(rf',\s*{re.escape(sub.name)}\b', title)
                  # after a town's name in an end, not heading it: "New York City To Toronto" names no state, its
                  # Toronto is no Toronto, New York (ikS5Ru3CV8s; user, 2026-10-16); nor do Kansas City, Washington DC
                  or any(re.search(rf'\S\s+{re.escape(sub.name)}\b', _without_longer(x, sub.name, names.values()))
                         for x in (first, last)))
             # a region that is one of the ends names no state: Lilongwe is a city and a district, and
             # "Lilongwe to Mponela" is no Mponela in that district (iWbS0H_Prdc: Mponela is in Dowa)
             and _name_key(sub.name) not in {_name_key(first), _name_key(last)}}
    if 'US' in (only or codes):  # by its code too: "Venice Beach to Malibu Beach, Los Angeles, CA" (k6V_9uIXVm4)
        named |= {sub.name for sub in pycountry.subdivisions.get(country_code='US') or []
                  if sub.code.split('-')[1] in title_states(title) and sub.name not in COUNTRIES}
    # an end's own state is its alone: "New York City To Washington DC" is no New York City in DC
    named -= {o for o in own if o}
    # a state heading an end is a city's name ("New York City To Toronto", "from New York to …"), unless the title
    # also writes it after a comma (ikS5Ru3CV8s: Toronto was looked for in New York State)
    named -= {n for n in named if any(re.match(rf'{re.escape(n)}\b', x) for x in (first, last))
              and not re.search(rf',\s*{re.escape(n)}\b', title)}
    if any(own):  # a US drive: no other country's region ("Bay Lake FL" names no Bay region of Somalia)
        named &= set(names.values())
    if len(named) == 1:
        state = next(iter(named))
        first, last = (x if state in x or o else f'{x}, {state}' for x, o in zip((first, last), own))
    # a named state says where, however far from home: no box around home (San Jose is 48 degrees from New York)
    near = (home[3], home[4]) if home and not named and not any(own) else None
    if only and not named:  # looked up in it, as with a state ("Carolina, Puerto Rico")
        country = countries[next(iter(only))]
        first, last = (x if country in x else f'{x}, {country}' for x in (first, last))

    def place(phrase, near=None, reach=None):
        # an end that names no place ("Airport to Downtown": St George, UT, not a Downtown near the channel)
        if _name_key(re.sub(r',.*', '', phrase)) in GENERIC_ENDS:
            return None
        hit = _osm_place(phrase, codes, near, reach)
        return hit if hit and (not only or hit[2] in only) else None
    a = place(first, near)
    b = place(last, a[4:6] if a else near, 300 if a else None)
    if not b and a and home:  # A was a far namesake (Longwood upstate for the one in the Bronx): B near home
        b = place(last, (home[3], home[4]))
    if not b and a:  # far from home too ("Sunrise- Fort Lauderdale": a Sunrise near New York): B anywhere, A near it
        b = place(last)
    # a drive between two towns is short: each end is also looked for near the other, and the closest pair is
    # meant ("from Plains to Hughestown": Plains Township next to Hughestown, not a Plains in New Jersey)
    pairs = [(x, y) for x in (a, b and place(first, b[4:6], 60))
             for y in (b, a and place(last, a[4:6], 60)) if x and y]

    def in_state(hit, o):  # in the end's own state, else in one the title names
        return hit[1] == o if o else not named or hit[1] in named
    if named or any(own):
        pairs = [(x, y) for x, y in pairs if in_state(x, own[0]) and in_state(y, own[1])]
    if pairs:
        a, b = min(pairs, key=lambda xy: _km(xy[0][4], xy[0][5], xy[1][4], xy[1][5]))
    elif named or any(own):  # one end only: it must be in the state named
        a, b = (x if x and in_state(x, o) else None for x, o in zip((a, b), own))
    # a major place by its name alone anchors the drive (x_M1WdCQ__8: "Mulholland to Hollywood" is no drive from a
    # Mulholland by Dublin to a Hollywood by Tracy: Hollywood is Los Angeles, and Mulholland Drive a road there that
    # OpenStreetMap gives no town for); the other end is looked for near it, else the drive is not split; not for a
    # famous namesake far from the channel (Paris, France for a Texas channel's Paris)
    for i, phrase in enumerate((first, last)) if not named else ():
        chosen, big = (a, b)[i], place(phrase)
        if (big and big[6] >= MAJOR_PLACE and (not chosen or _km(chosen[4], chosen[5], big[4], big[5]) > 30)
                and not (home and _km(home[3], home[4], big[4], big[5]) > ABROAD_KM)):
            other = place((last, first)[i], big[4:6], 60)
            # the other end a famous city by its own name too: a long drive between the two ("New York City To
            # Toronto", ikS5Ru3CV8s and 61x8XVlObEo; user, 2026-10-16)
            # ... or a town of its own by its name alone ("Lake George To New York City", 6cAuDtDOx3Q: importance
            # 0.44; a street or hamlet of a far namesake is far less)
            if not other and (alone := place((last, first)[i])) and alone[6] >= OWN_NAME_PLACE:
                other = alone
            a, b = (big, other) if i == 0 else (other, big)
            break
    end = b and _hit_locality(b, df, countries)
    # "…to West Farms in the Bronx", "…hasta el Barrio Portugues en Adjuntas": the drive ends in that place, and a
    # namesake of the end elsewhere (another Barrio Portugués) is not it
    if area := re.match(r"\s*(?:in|en)\s+(?:the\s+)?([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})", title[m.end(2):]):
        # near the drive: "in Nassau" is the county by Queens, not the village of Nassau near Albany
        q = area.group(1) + (f', {next(iter(named))}' if len(named) == 1 else '')
        c = _osm_place(q, codes, a[4:6] if a else b[4:6] if b else (home[3], home[4]) if home else None, 60)
        # not near the ends but a major place by its name: the ends were namesakes elsewhere ("Sugar Hill to
        # Washington Heights in Manhattan, New York" found both upstate, by Crown Point and Middletown, 2026-10-16)
        if not c and (far := _osm_place(q, codes)) and far[6] >= MAJOR_PLACE:
            c = far
            if a and _km(a[4], a[5], c[4], c[5]) > 60:
                a = None
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
    # with the region the title writes after it when no US state: "en Carolina,Puerto Rico" looked up alone, with
    # neither end found to search near, was a Carolina in Italy (kzMBsiZY_cQ; 2026-10-15)
    region = (area.group(2) or '').strip()
    q = area.group(1) + (f', {state[0]}' if len(state) == 1 else f', {region}' if region else '')
    near = near or ((home[3], home[4]) if home else None)
    hit = _osm_place(q, codes, near, 60) or _osm_place('the ' + q, codes, near, 60)
    if not hit and near and (far := _osm_place(q, codes)) and far[6] >= MAJOR_PLACE:
        hit = far  # the ends were namesakes far from it (see drive_ends: "… in Manhattan, New York")
    if hit and hit[0] and not re.search(r'\bCounty$', hit[0]):
        city = _hit_locality(hit, df, countries)
        # a start of that name inside C is part of it: "from Jamaica to Kew Gardens in Queens, New York" starts in
        # Queens' Jamaica, not the Jamaica Avenue of Wantagh (user, 2026-10-14: KJvEOnBJUHs had no locality)
        m = next((m for r in FROM_TO if (m := r.search(english_states(title)))), None)
        if m and (inside := _osm_place(m.group(1).strip(), codes, hit[4:6], 20)) \
                and _hit_locality(inside, df, countries) == city:
            return city
        # the start is a town of its own next to C ("from Alexandria to the Pentagon in Arlington"): not all in C;
        # a far start is a namesake of a place in C (Longwood upstate for the one in the Bronx)
        a, start = (drive_ends(title, df, home) or ((None, None),))[0]
        if a and start and start != city and _km(a[4], a[5], hit[4], hit[5]) <= METRO_KM:
            return None
        return city
    return None


def multi_city(meta, df, home=None):
    """Why a video, by its title or chapters, drives through more than one locality, else None: a drive from one
    town to another, to or away from a town, or chapters naming two or more localities (user, 2026-10-13: single
    cities first, these after everything else)."""
    title = meta.get('title') or ''
    if between_towns(title, df, home):
        return 'a drive from one town to another'
    # ... also with one end not found: held back is only later, not left out (wfq-d9YMXgI "Mzuzu to Nkhatabay",
    # LQzb0YUdzJs "Karonga to Chiweta"; user, 2026-10-16); not when the missing end names no place ("to Downtown")
    ends = drive_ends(title, df, home)
    m = next((m for r in FROM_TO if (m := r.search(english_states(title)))), None)
    # ... or two places of their own a small one of which counts for its neighbour in the mapping ("Drive Wąchock
    # – Starachowice", 3Xg0vN2LqKo: both Starachowice; user, 2026-10-16)
    (a, _), (b, _) = ends if ends else ((None, None), (None, None))
    if a and b and a[0] and b[0] and _name_key(a[0]) != _name_key(b[0]) and _km(a[4], a[5], b[4], b[5]) > 3 \
            and not area_city(title, df, home):
        return 'a drive from one town to another'
    if ends and m and sum(bool(loc) for _, loc in ends) == 1 and not area_city(title, df, home) \
            and not any(_name_key(re.sub(r',.*', '', x.strip())) in GENERIC_ENDS for x in m.groups()):
        return 'a drive from one town to another (one end not found)'
    if (loc := _title_locality(title, df)) and (role := title_role(title, loc)):
        return f"a drive {'to' if role == 'end' else 'away from'} {loc[0]}"
    # with the title's: "Pacific Coast in Los Angeles" whose chapters reach Malibu (5Du7Q0t-8Ns)
    named = {tuple(x) for _, _, x in chapter_localities(video_chapters(meta), df)}
    named |= {tuple(loc)} if loc else set()
    if len(named) >= 2:
        return f'its chapters name {len(named)} localities ({", ".join(sorted(n[0] for n in named))})'
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


def split_route(p, df, home=None, sig_path=None):
    """Split an untouched one-group proposal of a drive "from A to B" into the localities along its route, the
    driving time spread over them as on the route. Returns whether it was split. sig_path: the video's signals; without
    the object counts or town scenes that tell town driving from the highway, only A and B get segments."""
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
    stretches = ends_in_towns(stretches, segs, p.get('title'), df, home)
    p['entries'] = split_by_stretches(segs, stretches)
    # a town on the way is offered only where the picture can show it is town driving: without object counts or
    # town scenes no highway is cut, and its share of the route is highway as often as not (user, 2026-10-14:
    # Ie79UuTDKOw's Mountain Top, 26:21-28:06 on the highway)
    files = np.load(sig_path).files if sig_path and os.path.exists(sig_path) else None
    named = [tuple(loc) for _, _, loc in stretches if loc]
    if files is not None and 'urban' not in files and 'town_scene' not in files and len(named) > 2:
        ends = {named[0], named[-1]}
        left = sorted({e['locality'][0] for e in p['entries'] if tuple(e['locality']) not in ends})
        p['entries'] = [e for e in p['entries'] if tuple(e['locality']) in ends]
        if left:
            p['note'] = (f"{p.get('note') or ''}; towns on the way left out ({', '.join(left)}): no signal tells "
                         'their streets from the highway').lstrip('; ')
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


def ends_in_towns(stretches, segs, title, df, home=None):
    """The stretches with the proposal's first segment in the drive's start town A and its last in the end town B
    where the route has them as rural road (wAseuJ91-S4, "Kansas City (Legends Outlets) to Lawrence, KS": the
    outlets lie beyond the disc _urban gives Kansas City; the first segment, 0:01-6:55 after the rural and highway
    cuts, is the reviewer's Kansas City, the last, 23:02-35:02, their Lawrence; user, 2026-10-13)."""
    ends = drive_ends(title, df, home)
    if not ends or not segs:
        return stretches
    (_, start), (_, end) = ends
    first, last = min(segs, key=lambda x: x['start']), max(segs, key=lambda x: x['end'])

    def loc_at(t):
        return next((loc for a, b, loc in stretches if a <= t < b), stretches[-1][2])
    out = list(stretches)
    if start and loc_at(first['start']) is None:
        cut = first['end']
        out = [(0, cut, start)] + [(max(a, cut), b, loc) for a, b, loc in out if b > cut]
    if end and loc_at(max(last['end'] - 1, 0)) is None and last is not first:
        cut = last['start']
        out = [(a, min(b, cut), loc) for a, b, loc in out if a < cut] + [(cut, float('inf'), end)]
    return out


def _locality_anchor(loc, df):
    """(locality, state, country, lat, lon) of a mapping locality, like a channel's home, else None."""
    if not loc:
        return None
    _, row = add_video.get_existing_locality_row(df, *loc)
    return None if row is None else (loc[0], loc[1], loc[2], float(row['lat']), float(row['lon']))


def town_near(name, home, df=None):
    """[locality, state, country] of the town a place of that name near the channel's home is: itself, or for
    a neighbourhood the city it is part of (Echo Park is Los Angeles). A small town not in the mapping (df) next
    to a big city counts as that city, as for titles (Beverly Hills is Los Angeles: _metro_city)."""
    hit = _town_near(name, home)
    if not hit:
        return None
    town, lat, lon, osm_state = (*hit, None)[:4]  # results remembered before the state came as (town, lat, lon)
    rows = df[df['country'] == home[2]] if df is not None else None
    # the town's own state, as the mapping writes it (-VqR0H2f-JY: Zushi is in Kanagawa, not in Tokyo like home)
    state = _mapping_state(osm_state, rows, home[2]) or home[1]
    if rows is not None:
        known = (rows['locality'].map(_name_key).eq(_name_key(town))  # in that state: Pasadena, TX is another
                 & (rows['state'].isna() | rows['state'].eq(state))).any()
        if not known and (metro := _metro_city(town, (lat, lon), rows, state)):
            return metro
    return [town, state, home[2]]


def _mapping_state(osm_state, rows, country):
    """An OpenStreetMap state ("Kanagawa Prefecture", "California") as the mapping writes it ("Kanagawa", "CA"),
    else None."""
    if not osm_state:
        return None
    if country == 'United States':
        return add_video.US_STATE_CODES.get(osm_state.lower())
    bare = re.sub(r'\s+(?:Prefecture|Province|State|Region|Governorate|County)$', '', osm_state)
    states = set(rows['state'].dropna().unique()) if rows is not None else set()
    known = {_name_key(s): s for s in states}
    if _name_key(bare) in known:
        return known[_name_key(bare)]
    # the mapping may write the code: "New South Wales" is NSW
    iso2 = common.get_iso2_country_code(common.correct_country(country))
    code = next((sub.code.split('-')[1] for sub in pycountry.subdivisions.get(country_code=iso2) or []
                 if _name_key(sub.name) == _name_key(bare)), None)
    return code if code in states else bare


@_remembered
def _town_near(name, home):
    """(town, lat, lon) of a place of that name near home: OpenStreetMap tells a town from a neighbourhood (and
    gives the city it is part of); GeoNames calls both a section of a populated place."""
    r = _nominatim('geocode', f'{name}, {home[2]}', featuretype='settlement', addressdetails=True, timeout=15,
                   viewbox=[(home[3] - 1, home[4] - 1), (home[3] + 1, home[4] + 1)], bounded=True)
    if r is None or _km(home[3], home[4], r.latitude, r.longitude) > 60:
        return None
    address = r.raw.get('address', {})
    if _osm_type(r.raw) in ('city', 'town', 'village', 'municipality'):
        town = name
    else:
        town = address.get('city') or address.get('town') or address.get('village')
    # "Kamakura-shi" is Kamakura: a Japanese city's, ward's or town's suffix is no part of its name
    town = re.sub(r'-(?:shi|ku|cho|chō|machi|mura|gun)$', '', town or '') or None
    return (town, r.latitude, r.longitude, address.get('state') or address.get('province')) if town else None


def _km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


FAR_KM = 500  # a known place named in a description this far from the channel's home: its namesake near home?
ABROAD_KM = 1000  # a title place in another country counts only this close to the channel's home


def near_context(row, p, e, df, km=60):
    """Whether the place of mapping row is within km of the title's locality or of the video's other entries:
    then it is the one meant, no namesake near the channel's home (user, 2026-10-13: fUlJsPo_04c's La Jolla
    chapter among Del Mar and Encinitas ones of a "San Diego Coast" drive is no La Jolla near Los Angeles)."""
    around = [x['locality'] for x in p.get('entries', []) if x is not e and x.get('locality')]
    if (named := _title_locality(p.get('title') or '', df)):
        around.append(named)
    for loc in around:
        other = add_video.get_existing_locality_row(df, *loc)[1]
        if other is None or other['id'] == row['id']:  # the place itself says nothing about which one it is
            continue
        if _km(float(row['lat']), float(row['lon']), float(other['lat']), float(other['lon'])) <= km:
            return True
    return False


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


def channel_home(channel_id, df, plan, country=None):
    """(locality, state, country, lat, lon) where a channel mostly films: from its videos already in the mapping
    and localities set by hand in the plan, else from title guesses; None if nothing points anywhere. With the
    channel's country, a place in it comes first: vancouverdrive films trips to New York too, but lives in
    Vancouver."""
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
    ranked = sorted(counts.most_common(), key=lambda kv: country is not None and kv[0][2] != country)
    for (locality, state, where), _ in ranked:
        _, row = add_video.get_existing_locality_row(df, locality, state, where)
        if row is not None:
            return locality, state, where, float(row['lat']), float(row['lon'])
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


def one_end_city(title, df, home=None):
    """The locality of a drive "from A to B" with both ends in it, or with one end found and the other not, when a
    shorter part of the other ("Queens Village" or "Queens" of "Queens Village Queens") lies in that same locality,
    or with neither found, when B ends in the name of a big city ("to Elmhurst Queens"); else None."""
    ends = drive_ends(title, df, home)
    m = next((m for r in FROM_TO if (m := r.search(title or ''))), None)
    if not ends or not m:
        return None
    found = [loc for _, loc in ends if loc]
    if len(found) == 2 and found[0] == found[1]:  # "from Midtown East Manhattan to Garment District Manhattan"
        return found[0]
    state = [n.title() for n in add_video.US_STATE_CODES if re.search(rf'\b{re.escape(n)}\b', title, re.I)]
    if not found:
        # "from Corona to Elmhurst Queens": neither end found, but the end's last words name a big city the drive is in
        words = m.group(2).split()
        big = {tuple(r) for r in df.loc[df['population_locality'] >= METRO_POP,
                                        ['locality', 'state', 'country']].itertuples(index=False)}
        for part in [' '.join(words[i:]) for i in range(1, len(words))]:
            if part[:1].isupper() and (loc := place_locality([part] + state[:1], df, home)) and tuple(loc[0]) in big:
                return list(loc[0])
        return None
    if len(found) != 1:
        return None
    words = (m.group(1) if not ends[0][1] else m.group(2)).split()
    for part in [' '.join(words[:i]) for i in range(len(words) - 1, 0, -1)] + [' '.join(words[i:])
                                                                               for i in range(1, len(words))]:
        if part[:1].isupper() and (loc := place_locality([part] + state[:1], df, home)) and list(loc[0]) == found[0]:
            return found[0]
    return None


IN_PLACE = re.compile(r"\b(?:in|en) (?:the )?([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3})\s*(?:,|$)")


def locate_now(p, df, home, sig_path=None, meta=None, boilerplate=frozenset()):
    """The locality of a just-proposed video from its title: split along its route for a drive between towns, else the
    one town its title names, else its chapters (split at them when they name several), else its description
    (meta: the video's details); a video then in one place gets no rural or highway cuts (ONE_PLACE)."""
    e = p['entries'][0]
    # coordinates printed in the picture are ground truth: they decide before the title (user, 2026-10-09)
    if sig_path and split_gps(p, df, os.path.dirname(sig_path)):
        if 'entries' not in p:  # left out: the GPS passes no town
            return
    # a drive between two towns goes by its route, also when the title's guess named one of them (iWbS0H_Prdc:
    # "Lilongwe to Mponela - …" was all Lilongwe)
    elif (e.get('guessed') and any(r.search(p.get('title') or '') for r in FROM_TO)
          and split_route(p, df, home, sig_path)):
        pass
    elif not e.get('locality') and not split_route(p, df, home, sig_path):
        title = p.get('title') or ''
        if (same := one_end_city(title, df, home)):  # "… to Queens Village Queens": part of the end names it
            e['locality'], e['guessed'] = same, True
            p['note'] = f"{p['note']}; locality {same[0]}: both ends of the drive are in it".lstrip('; ')
        elif not any(r.search(title) for r in FROM_TO):
            hits = {tuple(loc): phrase for loc, phrase in title_places(title, df, home)}
            # "by Hillcrest in Queens, New York": the place is part of the city named after "in" (the Hillcrest
            # found on its own is a namesake upstate)
            # in a big city only: "Bellmore in Nassau" is in Nassau County, not in the village of Nassau upstate, so
            # Bellmore is looked up without it
            if (inside := IN_PLACE.search(title)) and place_phrases(title[:inside.start()]):
                big = {tuple(r) for r in df.loc[df['population_locality'] >= METRO_POP,
                                                ['locality', 'state', 'country']].itertuples(index=False)}
                hits = ({loc: ph for loc, ph in hits.items() if ph.split(',')[0] == inside.group(1) and loc in big}
                        or {tuple(loc): ph for loc, ph in place_localities(
                            place_phrases(title[:inside.start()] + title[inside.end(1):]), df, home)})
            if len(hits) > 1 and (city := metro_of_all(list(hits), df)):
                hits = {tuple(city): hits[tuple(city)]}
            if len(hits) == 1:
                (loc, phrase), = hits.items()
                e['locality'], e['guessed'] = list(loc), True
                p['note'] = f"{p['note']}; locality {loc[0]} from where '{phrase}' is".lstrip('; ')
    es = p.get('entries') or []
    if meta and len(es) == 1 and not es[0].get('locality') and not p.get('route') and sig_path \
            and 'screen_text' in (sig_files := np.load(sig_path)).files:
        # a town's name on the screen in the opening (QcYf2VZqS-Y: "Sacramento" at 0:26; user, 2026-10-13)
        texts = {float(t): x for t, x in json.loads(str(sig_files['screen_text'])).items()}
        # within the states and countries the title names: "Hatcher Pass, Alaska" is no Earth, TX from a map's
        # "Earth" at 0:01 (dqgfuV2HLeE; user, 2026-10-16)
        named_states, named_countries = title_states(meta.get('title') or ''), title_countries(meta.get('title') or '')
        where = df if not (named_states or named_countries) else df[
            df['state'].isin(named_states) | df['country'].isin(named_countries)]
        if (screen := screen_locality(texts, where)):
            es[0]['locality'], es[0]['guessed'] = screen[0], True
            p['note'] = (f"{p['note']}; locality {screen[0][0]} from a label on the screen at "
                         f"{int(screen[1]) // 60}:{int(screen[1]) % 60:02d}").lstrip('; ')
    if meta and len(es) == 1 and not es[0].get('locality') and not p.get('route'):
        e = es[0]
        near_home = (lambda name: town_near(name, home, df)) if home else None
        stretches = chapter_localities(video_chapters(meta), df, near_home)
        if len(stretches) == 1:
            e['locality'], e['guessed'] = stretches[0][2], True
            p['note'] = f"{p['note']}; locality {stretches[0][2][0]} from the chapters".lstrip('; ')
        elif stretches:
            p['entries'] = split_by_stretches(e['segments'], stretches)
            p['note'] = (f"{p['note']}; localities from the chapters: "
                         + ', '.join(f'{loc[0]} from {a // 60}:{a % 60:02d}' for a, _, loc in stretches)).lstrip('; ')
        elif found := description_locality(meta.get('description'), df, boilerplate):
            # far from the channel's home, the place of that name near home is meant (x_M1WdCQ__8: "Hollywood" in a
            # Los Angeles channel's description is no Hollywood, FL)
            row = add_video.get_existing_locality_row(df, *found)[1]
            if (home and row is not None and _km(home[3], home[4], float(row['lat']), float(row['lon'])) > FAR_KM
                    and (near := place_locality([found[0]], df, home)) and near[0] != list(found)):
                found = near[0]
            e['locality'], e['guessed'] = found, True
            p['note'] = f"{p['note']}; locality {found[0]} from the description".lstrip('; ')
        es = p.get('entries') or []
    # not after a GPS split that left one town (kddS8sGShKA: rural until 95:30, then Laughlin): the GPS decides
    if sig_path and len(es) == 1 and es[0].get('locality') and not p.get('route') and not es[0].get('gps'):
        es[0]['segments'] = propose(dict(np.load(sig_path)), ONE_PLACE)[0]
        p['note'] = re.sub(r'; (?:rural|highway) driving \d+-\d+s', '', p['note'])


# a suggested town the reviewer replaces with a bigger city nearby, as a rule for later suggestions (user, 2026-10-13:
# "Long Beach is considered part of LA"): {"town|state|country": {"to": [locality, state, country], "times": n,
# "videos": [...], "stated": bool}}; a rule once the reviewer said so or made the same change MERGE_TIMES times
# (one change can be the video's own: Kobe for Osaka where the title said Osaka)
MERGES = os.path.join('_output', 'proposals', 'locality_merges.json')
MERGE_TIMES, MERGE_KM = 2, 60


def _merge_key(loc):
    return '|'.join(str(x or '') for x in loc)


def load_merges():
    try:
        with open(MERGES) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_merge(old, new, video):
    """Note that the reviewer replaced the suggested locality old with new, when new is a bigger city within
    MERGE_KM of it (both in the mapping)."""
    df = add_video.load_csv(add_video.FILE_PATH)
    a, b = (add_video.get_existing_locality_row(df, *loc)[1] for loc in (old, new))
    if b is None:
        return
    # a suggested town not yet in the mapping (87XLoavxxSI's Long Beach) has nothing to compare: noted as it is
    if a is not None:
        far = _km(float(a['lat']), float(a['lon']), float(b['lat']), float(b['lon'])) > MERGE_KM
        smaller = float(pd.to_numeric(b['population_locality'], errors='coerce') or 0) \
            <= float(pd.to_numeric(a['population_locality'], errors='coerce') or 0)
        if far or smaller:
            return
    merges = load_merges()
    m = merges.setdefault(_merge_key(old), {'to': list(new), 'times': 0, 'videos': [], 'stated': False})
    if m['to'] != list(new):  # another city this time: start over with it
        m.update(to=list(new), times=0, videos=[])
    if video not in m['videos']:
        m['times'] += 1
        m['videos'].append(video)
    with open(MERGES + '.tmp', 'w') as f:
        json.dump(merges, f, indent=1, ensure_ascii=False)
    os.replace(MERGES + '.tmp', MERGES)


def merged_locality(loc, merges=None):
    """The bigger city the reviewer counts loc in (load_merges), else None."""
    m = (merges if merges is not None else load_merges()).get(_merge_key(loc))
    return m['to'] if m and (m.get('stated') or m['times'] >= MERGE_TIMES) else None


def process_channel(url, country, out_dir, pause_s=15, limit=None, download=True, single_city=True, new_only=False,
                    only=None, deferred=False):
    """Propose segments for every new video of a channel and write plan.json for the review page.

    Re-running resumes: videos already in the plan keep their entry and decisions, finished analyses are
    reused, and a run stopped by YouTube's sign-in check continues where it stopped. With limit, a run stops
    after that many new proposals, so the next run is the next batch. Without download, nothing is
    downloaded: each video is proposed whole from its metadata, and a later run with download analyses
    those not decided yet. only: just these video ids (to propose again ones taken out of the plan). single_city:
    a video through several localities (multi_city) is held back in the plan ("deferred"); deferred: propose those
    now, and nothing else ('us': the videos in the US of a channel elsewhere, held back until the US step).
    """
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, 'plan.json')
    plan = {}  # video -> plan entry, in plan order
    failed = set()  # videos an earlier run could not analyse: tried again, but after the others
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            for p in json.load(f):
                if str(p.get('exclude', '')).startswith('could not analyse'):
                    failed.add(p['video'])
                else:
                    plan[p['video']] = p

    if deferred:  # the drives held back, now that everything else is done
        kind = 'multi_city' if deferred is True else deferred
        # the US step: drives within one city first there too
        only, single_city = {v for v, p in plan.items() if p.get('deferred') == kind}, kind == 'us'
        for v in only:
            plan.pop(v)
    if os.path.exists(os.path.join(out_dir, CHANNEL_REJECTED)):  # "Reject the whole channel" on the review page
        print('Stopped early: the whole channel was rejected in review, so it is rejected; mark it "Rejected" in the '
              'channels sheet')
        return plan_path
    # marked in review ("Sped up", "Ghosts" on the review page): a channel's editing habit, so two such videos
    # reject it
    for key, (what, habit) in REVIEW_MARKS.items():
        marked = [v for v, p in plan.items() if p.get(key)]
        if len(marked) >= MARKED_REJECT:
            print(f'Stopped early: {len(marked)} videos marked {what} in review ({", ".join(marked[:5])}): this '
                  f'channel {habit}, so it is rejected; mark it "Rejected" in the channels sheet')
            return plan_path
    # most of its analysed videos bob at a walker's step rate: the channel films walks (Triple_A, 84 of 96; user,
    # 2026-10-16: 47 of them walks). Of the channels with 5 or more analysed, the next is at 60% of 10
    proposed = [p for p in plan.values() if p.get('entries') and p.get('analysed')]
    walks = [p for p in proposed if WALK_HINT in (p.get('note') or '')]
    if len(proposed) >= WALK_CHANNEL_MIN and len(walks) >= WALK_CHANNEL_SHARE * len(proposed):
        print(f"Stopped early: {len(walks)} of {len(proposed)} analysed videos bob like a walker's steps: this "
              'channel films walks, so it is rejected; mark it "Rejected" in the channels sheet')
        return plan_path

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
        night = int(bool(NIGHT_TITLE.search(title_text(p.get('title')))))
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
                if cur is None or ours or as_proposed(cur) or str(cur.get('exclude', '')).startswith('could not') \
                        or (deferred and cur.get('deferred')):
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
    early = None  # (home, localities in reach) for the locality of each video as soon as it is proposed
    failures_in_a_row = 0
    skips_in_a_row = 0
    # this channel opens with highlights: start after them (the reviews are on the reviewing machine, which marks
    # such a channel for the workers: sync_workers.py)
    montages = montage_channel(plan, out_dir) or os.path.exists(os.path.join(out_dir, MONTAGE_MARK))
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
        # chosen videos need no listing: a search URL's results or a channel YouTube will not list may lack them
        # (AbaoAmbience, CaliforniaGoldenState: 32 left out for good, 2026-10-14)
        if only is not None:
            ids = sorted(only)
        elif cached.get('date') == today and cached.get('url') == url:
            ids = cached['ids']
        else:
            try:
                ids = list_channel_api(url) if use_api else list_channel(url)
            except RuntimeError as e:
                if not use_api:
                    raise
                print(f'YouTube Data API refused ({str(e)[:80]}): listing with yt-dlp instead', flush=True)
                ids, use_api = list_channel(url), False  # the quota is used up: details come from yt-dlp too
            with open(listed + '.tmp', 'w') as f:
                json.dump({'date': today, 'url': url, 'ids': ids}, f)
            os.replace(listed + '.tmp', listed)
        todo = [v for v in ids if v not in in_mapping and (v not in plan or v in reanalyse)]
        if only is not None:
            todo = [v for v in todo if v in only]
        if new_only and not parse_qs(urlparse(url).query).get('query'):
            # a channel done before: its uploads (newest first) since its newest video in the mapping
            first = next((i for i, v in enumerate(ids) if v in in_mapping), len(ids))
            todo = [v for v in todo if v in set(ids[:first])]
        scope = set(todo) if new_only else None  # what "still to do" counts (US-4K: 0 new, 25 older unmapped)
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
    # a video YouTube keeps refusing (503 on its stream while others download) would otherwise stop every run
    for n, vid in sorted(enumerate(ids, 1), key=lambda x: x[1] in failed):
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
            if new_only and (newest := newest_upload(df).get(meta.get('channel_id'))) \
                    and _ymd(meta.get('upload_date')) < newest:
                scope.discard(vid)
                continue  # search results come in no date order: older than the channel's newest video in the mapping
            reason = exclusion_reason(meta)
            # footage from outside the US first (user, 2026-10-16): a channel elsewhere filming in the US waits
            if not reason and not deferred and country.strip().lower() not in US_NAMES and us_title(meta['title']):
                plan[vid] = {'video': vid, 'title': meta['title'], 'deferred': 'us',
                             'exclude': 'held back for later: the title names a place in the US (footage from '
                                        'outside the US first)'}
                touched.append(vid)
                commit([vid])
                continue
            if not reason and single_city:
                if single_home is None:
                    single_home = channel_home(meta.get('channel_id'), df, plan, country) or False
                named = title_countries(meta['title'])
                if (why := multi_city(meta, df[df['country'].isin([country] + named)], single_home or None)):
                    # held back, not left out: proposed once everything else is done (--deferred)
                    plan[vid] = {'video': vid, 'title': meta['title'], 'deferred': 'multi_city',
                                 'exclude': f'held back for later: {why} (drives within one city first)'}
                    touched.append(vid)
                    commit([vid])
                    continue
            if reason:
                plan[vid] = {'video': vid, 'title': meta['title'], 'exclude': reason}
                touched.append(vid)
                commit([vid])
                continue
            if download and not os.path.exists(sig_path):
                print(f'[{n}/{len(ids)}] analysing {vid} {meta["title"][:60]}')
                if (meta.get('duration') or 0) > 2 * 3600:  # run_channels.py waits longer for its silence
                    print(f'  long video: {int(meta["duration"]) // 60} min')
                analyse_video(vid, out_dir, meta)
                time.sleep(pause_s)  # spacing requests keeps YouTube's bot check away for longer
            failures_in_a_row = 0
        except BotCheck as e:
            stopped = f'YouTube asks to sign in ({e})'
            break
        except Exception as e:
            if re.search(r'Video unavailable|Private video|members-only|This video has been removed', str(e)):
                # gone for good (removed, private, members only): left out, not retried, no failure
                plan[vid] = {'video': vid, 'title': '', 'exclude': f'unavailable on YouTube: {str(e)[-120:]}'}
                touched.append(vid)
                commit([vid])
                continue
            failures_in_a_row += 1
            # no connection (a dropped network, an unresolved name): not the video's fault, so it is not recorded and
            # a later run tries it again
            offline = re.search(r'Failed to resolve|Unable to download API page|timed out|Connection (?:reset|refused|'
                                r'aborted)|Network is unreachable|nodename nor servname', str(e))
            if vid not in reanalyse and not offline:  # a whole-video proposal stays as it was
                plan[vid] = {'video': vid, 'title': '', 'exclude': f'could not analyse: {e}'}
                touched.append(vid)
                commit([vid])
            continue
        analysed = os.path.exists(sig_path)
        reason = skip_note = None
        if analysed:
            frames = sheet_frames(os.path.join(out_dir, f'{vid}.jpg'))
            if inset_rectangle(frames):
                reason = 'another camera is shown in a corner of the picture (picture-in-picture)'
            elif (n_driver := driver_frames(frames)) >= DRIVER_SHEET_FRAMES:
                reason = f'the camera faces the driver (a big face in {n_driver} of {len(frames)} sampled frames)'
            elif (n_walk := walking_frames(os.path.join(out_dir, f'{vid}.jpg'))) >= WALK_SHEET_FRAMES:
                reason = f'a walk, not a drive ({n_walk} of {len(frames)} sampled frames look taken on foot)'
            elif float(np.load(sig_path).get('step_bob', 0)) >= STEP_BOB_WALK:
                reason = (f"a walk, not a drive: the camera bobs at a walker's step rate "
                          f"({float(np.load(sig_path)['step_bob']):.0%} of its shake)")
            elif dashboard_view(os.path.join(out_dir, f'{vid}.jpg')):
                reason = 'filmed from inside the car: the dashboard fills the picture, not a dashcam view'
            elif fast := overlay_speedup(dict(np.load(sig_path))):  # the cross traffic is a note: see below
                reason = fast
            else:
                reason = footage_problem(dict(np.load(sig_path)))
            # an edit cut is offered with a note to check, not left out (user, 2026-10-11: gaX_-_Nna9k's cuts are
            # legit and miss no intersection); ten in a row still say the channel edits its drives
            notes_skip = []
            if at := footage_skip(dict(np.load(sig_path))):
                notes_skip.append(f'the video skips footage at {at} '
                                  f'({"edit cuts" if " and " in at else "an edit cut"}): '
                                  'check that no wait at a light is missing')
            # the camera's clock tells cuts the picture does not (a crossfade: B0XUv86Y2Uc at 2:42), so both
            if at := clock_jump(dict(np.load(sig_path))):
                notes_skip.append(f'the video skips footage at {at} (the time stamp in the picture jumps): check')
            skip_note = '; '.join(notes_skip) or None
            if 'skips' in np.load(sig_path):  # only videos analysed for skips count towards the streak
                skips_in_a_row = skips_in_a_row + 1 if skip_note and not reason else 0
            if skips_in_a_row >= SKIP_STREAK and only is None:  # videos proposed again do not judge the channel
                plan[vid] = {'video': vid, 'title': meta['title'], 'exclude': skip_note}
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
            segments, notes = night_from_title(*propose(sig := dict(np.load(sig_path))), sig, meta.get('title'))
            if skip_note:
                notes.insert(0, skip_note)
            if (bob := float(np.load(sig_path).get('step_bob', 0))) >= STEP_BOB_HINT:
                notes.insert(0, f"the camera bobs like a walker's steps ({bob:.0%} of its shake): a walk? check")
            elif (hints := walk_hints(segments, np.load(sig_path).get('bob_track', np.zeros((0, 2))).tolist())):
                notes.insert(0, "the camera bobs like a walker's steps at " + ', '.join(
                    f'{a // 60}:{a % 60:02d}-{b // 60}:{b % 60:02d}' for a, b in hints) + ': a walk? check')
            # fast cross traffic at a wait is a hint, no proof: a wait beside a fast road looks the same (3QeTN5BygQE:
            # 100 km/h at 17:02, not sped up; approved waits cross at medians of up to 73 km/h; user, 2026-10-13)
            if (fast := cross_speedup(dict(np.load(sig_path)))):
                notes.insert(0, f'possibly {fast}: check, it may be a fast road beside the wait')
            # a highway in the title is analysed, not left out: such drives pass through towns on the way, which
            # the rural and highway checks and a GPS overlay find (user, 2026-10-11: uVHGEdTyAXk, Galt to Tulare)
            if hit := HIGHWAY_TITLE.search(title_text(meta['title'])):
                notes.append(f"the title mentions '{hit.group(0)}': check that only driving through towns is left")
                # ... and a short piece between two rural or highway stretches is no town on the way (1Rz63qdJN38
                # 17:21-20:08; user, 2026-10-16): in reviewed videos with a highway title, of 31 such pieces up to
                # HIGHWAY_GAP_S 30 were rejected (of 147 in other videos 44 approved: those stay)
                found = re.findall(r'(?:rural|highway) driving (\d+)-(\d+)s', '; '.join(notes))
                spans = [(int(a), int(b)) for a, b in found]
                gaps = [g for g in segments if g['end'] - g['start'] <= HIGHWAY_GAP_S
                        and any(abs(b - g['start']) <= 3 for _, b in spans)
                        and any(abs(a - g['end']) <= 3 for a, _ in spans)]
                if gaps:
                    segments = [g for g in segments if g not in gaps]
                    notes.append('left out between highway stretches, no town: ' + ', '.join(
                        f"{g['start'] // 60}:{g['start'] % 60:02d}-{g['end'] // 60}:{g['end'] % 60:02d}"
                        for g in gaps))
            if not segments:
                plan[vid] = {'video': vid, 'title': meta['title'],
                             'exclude': 'no usable footage left: ' + ('; '.join(notes) or 'too short')}
                touched.append(vid)
                commit([vid])
                continue
        else:
            night = bool(NIGHT_TITLE.search(title_text(meta['title'])))
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
        if (title_vt := title_vehicle(title_text(meta['title']))) is not None:
            vt, vehicle_note = title_vt, f'vehicle from the title ({VEHICLE_TYPES[title_vt]})'
        if vid in reanalyse:  # keep a vehicle type picked in the review page
            vt, vehicle_note = plan[vid].get('vehicle_type', vt), 'vehicle as in the earlier proposal'
        named = title_countries(meta['title'])  # "…hacia Carolina, Puerto Rico": not San Juan, Texas
        locality = title_city(meta['title'], guess_locality(meta['title'], named[0] if len(named) == 1 else country,
                                                            df), df)
        touched.append(vid)
        plan[vid] = ({
            'video': vid, 'title': meta['title'], 'analysed': analysed,
            'entries': [{'locality': locality, 'guessed': locality is not None, 'segments': segments}],
            'vehicle_type': 0 if vt is None else vt, 'upload_date': meta['upload_date'], 'channel_id': ch,
            'note': '; '.join(filter(None, ['; '.join(notes), vehicle_note])),
        })
        try:  # its locality at once, before it shows in an open review page (the full step follows after the batch)
            if early is None:
                counts = Counter(q.get('channel_id') for q in plan.values() if q.get('channel_id'))
                early_home = channel_home(counts.most_common(1)[0][0], df, plan, country) if counts else None
                reach = (df['country'] == country) | df['country'].isin(filmed_countries(counts, df))
                if early_home:
                    reach |= np.array([_km(early_home[3], early_home[4], la, lo) <= ABROAD_KM
                                       for la, lo in zip(df['lat'], df['lon'])])
                early = early_home, reach
            locate_now(plan[vid], df[title_reach(df, early[1], meta['title'])], early[0],
                       sig_path if analysed else None, meta, boilerplate)
        except Exception as e:  # a lookup gone wrong must not stop the batch; the full step tries again
            print(f'  locality not found yet ({type(e).__name__})', flush=True)
        if montages and analysed:
            trim_montage(plan[vid], dict(np.load(sig_path)))
        if analysed and (lead := channel_lead_in(meta.get('channel_id'), df)):
            trim_lead_in(plan[vid], lead)
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
    home = channel_home(channels.most_common(1)[0][0], df, plan, country) if channels else None
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
        # a title naming a city puts the video there: its chapters split it only where they name another mapping
        # locality outright ("Kobe City Skyline"), not by sights and junctions looked up near home (zvCKHw9rKPg:
        # "Osaka Bay Sunset Drive" became Tokyo from "Awaza Junction"; JCsEdFxRjtE: Kyoto became Zushi and Nara)
        titled = p['entries'][0].get('locality')
        # chapters before the first that names a place are in the title's (5Du7Q0t-8Ns: Los Angeles until "Malibu"
        # at 16:09; user, 2026-10-13), and those of a drive "from A to B" in A, its chapters naming the towns on
        # the way however far (PvQnDdF145Q: San Francisco to Los Angeles, 11 h by Santa Cruz and Morro Bay)
        ends = drive_ends(p['title'], df[in_reach], home)
        start = ends[0][1] if ends else None
        reach, at = in_reach, None if ends else _locality_anchor(titled, df)
        if ends and start and start[1]:  # in the states of its ends: no Venice, FL on a drive along the coast of LA
            reach = in_reach & df['state'].isin({start[1]} | ({ends[1][1][1]} if ends[1][1] else set()))
        stretches = chapter_localities(video_chapters(metas[p['video']]), df[reach],
                                       (lambda name: town_near(name, home, df)) if home and not titled else None,
                                       near=at[3:5] if at else None, first=start or titled)
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
            for loc, phrase in title_places(p['title'], df[reach], home):
                in_title.setdefault(tuple(loc), phrase)
            if len(in_title) > 1 and (city := metro_of_all(list(in_title), df)):
                in_title = {tuple(city): in_title[tuple(city)]}
            # the channel's usual locality named in the title, also as part of a road's name, goes first: "Which
            # Nashville Road Should We Take?" is Nashville, not the "Roads" of a district in Ireland or the "Lead"
            # of Lead, SD (UCLgFDrgDENPf7a59tsj8d_A; user, 2026-10-16)
            if (usual := usual_locality(plan)) and re.search(rf'\b{re.escape(usual[0])}\b', p['title'] or '', re.I):
                in_title = {tuple(usual): usual[0]}
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
            # a freeway drive whose only place is a region or a sight, or none: no town to give it, unless its GPS
            # overlay says (HXJ9w8h4CwA, "Southern California freeway drive" was Los Angeles; user, 2026-10-16)
            sig_path = os.path.join(out_dir, f"{p['video']}_signals.npz")
            if (HIGHWAY_TITLE.search(p['title'] or '')
                    and not (found and _name_key(found[0][0]) in _name_key(found[1]))
                    and not (os.path.exists(sig_path) and 'gps' in np.load(sig_path).files)):
                p.pop('entries')
                p['exclude'] = 'freeway drive with no town named (title, description, chapters or GPS)'
                break
            if found:
                e['locality'], e['guessed'] = list(found[0]), True
                p['note'] = f"{p['note']}; locality {found[0][0]} from where '{found[1]}' is".lstrip('; ')
            elif (abroad := [c for c in title_countries(p['title']) if not (usual := usual_locality(plan))
                             or c != usual[2]]):
                # a title naming another country is not the channel's usual place (X53fh12knn8: "Pakistan's
                # Busiest City Karachi To Lahore" was New York; an hour cannot be Karachi to Lahore either, so it is
                # left to the reviewer; user, 2026-10-16)
                p['note'] = (f"{p['note']}; the title names {', '.join(abroad)}: set the locality").lstrip('; ')
            elif (usual := usual_locality(plan)):  # nothing names a place: where this channel mostly drives
                e['locality'], e['guessed'] = list(usual), True
                p['note'] = (f"{p['note']}; locality {usual[0]}: where most of this channel's reviewed videos are "
                             '(nothing in the title or description names a place: check)').lstrip('; ')
    # a drive "from A to B" without chapters passes through A, B and the towns between: split it along its route
    for p in new:
        split_gps(p, df, out_dir) or split_route(p, df[title_reach(df, in_reach, p['title'])], home,
                                                 os.path.join(out_dir, f"{p['video']}_signals.npz"))
    if home:
        iso2 = common.get_iso2_country_code(common.correct_country(home[2]))
        # a route's towns come from its points on the map, so none is a namesake of a place near home
        # (3HN-bS3aZk8: Miami to Orlando passed a Charleston that became New York; user, 2026-10-16)
        for p in (p for p in new if not p.get('route')):
            for e in p.get('entries', []):
                loc = e.get('locality')
                if not loc or not e.get('guessed') or e.get('gps') or tuple(loc) == home[:3]:
                    continue
                _, row = add_video.get_existing_locality_row(df, *loc)
                if row is None or _km(home[3], home[4], float(row['lat']), float(row['lon'])) < 100:
                    continue
                # a big city is meant as itself, not a place of its name near home (zvCKHw9rKPg: Osaka)
                if float(pd.to_numeric(row.get('population_locality'), errors='coerce') or 0) >= TITLE_CITY_POP:
                    continue
                # a title naming a state means that state: "Milton, Pennsylvania" is not the Milton near New York,
                # and "… in Arlington, Virginia" is no place near New York at all
                named = {code for name, code in add_video.US_STATE_CODES.items()
                         if re.search(rf'\b{re.escape(name)}\b|,\s*{code}\b', p['title'] or '', re.I)}
                if named and (loc[1] in named or home[1] not in named):
                    continue
                if near_context(row, p, e, df):
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
                and not es[0].get('gps')
                and not any(s.get('decision') or s.get('applied') for s in es[0]['segments'])):
            if (role := title_role(p['title'], es[0]['locality'])):
                p['entries'] = by_role(propose(dict(np.load(sig_path)))[0], es[0]['locality'], role)
                p['note'] = (f"{p['note']}; the title drives {'to' if role == 'end' else 'from'} "
                             f"{es[0]['locality'][0]}: only its {role} is there, set the locality of the rest")
                continue
            es[0]['segments'] = propose(dict(np.load(sig_path)), ONE_PLACE)[0]
            p['note'] = re.sub(r'; (?:rural|highway) driving \d+-\d+s', '', p.get('note') or '')
    merges = load_merges()
    for p in new:  # a town the reviewer counts in a bigger city nearby (merged_locality): that city
        for e in p.get('entries', []):
            if e.get('locality') and e.get('guessed') and (to := merged_locality(e['locality'], merges)):
                p['note'] = (f"{p.get('note') or ''}; {e['locality'][0]} counts as {to[0]} (as in your reviews)"
                             ).lstrip('; ')
                e['locality'] = list(to)
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
        # ... and one found in the US without the title saying so ("Driving in Miami" from a channel elsewhere)
        in_us = any(len(e.get('locality') or ()) > 2 and e['locality'][2] == 'United States'
                    for e in p.get('entries', []))
        if not deferred and country.strip().lower() not in US_NAMES and undecided(p) and in_us:
            plan[p['video']] = {'video': p['video'], 'title': p['title'], 'deferred': 'us',
                                'exclude': 'held back for later: proposed in the US (footage from outside the US '
                                           'first)'}
    commit(touched)
    kept = sum('entries' in p for p in new)
    print(f'{len(new)} videos added or re-analysed: {kept} proposed, {len(new) - kept} excluded -> {plan_path}')
    if stopped:
        print(f'Stopped early: {stopped}. Run the same command later to continue.')
    left = sum(v not in in_mapping and (v not in plan or (v in reanalyse and v not in touched))
               and (scope is None or v in scope) for v in ids)
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
CHANNEL_REJECTED = 'channel_rejected'  # a file in a channel's folder: the reviewer rejected the whole channel
WALK_HINT = "the camera bobs like a walker's steps"  # the start of propose's walk hint in a note
WALK_CHANNEL_MIN, WALK_CHANNEL_SHARE = 20, 0.7  # analysed videos, the share with WALK_HINT: a channel of walks
MARKED_REJECT = 2  # videos marked sped up (or with ghosts) in review that reject their channel (user, 2026-10-10)
# plan key -> (what the review button marks, the channel's habit it shows)
REVIEW_MARKS = {'sped_up': ('sped up', 'speeds up its drives'),
                'ghosts': ('with ghosts', 'joins scenes with dissolves (ghosts: people and cars fading out)')}

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
        if row is None and new_row is None:
            # another name of a locality the mapping has, accents aside ("Hà Nội" is Hanoi, dFeQ32cS3WM)
            df = add_video.load_csv(add_video.FILE_PATH)
            key, there = _name_key(locality), df[df['country'] == country]
            hit = next((r for r in there.itertuples() if key in {_name_key(n) for n in
                        [r.locality] + add_video._parse_flat_list_cell(r.locality_aka)}), None)
            if hit is not None:
                locality, state = hit.locality, None if pd.isna(hit.state) else hit.state
                _, row = add_video.get_existing_locality_row(df, locality, state, country)
        if row is None:
            if new_row is None:
                df = add_video.load_csv(add_video.FILE_PATH)
                same = df[(df['country'] == country) & (df['state'].fillna('') == (state or ''))]
                # a typo of a locality the mapping has ("Montouk" for Montauk) is no new locality
                close = difflib.get_close_matches(locality, same['locality'].dropna().tolist(), n=1, cutoff=0.85)
                if close and not at:
                    raise ValueError(f'"{locality}" is not in the mapping, but {close[0]} is: pick it, or put the new '
                                     f'locality on the map if it really is another place')
                new_row = add_video.new_locality_row(locality, state or None, country, near=near, at=at)
                if new_row['lat'] is None or new_row['lon'] is None:
                    raise ValueError(f'no coordinates found for "{locality}": check its spelling, '
                                     'or put it on the map')
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
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
<text y=%22.9em%22 font-size=%2290%22>🚕</text></svg>">
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f6f6f6; color: #222; }
header { position: sticky; top: 0; z-index: 1; background: #fff; border-bottom: 1px solid #ddd;
         display: flex; gap: 20px; padding: 10px 20px; align-items: center; }
#player { width: 480px; height: 270px; border: 0; background: #000; }
#status { font-size: 15px; line-height: 1.6; }
#apply { padding: 10px 18px; font-size: 15px; border: 0; border-radius: 4px;
         background: #007bff; color: #fff; cursor: pointer; }
#reject-channel { padding: 10px 18px; font-size: 15px; border: 0; border-radius: 4px;
                  background: #c62828; color: #fff; cursor: pointer; }
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
.note .warn { color: #c62828; font-weight: 600; }
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
.channel { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
#vid, #clock { cursor: pointer; }
#vid { font-family: monospace; }
#vid-yt { text-decoration: none; }
.channel a { color: #1a5fb4; text-decoration: none; }
.channel a:hover { text-decoration: underline; }
button.small.yes, button.small.go { background: #dff3e2; border-color: #7cbf86; color: #1d6b2a; }
button.small.no, button.small.stop { background: #fbe3e3; border-color: #d98a8a; color: #9b2222; }
.seg.approve td { background: #e8f6ea; }
.seg.reject td { background: #fbe9e9; color: #888; }
.seg.approve .yes, .seg.reject .no { font-weight: bold; border-color: #333; }
.seg.applied td { background: #e3ecf8; }
tr.split td { font-size: 12px; color: #777; border-top: 0; padding-top: 0; padding-bottom: 0; }
tr.split td:first-child { max-width: 0; }  /* the reason wraps in the columns the segments set, never widens them */
.excluded { color: #777; font-size: 14px; }
.night { color: #2a3f8f; font-weight: bold; }
.guess { color: #b36b00; font-size: 12px; }
.clock { font-size: 18px; margin-bottom: 4px; }
#clock { color: #c00; cursor: pointer; text-decoration: underline; font-variant-numeric: tabular-nums; }
#copied { color: green; font-size: 13px; }
.keys { font-size: 12px; color: #555; margin-bottom: 6px; }
tr.seg.active td:first-child { box-shadow: inset 4px 0 #007bff; }
tr.split.playing td:first-child { box-shadow: inset 4px 0 #e8890c; }
tr.split.playing { background: #fff4e5; }
/* its own stacking context: Leaflet's layers (z-index 400+) would otherwise scroll over the sticky player */
.routemap { width: 640px; max-width: 100%; height: 300px; margin: 8px 0; border: 1px solid #ccc;
            position: relative; z-index: 0; }
.routemap .leaflet-tooltip { font-size: 11px; padding: 1px 4px; }
html[data-theme=dark]{filter:invert(.9) hue-rotate(180deg);background:#fff}
html[data-theme=dark] img,html[data-theme=dark] iframe,html[data-theme=dark] video,
html[data-theme=dark] .leaflet-container,html[data-theme=dark] .locmap,html[data-theme=dark] .routemap
{filter:invert(1) hue-rotate(180deg)}
#theme{position:fixed;right:12px;bottom:12px;z-index:9999;padding:6px 10px;border:1px solid #bbb;border-radius:16px;
background:#fff;color:#222;font-size:13px;cursor:pointer}
</style>
<script>
// dark unless switched to light; a cookie, not localStorage: the hub and each review page run on their own port,
// and cookies are shared across the ports of a host
(function () {
  const m = document.cookie.match(/(?:^|; )theme=(dark|light)/);
  document.documentElement.dataset.theme = m ? m[1] : 'dark';
  addEventListener('DOMContentLoaded', () => {
    const b = document.createElement('button');
    b.id = 'theme';
    const label = () => { b.textContent = document.documentElement.dataset.theme === 'dark' ? '☀️ Light' : '🌙 Dark'; };
    b.onclick = () => {
      const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = t;
      document.cookie = `theme=${t}; path=/; max-age=31536000; SameSite=Lax`;
      label();
    };
    label();
    document.body.appendChild(b);
  });
})();
</script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head><body>
<header>
  <div id="player"></div>
  <div>
    {% if channel %}<div class="channel">🚕 <a href="{{ channel.url }}" target="_blank" rel="noopener">
      {{- channel.name }}</a></div>{% endif %}
    <div class="clock">Video: <span id="vid" title="click to copy">–</span>
      <a id="vid-yt" href="#" target="_blank" rel="noopener" title="open on YouTube here">↗</a>
      <span id="vid-copied"></span></div>
    <div class="clock">Time: <span id="clock" title="click to copy">0:00</span> <span id="copied"></span></div>
    <div class="keys">Keys for the outlined segment (the one clicked last): <b>A</b> start here · <b>S</b> end here ·
      <b>D</b> whole video · <b>Q</b> day · <b>W</b> night · <b>F</b> split time here · <b>C</b> copy the time</div>
    <div id="status"></div>
    {% if waiting > open_videos | length %}<div class="keys">Showing {{ open_videos | length }} of {{ waiting }}
      videos left to decide; the next appear as you finish these.</div>{% endif %}
    <button id="apply" disabled>Apply approved segments to the mapping</button>
    <button id="reject-channel" onclick="rejectChannel()" title="reject every undecided segment and stop the channel"
            >Reject the whole channel</button>
    <div id="log"></div>
  </div>
</header>
<main>
{% for p in plan if p.entries and p.video in open_videos %}
<div class="video" data-video="{{ p.video }}" data-duration="{{ durations.get(p.video) or '' }}"><div class="main">
  <h3>{{ p.title }}</h3>
  <a href="https://www.youtube.com/watch?v={{ p.video }}" target="_blank">{{ p.video }}</a>
  {% if p.upload_date and (p.upload_date | string).isdigit() %}{% set u = '%08d' % (p.upload_date | int) %}
  <span class="uploaded" title="upload date (day, month, year)">· uploaded {{ u[:2] }}.{{ u[2:4] }}.{{ u[4:] }}</span>
  {% endif %}
  <label class="vehicle">Vehicle (whole video):
    <select autocomplete="off" onchange="setVehicle('{{ p.video }}', this)">
    {% for code, name in vehicle_types.items() %}
      <option value="{{ code }}" {{ 'selected' if code == p.vehicle_type }}>{{ code }} ({{ name }})</option>
    {% endfor %}
    </select></label>
  {% if p.note %}<div class="note">{{ note_html(p.note) }}</div>{% endif %}
  {% if p.analysed is false %}
  <img class="thumb" src="https://i.ytimg.com/vi/{{ p.video }}/hqdefault.jpg" loading="lazy"
       alt="YouTube thumbnail of {{ p.video }} (not analysed)">
  {% else %}
  <img class="sheet" src="sheet/{{ p.video }}.jpg" loading="lazy" alt="frames from {{ p.video }}">
  {% set route = p.route or routes.get(p.video) %}
  {% if route %}{% set times = {} %}{% for e in p.entries if e.locality %}{% set _ = times.setdefault(
       e.locality[0], []).extend(e.segments | rejectattr('decision', 'equalto', 'reject')) %}{% endfor %}
  <div class="routemap" data-route='{{ route | tojson }}' data-times='{{ times | tojson }}'
       title="{{ 'the car route the localities were estimated from' if p.route else
                 'the localities in the order the video passes them' }}"></div>{% endif %}
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
        {% if not e.segments %}<button class="drop" data-video="{{ p.video }}" data-entry="{{ ei }}"
          onclick="dropGroup(this)" title="this group has no segments: remove it">✕ remove</button>{% endif %}
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
          <button class="small no" onclick="decide(this, 'reject', true)">Reject</button>{% endif %}</td>
      <td>{% if not s.applied %}
          <input class="at" placeholder="m:ss"
                 autocomplete="off" title="time to split at, e.g. where the drive enters another locality">
          <button class="small" onclick="splitAt(this)">Split</button>
          <button class="small" onclick="moveSegment(this)" title="put this segment under another locality">
          → other locality</button>{% endif %}</td>
    </tr>
    {% if not loop.last %}{% set n = e.segments[loop.index] %}
    {% set split = n.start - s.end <= max_merge_gap and n.night == s.night %}
    <tr class="split" data-video="{{ p.video }}" data-start="{{ s.end }}" data-end="{{ n.start }}">
      <td colspan="2">{% if split %}suggested split at {{ '%d:%02d' % (s.end // 60, s.end % 60) }}
        {% elif n.start > s.end %}left out {{ '%d:%02d' % (s.end // 60, s.end % 60) }}–{{
          '%d:%02d' % (n.start // 60, n.start % 60) }}{% else %}{% endif %}
        {%- set why = gap_reason(p.note, s.end, n.start, n.night != s.night) %}{% if why %}: {{ why }}{% endif %}</td>
      <td><button class="small" onclick="play('{{ p.video }}', {{ s.end }})">▶ {{
          'split' if split else 'gap' }}</button></td>
      <td>{% if not s.applied and not n.applied %}<button class="small"
          onclick="mergeSplit(this, '{{ p.video }}', {{ ei }}, {{ loop.index0 }})"{% if not split %}
          title="one segment from {{ '%d:%02d' % (s.start // 60, s.start % 60) }} to {{
          '%d:%02d' % (n.end // 60, n.end % 60) }}, the left-out part and the first one's day/night included"
          {% endif %}>{{ 'Remove split' if split else 'Join' }}</button>{% endif %}
          {%- if loop.index0 == 0 and e.segments | length > 2 and not e.segments | selectattr('applied') | list %}
          <button class="small" onclick="mergeAll(this, '{{ p.video }}', {{ ei }}, {{ e.segments | length }})"
          title="one segment of all {{ e.segments | length }} of this locality, the left-out parts included">Join all
          </button>{% endif %}</td>
    </tr>
    {% endif %}
  {% endfor %}
    <tr class="addrow"><td colspan="5">
      <button class="small newseg" data-video="{{ p.video }}" data-entry="{{ ei }}" onclick="newSegment(this)"
              title="from the current video time (or after the last segment) to the next segment or the end">
        + new segment{{ ' in ' ~ e.locality[0] if e.locality }}</button></td></tr>
  {% endfor %}
  </table>
  <button class="small" data-video="{{ p.video }}" onclick="addLocality(this)"
          title="an empty locality group: pick its locality, then add segments to it with + new segment">
    + add a locality</button>
  <button class="small go approve-rest" onclick="approveRest(this)"
          title="approve every undecided segment of this video">Approve rest</button>
  <button class="small stop" onclick="rejectRest(this)"
          title="reject every undecided segment of this video">Reject rest</button>
  <button class="small stop" onclick="rejectVideo(this)"
          title="reject every segment of this video">Reject video</button>
  <button class="small" data-video="{{ p.video }}" onclick="markSpedUp(this)"
          title="the footage plays faster than real time: rejects this video; two such videos reject the channel"
          >{{ '⏩ Sped up ✓' if p.sped_up else '⏩ Sped up' }}</button>
  <button class="small" data-video="{{ p.video }}" onclick="markGhosts(this)"
          title="scenes joined with a dissolve (people or cars fade out mid-picture): rejects this video; two such
                 videos reject the channel">{{ '👻 Ghosts ✓' if p.ghosts else '👻 Ghosts' }}</button>
  {% for t in (p.removed_splits or []) | unique | sort %}<button class="small" data-video="{{ p.video }}"
    data-at="{{ t }}" onclick="putSplitBack(this)" title="split the segment that contains {{ '%d:%02d' % (t // 60,
    t % 60) }} there again (a split removed earlier)">↺ split at {{ '%d:%02d' % (t // 60, t % 60) }}</button>
  {% endfor %}
  <button class="small" data-video="{{ p.video }}" onclick="undoRedo(this, 'undo')"{{ ' disabled' if not p.undo }}
          title="undo the last change to this video only (splits, joins, decisions, localities, times)">↶ Undo{{
          ' (%d)' % p.undo|length if p.undo }}</button>
  <button class="small" data-video="{{ p.video }}" onclick="undoRedo(this, 'redo')"{{ ' disabled' if not p.redo }}
          title="redo the change to this video just undone">↷ Redo{{ ' (%d)' % p.redo|length if p.redo }}</button>
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
{% elif more %}<p class="more">{{ more }} more videos to decide:
  they appear here as you finish these (reload the page).</p>{% endif %}
<p class="more">{% if show_done %}<a href="?">hide the videos already applied</a>{% elif applied %}
  <a href="?done=1">show the {{ applied }} videos already applied</a> (to correct one){% endif %}</p>
<details><summary>Excluded videos ({{ plan | selectattr('exclude') | list | length }})</summary>
{% for p in plan if p.exclude %}
<div class="excluded" data-video="{{ p.video }}"><a href="https://www.youtube.com/watch?v={{ p.video }}"
  target="_blank">{{ p.video }}</a> {{ p.title }} — <span class="why">{{ p.exclude }}</span></div>
{% endfor %}
</details>
</main>
<script>
let ytPlayer = null, ytReady = false, pending = null;
// a time in an exclusion reason or a proposal's note ("skips footage at 4:53", "highway driving 0-1856s", "camera
// not facing forward 48-154s") plays the video from there (notes: user, 2026-10-16); each text piece on its own,
// so the note's red warnings stay
document.querySelectorAll('.excluded .why, .video .note').forEach(el => {
  const times = /\\b(\\d+):(\\d\\d)(?::(\\d\\d))?\\b|\\b(\\d+)(?:-\\d+)?s\\b/g;  // m:ss, h:mm:ss, 123s, 123-456s
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(node => {
    const text = node.textContent, frag = document.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(times)) {
      const [all, a, b, c, sec] = m;
      const t = sec !== undefined ? +sec : c !== undefined ? +a * 3600 + +b * 60 + +c : +a * 60 + +b;
      frag.append(text.slice(last, m.index));
      const link = Object.assign(document.createElement('a'), { href: '#', className: 'at', textContent: all });
      link.dataset.t = t;
      frag.append(link);
      last = m.index + all.length;
    }
    if (!last) return;
    frag.append(text.slice(last));
    node.replaceWith(frag);
  });
});
document.addEventListener('click', e => {
  const a = e.target.closest('.excluded a.at, .video .note a.at');
  if (!a) return;
  e.preventDefault();
  play(a.closest('[data-video]').dataset.video, Math.max(+a.dataset.t - 3, 0));  // 3 s before, as ▶ end (user)
});
function onYouTubeIframeAPIReady() {
  ytPlayer = new YT.Player('player', { width: 480, height: 270, playerVars: { autoplay: 1, playsinline: 1, rel: 0 },
    events: { onReady: () => { ytReady = true; if (pending) play(...pending); } } });
}
// what play() asked for, until the player plays it: YouTube sometimes leaves a video it was switched to unstarted
// (state -1) at the right time (user, 2026-10-12: "the player gets stuck"; 2 of 3 quick switches in a test)
let want = null;
function play(video, t) {
  if (!ytReady) { pending = [video, t]; return; }
  pending = null;
  want = { video, t, at: Date.now(), tries: 0 };
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
  // the segment of the playing video that the current time is in, decided or not (the line follows the
  // playback); between segments the nearest undecided one, a decided one only when none is left
  const video = ytReady && playingVideo();
  if (!video) return null;
  const t = ytPlayer.getCurrentTime();
  const rows = [...document.querySelectorAll(`tr.seg:not(.applied)[data-video="${video}"]`)];
  const here = rows.find(r => +r.dataset.start <= t && t <= +r.dataset.end);
  if (here) return here;
  const open = rows.filter(r => !r.classList.contains('approve') && !r.classList.contains('reject'));
  let best = null, bestDistance = Infinity;
  for (const r of open.length ? open : rows) {
    const distance = Math.max(0, +r.dataset.start - t, t - +r.dataset.end);
    if (distance < bestDistance) { best = r; bestDistance = distance; }
  }
  return best;
}
// the keys act on the segment clicked last (any of its buttons or boxes) while its video plays, whatever part of
// the video is showing (user, 2026-10-11); else on the one the playback is in
let clickedRow = null;
document.addEventListener('click', e => { const r = e.target.closest('tr.seg'); if (r) clickedRow = r; }, true);
function keyRow() {
  // the row clicked last, until the playback reaches another segment of the video: then the outline follows it
  // (user, 2026-10-14: XDBi8ukETEA's stayed on the first segment while the second played)
  const video = ytReady && playingVideo(), here = activeRow();
  if (clickedRow && document.contains(clickedRow) && clickedRow.dataset.video === video) {
    const t = ytPlayer.getCurrentTime();
    if (!(here && here !== clickedRow && +here.dataset.start <= t && t <= +here.dataset.end)) return clickedRow;
    clickedRow = null;
  }
  return here;
}
setInterval(() => {
  if (!ytReady) return;
  if (want) {  // a switch not playing where asked after 2 s: seek and start it; after 6 s: load it again (3 tries)
    const state = ytPlayer.getPlayerState(), waited = Date.now() - want.at;
    const right = playingVideo() === want.video
      && Math.abs(ytPlayer.getCurrentTime() - want.t) <= 3 + waited / 1000;  // there, or played on from there
    if (state === YT.PlayerState.PLAYING && right) want = null;
    else if (want.tries >= 3) want = null;
    else if (waited > 6000) {
      ytPlayer.loadVideoById({ videoId: want.video, startSeconds: Math.max(0, Math.floor(want.t)) });
      want = { ...want, at: Date.now(), tries: want.tries + 1, nudged: false };
    } else if (waited > 2000 && !want.nudged && state !== YT.PlayerState.BUFFERING) {
      // a quick later switch can be lost while the one before still loads, and YouTube may leave it unstarted
      // or paused: once, go where asked and play
      if (playingVideo() === want.video) {
        ytPlayer.seekTo(Math.max(0, Math.floor(want.t)), true); ytPlayer.playVideo();
      }
      else ytPlayer.loadVideoById({ videoId: want.video, startSeconds: Math.max(0, Math.floor(want.t)) });
      want.nudged = true;
    }
  }
  document.getElementById('clock').textContent = fmt(ytPlayer.getCurrentTime() || 0);
  const vid = playingVideo();
  if (vid && document.getElementById('vid').textContent !== vid) document.getElementById('vid').textContent = vid;
  if (vid) document.getElementById('vid-yt').href =
    `https://www.youtube.com/watch?v=${vid}&t=${Math.floor(ytPlayer.getCurrentTime() || 0)}s`;
  // playing a left-out stretch between two segments: the line moves to that gap, in orange
  const t = ytPlayer.getCurrentTime(), video = playingVideo();
  const gap = [...document.querySelectorAll(`tr.split[data-video="${video}"]`)]
    .find(r => +r.dataset.start < t && t < +r.dataset.end);
  document.querySelectorAll('tr.split.playing').forEach(r => { if (r !== gap) r.classList.remove('playing'); });
  if (gap) gap.classList.add('playing');
  const row = gap ? null : keyRow();
  document.querySelectorAll('tr.seg.active').forEach(r => { if (r !== row) r.classList.remove('active'); });
  if (row) row.classList.add('active');
}, 250);
function copyText(text) {
  // the Clipboard API exists only on https and localhost; served from mini over http, the old way
  if (navigator.clipboard) return navigator.clipboard.writeText(text).catch(() => {});
  const box = Object.assign(document.createElement('textarea'), { value: text });
  document.body.appendChild(box);
  box.select();
  document.execCommand('copy');
  box.remove();
}
document.getElementById('vid').onclick = async () => {
  await copyText(document.getElementById('vid').textContent);
  document.getElementById('vid-copied').textContent = 'copied';
  setTimeout(() => { document.getElementById('vid-copied').textContent = ''; }, 1000);
};
document.getElementById('clock').onclick = async () => {
  await copyText(document.getElementById('clock').textContent);
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
  if (!['a', 's', 'd', 'q', 'w', 'f', 'c'].includes(key)) return;
  if (key === 'c') {  // the current time to the clipboard, as a click on it does (user, 2026-10-16)
    e.preventDefault();
    document.getElementById('clock').click();
    return;
  }
  const row = keyRow();
  if (!row) return;
  e.preventDefault();
  const t = Math.floor(ytPlayer.getCurrentTime());
  if (key === 'f') {  // current time into the Split box of the segment it falls in, and to the clipboard
    row.querySelector('input.at').value = fmt(t);
    copyText(fmt(t));
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
function nextOpen(row, otherVideo = false) {
  // next undecided segment: this video's first, also those above (unless otherVideo; the page moved on to the
  // next video while one above in this video was still open: user, 2026-10-16), then the following videos, then
  // from the top
  const mine = r => r.dataset.video === row.dataset.video;
  const open = r => !r.classList.contains('approve') && !r.classList.contains('reject') && !(otherVideo && mine(r));
  const rows = [...document.querySelectorAll('tr.seg:not(.applied)')];
  const i = rows.indexOf(row);
  const after = rows.slice(i + 1), before = rows.slice(0, Math.max(i, 0));
  return (!otherVideo && [...after, ...before].find(r => mine(r) && open(r))) || after.find(open) || before.find(open);
}
function goToNextOpen(row) {
  const next = nextOpen(row);
  if (!next) return;
  clickedRow = next;  // the keys go on with it
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
  updateApproveRest(box.closest('.video'));
  // the group's "+ new segment in …" names the locality just set, not the one the page was loaded with
  const add = document.querySelector(
    `button.newseg[data-video="${box.dataset.video}"][data-entry="${box.dataset.entry}"]`);
  if (add) add.textContent = `+ new segment in ${locality}`;
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
  // m:ss, h:mm:ss, seconds (754), YouTube's 754s and a link with t=754 or t=12m34s (user, 2026-10-16: videos that
  // only play on YouTube itself give the time in seconds)
  text = text.trim();
  const link = text.match(/[?&#]t=(?:(\\d+)h)?(?:(\\d+)m)?(\\d+)s?\\b/);
  if (link) return (+link[1] || 0) * 3600 + (+link[2] || 0) * 60 + +link[3];
  text = text.replace(/s$/, '');
  const parts = text.split(':').map(Number);
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
async function putSplitBack(btn) {
  // the segment of this video that now holds the moment a split was removed at
  const at = +btn.dataset.at;
  const row = [...document.querySelectorAll(`tr.seg[data-video="${btn.dataset.video}"]`)]
    .find(r => +r.dataset.start < at && at < +r.dataset.end && !r.classList.contains('applied'));
  if (!row) { alert(`No open segment of this video contains ${fmt(at)}.`); return; }
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
// all segments of a locality as one (user, 2026-10-16), after a confirmation
async function mergeAll(btn, video, entry, count) {
  const rows = [...Array(count).keys()].map(i =>
    document.querySelector(`tr.seg[data-video="${video}"][data-entry="${entry}"][data-seg="${i}"]`));
  const bounds = rows.map(r => [+r.dataset.start, +r.dataset.end]);
  const t = s => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  const a = Math.min(...bounds.map(b => b[0])), b = Math.max(...bounds.map(b => b[1]));
  if (!confirm(`Join all ${count} segments of this locality into one, ${t(a)}–${t(b)}? `
               + 'The left-out parts between them are included and their decisions cleared.')) return;
  btn.disabled = true;
  const res = await fetch('merge_all', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video, entry, seg: 0, bounds }) });
  if (!res.ok) { btn.disabled = false; alert((await res.json()).error); return; }
  reloadKeepingPlace();
}
// approving needs a locality: without one the approved segments could not be added to the mapping
function updateApproveRest(video) {
  const btn = video && video.querySelector('.approve-rest');
  if (!btn) return;
  const missing = [...video.querySelectorAll('.locbox')].some(b =>
    !b.querySelector('.loc-name').value.trim() || !b.querySelector('.loc-country').value);
  btn.disabled = missing;
  btn.title = missing ? 'set the locality (and its country) first' : '';
}
document.querySelectorAll('.video').forEach(updateApproveRest);
async function approveRest(btn) {
  const video = btn.closest('.video');
  for (const row of video.querySelectorAll('tr.seg:not(.approve):not(.reject):not(.applied)')) {
    if (!await decide(row.querySelector('.yes'), 'approve')) return;
  }
  const last = [...video.querySelectorAll('tr.seg')].pop();
  if (last) goToNextOpen(last);
}
async function rejectRest(btn) {
  const video = btn.closest('.video');
  for (const row of video.querySelectorAll('tr.seg:not(.approve):not(.reject):not(.applied)')) {
    if (!await decide(row.querySelector('.no'), 'reject')) return;
  }
  const last = [...video.querySelectorAll('tr.seg')].pop();
  if (last) goToNextOpen(last);
}
async function rejectChannel() {
  if (!confirm('Reject the whole channel? Every undecided segment in this batch is rejected, the fetcher stops ' +
               'the channel, and the channels sheet will say Rejected.')) return;
  const res = await fetch('reject_channel', { method: 'POST' });
  if (!res.ok) { alert((await res.json()).error); return; }
  const out = await res.json();
  alert(`${out.rejected} segments rejected; the channel is rejected.`);
  location.reload();
}
async function dropGroup(btn) {
  const res = await fetch('drop_group', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: btn.dataset.video, entry: +btn.dataset.entry }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  reloadKeepingPlace();
}
async function addLocality(btn) {
  const res = await fetch('add_locality', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video: btn.dataset.video }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  reloadKeepingPlace();
}
async function markVideo(btn, what, label) {
  const res = await fetch('mark', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ video: btn.dataset.video, what: what }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  const out = await res.json();
  if (out.rejects_channel) alert(`${out.marked} videos of this channel are marked ${label}: the fetcher rejects ` +
                                 'the channel before its next batch, and the sheet will say Rejected.');
  // the marked video leaves the page: go on with the next video's first undecided segment
  const next = nextOpen(btn.closest('.video').querySelector('tr.seg') || btn, true);
  reloadKeepingPlace(next ? next.dataset.video : null, next ? +next.dataset.start : 0);
}
const markSpedUp = btn => markVideo(btn, 'sped_up', 'sped up');
const markGhosts = btn => markVideo(btn, 'ghosts', 'with ghosts');
async function undoRedo(btn, what) {
  const res = await fetch(what, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ video: btn.dataset.video }) });
  if (!res.ok) { alert((await res.json()).error); return; }
  // stay on this video (at the player's moment if it is the one playing), not on another one in the player
  const v = btn.dataset.video;
  reloadKeepingPlace(v, ytReady && playingVideo() === v ? ytPlayer.getCurrentTime() : 0);
}
async function rejectVideo(btn) {
  const video = btn.closest('.video');
  const rows = [...video.querySelectorAll('tr.seg:not(.reject):not(.applied)')];
  if (rows.length) {  // one request for all: one per segment rewrote the plan as many times (slow on long videos)
    const res = await fetch('reject_segments', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments: rows.map(r => ({ video: r.dataset.video, entry: +r.dataset.entry,
        seg: +r.dataset.seg, bounds: [[+r.dataset.start, +r.dataset.end]] })) }) });
    if (!res.ok) { alert((await res.json()).error); return; }
    for (const row of rows) { row.classList.remove('approve'); row.classList.add('reject'); }
    update();
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
  let res = null, out;
  try {
    res = await fetch('apply', { method: 'POST' });
    out = await res.json();
  } catch (e) {  // the answer got lost (server restarted, sleep): what was added is marked applied in the plan
    out = { error: `No answer from the page server (${e.message}). Reload the page: segments already added ` +
                   'show as applied, the others can be applied again.' };
  }
  document.getElementById('log').textContent = out.error || out.log.join('\\n');
  btn.disabled = false; btn.textContent = 'Apply approved segments to the mapping';
  if (res && res.ok && !out.error) setTimeout(() => location.reload(), 4000);
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


def localities_route(p, plan_dir, mapping):
    """{'path', 'stops'} for the review page's map of a video with several localities and no car route (split by
    its chapters or a GPS overlay; user, 2026-10-11): the GPS track if the picture has one, else straight lines
    between the towns; a stop per locality where its first segment starts. None for fewer than two placed stops."""
    entries = sorted((e for e in p.get('entries', []) if e.get('locality') and e['segments']),
                     key=lambda e: min(s['start'] for s in e['segments']))
    if len({tuple(e['locality']) for e in entries}) < 2:
        return None
    sig_path = os.path.join(plan_dir, f"{p['video']}_signals.npz")
    gps = np.zeros((0, 3))
    if os.path.exists(sig_path):
        with np.load(sig_path) as z:
            gps = np.asarray(z['gps']).reshape(-1, 3) if 'gps' in z else gps
    stops, seen = [], set()
    for e in entries:
        name = e['locality'][0]
        if name in seen:
            continue
        seen.add(name)
        start = min(s['start'] for s in e['segments'])
        if len(gps):
            _, lat, lon = gps[np.argmin(np.abs(gps[:, 0] - start))]
        else:
            _, row = add_video.get_existing_locality_row(mapping, *e['locality'])
            if row is None:
                continue
            lat, lon = float(row['lat']), float(row['lon'])
        stops.append([float(lat), float(lon), name])
    if len(stops) < 2:
        return None
    path = gps[::max(1, len(gps) // 300), 1:].tolist() if len(gps) >= 2 else [st[:2] for st in stops]
    return {'path': path, 'stops': stops}


def review(plan_path, port=None):
    """Local page to approve or reject every proposed segment, then apply only the approved ones."""
    from flask import Flask, jsonify, render_template_string, request, send_from_directory
    from markupsafe import Markup, escape

    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    app = Flask(__name__)
    app.jinja_env.globals["note_html"] = note_html  # the note, its warnings in red
    app.jinja_env.globals["gap_reason"] = gap_reason  # why two segments are apart, on the split rows
    channel_cache = {}

    def channel_link(plan):
        """{name, url} of the batch's YouTube channel, shown above the clock (user, 2026-10-13): its title from the
        YouTube Data API, once per page server, else the folder's name."""
        if not channel_cache:
            cid = next((p['channel_id'] for p in plan if p.get('channel_id')), None)
            name = os.path.basename(plan_dir)
            if cid:
                try:
                    items = _api('channels', part='snippet', id=cid).get('items', [])
                    snippet = items[0]['snippet'] if items else {}
                    name = snippet.get('title') or name
                    handle = snippet.get('customUrl')
                    url = f'https://www.youtube.com/{handle}' if handle else f'https://www.youtube.com/channel/{cid}'
                except (RuntimeError, KeyError, IndexError, requests.RequestException):
                    url = f'https://www.youtube.com/channel/{cid}'
                channel_cache.update(name=name, url=url)
        return channel_cache or None

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

    # undo and redo per video, kept in the plan itself (p['undo'], p['redo']: the video's entry before each change,
    # newest last), so they outlast a restarted review page and are the same in every tab (user, 2026-10-10: an undo
    # held only in one page's memory looked like it changed other videos)
    UNDO_KEEP = 10
    before = {}

    def bare(p):
        return {k: v for k, v in p.items() if k not in ('undo', 'redo')}

    @app.before_request
    def take_lock():
        if request.method == 'POST':
            key = id(request._get_current_object())
            held[key] = lock = plan_lock(plan_path)
            lock.__enter__()
            video = (request.get_json(silent=True) or {}).get('video')
            if video and request.path not in ('/apply', '/undo', '/redo'):
                p = next((p for p in load() if p['video'] == video), None)
                if p is not None:
                    before[key] = (video, json.dumps(bare(p)))

    @app.after_request
    def remember(response):
        change = before.pop(id(request._get_current_object()), None)
        if change and response.status_code < 400:  # still under the plan lock: teardown releases it
            plan = load()
            p = next((p for p in plan if p['video'] == change[0]), None)
            if p is not None and json.dumps(bare(p)) != change[1]:
                p['undo'] = (p.get('undo', []) + [change[1]])[-UNDO_KEEP:]
                p.pop('redo', None)
                save(plan)
        return response

    @app.teardown_request
    def release_lock(exc):
        before.pop(id(request._get_current_object()), None)  # a request that failed leaves no change behind
        lock = held.pop(id(request._get_current_object()), None)
        if lock:
            lock.__exit__(None, None, None)

    def step(back, forward):
        """Put a video back to the state on top of its back stack ('undo' or 'redo'), its current one going on
        forward."""
        video = request.get_json()['video']
        plan = load()
        i = next((i for i, p in enumerate(plan) if p['video'] == video), None)
        if i is None:
            return jsonify(error=stale), 409
        p = plan[i]
        if not p.get(back):
            return jsonify(error=f'nothing to {back} for this video'), 400
        if any(s.get('applied') for e in p.get('entries', []) for s in e['segments']):
            return jsonify(error='this video is already (partly) in the mapping: change it there'), 409
        stacks = {back: p[back][:-1], forward: (p.get(forward, []) + [json.dumps(bare(p))])[-UNDO_KEEP:]}
        plan[i] = {**json.loads(p[back][-1]), **{k: v for k, v in stacks.items() if v}}
        save(plan)
        return jsonify(ok=True, undo=len(plan[i].get('undo', [])), redo=len(plan[i].get('redo', [])))

    @app.route('/undo', methods=['POST'])
    def undo():
        return step('undo', 'redo')

    @app.route('/redo', methods=['POST'])
    def redo():
        return step('redo', 'undo')

    @app.route('/')
    def index():
        # the first REVIEW_PAGE videos with something left to decide; finished ones stay in the plan but off the
        # page, and make room for the next on reload
        plan = load()
        segs = {p['video']: [s for e in p['entries'] for s in e['segments']] for p in plan if 'entries' in p}
        waiting = [v for v, ss in segs.items() if any(not s.get('decision') and not s.get('applied') for s in ss)]
        open_videos = set(waiting[:REVIEW_PAGE])
        # approved but held back for want of a locality: shown until it is set, or Apply can never add them
        open_videos |= {p['video'] for p in plan for e in p.get('entries', []) if not e.get('locality')
                        and any(s.get('decision') == 'approve' and not s.get('applied') for s in e['segments'])}
        # a locality group just added, with no segment yet: the reviewer is adding to it (a video whose segments were
        # all decided vanished on "+ add a locality": 5f3r-Q3Wtrc, vx4nUURTgHA; user, 2026-10-13)
        open_videos |= {p['video'] for p in plan for e in p.get('entries', []) if not e['segments']}
        # decided and not applied yet: stays, to be corrected before Apply (user, 2026-10-14: finished videos vanished
        # on the reload a merge makes); ?done=1 shows the applied ones too
        # (a rejection until the next Apply marks it done: it is never applied, and would stay for ever)

        def pending(s):
            return s.get('decision') == 'approve' and not s.get('applied') or \
                s.get('decision') == 'reject' and not s.get('done')
        open_videos |= {v for v, ss in segs.items() if any(pending(s) for s in ss)}
        applied = {v for v, ss in segs.items() if any(s.get('applied') for s in ss)}
        if request.args.get('done'):
            open_videos |= applied
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
                                      channel=channel_link(plan),
                                      video_country=video_country,
                                      state_countries=sorted(c for c, has in mapping.groupby('country')['state']
                                                             .agg(lambda x: x.notna().mean() > 0.5).items() if has),
                                      new_localities=new_localities, waiting=len(waiting),
                                      more=len(set(waiting) - open_videos), show_done=bool(request.args.get('done')),
                                      applied=len(applied - (open_videos if not request.args.get('done') else set())),
                                      durations={v: video_duration(v) for v in open_videos},
                                      routes={p['video']: r for p in plan if p['video'] in open_videos
                                              and not p.get('route')
                                              and (r := localities_route(p, plan_dir, mapping))},
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
        if not s['start'] < at < s['end'] - 1:
            return jsonify(error='the split time must be inside the segment'), 400
        # the second part from the next second: the two do not share a second (user, 2026-10-16)
        segs[d['seg']:d['seg'] + 1] = [{'start': s['start'], 'end': at, 'night': s['night']},
                                       {'start': at + 1, 'end': s['end'], 'night': s['night']}]
        if at in p.get('removed_splits', []):  # put back: no false cut after all
            p['removed_splits'] = [t for t in p['removed_splits'] if t != at]
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

    @app.route('/drop_group', methods=['POST'])
    def drop_group():
        """Remove a locality group with no segments (added by + add a locality and not used)."""
        d = request.get_json()
        plan = load()
        p = next((p for p in plan if p['video'] == d['video'] and 'entries' in p), None)
        if p is None or not 0 <= d['entry'] < len(p['entries']) or p['entries'][d['entry']]['segments']:
            return jsonify(error=stale), 409
        del p['entries'][d['entry']]
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
        suggested = e.get('locality') if e.get('guessed') else None
        locality, state, country = (str(x or '').strip() for x in d['locality'])
        state = state or None
        if not locality or country not in COUNTRIES:
            return jsonify(error='give a locality and pick a country'), 400
        rows = add_video.load_csv(add_video.FILE_PATH)
        rows = rows[rows['country'] == country]
        if not state and len(rows) and rows['state'].notna().mean() > 0.5:
            # without it Apply would make a second Northampton next to Northampton, MA
            return jsonify(error=f'give the state too: the mapping writes one for {country} (e.g. PA)'), 400
        if suggested and suggested != [locality, state, country]:
            try:  # a bigger city nearby for the town suggested: learnt for later suggestions
                record_merge(suggested, [locality, state, country], p['video'])
            except (OSError, ValueError, KeyError, TypeError):
                pass
        same = [o for o in p['entries'] if o is not e and o.get('locality') == [locality, state, country]]
        if same:  # this video already has a group for that locality: join it, overlapping segments as one
            joined = []
            for s in sorted(same[0]['segments'] + e['segments'], key=lambda s: s['start']):
                prev = joined[-1] if joined else None
                if (prev and s['start'] < prev['end'] and not prev.get('applied') and not s.get('applied')
                        and prev.get('decision') == s.get('decision')):
                    prev['end'] = max(prev['end'], s['end'])  # 40:19-46:27 inside 0:27-64:20 (rQ_b0lwCwCA)
                else:
                    joined.append(s)
            same[0]['segments'] = joined
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

    @app.route('/reject_channel', methods=['POST'])
    def reject_channel():
        """Reject every undecided segment of the batch and mark the channel rejected: the fetcher stops it (here,
        or on a worker after the hourly sync) and the sheet says Rejected."""
        plan = load()
        n = 0
        for p in plan:
            for e in p.get('entries', []):
                for seg in e['segments']:
                    if not seg.get('decision') and not seg.get('applied'):
                        seg['decision'] = 'reject'
                        n += 1
        save(plan)
        with open(os.path.join(plan_dir, CHANNEL_REJECTED), 'w') as f:
            f.write(f'rejected in review on {time.strftime("%Y-%m-%d")}')
        return jsonify(ok=True, rejected=n)

    @app.route('/mark', methods=['POST'])
    def mark():
        """Mark a video sped up or with ghosts (REVIEW_MARKS) and reject what is left of it; two such videos
        reject the channel (the fetcher checks before each batch)."""
        d = request.get_json()
        if d.get('what') not in REVIEW_MARKS:
            return jsonify(error='unknown mark'), 400
        plan = load()
        p = next((p for p in plan if p['video'] == d['video']), None)
        if p is None:
            return jsonify(error=stale), 409
        p[d['what']] = True
        for e in p.get('entries', []):
            for seg in e['segments']:
                if not seg.get('applied'):
                    seg['decision'] = 'reject'
        save(plan)
        n = sum(bool(q.get(d['what'])) for q in plan)
        return jsonify(ok=True, marked=n, rejects_channel=n >= MARKED_REJECT)

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

    @app.route('/merge_all', methods=['POST'])
    def merge_all():
        """All segments of one locality of a video as one (Join all)."""
        d = request.get_json()
        plan = load()
        p, segs = lookup(plan, d)
        if p is None or len(d['bounds']) != len(segs):
            return jsonify(error=stale), 409
        if any(s.get('applied') for s in segs):
            return jsonify(error='one of these segments is already in the mapping'), 409
        for a, b in zip(segs, segs[1:]):
            if 0 <= b['start'] - a['end'] <= MAX_MERGE_GAP_S and a['night'] == b['night']:
                p.setdefault('removed_splits', []).append(a['end'])  # false cuts, kept for tuning the detector
        segs[:] = [{'start': min(s['start'] for s in segs), 'end': max(s['end'] for s in segs),
                    'night': segs[0]['night']}]
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

    @app.route('/reject_segments', methods=['POST'])
    def reject_segments():
        """Reject several segments (Reject video) with one load and save of the plan, not one per segment."""
        plan = load()
        found = [(*lookup(plan, d), d) for d in request.get_json()['segments']]
        if any(p is None for p, _, _ in found):
            return jsonify(error=stale), 409
        for p, segs, d in found:
            if not segs[d['seg']].get('applied'):
                segs[d['seg']]['decision'] = 'reject'
        save(plan)
        return jsonify(ok=True)

    @app.route('/apply', methods=['POST'])
    def apply():
        log = []
        try:
            _apply(load(), log)
        except Exception as ex:  # an answer either way, or the page waits for ever
            log.append(f'stopped: {ex}')
            return jsonify(log=log, error='\n'.join(log)), 500
        # a group added and never used would keep its video on the page with nothing to decide (5f3r-Q3Wtrc);
        # and the rejections are done: off the page from now on, as the applied ones
        plan = load()
        for p in plan:
            if 'entries' in p:
                p['entries'] = [e for e in p['entries'] if e['segments']]
                for e in p['entries']:
                    for seg in e['segments']:
                        if seg.get('decision') == 'reject':
                            seg['done'] = True
        save(plan)
        return jsonify(log=log or ['nothing to add'])

    def _apply(plan, log):
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

    port = port or 5000 + random.randint(0, 999)
    Timer(1.25, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    app.run(host=os.environ.get('REVIEW_HOST', '127.0.0.1'), port=port, debug=False)  # see review_hub.py


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--channel', metavar='URL', help='propose segments for every new video of a channel')
    ap.add_argument('--country', help='country of the channel as written in the mapping, for locality guesses')
    ap.add_argument('--out', help='output folder (default: _output/proposals/<channel>)')
    ap.add_argument('--pause', type=float, default=15, help='seconds between downloads')
    ap.add_argument('--limit', type=int, help='stop after this many new proposals (one batch to review)')
    ap.add_argument('--new-only', action='store_true',
                    help='only uploads newer than the channel\'s newest video in the mapping')
    ap.add_argument('--no-download', action='store_true',
                    help='propose whole videos from YouTube Data API metadata, without downloading or analysing')
    ap.add_argument('--deferred', nargs='?', const='multi_city',
                    help='propose the drives through several localities that --single-city held back, or with "us" '
                         'the videos in the US of a channel elsewhere')
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
        name = channel_folder(args.channel)
        out = process_channel(args.channel, args.country, args.out or os.path.join('_output/proposals', name),
                              args.pause, args.limit, download=not args.no_download, single_city=args.single_city,
                              new_only=args.new_only, deferred=args.deferred)
        if out:
            with open(out) as f:
                if any('entries' in p for p in json.load(f)):
                    print(f'Review with: python propose_segments.py --review {out}')
    else:
        ap.print_help()
