"""Propose CROWD segments for a YouTube video from its pixels, for a curator to review.

Signals come from a low resolution copy decoded at a few frames per second:
camera motion (to trim stationary stretches), hard cuts and blank frames
(edited discontinuities), and sky brightness (day/night, split at transitions).
Vehicle type comes from the channel's history in the mapping file.
Nothing here writes to the mapping file.
"""
import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter

import cv2
import numpy as np

import add_video

FPS = 2                 # analysis frames per second
W, H = 160, 90          # analysis frame size

PARAMS = {
    'moving_flow': 0.25,      # median flow (px per analysis frame) above which the camera is moving
    'smooth_s': 5,            # rolling median window for motion
    'edge_stop_s': 15,        # stationary stretch at the start/end longer than this is trimmed
    'mid_stop_s': 180,        # stationary stretch inside a video longer than this is cut out
    'cut_ncc': 0.3,           # a frame this uncorrelated with the previous one is a hard cut...
    'cut_neighbour_ncc': 0.6,  # ...when the frames either side of it are this coherent
    'blank_std': 6.0,         # frame this flat is blank or a title card
    'night_sky': 95.0,        # sky brightness below this is night (calibrated on labelled segments)
    'night_window_s': 60,     # night/day must hold this long to count as a transition
    'min_segment_s': 30,      # drop proposed segments shorter than this
}


# pytubefix downloads now fail YouTube's PoToken check; a current yt-dlp (with node installed) does not
YT_DLP = os.environ.get('YT_DLP', 'yt-dlp')


def _url(video_id):
    return f'https://www.youtube.com/watch?v={video_id}'


def fetch_metadata(video_id):
    out = subprocess.run([YT_DLP, '-J', '--no-warnings', _url(video_id)],
                         capture_output=True, text=True, check=True)
    m = json.loads(out.stdout)
    d = m.get('upload_date')  # YYYYMMDD
    return {
        'id': video_id,
        'title': m.get('title'),
        'duration': m.get('duration'),
        'upload_date': f'{d[6:8]}{d[4:6]}{d[:4]}' if d else None,
        'channel_id': m.get('channel_id'),
        'description': m.get('description') or '',
        'chapters': [(c.get('title'), c.get('start_time')) for c in (m.get('chapters') or [])],
    }


def download_low_res(video_id, out_dir):
    template = os.path.join(out_dir, f'{video_id}_low.%(ext)s')
    subprocess.run([YT_DLP, '-q', '--no-warnings', '-f', 'bv*[height<=240][ext=mp4]/bv*[height<=360]/wv*',
                    '-o', template, _url(video_id)], check=True)
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


def signals(path):
    """Per analysis frame: camera motion, similarity to previous frame, flatness, sky brightness."""
    band = slice(int(H * 0.25), int(H * 0.70))   # road and buildings: no sky, no bonnet
    motion, ncc, std, sky = [], [], [], []
    prev = prev_thumb = None
    for f in frames(path):
        thumb = _normalised_thumb(f)
        if prev is None:
            motion.append(0.0)
            ncc.append(1.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, f, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mag = np.hypot(flow[band, :, 0], flow[band, :, 1])
            motion.append(float(np.median(mag)))  # median ignores other traffic moving past
            # correlation of brightness-normalised frames: lamps and exposure changes keep it high, cuts do not
            ncc.append(float((thumb * prev_thumb).mean()))
        std.append(float(f.std()))
        sky.append(float(f[: H // 3].mean()))
        prev, prev_thumb = f, thumb
    return {k: np.array(v) for k, v in
            (('motion', motion), ('ncc', ncc), ('std', std), ('sky', sky))}


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

    night = _rolling_median((sig['sky'] < p['night_sky']).astype(float), p['night_window_s'] * FPS) > 0.5

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


LOCALITY_FIELDS = ('lat', 'lon', 'gmp', 'population_locality', 'population_country', 'traffic_mortality',
                   'continent', 'locality_aka', 'literacy_rate', 'avg_height', 'med_age', 'gini', 'traffic_index')


def apply_segments(video_id, locality, state, country, segments, vehicle_type, upload_date, channel_id):
    """Add reviewed segments to an existing locality through the add-video form's own submit path,
    so its overlap and value checks apply. Writes the mapping file."""
    df = add_video.load_csv(add_video.FILE_PATH)
    idx, row = add_video.get_existing_locality_row(df, locality, state, country)
    if idx is None:
        raise ValueError(f'{locality}, {state}, {country} is not in the mapping; add its first video in the form')
    client = add_video.app.test_client()
    for seg in segments:
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
        row = add_video.load_csv(add_video.FILE_PATH).loc[idx]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video_ids', nargs='+')
    ap.add_argument('--out', default='_output/proposals')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for vid in args.video_ids:
        result = analyse_video(vid, args.out)
        vt, share = channel_vehicle_type(result['meta']['channel_id'])
        result['vehicle_type'] = {'value': vt, 'channel_share': round(share, 3)}
        with open(os.path.join(args.out, f'{vid}.json'), 'w') as f:
            json.dump(result, f)
        print(vid, result['meta']['title'], result['segments'], result['vehicle_type'])
