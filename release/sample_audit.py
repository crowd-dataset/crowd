"""Stratified random samples for the manual audits of CROWD (strata: continent x time-of-day label).

    python -m release.sample_audit [mapping.csv] [output folder] [--videos FOLDER ...] [--bbox FOLDER ...]

Writes three CSV files for people to fill in, from the segments of the mapping (seed 2026, so they can be redrawn):

    audit_labels.csv      segments to re-watch: are the start and end right, and the time-of-day and platform labels
                          (columns to fill in: boundary_ok, correct_start_s, correct_end_s, tod_audit, platform_audit)
    audit_frames.csv      frames on which to check the detector's boxes and mark the objects it missed (person,
                          bicycle, car, motorcycle, bus, truck, traffic light, stop sign), for
                          release.evaluate_detections; at most one frame per upload
    audit_clips.csv       10 s clips to watch with the dataset's tracks drawn over them, counting the tracking errors
                          (missed objects, false tracks, class confusions, identity switches, fragmentations); at most
                          one clip per upload

The samples are worked on with release.annotate, a browser tool that saves the results in the files that
release.audit_stats and release.evaluate_detections read.

Run on the machine that has the videos and the detection files, with their folders, the frames and clips are drawn
only from segments whose video and detection file are there. If the files already exist, the samples that can be worked
on here are kept and only the others are replaced, with new ids (an old sample's answers can never attach to a new
one); the label sample is made once and kept, since it needs only YouTube.

The contributor column is who added the video (analysis_data/video_contributors.csv): the auditor should not be
that person. Frames and clips are drawn with a probability proportional to the processed footage, so the samples
describe the footage and not the segments.
"""
import argparse
import csv
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: utils, release
from release import make_crowd_jsonl  # noqa: E402

SEED = 2026
LABEL_PER_STRATUM, FRAMES_PER_STRATUM, CLIPS_PER_STRATUM = 25, 40, 5
CLIP_S = 10
EDGE_S = 1  # no frame within a second of a segment boundary


def contributors(path="analysis_data/video_contributors.csv"):
    """video id -> who added it ('' if unknown)."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["video"]: r["contributor"] for r in csv.DictReader(f)}


def segments(mapping_csv):
    """One dict per segment: ids, times, processed duration, continent, time of day and platform labels."""
    records, _ = make_crowd_jsonl.build_records(Path(mapping_csv), Path(os.devnull))
    who = contributors()
    out = []
    for r in records:
        start, end = r["start_time_s"], r["end_time_s"]
        out.append({
            "segment_id": r["segment_id"], "video_id": r["video_id"], "start_s": start, "end_s": end,
            "processed_end_s": make_crowd_jsonl._processed_duration(start, end)[0],
            "duration_s": make_crowd_jsonl._processed_duration(start, end)[1],
            "continent": r["location"]["continent"], "locality": r["location"]["locality"],
            "iso3": r["location"]["iso3"], "tod": r["time_of_day_name"], "platform": r["vehicle_type_name"],
            "contributor": who.get(r["video_id"], ""),
        })
    return out


def by_stratum(segs):
    strata = defaultdict(list)
    for s in segs:
        strata[(s["continent"], s["tod"])].append(s)
    return dict(sorted(strata.items()))


def url(video, t):
    return f"https://www.youtube.com/watch?v={video}&t={int(t)}s"


def draw_segments(rng, pool, k, used=None):
    """k segments of distinct uploads, uniformly at random, none of an upload in `used` (which grows)."""
    if k <= 0:
        return []
    pool = pool[:]
    rng.shuffle(pool)
    chosen = []
    used = set() if used is None else used
    for s in pool:
        if s["video_id"] not in used:
            used.add(s["video_id"])
            chosen.append(s)
            if len(chosen) == k:
                break
    return chosen


def draw_times(rng, pool, k, length, used):
    """k (segment, offset) pairs with probability proportional to footage; one per upload, avoiding `used` uploads."""
    pool = [s for s in pool if s["duration_s"] > length + 2 * EDGE_S]
    weights = [s["duration_s"] for s in pool]
    out = []
    while len(out) < k and pool:
        s = rng.choices(pool, weights)[0]
        i = pool.index(s)
        pool.pop(i)
        weights.pop(i)
        if s["video_id"] in used:
            continue
        used.add(s["video_id"])
        offset = rng.uniform(EDGE_S, s["duration_s"] - length - EDGE_S)
        out.append((s, s["start_s"] + offset))
    return out


BBOX_NAME = re.compile(r"^(.+)_(\d+)_(\d+(?:\.\d+)?)\.csv$")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")


def bbox_keys(folders):
    """The (video id, start second) of the detection files in the folders (and in their bbox subfolders)."""
    keys = set()
    for d in folders:
        for folder in (Path(d).expanduser(), Path(d).expanduser() / "bbox"):
            if folder.is_dir():
                with os.scandir(folder) as entries:
                    for e in entries:
                        m = BBOX_NAME.match(e.name)
                        if m:
                            keys.add((m.group(1), int(m.group(2))))
    return keys


def video_ids(folders):
    """The ids of the videos in the folders."""
    found = set()
    for d in folders:
        folder = Path(d).expanduser()
        if folder.is_dir():
            with os.scandir(folder) as entries:
                found.update(Path(e.name).stem for e in entries if e.name.endswith(VIDEO_EXTENSIONS))
    return found


def availability(videos, bbox):
    """A function telling whether a sample (a dict with video_id and start_s) can be worked on with these folders:
    its video and its detection file are there. Without folders everything counts as available."""
    if not videos and not bbox:
        return lambda row: True
    keys, ids = bbox_keys(bbox), video_ids(videos)
    return lambda row: (not bbox or (row["video_id"], int(float(row["start_s"]))) in keys) and \
        (not videos or row["video_id"] in ids)


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


COMMON = ["continent", "tod", "segment_id", "video_id", "locality", "contributor"]


def draw_labels(rng, strata, out_dir, video_ok, keep_ids):
    """The sample for the label audit. Without folders it is made once and kept. With the folders of the videos
    (video_ok), the segments whose video is not there are replaced by segments whose video is (the downloaded file can
    be played instead of YouTube), except those that already have an answer (keep_ids)."""
    path = out_dir / "audit_labels.csv"
    old = list(csv.DictReader(open(path, newline="", encoding="utf-8"))) if path.exists() else []
    if path.exists() and video_ok is None:
        print(f"audit_labels: kept the existing {path}")
        return
    kept = []
    for key in {(r["continent"], r["tod"]) for r in old}:  # no more than LABEL_PER_STRATUM per stratum: answered first
        rs = [r for r in old if (r["continent"], r["tod"]) == key
              and (video_ok is None or video_ok(r) or r["segment_id"] in keep_ids)]
        rs.sort(key=lambda r: r["segment_id"] not in keep_ids)
        kept += rs[:max(LABEL_PER_STRATUM, sum(r["segment_id"] in keep_ids for r in rs))]
    used = {r["video_id"] for r in kept}
    rows, new, short = list(kept), 0, []
    for (continent, tod), pool in strata.items():
        have = sum(1 for r in kept if r["continent"] == continent and r["tod"] == tod)
        pool = [x for x in pool if video_ok is None or video_ok(x)]
        drawn = draw_segments(rng, pool, LABEL_PER_STRATUM - have, used)
        for s in drawn:
            rows.append({**{k: s[k] for k in COMMON}, "start_s": s["start_s"], "end_s": s["end_s"],
                         "platform": s["platform"], "url_start": url(s["video_id"], s["start_s"]),
                         "url_end": url(s["video_id"], max(s["end_s"] - 10, s["start_s"])),
                         "boundary_ok": "", "correct_start_s": "", "correct_end_s": "", "tod_audit": "",
                         "platform_audit": "", "notes": ""})
        new += len(drawn)
        if have + len(drawn) < LABEL_PER_STRATUM:
            short.append(f"{continent} {tod}: {have + len(drawn)} of {LABEL_PER_STRATUM}")
    rows.sort(key=lambda r: (r["continent"], r["tod"], r["segment_id"]))
    fields = [*COMMON, "start_s", "end_s", "platform", "url_start", "url_end", "boundary_ok", "correct_start_s",
              "correct_end_s", "tod_audit", "platform_audit", "notes"]
    write(path, fields, rows)
    print(f"audit_labels: {len(rows)} rows ({len(kept)} kept, {len(old) - len(kept)} dropped because their video is "
          f"not here, {new} drawn)" + (f"; fewer available than wanted: {'; '.join(short)}" if short else ""))


def draw_times_file(rng, strata, out_dir, name, per, length, extra, available, keep_ids=()):
    """Frames or clips: keep the existing samples that can be worked on here, and draw new ones, from the segments
    that can, to bring each stratum back to `per`. New ids continue after the highest existing one, so the answers
    saved for an old sample can never attach to a new one."""
    path = out_dir / f"{name}.csv"
    old = list(csv.DictReader(open(path, newline="", encoding="utf-8"))) if path.exists() else []
    kept = [r for r in old if available(r) or r["sample_id"] in keep_ids]  # an answered sample stays
    used = {r["video_id"] for r in kept}
    prefix = name[6:-1]
    rows, new, short = list(kept), 0, []
    for (continent, tod), pool in strata.items():
        stratum = f"{prefix}_{continent[:3].upper()}_{tod}_"
        have = [r for r in kept if r["sample_id"].startswith(stratum)]
        top = max([int(r["sample_id"][len(stratum):]) for r in old if r["sample_id"].startswith(stratum)], default=0)
        drawn = draw_times(rng, [x for x in pool if available(x)], per - len(have), length, used)
        for i, (s, t) in enumerate(drawn, top + 1):
            rows.append({"sample_id": f"{stratum}{i:03d}", **{k: s[k] for k in COMMON}, "start_s": s["start_s"],
                         "processed_end_s": s["processed_end_s"], extra: round(t, 3), "url": url(s["video_id"], t)})
        new += len(drawn)
        if len(have) + len(drawn) < per:
            short.append(f"{continent} {tod}: {len(have) + len(drawn)} of {per}")
    rows.sort(key=lambda r: r["sample_id"])
    write(path, ["sample_id", *COMMON, "start_s", "processed_end_s", extra, "url"], rows)
    dropped = len(old) - len(kept)
    print(f"{name}: {len(rows)} rows ({len(kept)} kept, {dropped} dropped because their video or detection file is "
          f"not here, {new} drawn)" + (f"; fewer available than wanted: {'; '.join(short)}" if short else ""))


def draw_samples(mapping_csv, out_dir, videos=(), bbox=(), frames_per=FRAMES_PER_STRATUM,
                 clips_per=CLIPS_PER_STRATUM, keep_ids=()):
    """Draw the three samples into out_dir. The frames and clips are refilled from the segments whose video and
    detection file are in the folders, and the labels from those whose video is (all segments if no folders are given);
    samples with an answer (keep_ids) are never dropped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    strata = by_stratum(segments(mapping_csv))
    available = availability(videos, bbox)
    keep_ids = set(keep_ids)
    draw_labels(rng, strata, out_dir, availability(videos, []) if videos else None, keep_ids)
    draw_times_file(rng, strata, out_dir, "audit_frames", frames_per, 0, "timestamp_s", available, keep_ids)
    draw_times_file(rng, strata, out_dir, "audit_clips", clips_per, CLIP_S, "clip_start_s", available, keep_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mapping", nargs="?", default="mapping.csv")
    p.add_argument("out", nargs="?", default="_output/audit")
    p.add_argument("--videos", nargs="*", default=[], help="folders with {video_id}.mp4")
    p.add_argument("--bbox", nargs="*", default=[], help="folders with the detection CSV files")
    a = p.parse_args()
    draw_samples(a.mapping, a.out, a.videos, a.bbox)


if __name__ == "__main__":
    main()
