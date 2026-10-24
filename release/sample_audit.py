"""Stratified random samples for the manual audits of CROWD (strata: continent x time-of-day label).

    python -m release.sample_audit [mapping.csv] [output folder]

Writes three CSV files for people to fill in, from the segments of the mapping (seed 2026, so they can be redrawn):

    audit_labels.csv      segments to re-watch: are the start and end right, and the time-of-day and platform labels
                          (columns to fill in: boundary_ok, correct_start_s, correct_end_s, tod_audit, platform_audit)
    audit_frames.csv      frames on which to check the detector's boxes and mark the objects it missed (person,
                          bicycle, car, motorcycle, bus, truck), for release.evaluate_detections; at most one frame
                          per upload
    audit_clips.csv       10 s clips to watch with the dataset's tracks drawn over them, counting the tracking errors
                          (missed objects, false tracks, class confusions, identity switches, fragmentations); at most
                          one clip per upload

The samples are worked on with release.annotate, a browser tool that saves the results in the files that
release.audit_stats and release.evaluate_detections read.

The contributor column is who added the video (analysis_data/video_contributors.csv): the auditor should not be
that person. Frames and clips are drawn with a probability proportional to the processed footage, so the samples
describe the footage and not the segments.
"""
import csv
import os
import random
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


def draw_segments(rng, pool, k):
    """k segments of distinct uploads, uniformly at random."""
    pool = pool[:]
    rng.shuffle(pool)
    chosen, used = [], set()
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


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    args = sys.argv[1:]
    mapping_csv = args[0] if args else "mapping.csv"
    out_dir = Path(args[1] if len(args) > 1 else "_output/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    strata = by_stratum(segments(mapping_csv))
    common = ["continent", "tod", "segment_id", "video_id", "locality", "contributor"]

    labels = []
    for (continent, tod), pool in strata.items():
        for s in draw_segments(rng, pool, LABEL_PER_STRATUM):
            labels.append({**{k: s[k] for k in common}, "start_s": s["start_s"], "end_s": s["end_s"],
                           "platform": s["platform"], "url_start": url(s["video_id"], s["start_s"]),
                           "url_end": url(s["video_id"], max(s["end_s"] - 10, s["start_s"])),
                           "boundary_ok": "", "correct_start_s": "", "correct_end_s": "", "tod_audit": "",
                           "platform_audit": "", "notes": ""})
    label_fields = [*common, "start_s", "end_s", "platform", "url_start", "url_end", "boundary_ok", "correct_start_s",
                    "correct_end_s", "tod_audit", "platform_audit", "notes"]
    write(out_dir / "audit_labels.csv", label_fields, labels)

    for name, per, length, extra in (("audit_frames", FRAMES_PER_STRATUM, 0, "timestamp_s"),
                                     ("audit_clips", CLIPS_PER_STRATUM, CLIP_S, "clip_start_s")):
        rows, used = [], set()
        for (continent, tod), pool in strata.items():
            for i, (s, t) in enumerate(draw_times(rng, pool, per, length, used), 1):
                rows.append({"sample_id": f"{name[6:-1]}_{continent[:3].upper()}_{tod}_{i:03d}",
                             **{k: s[k] for k in common}, "start_s": s["start_s"],
                             "processed_end_s": s["processed_end_s"], extra: round(t, 3),
                             "url": url(s["video_id"], t)})
        write(out_dir / f"{name}.csv", ["sample_id", *common, "start_s", "processed_end_s", extra, "url"], rows)
        print(f"{name}: {len(rows)} rows")
    print(f"audit_labels: {len(labels)} rows, written to {out_dir}")


if __name__ == "__main__":
    main()
