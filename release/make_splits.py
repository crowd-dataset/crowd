"""Canonical train / validation / test splits and balancing weights for CROWD (one row per segment of the index).

    python -m release.make_splits [mapping.csv] [crowd_splits.csv]

Two splits are given, both assigned by a hash of the group name, so a group always lands in the same split and the
assignment of existing segments does not change when videos are added:

    split_source    groups are source channels (an upload without a known channel is its own group), so no channel
                    and therefore no upload, route, camera or compression pipeline is in more than one split.
    split_locality  groups are localities, so a locality is never in more than one split (geographic generalisation).

70% of the groups go to "train", 15% to "val" and 15% to "test" (by hash, so the shares of segments and hours deviate
somewhat from these). The weights make a weighted random draw of segments balanced over the localities or countries:
weight_locality sums to 1 within each locality and weight_country to 1 within each country, in proportion to the
processed duration of the segment.
"""
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: utils, release
csv.field_size_limit(sys.maxsize)
from release import make_crowd_jsonl  # noqa: E402
from utils.core.parsing import parse_list, parse_videos  # noqa: E402

FIELDS = ["segment_id", "video_id", "channel_group", "locality_group", "iso3", "continent", "duration_s",
          "split_source", "split_locality", "weight_locality", "weight_country"]
BOUNDS = (("train", 70), ("val", 85), ("test", 100))  # upper bounds of the hash bucket (0-99)
NO_CHANNEL = {"", "none", "nan", "null"}


def bucket(group):
    """The hash bucket (0 to 99) of a group name; stable across runs, machines and Python versions."""
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100


def split_of(group):
    """'train', 'val' or 'test' for a group name."""
    b = bucket(group)
    return next(name for name, upper in BOUNDS if b < upper)


def channel_by_video(mapping_csv):
    """video id -> its channel id (the most frequent one if the rows differ), for videos with a known channel."""
    seen = defaultdict(Counter)
    with open(mapping_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            videos, channels = parse_videos(row["videos"]), parse_list(row["channel"])
            for video, channel in zip(videos, channels):
                if str(channel).strip().lower() not in NO_CHANNEL:
                    seen[video][str(channel).strip()] += 1
    return {video: counts.most_common(1)[0][0] for video, counts in seen.items()}


def build_rows(mapping_csv):
    """One dict per segment with its groups, splits and weights."""
    records, _ = make_crowd_jsonl.build_records(Path(mapping_csv), Path(os.devnull))
    channels = channel_by_video(mapping_csv)
    rows = []
    for r in records:
        start, end = r["start_time_s"], r["end_time_s"]
        loc = r["location"]
        channel = channels.get(r["video_id"])
        channel_group = f"channel:{channel}" if channel else f"video:{r['video_id']}"
        locality_group = f"{loc['iso3']}|{loc['locality']}|{loc.get('state') or ''}"
        rows.append({
            "segment_id": r["segment_id"], "video_id": r["video_id"], "channel_group": channel_group,
            "locality_group": locality_group, "iso3": loc["iso3"], "continent": loc["continent"],
            "duration_s": make_crowd_jsonl._processed_duration(start, end)[1],
            "split_source": split_of(channel_group), "split_locality": split_of(locality_group),
        })
    for key, weight in (("locality_group", "weight_locality"), ("iso3", "weight_country")):
        total = defaultdict(float)
        for r in rows:
            total[r[key]] += r["duration_s"]
        for r in rows:
            r[weight] = r["duration_s"] / total[r[key]] if total[r[key]] else 0.0
    return rows


def write_rows(rows, path):
    """Write the rows to path atomically."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(tmp, path)


def main():
    args = sys.argv[1:]
    mapping_csv = args[0] if args else "mapping.csv"
    out = args[1] if len(args) > 1 else "crowd_splits.csv"
    rows = build_rows(mapping_csv)
    write_rows(rows, out)
    print(f"Wrote {len(rows):,} segments to {out}")
    for split in ("split_source", "split_locality"):
        hours = Counter()
        count = Counter()
        for r in rows:
            hours[r[split]] += r["duration_s"] / 3600
            count[r[split]] += 1
        total = sum(hours.values())
        print(split, {k: f"{count[k]:,} segments, {hours[k]:,.0f} h ({100 * hours[k] / total:.1f}%)"
                      for k in sorted(hours)})


if __name__ == "__main__":
    main()
