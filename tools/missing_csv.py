#!/usr/bin/env python3
"""Write mapping_remaining.csv: the mapping ("mapping" of the config) without the segments that already have a bbox
CSV, i.e. the segments still to process.

A segment counts as processed when a folder of "data" has bbox/{video_id}_{start_time}_{fps}.csv (main.py's layout;
CSVs right in the data folder count too). A video without segments left is dropped from its row, a row without
videos left is dropped.

    python -m tools.missing_csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from utils.core.parsing import parse_list, parse_videos  # noqa: E402

INPUT_CSV = common.get_configs("mapping")
OUTPUT_CSV = "mapping_remaining.csv"
BBOX_OUTPUT_DIRS = common.get_configs("data")  # the data folders; their bbox/ has the bbox CSVs
PRINT_REMOVED_CITIES = False  # set True to print which locality rows are dropped


def _int(x: Any) -> int:
    """x as an int, 0 where it is none."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def int_lists(cell: Any) -> List[List[int]]:
    """A nested list cell ('[[1,2],[3]]') as lists of ints; a bare number is a list of one, an inner list with a
    value that is not a number becomes []."""
    out: List[List[int]] = []
    for item in parse_list(cell):
        try:
            out.append([int(x) for x in (item if isinstance(item, list) else [item])])
        except (TypeError, ValueError):
            out.append([])
    return out


def normalize_len(lst: List[Any], n: int, pad: Any) -> List[Any]:
    """Pad or truncate list to length n."""
    return lst[:n] + [pad] * (n - len(lst))


def compact(values) -> str:
    """A list as the mapping writes it: [a,b] and [[0,10],[5]]."""
    return "[" + ",".join(compact(v) if isinstance(v, list) else str(v) for v in values) + "]"


def index_existing_bbox_segments(bbox_dirs: List[Path]) -> Set[Tuple[str, int]]:
    """(video_id, start_time) of each bbox CSV {video_id}_{start}_{fps}.csv in the data folders (and their bbox/).
    A video id may contain underscores, so the name is split from the right."""
    found: Set[Tuple[str, int]] = set()
    for d in bbox_dirs:
        for folder in (d / "bbox", d):
            if not folder.is_dir():
                continue
            for p in folder.glob("*.csv"):
                parts = p.stem.split("_")
                if len(parts) >= 3 and parts[-2].lstrip("-").isdigit():
                    found.add(("_".join(parts[:-2]), int(parts[-2])))
    return found


def build_remaining() -> None:
    """Write OUTPUT_CSV and print what was removed."""
    bbox_present = index_existing_bbox_segments([Path(p) for p in BBOX_OUTPUT_DIRS])
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    for c in ["videos", "start_time", "end_time", "time_of_day"] + (["locality"] if PRINT_REMOVED_CITIES else []):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    per_video = [c for c in ("vehicle_type", "upload_date", "channel") if c in df.columns]

    kept_rows: List[Dict[str, Any]] = []
    removed_segments = kept_segments = 0
    removed_cities: List[str] = []  # locality names of dropped rows

    for _, row in df.iterrows():
        videos = parse_videos(row["videos"])
        n = len(videos)
        times = {c: normalize_len(int_lists(row[c]), n, []) for c in ("start_time", "end_time", "time_of_day")}
        lists = {c: normalize_len(parse_list(row[c]), n, "" if c == "channel" else 0) for c in per_video}
        lists = {c: v if c == "channel" else [_int(x) for x in v] for c, v in lists.items()}
        out = {c: [] for c in ["videos", *times, *per_video]}

        for i, vid in enumerate(videos):
            segments = list(zip(*(times[c][i] for c in times)))  # (start, end, time of day)
            kept = [s for s in segments if (vid, s[0]) not in bbox_present]
            removed_segments += len(segments) - len(kept)
            kept_segments += len(kept)
            if kept:  # this video still has segments to process
                out["videos"].append(vid)
                for c, values in zip(times, zip(*kept)):
                    out[c].append(list(values))
                for c in per_video:
                    out[c].append(lists[c][i])

        if not out["videos"]:  # nothing left in this row
            removed_cities.append(str(row.get("locality", "")).strip())
            continue
        kept_rows.append({**dict(row), **{c: compact(v) for c, v in out.items()}})

    out_df = pd.DataFrame(kept_rows, columns=df.columns)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Segments removed (bbox exists): {removed_segments}")
    print(f"Segments remaining: {kept_segments}")
    print(f"Rows dropped (fully processed): {len(removed_cities)}")
    print(f"Rows kept: {len(out_df)} / {len(df)}")
    if PRINT_REMOVED_CITIES:
        print("\nCities removed (rows dropped):")
        for c in dict.fromkeys(c or "(missing locality)" for c in removed_cities):  # first-seen order
            print(f"  - {c}")
        print(f"Total removed rows printed: {len(removed_cities)}")


if __name__ == "__main__":
    build_remaining()
