#!/usr/bin/env python3
"""
Create mapping_remaining.csv from mapping_queue.csv by removing *segments* that
already have bbox CSV outputs.

Segment is considered processed if a file exists matching:
    <bbox_dir>/{video_id}_{start_time}_*.csv

Row is removed entirely if, after segment removal, there are no videos left in that row.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import common


# =========================
# EDIT THESE PATHS
# =========================

INPUT_CSV = common.get_configs("mapping")
OUTPUT_CSV = "mapping_remaining.csv"

# Put ALL bbox output directories you want to check here.
BBOX_OUTPUT_DIRS = common.get_configs("data")


# =========================
# FLAG: print removed cities
# =========================
PRINT_REMOVED_CITIES = False  # set True to print which locality rows are dropped


# =========================
# Parsing helpers
# =========================

def parse_bracketed_str_list(value: Any) -> List[str]:
    """Parse '[a,b,c]' (items not quoted) -> ['a','b','c']."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_bracketed_int_list(value: Any) -> List[int]:
    """Parse '[1,2,3]' -> [1,2,3]."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except Exception:
        pass
    out: List[int] = []
    for it in parse_bracketed_str_list(s):
        try:
            out.append(int(it))
        except Exception:
            out.append(0)
    return out


def parse_list_of_lists(value: Any) -> List[List[int]]:
    """Parse '[[1,2],[3]]' -> [[1,2],[3]]."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[List[int]] = []
    for item in obj:
        if isinstance(item, list):
            try:
                out.append([int(x) for x in item])
            except Exception:
                out.append([])
        else:
            try:
                out.append([int(item)])
            except Exception:
                out.append([])
    return out


def normalize_len(lst: List[Any], n: int, pad: Any) -> List[Any]:
    """Pad or truncate list to length n."""
    if len(lst) >= n:
        return lst[:n]
    return lst + [pad] * (n - len(lst))


# =========================
# Formatting helpers
# =========================

def format_str_list_no_quotes(items: List[str]) -> str:
    return "[" + ",".join(items) + "]"


def format_int_list(items: List[int]) -> str:
    return "[" + ",".join(str(x) for x in items) + "]"


def format_list_of_lists(lol: List[List[int]]) -> str:
    inner = []
    for lst in lol:
        inner.append("[" + ",".join(str(x) for x in lst) + "]")
    return "[" + ",".join(inner) + "]"


# =========================
# Bbox presence indexing
# =========================

def index_existing_bbox_segments(bbox_dirs: List[Path]) -> Set[Tuple[str, int]]:
    """
    Build a set of (video_id, start_time) from bbox files.

    Expected bbox filename: {video_id}_{start}_{fps}.csv
    video_id may contain underscores, so parse from the right:
        parts[-2] = start_time
        parts[:-2] join back to video_id
    """
    found: Set[Tuple[str, int]] = set()
    for d in bbox_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in d.glob("*.csv"):
            stem = p.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            st_str = parts[-2]
            if not st_str.lstrip("-").isdigit():
                continue
            vid = "_".join(parts[:-2])
            found.add((vid, int(st_str)))
    return found


# =========================
# Main transformation
# =========================

def build_remaining() -> None:
    bbox_dirs = [Path(p) for p in BBOX_OUTPUT_DIRS]
    bbox_present = index_existing_bbox_segments(bbox_dirs)

    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

    required_cols = ["videos", "start_time", "end_time", "time_of_day"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Only enforce locality column if printing is enabled (keeps change minimal)
    if PRINT_REMOVED_CITIES and "locality" not in df.columns:
        raise ValueError("Missing required column: locality (required when PRINT_REMOVED_CITIES=True)")

    has_vehicle = "vehicle_type" in df.columns
    has_upload = "upload_date" in df.columns
    has_channel = "channel" in df.columns

    kept_rows: List[Dict[str, Any]] = []
    removed_segments = 0
    kept_segments = 0
    dropped_rows = 0

    removed_cities: List[str] = []  # collects locality names of dropped rows (optional)

    for _, row in df.iterrows():
        videos = parse_bracketed_str_list(row.get("videos", ""))
        n = len(videos)
        if n == 0:
            dropped_rows += 1
            if PRINT_REMOVED_CITIES:
                removed_cities.append(str(row.get("locality", "")).strip())
            continue

        st_ll = normalize_len(parse_list_of_lists(row.get("start_time", "")), n, [])
        et_ll = normalize_len(parse_list_of_lists(row.get("end_time", "")), n, [])
        tod_ll = normalize_len(parse_list_of_lists(row.get("time_of_day", "")), n, [])

        vehicle = normalize_len(parse_bracketed_int_list(row.get("vehicle_type", "[]")), n, 0) if has_vehicle else []
        upload = normalize_len(parse_bracketed_int_list(row.get("upload_date", "[]")), n, 0) if has_upload else []
        channel = normalize_len(parse_bracketed_str_list(row.get("channel", "[]")), n, "") if has_channel else []

        new_videos: List[str] = []
        new_st_ll: List[List[int]] = []
        new_et_ll: List[List[int]] = []
        new_tod_ll: List[List[int]] = []
        new_vehicle: List[int] = []
        new_upload: List[int] = []
        new_channel: List[str] = []

        for i, vid in enumerate(videos):
            st_list = st_ll[i] if isinstance(st_ll[i], list) else []
            et_list = et_ll[i] if isinstance(et_ll[i], list) else []
            tod_list = tod_ll[i] if isinstance(tod_ll[i], list) else []

            kept_st: List[int] = []
            kept_et: List[int] = []
            kept_tod: List[int] = []

            # Segment-level pruning (not video-level)
            for st, et, tod in zip(st_list, et_list, tod_list):
                try:
                    st_i = int(st)
                except Exception:
                    # Malformed start_time: keep it (safer than dropping incorrectly)
                    kept_st.append(st)
                    kept_et.append(et)
                    kept_tod.append(tod)
                    kept_segments += 1
                    continue

                if (vid, st_i) in bbox_present:
                    removed_segments += 1
                    continue

                kept_st.append(int(st))
                kept_et.append(int(et) if str(et).lstrip("-").isdigit() else et)
                kept_tod.append(int(tod) if str(tod).lstrip("-").isdigit() else tod)
                kept_segments += 1

            # Keep this video only if it still has at least one remaining segment
            if kept_st:
                new_videos.append(vid)
                new_st_ll.append(kept_st)
                new_et_ll.append(kept_et)
                new_tod_ll.append(kept_tod)
                if has_vehicle:
                    new_vehicle.append(vehicle[i])
                if has_upload:
                    new_upload.append(upload[i])
                if has_channel:
                    new_channel.append(channel[i])

        # DROP THE WHOLE ROW if all videos are already processed (i.e., nothing remains)
        if not new_videos:
            dropped_rows += 1
            if PRINT_REMOVED_CITIES:
                removed_cities.append(str(row.get("locality", "")).strip())
            continue

        out_row = dict(row)
        out_row["videos"] = format_str_list_no_quotes(new_videos)
        out_row["start_time"] = format_list_of_lists(new_st_ll)
        out_row["end_time"] = format_list_of_lists(new_et_ll)
        out_row["time_of_day"] = format_list_of_lists(new_tod_ll)

        if has_vehicle:
            out_row["vehicle_type"] = format_int_list(new_vehicle)
        if has_upload:
            out_row["upload_date"] = format_int_list(new_upload)
        if has_channel:
            out_row["channel"] = format_str_list_no_quotes(new_channel)

        kept_rows.append(out_row)

    out_df = pd.DataFrame(kept_rows, columns=df.columns)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Segments removed (bbox exists): {removed_segments}")
    print(f"Segments remaining: {kept_segments}")
    print(f"Rows dropped (fully processed): {dropped_rows}")
    print(f"Rows kept: {len(out_df)} / {len(df)}")

    if PRINT_REMOVED_CITIES:
        print("\nCities removed (rows dropped):")
        # Print unique cities while preserving first-seen order
        seen: Set[str] = set()
        for c in removed_cities:
            c = c or "(missing locality)"
            if c in seen:
                continue
            seen.add(c)
            print(f"  - {c}")
        print(f"Total removed rows printed: {len(removed_cities)}")


if __name__ == "__main__":
    build_remaining()
