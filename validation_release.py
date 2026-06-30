#!/usr/bin/env python3
"""validate_release.py

Release validator for the dataset layout you actually have:

INPUTS (no parquet required)
---------------------------
1) mapping CSV (locality/state rows) containing list-like columns:
   - videos, time_of_day, start_time, end_time, vehicle_type, upload_date, channel, continent, country, locality, iso3

2) mapping_metadata.csv (per-video metadata):
   - id, video, title, upload_date, channel, views, description, chapters, segments, date_updated

3) YOLO processed CSV folder:
   - one CSV per processed segment, filename pattern: {video_id}_{start_time}_{fps}.csv
     Example: XsQuWRGgGNk_5_60.csv

WHAT IT DOES
------------
- Builds paper-aligned rollups:
  * unique videos (uploads)
  * upload_records (sum of videos per mapping row)
  * segment_records (label-entry count = sum of time_of_day entries)
  * total duration (sum over (end-start) across segments)
  * continent rollups (segments + duration + shares)
  * continent day/night label-entry distribution
  * global + continent upload day/night composition (only_day / only_night / both)

- Validates internal consistency:
  * len(videos) matches len(time_of_day/start/end/vehicle_type) outer list per row
  * per-video segment alignment (start/end pairs)
  * start<end, non-negative

- Validates cross-file integrity:
  * mapping vs mapping_metadata (missing videos either side)
  * mapping_metadata.segments matches segments derived from mapping (deduped by (start,end))
  * YOLO CSV filenames map to known videos and known segment start times (with tolerance)
  * YOLO coverage: how many segments have a YOLO CSV

- Writes:
  * _output/validate_release_report.json
  * _output/validate_release.log
  * _output/sha256_manifest.txt

-----------------
Edit CONFIG below and run:
  python validate_release.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Progress bars
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# Allow very large CSV fields (mapping rows can contain extremely long list strings)
try:
    csv.field_size_limit(min(sys.maxsize, 1024 * 1024 * 1024))
except Exception:
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass


# =============================================================================
# CONFIG (edit these paths)
# =============================================================================
# =============================================================================
# FLAGS (edit these)
# =============================================================================
# Interactive segment overlap cleanup (prompts in terminal)
FIX_OVERLAPS: bool = True
FIX_OVERLAPS_INPLACE: bool = False  # True overwrites mapping.csv, False writes _output/mapping.fixed.csv
OVERLAP_MATCH_EPS: float = 1e-3     # match tolerance for start and end when deleting


# Export a CSV listing all overlaps found in the current mapping file
EXPORT_OVERLAPS_FOUND_CSV: bool = True
OVERLAPS_FOUND_CSV_NAME: str = "overlaps_found.csv"

# Export a CSV listing videos involved in selected validation checks
# Categories included:
#   1) mapping_csv.multi_continent_videos
#   2) mapping_metadata.duplicate_video_rows
#   3) mapping_vs_metadata.missing_metadata_rows
#   4) mapping_vs_metadata.segment_count_match (mismatches only)
#   5) metadata_vs_mapping.missing_in_mapping
EXPORT_VALIDATION_VIDEO_ISSUES_CSV: bool = True
VALIDATION_VIDEO_ISSUES_CSV_NAME: str = "validation_video_issues.csv"

# Keep mapping list like columns in the original compact style (no quotes, no spaces)
WRITE_MAPPING_LISTS_COMPACT: bool = True

# YOLO: tolerance for matching filename start times to mapping segment start times (seconds)
YOLO_START_MATCH_TOLERANCE_S: float = 0.75

CONFIG: Dict[str, Any] = {
    "paths": {
        # Root folder that contains the mapping files and (optionally) _output/
        "data_dir": Path(__file__).resolve().parent,

        # REQUIRED
        "mapping_csv": "mapping.csv",
        "mapping_metadata_csv": "mapping_metadata.csv",

        # REQUIRED (folder of YOLO processed CSVs) - ABSOLUTE path is fine
        "yolo_dir": "bbox",

        # OPTIONAL: if provided, we will try to find claimed numbers in the LaTeX source
        "main_tex": None,  # e.g. "main.tex"
    },

    # Filename pattern for YOLO csvs
    "yolo_filename_regex": r"^(?P<video_id>[A-Za-z0-9_-]{6,})_(?P<start>[0-9]+(?:\.[0-9]+)?)_(?P<fps>[0-9]+(?:\.[0-9]+)?)\.csv$",  # noqa: E501

    # Allowed / expected categories
    "allowed": {
        "continents": ["Europe", "Asia", "North America", "Africa", "Oceania", "South America"],
        "time_of_day_codes": {0: "Day", 1: "Night"},
        "vehicle_type_codes": {
            0: "Car",
            1: "Bus",
            2: "Truck",
            3: "Two-wheeler",
            4: "Bicycle",
            5: "Automated car",
            6: "Electric scooter",
            7: "Monowheel/unicycle",
            8: "Automated bus",
            9: "Automated truck",
            10: "Automated two-wheeler",
            11: "Non-electric scooter",
            12: "Pedestrian",
        },
    },

    # OPTIONAL: reproducibility environment capture for detection and tracking.
    # This writes _output/reproducibility_environment.json and .txt.
    "reproducibility": {
        "enabled": True,
        "project_config_candidates": [
            "config",
            "config.yaml",
            "config.yml",
            "config.json",
            "common.yaml",
            "settings.yaml",
        ],
        "model_candidates": [
            "yolo11x.pt",
            "yolo11x-seg.pt",
        ],
        "tracker_candidates": [
            "bbox_custom_tracker.yaml",
            "seg_custom_tracker.yaml",
            "botsort.yaml",
            "bytetrack.yaml",
        ],
        "output_json": "reproducibility_environment.json",
        "output_txt": "reproducibility_environment.txt",
    },

    # OPTIONAL: known expected totals from the paper/logs (set to None to skip)
    "expected": {
        "unique_videos": 42032,
        "duration_s": 72992029,
        "duration_h": 20275.56,
        "segment_records": 51754,
        "upload_records": 44757,
        "countries": 238,
        "rows_locality_state_iso3": 7103,
    },

    # OPTIONAL: LaTeX number checks (works only if you set paths.main_tex)
    # Each item: (name, regex_to_capture_number, expected_key)
    # The regex MUST have exactly one capturing group for the numeric string.
    "latex_claim_checks": [
        (
            "unique_videos",
            r"(?i)\b(?:a\s+total\s+of|total\s+of)\s+([0-9]{1,3}(?:,[0-9]{3})*)\s+videos\b",
            "unique_videos",
        ),
        ("duration_hours", r"(?i)\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)\s+hours\b", "duration_h"),
        (
            "segment_records",
            r"(?i)\b([0-9]{1,3}(?:,[0-9]{3})*)\s+(?:segments|segment\s+records|label\s+entries)\b",
            "segment_records",
        ),
    ],
}

# =============================================================================
# Logging / report helpers
# =============================================================================


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_release")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-run in notebooks
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(out_dir / "validate_release.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_csv_data_rows(path: Path) -> Optional[int]:
    """Return number of data rows in a CSV (excluding header). None if unreadable."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            n_lines = sum(1 for _ in f)
        return max(int(n_lines) - 1, 0)
    except Exception:
        return None


def _num_from_str(s: str) -> Optional[float]:
    if s is None:
        return None
    ss = str(s).strip().replace(",", "")
    if ss.endswith("%"):
        ss = ss[:-1].strip()
    if ss == "":
        return None
    try:
        return float(ss)
    except Exception:
        return None


def _int_from_any(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except Exception:
        try:
            f = float(str(v).strip())
            if math.isfinite(f):
                return int(f)
        except Exception:
            return None
    return None


def safe_eval_list(x: Any) -> List[Any]:
    """Parse list-like cell robustly."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []

    s = x.strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    # strip wrapping quotes if the entire cell is quoted
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in {"'", '"'}):
        s = s[1:-1].strip()

    if s == "" or s == "[]":
        return []

    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        # Fallback: [a,b,c] with bare tokens
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            return [p for p in parts if p != ""]
        return []


# =============================================================================
# Interactive overlap cleanup
# =============================================================================
@dataclass(frozen=True)
class _SegOcc:
    video: str
    row_index: int      # 0 based data row index
    row_number: int     # 1 based data row number (excluding header)
    location: str       # locality|state|country
    start: float
    end: float


def dump_list_compact(obj: Any) -> str:
    """
    Serialise list like objects in the same compact style as the original mapping CSV.
    Strings are written without quotes and with no spaces after commas.
    """
    def _fmt(v: Any) -> str:
        if isinstance(v, list):
            return "[" + ",".join(_fmt(x) for x in v) + "]"
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            if math.isfinite(v) and abs(v - round(v)) < 1e-12:
                return str(int(round(v)))
            # Use a compact float representation
            s = repr(float(v))
            return s
        # Default: treat as token string without quotes
        s = str(v).strip()
        return s

    if isinstance(obj, list):
        return "[" + ",".join(_fmt(x) for x in obj) + "]"
    return _fmt(obj)


def export_overlaps_found_csv(mapping_path: Path, out_csv_path: Path, logger: logging.Logger, eps: float) -> int:
    """
    Scan mapping.csv for overlapping segments and write every overlap pair to a CSV.
    Returns the number of overlap pairs written.
    """
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return 0

    by_vid = _build_occurrences_for_all_videos(rows, fieldnames)  # type: ignore

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "a_row_number", "a_location", "a_start", "a_end",
                "b_row_number", "b_location", "b_start", "b_end",
                "overlap_s",
            ],
        )
        w.writeheader()

        for vid, occs in sorted(by_vid.items(), key=lambda kv: kv[0]):
            overlaps = _detect_overlaps_for_video(occs)
            for a, b in overlaps:
                overlap_s = max(0.0, min(a.end, b.end) - max(a.start, b.start))
                w.writerow(
                    {
                        "video": vid,
                        "a_row_number": a.row_number,
                        "a_location": a.location,
                        "a_start": a.start,
                        "a_end": a.end,
                        "b_row_number": b.row_number,
                        "b_location": b.location,
                        "b_start": b.start,
                        "b_end": b.end,
                        "overlap_s": overlap_s,
                    }
                )
                n_pairs += 1

    logger.info(f"Overlaps CSV written: {out_csv_path} (pairs: {n_pairs})")
    return n_pairs


def _looks_like_list_cell(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    return s.startswith("[") and s.endswith("]")


def _to_float_maybe(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None


def _build_occurrences_for_all_videos(rows: List[Dict[str, Any]], fieldnames: List[str]) -> Dict[str, List[_SegOcc]]:
    by_vid: Dict[str, List[_SegOcc]] = defaultdict(list)

    for ridx, r in enumerate(rows):
        rownum = ridx + 1
        country = (r.get("country") or "").strip()
        locality = (r.get("locality") or "").strip()
        state = (r.get("state") or "").strip()
        loc = f"{locality}|{state}|{country}"

        vids = [str(v).strip() for v in safe_eval_list(r.get("videos")) if str(v).strip() != ""]
        st_lol = safe_eval_list(r.get("start_time"))
        en_lol = safe_eval_list(r.get("end_time"))

        if not isinstance(st_lol, list) or not isinstance(en_lol, list):
            continue

        n_outer = min(len(vids), len(st_lol), len(en_lol))
        for i in range(n_outer):
            vid = vids[i]
            if not vid:
                continue

            starts = st_lol[i] if isinstance(st_lol[i], list) else []
            ends = en_lol[i] if isinstance(en_lol[i], list) else []

            for s_raw, e_raw in zip(starts, ends):
                s = _to_float_maybe(s_raw)
                e = _to_float_maybe(e_raw)
                if s is None or e is None:
                    continue
                if s < 0 or e <= s:
                    continue

                by_vid[vid].append(
                    _SegOcc(
                        video=vid,
                        row_index=ridx,
                        row_number=rownum,
                        location=loc,
                        start=float(s),
                        end=float(e),
                    )
                )

    return by_vid


def _detect_overlaps_for_video(occs: List[_SegOcc]) -> List[Tuple[_SegOcc, _SegOcc]]:
    if len(occs) < 2:
        return []

    segs_sorted = sorted(occs, key=lambda o: (o.start, o.end, o.location, o.row_number))
    overlaps: List[Tuple[_SegOcc, _SegOcc]] = []

    prev = segs_sorted[0]
    for cur in segs_sorted[1:]:
        if cur.start < prev.end:
            overlaps.append((prev, cur))
            if cur.end > prev.end:
                prev = cur
            continue
        prev = cur

    return overlaps


def _remove_inner_segment(row: Dict[str, Any], outer_i: int, start: float, end: float, eps: float) -> bool:
    vids = [str(v).strip() for v in safe_eval_list(row.get("videos"))]
    if outer_i < 0 or outer_i >= len(vids):
        return False

    st_lol = safe_eval_list(row.get("start_time"))
    en_lol = safe_eval_list(row.get("end_time"))
    tod_lol = safe_eval_list(row.get("time_of_day"))
    veh_lol = safe_eval_list(row.get("vehicle_type")) if "vehicle_type" in row else None

    if not (isinstance(st_lol, list) and isinstance(en_lol, list)):
        return False
    if outer_i >= len(st_lol) or outer_i >= len(en_lol):
        return False
    if not (isinstance(st_lol[outer_i], list) and isinstance(en_lol[outer_i], list)):
        return False

    starts = st_lol[outer_i]
    ends = en_lol[outer_i]

    match_j: Optional[int] = None
    for j, (s_raw, e_raw) in enumerate(zip(starts, ends)):
        s = _to_float_maybe(s_raw)
        e = _to_float_maybe(e_raw)
        if s is None or e is None:
            continue
        if abs(s - start) <= eps and abs(e - end) <= eps:
            match_j = j
            break

    if match_j is None:
        return False

    j = match_j

    try:
        starts.pop(j)
        ends.pop(j)
    except Exception:
        return False

    if isinstance(tod_lol, list) and outer_i < len(tod_lol) and isinstance(tod_lol[outer_i], list) and j < len(tod_lol[outer_i]):  # noqa:E501
        tod_lol[outer_i].pop(j)

    if isinstance(veh_lol, list) and outer_i < len(veh_lol) and isinstance(veh_lol[outer_i], list) and j < len(veh_lol[outer_i]):  # noqa:E501
        veh_lol[outer_i].pop(j)

    if WRITE_MAPPING_LISTS_COMPACT:
        row["start_time"] = dump_list_compact(st_lol)
        row["end_time"] = dump_list_compact(en_lol)
        if "time_of_day" in row:
            row["time_of_day"] = dump_list_compact(tod_lol)
        if "vehicle_type" in row and veh_lol is not None:
            row["vehicle_type"] = dump_list_compact(veh_lol)
    else:
        row["start_time"] = repr(st_lol)
        row["end_time"] = repr(en_lol)
        if "time_of_day" in row:
            row["time_of_day"] = repr(tod_lol)
        if "vehicle_type" in row and veh_lol is not None:
            row["vehicle_type"] = repr(veh_lol)

    return True


def _maybe_remove_empty_video_outer_entry(row: Dict[str, Any], outer_i: int, fieldnames: List[str]) -> bool:
    vids = [str(v).strip() for v in safe_eval_list(row.get("videos"))]
    st_lol = safe_eval_list(row.get("start_time"))
    en_lol = safe_eval_list(row.get("end_time"))

    if not (isinstance(st_lol, list) and isinstance(en_lol, list)):
        return False
    if outer_i >= len(vids) or outer_i >= len(st_lol) or outer_i >= len(en_lol):
        return False
    if not (isinstance(st_lol[outer_i], list) and isinstance(en_lol[outer_i], list)):
        return False

    if len(st_lol[outer_i]) != 0 or len(en_lol[outer_i]) != 0:
        return False

    outer_len = len(vids)

    for col in fieldnames:
        raw = row.get(col)
        if not _looks_like_list_cell(raw):
            continue
        parsed = safe_eval_list(raw)
        if not isinstance(parsed, list):
            continue
        if len(parsed) != outer_len:
            continue
        try:
            parsed.pop(outer_i)
            row[col] = dump_list_compact(parsed) if WRITE_MAPPING_LISTS_COMPACT else repr(parsed)
        except Exception:
            continue

    return True


def _find_outer_index_for_segment(row: Dict[str, Any], video_id: str, start: float,
                                  end: float, eps: float) -> Optional[int]:
    vids = [str(v).strip() for v in safe_eval_list(row.get("videos")) if str(v).strip() != ""]
    if not vids:
        return None

    st_lol = safe_eval_list(row.get("start_time"))
    en_lol = safe_eval_list(row.get("end_time"))
    if not (isinstance(st_lol, list) and isinstance(en_lol, list)):
        return None

    candidates = [i for i, v in enumerate(vids) if v == video_id]
    for i in candidates:
        if i >= len(st_lol) or i >= len(en_lol):
            continue
        starts = st_lol[i] if isinstance(st_lol[i], list) else None
        ends = en_lol[i] if isinstance(en_lol[i], list) else None
        if starts is None or ends is None:
            continue
        for s_raw, e_raw in zip(starts, ends):
            s = _to_float_maybe(s_raw)
            e = _to_float_maybe(e_raw)
            if s is None or e is None:
                continue
            if abs(s - start) <= eps and abs(e - end) <= eps:
                return i
    return None


def interactive_fix_overlaps_in_mapping_csv(
    mapping_path: Path,
    output_path: Path,
    logger: logging.Logger,
    inplace: bool,
    eps: float,
) -> Path:
    """
    Interactively resolve overlaps by asking which side to delete.
    Deleting means deleting the segment and all aligned values for that segment.
    If a video outer entry becomes empty, it is removed from videos and all aligned list columns.
    """
    if not sys.stdin.isatty():
        logger.warning("Interactive overlap fixer is enabled but stdin is not a TTY, skipping.")
        return mapping_path

    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        logger.info("mapping.csv is empty, nothing to fix.")
        return mapping_path

    by_vid = _build_occurrences_for_all_videos(rows, fieldnames)  # type: ignore

    vids_with_overlaps = []
    for vid, occs in by_vid.items():
        if _detect_overlaps_for_video(occs):
            vids_with_overlaps.append(vid)

    if not vids_with_overlaps:
        logger.info("No overlapping segments found. No changes made.")
        return mapping_path

    logger.info(f"Found overlaps in {len(vids_with_overlaps)} videos. Starting interactive cleanup.")

    skipped_pairs: Set[Tuple[str, int, float, float, int, float, float]] = set()
    quit_requested = False

    for vid in sorted(vids_with_overlaps):
        while True:
            occs = by_vid.get(vid, [])
            overlaps = _detect_overlaps_for_video(occs)
            if not overlaps:
                break

            chosen: Optional[Tuple[_SegOcc, _SegOcc]] = None
            for a, b in overlaps:
                sig = (
                    vid,
                    a.row_index, round(a.start, 3), round(a.end, 3),
                    b.row_index, round(b.start, 3), round(b.end, 3),
                )
                if sig not in skipped_pairs:
                    chosen = (a, b)
                    break

            if chosen is None:
                logger.warning(f"All remaining overlaps for video {vid} were kept. Leaving them as is.")
                break

            a, b = chosen
            sig = (
                vid,
                a.row_index, round(a.start, 3), round(a.end, 3),
                b.row_index, round(b.start, 3), round(b.end, 3),
            )

            logger.info("")
            logger.info(f"Video: {vid}")
            logger.info("Overlap detected. Choose what to delete.")
            logger.info(f"A) row {a.row_number}  {a.location}  [{a.start} , {a.end}]")
            logger.info(f"B) row {b.row_number}  {b.location}  [{b.start} , {b.end}]")
            logger.info("Options: 1 delete A, 2 delete B, 3 delete both, k keep both, q quit")
            choice = input("Your choice: ").strip().lower()

            if choice == "q":
                quit_requested = True
                logger.info("Quitting interactive overlap fixer early.")
                break

            if choice == "k" or choice == "":
                skipped_pairs.add(sig)
                continue

            def _apply_delete(target: _SegOcc) -> None:
                row = rows[target.row_index]
                outer_i = _find_outer_index_for_segment(row, target.video, target.start, target.end, eps=eps)
                if outer_i is None:
                    logger.warning(
                        f"Could not find segment in row {target.row_number} for deletion: {target.video} [{target.start},{target.end}]"  # noqa:E501
                    )
                    return

                ok = _remove_inner_segment(row, outer_i, target.start, target.end, eps=eps)
                if not ok:
                    logger.warning(
                        f"Failed to delete segment in row {target.row_number}: {target.video} [{target.start},{target.end}]"  # noqa:E501
                    )
                    return

                removed_outer = _maybe_remove_empty_video_outer_entry(row, outer_i, fieldnames)  # type: ignore

                if removed_outer:
                    by_vid[target.video] = [o for o in by_vid.get(target.video, []) if o.row_index != target.row_index]
                else:
                    by_vid[target.video] = [
                        o for o in by_vid.get(target.video, [])
                        if not (
                            o.row_index == target.row_index
                            and abs(o.start - target.start) <= eps
                            and abs(o.end - target.end) <= eps
                        )
                    ]

            if choice == "1":
                _apply_delete(a)
            elif choice == "2":
                _apply_delete(b)
            elif choice == "3":
                _apply_delete(a)
                _apply_delete(b)
            else:
                logger.warning("Unrecognised choice. Keeping both for now.")
                skipped_pairs.add(sig)
                continue

        if quit_requested:
            break

    if inplace:
        tmp_path = mapping_path.with_suffix(".tmp.fixed.csv")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(mapping_path)
        logger.info(f"Wrote fixed mapping in place: {mapping_path}")
        return mapping_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote fixed mapping to: {output_path}")
    return output_path


def to_list_of_ints(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[int] = []
        for t in x:
            iv = _int_from_any(t)
            if iv is not None:
                out.append(iv)
        return out
    iv = _int_from_any(x)
    return [iv] if iv is not None else []


def norm_list_of_lists_int(x: Any) -> List[List[int]]:
    lst = safe_eval_list(x)
    out: List[List[int]] = []
    for item in lst:
        if isinstance(item, list):
            arr = to_list_of_ints(item)
        else:
            arr = to_list_of_ints(item)
        out.append(arr if arr else [])
    return out


def norm_list_of_lists_float(x: Any) -> List[List[float]]:
    lst = safe_eval_list(x)
    out: List[List[float]] = []
    for item in lst:
        if isinstance(item, list):
            arr: List[float] = []
            for t in item:
                try:
                    fv = float(t)
                    if math.isfinite(fv):
                        arr.append(fv)
                except Exception:
                    continue
            out.append(arr)
        else:
            try:
                fv = float(item)
                if math.isfinite(fv):
                    out.append([fv])
                else:
                    out.append([])
            except Exception:
                out.append([])
    return out


def min_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    return min(len(a), len(b))


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_float(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


@dataclass
class Check:
    name: str
    ok: bool
    severity: str  # PASS/WARN/FAIL
    details: Dict[str, Any]


class Report:
    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(self, name: str, ok: bool, details: Optional[Dict[str, Any]] = None, warn: bool = False) -> None:
        if details is None:
            details = {}
        severity = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.checks.append(Check(name=name, ok=ok, severity=severity, details=details))

    def summary(self) -> Dict[str, int]:
        c = Counter(ch.severity for ch in self.checks)
        return {"pass": int(c.get("PASS", 0)), "warn": int(c.get("WARN", 0)), "fail": int(c.get("FAIL", 0))}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "checks": [
                {"name": ch.name, "ok": ch.ok, "severity": ch.severity, "details": ch.details} for ch in self.checks
            ],
        }


# =============================================================================
# Core aggregation from mapping.csv
# =============================================================================
@dataclass
class VideoAgg:
    continents: Counter
    has_day: bool = False
    has_night: bool = False
    segments: Set[Tuple[float, float]] = None  # type: ignore

    def __post_init__(self):
        if self.segments is None:
            self.segments = set()


@dataclass
class MappingAgg:
    rows: int
    unique_locality_state_iso3: int
    unique_countries: int
    upload_records: int
    unique_videos: int
    segment_records: int
    duration_s: float

    # Rollups
    continent_segment_records: Dict[str, int]
    continent_duration_s: Dict[str, float]
    continent_unique_uploads: Dict[str, int]

    continent_day_entries: Dict[str, int]
    continent_night_entries: Dict[str, int]

    global_upload_daynight: Dict[str, int]  # only_day, only_night, both_day_night, unknown
    continent_upload_daynight: Dict[str, Dict[str, int]]

    # Diagnostics
    n_rows_outer_mismatch: int
    n_videos_multi_continent: int


def parse_mapping_csv(path: Path, logger: logging.Logger, report: Report) -> Tuple[MappingAgg, Dict[str, VideoAgg]]:
    allowed_cont = set(CONFIG["allowed"]["continents"])
    allowed_tod_vals = set(CONFIG["allowed"]["time_of_day_codes"].keys())  # {0, 1}
    allowed_vehicle_vals = set(CONFIG["allowed"]["vehicle_type_codes"].keys())
    cols: List[str] = []

    rows = 0
    locality_state_iso3_keys: Set[str] = set()
    countries: Set[str] = set()

    upload_records = 0
    segment_records = 0
    duration_s = 0.0

    cont_seg: Dict[str, int] = defaultdict(int)
    cont_dur: Dict[str, float] = defaultdict(float)

    cont_day: Dict[str, int] = defaultdict(int)
    cont_night: Dict[str, int] = defaultdict(int)

    video_aggs: Dict[str, VideoAgg] = {}

    n_rows_outer_mismatch = 0
    n_bad_segment_pairs = 0
    bad_segment_examples: List[Dict[str, Any]] = []

    # Rows with zero videos
    n_rows_zero_videos = 0
    zero_video_examples: List[Dict[str, Any]] = []

    # time_of_day contains only 0 or 1
    n_invalid_tod_values = 0
    invalid_tod_examples: List[Dict[str, Any]] = []

    # vehicle_type contains only known mapped codes
    n_invalid_vehicle_values = 0
    invalid_vehicle_examples: List[Dict[str, Any]] = []

    # Duplicate locality keys (locality + state + country)
    locality_state_country_counts: Counter[str] = Counter()

    # Per-video segment occurrences with location info (for overlap / duplicate checks)
    # Stored as (start_s, end_s, locality|state|country, row_number)
    video_segment_details: Dict[str, List[Tuple[float, float, str, int]]] = defaultdict(list)

    # avg_height consistency per country (all rows within a country should share the same avg_height)
    has_avg_height_col = False
    country_avg_height_values: Dict[str, Set[str]] = defaultdict(set)
    country_avg_height_rows: Counter[str] = Counter()
    country_avg_height_missing: Counter[str] = Counter()
    n_invalid_avg_height = 0
    invalid_avg_height_examples: List[Dict[str, Any]] = []

    # med_age consistency per country (all rows within a country should share the same med_age)
    has_med_age_col = False
    country_med_age_values: Dict[str, Set[str]] = defaultdict(set)
    country_med_age_rows: Counter[str] = Counter()
    country_med_age_missing: Counter[str] = Counter()
    n_invalid_med_age = 0
    invalid_med_age_examples: List[Dict[str, Any]] = []

    # traffic_mortality consistency per country (all rows within a country should share the same traffic_mortality)
    has_traffic_mortality_col = False
    traffic_mortality_col_name = ""
    country_traffic_mortality_values: Dict[str, Set[str]] = defaultdict(set)
    country_traffic_mortality_rows: Counter[str] = Counter()
    country_traffic_mortality_missing: Counter[str] = Counter()
    n_invalid_traffic_mortality = 0
    invalid_traffic_mortality_examples: List[Dict[str, Any]] = []

    # literacy_rate consistency per country (all rows within a country should share the same literacy_rate)
    has_literacy_rate_col = False
    literacy_rate_col_name = ""
    country_literacy_rate_values: Dict[str, Set[str]] = defaultdict(set)
    country_literacy_rate_rows: Counter[str] = Counter()
    country_literacy_rate_missing: Counter[str] = Counter()
    n_invalid_literacy_rate = 0
    invalid_literacy_rate_examples: List[Dict[str, Any]] = []

    total_rows = count_csv_data_rows(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []  # type: ignore
        required = {"continent", "country", "locality", "iso3", "videos", "time_of_day", "start_time", "end_time"}
        missing = sorted(list(required - set(cols)))
        report.add(
            "mapping_csv.required_columns",
            ok=(len(missing) == 0),
            details={"missing": missing, "present": cols},
            warn=(len(missing) > 0),
        )

        colset = set(cols)
        has_avg_height_col = "avg_height" in colset
        report.add(
            "mapping_csv.avg_height_column_present",
            ok=has_avg_height_col,
            details={"present": has_avg_height_col},
            warn=(not has_avg_height_col),
        )

        has_med_age_col = "med_age" in colset
        report.add(
            "mapping_csv.med_age_column_present",
            ok=has_med_age_col,
            details={"present": has_med_age_col},
            warn=(not has_med_age_col),
        )

        # traffic_mortality column detection (support a couple of common variants)
        traffic_candidates = ["traffic_mortality", "traffic mortality"]
        for c in traffic_candidates:
            if c in colset:
                has_traffic_mortality_col = True
                traffic_mortality_col_name = c
                break
        report.add(
            "mapping_csv.traffic_mortality_column_present",
            ok=has_traffic_mortality_col,
            details={"present": has_traffic_mortality_col, "column_name": traffic_mortality_col_name or None},
            warn=(not has_traffic_mortality_col),
        )

        # literacy_rate column detection (support a couple of common variants)
        literacy_candidates = ["literacy_rate", "literacy rate"]
        for c in literacy_candidates:
            if c in colset:
                has_literacy_rate_col = True
                literacy_rate_col_name = c
                break
        report.add(
            "mapping_csv.literacy_rate_column_present",
            ok=has_literacy_rate_col,
            details={"present": has_literacy_rate_col, "column_name": literacy_rate_col_name or None},
            warn=(not has_literacy_rate_col),
        )

        it = tqdm(reader, total=total_rows, desc="Parsing mapping.csv", unit="row")
        for r in it:
            rows += 1
            continent = (r.get("continent") or "").strip()
            country = (r.get("country") or "").strip()
            locality = (r.get("locality") or "").strip()
            iso3 = (r.get("iso3") or "").strip()
            state = (r.get("state") or "").strip()

            loc_key_csc = f"{locality}|{state}|{country}"

            if country:
                countries.add(country)
            if locality and iso3:
                locality_state_iso3_keys.add(f"{locality}|{state}|{iso3}")

            # Duplicate key count (locality + state + country)
            if locality and country:
                locality_state_country_counts[f"{locality}|{state}|{country}"] += 1

            # avg_height consistency per country
            if has_avg_height_col and country:
                country_avg_height_rows[country] += 1
                raw_h = (r.get("avg_height") or "").strip()
                if raw_h == "" or raw_h.lower() in {"nan", "none", "null"}:
                    country_avg_height_missing[country] += 1
                else:
                    num_h = _num_from_str(raw_h)
                    if num_h is None or (not math.isfinite(float(num_h))):
                        n_invalid_avg_height += 1
                        country_avg_height_values[country].add(raw_h)
                        if len(invalid_avg_height_examples) < 25:
                            invalid_avg_height_examples.append(
                                {
                                    "row": rows,
                                    "country": country,
                                    "state": state,
                                    "locality": locality,
                                    "value": raw_h,
                                }
                            )
                    else:
                        norm_h = f"{round(float(num_h), 6):.6f}".rstrip("0").rstrip(".")
                        country_avg_height_values[country].add(norm_h)

            # med_age consistency per country
            if has_med_age_col and country:
                country_med_age_rows[country] += 1
                raw_ma = (r.get("med_age") or "").strip()
                if raw_ma == "" or raw_ma.lower() in {"nan", "none", "null"}:
                    country_med_age_missing[country] += 1
                else:
                    num_ma = _num_from_str(raw_ma)
                    if num_ma is None or (not math.isfinite(float(num_ma))):
                        n_invalid_med_age += 1
                        country_med_age_values[country].add(raw_ma)
                        if len(invalid_med_age_examples) < 25:
                            invalid_med_age_examples.append(
                                {
                                    "row": rows,
                                    "country": country,
                                    "state": state,
                                    "locality": locality,
                                    "value": raw_ma,
                                }
                            )
                    else:
                        norm_ma = f"{round(float(num_ma), 6):.6f}".rstrip("0").rstrip(".")
                        country_med_age_values[country].add(norm_ma)

            # traffic_mortality consistency per country
            if has_traffic_mortality_col and country:
                country_traffic_mortality_rows[country] += 1
                raw_tm = (r.get(traffic_mortality_col_name) or "").strip()
                if raw_tm == "" or raw_tm.lower() in {"nan", "none", "null"}:
                    country_traffic_mortality_missing[country] += 1
                else:
                    num_tm = _num_from_str(raw_tm)
                    if num_tm is None or (not math.isfinite(float(num_tm))):
                        n_invalid_traffic_mortality += 1
                        country_traffic_mortality_values[country].add(raw_tm)
                        if len(invalid_traffic_mortality_examples) < 25:
                            invalid_traffic_mortality_examples.append(
                                {"row": rows, "country": country, "state": state,
                                 "locality": locality, "value": raw_tm}
                            )
                    else:
                        norm_tm = f"{round(float(num_tm), 6):.6f}".rstrip("0").rstrip(".")
                        country_traffic_mortality_values[country].add(norm_tm)

            # literacy_rate consistency per country
            if has_literacy_rate_col and country:
                country_literacy_rate_rows[country] += 1
                raw_lr = (r.get(literacy_rate_col_name) or "").strip()
                if raw_lr == "" or raw_lr.lower() in {"nan", "none", "null"}:
                    country_literacy_rate_missing[country] += 1
                else:
                    num_lr = _num_from_str(raw_lr)
                    if num_lr is None or (not math.isfinite(float(num_lr))):
                        n_invalid_literacy_rate += 1
                        country_literacy_rate_values[country].add(raw_lr)
                        if len(invalid_literacy_rate_examples) < 25:
                            invalid_literacy_rate_examples.append(
                                {"row": rows, "country": country,
                                 "state": state, "locality": locality, "value": raw_lr}
                            )
                    else:
                        norm_lr = f"{round(float(num_lr), 6):.6f}".rstrip("0").rstrip(".")
                        country_literacy_rate_values[country].add(norm_lr)

            vids = [str(v).strip() for v in safe_eval_list(r.get("videos")) if str(v).strip() != ""]
            tod_lol = norm_list_of_lists_int(r.get("time_of_day"))
            st_lol = norm_list_of_lists_float(r.get("start_time"))
            en_lol = norm_list_of_lists_float(r.get("end_time"))
            veh_lol = norm_list_of_lists_int(r.get("vehicle_type")) if "vehicle_type" in set(cols) else []

            # Rows that have no videos
            if len(vids) == 0:
                n_rows_zero_videos += 1
                if len(zero_video_examples) < 25:
                    zero_video_examples.append(
                        {"row": rows, "continent": continent, "country": country,
                         "state": state, "locality": locality, "iso3": iso3})

            # time_of_day values are only 0 or 1 (scan entire cell content)
            for outer_i, tods in enumerate(tod_lol):
                for inner_i, t in enumerate(tods):
                    if t not in allowed_tod_vals:
                        n_invalid_tod_values += 1
                        if len(invalid_tod_examples) < 25:
                            invalid_tod_examples.append(
                                {
                                    "row": rows,
                                    "continent": continent,
                                    "country": country,
                                    "state": state,
                                    "locality": locality,
                                    "iso3": iso3,
                                    "outer_index": outer_i,
                                    "inner_index": inner_i,
                                    "value": t,
                                }
                            )

            # vehicle_type values are only known mapped codes (scan entire cell content)
            for outer_i, vehs in enumerate(veh_lol):
                for inner_i, v in enumerate(vehs):
                    if v not in allowed_vehicle_vals:
                        n_invalid_vehicle_values += 1
                        if len(invalid_vehicle_examples) < 25:
                            invalid_vehicle_examples.append(
                                {
                                    "row": rows,
                                    "continent": continent,
                                    "country": country,
                                    "state": state,
                                    "locality": locality,
                                    "iso3": iso3,
                                    "outer_index": outer_i,
                                    "inner_index": inner_i,
                                    "value": v,
                                }
                            )

            upload_records += len(vids)

            # Outer list alignment diagnostics should match len(vids)
            outer_lens = {
                "videos": len(vids),
                "time_of_day": len(tod_lol),
                "start_time": len(st_lol),
                "end_time": len(en_lol),
            }
            if "vehicle_type" in set(cols):
                outer_lens["vehicle_type"] = len(veh_lol)

            if len({outer_lens[k] for k in outer_lens}) != 1:
                n_rows_outer_mismatch += 1

            # For rollups, trust only aligned portion (zip truncation)
            n_outer = len(vids)
            n_outer = min(n_outer, len(tod_lol), len(st_lol), len(en_lol))
            if "vehicle_type" in set(cols):
                n_outer = min(n_outer, len(veh_lol))

            row_seg_records = 0
            row_dur = 0.0

            if continent and continent not in allowed_cont:
                report.add(
                    "mapping_csv.continent_allowed",
                    ok=False,
                    details={"continent": continent, "row": rows},
                    warn=True,
                )

            for i in range(n_outer):
                vid = vids[i]
                if not vid:
                    continue

                tods = tod_lol[i] if i < len(tod_lol) else []
                starts = st_lol[i] if i < len(st_lol) else []
                ends = en_lol[i] if i < len(en_lol) else []

                row_seg_records += len(tods)

                for s, e in zip(starts, ends):
                    try:
                        s2 = float(s)
                        e2 = float(e)
                    except Exception:
                        n_bad_segment_pairs += 1
                        if len(bad_segment_examples) < 25:
                            bad_segment_examples.append({"row": rows, "video": vid, "start": s,
                                                         "end": e, "reason": "non_numeric"})
                        continue

                    if (not math.isfinite(s2)) or (not math.isfinite(e2)):
                        n_bad_segment_pairs += 1
                        if len(bad_segment_examples) < 25:
                            bad_segment_examples.append({"row": rows, "video": vid, "start": s2,
                                                         "end": e2, "reason": "non_finite"})
                        continue

                    if s2 < 0 or e2 < 0 or e2 <= s2:
                        n_bad_segment_pairs += 1
                        if len(bad_segment_examples) < 25:
                            bad_segment_examples.append({"row": rows, "video": vid, "start": s2,
                                                         "end": e2, "reason": "invalid_order_or_negative"})
                        continue

                    row_dur += (e2 - s2)

                    # Track segment occurrences for overlap / duplicate checks
                    video_segment_details[vid].append((s2, e2, loc_key_csc, rows))

                if vid not in video_aggs:
                    video_aggs[vid] = VideoAgg(continents=Counter())

                if continent:
                    video_aggs[vid].continents[continent] += 1

                for t in (tods or []):
                    if t == 0:
                        video_aggs[vid].has_day = True
                    elif t == 1:
                        video_aggs[vid].has_night = True

                for s, e in zip(starts, ends):
                    try:
                        s2 = float(s)
                        e2 = float(e)
                        if math.isfinite(s2) and math.isfinite(e2) and (s2 >= 0) and (e2 > s2):
                            video_aggs[vid].segments.add((s2, e2))
                    except Exception:
                        continue

            day_ct = 0
            night_ct = 0
            for i in range(min_len(vids, tod_lol)):
                for t in tod_lol[i]:
                    if t == 0:
                        day_ct += 1
                    elif t == 1:
                        night_ct += 1

            if continent:
                cont_seg[continent] += row_seg_records
                cont_dur[continent] += row_dur
                cont_day[continent] += day_ct
                cont_night[continent] += night_ct

            segment_records += row_seg_records
            duration_s += row_dur

    video_canon_cont: Dict[str, str] = {}
    n_multi_cont = 0
    for vid, agg in video_aggs.items():
        if not agg.continents:
            continue
        if len(agg.continents) > 1:
            n_multi_cont += 1
        canon = sorted(agg.continents.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        video_canon_cont[vid] = canon

    cont_unique: Dict[str, int] = defaultdict(int)
    for vid, cont in video_canon_cont.items():
        cont_unique[cont] += 1

    global_dn = Counter({"only_day": 0, "only_night": 0, "both_day_night": 0, "unknown": 0})
    cont_dn: Dict[str, Counter] = defaultdict(
        lambda: Counter({"only_day": 0, "only_night": 0, "both_day_night": 0, "unknown": 0})
    )

    for vid, agg in video_aggs.items():
        if agg.has_day and agg.has_night:
            cat = "both_day_night"
        elif agg.has_day:
            cat = "only_day"
        elif agg.has_night:
            cat = "only_night"
        else:
            cat = "unknown"

        global_dn[cat] += 1
        cont = video_canon_cont.get(vid)
        if cont:
            cont_dn[cont][cat] += 1

    agg = MappingAgg(
        rows=rows,
        unique_locality_state_iso3=len(locality_state_iso3_keys),
        unique_countries=len(countries),
        upload_records=upload_records,
        unique_videos=len(video_aggs),
        segment_records=segment_records,
        duration_s=duration_s,
        continent_segment_records=dict(cont_seg),
        continent_duration_s=dict(cont_dur),
        continent_unique_uploads=dict(cont_unique),
        continent_day_entries=dict(cont_day),
        continent_night_entries=dict(cont_night),
        global_upload_daynight=dict(global_dn),
        continent_upload_daynight={k: dict(v) for k, v in cont_dn.items()},
        n_rows_outer_mismatch=n_rows_outer_mismatch,
        n_videos_multi_continent=n_multi_cont,
    )

    report.add(
        "mapping_csv.outer_list_alignment",
        ok=(n_rows_outer_mismatch == 0),
        details={"rows": rows, "rows_with_outer_len_mismatch": n_rows_outer_mismatch},
        warn=(n_rows_outer_mismatch > 0),
    )

    report.add(
        "mapping_csv.multi_continent_videos",
        ok=(n_multi_cont == 0),
        details={"videos_with_multiple_continents": n_multi_cont},
        warn=(n_multi_cont > 0),
    )

    report.add(
        "mapping_csv.segment_pairs_valid",
        ok=(n_bad_segment_pairs == 0),
        details={"bad_pair_count": n_bad_segment_pairs, "examples": bad_segment_examples},
        warn=(n_bad_segment_pairs > 0),
    )

    report.add(
        "mapping_csv.rows_with_zero_videos",
        ok=(n_rows_zero_videos == 0),
        details={"rows_with_zero_videos": n_rows_zero_videos, "examples": zero_video_examples},
        warn=(n_rows_zero_videos > 0),
    )

    report.add(
        "mapping_csv.time_of_day_values_valid",
        ok=(n_invalid_tod_values == 0),
        details={"invalid_value_count": n_invalid_tod_values, "examples": invalid_tod_examples,
                 "allowed_values": sorted(list(allowed_tod_vals))},
        warn=(n_invalid_tod_values > 0),
    )

    report.add(
        "mapping_csv.vehicle_type_values_valid",
        ok=(n_invalid_vehicle_values == 0),
        details={
            "invalid_value_count": n_invalid_vehicle_values,
            "examples": invalid_vehicle_examples,
            "allowed_values": sorted(list(allowed_vehicle_vals)),
            "allowed_labels": {str(k): v for k, v in sorted(CONFIG["allowed"]["vehicle_type_codes"].items())},
        },
        warn=(n_invalid_vehicle_values > 0),
    )

    dups = [(k, v) for k, v in locality_state_country_counts.items() if v > 1]
    dups_sorted = sorted(dups, key=lambda kv: (-kv[1], kv[0]))[:25]
    dup_examples: List[Dict[str, Any]] = []
    for k, v in dups_sorted:
        c, s, co = k.split("|", 2)
        dup_examples.append({"locality": c, "state": s, "country": co, "rows": v})

    report.add(
        "mapping_csv.duplicate_locality_state_country",
        ok=(len(dups) == 0),
        details={"duplicate_key_count": len(dups), "examples": dup_examples},
        warn=(len(dups) > 0),
    )

    # avg_height consistency report (only if the column exists)
    if has_avg_height_col:
        mismatched_countries: List[str] = []
        for co, vals in country_avg_height_values.items():
            if len(vals) > 1:
                mismatched_countries.append(co)

        examples: List[Dict[str, Any]] = []
        for co in sorted(mismatched_countries, key=lambda x: (-int(country_avg_height_rows.get(x, 0)), x))[:25]:
            vals = sorted(list(country_avg_height_values.get(co, set())))
            examples.append(
                {
                    "country": co,
                    "rows": int(country_avg_height_rows.get(co, 0)),
                    "unique_value_count": len(country_avg_height_values.get(co, set())),
                    "unique_values": vals[:10],
                    "missing_rows": int(country_avg_height_missing.get(co, 0)),
                }
            )

        report.add(
            "mapping_csv.avg_height_consistent_by_country",
            ok=(len(mismatched_countries) == 0),
            details={"mismatch_country_count": len(mismatched_countries), "examples": examples},
            warn=(len(mismatched_countries) > 0),
        )

        report.add(
            "mapping_csv.avg_height_values_parseable",
            ok=(n_invalid_avg_height == 0),
            details={"invalid_value_count": n_invalid_avg_height, "examples": invalid_avg_height_examples},
            warn=(n_invalid_avg_height > 0),
        )

    # med_age consistency report (only if the column exists)
    if has_med_age_col:
        mismatched_countries: List[str] = []
        for co, vals in country_med_age_values.items():
            if len(vals) > 1:
                mismatched_countries.append(co)

        examples: List[Dict[str, Any]] = []
        for co in sorted(mismatched_countries, key=lambda x: (-int(country_med_age_rows.get(x, 0)), x))[:25]:
            vals = sorted(list(country_med_age_values.get(co, set())))
            examples.append(
                {
                    "country": co,
                    "rows": int(country_med_age_rows.get(co, 0)),
                    "unique_value_count": len(country_med_age_values.get(co, set())),
                    "unique_values": vals[:10],
                    "missing_rows": int(country_med_age_missing.get(co, 0)),
                }
            )

        report.add(
            "mapping_csv.med_age_consistent_by_country",
            ok=(len(mismatched_countries) == 0),
            details={"mismatch_country_count": len(mismatched_countries), "examples": examples},
            warn=(len(mismatched_countries) > 0),
        )

        report.add(
            "mapping_csv.med_age_values_parseable",
            ok=(n_invalid_med_age == 0),
            details={"invalid_value_count": n_invalid_med_age, "examples": invalid_med_age_examples},
            warn=(n_invalid_med_age > 0),
        )

    # traffic_mortality consistency report (only if the column exists)
    if has_traffic_mortality_col:
        mismatched_countries: List[str] = []
        for co, vals in country_traffic_mortality_values.items():
            if len(vals) > 1:
                mismatched_countries.append(co)

        examples: List[Dict[str, Any]] = []
        for co in sorted(mismatched_countries, key=lambda x: (-int(country_traffic_mortality_rows.get(x, 0)), x))[:25]:
            vals = sorted(list(country_traffic_mortality_values.get(co, set())))
            examples.append(
                {
                    "country": co,
                    "rows": int(country_traffic_mortality_rows.get(co, 0)),
                    "unique_value_count": len(country_traffic_mortality_values.get(co, set())),
                    "unique_values": vals[:10],
                    "missing_rows": int(country_traffic_mortality_missing.get(co, 0)),
                }
            )

        report.add(
            "mapping_csv.traffic_mortality_consistent_by_country",
            ok=(len(mismatched_countries) == 0),
            details={"mismatch_country_count": len(mismatched_countries), "examples": examples},
            warn=(len(mismatched_countries) > 0),
        )

        report.add(
            "mapping_csv.traffic_mortality_values_parseable",
            ok=(n_invalid_traffic_mortality == 0),
            details={"invalid_value_count": n_invalid_traffic_mortality,
                     "examples": invalid_traffic_mortality_examples},
            warn=(n_invalid_traffic_mortality > 0),
        )

    # literacy_rate consistency report (only if the column exists)
    if has_literacy_rate_col:
        mismatched_countries: List[str] = []
        for co, vals in country_literacy_rate_values.items():
            if len(vals) > 1:
                mismatched_countries.append(co)

        examples: List[Dict[str, Any]] = []
        for co in sorted(mismatched_countries, key=lambda x: (-int(country_literacy_rate_rows.get(x, 0)), x))[:25]:
            vals = sorted(list(country_literacy_rate_values.get(co, set())))
            examples.append(
                {
                    "country": co,
                    "rows": int(country_literacy_rate_rows.get(co, 0)),
                    "unique_value_count": len(country_literacy_rate_values.get(co, set())),
                    "unique_values": vals[:10],
                    "missing_rows": int(country_literacy_rate_missing.get(co, 0)),
                }
            )

        report.add(
            "mapping_csv.literacy_rate_consistent_by_country",
            ok=(len(mismatched_countries) == 0),
            details={"mismatch_country_count": len(mismatched_countries), "examples": examples},
            warn=(len(mismatched_countries) > 0),
        )

        report.add(
            "mapping_csv.literacy_rate_values_parseable",
            ok=(n_invalid_literacy_rate == 0),
            details={"invalid_value_count": n_invalid_literacy_rate, "examples": invalid_literacy_rate_examples},
            warn=(n_invalid_literacy_rate > 0),
        )

    # ---------------------------------------------------------------------
    # Segment overlap and duplicate location checks (per video)
    # A given video should not have overlapping segments. Also, the same segment
    # (start,end) should not appear under multiple locality|state|country locations.
    # ---------------------------------------------------------------------
    n_overlapping_pairs = 0
    overlap_examples: List[Dict[str, Any]] = []

    n_duplicate_segments_across_locations = 0
    duplicate_segment_examples: List[Dict[str, Any]] = []

    it_vids = tqdm(
        video_segment_details.items(),
        total=len(video_segment_details),
        desc="Checking segment overlaps",
        unit="video",
    )

    for vid, seg_list in it_vids:
        if not seg_list or len(seg_list) < 2:
            continue

        # 4a) Overlap check
        segs_sorted = sorted(seg_list, key=lambda x: (x[0], x[1], x[2], x[3]))
        prev_s, prev_e, prev_loc, prev_row = segs_sorted[0]
        for cur_s, cur_e, cur_loc, cur_row in segs_sorted[1:]:
            # Overlap if next segment starts before previous ends
            if cur_s < prev_e:
                n_overlapping_pairs += 1
                if len(overlap_examples) < 25:
                    overlap_examples.append(
                        {
                            "video": vid,
                            "prev": {"start": prev_s, "end": prev_e, "location": prev_loc, "row": prev_row},
                            "curr": {"start": cur_s, "end": cur_e, "location": cur_loc, "row": cur_row},
                        }
                    )
                # keep the interval with the later end as the active one
                if cur_e > prev_e:
                    prev_s, prev_e, prev_loc, prev_row = cur_s, cur_e, cur_loc, cur_row
                continue

            prev_s, prev_e, prev_loc, prev_row = cur_s, cur_e, cur_loc, cur_row

        # 4b) Duplicate exact segments across different locations
        seg_to_locs: Dict[Tuple[float, float], Set[str]] = defaultdict(set)
        seg_to_rows: Dict[Tuple[float, float], Set[int]] = defaultdict(set)
        for s, e, loc, rown in seg_list:
            key = (round(float(s), 3), round(float(e), 3))
            seg_to_locs[key].add(loc)
            seg_to_rows[key].add(int(rown))

        for (s, e), locs in seg_to_locs.items():
            if len(locs) > 1:
                n_duplicate_segments_across_locations += 1
                if len(duplicate_segment_examples) < 25:
                    duplicate_segment_examples.append(
                        {
                            "video": vid,
                            "start": s,
                            "end": e,
                            "location_count": len(locs),
                            "locations": sorted(list(locs))[:10],
                            "rows": sorted(list(seg_to_rows.get((s, e), set())))[:15],
                        }
                    )

    report.add(
        "mapping_csv.video_segments_non_overlapping",
        ok=(n_overlapping_pairs == 0),
        details={"overlapping_pair_count": n_overlapping_pairs, "examples": overlap_examples},
        warn=(n_overlapping_pairs > 0),
    )

    report.add(
        "mapping_csv.video_segment_duplicate_locations",
        ok=(n_duplicate_segments_across_locations == 0),
        details={
            "duplicate_segment_count": n_duplicate_segments_across_locations,
            "examples": duplicate_segment_examples,
        },
        warn=(n_duplicate_segments_across_locations > 0),
    )

    return agg, video_aggs


# =============================================================================
# mapping_metadata.csv checks
# =============================================================================
@dataclass
class MetadataRow:
    id: str
    video: str
    title: str
    upload_date: str
    channel: str
    views: str
    segments: str
    date_updated: str


def load_mapping_metadata(
    path: Path,
    logger: logging.Logger,
    report: Report,
) -> Tuple[Dict[str, MetadataRow], Dict[str, int]]:
    out: Dict[str, MetadataRow] = {}

    total_rows = count_csv_data_rows(path)

    # NEW: detect duplicate video ids in mapping_metadata.csv
    first_seen_row: Dict[str, int] = {}
    dup_row_count = 0
    dup_video_counts: Counter[str] = Counter()
    dup_examples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        required = {"video", "segments"}
        missing = sorted(list(required - set(cols)))
        report.add(
            "mapping_metadata.required_columns",
            ok=(len(missing) == 0),
            details={"missing": missing, "present": cols},
            warn=(len(missing) > 0),
        )

        it = tqdm(reader, total=total_rows, desc="Loading mapping_metadata.csv", unit="row")
        rownum = 1  # data row counter (excluding header)
        for r in it:
            video = (r.get("video") or "").strip()
            if not video:
                rownum += 1
                continue

            if video in first_seen_row:
                dup_row_count += 1
                dup_video_counts[video] += 1
                if len(dup_examples) < 25:
                    dup_examples.append(
                        {"video": video, "first_row": first_seen_row[video], "duplicate_row": rownum}
                    )
                # Keep the first occurrence in `out` to avoid silent overwrites
                rownum += 1
                continue

            first_seen_row[video] = rownum

            mr = MetadataRow(
                id=(r.get("id") or "").strip(),
                video=video,
                title=(r.get("title") or "").strip(),
                upload_date=(r.get("upload_date") or "").strip(),
                channel=(r.get("channel") or "").strip(),
                views=(r.get("views") or "").strip(),
                segments=(r.get("segments") or "").strip(),
                date_updated=(r.get("date_updated") or "").strip(),
            )
            out[video] = mr
            rownum += 1

    report.add(
        "mapping_metadata.duplicate_video_rows",
        ok=(dup_row_count == 0),
        details={"duplicate_row_count": dup_row_count, "examples": dup_examples},
        warn=(dup_row_count > 0),
    )

    report.add(
        "mapping_metadata.unique_video_keys",
        ok=True,
        details={"n_unique_videos": len(out), "n_rows_total": total_rows, "duplicate_row_count": dup_row_count},
    )
    return out, dict(dup_video_counts)


def compare_metadata_segments(
    video_aggs: Dict[str, VideoAgg],
    meta: Dict[str, MetadataRow],
    report: Report,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    mismatches: List[Dict[str, Any]] = []
    missing_meta: List[str] = []

    it = tqdm(video_aggs.items(), total=len(video_aggs), desc="Comparing mapping vs metadata", unit="video")
    for vid, agg in it:
        mr = meta.get(vid)
        if mr is None:
            missing_meta.append(vid)
            continue

        derived = len(agg.segments)
        claimed = _int_from_any(mr.segments)

        if claimed is None:
            mismatches.append({"video": vid, "claimed": mr.segments, "derived": derived, "reason": "claimed_not_int"})
            continue

        if int(claimed) != int(derived):
            mismatches.append({"video": vid, "claimed": int(claimed), "derived": derived})

    report.add(
        "mapping_vs_metadata.missing_metadata_rows",
        ok=(len(missing_meta) == 0),
        details={"missing_count": len(missing_meta), "examples": missing_meta[:25]},
        warn=(len(missing_meta) > 0),
    )

    report.add(
        "mapping_vs_metadata.segment_count_match",
        ok=(len(mismatches) == 0),
        details={"mismatch_count": len(mismatches), "examples": mismatches[:25]},
        warn=(len(mismatches) > 0),
    )

    return missing_meta, mismatches


def export_validation_video_issues_csv(
    out_csv_path: Path,
    multi_continent_videos: Sequence[str],
    metadata_duplicate_counts: Dict[str, int],
    missing_metadata_videos: Sequence[str],
    segment_count_mismatches: Sequence[Dict[str, Any]],
    missing_in_mapping_videos: Sequence[str],
    logger: logging.Logger,
) -> None:
    """Write a single CSV containing videos involved in selected checks."""
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "video",
                "claimed_segments",
                "derived_segments",
                "reason",
            ],
        )
        w.writeheader()

        # 1) mapping_csv.multi_continent_videos
        for vid in sorted(set(multi_continent_videos)):
            w.writerow(
                {
                    "category": "mapping_csv.multi_continent_videos",
                    "video": vid,
                    "claimed_segments": "",
                    "derived_segments": "",
                    "reason": "multiple_continents",
                }
            )

        # 2) mapping_metadata.duplicate_video_rows
        for vid in sorted(metadata_duplicate_counts.keys()):
            w.writerow(
                {
                    "category": "mapping_metadata.duplicate_video_rows",
                    "video": vid,
                    "claimed_segments": "",
                    "derived_segments": "",
                    "reason": f"duplicate_rows:{int(metadata_duplicate_counts.get(vid, 0))}",
                }
            )

        # 3) mapping_vs_metadata.missing_metadata_rows
        for vid in sorted(set(missing_metadata_videos)):
            w.writerow(
                {
                    "category": "mapping_vs_metadata.missing_metadata_rows",
                    "video": vid,
                    "claimed_segments": "",
                    "derived_segments": "",
                    "reason": "missing_metadata",
                }
            )

        # 4) mapping_vs_metadata.segment_count_match (mismatches only)
        for mm in segment_count_mismatches:
            vid = str(mm.get("video", "")).strip()
            if not vid:
                continue
            w.writerow(
                {
                    "category": "mapping_vs_metadata.segment_count_match",
                    "video": vid,
                    "claimed_segments": "" if mm.get("claimed") is None else str(mm.get("claimed")),
                    "derived_segments": "" if mm.get("derived") is None else str(mm.get("derived")),
                    "reason": str(mm.get("reason", "mismatch")),
                }
            )

        # 5) metadata_vs_mapping.missing_in_mapping
        for vid in sorted(set(missing_in_mapping_videos)):
            w.writerow(
                {
                    "category": "metadata_vs_mapping.missing_in_mapping",
                    "video": vid,
                    "claimed_segments": "",
                    "derived_segments": "",
                    "reason": "missing_in_mapping",
                }
            )

    logger.info(f"Validation video issues CSV written: {out_csv_path}")


# =============================================================================
# Reproducibility environment capture
# =============================================================================
def _package_version(package_name: str) -> str:
    """Return the installed package version, or a clear missing value."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"unavailable: {exc}"


def _run_command_quiet(command: List[str]) -> Dict[str, Any]:
    """Run a command and return stdout/stderr without raising."""
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": f"failed: {exc}"}


def _candidate_search_roots(data_dir: Path, yolo_dir: Path) -> List[Path]:
    roots: List[Path] = []
    for candidate in [Path.cwd(), data_dir, data_dir.parent, yolo_dir.parent]:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def _find_named_file(filename: str, roots: Sequence[Path]) -> Optional[Path]:
    """Find a named file while pruning large generated folders."""
    direct = Path(filename).expanduser()
    if direct.exists() and direct.is_file():
        return direct.resolve()

    skip_dirs = {".git", "__pycache__", "_output", "bbox", "seg", "frames", "videos", ".venv", "venv"}
    for root in roots:
        direct_under_root = root / filename
        if direct_under_root.exists() and direct_under_root.is_file():
            return direct_under_root.resolve()

        try:
            for current_root, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                if filename in filenames:
                    return (Path(current_root) / filename).resolve()
        except Exception:
            continue
    return None


def _read_small_text(path: Path, max_chars: int = 20000) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return {"path": str(path), "sha256": sha256_file(path), "text": text, "truncated": truncated}
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def _extract_yaml_like_values(text: str, keys: Sequence[str]) -> Dict[str, str]:
    """Extract simple key: value lines from tracker/config text without interpreting the file."""
    out: Dict[str, str] = {}
    wanted = set(keys)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in wanted:
            out[key] = value.strip()
    return out


def _ultralytics_default_settings() -> Dict[str, Any]:
    keys = [
        "imgsz",
        "iou",
        "max_det",
        "agnostic_nms",
        "classes",
        "rect",
        "half",
        "dnn",
        "augment",
        "vid_stride",
        "stream_buffer",
        "visualize",
        "retina_masks",
        "embed",
        "conf",
        "tracker",
        "device",
    ]

    defaults: Dict[str, Any] = {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT  # type: ignore
        if isinstance(DEFAULT_CFG_DICT, dict):
            defaults.update(DEFAULT_CFG_DICT)
    except Exception:
        pass

    if not defaults:
        try:
            from ultralytics.utils import DEFAULT_CFG  # type: ignore
            if isinstance(DEFAULT_CFG, dict):
                defaults.update(DEFAULT_CFG)
            elif hasattr(DEFAULT_CFG, "__dict__"):
                defaults.update(DEFAULT_CFG.__dict__)
        except Exception:
            pass

    return {key: defaults.get(key, "not found") for key in keys}


def _collect_torch_cuda_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = str(torch.version.cuda)
        info["torch_cudnn_version"] = torch.backends.cudnn.version()
        info["torch_cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        info["torch_cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        info["torch_cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        try:
            info["torch_use_deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        except Exception as exc:
            info["torch_use_deterministic_algorithms"] = f"unavailable: {exc}"
        try:
            info["torch_allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
            info["torch_allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
        except Exception as exc:
            info["torch_tf32_settings"] = f"unavailable: {exc}"

        devices: List[Dict[str, Any]] = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
        info["cuda_devices"] = devices
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _collect_model_weight_info(model_name: str, roots: Sequence[Path]) -> Dict[str, Any]:
    path = _find_named_file(model_name, roots)
    if path is None:
        return {"requested": model_name, "found": False, "path": None, "sha256": None}

    item: Dict[str, Any] = {
        "requested": model_name,
        "found": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(path))
        item["model_task"] = str(getattr(model, "task", "not available"))
        item["model_overrides"] = getattr(model, "overrides", {})
    except Exception as exc:
        item["model_load_error"] = str(exc)
    return item


def _collect_tracker_info(tracker_name: str, roots: Sequence[Path]) -> Dict[str, Any]:
    path = _find_named_file(tracker_name, roots)
    if path is None:
        return {"requested": tracker_name, "found": False, "path": None}

    text_info = _read_small_text(path)
    important = [
        "tracker_type",
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
        "gmc_method",
        "proximity_thresh",
        "appearance_thresh",
        "with_reid",
        "model",
    ]
    return {
        "requested": tracker_name,
        "found": True,
        "path": str(path),
        "sha256": text_info.get("sha256"),
        "important_fields": _extract_yaml_like_values(str(text_info.get("text", "")), important),
        "raw_text": text_info.get("text", ""),
        "raw_text_truncated": text_info.get("truncated", False),
    }


def _format_reproducibility_text(info: Dict[str, Any]) -> str:
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 90)
        lines.append(title)
        lines.append("=" * 90)

    section("SYSTEM")
    for key, value in info.get("system", {}).items():
        lines.append(f"{key}: {value}")

    section("PACKAGE VERSIONS")
    for key, value in info.get("packages", {}).items():
        lines.append(f"{key}: {value}")

    section("CUDA, cuDNN, GPU")
    lines.append(json.dumps(info.get("cuda", {}), indent=2, default=str))
    lines.append("nvidia-smi:")
    lines.append(json.dumps(info.get("nvidia_smi", {}), indent=2, default=str))
    lines.append("nvcc:")
    lines.append(json.dumps(info.get("nvcc", {}), indent=2, default=str))

    section("MODEL WEIGHTS")
    for item in info.get("model_weights", []):
        lines.append(json.dumps(item, indent=2, default=str))

    section("ULTRALYTICS DEFAULT INFERENCE SETTINGS")
    lines.append(json.dumps(info.get("ultralytics_defaults", {}), indent=2, default=str))

    section("EXPLICIT TRACK CALL PARAMETERS")
    lines.append(json.dumps(info.get("explicit_track_call_parameters", {}), indent=2, default=str))

    section("TRACKER CONFIGURATION")
    for item in info.get("trackers", []):
        lines.append(json.dumps(item, indent=2, default=str))

    section("PROJECT CONFIG FILES")
    for item in info.get("project_config_files", []):
        lines.append(json.dumps(item, indent=2, default=str))

    section("DETERMINISTIC ENVIRONMENT VARIABLES")
    for key, value in info.get("deterministic_environment_variables", {}).items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines).lstrip() + "\n"


def collect_reproducibility_environment(
    data_dir: Path,
    yolo_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
    report: Report,
) -> Dict[str, Any]:
    """Capture exact detection/tracking environment details requested by reviewers."""
    repro_cfg = CONFIG.get("reproducibility", {})
    roots = _candidate_search_roots(data_dir, yolo_dir)

    packages = [
        "ultralytics",
        "torch",
        "torchvision",
        "torchaudio",
        "opencv-python",
        "opencv-contrib-python",
        "numpy",
        "pandas",
        "polars",
        "pytubefix",
        "yt-dlp",
        "moviepy",
        "PyYAML",
        "scipy",
        "plotly",
    ]

    project_config_files: List[Dict[str, Any]] = []
    for name in repro_cfg.get("project_config_candidates", []):
        found = _find_named_file(str(name), roots)
        if found is not None:
            project_config_files.append(_read_small_text(found))

    model_infos = [
        _collect_model_weight_info(str(name), roots)
        for name in repro_cfg.get("model_candidates", [])
    ]

    tracker_infos = [
        _collect_tracker_info(str(name), roots)
        for name in repro_cfg.get("tracker_candidates", [])
    ]

    info: Dict[str, Any] = {
        "system": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "working_directory": str(Path.cwd()),
            "data_dir": str(data_dir),
            "yolo_dir": str(yolo_dir),
        },
        "packages": {package: _package_version(package) for package in packages},
        "cuda": _collect_torch_cuda_environment(),
        "nvidia_smi": _run_command_quiet(
            ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version,memory.total", "--format=csv,noheader"]
        ) if shutil.which("nvidia-smi") else {"available": False, "reason": "nvidia-smi not found"},
        "nvcc": _run_command_quiet(["nvcc", "--version"]) if shutil.which("nvcc") else {
            "available": False,
            "reason": "nvcc not found",
        },
        "project_config_files": project_config_files,
        "model_weights": model_infos,
        "ultralytics_defaults": _ultralytics_default_settings(),
        "explicit_track_call_parameters": {
            "call": "YOLO(model).track",
            "tracker": "bbox_tracker or seg_tracker YAML from project configuration",
            "persist": True,
            "conf": 0.0,
            "save": False,
            "device": "cuda when available, otherwise cpu",
            "track_buffer": "track_buffer_sec * rounded_video_fps when configured by the project code",
            "note": "Any setting not listed here should be copied from ultralytics_defaults or the tracker YAML above.",
        },
        "trackers": tracker_infos,
        "deterministic_environment_variables": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not set"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not set"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not set"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not set"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / str(repro_cfg.get("output_json", "reproducibility_environment.json"))
    txt_path = out_dir / str(repro_cfg.get("output_txt", "reproducibility_environment.txt"))
    json_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(_format_reproducibility_text(info), encoding="utf-8")

    logger.info(f"Reproducibility environment JSON written: {json_path}")
    logger.info(f"Reproducibility environment text written: {txt_path}")
    logger.info("\n=== Reproducibility summary ===")
    logger.info(f"Ultralytics: {info['packages'].get('ultralytics')}")
    logger.info(f"PyTorch: {info['packages'].get('torch')}")
    logger.info(f"OpenCV: {info['packages'].get('opencv-python')}")
    logger.info(f"CUDA available: {info.get('cuda', {}).get('torch_cuda_available')}")
    logger.info(f"Ultralytics defaults: {json.dumps(info.get('ultralytics_defaults', {}), default=str)}")

    report.add(
        "reproducibility_environment_written",
        ok=True,
        details={
            "json_path": str(json_path),
            "txt_path": str(txt_path),
            "ultralytics_version": info["packages"].get("ultralytics"),
            "torch_version": info["packages"].get("torch"),
            "opencv_python_version": info["packages"].get("opencv-python"),
            "cuda": info.get("cuda", {}),
            "model_weights": [
                {k: v for k, v in item.items() if k in {"requested", "found", "path", "sha256", "size_bytes"}}
                for item in model_infos
            ],
            "trackers": [
                {k: v for k, v in item.items() if k in {"requested", "found", "path", "sha256", "important_fields"}}
                for item in tracker_infos
            ],
        },
    )

    return info


# =============================================================================
# YOLO folder checks
# =============================================================================
@dataclass
class YoloFileInfo:
    path: str
    video_id: str
    start_time: float
    fps: float
    n_rows: int
    n_cols: int
    header: List[str]


def iter_csv_rowcount(path: Path, max_rows: Optional[int] = None) -> Tuple[int, int, List[str]]:
    """Count rows quickly without loading whole file; returns (rows, cols, header)."""
    n_rows = 0
    header: List[str] = []
    n_cols = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0, []
        n_cols = len(header)
        for _ in reader:
            n_rows += 1
            if max_rows is not None and n_rows >= max_rows:
                break
    return n_rows, n_cols, header


def check_yolo_folder(
    yolo_dir: Path,
    video_aggs: Dict[str, VideoAgg],
    report: Report,
    logger: logging.Logger,
) -> Tuple[List[YoloFileInfo], Dict[str, Any]]:
    rex = re.compile(CONFIG["yolo_filename_regex"])
    tol = float(YOLO_START_MATCH_TOLERANCE_S)

    # Build per-video set of known start times (from mapping)
    starts_by_video: Dict[str, Set[float]] = {}
    for vid, agg in video_aggs.items():
        starts_by_video[vid] = {s for (s, _e) in agg.segments}

    infos: List[YoloFileInfo] = []

    bad_name: List[str] = []
    unknown_video: List[str] = []
    unmatched_start: List[Dict[str, Any]] = []

    matched_segments: Set[Tuple[str, float]] = set()
    duplicate_match: List[Dict[str, Any]] = []

    if not yolo_dir.exists() or not yolo_dir.is_dir():
        report.add("yolo_dir.exists", ok=False, details={"path": str(yolo_dir)}, warn=False)
        return [], {"coverage": None}

    csv_files = sorted([p for p in yolo_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    report.add("yolo_dir.exists", ok=True, details={"path": str(yolo_dir), "n_csv": len(csv_files)})

    for p in tqdm(csv_files, total=len(csv_files), desc="Scanning YOLO CSVs", unit="file"):
        m = rex.match(p.name)
        if not m:
            bad_name.append(p.name)
            continue

        vid = m.group("video_id")
        st = float(m.group("start"))
        fps = float(m.group("fps"))

        if vid not in video_aggs:
            unknown_video.append(p.name)

        n_rows, n_cols, header = iter_csv_rowcount(p)
        infos.append(
            YoloFileInfo(
                path=str(p),
                video_id=vid,
                start_time=st,
                fps=fps,
                n_rows=n_rows,
                n_cols=n_cols,
                header=header[:50],
            )
        )

        if n_rows == 0:
            report.add("yolo_csv.non_empty", ok=False, details={"file": p.name}, warn=True)

        # Match start time to known segment start times
        known_starts = starts_by_video.get(vid, set())
        if known_starts:
            nearest = min(known_starts, key=lambda x: abs(x - st))
            if abs(nearest - st) <= tol:
                key = (vid, nearest)
                if key in matched_segments:
                    duplicate_match.append({"file": p.name, "video": vid, "start": st, "matched_start": nearest})
                matched_segments.add(key)
            else:
                unmatched_start.append(
                    {"file": p.name, "video": vid, "start": st,
                     "nearest_known_start": nearest, "delta": abs(nearest - st)}
                )
        else:
            unmatched_start.append({"file": p.name, "video": vid, "start": st, "reason": "no_known_segment_starts"})

    report.add(
        "yolo_filenames.pattern_match",
        ok=(len(bad_name) == 0),
        details={"bad_count": len(bad_name), "examples": bad_name[:50], "regex": CONFIG["yolo_filename_regex"]},
        warn=(len(bad_name) > 0),
    )

    report.add(
        "yolo_integrity.unknown_video_ids",
        ok=(len(unknown_video) == 0),
        details={"unknown_count": len(unknown_video), "examples": unknown_video[:50]},
        warn=(len(unknown_video) > 0),
    )

    report.add(
        "yolo_integrity.start_time_match",
        ok=(len(unmatched_start) == 0),
        details={"unmatched_count": len(unmatched_start), "examples": unmatched_start[:25], "tolerance_s": tol},
        warn=(len(unmatched_start) > 0),
    )

    report.add(
        "yolo_integrity.duplicate_segment_matches",
        ok=(len(duplicate_match) == 0),
        details={"duplicate_count": len(duplicate_match), "examples": duplicate_match[:25]},
        warn=(len(duplicate_match) > 0),
    )

    total_segments = sum(len(agg.segments) for agg in video_aggs.values())
    covered_segments = len(matched_segments)
    coverage = covered_segments / max(total_segments, 1)

    cov = {
        "total_segments": total_segments,
        "covered_segments": covered_segments,
        "coverage_pct": round(coverage * 100, 2),
    }
    report.add(
        "yolo_coverage.segment_coverage",
        ok=(coverage > 0.98) if total_segments > 0 else False,
        details=cov,
        warn=True,
    )

    return infos, cov


# =============================================================================
# Optional LaTeX checks
# =============================================================================
def _extract_numbers_from_latex(tex_path: Path, checks: List[Tuple[str, str, str]], report: Report) -> None:
    """Try to find paper-claimed numbers from main.tex using regex patterns."""
    text = tex_path.read_text(encoding="utf-8", errors="replace")

    for name, pattern, expected_key in checks:
        rex = re.compile(pattern)
        matches = rex.findall(text)
        if not matches:
            report.add(f"latex_claim:{name}", ok=False, details={"pattern": pattern, "reason": "no_match"}, warn=True)
            continue

        flat: List[str] = []
        for m in matches:
            if isinstance(m, tuple):
                flat.append(m[0])
            else:
                flat.append(m)

        if len(flat) > 1:
            report.add(
                f"latex_claim:{name}",
                ok=False,
                details={"pattern": pattern, "matches": flat[:25], "reason": "multiple_matches"},
                warn=True,
            )
            continue

        claimed_raw = flat[0]
        claimed_num = _num_from_str(claimed_raw)
        expected = CONFIG.get("expected", {}).get(expected_key)

        if claimed_num is None:
            report.add(f"latex_claim:{name}", ok=False, details={"claimed_raw": claimed_raw,
                                                                 "reason": "not_numeric"}, warn=True)
            continue

        if expected is None:
            report.add(f"latex_claim:{name}", ok=True, details={"claimed": claimed_num,
                                                                "expected": None, "note": "no_expected_config"})
            continue

        if isinstance(expected, int):
            ok = int(round(claimed_num)) == int(expected)
            report.add(f"latex_claim:{name}", ok=ok, details={"claimed": claimed_num,
                                                              "expected": expected}, warn=not ok)
        else:
            ok = abs(float(claimed_num) - float(expected)) <= 0.01
            report.add(
                f"latex_claim:{name}",
                ok=ok,
                details={"claimed": claimed_num, "expected": expected, "delta": float(claimed_num) - float(expected)},
                warn=not ok,
            )


# =============================================================================
# Pretty printing
# =============================================================================
def _print_mapping_summary(agg: MappingAgg, logger: logging.Logger) -> None:
    logger.info("\n=== Dataset summary (from mapping.csv) ===")
    logger.info(f"Rows (mapping): {fmt_int(agg.rows)}")
    logger.info(f"Unique locality+state+iso3 keys: {fmt_int(agg.unique_locality_state_iso3)}")
    logger.info(f"Unique countries/territories: {fmt_int(agg.unique_countries)}")
    logger.info(f"Upload records (sum of videos per row): {fmt_int(agg.upload_records)}")
    logger.info(f"Unique videos (global uploads): {fmt_int(agg.unique_videos)}")
    logger.info(f"Segment records (label entries): {fmt_int(agg.segment_records)}")
    logger.info(f"Duration: {fmt_int(int(round(agg.duration_s)))} s | {fmt_float(agg.duration_s/3600.0, 2)} h")


def _print_continent_table(agg: MappingAgg, logger: logging.Logger) -> None:
    order = CONFIG["allowed"]["continents"]
    seg_tot = sum(agg.continent_segment_records.get(c, 0) for c in order)
    dur_tot = sum(agg.continent_duration_s.get(c, 0.0) for c in order)

    logger.info("\n=== A) Continent segment distribution (label-entries; paper) ===")
    header = f"{'Continent':<14} {'Segments':>10} {'Seg%':>7} {'Dur(h)':>10} {'Dur%':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    for c in order:
        seg = int(agg.continent_segment_records.get(c, 0))
        dur_h = float(agg.continent_duration_s.get(c, 0.0)) / 3600.0
        seg_pct = 100.0 * seg / max(seg_tot, 1)
        dur_pct = 100.0 * (dur_h * 3600.0) / max(dur_tot, 1e-9)
        logger.info(f"{c:<14} {seg:>10} {seg_pct:>6.2f} {dur_h:>10.2f} {dur_pct:>6.2f}")

    logger.info(f"{'Total':<14} {seg_tot:>10} {100.0:>6.2f} {dur_tot/3600.0:>10.2f} {100.0:>6.2f}")


def _print_daynight_entries(agg: MappingAgg, logger: logging.Logger) -> None:
    order = CONFIG["allowed"]["continents"]
    logger.info("\n=== D) Continent x time-of-day label entries (paper) ===")
    header = f"{'Continent':<14} {'Day':>12} {'Night':>12} {'Total':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    tot_day = 0
    tot_night = 0
    for c in order:
        d = int(agg.continent_day_entries.get(c, 0))
        n = int(agg.continent_night_entries.get(c, 0))
        t = d + n
        tot_day += d
        tot_night += n
        d_pct = (100.0 * d / max(t, 1))
        n_pct = (100.0 * n / max(t, 1))
        logger.info(
            f"{c:<14} {fmt_int(d):>12} ({d_pct:>6.2f}%) {fmt_int(n):>12} ({n_pct:>6.2f}%) {fmt_int(t):>12}"
        )

    tot = tot_day + tot_night
    logger.info(
        f"{'Total':<14} {fmt_int(tot_day):>12} ({100.0*tot_day/max(tot, 1):>6.2f}%) "
        f"{fmt_int(tot_night):>12} ({100.0*tot_night/max(tot, 1):>6.2f}%) {fmt_int(tot):>12}"
    )


def _print_upload_daynight(agg: MappingAgg, logger: logging.Logger) -> None:
    cats = ["only_day", "only_night", "both_day_night", "unknown"]
    tot = sum(int(agg.global_upload_daynight.get(k, 0)) for k in cats)
    logger.info("\n=== E) Upload day/night composition (global; paper) ===")
    for k in cats:
        v = int(agg.global_upload_daynight.get(k, 0))
        logger.info(f"{k:<16} {fmt_int(v):>10} ({100.0*v/max(tot, 1):.2f}%)")
    logger.info(f"{'Total':<16} {fmt_int(tot):>10} (100.00%)")


# =============================================================================
# Main
# =============================================================================
def _resolve_path_maybe_absolute(base: Path, p: Any) -> Path:
    """If p is absolute, return it; else return base/p. Always .resolve()'d."""
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    return (base / pp).resolve()


def _infer_yolo_root(yolo_dir: Path) -> Path:
    """
    Try to create a nicer 'root' for manifest paths.
    If yolo_dir looks like .../data/bbox, root becomes the parent of 'data'.
    """
    try:
        if yolo_dir.parent.name == "data":
            return yolo_dir.parent.parent
    except Exception:
        pass
    return yolo_dir


def _manifest_relpath(p_abs: Path, data_dir: Path, yolo_root: Path, yolo_dir: Path) -> str:
    """
    Avoid ValueError from Path.relative_to when files aren't under data_dir.
    Prefer relative_to(data_dir), then yolo_root, then yolo_dir, else filename.
    """
    try:
        return p_abs.relative_to(data_dir).as_posix()
    except ValueError:
        pass
    try:
        return p_abs.relative_to(yolo_root).as_posix()
    except ValueError:
        pass
    try:
        return p_abs.relative_to(yolo_dir).as_posix()
    except ValueError:
        return p_abs.name


def main() -> int:
    data_dir = Path(CONFIG["paths"]["data_dir"]).resolve()
    out_dir = data_dir / "_output"
    logger = _setup_logger(out_dir)
    report = Report()

    logger.info(f"Data dir: {data_dir}")

    # ---------------------------------------------------------------------
    # 0) Resolve required paths
    # ---------------------------------------------------------------------
    mapping_path = _resolve_path_maybe_absolute(data_dir, CONFIG["paths"]["mapping_csv"])
    meta_path = _resolve_path_maybe_absolute(data_dir, CONFIG["paths"]["mapping_metadata_csv"])
    yolo_dir = _resolve_path_maybe_absolute(data_dir, CONFIG["paths"]["yolo_dir"])
    yolo_root = _infer_yolo_root(yolo_dir)

    tex_rel = CONFIG["paths"].get("main_tex")
    tex_path = _resolve_path_maybe_absolute(data_dir, tex_rel) if tex_rel else None

    report.add("path.exists.mapping_csv", mapping_path.exists(), {"path": str(mapping_path)}, warn=False)
    report.add("path.exists.mapping_metadata_csv", meta_path.exists(), {"path": str(meta_path)}, warn=False)
    report.add("path.exists.yolo_dir", yolo_dir.exists(), {"path": str(yolo_dir)}, warn=False)
    if tex_path is not None:
        report.add("path.exists.main_tex", tex_path.exists(), {"path": str(tex_path)}, warn=True)

    if CONFIG.get("reproducibility", {}).get("enabled", True):
        collect_reproducibility_environment(data_dir, yolo_dir, out_dir, logger, report)

    if not mapping_path.exists() or not meta_path.exists() or not yolo_dir.exists():
        logger.error("Missing required inputs. Edit CONFIG['paths'].")
        (out_dir / "validate_release_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return 1
    # ---------------------------------------------------------------------
    # 0b) Optional: export full overlap list and or interactively fix overlaps
    # ---------------------------------------------------------------------
    if EXPORT_OVERLAPS_FOUND_CSV:
        export_overlaps_found_csv(
            mapping_path=mapping_path,
            out_csv_path=(out_dir / OVERLAPS_FOUND_CSV_NAME),
            logger=logger,
            eps=float(OVERLAP_MATCH_EPS),
        )

    if FIX_OVERLAPS:
        fixed_path = out_dir / "mapping.fixed.csv"
        mapping_path = interactive_fix_overlaps_in_mapping_csv(
            mapping_path=mapping_path,
            output_path=fixed_path,
            logger=logger,
            inplace=bool(FIX_OVERLAPS_INPLACE),
            eps=float(OVERLAP_MATCH_EPS),
        )

    # ---------------------------------------------------------------------
    # 1) SHA256 manifest
    # ---------------------------------------------------------------------
    sha_lines: List[str] = []

    manifest_files: List[Path] = []
    manifest_files.extend([mapping_path, meta_path])

    yolo_manifest_files: List[Path] = []
    if yolo_dir.is_dir():
        # Only include a few safe suffixes, as in the original code
        for p in yolo_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".csv", ".txt", ".json", ".md"}:
                continue
            yolo_manifest_files.append(p)

    manifest_files.extend(sorted(yolo_manifest_files))

    for p in tqdm(manifest_files, desc="Computing SHA256", unit="file"):
        d = sha256_file(p)
        if p == mapping_path or p == meta_path:
            sha_lines.append(f"{d}  {p.name}")
        else:
            rel = _manifest_relpath(p.resolve(), data_dir=data_dir, yolo_root=yolo_root, yolo_dir=yolo_dir)
            sha_lines.append(f"{d}  {rel}")

    (out_dir / "sha256_manifest.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    report.add("sha256_manifest_written", True,
               {"path": str(out_dir / "sha256_manifest.txt"), "n_entries": len(sha_lines)})

    # ---------------------------------------------------------------------
    # 2) Parse mapping.csv and compute paper-aligned rollups
    # ---------------------------------------------------------------------
    mapping_agg, video_aggs = parse_mapping_csv(mapping_path, logger, report)

    multi_continent_videos = sorted(
        [vid for vid, agg in video_aggs.items() if agg.continents and len(agg.continents) > 1]
    )

    _print_mapping_summary(mapping_agg, logger)
    _print_continent_table(mapping_agg, logger)
    _print_daynight_entries(mapping_agg, logger)
    _print_upload_daynight(mapping_agg, logger)

    # ---------------------------------------------------------------------
    # 3) Compare to expected totals (if provided)
    # ---------------------------------------------------------------------
    exp = CONFIG.get("expected", {})

    def _cmp(name: str, computed: Any, expected_key: str, tol: float = 0.0) -> None:
        expected = exp.get(expected_key)
        if expected is None:
            report.add(f"expected:{name}", True, {"computed": computed, "expected": None, "note": "not_configured"})
            return
        if isinstance(expected, int):
            ok = int(computed) == int(expected)
            report.add(f"expected:{name}", ok, {"computed": int(computed), "expected": int(expected)}, warn=not ok)
        else:
            ok = abs(float(computed) - float(expected)) <= tol
            report.add(
                f"expected:{name}",
                ok,
                {"computed": float(computed), "expected": float(expected), "delta": float(computed) - float(expected)},
                warn=not ok,
            )

    _cmp("unique_videos", mapping_agg.unique_videos, "unique_videos")
    _cmp("upload_records", mapping_agg.upload_records, "upload_records")
    _cmp("segment_records", mapping_agg.segment_records, "segment_records")
    _cmp("duration_s", int(round(mapping_agg.duration_s)), "duration_s")
    _cmp("duration_h", round(mapping_agg.duration_s / 3600.0, 2), "duration_h", tol=0.01)
    _cmp("countries", mapping_agg.unique_countries, "countries")
    _cmp("rows_locality_state_iso3", mapping_agg.unique_locality_state_iso3, "rows_locality_state_iso3")

    # ---------------------------------------------------------------------
    # 4) Load mapping_metadata.csv and cross-check
    # ---------------------------------------------------------------------
    meta, metadata_duplicate_counts = load_mapping_metadata(meta_path, logger, report)

    missing_metadata_videos, segment_count_mismatches = compare_metadata_segments(video_aggs, meta, report)

    missing_in_mapping = [vid for vid in meta.keys() if vid not in video_aggs and vid != "#NAME?"]
    report.add(
        "metadata_vs_mapping.missing_in_mapping",
        ok=(len(missing_in_mapping) == 0),
        details={"missing_count": len(missing_in_mapping), "examples": missing_in_mapping[:25]},
        warn=(len(missing_in_mapping) > 0),
    )

    if EXPORT_VALIDATION_VIDEO_ISSUES_CSV:
        out_csv = out_dir / VALIDATION_VIDEO_ISSUES_CSV_NAME
        export_validation_video_issues_csv(
            out_csv_path=out_csv,
            multi_continent_videos=multi_continent_videos,
            metadata_duplicate_counts=metadata_duplicate_counts,
            missing_metadata_videos=missing_metadata_videos,
            segment_count_mismatches=segment_count_mismatches,
            missing_in_mapping_videos=missing_in_mapping,
            logger=logger,
        )
        report.add(
            "export.validation_video_issues_csv",
            ok=True,
            details={
                "path": str(out_csv),
                "counts": {
                    "mapping_csv.multi_continent_videos": len(multi_continent_videos),
                    "mapping_metadata.duplicate_video_rows": len(metadata_duplicate_counts),
                    "mapping_vs_metadata.missing_metadata_rows": len(missing_metadata_videos),
                    "mapping_vs_metadata.segment_count_match_mismatches": len(segment_count_mismatches),
                    "metadata_vs_mapping.missing_in_mapping": len(missing_in_mapping),
                },
            },
        )

    # ---------------------------------------------------------------------
    # 5) Check YOLO folder
    # ---------------------------------------------------------------------
    _infos, _cov = check_yolo_folder(yolo_dir, video_aggs, report, logger)

    # ---------------------------------------------------------------------
    # 6) Optional: validate LaTeX claimed numbers
    # ---------------------------------------------------------------------
    if tex_path is not None and tex_path.exists():
        _extract_numbers_from_latex(tex_path, CONFIG.get("latex_claim_checks", []), report)

    # ---------------------------------------------------------------------
    # 7) Write report
    # ---------------------------------------------------------------------
    report_path = out_dir / "validate_release_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    summ = report.summary()
    logger.info("\n=== Validation summary ===")
    logger.info(json.dumps(summ, indent=2))
    logger.info(f"Report: {report_path}")

    return 1 if summ.get("fail", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
