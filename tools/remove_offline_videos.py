#!/usr/bin/env python3
"""
remove_offline_videos.py

Interactive tool to delete video information from mapping.csv.

Keeps list columns aligned.

Always aligned (per video occurrence):
- videos
- upload_date (if present)
- channel (if present)
- views (if present)
- date_updated (if present)
- title (if present)

Aligned per segment (nested lists per video occurrence):
- time_of_day
- start_time
- end_time
- vehicle_type (supports flat list or nested list)

If a row ends up with zero videos after deletion, the whole row is removed.
If an id column exists (serial number), it is reassigned sequentially.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from custom_logger import CustomLogger


def _parse_log_level(name: str) -> int:
    level_name = (name or "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def setup_terminal_logging(py_logger: logging.Logger) -> int:
    """
    Ensure logs are visible in the terminal.

    Control verbosity with LOG_LEVEL (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = _parse_log_level(os.getenv("LOG_LEVEL", "INFO"))

    py_logger.setLevel(level)
    py_logger.propagate = False

    fmt = "%(message)s"
    if level <= logging.DEBUG:
        fmt = "%(levelname)s: %(message)s"

    formatter = logging.Formatter(fmt)

    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in py_logger.handlers)
    if not has_stream_handler:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        py_logger.addHandler(handler)
    else:
        for h in py_logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level)
                h.setFormatter(formatter)

    return level


logger = CustomLogger(__name__)
setup_terminal_logging(logger.logger)

# Allow very large CSV fields
try:
    csv.field_size_limit(min(sys.maxsize, 1024 * 1024 * 1024))
except Exception:
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass


PER_VIDEO_LIST_COLS = ["upload_date", "channel", "views", "date_updated", "title"]


def safe_eval_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []

    s = x.strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    if (len(s) >= 2) and (s[0] == s[-1]) and s[0] in {"'", '"'}:
        s = s[1:-1].strip()

    if s == "" or s == "[]":
        return []

    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            return [p for p in parts if p != ""]
        return []


def _to_int_maybe(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except Exception:
        return None


def _to_float_maybe(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            f = float(x)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def norm_list_of_lists_int(cell: Any) -> List[List[int]]:
    outer = safe_eval_list(cell)
    out: List[List[int]] = []
    for item in outer:
        if isinstance(item, list):
            inner_vals: List[int] = []
            for v in item:
                iv = _to_int_maybe(v)
                if iv is not None:
                    inner_vals.append(iv)
            out.append(inner_vals)
        else:
            out.append([])
    return out


def norm_list_of_lists_float(cell: Any) -> List[List[float]]:
    outer = safe_eval_list(cell)
    out: List[List[float]] = []
    for item in outer:
        if isinstance(item, list):
            inner_vals: List[float] = []
            for v in item:
                fv = _to_float_maybe(v)
                if fv is not None:
                    inner_vals.append(float(fv))
            out.append(inner_vals)
        else:
            out.append([])
    return out


def norm_flat_list_int(cell: Any) -> List[int]:
    outer = safe_eval_list(cell)
    out: List[int] = []
    for item in outer:
        iv = _to_int_maybe(item)
        out.append(iv if iv is not None else 0)
    return out


def norm_flat_list_str(cell: Any) -> List[str]:
    outer = safe_eval_list(cell)
    out: List[str] = []
    for item in outer:
        s = "" if item is None else str(item).strip()
        out.append(s)
    return out


def _fmt_float(x: float) -> str:
    if not math.isfinite(x):
        return "0"
    v = round(float(x), 6)
    if abs(v) < 1e-12:
        v = 0.0
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s != "" else "0"


def _fmt_atom(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return _fmt_float(x)

    s = str(x)

    # Keep ids and channel ids unquoted (allows '-' and '_')
    if re.fullmatch(r"[A-Za-z0-9_-]+", s):
        return s

    # Fallback for anything else
    return json.dumps(s, ensure_ascii=False, separators=(",", ":"))


def to_compact_list(obj: Any) -> str:
    if isinstance(obj, list):
        return "[" + ",".join(to_compact_list(x) if isinstance(x, list) else _fmt_atom(x) for x in obj) + "]"
    return _fmt_atom(obj)


def load_mapping(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        rows: List[Dict[str, Any]] = []
        for r in reader:
            vids = [str(v).strip() for v in safe_eval_list(r.get("videos")) if str(v).strip() != ""]

            tod = norm_list_of_lists_int(r.get("time_of_day"))
            st = norm_list_of_lists_float(r.get("start_time"))
            en = norm_list_of_lists_float(r.get("end_time"))

            # vehicle_type can be flat or nested
            vehicle_type = None
            vehicle_type_is_nested = False
            if "vehicle_type" in fieldnames:
                vt_raw = safe_eval_list(r.get("vehicle_type"))
                vehicle_type_is_nested = any(isinstance(x, list) for x in vt_raw)
                if vehicle_type_is_nested:
                    vehicle_type = norm_list_of_lists_int(r.get("vehicle_type"))
                else:
                    vehicle_type = norm_flat_list_int(r.get("vehicle_type"))

            # lists aligned with videos
            per_video: Dict[str, Any] = {}
            for col in PER_VIDEO_LIST_COLS:
                if col not in fieldnames:
                    continue
                raw_list = safe_eval_list(r.get(col))
                if isinstance(raw_list, list) and len(raw_list) == len(vids):
                    if col in {"upload_date", "views", "date_updated"}:
                        per_video[col] = norm_flat_list_int(r.get(col))
                    else:
                        per_video[col] = norm_flat_list_str(r.get(col))
                else:
                    per_video[col] = None

            rows.append(
                {
                    "raw": r,
                    "videos": vids,
                    "time_of_day": tod,
                    "start_time": st,
                    "end_time": en,
                    "vehicle_type": vehicle_type,
                    "vehicle_type_is_nested": vehicle_type_is_nested,
                    "per_video": per_video,
                }
            )

    return fieldnames, rows  # type: ignore


def write_mapping(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rd in rows:
            rd["raw"]["videos"] = to_compact_list(rd["videos"])
            rd["raw"]["time_of_day"] = to_compact_list(rd["time_of_day"])
            rd["raw"]["start_time"] = to_compact_list(rd["start_time"])
            rd["raw"]["end_time"] = to_compact_list(rd["end_time"])

            if rd["vehicle_type"] is not None and "vehicle_type" in fieldnames:
                rd["raw"]["vehicle_type"] = to_compact_list(rd["vehicle_type"])

            for col, val in rd["per_video"].items():
                if col in fieldnames and val is not None:
                    rd["raw"][col] = to_compact_list(val)

            writer.writerow(rd["raw"])


def _loc_string(raw: Dict[str, str]) -> str:
    cont = (raw.get("continent") or "").strip()
    country = (raw.get("country") or "").strip()
    state = (raw.get("state") or "").strip()
    locality = (raw.get("locality") or "").strip()
    iso3 = (raw.get("iso3") or "").strip()
    parts = [p for p in [locality, state, country, cont] if p]
    loc = ", ".join(parts) if parts else "(unknown location)"
    if iso3:
        loc += " [" + iso3 + "]"
    return loc


def remove_outer_at_index(rd: Dict[str, Any], outer_i: int) -> None:
    del rd["videos"][outer_i]
    if outer_i < len(rd["time_of_day"]):
        del rd["time_of_day"][outer_i]
    if outer_i < len(rd["start_time"]):
        del rd["start_time"][outer_i]
    if outer_i < len(rd["end_time"]):
        del rd["end_time"][outer_i]

    vt = rd.get("vehicle_type")
    if vt is not None and outer_i < len(vt):
        del vt[outer_i]

    for _, val in rd["per_video"].items():
        if val is not None and outer_i < len(val):
            del val[outer_i]


def delete_all(video_id: str, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    removed_video_slots = 0
    removed_segments = 0

    for rd in rows:
        idxs = [i for i, v in enumerate(rd["videos"]) if v == video_id]
        if not idxs:
            continue

        for outer_i in reversed(idxs):
            segs = 0
            if outer_i < len(rd["start_time"]) and outer_i < len(rd["end_time"]):
                segs = min(len(rd["start_time"][outer_i]), len(rd["end_time"][outer_i]))
            removed_segments += segs

            remove_outer_at_index(rd, outer_i)
            removed_video_slots += 1

    return removed_video_slots, removed_segments


def list_instances(video_id: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []

    for row_i, rd in enumerate(rows):
        for outer_i, vid in enumerate(rd["videos"]):
            if vid != video_id:
                continue

            starts = rd["start_time"][outer_i] if outer_i < len(rd["start_time"]) else []
            ends = rd["end_time"][outer_i] if outer_i < len(rd["end_time"]) else []
            tods = rd["time_of_day"][outer_i] if outer_i < len(rd["time_of_day"]) else []

            vt = rd.get("vehicle_type")
            vt_nested = bool(rd.get("vehicle_type_is_nested"))
            vt_value = None
            if vt is not None:
                if vt_nested and outer_i < len(vt) and isinstance(vt[outer_i], list):
                    vt_value = vt[outer_i]
                else:
                    vt_value = vt[outer_i] if outer_i < len(vt) else None

            n = min(len(starts), len(ends))
            for inner_i in range(n):
                vehicle_type_val = None
                if vt is not None:
                    if vt_nested and isinstance(vt_value, list):
                        vehicle_type_val = vt_value[inner_i] if inner_i < len(vt_value) else None
                    else:
                        vehicle_type_val = vt_value

                instances.append(
                    {
                        "row_i": row_i,
                        "outer_i": outer_i,
                        "inner_i": inner_i,
                        "location": _loc_string(rd["raw"]),
                        "start": starts[inner_i],
                        "end": ends[inner_i],
                        "time_of_day": tods[inner_i] if inner_i < len(tods) else None,
                        "vehicle_type": vehicle_type_val,
                    }
                )

    return instances


def delete_one(inst: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    rd = rows[inst["row_i"]]
    outer_i = inst["outer_i"]
    inner_i = inst["inner_i"]

    if outer_i < len(rd["start_time"]) and inner_i < len(rd["start_time"][outer_i]):
        del rd["start_time"][outer_i][inner_i]
    if outer_i < len(rd["end_time"]) and inner_i < len(rd["end_time"][outer_i]):
        del rd["end_time"][outer_i][inner_i]
    if outer_i < len(rd["time_of_day"]) and inner_i < len(rd["time_of_day"][outer_i]):
        del rd["time_of_day"][outer_i][inner_i]

    vt = rd.get("vehicle_type")
    vt_nested = bool(rd.get("vehicle_type_is_nested"))
    if vt is not None and vt_nested:
        if outer_i < len(vt) and isinstance(vt[outer_i], list) and inner_i < len(vt[outer_i]):
            del vt[outer_i][inner_i]

    starts = rd["start_time"][outer_i] if outer_i < len(rd["start_time"]) else []
    ends = rd["end_time"][outer_i] if outer_i < len(rd["end_time"]) else []
    if min(len(starts), len(ends)) == 0:
        remove_outer_at_index(rd, outer_i)


def _prompt(text: str) -> str:
    logger.info(text)
    return input("> ").strip()


def _detect_id_column(fieldnames: List[str]) -> Optional[str]:
    preferred = ["id", "ID", "Id", "index", "idx", "serial", "sr_no", "srno"]
    for name in preferred:
        if name in fieldnames:
            return name
    return None


def _reassign_ids(rows: List[Dict[str, Any]], id_col: str) -> None:
    ints: List[int] = []
    for rd in rows:
        iv = _to_int_maybe(rd["raw"].get(id_col))
        if iv is not None:
            ints.append(iv)

    start = 0 if ints and min(ints) == 0 else 1

    for i, rd in enumerate(rows):
        rd["raw"][id_col] = str(start + i)


def _prune_empty_rows_and_reindex(fieldnames: List[str], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    before = len(rows)
    kept = [rd for rd in rows if len(rd.get("videos", [])) > 0]
    removed = before - len(kept)
    if removed > 0:
        logger.info("Removed {} row(s) with zero videos.", removed)

    id_col = _detect_id_column(fieldnames)
    if id_col and kept:
        _reassign_ids(kept, id_col)
        logger.info("Reassigned {} sequentially.", id_col)

    return kept


def main() -> None:
    mapping_path = _prompt("Enter path to mapping.csv:")
    if not mapping_path:
        logger.error("No path provided.")
        return

    try:
        logger.info("Loading mapping from {}", mapping_path)
        fieldnames, rows = load_mapping(mapping_path)
    except FileNotFoundError:
        logger.error("File not found: {}", mapping_path)
        return
    except Exception as e:
        logger.error("Failed to load mapping: {}", e)
        return

    logger.info("Loaded {} rows with {} columns.", len(rows), len(fieldnames))

    video_id = _prompt("Enter video_id:")
    if not video_id:
        logger.error("No video_id provided.")
        return

    if not any(video_id in rd["videos"] for rd in rows):
        logger.error("video_id not found: {}", video_id)
        return

    total_occurrences = sum(rd["videos"].count(video_id) for rd in rows)
    logger.info("Found {} occurrence(s) of {}", total_occurrences, video_id)

    logger.info("")
    logger.info("Delete mode:")
    logger.info("  1) Delete all segments for this video_id (every occurrence)")
    logger.info("  2) Delete a particular segment instance")
    mode = _prompt("Select 1 or 2:")

    if mode == "1":
        removed_slots, removed_segs = delete_all(video_id, rows)
        logger.info(
            "Removed {} video occurrence(s) and about {} segment(s) for {}.",
            removed_slots,
            removed_segs,
            video_id,
        )

    elif mode == "2":
        instances = list_instances(video_id, rows)
        if not instances:
            logger.warning("No segment instances found for this video_id.")
            return

        logger.info("")
        logger.info("Instances:")
        for i, inst in enumerate(instances, start=1):
            tod = inst["time_of_day"]
            veh = inst["vehicle_type"]
            tod_s = "?" if tod is None else str(tod)
            veh_s = "" if veh is None else ", vehicle_type {}".format(veh)
            logger.info(
                "{:>4}. {} | start {} end {} | time_of_day {}{}",
                i,
                inst["location"],
                _fmt_float(inst["start"]),
                _fmt_float(inst["end"]),
                tod_s,
                veh_s,
            )

        sel_raw = _prompt("Type the number to delete:")
        if not sel_raw.isdigit():
            logger.error("Invalid selection.")
            return
        sel = int(sel_raw)
        if sel < 1 or sel > len(instances):
            logger.warning("Selection out of range.")
            return

        delete_one(instances[sel - 1], rows)
        logger.info("Deleted selected segment instance.")

    else:
        logger.error("Invalid choice.")
        return

    rows = _prune_empty_rows_and_reindex(fieldnames, rows)

    logger.info("")
    logger.info("Save options:")
    logger.info("  1) Overwrite the existing mapping.csv")
    logger.info("  2) Save to a new file")
    save_mode = _prompt("Select 1 or 2:")

    if save_mode == "1":
        out_path = mapping_path
    elif save_mode == "2":
        out_path = _prompt("Enter output csv path:")
        if not out_path:
            logger.error("No output path provided.")
            return
    else:
        logger.error("Invalid choice.")
        return

    write_mapping(out_path, fieldnames, rows)
    logger.info("Saved: {}", out_path)


if __name__ == "__main__":
    main()
