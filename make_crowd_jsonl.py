#!/usr/bin/env python3
"""
Create a single JSON Lines index file from mapping.csv and mapping_metadata.csv.

Output design
-------------
- One JSON object per retained segment.
- Every line is valid JSON.
- Python-style list strings from the legacy CSV files are converted to real JSON
  arrays or objects in the output.
- The script is idempotent: running it again fully rebuilds the JSONL from the
  current CSV files and replaces the old JSONL atomically.

Default use
-----------
    python make_crowd_jsonl.py

Optional positional use, without argparse:
    python make_crowd_jsonl.py mapping.csv mapping_metadata.csv crowd_index.jsonl

Notes
-----
This converter uses ast.literal_eval only to read legacy CSV cells. Downstream
users of the released JSONL file do not need Python-specific parsing.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Edit these paths when running without positional arguments.
# -----------------------------------------------------------------------------
MAPPING_CSV = "mapping.csv"
MAPPING_METADATA_CSV = "mapping_metadata.csv"
OUTPUT_JSONL = "crowd_index.jsonl"

# The detector/tracker pipeline processes each segment to end_time - 1 second.
ENDPOINT_ADJUSTMENT_SECONDS = 1

# Privacy-safe default: do not release source-provided free-text metadata.
# Set to False only if you explicitly decide to release these fields.
EXCLUDE_SOURCE_FREE_TEXT_METADATA = True
SOURCE_FREE_TEXT_METADATA_FIELDS = {
    "title",
    "description",
    "channel",
    "chapters",
    "tags",
    "categories",
}

TIME_OF_DAY_NAMES = {
    0: "day",
    1: "night",
    "0": "day",
    "1": "night",
    "day": "day",
    "night": "night",
    "Day": "day",
    "Night": "night",
}

VEHICLE_TYPE_NAMES = {
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
    "0": "Car",
    "1": "Bus",
    "2": "Truck",
    "3": "Two-wheeler",
    "4": "Bicycle",
    "5": "Automated car",
    "6": "Electric scooter",
    "7": "Monowheel/unicycle",
    "8": "Automated bus",
    "9": "Automated truck",
    "10": "Automated two-wheeler",
    "11": "Non-electric scooter",
    "12": "Pedestrian",
}

# Known fields in mapping.csv that describe per-video or per-segment nested values.
CORE_LIST_LIKE_MAPPING_FIELDS = {
    "videos",
    "video",
    "video_id",
    "start_time",
    "end_time",
    "time_of_day",
    "vehicle_type",
    "upload_date",
    "channel",
}

# Fields that should usually appear under the location object when present.
LOCATION_FIELD_CANDIDATES = [
    "locality",
    "locality_aliases",
    "state",
    "region",
    "country",
    "iso3",
    "iso2",
    "continent",
    "lat",
    "latitude",
    "lon",
    "lng",
    "long",
    "longitude",
]

# -----------------------------------------------------------------------------
# CSV and parsing helpers
# -----------------------------------------------------------------------------
try:
    csv.field_size_limit(min(sys.maxsize, 1024 * 1024 * 1024))
except Exception:
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        s = value.strip()
        return s == "" or s.lower() in {"nan", "none", "null", "na", "n/a"}
    return False


def _clean_scalar(value: Any) -> Any:
    """Convert CSV scalar strings to JSON-friendly scalars."""
    if _is_missing(value):
        return None

    if isinstance(value, (int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    if not isinstance(value, str):
        return value

    s = value.strip()

    # Remove one layer of wrapping quotes only when they wrap the whole value.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        s = s[1:-1].strip()

    lower = s.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"nan", "none", "null", "na", "n/a", ""}:
        return None

    # Preserve ISO-like dates as strings.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", s):
        return s

    # Convert numeric strings when doing so is safe.
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except Exception:
            return s
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", s):
        try:
            f = float(s)
            if math.isfinite(f):
                return f
        except Exception:
            return s

    return s


def _json_friendly(value: Any) -> Any:
    """Recursively convert values to JSON-serialisable Python objects."""
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_friendly(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(v) for v in value]
    return _clean_scalar(value)


def _parse_cell(value: Any) -> Any:
    """Parse a CSV cell into a JSON-friendly Python object.

    Supports valid JSON, Python literal strings, and legacy bare-token list cells
    such as [abc123,def456].
    """
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        return _clean_scalar(value)

    s = value.strip()
    if s == "":
        return None

    # Valid JSON first.
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return _json_friendly(json.loads(s))
        except Exception:
            pass

    # Python literal strings, used in the legacy CSV.
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return _json_friendly(ast.literal_eval(s))
        except Exception:
            pass

    # Bare-token list fallback: [abc,def] -> ["abc", "def"].
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if inner == "":
            return []
        return [_clean_scalar(part.strip().strip("'").strip('"')) for part in inner.split(",")]

    return _clean_scalar(s)


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def _extract_youtube_id(value: Any) -> Optional[str]:
    """Extract a YouTube id from an id, URL, or watch URL string."""
    if _is_missing(value):
        return None
    s = str(value).strip().strip("'").strip('"')

    # Standard watch URL.
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", s)
    if m:
        return m.group(1)

    # youtu.be short URL.
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", s)
    if m:
        return m.group(1)

    # Bare id.
    if re.fullmatch(r"[A-Za-z0-9_-]{6,}", s):
        return s

    return None


def _parse_video_ids(value: Any) -> List[str]:
    parsed = _parse_cell(value)
    ids: List[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            vid = _extract_youtube_id(item)
            if vid:
                ids.append(vid)
    else:
        vid = _extract_youtube_id(parsed)
        if vid:
            ids.append(vid)

    # Fallback for awkward cells.
    if not ids and isinstance(value, str):
        ids.extend(re.findall(r"[A-Za-z0-9_-]{6,}", value))

    # Preserve order and remove duplicates.
    seen = set()
    out = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def _get_group_for_video(value: Any, video_index: int, n_videos: int) -> List[Any]:
    """Return the per-video group from a nested mapping column."""
    parsed = _parse_cell(value)
    if parsed is None:
        return []

    if not isinstance(parsed, list):
        return [parsed]

    if not parsed:
        return []

    has_nested = any(isinstance(item, list) for item in parsed)

    # Typical shape: one outer list per video, each containing segment values.
    if has_nested and len(parsed) == n_videos:
        return _ensure_list(parsed[video_index]) if video_index < len(parsed) else []

    # Single video with a flat or nested list.
    if n_videos == 1:
        if has_nested and len(parsed) == 1:
            return _ensure_list(parsed[0])
        return parsed

    # Per-video scalar list, e.g. upload_date = [date1,date2,...].
    if len(parsed) == n_videos and video_index < len(parsed):
        return _ensure_list(parsed[video_index])

    # Fallback.
    if video_index < len(parsed):
        return _ensure_list(parsed[video_index])
    return []


def _get_segment_value(row: Dict[str, Any], field: str, video_index: int, n_videos: int, segment_index: int) -> Any:
    group = _get_group_for_video(row.get(field), video_index, n_videos)
    if not group:
        return None
    if segment_index < len(group):
        return _json_friendly(group[segment_index])
    return _json_friendly(group[0])


def _to_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def _format_number_for_id(value: Any) -> str:
    f = _to_float(value)
    if f is None:
        return "missing"
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    s = ("%.6f" % f).rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _processed_duration(start_time: Any, end_time: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    start = _to_float(start_time)
    end = _to_float(end_time)
    if start is None or end is None:
        return None, None, None

    processed_end = end - ENDPOINT_ADJUSTMENT_SECONDS
    if processed_end <= start:
        processed_end = end

    duration = max(0.0, processed_end - start)
    return processed_end, duration, duration / 60.0


def _normalise_date(value: Any) -> Any:
    """Keep upload date strings stable and JSON-friendly without inventing recording dates."""
    if _is_missing(value):
        return None
    return _clean_scalar(value)


def _scalar_mapping_value(row: Dict[str, Any], field: str) -> Any:
    return _json_friendly(_parse_cell(row.get(field)))


def _build_location(row: Dict[str, Any], fieldnames: Iterable[str]) -> Dict[str, Any]:
    location: Dict[str, Any] = {}
    lower_to_actual = {name.lower(): name for name in fieldnames}

    for candidate in LOCATION_FIELD_CANDIDATES:
        actual = lower_to_actual.get(candidate.lower())
        if actual is None:
            continue
        value = _scalar_mapping_value(row, actual)
        if value is not None:
            key = candidate
            if key == "latitude":
                key = "lat"
            elif key in {"lng", "long", "longitude"}:
                key = "lon"
            location[key] = value

    return location


def _build_mapping_extra(row: Dict[str, Any], fieldnames: Iterable[str]) -> Dict[str, Any]:
    location_fields = {x.lower() for x in LOCATION_FIELD_CANDIDATES}
    core_fields = {x.lower() for x in CORE_LIST_LIKE_MAPPING_FIELDS}
    extra: Dict[str, Any] = {}

    for field in fieldnames:
        fl = field.lower()
        if fl in location_fields or fl in core_fields:
            continue
        value = _scalar_mapping_value(row, field)
        if value is not None:
            extra[field] = value

    return extra


def _clean_metadata_row(row: Dict[str, Any], fieldnames: Iterable[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for field in fieldnames:
        fl = field.lower().strip()
        if fl in {"id", "video", "video_id"}:
            continue
        if EXCLUDE_SOURCE_FREE_TEXT_METADATA and fl in SOURCE_FREE_TEXT_METADATA_FIELDS:
            continue

        value = _json_friendly(_parse_cell(row.get(field)))
        if value is not None:
            metadata[field] = value

    return metadata


def _build_metadata_index(metadata_csv: Path) -> Dict[str, Dict[str, Any]]:
    if not metadata_csv.exists():
        return {}

    fieldnames, rows = _read_csv_rows(metadata_csv)
    index: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        vid = None
        for key in ("id", "video_id", "video"):
            if key in row:
                vid = _extract_youtube_id(row.get(key))
                if vid:
                    break

        if not vid:
            continue

        cleaned = _clean_metadata_row(row, fieldnames)
        if vid in index:
            # Merge duplicate metadata rows, later non-empty values take precedence.
            merged = dict(index[vid])
            merged.update(cleaned)
            index[vid] = merged
        else:
            index[vid] = cleaned

    return index


def _make_segment_record(
    row: Dict[str, Any],
    fieldnames: List[str],
    row_number: int,
    video_id: str,
    video_index: int,
    n_videos: int,
    segment_index: int,
    duplicate_suffix: int,
    metadata_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    start_time = _get_segment_value(row, "start_time", video_index, n_videos, segment_index)
    end_time = _get_segment_value(row, "end_time", video_index, n_videos, segment_index)

    start_float = _to_float(start_time)
    end_float = _to_float(end_time)
    if start_float is None or end_float is None:
        return None
    if end_float <= start_float:
        return None

    processed_end, processed_duration_s, processed_duration_min = _processed_duration(start_float, end_float)

    time_of_day = _get_segment_value(row, "time_of_day", video_index, n_videos, segment_index)
    vehicle_type = _get_segment_value(row, "vehicle_type", video_index, n_videos, segment_index)
    upload_date_mapping = _get_segment_value(row, "upload_date", video_index, n_videos, segment_index)

    metadata = metadata_index.get(video_id, {})
    upload_date_metadata = metadata.get("upload_date")
    upload_date = _normalise_date(upload_date_mapping if upload_date_mapping is not None else upload_date_metadata)

    start_id = _format_number_for_id(start_float)
    end_id = _format_number_for_id(end_float)
    base_segment_id = f"{video_id}_{start_id}_{end_id}"
    segment_id = base_segment_id if duplicate_suffix == 0 else f"{base_segment_id}_{duplicate_suffix}"

    time_of_day_name = TIME_OF_DAY_NAMES.get(time_of_day, TIME_OF_DAY_NAMES.get(str(time_of_day), None))
    vehicle_type_name = VEHICLE_TYPE_NAMES.get(vehicle_type, VEHICLE_TYPE_NAMES.get(str(vehicle_type), None))

    location = _build_location(row, fieldnames)
    mapping_extra = _build_mapping_extra(row, fieldnames)

    record: Dict[str, Any] = {
        "record_type": "crowd_segment",
        "schema_version": "1.0",
        "segment_id": segment_id,
        "video_id": video_id,
        "source": {
            "platform": "YouTube",
            "watch_url": f"https://www.youtube.com/watch?v={video_id}",
            "underlying_video_redistributed": False,
        },
        "upload": {
            "upload_date": upload_date,
            "recording_date": None,
            "recording_date_provenance": None,
            "recording_date_uncertainty": "not_available",
        },
        "segment": {
            "start_time_s": start_float,
            "end_time_s": end_float,
            "processed_end_time_s": processed_end,
            "endpoint_adjustment_s": ENDPOINT_ADJUSTMENT_SECONDS,
            "processed_duration_s": processed_duration_s,
            "processed_duration_min": processed_duration_min,
        },
        "labels": {
            "time_of_day_code": time_of_day,
            "time_of_day_name": time_of_day_name,
            "vehicle_type_code": vehicle_type,
            "vehicle_type_name": vehicle_type_name,
        },
        "location": location,
        "mapping_row": {
            "row_number": row_number,
        },
        "automatic_outputs": {
            "bbox_csv_expected_prefix": f"{video_id}_{start_id}_",
            "type": "YOLOv11x and BoT-SORT detection/tracking pseudo-labels",
            "object_level_ground_truth": False,
        },
    }

    if metadata:
        record["upload_metadata"] = metadata
    if mapping_extra:
        record["mapping_extra"] = mapping_extra

    return _json_friendly(record)


def build_records(mapping_csv: Path, metadata_csv: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_index = _build_metadata_index(metadata_csv)

    fieldnames, rows = _read_csv_rows(mapping_csv)
    records: List[Dict[str, Any]] = []
    skipped_rows_without_video = 0
    skipped_segments_bad_times = 0
    missing_metadata_videos = set()
    seen_segment_ids: Counter[str] = Counter()

    for row_index, row in enumerate(rows, start=1):
        video_ids = _parse_video_ids(row.get("videos") or row.get("video_id") or row.get("video"))
        if not video_ids:
            skipped_rows_without_video += 1
            continue

        n_videos = len(video_ids)

        for video_index, video_id in enumerate(video_ids):
            starts = _get_group_for_video(row.get("start_time"), video_index, n_videos)
            ends = _get_group_for_video(row.get("end_time"), video_index, n_videos)
            segment_count = max(len(starts), len(ends))

            if video_id not in metadata_index:
                missing_metadata_videos.add(video_id)

            for segment_index in range(segment_count):
                start_value = starts[segment_index] if segment_index < len(starts) else None
                end_value = ends[segment_index] if segment_index < len(ends) else None
                base_id = f"{video_id}_{_format_number_for_id(start_value)}_{_format_number_for_id(end_value)}"
                duplicate_suffix = seen_segment_ids[base_id]
                seen_segment_ids[base_id] += 1

                record = _make_segment_record(
                    row=row,
                    fieldnames=fieldnames,
                    row_number=row_index,
                    video_id=video_id,
                    video_index=video_index,
                    n_videos=n_videos,
                    segment_index=segment_index,
                    duplicate_suffix=duplicate_suffix,
                    metadata_index=metadata_index,
                )
                if record is None:
                    skipped_segments_bad_times += 1
                    continue
                records.append(record)

    records.sort(key=lambda r: (str(r.get("video_id", "")), str(r.get("segment_id", ""))))

    total_duration_s = sum(
        float(r.get("segment", {}).get("processed_duration_s") or 0.0)
        for r in records
        if isinstance(r.get("segment"), dict)
    )

    summary = {
        "mapping_rows": len(rows),
        "jsonl_records": len(records),
        "unique_uploads_in_jsonl": len({r.get("video_id") for r in records}),
        "total_processed_duration_s": round(total_duration_s, 2),
        "total_processed_duration_h": round(total_duration_s / 3600.0, 2),
        "skipped_mapping_rows_without_video": skipped_rows_without_video,
        "skipped_segments_with_bad_times": skipped_segments_bad_times,
        "videos_missing_metadata_count": len(missing_metadata_videos),
        "videos_missing_metadata_sample": sorted(missing_metadata_videos)[:25],
    }

    return records, summary


def write_jsonl_atomic(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                f.write("\n")
        os.replace(tmp_name, output_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _resolve_paths_from_argv() -> Tuple[Path, Path, Path]:
    args = sys.argv[1:]
    mapping_csv = Path(args[0]) if len(args) >= 1 else Path(MAPPING_CSV)
    metadata_csv = Path(args[1]) if len(args) >= 2 else Path(MAPPING_METADATA_CSV)
    output_jsonl = Path(args[2]) if len(args) >= 3 else Path(OUTPUT_JSONL)
    return mapping_csv, metadata_csv, output_jsonl


def main() -> None:
    mapping_csv, metadata_csv, output_jsonl = _resolve_paths_from_argv()

    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    records, summary = build_records(mapping_csv, metadata_csv)
    write_jsonl_atomic(records, output_jsonl)

    print(f"Wrote {summary['jsonl_records']:,} JSONL records to {output_jsonl}")
    print(f"Unique uploads: {summary['unique_uploads_in_jsonl']:,}")
    print(f"Total processed duration: {summary['total_processed_duration_h']:,} h")
    if summary.get("videos_missing_metadata_count"):
        print(f"Videos missing metadata: {summary['videos_missing_metadata_count']:,}")
    if summary.get("skipped_mapping_rows_without_video"):
        print(f"Rows skipped without video id: {summary['skipped_mapping_rows_without_video']:,}")
    if summary.get("skipped_segments_with_bad_times"):
        print(f"Segments skipped because of invalid times: {summary['skipped_segments_with_bad_times']:,}")


if __name__ == "__main__":
    main()
