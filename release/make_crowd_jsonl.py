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
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
from release import split_large_files  # noqa: E402
from utils.core.parsing import ENDPOINT_ADJUSTMENT_SECONDS, parse_videos  # noqa: E402

# -----------------------------------------------------------------------------
# Edit these paths when running without positional arguments.
# -----------------------------------------------------------------------------
MAPPING_CSV = "mapping.csv"
MAPPING_METADATA_CSV = "mapping_metadata.csv"
OUTPUT_JSONL = "crowd_index.jsonl"

# Privacy-safe default: do not release source-provided free-text metadata.
# Set to False only if you explicitly decide to release these fields.
EXCLUDE_SOURCE_FREE_TEXT_METADATA = True
SOURCE_FREE_TEXT_METADATA_FIELDS = {
    "title",
    "description",
    "chapters",
    "tags",
    "categories",
}

TIME_OF_DAY_NAMES = {0: "day", 1: "night"}

VEHICLE_TYPE_NAMES = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Two-wheeler",
    4: "Bicycle",
    5: "Automated car",
    6: "Electric scooter",
    7: "Monowheel/unicycle",
    8: "Emergency vehicle",
    9: "Automated bus",
    10: "Automated truck",
    11: "Automated two-wheeler",
    12: "Non-electric scooter",
    13: "Pedestrian",
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

REQUIRED_JSONL_FIELDS = {
    "video_id", "segment_id", "youtube_url", "start_time_s", "end_time_s",
    "upload_date", "time_of_day_code", "time_of_day_name",
    "vehicle_type_code", "vehicle_type_name", "location",
    "mapping_row_number", "metadata", "mapping_extra",
}
REQUIRED_LOCATION_FIELDS = {"locality", "country", "iso3", "continent", "lat", "lon"}

# -----------------------------------------------------------------------------
# CSV and parsing helpers
# -----------------------------------------------------------------------------
csv.field_size_limit(1024 * 1024 * 1024)  # long list cells; fits a C long everywhere


def _is_missing(value: Any) -> bool:
    """True for None, NaN and empty or nan/none/null/na/n/a strings."""
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
        return int(s)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", s):
        f = float(s)
        if math.isfinite(f):  # 1e999 stays a string
            return f

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
    """Parse a CSV cell; string cells are cached (the same large list cell is read once per segment).

    Callers treat the result as read-only: they index it or copy it through _json_friendly.
    """
    if isinstance(value, str):
        return _parse_str_cell(value)
    return _parse_cell_uncached(value)


@lru_cache(maxsize=64)
def _parse_str_cell(value: str) -> Any:
    return _parse_cell_uncached(value)


def _parse_cell_uncached(value: Any) -> Any:
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return _json_friendly(ast.literal_eval(s))
        except (ValueError, SyntaxError, TypeError):
            pass

    # Bare-token list fallback: [abc,def] -> ["abc", "def"].
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if inner == "":
            return []
        return [_clean_scalar(part.strip().strip("'").strip('"')) for part in inner.split(",")]

    return _clean_scalar(s)


def _ensure_list(value: Any) -> List[Any]:
    """value as a list: [] for None, itself for a list, else [value]."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """The header and the rows (dicts of strings) of a CSV file."""
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
    """The value of one segment in a nested mapping column (the video's first value when it has fewer)."""
    group = _get_group_for_video(row.get(field), video_index, n_videos)
    if not group:
        return None
    if segment_index < len(group):
        return _json_friendly(group[segment_index])
    return _json_friendly(group[0])


def _to_float(value: Any) -> Optional[float]:
    """value as a finite float, else None."""
    if _is_missing(value):
        return None
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def _to_int(value: Any) -> Optional[int]:
    """Return an integer only when the input is finite and integral."""
    number = _to_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _format_number_for_id(value: Any) -> str:
    """A time for a segment id: 12 -> '12', 12.5 -> '12p5', None -> 'missing'."""
    f = _to_float(value)
    if f is None:
        return "missing"
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    s = ("%.6f" % f).rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _processed_duration(start_time: Any, end_time: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(processed end, duration s, duration min) as main.py trims a segment (float times kept, unlike
    utils.core.parsing.processed_segment_duration_seconds); Nones for non-numeric times."""
    start = _to_float(start_time)
    end = _to_float(end_time)
    if start is None or end is None:
        return None, None, None

    processed_end = end - ENDPOINT_ADJUSTMENT_SECONDS
    if processed_end <= start:
        processed_end = end

    duration = max(0.0, processed_end - start)
    return processed_end, duration, duration / 60.0


def _scalar_mapping_value(row: Dict[str, Any], field: str) -> Any:
    """A mapping cell as a JSON value."""
    return _json_friendly(_parse_cell(row.get(field)))


def _build_location(row: Dict[str, Any], fieldnames: Iterable[str]) -> Dict[str, Any]:
    """The location object of a mapping row (lat/lon under those names); required keys are null when missing."""
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

    # These keys are required by the public JSONL schema. Missing source values
    # are represented explicitly as JSON null rather than by omitting the key.
    for required in sorted(REQUIRED_LOCATION_FIELDS):
        location.setdefault(required, None)

    return location


def _build_mapping_extra(row: Dict[str, Any], fieldnames: Iterable[str]) -> Dict[str, Any]:
    """The other non-empty mapping columns of a row (gmp, population, ...), keyed by their names."""
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
    """The non-empty mapping_metadata.csv fields of a row, without ids and (by default) free text."""
    metadata: Dict[str, Any] = {}

    for field in fieldnames:
        fl = field.lower().strip()
        if fl in {"id", "video", "video_id"}:
            continue
        if EXCLUDE_SOURCE_FREE_TEXT_METADATA and fl in SOURCE_FREE_TEXT_METADATA_FIELDS:
            continue

        value = _scalar_mapping_value(row, field)
        if value is not None:
            metadata[field] = value

    return metadata


def _build_metadata_index(metadata_csv: Path) -> Dict[str, Dict[str, Any]]:
    """YouTube id -> its cleaned metadata (joined from its parts first when split); {} without the file."""
    split_large_files.join(metadata_csv)
    if not metadata_csv.exists():
        return {}

    fieldnames, rows = _read_csv_rows(metadata_csv)
    index: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        vid = None
        # In mapping_metadata.csv, `id` is the numeric row identifier and
        # `video` is the YouTube upload identifier. Prefer the latter.
        for key in ("video_id", "video", "id"):
            if key in row:
                vid = _extract_youtube_id(row.get(key))
                if vid:
                    break

        if not vid:
            continue

        cleaned = _clean_metadata_row(row, fieldnames)
        # Merge duplicate metadata rows, later non-empty values take precedence.
        index.setdefault(vid, {}).update(cleaned)

    return index


def _make_segment_record(
    row: Dict[str, Any],
    row_number: int,
    video_id: str,
    video_index: int,
    n_videos: int,
    segment_index: int,
    duplicate_suffix: int,
    metadata: Dict[str, Any],
    location: Dict[str, Any],
    mapping_extra: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """The JSONL record of one segment; None when its times are missing or end <= start."""
    start_time = _get_segment_value(row, "start_time", video_index, n_videos, segment_index)
    end_time = _get_segment_value(row, "end_time", video_index, n_videos, segment_index)

    start_float = _to_float(start_time)
    end_float = _to_float(end_time)
    if start_float is None or end_float is None or end_float <= start_float:
        return None

    time_of_day = _to_int(_get_segment_value(row, "time_of_day", video_index, n_videos, segment_index))
    vehicle_type = _to_int(_get_segment_value(row, "vehicle_type", video_index, n_videos, segment_index))
    upload_date = _get_segment_value(row, "upload_date", video_index, n_videos, segment_index)
    upload_date = _clean_scalar(upload_date if upload_date is not None else metadata.get("upload_date"))

    start_id = _format_number_for_id(start_float)
    end_id = _format_number_for_id(end_float)
    base_segment_id = f"{video_id}_{start_id}_{end_id}"
    segment_id = base_segment_id if duplicate_suffix == 0 else f"{base_segment_id}_{duplicate_suffix}"

    mapping_row_number = _to_int(row.get("id")) or row_number

    # Keep the released representation deliberately flat. These names and
    # nesting levels match both the manuscript schema and validation_release.py.
    record: Dict[str, Any] = {
        "video_id": video_id,
        "segment_id": segment_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "start_time_s": start_float,
        "end_time_s": end_float,
        "upload_date": upload_date,
        "time_of_day_code": time_of_day,
        "time_of_day_name": TIME_OF_DAY_NAMES.get(time_of_day),
        "vehicle_type_code": vehicle_type,
        "vehicle_type_name": VEHICLE_TYPE_NAMES.get(vehicle_type),
        "location": location,
        "mapping_row_number": mapping_row_number,
        "metadata": metadata,
        "mapping_extra": mapping_extra,
    }

    return _json_friendly(record)


def build_records(mapping_csv: Path, metadata_csv: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The JSONL records of all segments of the mapping, sorted by video and segment id, and a summary of counts."""
    metadata_index = _build_metadata_index(metadata_csv)

    fieldnames, rows = _read_csv_rows(mapping_csv)
    records: List[Dict[str, Any]] = []
    skipped_rows_without_video = 0
    skipped_segments_bad_times = 0
    missing_metadata_videos = set()
    seen_segment_ids: Counter[str] = Counter()

    for row_index, row in enumerate(rows, start=1):
        video_ids = parse_videos(row.get("videos") or row.get("video_id") or row.get("video"))
        if not video_ids:
            skipped_rows_without_video += 1
            continue

        n_videos = len(video_ids)
        # The same for every segment of the row; _json_friendly(record) copies them.
        location = _build_location(row, fieldnames)
        mapping_extra = _build_mapping_extra(row, fieldnames)

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
                    row_number=row_index,
                    video_id=video_id,
                    video_index=video_index,
                    n_videos=n_videos,
                    segment_index=segment_index,
                    duplicate_suffix=duplicate_suffix,
                    metadata=metadata_index.get(video_id, {}),
                    location=location,
                    mapping_extra=mapping_extra,
                )
                if record is None:
                    skipped_segments_bad_times += 1
                    continue
                records.append(record)

    records.sort(key=lambda r: (str(r.get("video_id", "")), str(r.get("segment_id", ""))))

    total_duration_s = sum(
        _processed_duration(r.get("start_time_s"), r.get("end_time_s"))[1] or 0.0
        for r in records
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
    """Write one compact, key-sorted JSON object per line to output_path via a temporary file."""
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


def validate_generated_schema(records: List[Dict[str, Any]]) -> None:
    """Fail before writing if a generated record drifts from the release schema."""
    for record_number, record in enumerate(records, start=1):
        missing = sorted(REQUIRED_JSONL_FIELDS - set(record))
        location = record.get("location")
        if not isinstance(location, dict):
            missing.append("location<object>")
        else:
            missing.extend(
                f"location.{field}"
                for field in sorted(REQUIRED_LOCATION_FIELDS - set(location))
            )
        if missing:
            raise ValueError(
                f"Generated JSONL record {record_number} ({record.get('segment_id')!r}) "
                f"is missing required fields: {missing}"
            )


def _resolve_paths_from_argv() -> Tuple[Path, Path, Path]:
    """Mapping, metadata and output paths from the optional positional arguments."""
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
    validate_generated_schema(records)
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
