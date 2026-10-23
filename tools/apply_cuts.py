#!/usr/bin/env python3
"""Apply the confirmed cuts of cut_times.csv to the mapping and to the detection files (step 3 of the review of cuts).

A candidate marked "yes" in cut_times.csv at time t splits its segment in the mapping into a part that ends at
floor(t) and a part that starts at ceil(t), so one to two seconds around the cut are dropped. Parts shorter than
MIN_SEGMENT_LENGTH_SEC are discarded. The detection CSV of the segment is split at the same frames, with frame
numbers restarted at 1 and the true frame rate read from the video file (not the rounded one in the file name).

    python -m tools.apply_cuts [--videos FOLDER ...] [--bbox FOLDER ...]

Writes _output/mapping_split.csv, the split detection files in _output/bbox_split/ and logs; copy them back by hand
and delete the source files listed in _output/source_csvs_to_remove.csv. The folders of the videos and of the detection
files are the "videos" and (the bbox of) the "data" of the config unless the options give others (any number).
"""
import argparse
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402


# =========================
# FILES AND FOLDERS
# =========================

MAPPING_CSV = Path(common.get_configs("mapping"))
CUT_TIMES_CSV = Path("cut_times.csv")

# folders of the detection files: the data folders of the config, each also with its bbox subfolder
CSV_INPUT_DIRS = [Path(d).expanduser() for d in common.get_configs("data")]

OUTPUT_MAPPING_CSV = Path("_output/mapping_split.csv")
OUTPUT_LOG_CSV = Path("_output/mapping_split_log.csv")
UNMATCHED_CUTS_LOG_CSV = Path("_output/unmatched_cuts_log.csv")

CSV_OUTPUT_DIR = Path("_output/bbox_split")
CSV_SPLIT_LOG_CSV = Path("_output/csv_split_log.csv")
SOURCE_CSVS_TO_REMOVE_CSV = Path("_output/source_csvs_to_remove.csv")

# Folders holding the original {video_id}.mp4 files. Used to read the true
# (unrounded) frame rate, e.g. 29.97 instead of the rounded 30 that appears
# in the CSV file name.
VIDEO_DIRS = [Path(d).expanduser() for d in common.get_configs("videos")]

OVERWRITE_OUTPUTS = True

USE_ONLY_YES_ROWS = True

MIN_SEGMENT_LENGTH_SEC = 1

CSV_CHUNKSIZE = 500_000

# The fps in the CSV file name is rounded to an integer, but frame-count in
# the CSV follows the true frame rate (e.g. 29.97). Using the rounded value
# shifts the split point by about 0.03 s per second of video, i.e. several
# seconds for long segments, so the true fps is resolved per segment:
#   1. read from the original video file (VIDEO_DIRS), if available;
#   2. otherwise inferred from the cut timestamps, which lie on the frame
#      grid of the true fps (integer vs. NTSC 1000/1001 rate);
#   3. otherwise inferred from the highest frame-count in the source CSV.
# If none of these works, the rounded fps is used and the segment is
# flagged as "filename_unverified" in the CSV split log (or skipped, if
# SKIP_UNVERIFIED_FPS is True).
SKIP_UNVERIFIED_FPS = False
FRAME_GRID_TOLERANCE = 0.05
FRAME_COUNT_TOLERANCE = 2

# The paper states that mapping.csv keeps the manually selected segment
# end_time, while the processed pseudo-label CSVs are generated using
# end_time - 1 second. Keep this True to make split CSVs follow the same
# convention as the released CROWD files.
APPLY_PAPER_ENDPOINT_ADJUSTMENT_TO_CSV = True
PROCESSED_ENDPOINT_OFFSET_SEC = 1

# Force split CSV outputs to the per-segment pseudo-label schema described
# in the paper. Extra source columns are dropped; missing required columns
# raise an error.
FORCE_PAPER_CSV_SCHEMA = True

PAPER_CSV_COLUMNS = [
    "yolo-id",
    "x-center",
    "y-center",
    "width",
    "height",
    "unique-id",
    "confidence",
    "frame-count",
]


# =========================
# MAPPING COLUMNS
# =========================

NESTED_COLUMNS = [
    "time_of_day",
    "start_time",
    "end_time",
]

FLAT_COLUMNS = [
    "videos",
    "vehicle_type",
    "upload_date",
    "channel",
]


# =========================
# BASIC HELPERS
# =========================

def floor_time(value):
    return int(math.floor(float(value)))


def ceil_time(value):
    return int(math.ceil(float(value)))


def normalise_time_key(value):
    return round(float(value), 3)


def format_number(value):
    value = float(value)

    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))

    return str(value)


def parse_csv_file_name(csv_file):
    """
    Expected:
    {video_id}_{segment_start_time}_{fps}.csv

    Example:
    -0fvEC1AZRU_16619_30.csv
    """
    stem = Path(str(csv_file)).stem
    parts = stem.rsplit("_", 2)

    if len(parts) != 3:
        return None, None, None, None

    video_id = parts[0]

    try:
        segment_start_time = float(parts[1])
    except ValueError:
        return None, None, None, None

    fps_token = parts[2]

    try:
        fps = float(fps_token)
    except ValueError:
        return None, None, None, None

    if fps <= 0 or not math.isfinite(fps):
        return None, None, None, None

    return video_id, segment_start_time, fps, fps_token


def make_csv_name(video_id, segment_start_time, fps_token):
    return f"{video_id}_{int(segment_start_time)}_{fps_token}.csv"


def get_processed_csv_end_time(segment_start, segment_end):
    """
    Return the endpoint used for the detection/tracking CSV.

    The mapping keeps the manually selected end_time. The paper describes
    the actual processed CSV/video endpoint as end_time - 1 second, with a
    safeguard that leaves the endpoint unchanged if the adjusted interval
    would be non-positive.
    """
    segment_start = int(segment_start)
    segment_end = int(segment_end)

    if not APPLY_PAPER_ENDPOINT_ADJUSTMENT_TO_CSV:
        return segment_end

    adjusted_end = segment_end - int(PROCESSED_ENDPOINT_OFFSET_SEC)

    if adjusted_end <= segment_start:
        return segment_end

    return adjusted_end


def get_frame_bounds_for_processed_csv(segment_start, segment_end, original_start, fps):
    """
    Convert a mapping segment into source CSV frame-count bounds.

    Source and output CSV frame-count values are segment-local and 1-based.
    For a segment starting at t_start, frame 1 corresponds to t_start.
    """
    processed_end = get_processed_csv_end_time(
        segment_start=segment_start,
        segment_end=segment_end,
    )

    start_offset_frames = int(round((int(segment_start) - int(original_start)) * fps))
    end_offset_frames = int(round((processed_end - int(original_start)) * fps))

    return {
        "processed_end": processed_end,
        "start_offset_frames": start_offset_frames,
        "first_old_frame": start_offset_frames + 1,
        "last_old_frame": end_offset_frames,
    }


def validate_paper_csv_columns(csv_path, columns):
    missing_columns = [
        column
        for column in PAPER_CSV_COLUMNS
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required paper CSV column(s): "
            f"{missing_columns}"
        )


# =========================
# TRUE FPS RESOLUTION
# =========================

_video_fps_cache = {}


def ntsc_fps(rounded_fps):
    return float(rounded_fps) * 1000.0 / 1001.0


def is_plausible_fps(true_fps, rounded_fps):
    return abs(float(true_fps) - float(rounded_fps)) <= 1.0


def probe_video_fps(video_id):
    """Return the unrounded fps of {video_id}.mp4, or None if not found."""
    if video_id in _video_fps_cache:
        return _video_fps_cache[video_id]

    fps = None

    for video_dir in VIDEO_DIRS:
        video_path = Path(video_dir) / f"{video_id}.mp4"

        if not video_path.is_file():
            continue

        try:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            value = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
        except Exception as error:
            print(f"  could not read fps from {video_path}: {error}")
            continue

        if value > 0 and math.isfinite(value):
            fps = value
            break

    _video_fps_cache[video_id] = fps
    return fps


def is_on_frame_grid(offset_sec, fps):
    frames = float(offset_sec) * float(fps)
    return abs(frames - round(frames)) <= FRAME_GRID_TOLERANCE


def infer_fps_from_cut_times(cut_times, original_start, rounded_fps):
    """
    Cut timestamps are frame times of the true fps, e.g. 230.13 s after the
    segment start is frame 6897 at 29.97 fps but not a whole frame at 30 fps.
    Return the candidate that every cut agrees with, or None if ambiguous.
    """
    candidates = [float(rounded_fps), ntsc_fps(rounded_fps)]
    offsets = [float(t) - float(original_start) for t in cut_times]

    if not offsets:
        return None

    matching = [
        fps
        for fps in candidates
        if all(is_on_frame_grid(offset, fps) for offset in offsets)
    ]

    if len(matching) == 1:
        return matching[0]

    return None


def read_max_frame_count(csv_path):
    max_frame = None

    for chunk in pd.read_csv(csv_path, usecols=["frame-count"], chunksize=CSV_CHUNKSIZE):
        value = pd.to_numeric(chunk["frame-count"], errors="coerce").max()

        if pd.notna(value) and (max_frame is None or value > max_frame):
            max_frame = float(value)

    return max_frame


def infer_fps_from_frame_count(csv_path, original_start, original_end, rounded_fps):
    """
    The source CSV covers original_start .. processed_end, so its highest
    frame-count is close to duration * true_fps. Trailing frames without
    detections can only lower it, so a value above the NTSC frame total
    proves the integer rate, and a value right at the NTSC total points to
    the NTSC rate. Anything lower is inconclusive.
    """
    max_frame = read_max_frame_count(csv_path)

    if max_frame is None:
        return None

    processed_end = get_processed_csv_end_time(original_start, original_end)
    duration = processed_end - int(original_start)

    ntsc_total = duration * ntsc_fps(rounded_fps)

    if max_frame > ntsc_total + FRAME_COUNT_TOLERANCE:
        return float(rounded_fps)

    if max_frame >= ntsc_total - FRAME_COUNT_TOLERANCE:
        return ntsc_fps(rounded_fps)

    return None


def resolve_true_fps(plan, csv_info):
    """Return (fps, source) for the frame arithmetic of one source CSV."""
    rounded_fps = float(csv_info["fps"])

    video_fps = probe_video_fps(plan["video_id"])

    if video_fps is not None:
        if is_plausible_fps(video_fps, rounded_fps):
            return video_fps, "video"

        print(
            f"  video fps {video_fps:.3f} does not match file name fps "
            f"{csv_info['fps_token']}; ignoring video"
        )

    grid_fps = infer_fps_from_cut_times(
        cut_times=plan["cut_times"],
        original_start=plan["original_start"],
        rounded_fps=rounded_fps,
    )

    if grid_fps is not None:
        return grid_fps, "cut_times"

    count_fps = infer_fps_from_frame_count(
        csv_path=csv_info["path"],
        original_start=plan["original_start"],
        original_end=plan["original_end"],
        rounded_fps=rounded_fps,
    )

    if count_fps is not None:
        return count_fps, "csv_frame_count"

    return rounded_fps, "filename_unverified"


# =========================
# LIST PARSING AND FORMATTING
# =========================

def parse_flat_list(value):
    text = str(value).strip()

    if text == "" or text.lower() == "nan":
        return []

    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"Expected flat list, got: {text[:100]}")

    inner = text[1:-1].strip()

    if inner == "":
        return []

    return [item.strip() for item in inner.split(",")]


def parse_number_token(token):
    token = str(token).strip()

    if token == "":
        return None

    value = float(token)

    if math.isclose(value, round(value), abs_tol=1e-9):
        return int(round(value))

    return value


def parse_nested_number_list(value):
    """
    Parses:
    [[0,6639,9696],[25],[100,500]]
    """
    text = str(value).strip()

    if text == "" or text.lower() == "nan":
        return []

    if text == "[]":
        return []

    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"Expected nested list, got: {text[:100]}")

    result = []
    current_group = []
    token = ""
    inside_group = False

    for char in text[1:-1]:
        if char == "[":
            inside_group = True
            current_group = []
            token = ""

        elif char == "]":
            if token.strip() != "":
                current_group.append(parse_number_token(token))

            result.append(current_group)
            current_group = []
            token = ""
            inside_group = False

        elif char == ",":
            if inside_group:
                if token.strip() != "":
                    current_group.append(parse_number_token(token))
                    token = ""

        else:
            if inside_group:
                token += char

    return result


def format_flat_list(values):
    return "[" + ",".join(str(value) for value in values) + "]"


def format_nested_number_list(values):
    groups = []

    for group in values:
        group_text = "[" + ",".join(format_number(value) for value in group) + "]"
        groups.append(group_text)

    return "[" + ",".join(groups) + "]"


def parse_mapping_row(row):
    parsed = {}

    for column in FLAT_COLUMNS:
        parsed[column] = parse_flat_list(row[column])

    for column in NESTED_COLUMNS:
        parsed[column] = parse_nested_number_list(row[column])

    return parsed


def validate_mapping_lists(row_index, parsed):
    videos = parsed["videos"]
    expected_len = len(videos)

    for column_name in [
        "time_of_day",
        "start_time",
        "end_time",
        "vehicle_type",
        "upload_date",
        "channel",
    ]:
        values = parsed[column_name]

        if len(values) != expected_len:
            raise ValueError(
                f"Row {row_index}: videos has length {expected_len}, "
                f"but {column_name} has length {len(values)}"
            )

    for video_index in range(expected_len):
        video_id = videos[video_index]

        start_len = len(parsed["start_time"][video_index])
        end_len = len(parsed["end_time"][video_index])
        tod_len = len(parsed["time_of_day"][video_index])

        if start_len != end_len or start_len != tod_len:
            raise ValueError(
                f"Row {row_index}, video {video_id}: "
                f"start_time length={start_len}, "
                f"end_time length={end_len}, "
                f"time_of_day length={tod_len}"
            )


# =========================
# CUT LOADING
# =========================

def load_yes_cuts(cut_times_csv):
    df = pd.read_csv(cut_times_csv, dtype={"csv_file": str})

    required_columns = ["csv_file", "absolute_time_sec"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"cut_times.csv is missing required column: {column}")

    if USE_ONLY_YES_ROWS:
        if "is_correct" not in df.columns:
            raise ValueError(
                "USE_ONLY_YES_ROWS=True, but cut_times.csv has no is_correct column."
            )

        df = df[
            df["is_correct"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        ].copy()

    cuts_by_segment = {}

    for _, row in df.iterrows():
        csv_file = str(row["csv_file"])
        raw_cut_time = float(row["absolute_time_sec"])

        video_id, segment_start_time, fps, fps_token = parse_csv_file_name(csv_file)

        if video_id is None:
            print(f"Skipping malformed csv_file: {csv_file}")
            continue

        key = (
            video_id,
            normalise_time_key(segment_start_time),
        )

        cuts_by_segment.setdefault(key, []).append({
            "csv_file": csv_file,
            "raw_cut_time": raw_cut_time,
            "previous_end_time": floor_time(raw_cut_time),
            "next_start_time": ceil_time(raw_cut_time),
        })

    for key in cuts_by_segment:
        deduped = {}

        for item in cuts_by_segment[key]:
            dedupe_key = (
                item["previous_end_time"],
                item["next_start_time"],
            )
            deduped[dedupe_key] = item

        cuts_by_segment[key] = sorted(
            deduped.values(),
            key=lambda item: item["raw_cut_time"],
        )

    return cuts_by_segment


def get_valid_cuts_for_segment(cuts, segment_start, segment_end):
    valid_cuts = []

    for item in cuts:
        next_start = int(item["next_start_time"])

        # A cut inside the first second (e.g. 43.5 in a segment starting at
        # 43) is kept: make_split_segments drops the empty piece before it
        # and the segment then starts at ceil(cut).
        if float(item["raw_cut_time"]) <= segment_start:
            continue

        # A cut inside the last second lies after the processed CSV
        # endpoint (end_time - 1), so it is already excluded.
        if next_start >= segment_end:
            continue

        valid_cuts.append(item)

    return valid_cuts


def make_split_segments(segment_start, segment_end, valid_cuts):
    """
    For a fractional cut:
    previous end = floor(cut)
    next start   = ceil(cut)

    Example:
    original segment: 43 to 200
    cuts: 143.533, 154.367

    mapping output:
    43 to 143
    144 to 154
    155 to 200

    The split mapping keeps these end_time values. When writing the
    pseudo-label CSVs, each segment uses the paper's processed endpoint
    convention: CSV endpoint = mapping end_time - 1 second, unless that
    would make the processed interval non-positive.
    """
    segment_start = int(segment_start)
    segment_end = int(segment_end)

    segments = []

    current_start = segment_start

    for cut in valid_cuts:
        previous_end = int(cut["previous_end_time"])
        next_start = int(cut["next_start_time"])

        if previous_end - current_start >= MIN_SEGMENT_LENGTH_SEC:
            segments.append({
                "start": current_start,
                "end": previous_end,
            })

        current_start = next_start

    if segment_end - current_start >= MIN_SEGMENT_LENGTH_SEC:
        segments.append({
            "start": current_start,
            "end": segment_end,
        })

    return segments


# =========================
# MAPPING SPLITTING
# =========================

def register_split_plan(split_plans, key, video_id, original_start, original_end, segments, cut_times):
    plan = {
        "video_id": video_id,
        "original_start": int(original_start),
        "original_end": int(original_end),
        "segments": segments,
        "cut_times": list(cut_times),
    }

    if key not in split_plans:
        split_plans[key] = plan
        return

    old_segments = split_plans[key]["segments"]

    if old_segments != segments:
        print(f"Warning: duplicate split plan differs for {key}")


def split_mapping_and_build_plans(mapping_csv, cut_times_csv):
    mapping_df = pd.read_csv(
        mapping_csv,
        dtype=str,
        keep_default_na=False,
    )

    cuts_by_segment = load_yes_cuts(cut_times_csv)

    split_plans = {}
    split_logs = []
    used_cut_keys = set()

    for row_index, row in mapping_df.iterrows():
        parsed = parse_mapping_row(row)
        validate_mapping_lists(row_index, parsed)

        videos = parsed["videos"]
        start_time = parsed["start_time"]
        end_time = parsed["end_time"]
        time_of_day = parsed["time_of_day"]

        for video_index, video_id in enumerate(videos):
            new_start_list = []
            new_end_list = []
            new_time_of_day_list = []

            for segment_index in range(len(start_time[video_index])):
                segment_start = int(start_time[video_index][segment_index])
                segment_end = int(end_time[video_index][segment_index])
                time_of_day_value = time_of_day[video_index][segment_index]

                key = (
                    video_id,
                    normalise_time_key(segment_start),
                )

                candidate_cuts = cuts_by_segment.get(key, [])

                valid_cuts = get_valid_cuts_for_segment(
                    cuts=candidate_cuts,
                    segment_start=segment_start,
                    segment_end=segment_end,
                )

                if not valid_cuts:
                    new_start_list.append(segment_start)
                    new_end_list.append(segment_end)
                    new_time_of_day_list.append(time_of_day_value)
                    continue

                segments = make_split_segments(
                    segment_start=segment_start,
                    segment_end=segment_end,
                    valid_cuts=valid_cuts,
                )

                for segment in segments:
                    new_start_list.append(segment["start"])
                    new_end_list.append(segment["end"])
                    new_time_of_day_list.append(time_of_day_value)

                register_split_plan(
                    split_plans=split_plans,
                    key=key,
                    video_id=video_id,
                    original_start=segment_start,
                    original_end=segment_end,
                    segments=segments,
                    cut_times=[cut["raw_cut_time"] for cut in valid_cuts],
                )

                used_cut_keys.add(key)

                for cut in valid_cuts:
                    split_logs.append({
                        "mapping_row_index": row_index,
                        "locality": row.get("locality", ""),
                        "country": row.get("country", ""),
                        "video_id": video_id,
                        "csv_file": cut["csv_file"],
                        "original_start_time": segment_start,
                        "original_end_time": segment_end,
                        "raw_cut_time": cut["raw_cut_time"],
                        "previous_end_time": cut["previous_end_time"],
                        "next_start_time": cut["next_start_time"],
                    })

            start_time[video_index] = new_start_list
            end_time[video_index] = new_end_list
            time_of_day[video_index] = new_time_of_day_list

        mapping_df.at[row_index, "videos"] = format_flat_list(parsed["videos"])
        mapping_df.at[row_index, "vehicle_type"] = format_flat_list(parsed["vehicle_type"])
        mapping_df.at[row_index, "upload_date"] = format_flat_list(parsed["upload_date"])
        mapping_df.at[row_index, "channel"] = format_flat_list(parsed["channel"])

        mapping_df.at[row_index, "start_time"] = format_nested_number_list(parsed["start_time"])
        mapping_df.at[row_index, "end_time"] = format_nested_number_list(parsed["end_time"])
        mapping_df.at[row_index, "time_of_day"] = format_nested_number_list(parsed["time_of_day"])

    OUTPUT_MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_MAPPING_CSV, index=False)

    log_df = pd.DataFrame(
        split_logs,
        columns=[
            "mapping_row_index",
            "locality",
            "country",
            "video_id",
            "csv_file",
            "original_start_time",
            "original_end_time",
            "raw_cut_time",
            "previous_end_time",
            "next_start_time",
        ],
    )

    OUTPUT_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(OUTPUT_LOG_CSV, index=False)

    all_cut_keys = set(cuts_by_segment.keys())
    unused_cut_keys = all_cut_keys - used_cut_keys

    print(f"Input mapping rows: {len(mapping_df)}")
    print(f"Verified cut segment keys: {len(all_cut_keys)}")
    print(f"Matched and split segment keys: {len(used_cut_keys)}")
    print(f"Unmatched cut segment keys: {len(unused_cut_keys)}")
    print(f"Total split cuts applied: {len(split_logs)}")
    print(f"Saved updated mapping: {OUTPUT_MAPPING_CSV}")
    print(f"Saved mapping split log: {OUTPUT_LOG_CSV}")

    unmatched_cut_rows = []

    for video_id, segment_start_time in sorted(unused_cut_keys):
        unmatched_cut_rows.append({
            "video_id": video_id,
            "segment_start_time": format_number(segment_start_time),
            "cut_key": f"{video_id}_{format_number(segment_start_time)}",
        })

    unmatched_cut_df = pd.DataFrame(
        unmatched_cut_rows,
        columns=[
            "video_id",
            "segment_start_time",
            "cut_key",
        ],
    )

    UNMATCHED_CUTS_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    unmatched_cut_df.to_csv(UNMATCHED_CUTS_LOG_CSV, index=False)
    print(f"Saved unmatched cuts log: {UNMATCHED_CUTS_LOG_CSV}")

    if unused_cut_keys:
        print()
        print(f"Unmatched cuts, all {len(unused_cut_keys)}:")

        for row in unmatched_cut_rows:
            print(f"  {row['cut_key']}")

    return split_plans


# =========================
# CSV SPLITTING
# =========================

def build_csv_index(csv_input_dirs):
    """Index the detection files of all folders (and of their bbox subfolders) by video and segment start."""
    csv_index = {}

    csv_paths = sorted(
        (p for d in csv_input_dirs for folder in (Path(d), Path(d) / "bbox") if folder.is_dir()
         for p in folder.glob("*.csv")),
        key=lambda path: path.name,
    )

    for csv_path in csv_paths:
        video_id, segment_start_time, fps, fps_token = parse_csv_file_name(csv_path.name)

        if video_id is None:
            continue

        key = (
            video_id,
            normalise_time_key(segment_start_time),
        )

        csv_index.setdefault(key, []).append({
            "path": csv_path,
            "video_id": video_id,
            "segment_start_time": segment_start_time,
            "fps": fps,
            "fps_token": fps_token,
        })

    return csv_index


def prepare_csv_output_folder():
    if OVERWRITE_OUTPUTS and CSV_OUTPUT_DIR.exists():
        shutil.rmtree(CSV_OUTPUT_DIR)

    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_output_segment_path(video_id, segment_start, fps_token):
    csv_name = make_csv_name(
        video_id=video_id,
        segment_start_time=segment_start,
        fps_token=fps_token,
    )

    return CSV_OUTPUT_DIR / csv_name


def initialise_empty_csv(output_path, columns):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=columns).to_csv(
        output_path,
        index=False,
    )


def split_one_csv_file(plan, csv_info, fps, fps_source):
    original_path = csv_info["path"]
    video_id = csv_info["video_id"]
    fps = float(fps)
    fps_token = csv_info["fps_token"]

    original_start = int(plan["original_start"])
    segments = plan["segments"]

    output_paths = []

    for segment in segments:
        output_path = get_output_segment_path(
            video_id=video_id,
            segment_start=segment["start"],
            fps_token=fps_token,
        )

        output_paths.append(output_path)

        if output_path.exists():
            output_path.unlink()

    wrote_header = {
        str(path): False
        for path in output_paths
    }

    detected_columns = None
    output_columns = None

    written_counts = {
        str(path): 0
        for path in output_paths
    }

    segment_frame_bounds = []

    for segment in segments:
        bounds = get_frame_bounds_for_processed_csv(
            segment_start=segment["start"],
            segment_end=segment["end"],
            original_start=original_start,
            fps=fps,
        )
        segment_frame_bounds.append(bounds)

    for chunk in pd.read_csv(original_path, chunksize=CSV_CHUNKSIZE):
        if detected_columns is None:
            detected_columns = list(chunk.columns)
            validate_paper_csv_columns(
                csv_path=original_path,
                columns=detected_columns,
            )

            if FORCE_PAPER_CSV_SCHEMA:
                output_columns = PAPER_CSV_COLUMNS
            else:
                output_columns = detected_columns

        original_frame_count = pd.to_numeric(
            chunk["frame-count"],
            errors="coerce",
        )

        for segment, output_path, bounds in zip(segments, output_paths, segment_frame_bounds):
            first_old_frame = int(bounds["first_old_frame"])
            last_old_frame = int(bounds["last_old_frame"])
            start_offset_frames = int(bounds["start_offset_frames"])

            mask = (
                original_frame_count.notna()
                & (original_frame_count >= first_old_frame)
                & (original_frame_count <= last_old_frame)
            )

            if not mask.any():
                continue

            part = chunk.loc[mask].copy()

            part["frame-count"] = (
                original_frame_count.loc[mask].astype(float)
                - start_offset_frames
            ).round().astype(int)

            if FORCE_PAPER_CSV_SCHEMA:
                part = part[PAPER_CSV_COLUMNS]

            part.to_csv(
                output_path,
                mode="a",
                header=not wrote_header[str(output_path)],
                index=False,
            )

            wrote_header[str(output_path)] = True
            written_counts[str(output_path)] += len(part)

    if detected_columns is None:
        detected_columns = []
        output_columns = PAPER_CSV_COLUMNS if FORCE_PAPER_CSV_SCHEMA else []

    for output_path in output_paths:
        if not wrote_header[str(output_path)]:
            initialise_empty_csv(
                output_path=output_path,
                columns=output_columns,
            )

    csv_logs = []

    for segment, output_path, bounds in zip(segments, output_paths, segment_frame_bounds):
        csv_logs.append({
            "source_csv": original_path.name,
            "output_csv": output_path.name,
            "video_id": video_id,
            "segment_start": int(segment["start"]),
            "segment_end": int(segment["end"]),
            "csv_processed_end": int(bounds["processed_end"]),
            "fps": fps_token,
            "fps_used": round(fps, 6),
            "fps_source": fps_source,
            "first_old_frame": int(bounds["first_old_frame"]),
            "last_old_frame": int(bounds["last_old_frame"]),
            "rows_written": written_counts[str(output_path)],
        })

    return csv_logs


def split_csv_files(split_plans):
    prepare_csv_output_folder()

    csv_index = build_csv_index(CSV_INPUT_DIRS)

    all_csv_logs = []
    source_csvs_to_remove = []
    fps_source_counts = {}

    print(f"CSV segments to split: {len(split_plans)}")
    print(f"Writing only split CSV files into: {CSV_OUTPUT_DIR}")

    for index, (key, plan) in enumerate(sorted(split_plans.items()), start=1):
        matches = csv_index.get(key, [])

        if not matches:
            print(f"[{index}/{len(split_plans)}] Missing source CSV for {key}")
            continue

        if len(matches) > 1:
            print(f"[{index}/{len(split_plans)}] Multiple CSVs found for {key}; using first")

        csv_info = matches[0]

        fps, fps_source = resolve_true_fps(plan, csv_info)
        fps_source_counts[fps_source] = fps_source_counts.get(fps_source, 0) + 1

        print(
            f"[{index}/{len(split_plans)}] Splitting {csv_info['path'].name} "
            f"(fps {fps:.3f}, from {fps_source})"
        )

        if fps_source == "filename_unverified" and SKIP_UNVERIFIED_FPS:
            print("  skipped: true fps could not be verified")
            continue

        csv_logs = split_one_csv_file(
            plan=plan,
            csv_info=csv_info,
            fps=fps,
            fps_source=fps_source,
        )

        all_csv_logs.extend(csv_logs)

        # The split files replace the source CSV. If no split file has the
        # source name (the first piece was dropped), the source must be
        # deleted by hand when the split files are copied back.
        output_names = {log["output_csv"] for log in csv_logs}

        if csv_info["path"].name not in output_names:
            source_csvs_to_remove.append({
                "source_csv": csv_info["path"].name,
                "source_path": str(csv_info["path"]),
                "replaced_by": ";".join(sorted(output_names)),
            })

        print(f"  created {len(csv_logs)} split CSV files")

    csv_log_df = pd.DataFrame(
        all_csv_logs,
        columns=[
            "source_csv",
            "output_csv",
            "video_id",
            "segment_start",
            "segment_end",
            "csv_processed_end",
            "fps",
            "fps_used",
            "fps_source",
            "first_old_frame",
            "last_old_frame",
            "rows_written",
        ],
    )

    CSV_SPLIT_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_log_df.to_csv(CSV_SPLIT_LOG_CSV, index=False)

    remove_df = pd.DataFrame(
        source_csvs_to_remove,
        columns=["source_csv", "source_path", "replaced_by"],
    )
    SOURCE_CSVS_TO_REMOVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    remove_df.to_csv(SOURCE_CSVS_TO_REMOVE_CSV, index=False)

    print(f"fps source counts: {fps_source_counts}")
    print(f"Saved split CSV folder: {CSV_OUTPUT_DIR}")
    print(f"Saved CSV split log: {CSV_SPLIT_LOG_CSV}")
    print(
        f"Source CSVs with no same-name replacement (delete these when "
        f"copying split files back): {len(remove_df)} -> {SOURCE_CSVS_TO_REMOVE_CSV}"
    )


# =========================
# MAIN
# =========================

def main():
    global VIDEO_DIRS, CSV_INPUT_DIRS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", help="folders with {video_id}.mp4, instead of the config")
    parser.add_argument("--bbox", nargs="+", help="folders with the detection files, instead of the config data")
    args = parser.parse_args()
    if args.videos:
        VIDEO_DIRS = [Path(d).expanduser() for d in args.videos]
    if args.bbox:
        CSV_INPUT_DIRS = [Path(d).expanduser() for d in args.bbox]

    split_plans = split_mapping_and_build_plans(
        mapping_csv=MAPPING_CSV,
        cut_times_csv=CUT_TIMES_CSV,
    )

    split_csv_files(split_plans)

    print()
    print("Done.")
    print(f"New mapping file: {OUTPUT_MAPPING_CSV}")
    print(f"New split CSV folder: {CSV_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
