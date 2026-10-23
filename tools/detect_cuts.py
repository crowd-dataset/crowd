#!/usr/bin/env python3
"""Screen the segments of the mapping for candidate cuts (step 1 of the review of cuts).

For each segment, FFmpeg scene-change detection (pixel-level scene score, threshold SCENE_THRESHOLD) runs on the
original video file over the segment only. Times closer than MERGE_NEARBY_SEC are merged into one candidate. The
candidates are appended to _output/cut_times.csv (absolute time in the upload, segment start, frame rate and the
window to play); copy them to cut_times.csv, where tools/review_cuts.py reads them. Segments that were screened
before (_output/cut_processing_tracker.csv) are not screened again.

    python -m tools.detect_cuts [--videos FOLDER ...]

The folders of the videos are the "videos" of the config unless --videos gives others (any number).
"""
import argparse
from pathlib import Path
from datetime import datetime
import math
import os
import re
import subprocess

import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402


# =========================
# FILES AND FOLDERS
# =========================

MAPPING_CSV = Path(common.get_configs("mapping"))

VIDEO_DIRS = [Path(d).expanduser() for d in common.get_configs("videos")]

# Existing files are kept. New rows are appended.
OUTPUT_CSV = Path("_output/cut_times.csv")
TRACKER_CSV = Path("_output/cut_processing_tracker.csv")


# =========================
# CUT DETECTION SETTINGS
# =========================

# Higher = fewer false positives.
# Try 0.45 first. Use 0.55 if still too sensitive.
SCENE_THRESHOLD = 0.45

MERGE_NEARBY_SEC = 2.0

IGNORE_FIRST_SEC = 0
IGNORE_LAST_SEC = 0


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".m4v",
    ".avi",
}


def format_number(value):
    """
    Formats numbers for the old CSV filename style.

    Examples:
    25.0  -> "25"
    29.97 -> "29.97"
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    return f"{value:.6f}".rstrip("0").rstrip(".")


def parse_old_csv_name(csv_file):
    """
    Reads the old identifier format:

    {video_id}_{start_time}_{fps}.csv

    Returns only video_id and start_time because those are enough to identify
    a mapped segment that was already completed.
    """
    stem = Path(str(csv_file)).stem
    parts = stem.rsplit("_", 2)

    if len(parts) != 3:
        return None, None

    video_id = parts[0]

    try:
        start_time = float(parts[1])
    except ValueError:
        return None, None

    return video_id, start_time


def split_simple_list(value):
    """
    Converts:
    [video1,video2,video3]

    into:
    ["video1", "video2", "video3"]
    """
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    if not text.strip():
        return []

    values = []

    for item in text.split(","):
        item = item.strip().strip("'\"")

        if item:
            values.append(item)

    return values


def split_nested_number_lists(value):
    """
    Converts:
    [[25],[30,400],[0]]

    into:
    [[25.0], [30.0, 400.0], [0.0]]
    """
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    groups = re.findall(r"\[([^\[\]]*)\]", text)
    result = []

    for group in groups:
        numbers = []

        for item in group.split(","):
            item = item.strip()

            if not item:
                continue

            try:
                number = float(item)
            except ValueError:
                continue

            if math.isfinite(number):
                numbers.append(number)

        result.append(numbers)

    return result


def load_mapping_segments(mapping_csv):
    """
    Reads mapping.csv and returns unique tuples:

    (video_id, segment_start, segment_end)

    The mapping columns are parallel nested lists:
    videos[i] -> start_time[i] -> end_time[i]
    """
    required_columns = ["videos", "start_time", "end_time"]

    try:
        mapping_df = pd.read_csv(
            mapping_csv,
            usecols=required_columns,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not read mapping CSV: {exc}") from exc

    segments = []
    seen_segments = set()
    skipped_entries = 0

    for row_number, row in mapping_df.iterrows():
        video_ids = split_simple_list(row["videos"])
        start_groups = split_nested_number_lists(row["start_time"])
        end_groups = split_nested_number_lists(row["end_time"])

        if not (
            len(video_ids) == len(start_groups) == len(end_groups)
        ):
            print(
                f"Mapping row {row_number + 2}: list lengths do not match "
                f"(videos={len(video_ids)}, "
                f"start_time={len(start_groups)}, "
                f"end_time={len(end_groups)})"
            )

        item_count = min(
            len(video_ids),
            len(start_groups),
            len(end_groups),
        )

        for item_index in range(item_count):
            video_id = video_ids[item_index]
            starts = start_groups[item_index]
            ends = end_groups[item_index]

            if len(starts) != len(ends):
                print(
                    f"Mapping row {row_number + 2}, video {video_id}: "
                    f"start/end counts do not match "
                    f"({len(starts)} starts, {len(ends)} ends)"
                )

            segment_count = min(len(starts), len(ends))

            for segment_index in range(segment_count):
                segment_start = float(starts[segment_index])
                segment_end = float(ends[segment_index])

                if segment_start < 0 or segment_end <= segment_start:
                    skipped_entries += 1
                    continue

                dedupe_key = (
                    video_id,
                    round(segment_start, 6),
                    round(segment_end, 6),
                )

                if dedupe_key in seen_segments:
                    continue

                seen_segments.add(dedupe_key)
                segments.append(
                    (video_id, segment_start, segment_end)
                )

    segments.sort(key=lambda item: (item[0], item[1], item[2]))

    print(f"Loaded {len(segments)} unique mapped segments.")

    if skipped_entries:
        print(f"Skipped {skipped_entries} invalid mapped segments.")

    return segments


def build_video_index(video_dirs):
    """
    Builds dictionary:
    video_id -> video_path

    This avoids searching folders repeatedly.
    """
    video_index = {}

    for video_dir in video_dirs:
        if not video_dir.exists():
            print(f"Video folder does not exist: {video_dir}")
            continue

        print(f"Indexing videos in: {video_dir}")

        for root, _, files in os.walk(video_dir):
            root_path = Path(root)

            for file_name in files:
                path = root_path / file_name

                if path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                video_id = path.stem

                if video_id not in video_index:
                    video_index[video_id] = path

    print(f"Indexed {len(video_index)} video files.")

    return video_index


def find_video_path(video_id, video_index):
    """
    First tries exact indexed match.
    Then tries direct matching in each video folder.
    """
    if video_id in video_index:
        return video_index[video_id]

    for video_dir in VIDEO_DIRS:
        if not video_dir.exists():
            continue

        for ext in VIDEO_EXTENSIONS:
            candidate = video_dir / f"{video_id}{ext}"

            if candidate.exists():
                return candidate

        matches = list(video_dir.glob(f"{video_id}.*"))

        if matches:
            return matches[0]

    return None


def get_video_fps(video_path):
    """
    Reads the video's average frame rate using ffprobe.

    FPS is used only to keep the old csv_file naming format:
    {video_id}_{start_time}_{fps}.csv
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        message = (
            result.stderr.splitlines()[-1]
            if result.stderr
            else "ffprobe_failed"
        )
        return None, message

    value = result.stdout.strip()

    try:
        if "/" in value:
            numerator_text, denominator_text = value.split("/", 1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)

            if denominator == 0:
                return None, "invalid_fps"

            fps = numerator / denominator
        else:
            fps = float(value)
    except ValueError:
        return None, f"invalid_fps: {value}"

    if fps <= 0 or not math.isfinite(fps):
        return None, f"invalid_fps: {value}"

    # Old CSV names generally use normal decimal/integer FPS values.
    if abs(fps - round(fps)) < 0.02:
        fps = float(round(fps))
    else:
        fps = round(fps, 3)

    return fps, ""


def make_old_csv_file_name(video_id, segment_start, fps):
    """
    Keeps the previous output value style:

    {video_id}_{start_time}_{fps}.csv
    """
    return (
        f"{video_id}_"
        f"{format_number(segment_start)}_"
        f"{format_number(fps)}.csv"
    )


def run_ffmpeg_scene_detection(
    video_path,
    segment_start,
    segment_duration,
):
    """
    Uses FFmpeg pixel-level scene cut detection.

    FFmpeg reports pts_time relative to the segment because of:
    setpts=PTS-STARTPTS

    Final output:
    absolute_time = segment_start + pts_time
    """
    if segment_duration is None or segment_duration <= 0:
        return [], "invalid_duration"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-ss",
        f"{segment_start:.3f}",
        "-t",
        f"{segment_duration:.3f}",
        "-i",
        str(video_path),
        "-vf",
        f"setpts=PTS-STARTPTS,select='gt(scene,{SCENE_THRESHOLD})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        message = (
            result.stderr.splitlines()[-1]
            if result.stderr
            else "ffmpeg_failed"
        )
        return [], message

    cut_times = []

    for line in result.stderr.splitlines():
        match = re.search(r"pts_time:([0-9.]+)", line)

        if match:
            relative_time = float(match.group(1))
            absolute_time = segment_start + relative_time
            cut_times.append(absolute_time)

    return cut_times, ""


def merge_nearby_times(times):
    if not times:
        return []

    times = sorted(times)

    merged = []
    group = [times[0]]

    for value in times[1:]:
        if value - group[-1] <= MERGE_NEARBY_SEC:
            group.append(value)
        else:
            merged.append(group[0])
            group = [value]

    merged.append(group[0])

    return merged


def filter_segment_edges(
    times,
    segment_start,
    segment_duration,
):
    if segment_duration is None:
        return times

    segment_end = segment_start + segment_duration
    filtered = []

    for value in times:
        if value < segment_start + IGNORE_FIRST_SEC:
            continue

        if value > segment_end - IGNORE_LAST_SEC:
            continue

        filtered.append(value)

    return filtered


def initialise_files():
    """
    Creates files only when they do not already exist.

    Existing output and tracker files are never deleted or overwritten.
    """
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_CSV.exists():
        pd.DataFrame(
            columns=["csv_file", "absolute_time_sec"]
        ).to_csv(
            OUTPUT_CSV,
            index=False,
        )

    if not TRACKER_CSV.exists():
        pd.DataFrame(
            columns=[
                "csv_file",
                "video_id",
                "video_path",
                "status",
                "saved_rows",
                "processed_at",
                "message",
            ]
        ).to_csv(
            TRACKER_CSV,
            index=False,
        )


def get_completed_segments():
    """
    Returns completed (video_id, start_time) pairs from the existing tracker.

    Only status=done is skipped. Failed segments can be retried.
    """
    if not TRACKER_CSV.exists():
        return set()

    try:
        tracker_df = pd.read_csv(TRACKER_CSV)
    except Exception:
        return set()

    required_columns = {"csv_file", "status"}

    if not required_columns.issubset(tracker_df.columns):
        return set()

    completed = set()

    done_rows = tracker_df[
        tracker_df["status"].astype(str).str.lower() == "done"
    ]

    for _, row in done_rows.iterrows():
        video_id, start_time = parse_old_csv_name(
            row["csv_file"]
        )

        if video_id is None:
            continue

        completed.add(
            (video_id, round(float(start_time), 6))
        )

    return completed


def append_output_rows(rows):
    """
    Keeps the old output format exactly:

    csv_file,absolute_time_sec
    """
    if not rows:
        return

    pd.DataFrame(
        rows,
        columns=["csv_file", "absolute_time_sec"],
    ).to_csv(
        OUTPUT_CSV,
        mode="a",
        header=False,
        index=False,
    )


def append_tracker_row(
    csv_file,
    video_id,
    video_path,
    status,
    saved_rows,
    message="",
):
    """
    Keeps the old tracker format exactly.
    """
    row = pd.DataFrame(
        [{
            "csv_file": csv_file,
            "video_id": video_id,
            "video_path": (
                ""
                if video_path is None
                else str(video_path)
            ),
            "status": status,
            "saved_rows": saved_rows,
            "processed_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "message": message,
        }]
    )

    row.to_csv(
        TRACKER_CSV,
        mode="a",
        header=False,
        index=False,
    )


def process_one_segment(
    video_id,
    segment_start,
    segment_end,
    video_index,
    fps_cache,
):
    video_path = find_video_path(
        video_id,
        video_index,
    )

    if video_path is None:
        csv_file = (
            f"{video_id}_"
            f"{format_number(segment_start)}_unknown.csv"
        )

        return [], {
            "csv_file": csv_file,
            "video_id": video_id,
            "video_path": None,
            "status": "video_not_found",
            "message": "No matching local video file found",
        }

    if video_id in fps_cache:
        fps, fps_message = fps_cache[video_id]
    else:
        fps, fps_message = get_video_fps(video_path)
        fps_cache[video_id] = (fps, fps_message)

    if fps is None:
        csv_file = (
            f"{video_id}_"
            f"{format_number(segment_start)}_unknown.csv"
        )

        return [], {
            "csv_file": csv_file,
            "video_id": video_id,
            "video_path": video_path,
            "status": "fps_failed",
            "message": fps_message,
        }

    csv_file = make_old_csv_file_name(
        video_id=video_id,
        segment_start=segment_start,
        fps=fps,
    )

    segment_duration = (
        float(segment_end) - float(segment_start)
    )

    if segment_duration <= 0:
        return [], {
            "csv_file": csv_file,
            "video_id": video_id,
            "video_path": video_path,
            "status": "duration_failed",
            "message": "end_time must be greater than start_time",
        }

    cut_times, ffmpeg_message = run_ffmpeg_scene_detection(
        video_path=video_path,
        segment_start=segment_start,
        segment_duration=segment_duration,
    )

    if ffmpeg_message:
        return [], {
            "csv_file": csv_file,
            "video_id": video_id,
            "video_path": video_path,
            "status": "ffmpeg_failed",
            "message": ffmpeg_message,
        }

    cut_times = filter_segment_edges(
        times=cut_times,
        segment_start=segment_start,
        segment_duration=segment_duration,
    )

    cut_times = merge_nearby_times(cut_times)

    rows = []

    for cut_time in cut_times:
        rows.append({
            "csv_file": csv_file,
            "absolute_time_sec": round(
                float(cut_time),
                3,
            ),
        })

    return rows, {
        "csv_file": csv_file,
        "video_id": video_id,
        "video_path": video_path,
        "status": "done",
        "message": "",
    }


def main():
    global VIDEO_DIRS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", help="folders with {video_id}.mp4, instead of the config")
    args = parser.parse_args()
    if args.videos:
        VIDEO_DIRS = [Path(d).expanduser() for d in args.videos]
    initialise_files()

    completed_segments = get_completed_segments()
    video_index = build_video_index(VIDEO_DIRS)
    mapped_segments = load_mapping_segments(MAPPING_CSV)

    remaining_segments = [
        segment
        for segment in mapped_segments
        if (
            segment[0],
            round(float(segment[1]), 6),
        ) not in completed_segments
    ]

    print(f"Total mapped segments: {len(mapped_segments)}")
    print(
        "Already completed in tracker: "
        f"{len(mapped_segments) - len(remaining_segments)}"
    )
    print(
        f"Remaining to process: {len(remaining_segments)}"
    )

    fps_cache = {}

    for index, segment in enumerate(
        remaining_segments,
        start=1,
    ):
        video_id, segment_start, segment_end = segment

        print(
            f"[{index}/{len(remaining_segments)}] "
            f"{video_id}: "
            f"{format_number(segment_start)}-"
            f"{format_number(segment_end)}"
        )

        try:
            rows, info = process_one_segment(
                video_id=video_id,
                segment_start=segment_start,
                segment_end=segment_end,
                video_index=video_index,
                fps_cache=fps_cache,
            )

            append_output_rows(rows)

            append_tracker_row(
                csv_file=info["csv_file"],
                video_id=info["video_id"],
                video_path=info["video_path"],
                status=info["status"],
                saved_rows=len(rows),
                message=info["message"],
            )

            print(
                f"    status={info['status']}, "
                f"saved={len(rows)}"
            )

        except Exception as exc:
            fallback_csv_file = (
                f"{video_id}_"
                f"{format_number(segment_start)}_unknown.csv"
            )

            append_tracker_row(
                csv_file=fallback_csv_file,
                video_id=video_id,
                video_path=find_video_path(
                    video_id,
                    video_index,
                ),
                status="crashed",
                saved_rows=0,
                message=str(exc),
            )

            print(f"    crashed: {exc}")

    print("Done.")
    print(f"Output:  {OUTPUT_CSV}")
    print(f"Tracker: {TRACKER_CSV}")


if __name__ == "__main__":
    main()
