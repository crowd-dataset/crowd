#!/usr/bin/env python3
"""
Scan specific folders for likely-corrupt videos (e.g., container duration looks fine
but video frames stop early) and write a TXT report.

Requires:
  - ffprobe and ffmpeg on PATH (FFmpeg)

Edit ROOT_FOLDERS below to your folder paths, then run:
  python find_corrupt_videos_hardcoded.py
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import common
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

# =========================
# EDIT THESE PATHS
# =========================
ROOT_FOLDERS = common.get_configs("videos")

# Output report file (TXT)
OUTPUT_TXT = r"corrupt_videos.txt"

# Tuning parameters
TAIL_SECONDS = 15                 # seconds from end to probe/decode
ALLOWED_MISSING_RATIO = 0.03      # 3% of duration
MIN_MISSING_SECONDS = 30.0        # flag only if >= 30s missing

# Parallelism
WORKERS = max(1, (os.cpu_count() or 4) // 2)

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts"}


@dataclass
class CheckResult:
    path: Path
    is_corrupt: bool
    reason: str
    format_duration: Optional[float] = None
    last_video_frame_ts: Optional[float] = None


def which_or_exit(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        print(
            f"ERROR: '{bin_name}' not found on PATH. Install FFmpeg and ensure {bin_name} is available.",
            file=sys.stderr,
        )
        sys.exit(2)
    return p


def iter_video_files(roots: List[Path]) -> Iterable[Path]:
    for r in roots:
        if r.is_file():
            if r.suffix.lower() in VIDEO_EXTS:
                yield r
            continue
        if r.is_dir():
            for dirpath, _, filenames in os.walk(r):
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if p.suffix.lower() in VIDEO_EXTS:
                        yield p


def run_cmd(cmd: List[str], timeout_s: int = 120) -> Tuple[int, str, str]:
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def ffprobe_metadata(ffprobe: str, path: Path) -> Tuple[Optional[float], bool, str]:
    """
    Returns (format_duration_seconds, has_video_stream, raw_error_if_any)
    """
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    rc, out, err = run_cmd(cmd, timeout_s=60)
    if rc != 0 or not out.strip():
        return None, False, f"ffprobe_failed: {err.strip() or 'no_output'}"

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None, False, "ffprobe_failed: invalid_json"

    fmt = data.get("format", {}) or {}
    dur = fmt.get("duration", None)
    format_duration = None
    if dur is not None:
        try:
            format_duration = float(dur)
        except ValueError:
            format_duration = None

    streams = data.get("streams", []) or []
    has_video = any((s.get("codec_type") == "video") for s in streams)

    return format_duration, has_video, ""


def ffprobe_last_video_frame_ts(ffprobe: str, path: Path, tail_seconds: int) -> Tuple[Optional[float], str]:
    """
    Try to read video frames near the end of file and return the last frame timestamp.
    If no frames are returned, returns (None, reason).
    """
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-sseof", f"-{tail_seconds}",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0",
        str(path),
    ]
    rc, out, err = run_cmd(cmd, timeout_s=60)
    if rc != 0:
        return None, f"ffprobe_tail_failed: {err.strip() or 'no_error_text'}"

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None, "no_video_frames_in_tail"

    for ln in reversed(lines):
        try:
            return float(ln), ""
        except ValueError:
            continue

    return None, "no_parsable_tail_timestamps"


def ffmpeg_decode_tail_errors(ffmpeg: str, path: Path, tail_seconds: int) -> Tuple[bool, str]:
    """
    Attempt to decode the last N seconds of video. If ffmpeg reports errors, flag it.
    Returns (has_error, error_summary).
    """
    cmd = [
        ffmpeg,
        "-v", "error",
        "-hide_banner",
        "-nostdin",
        "-sseof", f"-{tail_seconds}",
        "-i", str(path),
        "-map", "0:v:0",
        "-f", "null",
        "-",
    ]
    rc, _out, err = run_cmd(cmd, timeout_s=120)

    err_clean = (err or "").strip()
    if rc != 0 and err_clean:
        return True, f"ffmpeg_decode_tail_failed: {err_clean.splitlines()[-1][:300]}"
    if err_clean:
        last_line = err_clean.splitlines()[-1][:300]
        return True, f"ffmpeg_decode_tail_error: {last_line}"
    return False, ""


def check_video(path: Path, ffprobe: str, ffmpeg: str) -> CheckResult:
    format_duration, has_video, meta_err = ffprobe_metadata(ffprobe, path)
    if meta_err:
        return CheckResult(path, True, meta_err, format_duration, None)

    if not has_video:
        return CheckResult(path, True, "no_video_stream_found", format_duration, None)

    last_ts, tail_reason = ffprobe_last_video_frame_ts(ffprobe, path, tail_seconds=TAIL_SECONDS)
    ffmpeg_tail_err, ffmpeg_reason = ffmpeg_decode_tail_errors(ffmpeg, path, tail_seconds=TAIL_SECONDS)

    reasons = []
    is_corrupt = False

    if ffmpeg_tail_err:
        is_corrupt = True
        reasons.append(ffmpeg_reason)

    if format_duration is not None:
        if last_ts is None:
            if format_duration > float(TAIL_SECONDS) + 1.0:
                is_corrupt = True
                reasons.append(tail_reason or "no_video_frames_in_tail")
        else:
            missing = format_duration - last_ts
            if missing > MIN_MISSING_SECONDS and missing > (ALLOWED_MISSING_RATIO * format_duration):
                is_corrupt = True
                reasons.append(
                    f"video_ends_early: missing≈{missing:.1f}s (dur={format_duration:.1f}s last_frame={last_ts:.1f}s)"
                )
    else:
        if last_ts is None:
            is_corrupt = True
            reasons.append(tail_reason or "no_video_frames_in_tail")

    if not is_corrupt:
        return CheckResult(path, False, "ok", format_duration, last_ts)

    return CheckResult(
        path=path,
        is_corrupt=True,
        reason="; ".join(reasons) if reasons else "suspicious",
        format_duration=format_duration,
        last_video_frame_ts=last_ts,
    )


def main() -> int:
    ffprobe = which_or_exit("ffprobe")
    ffmpeg = which_or_exit("ffmpeg")

    roots = [Path(p).expanduser().resolve() for p in ROOT_FOLDERS]
    files = sorted(set(iter_video_files(roots)))

    if not files:
        print("No video files found under ROOT_FOLDERS.", file=sys.stderr)
        return 1

    print(f"Scanning {len(files)} video file(s) with {WORKERS} worker(s)...")

    results: List[CheckResult] = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(check_video, p, ffprobe, ffmpeg) for p in files]
        for i, f in enumerate(cf.as_completed(futs), start=1):
            results.append(f.result())
            if i % 25 == 0 or i == len(futs):
                print(f"  Progress: {i}/{len(futs)}")

    corrupt = sorted((r for r in results if r.is_corrupt), key=lambda x: str(x.path))

    out_path = Path(OUTPUT_TXT).expanduser().resolve()
    with out_path.open("w", encoding="utf-8") as w:
        w.write("Corrupt/Suspicious videos report\n")
        w.write("=" * 40 + "\n\n")
        w.write(f"Roots: {', '.join(str(r) for r in roots)}\n")
        w.write(f"Scanned: {len(files)}\n")
        w.write(f"Flagged: {len(corrupt)}\n\n")
        w.write("PATH\tREASON\tFORMAT_DURATION_S\tLAST_VIDEO_FRAME_TS_S\n\n")
        for r in corrupt:
            w.write(
                f"{r.path}\t{r.reason}\t"
                f"{'' if r.format_duration is None else f'{r.format_duration:.3f}'}\t"
                f"{'' if r.last_video_frame_ts is None else f'{r.last_video_frame_ts:.3f}'}\n"
            )

    print(f"Done. Flagged {len(corrupt)} / {len(files)}. Report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
