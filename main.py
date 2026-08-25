# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# Pipeline overview:
# - Reads a mapping CSV describing video IDs and segment intervals; runs in a loop.
# - Optionally updates metadata (ISO3, population, mortality, Gini, upload dates).
# - Retrieves videos (FTP first → YouTube fallback). With external_ssd=True:
#     * downloads go to videos[-1], and
#     * if a file already exists in videos[-1] and FTP is available, we refresh it:
#         download to a temp folder → delete old file → replace with fresh copy.
# - For each segment, run bounding box tracking only if the corresponding CSV
#   is missing; otherwise skip work. If tracking is disabled, still download mp4s.
# - Cleans temp files and emails a summary or crash report.
# -----------------------------------------------------------------------------

import shutil
import os
import threading
import uuid
import hashlib
import glob  # checks for "do we already have CSVs?" without knowing FPS
from datetime import datetime
from helper_script import Youtube_Helper         # download/trim/track utilities
import pandas as pd
from custom_logger import CustomLogger           # structured logging
from logmod import logs                          # log level/color setup
import ast                                       # safe string->python list parsing
import common                                    # configs, secrets, email, git utils
import time                                      # sleep between passes
from types import SimpleNamespace                # lightweight config container
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Configure logging based on config file (verbosity & ANSI colors)
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)
helper = Youtube_Helper()
# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------
_worker_local = threading.local()


def _get_worker_helper() -> Youtube_Helper:
    """Return a per-thread helper instance (avoids shared mutable state)."""
    h = getattr(_worker_local, "helper", None)
    if h is None:
        h = Youtube_Helper()
        _worker_local.helper = h
    return h


def _cfg(key: str, default=None):
    """Backwards-compatible config lookup: returns default if key missing."""
    try:
        v = common.get_configs(key)
        return default if v is None else v
    except Exception:
        return default


def _slurm_task_info():
    """Return (global_rank, world_size, local_rank) for a Slurm job step.

    Falls back to (0, 1, 0) when not running under Slurm.
    """
    try:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
    except Exception:
        rank = 0

    try:
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    except Exception:
        world_size = 1

    try:
        local_rank = int(os.environ.get("SLURM_LOCALID", str(rank)))
    except Exception:
        local_rank = rank

    if world_size <= 0:
        world_size = 1
    if rank < 0:
        rank = 0
    if local_rank < 0:
        local_rank = 0

    return rank, world_size, local_rank


def _segment_to_shard(vid: str, st: int, et: int, world_size: int) -> int:
    """Deterministically assign a segment to a shard in [0, world_size).

    Uses MD5 to avoid Python hash randomization across processes.
    """
    if world_size <= 1:
        return 0

    payload = f"{vid}|{int(st)}|{int(et)}".encode("utf-8", errors="ignore")
    digest = hashlib.md5(payload).digest()
    return int.from_bytes(digest[:4], "little") % world_size


def process_mapping_concurrently(mapping, config, secret, logger) -> int:
    """Concurrent video download + segment processing with fairness constraints."""

    # Local aliases
    data_folders = list(config.data)
    video_paths = list(config.videos)
    internal_ssd = video_paths[-1] if video_paths else "."
    os.makedirs(internal_ssd, exist_ok=True)

    # ------------------------------------------------------------------
    # Segment completion policy
    # ------------------------------------------------------------------
    # IMPORTANT:
    # Do not use loose root-level marker files to decide that a segment is done.
    # A temporary/leftover filename such as "-xvmIQqC3dw_0_475_30_mod_....mp4"
    # can accidentally match the old marker pattern "{video_id}_{start_time}_*".
    # That made videos whose IDs start with '-' download successfully, then get
    # finalised without trimming or YOLO processing.  The pipeline now decides
    # completion only from the actual expected output CSVs in data/*/bbox.

    # Snellius multi-task sharding (1 task == 1 GPU)
    snellius_mode = bool(getattr(config, "snellius_mode", False))
    shard_mode = bool(getattr(config, "snellius_shard_mode", False))
    rank, world_size, local_rank = (0, 1, 0)
    if snellius_mode:
        rank, world_size, local_rank = _slurm_task_info()

    # ------------------------------------------------------------------
    # Build a per-video segment plan (deduplicated).
    # ------------------------------------------------------------------
    video_plan = OrderedDict()  # vid -> dict(segments=list[(st, et, tod)], meta=dict)

    countries_analyse = getattr(config, "countries_analyse", [])
    for _, row in mapping.iterrows():
        try:
            iso3 = str(row.get("iso3", ""))
            if countries_analyse and iso3 not in countries_analyse:
                continue

            video_ids = [v.strip() for v in str(row["videos"]).strip("[]").split(",") if v.strip()]
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])

            for vid, st_list, et_list, tod_list in zip(video_ids, start_times, end_times, time_of_day):
                if vid not in video_plan:
                    video_plan[vid] = {"segments": [], "seen": set()}
                for st, et, tod in zip(st_list, et_list, tod_list):
                    key = (int(st), int(et), str(tod))
                    if key not in video_plan[vid]["seen"]:
                        video_plan[vid]["seen"].add(key)
                        video_plan[vid]["segments"].append(key)
        except Exception as e:
            logger.warning(f"Skipping malformed mapping row due to parse error: {e!r}")
            continue

    # ------------------------------------------------------------------
    # Snellius sharding: each Slurm task processes a disjoint subset of segments.
    # This is intentionally deterministic (MD5) so all tasks agree on ownership.
    # ------------------------------------------------------------------
    if snellius_mode and shard_mode and world_size > 1:
        sharded_plan = OrderedDict()
        assigned_segments = 0
        for vid, info in video_plan.items():
            segs = [
                seg for seg in info["segments"]
                if _segment_to_shard(vid, seg[0], seg[1], world_size) == rank
            ]
            if segs:
                sharded_plan[vid] = {"segments": segs, "seen": set(segs)}
                assigned_segments += len(segs)

        video_plan = sharded_plan
        logger.info(
            f"[Snellius shard {rank}/{world_size}] Assigned {assigned_segments} segments across {len(video_plan)} videos."  # noqa: E501
        )

    if not video_plan:
        logger.info("No videos found after filtering; nothing to do.")
        return 0

    # ------------------------------------------------------------------
    # Decide which videos actually need downloading/processing.
    # - If bbox tracking is disabled: always download (archival).
    # - Otherwise: download only if any required bbox output is missing (wildcard FPS).
    # ------------------------------------------------------------------
    bbox_mode_cfg = bool(getattr(config, "tracking_mode", False))

    vids_to_handle = []
    for vid, info in video_plan.items():
        segments = info["segments"]

        if not bbox_mode_cfg:
            vids_to_handle.append(vid)
            continue

        needs_any = False
        for st, _, _ in segments:
            has_bbox = any(glob.glob(os.path.join(folder, "bbox", f"{vid}_{st}_*.csv")) for folder in data_folders)
            if bbox_mode_cfg and not has_bbox:
                needs_any = True
                break

        if needs_any:
            vids_to_handle.append(vid)

    if not vids_to_handle:
        logger.info("All required outputs already exist; nothing to process.")
        return 0

    # Concurrency controls
    max_workers = int(getattr(config, "max_workers", 1))
    download_workers = int(getattr(config, "download_workers", 2))
    max_active_per_video = int(getattr(config, "max_active_segments_per_video", 1))

    # Ensure per-task run isolation root exists
    runs_root = getattr(config, "runs_root", "runs")
    os.makedirs(runs_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Video preparation (download / refresh / locate).
    # ------------------------------------------------------------------
    def _prepare_video(vid: str):
        h = _get_worker_helper()

        # Preferred cache destination:
        # - if external_ssd=True, we typically use the last configured path (internal_ssd)
        # - otherwise we use the first configured path
        output_path = internal_ssd if getattr(config, "external_ssd", False) else (video_paths[0] if video_paths else ".")  # noqa: E501
        os.makedirs(output_path, exist_ok=True)

        snellius_mode = bool(getattr(config, "snellius_mode", False))
        ftp_server = getattr(config, "ftp_server", None)

        preferred_path = os.path.join(output_path, f"{vid}.mp4")

        video_title = vid
        resolution = ""
        video_fps = 0.0
        base_video_path = None

        # Cleanup intent:
        # - always_delete_paths: local copies created from other folders (and any processing-speed working copies)
        # - downloaded_paths: FTP/YouTube downloads (deleted only when delete_youtube_video=True)
        always_delete_paths = []
        downloaded_paths = []
        download_source = "existing"

        # ------------------------------------------------------------------
        # 0) If the canonical cache file already exists in output_path, use it.
        #    Do NOT re-download (no FTP refresh).
        # ------------------------------------------------------------------
        if os.path.isfile(preferred_path):
            base_video_path = preferred_path

        # ------------------------------------------------------------------
        # 1) Non-Snellius only: if preferred is missing, look for an existing copy
        #    in other configured video folders and copy it into output_path.
        #    Copied MP4s are deleted by default once processing finishes.
        # ------------------------------------------------------------------
        if base_video_path is None and not snellius_mode:
            for vp in video_paths:
                try:
                    if os.path.abspath(vp) == os.path.abspath(output_path):
                        continue
                except Exception:
                    pass

                candidate = os.path.join(vp, f"{vid}.mp4")
                if not os.path.isfile(candidate):
                    continue

                try:
                    out = h.copy_video_safe(candidate, output_path, vid)
                    if out:
                        base_video_path = out
                    else:
                        shutil.copy2(candidate, preferred_path)
                        base_video_path = preferred_path

                    download_source = "copy"
                    always_delete_paths.append(base_video_path)
                except Exception as e:
                    logger.warning(f"{vid}: failed to copy from {candidate} ({e!r}); will try download.")
                break

        # ------------------------------------------------------------------
        # 2) If still missing: download from FTP (if configured).
        #    Snellius mode: ONLY FTP (no YouTube, no local-copy reuse beyond preferred_path).
        # ------------------------------------------------------------------
        if base_video_path is None and ftp_server:
            tmp_dir = os.path.join(output_path, ".ftp_tmp", f"{vid}_{uuid.uuid4().hex}")
            os.makedirs(tmp_dir, exist_ok=True)
            try:
                tmp = h.download_videos_from_ftp(
                    filename=vid,
                    base_url=ftp_server,
                    out_dir=tmp_dir,
                    username=getattr(secret, "ftp_username", None),
                    password=getattr(secret, "ftp_password", None),
                    token=getattr(secret, "ftp_token", None),
                    debug=False,
                )
            except Exception as e:
                tmp = None
                logger.warning(f"{vid}: FTP download failed ({e!r}).")

            if tmp is not None:
                tmp_video_path, video_title, resolution, fps = tmp

                try:
                    os.remove(preferred_path)
                except FileNotFoundError:
                    pass
                try:
                    os.replace(tmp_video_path, preferred_path)
                except Exception:
                    shutil.move(tmp_video_path, preferred_path)

                base_video_path = preferred_path
                downloaded_paths.append(preferred_path)
                download_source = "ftp"
                video_fps = float(fps or 0.0)
                try:
                    h.set_video_title(video_title)
                except Exception:
                    pass
            else:
                if snellius_mode:
                    logger.info(f"{vid}: not available on FTP; skipping (Snellius mode).")
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass
                    return None

            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 3) If still missing: download from YouTube ONLY when snellius_mode is False.
        # ------------------------------------------------------------------
        if base_video_path is None:
            if snellius_mode:
                logger.info(f"{vid}: no local file and FTP unavailable; skipping (Snellius mode).")
                return None

            yt = h.download_video_with_resolution(vid, output_path=output_path)
            if yt is None:
                raise RuntimeError(f"{vid}: could not download video via FTP or YouTube.")
            base_video_path, video_title, resolution, fps = yt
            video_fps = float(fps or 0.0)
            downloaded_paths.append(base_video_path)
            download_source = "youtube"
            try:
                h.set_video_title(video_title)
            except Exception:
                pass

        # Ensure FPS
        if not video_fps or video_fps <= 0:
            try:
                video_fps = float(h.get_video_fps(base_video_path))  # type: ignore
            except Exception:
                video_fps = 0.0

        # Optional processing-speed working copy (safe to delete after processing)
        if getattr(config, "external_ssd", False) and base_video_path and not base_video_path.startswith(internal_ssd):
            try:
                out = h.copy_video_safe(base_video_path, internal_ssd, vid)
                if out and out != base_video_path:
                    always_delete_paths.append(out)
                    base_video_path = out
            except Exception as e:
                logger.warning(f"{vid}: failed to copy to internal SSD ({e!r}); processing from source.")

        # Ensure helper has a title
        try:
            h.set_video_title(video_title or vid)
        except Exception:
            pass

        return {
            "vid": vid,
            "base_video_path": base_video_path,
            "output_path": output_path,
            "video_title": video_title or vid,
            "resolution": resolution,
            "video_fps": float(video_fps),
            "download_source": download_source,
            "downloaded_paths": list(downloaded_paths),
            "always_delete_paths": list(always_delete_paths),
        }

    # ------------------------------------------------------------------
    # Segment processing worker
    # ------------------------------------------------------------------
    def _process_segment(vid: str, base_video_path: str, output_path: str, video_fps: float, st: int, et: int, tod: str, run_bbox: bool) -> int:  # noqa: E501
        h = _get_worker_helper()
        h.set_video_title(vid)

        # Unique per-segment run dir (prevents collisions under concurrency)
        seg_run_root = os.path.join(runs_root, f"{vid}_{st}_{int(video_fps)}_{uuid.uuid4().hex}")
        os.makedirs(seg_run_root, exist_ok=True)

        # Unique trimmed video (avoid clobbering)
        tmp_dir = internal_ssd if getattr(config, "external_ssd", False) else output_path
        os.makedirs(tmp_dir, exist_ok=True)
        # Prefix with "yt_" so the basename never starts with "-".
        # This keeps downstream tools safe if they ever pass a basename to a CLI.
        trimmed_video_path = os.path.join(tmp_dir, f"yt_{vid}_{st}_{et}_{int(video_fps)}_mod_{uuid.uuid4().hex}.mp4")

        # Trim as needed
        if run_bbox:
            end_time_adj = int(et) - 1
            if end_time_adj <= int(st):
                end_time_adj = int(et)
            logger.info(f"{vid}: trimming segment {st}-{et}s.")
            h.trim_video(base_video_path, trimmed_video_path, int(st), int(end_time_adj))

            # Avoid annotated-video filename collisions
            annotated_name = f"{vid}_{st}_{et}_mod.mp4"

            h.tracking_mode(
                trimmed_video_path,
                output_path,
                video_title=annotated_name,
                video_fps=float(video_fps),
                bbox_mode=bool(run_bbox),
                flag=int(getattr(config, "save_annotated_video", 0)),
                run_root=seg_run_root,
            )

        # Move CSVs into the final dataset location (same behaviour as original script)
        data_path = config.data[-1] if config.data else "."
        processed = 0

        if run_bbox:
            old_bbox = os.path.join(seg_run_root, "detect", f"{vid}.csv")
            new_bbox = os.path.join(seg_run_root, "detect", f"{vid}_{st}_{int(video_fps)}.csv")
            if os.path.isfile(old_bbox):
                os.rename(old_bbox, new_bbox)
            dest_bbox = os.path.join(data_path, "bbox", os.path.basename(new_bbox))
            os.makedirs(os.path.dirname(dest_bbox), exist_ok=True)
            if os.path.isfile(new_bbox):
                shutil.copy(new_bbox, dest_bbox)
                processed = 1

        # Cleanup
        if getattr(config, "delete_runs_files", True):
            try:
                shutil.rmtree(seg_run_root)
            except Exception:
                pass
        if getattr(config, "delete_mod_video", True):
            try:
                os.remove(trimmed_video_path)
            except FileNotFoundError:
                pass

        return processed

    # ------------------------------------------------------------------
    # Run pipeline with a bounded "video buffer".
    #
    # By default, the downloader will stay only (2 * max_workers) videos ahead:
    # a video counts toward the buffer while it is
    #   (a) downloading/preparing, OR
    #   (b) downloaded but still has pending/active segments.
    #
    # This prevents the pipeline from downloading the entire mapping upfront.
    # You can override via config key "download_buffer" if desired.
    # ------------------------------------------------------------------
    download_pool = ThreadPoolExecutor(max_workers=download_workers)
    segment_pool = ThreadPoolExecutor(max_workers=max_workers)

    download_buffer = int(getattr(config, "download_buffer", 0) or (2 * max_workers))
    download_buffer = max(1, min(download_buffer, len(vids_to_handle)))

    vid_queue = deque(vids_to_handle)

    download_futures = {}          # vid -> Future
    pending = defaultdict(deque)   # vid -> deque[(st, et, tod, run_bbox)]
    active = defaultdict(int)      # vid -> active segment tasks
    seg_futures = {}              # Future -> vid
    video_info = {}               # vid -> prepare dict
    processed_any = defaultdict(bool)

    rr = 0
    total_processed = 0

    def _video_is_active(v: str) -> bool:
        return (active.get(v, 0) > 0) or bool(pending.get(v))

    def _count_inflight_videos() -> int:
        active_vids = sum(1 for v in video_info.keys() if _video_is_active(v))
        return len(download_futures) + active_vids

    def _maybe_submit_downloads():
        while vid_queue and _count_inflight_videos() < download_buffer:
            v = vid_queue.popleft()
            download_futures[v] = download_pool.submit(_prepare_video, v)

    def _finalize_video(v: str):
        info = video_info.get(v)
        if not info:
            return

        ext_ssd = bool(getattr(config, "external_ssd", False))
        keep_root = None
        if ext_ssd:
            try:
                keep_root = os.path.abspath(internal_ssd)
            except Exception:
                keep_root = None

        def _ok_to_delete(path: str) -> bool:
            """When external_ssd=True, only delete files that live in videos[-1]."""
            if not keep_root:
                return True
            try:
                ap = os.path.abspath(path)
            except Exception:
                return False
            return ap == keep_root or ap.startswith(keep_root + os.sep)

        # Default cleanup: delete MP4s created by COPYING from other folders
        # (and any processing-speed working copies).
        for p in set(info.get("always_delete_paths") or []):
            if not _ok_to_delete(p):
                continue
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        # Conditional cleanup: delete FTP/YouTube downloads only when explicitly enabled.
        # When external_ssd=True, never delete anything outside videos[-1]
        # even if delete_youtube_video=True.
        if getattr(config, "delete_youtube_video", False):
            for p in set(info.get("downloaded_paths") or []):
                if not _ok_to_delete(p):
                    continue
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        # Drop bookkeeping to free buffer slots and memory
        video_info.pop(v, None)
        pending.pop(v, None)
        active.pop(v, None)

    # Prime the download buffer
    _maybe_submit_downloads()

    # Snellius: warm up downloads so segment workers do not stall on I/O.
    # Target: keep at least 2×max_workers videos prepared (downloaded + queued) before starting segment work.
    warmup_ready_target = 0
    warmup_phase = False
    if snellius_mode and max_workers > 0:
        warmup_ready_target = min(len(vids_to_handle), 2 * max_workers)
        warmup_phase = warmup_ready_target > 0

    try:
        while True:
            # ----------------------------------------------------------
            # 1) Harvest completed downloads and enqueue missing segments
            # ----------------------------------------------------------
            for v, fut in list(download_futures.items()):
                if not fut.done():
                    continue

                download_futures.pop(v, None)

                try:
                    info = fut.result()
                except Exception as e:
                    logger.error(f"{v}: download/prepare failed: {e!r}")
                    continue

                if info is None:
                    logger.info(f"{v}: skipped (not available via FTP).")
                    continue

                video_info[v] = info
                fps = info["video_fps"]  # noqa: F841

                if not bbox_mode_cfg:
                    # Download-only mode: nothing else to do.
                    _finalize_video(v)
                    continue

                for (st, et, tod) in video_plan[v]["segments"]:
                    has_bbox = any(glob.glob(os.path.join(folder, "bbox",
                                                          f"{v}_{st}_*.csv")) for folder in data_folders)
                    run_bbox = bool(bbox_mode_cfg and not has_bbox)

                    logger.info(
                        f"{v}: segment {st}-{et}s output status: "
                        f"has_bbox={has_bbox}, run_bbox={run_bbox}"
                    )

                    if run_bbox:
                        pending[v].append((int(st), int(et), str(tod), run_bbox))

                # If nothing is required for this video, finalize immediately.
                if not pending.get(v):
                    _finalize_video(v)

            # Submit additional downloads if buffer has room
            _maybe_submit_downloads()

            # Snellius: ensure we have a backlog of prepared videos before consuming GPU slots.
            if warmup_phase:
                ready_now = sum(1 for vv in video_info.keys() if pending.get(vv))
                if ready_now < warmup_ready_target and (vid_queue or download_futures):
                    if download_futures:
                        wait(set(download_futures.values()), return_when=FIRST_COMPLETED, timeout=0.5)
                    continue
                warmup_phase = False

            # ----------------------------------------------------------
            # 2) Schedule segment jobs (fair round-robin across videos)
            # ----------------------------------------------------------
            if max_workers > 0 and len(seg_futures) < max_workers:
                active_vids = [v for v in video_info.keys() if pending.get(v)]
                if active_vids:
                    scheduled_any = True
                    while scheduled_any and len(seg_futures) < max_workers:
                        scheduled_any = False
                        for _ in range(len(active_vids)):
                            v = active_vids[rr % len(active_vids)]
                            rr += 1
                            if active[v] >= max_active_per_video:
                                continue
                            if not pending.get(v):
                                continue

                            st, et, tod, run_bbox = pending[v].popleft()
                            info = video_info[v]
                            active[v] += 1

                            f = segment_pool.submit(
                                _process_segment,
                                v,
                                info["base_video_path"],
                                info["output_path"],
                                info["video_fps"],
                                st, et, tod,
                                run_bbox,
                            )
                            seg_futures[f] = v
                            scheduled_any = True

                            if len(seg_futures) >= max_workers:
                                break

            # ----------------------------------------------------------
            # 3) Collect completed segment jobs
            # ----------------------------------------------------------
            if seg_futures:
                done, _ = wait(set(seg_futures.keys()), return_when=FIRST_COMPLETED, timeout=0.5)
                for f in list(done):
                    v = seg_futures.pop(f, None)
                    if v is None:
                        continue
                    active[v] = max(0, active[v] - 1)
                    try:
                        processed = int(f.result() or 0)
                        if processed:
                            processed_any[v] = True
                            total_processed += processed
                    except Exception as e:
                        logger.error(f"{v}: segment task failed: {e!r}")
            else:
                # No segment jobs right now; avoid busy loop while downloads are running.
                if download_futures:
                    wait(set(download_futures.values()), return_when=FIRST_COMPLETED, timeout=0.5)

            # ----------------------------------------------------------
            # 4) Finalize drained videos to free buffer slots
            # ----------------------------------------------------------
            for v in list(video_info.keys()):
                if (active.get(v, 0) == 0) and (not pending.get(v)):
                    _finalize_video(v)

            # Make room for more downloads after finalization
            _maybe_submit_downloads()

            # Exit when nothing remains
            if not vid_queue and not download_futures and not seg_futures and not any(pending.values()):
                break

        return total_processed

    finally:
        download_pool.shutdown(wait=True, cancel_futures=False)
        segment_pool.shutdown(wait=True, cancel_futures=False)


# Guard to avoid duplicate crash emails (kept for compatibility with some runners)
email_already_sent = False

# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    try:
        # ---------------------------------------------------------------------
        # Load static configuration (paths, switches) once
        # ---------------------------------------------------------------------
        config = SimpleNamespace(
            mapping=common.get_configs("mapping"),                 # path to mapping CSV
            videos=common.get_configs("videos"),                   # ordered list of video dirs
            delete_runs_files=common.get_configs("delete_runs_files"),
            delete_youtube_video=common.get_configs("delete_youtube_video"),
            data=common.get_configs("data"),                       # ordered list of data dirs
            countries_analyse=common.get_configs("countries_analyse"),
            update_pop_country=common.get_configs("update_pop_country"),
            update_gini_value=common.get_configs("update_gini_value"),
            tracking_mode=common.get_configs("tracking_mode"),
            save_annotated_video=common.get_configs("save_annotated_video"),
            sleep_sec=common.get_configs("sleep_sec"),
            git_pull=common.get_configs("git_pull"),
            machine_name=common.get_configs("machine_name"),
            email_send=common.get_configs("email_send"),
            email_sender=common.get_configs("email_sender"),
            email_recipients=common.get_configs("email_recipients"),
            external_ssd=common.get_configs("external_ssd"),       # True => prefer SSD target
            ftp_server=common.get_configs("ftp_server"),
            snellius_mode=_cfg("snellius_mode", False),
            snellius_shard_mode=_cfg("snellius_shard_mode", True),

            max_workers=_cfg("max_workers", 1),
            download_workers=_cfg("download_workers", 2),
            max_active_segments_per_video=_cfg("max_active_segments_per_video", 1),
            runs_root=_cfg("runs_root", "runs"),
        )

        # ---------------------------------------------------------------------
        # Load secrets (email + FTP credentials)
        # ---------------------------------------------------------------------
        secret = SimpleNamespace(
            email_smtp=common.get_secrets("email_smtp"),
            email_account=common.get_secrets("email_account"),
            email_password=common.get_secrets("email_password"),
            ftp_username=common.get_secrets("ftp_username"),
            ftp_password=common.get_secrets("ftp_password"),
        )

        # ---------------------------------------------------------------------
        # Snellius: derive task rank information (for multi-task GPU runs)
        # ---------------------------------------------------------------------
        if bool(getattr(config, "snellius_mode", False)):
            r, w, lr = _slurm_task_info()
            config.snellius_rank = r
            config.snellius_world_size = w
            config.snellius_local_rank = lr

            # Ensure each task uses the intended GPU when multiple GPUs are visible.
            # try:
            #     import torch
            #     if torch.cuda.is_available():
            #         torch.cuda.set_device(lr)
            #         logger.info(f"[Snellius] Using CUDA device local_rank={lr} (rank={r}/{w}).")
            # except Exception as e:
            #     logger.warning(f"[Snellius] Unable to set CUDA device for local_rank={lr}: {e!r}")

        # =========================================================================
        # Endless loop: process mapping; sleep; repeat
        # =========================================================================
        while True:
            # Read mapping each pass to pick up edits
            mapping = pd.read_csv(config.mapping)
            video_paths = config.videos  # convenience alias

            # Decide where to write downloads
            if config.external_ssd:
                # With SSD, ensure mount exists and set output to videos[-1]
                internal_ssd = config.videos[-1]
                os.makedirs(internal_ssd, exist_ok=True)
                output_path = config.videos[-1]
            else:
                output_path = config.videos[-1]

            # Convenience locals; keep behavior intact
            delete_runs_files = config.delete_runs_files
            delete_youtube_video = config.delete_youtube_video
            data_folders = config.data
            data_path = config.data[-1]  # final data root (where bbox CSVs land)
            countries_analyse = config.countries_analyse
            counter_processed = 0        # segments processed this pass

            # -----------------------------------------------------------------
            # Optional mapping maintenance (all in-place on `mapping`)
            # -----------------------------------------------------------------

            if config.update_pop_country:
                helper.update_population_in_csv(mapping)
            if config.update_gini_value:
                helper.fill_gini_data(mapping)

            # -----------------------------------------------------------------
            # Workspace setup: clean YOLO runs/* and ensure data dirs exist
            # -----------------------------------------------------------------
            helper.delete_folder(folder_path=getattr(config, "runs_root", "runs"))  # idempotent cleanup
            os.makedirs(data_path, exist_ok=True)
            os.makedirs(os.path.join(data_path, "bbox"), exist_ok=True)

            # =========================================================================
            # Iterate mapping rows; each may define multiple videos & segment lists
            # =========================================================================
            counter_processed = process_mapping_concurrently(mapping, config, secret, logger)

            # -----------------------------------------------------------------
            # Email success summary (only if anything was processed)
            # -----------------------------------------------------------------
            if config.email_send and counter_processed and (not getattr(config, "snellius_mode", False) or getattr(config, "snellius_rank", 0) == 0):  # noqa: E501
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                common.send_email(
                    subject=f"✅ Processing job finished on machine {config.machine_name}",
                    content=(
                        f"Processing job finished on {config.machine_name} at {time_now}. "
                        f"{counter_processed} segments were processed."
                    ),
                    sender=config.email_sender,
                    recipients=config.email_recipients
                )

            # -----------------------------------------------------------------
            # Sleep between passes; clean *_mod.mp4 remnants; optionally git pull
            # -----------------------------------------------------------------
            if config.sleep_sec:
                helper.delete_youtube_mod_videos(video_paths)
                logger.info(f"Sleeping for {config.sleep_sec} s before attempting to go over mapping again.")
                time.sleep(config.sleep_sec)

            if config.git_pull:
                common.git_pull()

    # =============================================================================
    # Crash handling: email details then re-raise for visibility
    # =============================================================================
    except Exception as e:
        try:
            if config.email_send and (not getattr(config, "snellius_mode",  # type: ignore
                                                  False) or getattr(config, "snellius_rank", 0) == 0):  # type: ignore
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # kept from original (fun image): fine to send as plain link or ignore
                image_url = "https://i.pinimg.com/474x/20/82/0f/20820fd73c946d3e1d2e6efe23e1b2f3.jpg"
                common.send_email(
                    subject=f"‼️ Processing job crashed on machine {config.machine_name}",  # type: ignore
                    content=(
                        f"Processing job crashed on {config.machine_name} at {time_now}. "  # type: ignore
                        f"{counter_processed} segments were processed. Error message: {e}."  # type: ignore
                    ),
                    sender=config.email_sender,  # type: ignore
                    recipients=config.email_recipients  # type: ignore
                )
        except Exception:
            # If config/email not ready or mail fails, swallow and re-raise
            pass
        raise
