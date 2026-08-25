# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import re
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from ultralytics import YOLO  # type: ignore
from collections import defaultdict
from typing import Optional, Set
import shutil
import numpy as np
import pandas as pd
import world_bank_data as wb
import pycountry
from custom_logger import CustomLogger
import common
import torch
import subprocess
import sys
import logging
from tqdm import tqdm
import datetime
import json
import yaml
import pathlib
import requests
import yt_dlp
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = CustomLogger(__name__)  # use custom logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

# Consts
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

# logging of attempts to upgrade packages
UPGRADE_LOG_FILE = "upgrade_log.json"


class Youtube_Helper:
    """ A helper class for managing YouTube video downloads, processing, and analytics.
    Features:
              - Downloads videos via pytube or yt_dlp with resolution preference.
              - Handles video compression, trimming, and FPS extraction.
              - Applies object detection and tracking using YOLO models.
              - Updates and maintains CSV datasets with video metadata.
              - Interfaces with World Bank data to supplement mapping files.

    Attributes:
        model (str): Path or identifier for the YOLO model to use.
        resolution (str): Target resolution for downloaded videos.
        video_title (str): Title of the currently processed video.
        """

    def __init__(self, video_title=None):
        """
        Initialises a new instance of the class.
        Parameters:
        video_title (str, optional): The title of the video. Defaults to None.
        Instance Variables: self.model (str): The model configuration loaded from common.get_configs("model").
        self.resolution (str): The video resolution. Initialised as None and set later when needed.
        self.video_title (str): The title of the video.
        """
        self.tracking_model = common.get_configs("tracking_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.resolution = None
        self.mapping = pd.read_csv(common.get_configs("mapping"))
        self.confidence = 0.0
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.output_path = common.get_configs("videos")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.delete_frames = common.get_configs("delete_frames")
        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")

    def set_video_title(self, title):
        """
        Sets the video title for the instance.
        Parameters: title (str): The new title for the video.
        """
        self.video_title = title

    def rename_folder(self, old_name, new_name):
        """
        Renames a folder from old_name to new_name.
        Parameters: old_name (str): The current name (or path) of the folder.
        new_name (str): The new name (or path) to assign to the folder.
        Error Handling: - Logs an error if the folder with old_name is not found.
        - Logs an error if a folder with new_name already exists.
        """
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    def load_upgrade_log(self):
        """
        Load package upgrade attempt log from file.
        Returns: dict: Dictionary with package names and last upgrade date.
        """
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    def save_upgrade_log(self, log_data):
        """ Save package upgrade log to a JSON file.
        Parameters: log_data (dict): Dictionary of package upgrade dates.
        """
        with open(UPGRADE_LOG_FILE, "w") as file:
            json.dump(log_data, file)

    def was_upgraded_today(self, package_name):
        """ Check whether the given package was already upgraded today.
        Parameters:
                package_name (str): Name of the package.
        Returns: bool: True if upgraded today, False otherwise.
        """
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name):
        """
        Mark a package as upgraded by saving today's date in the log.
        Parameters: package_name (str): Name of the package.
        """
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name: str) -> None:
        """
        Upgrade a Python package using `uv` (pip-compatible mode), once per day.

        This method avoids repeated upgrade attempts within a single day by consulting
        `was_upgraded_today(package_name)`. Upgrades are executed using uv's pip interface:

            uv pip install --upgrade <package>

        Cross-platform behavior:
          - Resolves the `uv` executable via PATH (shutil.which) and common user install locations.
          - Targets the currently running interpreter's environment by passing:
                --python sys.executable
            This ensures the upgrade applies to the same virtual environment (.venv) that is
            executing this code, regardless of OS.

        Parameters:
            package_name: The distribution name to upgrade (e.g., "requests", "pydantic").

        Notes:
            If `uv` cannot be found, the method logs an error and records the attempt for today
            to avoid repeated failures.
        """
        if self.was_upgraded_today(package_name):
            logging.debug("%s upgrade already attempted today. Skipping.", package_name)
            return

        uv_exe = Youtube_Helper._resolve_uv_executable()

        if not uv_exe:
            logging.error(
                "Cannot upgrade %s because `uv` was not found on PATH (or common install locations).",
                package_name,
            )
            # Still record the attempt to avoid retry loops.
            self.mark_as_upgraded(package_name)
            return

        cmd = [
            uv_exe,
            "pip",
            "install",
            "--upgrade",
            package_name,
            # Ensures the operation applies to the environment of this running Python.
            "--python",
            sys.executable,
        ]

        try:
            logging.info("Upgrading %s with uv (targeting %s)...", package_name, sys.executable)
            subprocess.check_call(cmd)
            logging.info("%s upgraded successfully.", package_name)
            self.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError as e:
            logging.error("Failed to upgrade %s with uv: %s", package_name, e)
            # Still record the attempt to avoid retry loops.
            self.mark_as_upgraded(package_name)

    @staticmethod
    def _resolve_uv_executable() -> str | None:
        """
        Resolve the `uv` executable in a cross-platform manner.

        Resolution order:
          1) Use PATH resolution via shutil.which("uv")
          2) Check common user install locations:
             - Unix/macOS: ~/.local/bin/uv
             - Windows: %USERPROFILE%\\.local\\bin\\uv.exe (less common, but included)
             - Windows: %APPDATA%\\uv\\uv.exe (possible for some installers)
          3) Check venv scripts/bin directory (if uv is installed into the venv)

        Returns:
            Absolute path to `uv` if found; otherwise None.
        """
        # 1) PATH resolution
        uv_on_path = shutil.which("uv")
        if uv_on_path:
            return uv_on_path

        candidates: list[str] = []

        home = os.path.expanduser("~")

        # 2) Common user locations
        # Unix-like
        candidates.append(os.path.join(home, ".local", "bin", "uv"))

        # Windows common knowledge: uv.exe naming
        candidates.append(os.path.join(home, ".local", "bin", "uv.exe"))

        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(os.path.join(appdata, "uv", "uv.exe"))

        # 3) Venv-local (if uv installed inside this environment)
        # Works for both Windows and Unix-like: sys.executable points into venv.
        py_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(py_dir, "uv"))       # Unix-like venv: .venv/bin/uv
        candidates.append(os.path.join(py_dir, "uv.exe"))   # Windows venv: .venv/Scripts/uv.exe

        for path in candidates:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    @staticmethod
    def _wait_for_stable_file(src, checks=2, interval=0.5, timeout=30):
        """
        Wait until src exists and its size is unchanged for checks consecutive checks.
        Returns True if stable within timeout, else False.
        """
        deadline = time.time() + timeout
        last = -1
        stable = 0

        while time.time() < deadline:
            if os.path.isfile(src):
                try:
                    size = os.stat(src).st_size
                except OSError:
                    size = -1

                if size == last:
                    stable += 1
                    if stable >= checks:
                        return True
                else:
                    stable = 0

                last = size

            time.sleep(interval)

        return False

    def copy_video_safe(self, base_video_path, internal_ssd, vid, max_attempts=5, backoff=0.6):
        """
        Copies base_video_path -> os.path.join(internal_ssd, f"{vid}.mp4")
        - Ensures dest dir exists - Waits for source file to be stable
        - Copies to temp, then atomic replace - Verifies size - Retries on transient OS errors
        """
        if not vid or str(vid).strip() == "":
            raise ValueError("vid must be a non-empty string")

        dest_dir = internal_ssd
        dest = os.path.join(dest_dir, f"{vid}.mp4")
        tmp = dest + ".tmp"

        # Ensure destination directory
        os.makedirs(dest_dir, exist_ok=True)

        # Avoid copying onto itself (only works if both exist)
        try:
            if os.path.exists(dest) and os.path.exists(base_video_path) and os.path.samefile(base_video_path, dest):
                return dest
        except OSError:
            # samefile can raise if either path doesn't fully exist yet; ignore
            pass

        # Wait for source to appear and stabilize
        if not Youtube_Helper._wait_for_stable_file(base_video_path):
            raise FileNotFoundError(f"Source never became available or stayed unstable: {base_video_path}")

        # Cache expected size for verification
        try:
            src_size = os.stat(base_video_path).st_size
        except OSError as e:
            raise FileNotFoundError(f"Cannot stat source: {base_video_path}: {e!r}")

        attempt = 0
        while True:
            attempt += 1
            try:
                # Copy to temp
                shutil.copy2(base_video_path, tmp)

                # Best-effort flush of temp file contents
                try:
                    with open(tmp, "rb") as f:
                        os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass

                # Atomic replace final
                os.replace(tmp, dest)

                # Verify size
                if os.stat(dest).st_size != src_size:
                    raise OSError(f"Size mismatch after copy: src={src_size}, dst={os.stat(dest).st_size}")

                return dest

            except (OSError, IOError):
                # Clean up temp if present
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

                if attempt >= max_attempts:
                    raise

                time.sleep(backoff * attempt)

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500) -> Optional[tuple[str, str, str, float]]:
        """
        Search and download a specific .mp4 file from a multi-directory FastAPI-based
        HTTP file server (e.g., files.mobility-squad.com). This function attempts direct
        download from known /files/ paths (tue1/tue2/tue3), and if not found, recursively
        crawls the /browse pages to locate the video file. Progress is shown with tqdm.
        Args: filename (str): Target file name (with or without .mp4 extension).
        base_url (str, optional): Base URL of the file server.
        Must include protocol, e.g. "https://files.mobility-squad.com/".
        out_dir (str, optional): Local output directory to save the video.
        Defaults to current directory ".".
        username (str, optional): Username for HTTP Basic Auth.
        password (str, optional): Password for HTTP Basic Auth.
        token (str, optional): Token string for token-based authentication.
        Sent as a query parameter ?token=.... timeout (int, optional):
        Request timeout in seconds. Default is 20. max_pages (int, optional):
        Safety limit for crawl depth/pages. Default is 500.
        Returns: Optional[Tuple[str, str, str, float]]: Returns a tuple
        (local_path, filename, resolution_label, fps) if the download succeeds,
        or None if the file is not found or download fails.
        Logging: - logger.info: start, success summaries.
        - logger.debug: HTTP requests, crawl steps, file matches.
        - logger.warning: non-fatal issues (metadata failures, skipped pages).
        - logger.error: fatal errors (network/IO exceptions). Example:

            result = self.download_videos_from_http_fileserver(
                filename="3ai7SUaPoHM",
                base_url="https://files.mobility-squad.com/",
                out_dir="./downloads",
                username="mobility",
                password="your_password"
            )
            if result:
                path, name, res, fps = result
                print(f"Downloaded {name} ({res}, {fps} fps) to {path}")
            else:
                print("File not found or failed.")
        """
        # -------------------- Input Preparation --------------------
        if not base_url:
            logger.error("Base URL is missing.")
            return None

        base = base_url if base_url.endswith("/") else base_url + "/"

        if username == "":
            username = None
        if password == "":
            password = None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        filename_lower = filename_with_ext.lower()
        aliases = ["tue4", "tue5"]

        req_params = {"token": token} if token else None

        logger.info(f"Starting download for '{filename_with_ext}'")
        logger.debug(
            f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"  # noqa:E501
        )  # noqa: E501

        # ---------- Session ----------
        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET with logging and safe error handling."""
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # ---------- 1. Try direct /files paths ----------
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                logger.debug(f"Trying direct URL: {direct_url}")

                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                logger.info(f"Found file via direct URL: {direct_url}")
                content_len = int(r.headers.get("content-length", 0))
                logger.debug(f"Content-Length: {content_len or 'unknown'} bytes")

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # Avoid overwriting
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                # ---------- Download ----------
                try:
                    total = content_len or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading from ftp: {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Download complete: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download failed for {filename_with_ext}: {e}")
                    return None

                # ---------- Metadata ----------
                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Youtube_Helper.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata extracted: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                logger.info(f"✅ Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename, resolution, fps

            # ---------- 2. Crawl /browse fallback ----------
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                """Recursively traverse /browse pages."""
                stack = [start_url]
                pages_seen = 0

                while stack:
                    url = stack.pop()

                    if url in visited:
                        continue

                    visited.add(url)
                    pages_seen += 1
                    if pages_seen > max_pages:
                        logger.warning(f"Crawl aborted after {max_pages} pages.")
                        return None

                    resp = fetch(url)
                    if resp is None:
                        continue

                    try:
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception as e:
                        logger.warning(f"HTML parse failed at {url}: {e}")
                        continue

                    for a in soup.find_all("a"):
                        href = (a.get("href") or "").strip()  # type: ignore
                        if not href:
                            continue

                        full = urljoin(url, href)

                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                logger.info(f"File located via crawl: {full}")
                                return full

                        if is_dir_link(href):
                            stack.append(full)

                logger.debug("Crawl finished — no file found.")
                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                logger.debug(f"Crawling alias: {alias} -> {start_url}")

                found = crawl(start_url)
                if not found:
                    continue

                r = fetch(found, stream=True)
                if not r:
                    continue

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                try:
                    total = int(r.headers.get("content-length", 0)) or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Downloaded via crawl: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download during crawl failed: {e}")
                    return None

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Youtube_Helper.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    def download_video_with_resolution(self, vid, resolutions=["720p", "480p", "360p", "144p"], output_path="."):
        """
        Downloads a YouTube video in one of the specified resolutions and returns video details.
        Uses pytubefix first, then yt-dlp as fallback.
        Downloads into output_path/.ftp_tmp first, then moves the completed file to output_path.
        """
        selected_resolution = None

        temp_dir = os.path.join(output_path, ".ftp_tmp")
        final_video_file_path = os.path.join(output_path, f"{vid}.mp4")
        temp_video_file_path = os.path.join(temp_dir, f"{vid}.mp4")

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        try:
            if self.update_package and datetime.datetime.today().weekday() == 0:
                self.upgrade_package_if_needed("pytubefix")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"

            if self.need_authentication:
                youtube_object = YouTube(
                    youtube_url,
                    self.client,
                    use_oauth=True,
                    allow_oauth_cache=True,
                    on_progress_callback=on_progress,
                )
            else:
                youtube_object = YouTube(youtube_url, self.client, on_progress_callback=on_progress)

            selected_stream = None
            selected_resolution = None

            # 1) Preferred resolutions (exact match)
            for resolution in resolutions:
                video_streams = youtube_object.streams.filter(res=resolution)
                if video_streams:
                    selected_resolution = resolution
                    logger.debug(f"Found video {vid} in {resolution}.")
                    selected_stream = video_streams.first() if hasattr(video_streams, "first") else video_streams[0]
                    break

            # 2) Fallback logic if no preferred match
            if not selected_stream:

                def _height(s) -> int:
                    h = getattr(s, "height", None)
                    if isinstance(h, int) and h > 0:
                        return h
                    res_attr = getattr(s, "resolution", None) or getattr(s, "res", None) or ""
                    m = re.search(r"(\d{3,4})p", str(res_attr))
                    return int(m.group(1)) if m else -1

                def _is_mp4(s) -> bool:
                    mime = getattr(s, "mime_type", "") or ""
                    return "mp4" in mime.lower()

                def _pick_prefer_progressive(streams_at_height):
                    progressive = [s for s in streams_at_height if getattr(s, "is_progressive", False)]
                    return progressive[0] if progressive else streams_at_height[0]

                streams = list(youtube_object.streams)

                # 2a) Highest available <= 720p
                le_720 = [s for s in streams if _is_mp4(s) and 0 < _height(s) <= 720]
                if le_720:
                    max_h = max(_height(s) for s in le_720)
                    at_max_h = [s for s in le_720 if _height(s) == max_h]
                    chosen = _pick_prefer_progressive(at_max_h)
                    selected_stream = chosen
                    selected_resolution = getattr(chosen, "resolution", None) or getattr(chosen, "res", None) or f"{max_h}p"  # noqa:E501
                    logger.debug(f"{vid}: no preferred match; picked highest available {selected_resolution} (≤720p).")
                else:
                    # 2b) Lowest available > 720p
                    gt_720 = [s for s in streams if _is_mp4(s) and _height(s) > 720]
                    if not gt_720:
                        logger.error(f"{vid}: no MP4 stream available at any resolution.")
                        return None

                    min_h = min(_height(s) for s in gt_720)
                    at_min_h = [s for s in gt_720 if _height(s) == min_h]
                    chosen = _pick_prefer_progressive(at_min_h)

                    selected_stream = chosen
                    selected_resolution = getattr(chosen, "resolution", None) or getattr(chosen, "res", None) or f"{min_h}p"  # noqa:E501
                    logger.debug(f"{vid}: no ≤720p stream; picked lowest available >720p ({selected_resolution}).")

            logger.info(f"{vid}: download in {selected_resolution} started with pytubefix.")

            if os.path.exists(temp_video_file_path):
                os.remove(temp_video_file_path)

            selected_stream.download(temp_dir, filename=f"{vid}.mp4")
            self.video_title = youtube_object.title

            os.replace(temp_video_file_path, final_video_file_path)

            fps = self.get_video_fps(final_video_file_path)
            logger.info(f"{vid}: FPS={fps}.")

            return final_video_file_path, vid, selected_resolution, fps

        except Exception as e:
            logger.error(f"{vid}: pytubefix download method failed: {e}")

            try:
                if self.update_package and datetime.datetime.today().weekday() == 0:
                    self.upgrade_package_if_needed("yt-dlp")

                youtube_url = f"https://www.youtube.com/watch?v={vid}"

                yt_dlp_common_opts = {
                    # Same as CLI: --cookies-from-browser chrome
                    "cookiesfrombrowser": ("brave",),
                    # Same as CLI: --remote-components ejs:github
                    "remote_components": ["ejs:github"],
                    "noplaylist": True,
                }

                # Keep this broad. The working CLI selected formats such as 299+251
                # after reading m3u8 information, so do not reject HLS formats here.
                format_spec = "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"
                selected_resolution = "best<=720p"

                ydl_opts = {
                    **yt_dlp_common_opts,
                    "outtmpl": os.path.join(temp_dir, f"{vid}.%(ext)s"),
                    "format": format_spec,
                    "merge_output_format": "mp4",
                    "final_ext": "mp4",
                    # Temporarily set this to True if you need to confirm cookies are loaded.
                    "verbose": False,
                }

                logger.info(f"{vid}: download in {selected_resolution} started with yt-dlp.")

                if os.path.exists(temp_video_file_path):
                    os.remove(temp_video_file_path)

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # pyright: ignore[reportArgumentType]
                    info = ydl.extract_info(youtube_url, download=True)

                self.video_title = info.get("title") if isinstance(info, dict) else None

                os.replace(temp_video_file_path, final_video_file_path)

                fps = self.get_video_fps(final_video_file_path)
                logger.info(f"{vid}: FPS={fps}.")

                return final_video_file_path, vid, selected_resolution, fps

            except Exception as ytdlp_error:
                logger.error(f"{vid}: yt-dlp download method failed: {ytdlp_error}")
                return None

    def get_video_fps(self, video_file_path):
        """
        Retrieves the frames per second (FPS) of a video file using OpenCV.
        Parameters: video_file_path (str): The file path to the video whose FPS is to be determined.
        Returns: int or None: The rounded FPS value of the video if successful;
        otherwise, returns None if an error occurs.
        The function performs the following steps:
        1. Opens the video file using OpenCV's VideoCapture.
        2. Retrieves the FPS using the CAP_PROP_FPS property.
        3. Rounds the FPS value to the nearest integer.
        4. Releases the video resource.
        5. Returns the rounded FPS value, or None if an exception is encountered.
        """
        try:
            # Open the video file using OpenCV
            video = cv2.VideoCapture(video_file_path)

            # Retrieve the FPS using OpenCV's CAP_PROP_FPS property
            fps = video.get(cv2.CAP_PROP_FPS)

            # Release the video resource
            video.release()

            # Return the FPS rounded to the nearest integer
            return round(fps, 0)
        except Exception as e:
            # Log an error message if FPS retrieval fails
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    @staticmethod
    def get_video_resolution_label(video_path: str) -> str:
        """
        Return a resolution label for a local video file using an "exact, truthful" policy.

        Policy
        -----------------
        - Read the frame height (pixels) from the file via OpenCV.
        - If the height matches a known standard, return its label (e.g., "720p", "1080p").
        - If the height is close to a known standard within a small tolerance (to account for
          encoder/container padding such as 1088 instead of 1080), return the nearest
          standard label.
        - Otherwise, return the exact height in the form "<height>p" (e.g., "540p", "768p").

        This approach remains compatible with the updated download selection logic, which
        may select non-standard heights when they are the best available option.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.

        Returns
        -------
        str
            A resolution label (e.g., "144p", "360p", "720p", "1080p") or "<height>p" for
            non-standard heights.

        Raises
        ------
        FileNotFoundError
            If `video_path` does not exist.
        RuntimeError
            If the video cannot be opened or the frame height cannot be determined.

        Notes
        -----
        - The label is derived from frame height only (not bitrate, codec, aspect ratio, etc.).
        - Some videos may report padded heights (e.g., 544, 736, 1088). These are mapped to
          the nearest standard label only when within the configured tolerance.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        if height <= 0:
            raise RuntimeError(f"Could not determine frame height for video: {video_path}")

        # Canonical/common heights (add more if we want "named" labels, but non-matches still fall back to "<height>p")
        labels = {
            144: "144p",
            240: "240p",
            360: "360p",
            480: "480p",
            540: "540p",
            576: "576p",
            720: "720p",
            900: "900p",
            1080: "1080p",
            1440: "1440p",
            2160: "2160p",  # 4K UHD
            4320: "4320p",  # 8K UHD
        }

        # Small tolerance to normalize padded encodes (e.g., 1088 -> 1080).
        tolerance_px = 16

        # Exact match
        if height in labels:
            return labels[height]

        # Nearest label within tolerance (padding normalization only)
        closest_h = min(labels.keys(), key=lambda h: abs(height - h))
        if abs(height - closest_h) <= tolerance_px:
            return labels[closest_h]

        # Truthful fallback for truly non-standard heights
        return f"{height}p"

    def trim_video(self, input_path, output_path, start_time, end_time):
        """
        Trims a segment from a video and saves the result to a specified file.
        Parameters: input_path (str): The file path to the original video.
        output_path (str): The destination file path where the trimmed video will be saved.
        start_time (float or str): The start time for the trimmed segment.
        This can be specified in seconds or in a time format recognised by MoviePy.
        nd_time (float or str): The end time for the trimmed segment.
        Similar to start_time, it can be in seconds or another supported time format.
        Returns: None The function performs the following steps:
        1. Loads the original video using MoviePy's VideoFileClip.
        2. Creates a subclip from the original video based on the provided start_time and end_time.
        3. Writes the subclip to the output_path using the H.264 video codec and AAC audio codec.
        4. Closes the video file to free up resources.
        """
        # Load the video and create a subclip using the provided start and end times.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()

    def draw_yolo_boxes_on_video(self, df, fps, video_path, output_path):
        """
        Draw YOLO-style bounding boxes on a video and save the annotated output.
        This method takes a DataFrame containing normalised bounding box coordinates
        (in YOLO format), matches them frame-by-frame to the input video,
        draws the corresponding boxes and labels, and writes the resulting video to disk.
        Args: df (pd.DataFrame): DataFrame containing at least the following columns:
        - 'frame-count': Original frame indices in the source video.
        - 'X-center', 'Y-center': Normalised center coordinates (0 to 1).
        - 'width', 'Height': Normalised width and height (0 to 1).
        - 'unique-id': Identifier to display in the label.
        fps (float): Frames per second for the output video.
        video_path (str): Path to the input video file.
        output_path (str): Path to save the annotated output video.

        Raises: IOError: If the input video cannot be opened.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Normalise frame indices to start from 0
        min_frame = df["frame-count"].min()
        df["Frame Index"] = df["frame-count"] - min_frame

        # Attempt to open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video dimensions and total number of frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer with the same resolution and specified fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Writing to {output_path} ({width}x{height} @ {fps}fps)")

        frame_index = 0

        # Process each frame
        while frame_index < total_frames:
            success, frame = cap.read()
            if not success:
                logger.error(f"Failed to read frame {frame_index}")
                break

            # Filter YOLO data for this adjusted frame index
            frame_data = df[df["Frame Index"] == frame_index]
            for _, row in frame_data.iterrows():
                # Convert normalised coordinates to absolute pixel values
                x_center = row["x-center"] * width
                y_center = row["y-center"] * height
                w = row["width"] * width
                h = row["height"] * height

                # Calculate top-left and bottom-right corners of the box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle and label with unique ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {int(row['unique-id'])}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the modified frame to the output video
            out.write(frame)
            frame_index += 1

        # Release video objects
        cap.release()
        out.release()

    def detect_gpu(self):
        """
        Detects whether an NVIDIA or Intel GPU is available and returns the appropriate FFmpeg encoder.
        Returns: str: 'hevc_nvenc' for NVIDIA, 'hevc_qsv' for Intel, or None if no compatible GPU is found.
        """
        try:
            # Check for NVIDIA GPU
            nvidia_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_check.returncode == 0:
                return "hevc_nvenc"

            # Check for Intel QuickSync GPU
            intel_check = subprocess.run(["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Intel" in intel_check.stdout:
                return "hevc_qsv"

        except FileNotFoundError:
            pass  # Command not found, meaning the hardware isn't available or the driver isn't installed

        return None  # No compatible GPU found

    def extract_youtube_id(self, file_path):
        """
        Extracts the YouTube video ID from a given file path.
        Args: file_path (str): The full path of the video file.
        Returns: str: The extracted YouTube ID.
        Raises: ValueError: If no valid YouTube ID is found.
        """
        filename = os.path.basename(file_path)  # Get only the filename
        youtube_id, ext = os.path.splitext(filename)  # Remove the file extension
        if not youtube_id or len(youtube_id) < 5:  # Basic validation
            raise ValueError("Invalid YouTube ID extracted.")
        return youtube_id

    def create_video_from_images(self, image_folder, output_path,
                                 video_title, bbox_mode=False, frame_rate=30):
        """
        Creates a video file from a sequence of image frames. The output filename will reflect the mode used.
        Parameters: image_folder (str): Folder containing frame images.
        output_path (str): Directory where the output video will be saved.
        video_title (str): Base title for the video.
        bbox_mode (bool): Whether bounding box mode is used.
        frame_rate (int or float): Frame rate for the video.
        """
        # Decide on the output filename based on mode
        if bbox_mode:
            output_filename = f"{video_title}_mod_bbox.mp4"
        else:
            output_filename = f"{video_title}_mod.mp4"

        output_video_path = os.path.join(output_path, output_filename)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        def extract_frame_number(filename):
            match = re.search(r'frame_tracked_(\d+)\.jpg', filename)
            return int(match.group(1)) if match else float('inf')  # Push invalid ones to the end

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        if not images:
            logger.error("No JPG images found in the specified folder.")
            return

        images.sort(key=extract_frame_number)

        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            logger.error(f"Could not read the first image: {first_image_path}")
            return

        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            if frame is not None:
                video.write(frame)
            else:
                logger.error(f"Failed to read frame: {img_path}")

        video.release()
        logger.info(f"Video created successfully at: {output_video_path}")
        return output_video_path

    def merge_txt_to_csv_dynamically_bbox(self, txt_location, output_csv, frame_count):
        """
        Merges YOLO-format label data from a .txt file into a CSV, frame by frame.
        Parameters: txt_location (str): Directory containing label .txt files.
        output_csv (str): Path to the CSV file to update.
        frame_count (int): Frame number being processed (used for naming).
        """
        # Define the path for the new text file
        new_txt_file_name = os.path.join(txt_location, f"label_{frame_count}.txt")

        # Read data from the new text file
        with open(new_txt_file_name, 'r') as text_file:
            data = text_file.read()

        # Save the data into the new text file
        with open(new_txt_file_name, 'w') as new_file:
            new_file.write(data)

        # Read the newly created text file into a DataFrame
        df = pd.read_csv(
            new_txt_file_name,
            delimiter=" ",
            header=None,
            names=["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence"],
        )
        df['frame-count'] = frame_count

        # Append the DataFrame to the CSV file
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode='w')  # If the CSV does not exist, create it
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)  # If it exists, append without header

    def delete_folder(self, folder_path):
        """
        Deletes the folder and all its contents recursively.
        Parameters: folder_path (str): The path of the folder to delete.
        Returns: bool: True if the folder was successfully deleted, False otherwise.
        """
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Folder '{folder_path}' deleted successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to delete folder '{folder_path}': {e}")
                return False
        else:
            logger.info(f"Folder '{folder_path}' does not exist.")
            return False

    def delete_youtube_mod_videos(self, folders):
        """
        Deletes files of the form {youtube_id}_mod.mp4 from the given list of folders.
        Args: folders (list): List of folder paths to scan.
        """
        pattern = re.compile(r"^[A-Za-z0-9_-]{11}_mod\.mp4$")
        for folder in folders:
            if not os.path.exists(folder):
                logger.info(f"Skipping missing folder: {folder}")
                continue
            for filename in os.listdir(folder):
                if pattern.match(filename):
                    file_path = os.path.join(folder, filename)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.info(f"Failed to delete {file_path}: {e}")

    def get_iso_alpha_3(self, country_name, existing_iso):
        """
        Converts a country name to ISO 3166-1 alpha-3 format.
        Parameters: country_name (str): Full country name.
        existing_iso (str): Existing ISO code as fallback.
        Returns: str or None: ISO 3166-1 alpha-3 code or fallback value.
        """
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except LookupError:
            if country_name.strip().upper() == "KOSOVO":
                return "XKX"  # User-assigned code for Kosovo
            return existing_iso if existing_iso else None

    def get_latest_population(self):
        """
        Fetches the latest available population data from World Bank.
        Returns: pd.DataFrame: Population data with columns ['iso3', 'Year', 'Population'].
        """
        # Search for the population indicator
        indicator = 'SP.POP.TOTL'  # Total Population (World Bank indicator code)

        # Fetch the latest population data
        population_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        population_df = population_data.reset_index()

        # Rename columns appropriately
        population_df = population_df.rename(columns={
            population_df.columns[0]: 'iso3',  # type: ignore
            population_df.columns[2]: 'Year',  # type: ignore
            population_df.columns[3]: 'Population'  # type: ignore
        })

        # Divide population by 1000
        population_df['Population'] = population_df['Population'] / 1000
        return population_df

    def update_population_in_csv(self, data):
        """
        Updates the mapping DataFrame with the latest country population data.
        Parameters: data (pd.DataFrame): The mapping DataFrame to update.
        """
        # Ensure the required columns exist in the CSV
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")
        if "population_country" not in data.columns:
            data["population_country"] = None  # Initialise the column if it doesn't exist

        # Get the latest population data
        latest_population = self.get_latest_population()

        # Create a dictionary for quick lookup
        population_dict = dict(zip(latest_population['iso3'], latest_population['Population']))

        # Update the population_country column
        for index, row in data.iterrows():
            iso3 = row["iso3"]
            population = population_dict.get(iso3, None)
            # Always update with the latest population data
            data.at[index, "population_country"] = population

        # Save the updated DataFrame back to the same CSV
        data.to_csv(self.mapping, index=False)
        logger.info("Mapping file updated successfully with country population.")

    def get_latest_gini_values(self):
        """
        Fetch the latest GINI index data from the World Bank.
        Returns: pd.DataFrame: DataFrame with iso3 and the latest GINI index values.
        """
        # World Bank indicator for GINI index
        indicator = 'SI.POV.GINI'  # GINI index

        # Fetch the most recent data (mrv=1 for the most recent value)
        geni_data = wb.get_series(indicator, id_or_value='id', mrv=1)

        # Convert the data to a DataFrame
        geni_df = geni_data.reset_index()

        # Rename columns appropriately
        geni_df = geni_df.rename(columns={
            geni_df.columns[0]: 'iso3',  # type: ignore
            geni_df.columns[2]: 'Year',  # type: ignore
            geni_df.columns[3]: 'gini'  # type: ignore
        })

        # Keep only the latest value for each country
        geni_df = geni_df.sort_values(by=['iso3', 'Year'], ascending=[True, False]).drop_duplicates(subset=['iso3'])
        return geni_df[['iso3', 'gini']]

    def fill_gini_data(self, df):
        """
        Fill the GINI index column in a CSV file using World Bank data.
        Args: file_path (str): Path to the input CSV file.
        """
        try:
            # Ensure the required column exists
            if 'gini' not in df.columns:
                logger.error("The required columns 'iso3' and/or 'gini' are missing from the file.")
                return

            # Get the latest GINI index data
            geni_df = self.get_latest_gini_values()

            # Merge the GINI index data with the existing DataFrame
            updated_df = pd.merge(df, geni_df, on='iso3', how='left', suffixes=('', '_new'))

            # Update the geni column with the new data
            updated_df['gini'] = updated_df['gini_new'].combine_first(updated_df['gini'])

            # Drop the temporary column
            updated_df = updated_df.drop(columns=['gini_new'])

            # Update in-memory mapping
            self.mapping = updated_df

            # Persist to the original CSV path
            mapping_path = common.get_configs("mapping")  # path to the mapping CSV
            updated_df.to_csv(mapping_path, index=False)
            logger.info("Mapping file updated successfully with GINI value.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def update_track_buffer_in_yaml(self, yaml_path, video_fps):
        # Load existing YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update the track_buffer value
        config['track_buffer'] = common.get_configs("track_buffer_sec") * video_fps

        # Write it back to the YAML file (overwrite)
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def tracking_mode(self, input_video_path, output_video_path, video_title,
                      video_fps, bbox_mode, flag=0, run_root='runs'):
        """
        Performs object tracking on a video using YOLO models and saves results.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ------------------------------------------------------------------
        # Multi-thread safety: isolate all YOLO "runs/*" side-effects per task
        # and avoid in-place modifications of shared tracker YAML files.
        # ------------------------------------------------------------------
        run_root = run_root or "runs"
        detect_root = os.path.join(run_root, "detect")
        os.makedirs(detect_root, exist_ok=True)

        def _resolve_yaml(src_name: str) -> str:
            # Accept either a filename (repo-relative) or an absolute path.
            if os.path.isabs(src_name) and os.path.exists(src_name):
                return src_name
            # try current working directory
            if os.path.exists(src_name):
                return src_name
            # try relative to this file
            here = os.path.dirname(__file__)
            candidate = os.path.join(here, src_name)
            return candidate if os.path.exists(candidate) else src_name

        bbox_tracker_path = self.bbox_tracker

        if bbox_mode and self.bbox_tracker == "bbox_custom_tracker.yaml":
            src = _resolve_yaml(self.bbox_tracker)
            bbox_tracker_path = os.path.join(detect_root, os.path.basename(self.bbox_tracker))
            try:
                shutil.copyfile(src, bbox_tracker_path)
                self.update_track_buffer_in_yaml(bbox_tracker_path, video_fps)
            except Exception as e:
                logger.warning(f"Failed to prepare bbox tracker YAML copy: {e!r}. Using original tracker path.")

        # --- Model Initialisation ---
        if bbox_mode:
            bbox_model = YOLO(self.tracking_model)

        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        # Output directories
        if bbox_mode:
            bbox_frames_output_path = os.path.join(detect_root, "frames")
            bbox_annotated_frame_output_path = os.path.join(detect_root, "annotated_frames")
            bbox_tracked_frame_output_path = os.path.join(detect_root, "tracked_frame")
            bbox_txt_output_path = os.path.join(detect_root, "labels")
            bbox_text_filename = os.path.join(detect_root, "track", "labels", "image0.txt")
            bbox_display_video_output_path = os.path.join(detect_root, "display_video.mp4")
            os.makedirs(bbox_frames_output_path, exist_ok=True)
            os.makedirs(bbox_txt_output_path, exist_ok=True)
            os.makedirs(bbox_annotated_frame_output_path, exist_ok=True)
            os.makedirs(bbox_tracked_frame_output_path, exist_ok=True)

        # Video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        if bbox_mode and self.display_frame_tracking:
            bbox_video_writer = cv2.VideoWriter(
                bbox_display_video_output_path, fourcc, video_fps, (frame_width, frame_height)  # type: ignore
            )

        # Progress bar
        if total_frames == 0:
            logger.warning("Warning: Could not determine total frames. Progress bar may not work correctly.")
            total_frames = None

        progress_bar = tqdm(total=total_frames, unit="frames", dynamic_ncols=True)
        frame_count = 0

        # Track history
        if bbox_mode:
            bbox_track_history = defaultdict(lambda: [])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # -------- BBOX MODE --------
            bbox_failed = False
            bbox_boxes_xywh = None
            bbox_track_ids = []
            bbox_confidences = []
            bbox_annotated_frame = frame.copy()

            if bbox_mode:
                try:
                    bbox_results = bbox_model.track(  # type: ignore
                        frame,
                        tracker=bbox_tracker_path,
                        persist=True,
                        conf=self.confidence,
                        save=False,
                        save_txt=False,
                        project=detect_root,
                        name="track",
                        exist_ok=True,
                        line_width=LINE_THICKNESS,
                        show_labels=SHOW_LABELS,
                        show_conf=SHOW_CONF,
                        show=RENDER,
                        verbose=False,
                        device=device,
                    )

                    bbox_boxes_obj = bbox_results[0].boxes
                    bbox_boxes_xywh = bbox_boxes_obj.xywh.cpu() if bbox_boxes_obj is not None else None  # type: ignore
                    if bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                        bbox_track_ids = (
                            bbox_boxes_obj.id.int().cpu().tolist()  # type: ignore
                            if hasattr(bbox_boxes_obj, "id") and bbox_boxes_obj.id is not None  # type: ignore
                            else []
                        )
                        bbox_confidences = (  # noqa: F841
                            bbox_boxes_obj.conf.cpu().tolist()  # type: ignore
                            if hasattr(bbox_boxes_obj, "conf") and bbox_boxes_obj.conf is not None  # type: ignore
                            else []
                        )

                        # Only render overlays if requested (plotting is CPU-expensive)
                        if self.save_annoted_img or self.save_tracked_img or self.display_frame_tracking:
                            bbox_annotated_frame = bbox_results[0].plot()
                        else:
                            bbox_annotated_frame = frame.copy()
                    else:
                        logger.info(f"[Frame {frame_count}] BBox: No objects found. Using original frame.")
                        with open(bbox_text_filename, 'w') as file:  # noqa:F841   # type: ignore
                            pass
                except Exception as e:
                    logger.error(f"[Frame {frame_count}] BBox failed: {e}. Using original frame.")
                    bbox_failed = True

            progress_bar.update(1)

            # Save annotated frames
            if self.save_annoted_img:
                if bbox_mode:
                    bbox_frame_filename = os.path.join(bbox_annotated_frame_output_path,  # type: ignore
                                                       f"frame_{frame_count}.jpg")
                    cv2.imwrite(bbox_frame_filename, bbox_annotated_frame)

            # Save txt files (only if detections found and not failed)
            if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                # Write labels directly from model outputs (avoid Ultralytics save_txt I/O)
                new_txt_file_name_bbox = os.path.join(detect_root, "labels", f"label_{frame_count}.txt")
                try:
                    bbox_cls = bbox_boxes_obj.cls.int().cpu().tolist() if hasattr(bbox_boxes_obj, "cls") and bbox_boxes_obj.cls is not None else []  # type: ignore # noqa: E501
                    bbox_ids = (
                        bbox_boxes_obj.id.int().cpu().tolist()  # type: ignore
                        if hasattr(bbox_boxes_obj, "id") and bbox_boxes_obj.id is not None  # type: ignore
                        else [-1] * len(bbox_cls)
                    )
                    bbox_confs = (
                        bbox_boxes_obj.conf.float().cpu().tolist()  # type: ignore
                        if hasattr(bbox_boxes_obj, "conf") and bbox_boxes_obj.conf is not None  # type: ignore
                        else [0.0] * len(bbox_cls)
                    )

                    # Normalized xywh; fall back to manual normalization if needed
                    if hasattr(bbox_boxes_obj, "xywhn") and bbox_boxes_obj.xywhn is not None:  # type: ignore
                        xywhn = bbox_boxes_obj.xywhn.cpu().tolist()  # type: ignore
                    else:
                        xywh = bbox_boxes_obj.xywh.cpu().tolist() if hasattr(bbox_boxes_obj, "xywh") and bbox_boxes_obj.xywh is not None else []  # type: ignore # noqa: E501
                        xywhn = []
                        for (x, y, w, h) in xywh:
                            xywhn.append([x / frame_width, y / frame_height, w / frame_width, h / frame_height])

                    with open(new_txt_file_name_bbox, "w") as f:
                        for cls_i, box_i, tid_i, conf_i in zip(bbox_cls, xywhn, bbox_ids, bbox_confs):
                            x, y, w, h = box_i
                            f.write(f"{int(cls_i)} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f} {int(tid_i)} {float(conf_i):.6f}\n")  # noqa: E501
                except Exception as e:
                    logger.warning(f"[Frame {frame_count}] BBox label write failed: {e}")
                    open(new_txt_file_name_bbox, "a").close()

                bbox_labels_path = os.path.join(detect_root, "labels")
                bbox_output_csv_path = os.path.join(detect_root, f"{self.video_title}.csv")
                self.merge_txt_to_csv_dynamically_bbox(bbox_labels_path, bbox_output_csv_path, frame_count)

            if self.delete_labels is True:
                if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                    os.remove(os.path.join(detect_root, "labels", f"label_{frame_count}.txt"))

            # Save labeled image
            if self.delete_frames is False:
                if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                    bbox_image_filename = os.path.join(detect_root, "track", "image0.jpg")
                    bbox_new_img_file_name = os.path.join(detect_root, "frames", f"frame_{frame_count}.jpg")
                    if os.path.exists(bbox_image_filename):
                        shutil.move(bbox_image_filename, bbox_new_img_file_name)

            # Plot the tracks
            try:
                if bbox_mode and not bbox_failed and bbox_boxes_xywh is not None and bbox_boxes_xywh.size(0) > 0:
                    for box, track_id in zip(bbox_boxes_xywh, bbox_track_ids):
                        x, y, w, h = box
                        track = bbox_track_history[track_id]  # type: ignore
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(bbox_annotated_frame, [points], isClosed=False,
                                      color=(230, 230, 230), thickness=LINE_THICKNESS * 5)
            except Exception:
                pass

            # Display the annotated frame
            if self.display_frame_tracking:
                if bbox_mode:
                    cv2.imshow("YOLOv11 Tracking", bbox_annotated_frame)
                    bbox_video_writer.write(bbox_annotated_frame)  # type: ignore

            # Save the tracked frame here
            if self.save_tracked_img:
                if bbox_mode:
                    bbox_frame_filename = os.path.join(bbox_tracked_frame_output_path,  # type: ignore
                                                       f"frame_tracked_{frame_count}.jpg")
                    cv2.imwrite(bbox_frame_filename, bbox_annotated_frame)

            # Break the loop if 'q' is pressed, only when display is enabled
            if self.display_frame_tracking:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release resources
        cap.release()
        if self.display_frame_tracking:
            cv2.destroyAllWindows()
        progress_bar.close()

        if flag:
            if bbox_mode:
                self.create_video_from_images(
                    image_folder=bbox_tracked_frame_output_path,  # type: ignore
                    output_path=output_video_path,
                    video_title=video_title,
                    bbox_mode=bbox_mode,
                    frame_rate=video_fps,
                )
