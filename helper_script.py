# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import re
import time
from contextlib import contextmanager
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from ultralytics import YOLO  # type: ignore
import track_ids
from collections import defaultdict
from typing import Optional
import shutil
import numpy as np
import pandas as pd
import world_bank_data as wb
from custom_logger import CustomLogger
import common
import torch
import subprocess
import sys
import logging
from tqdm import tqdm
import csv
import datetime
import json
import yaml
import pathlib
import requests
import yt_dlp
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
try:
    import fcntl
except ImportError:  # Windows: mapping.csv is rewritten without the lock
    fcntl = None

logger = CustomLogger(__name__)  # use custom logger
track_ids.install()  # a counter of track ids per thread: segments run in threads (max_workers), see track_ids.py
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

# Consts
LINE_THICKNESS = 1
# columns of the bbox CSVs, one row per detection (the label files' columns, then the frame)
BBOX_COLUMNS = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]

# logging of attempts to upgrade packages
UPGRADE_LOG_FILE = "upgrade_log.json"


@contextmanager
def _mapping_lock(mapping_path):
    """Hold mapping_path + '.lock', the lock the review pages and add_video take while they rewrite the mapping."""
    with open(mapping_path + ".lock", "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _no_query(text):
    """text without the query strings of the URLs in it: a request error names its URL with ?token=..."""
    return re.sub(r"\?[^\s'\"]*", "", str(text))


class Youtube_Helper:
    """ A helper class for managing YouTube video downloads, processing, and analytics.
    Features:
              - Downloads videos via pytube or yt_dlp with resolution preference.
              - Handles video compression, trimming, and FPS extraction.
              - Applies object detection and tracking using YOLO models.
              - Updates and maintains CSV datasets with video metadata.
              - Interfaces with World Bank data to supplement mapping files.

    Attributes:
        tracking_model (str): Path or identifier for the YOLO model to use.
        video_title (str): Title of the currently processed video.
        mapping (pd.DataFrame): the mapping, read on first use.
        """

    def __init__(self, video_title=None):
        """Read the settings of the config; video_title is the title of the video (None until one is set)."""
        self.video_title = video_title
        self.tracking_model = common.get_configs("tracking_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self._mapping = None
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.output_path = common.get_configs("videos")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")
        # optional: the browser yt-dlp reads YouTube cookies from, BROWSER[+KEYRING] (default brave, unless a
        # "cookies_file" is set)
        self.cookies_browser = common.get_configs("cookies_browser", default=None)
        # optional: yt-dlp's JavaScript runtime, as its --js-runtimes RUNTIME[:PATH] (default: deno, if any)
        runtime, _, path = (common.get_configs("js_runtime", default=None) or "").partition(":")
        self.js_runtimes = {runtime: {"path": path} if path else {}} if runtime else None
        # optional: "use_pytubefix" false goes straight to yt-dlp (where pytubefix always fails, it only adds requests
        # to YouTube); "cookies_file", a Netscape cookies file (another login), is tried before the browser's
        self.use_pytubefix = bool(common.get_configs("use_pytubefix", default=True))
        self.cookies_file = common.get_configs("cookies_file", default=None) or None

    @property
    def mapping(self):
        """The mapping DataFrame, read on first use (6.5 MB, and most helpers never need it)."""
        if self._mapping is None:
            self._mapping = pd.read_csv(common.get_configs("mapping"))
        return self._mapping

    @mapping.setter
    def mapping(self, df):
        self._mapping = df

    def set_video_title(self, title):
        """
        Sets the video title for the instance.
        Parameters: title (str): The new title for the video.
        """
        self.video_title = title

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
            logger.debug(f"{package_name} upgrade already attempted today. Skipping.")
            return

        uv_exe = Youtube_Helper._resolve_uv_executable()

        if not uv_exe:
            logger.error(f"Cannot upgrade {package_name} because `uv` was not found on PATH (or common install "
                         "locations).")
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
            # yt-dlp's nightly builds: YouTube changes break its stable release for weeks (yt-dlp's own advice)
            *(["--prerelease=allow"] if package_name == "yt-dlp" else []),
        ]

        try:
            logger.info(f"Upgrading {package_name} with uv (targeting {sys.executable})...")
            subprocess.check_call(cmd)
            logger.info(f"{package_name} upgraded successfully.")
            self.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade {package_name} with uv: {e}")
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

    # The file server's folders, and how long a listing of one is kept: a video missing from the server used to cost a
    # crawl of up to max_pages pages per folder, now the folder is listed once and looked up
    FTP_ALIASES = ("tue3", "tue4", "tue5")
    FTP_INDEX_S = 3600
    _ftp_index: dict = {}  # (base URL, folder) -> (time listed, {file name in lower case: URL})

    @staticmethod
    def _save_response(r, path: str, desc: str) -> int:
        """Write a streamed response to path with a progress bar; returns the number of bytes written."""
        total = int(r.headers.get("content-length", 0)) or None
        written = 0
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    bar.update(len(chunk))
        return written

    def _video_info(self, path: str):
        """(resolution label, fps) of a video file; ("unknown", 0.0) when they cannot be read."""
        try:
            return Youtube_Helper.get_video_resolution_label(path), float(self.get_video_fps(path) or 0.0)
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {path}: {e}")
            return "unknown", 0.0

    @staticmethod
    def _free_path(path: str) -> str:
        """path, or path with " (1)", " (2)", ... before the extension when that file exists already."""
        stem, suffix = os.path.splitext(path)
        i = 0
        while os.path.exists(path):
            i += 1
            path = f"{stem} ({i}){suffix}"
        return path

    def _ftp_files(self, fetch, base: str, alias: str, max_pages: int) -> dict:
        """{file name in lower case: URL} of one folder of the file server, from its /browse pages (kept
        FTP_INDEX_S)."""
        cached = Youtube_Helper._ftp_index.get((base, alias))
        if cached and time.time() - cached[0] < Youtube_Helper.FTP_INDEX_S:
            return cached[1]
        files, stack, seen = {}, [urljoin(base, f"v/{alias}/browse")], set()
        while stack and len(seen) < max_pages:
            url = stack.pop()
            if url in seen:
                continue
            seen.add(url)
            resp = fetch(url)
            if resp is None:
                continue
            for a in BeautifulSoup(resp.text, "html.parser").find_all("a"):
                href = (a.get("href") or "").strip()  # type: ignore
                full = urljoin(url, href)
                if "/files/" in href:
                    files[pathlib.PurePosixPath(urlparse(full).path).name.lower()] = full
                    files.setdefault((a.text or "").strip().lower(), full)
                elif href.startswith("/v/") and "/browse" in href:
                    stack.append(full)
        if stack:
            logger.warning(f"File server listing of {alias} stopped after {max_pages} pages.")
        Youtube_Helper._ftp_index[(base, alias)] = (time.time(), files)
        logger.info(f"File server listing of {alias}: {len(files)} entries.")
        return files

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500) -> Optional[tuple[str, str, str, float]]:
        """
        Download a video (.mp4) from the multi-folder FastAPI file server (e.g. https://files.mobility-squad.com/).

        Each folder of FTP_ALIASES is tried at the file's own /files/ URL first; when that misses, the folder's
        listing (its /browse pages, crawled once and kept FTP_INDEX_S) says where the file is.

        Args:
            filename: the file name, with or without .mp4 (a YouTube id).
            base_url: the server, with protocol.
            out_dir: where to save it; an existing file of that name is kept and the new one saved as "NAME (1).mp4".
            username, password: HTTP Basic Auth. token: sent as ?token=... when given.
            timeout: seconds per request. debug: log each request. max_pages: limit of a folder listing.

        Returns:
            (local_path, filename, resolution_label, fps), or None when the file is not there or the download fails.
        """
        if not base_url:
            logger.error("Base URL is missing.")
            return None
        base = base_url if base_url.endswith("/") else base_url + "/"
        name = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        params = {"token": token} if token else None
        logger.info(f"Starting download for '{name}'")

        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET, or None when it fails (a missing file is expected, so it is not a warning)."""
                try:
                    r = session.get(url, timeout=timeout, params=params, stream=stream)
                    if debug:
                        logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    missing = getattr(getattr(e, "response", None), "status_code", None) == 404
                    (logger.debug if missing else logger.warning)(f"Request failed [{url}]: {_no_query(e)}")
                    return None

            r = None
            for alias in self.FTP_ALIASES:
                r = fetch(urljoin(base, f"v/{alias}/files/{name}"), stream=True)
                if r is not None:
                    break
            if r is None:
                for alias in self.FTP_ALIASES:
                    url = self._ftp_files(fetch, base, alias, max_pages).get(name.lower())
                    if url and (r := fetch(url, stream=True)) is not None:
                        break
            if r is None:
                logger.warning(f"File '{name}' not found on the file server.")
                return None

            os.makedirs(out_dir, exist_ok=True)
            local_path = self._free_path(os.path.join(out_dir, name))
            try:
                written = self._save_response(r, local_path, f"Downloading from ftp: {name}")
            except Exception as e:
                logger.error(f"Download failed for {name}: {_no_query(e)}")
                return None

        resolution, fps = self._video_info(local_path)
        logger.info(f"✅ Saved '{name}' ({written} bytes, res={resolution}, fps={fps})")
        return local_path, filename, resolution, fps

    @staticmethod
    def _stream_height(s) -> int:
        """A pytubefix stream's height in pixels, -1 when unknown."""
        h = getattr(s, "height", None)
        if isinstance(h, int) and h > 0:
            return h
        m = re.search(r"(\d{3,4})p", str(getattr(s, "resolution", None) or getattr(s, "res", None) or ""))
        return int(m.group(1)) if m else -1

    @staticmethod
    def _closest_to_720(streams):
        """(stream, resolution label) of the MP4 streams: the highest at most 720p, else the lowest above it (a
        progressive one when there is one at that height); (None, None) when there is no MP4 stream."""
        mp4 = [s for s in streams if "mp4" in (getattr(s, "mime_type", "") or "").lower()]
        low = [s for s in mp4 if 0 < Youtube_Helper._stream_height(s) <= 720]
        pool = low or [s for s in mp4 if Youtube_Helper._stream_height(s) > 720]
        if not pool:
            return None, None
        h = (max if low else min)(Youtube_Helper._stream_height(s) for s in pool)
        at_h = [s for s in pool if Youtube_Helper._stream_height(s) == h]
        chosen = next((s for s in at_h if getattr(s, "is_progressive", False)), at_h[0])
        return chosen, getattr(chosen, "resolution", None) or getattr(chosen, "res", None) or f"{h}p"

    @staticmethod
    def _browser_cookies(setting):
        """yt-dlp's cookiesfrombrowser for a "cookies_browser" setting, written as its --cookies-from-browser
        BROWSER[+KEYRING] (chrome+gnomekeyring where a background job would not find the desktop keyring by itself)."""
        browser, _, keyring = setting.partition("+")
        return (browser, None, keyring.upper() or None, None)

    @staticmethod
    def _video_seconds(path):
        """A video file's length in s (ffprobe), None when it cannot be read."""
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  path], capture_output=True, text=True, timeout=60).stdout
            return float(out.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _finish_download(self, temp_path, final_path, vid, resolution):
        """Move the finished download into place: (final_path, vid, resolution, fps)."""
        os.replace(temp_path, final_path)
        fps = self.get_video_fps(final_path)
        logger.info(f"{vid}: FPS={fps}.")
        return final_path, vid, resolution, fps

    def download_video_with_resolution(self, vid, resolutions=["720p", "480p", "360p", "144p"], output_path="."):
        """
        Download a YouTube video, the first of resolutions it has (else the highest at most 720p, else the lowest
        above), with pytubefix, then yt-dlp when that fails: with the "cookies_file" login, then the "cookies_browser"
        one (else Brave), then none.
        The file goes to output_path/.ftp_tmp first and is moved to output_path/VID.mp4 when complete.

        Returns:
            (path, vid, resolution, fps), or None when both fail.
        """
        temp_dir = os.path.join(output_path, ".ftp_tmp")
        final_path = os.path.join(output_path, f"{vid}.mp4")
        temp_path = os.path.join(temp_dir, f"{vid}.mp4")
        os.makedirs(temp_dir, exist_ok=True)
        youtube_url = f"https://www.youtube.com/watch?v={vid}"
        monday = datetime.datetime.today().weekday() == 0

        try:
            if not self.use_pytubefix:
                raise RuntimeError("pytubefix is off (use_pytubefix)")
            if self.update_package and monday:
                self.upgrade_package_if_needed("pytubefix")
            if self.need_authentication:
                yt = YouTube(youtube_url, self.client, use_oauth=True, allow_oauth_cache=True,
                             on_progress_callback=on_progress)
            else:
                yt = YouTube(youtube_url, self.client, on_progress_callback=on_progress)

            stream, resolution = None, None
            for res in resolutions:  # the preferred resolutions, exactly
                found = yt.streams.filter(res=res)
                if found:
                    stream, resolution = (found.first() if hasattr(found, "first") else found[0]), res
                    logger.debug(f"Found video {vid} in {res}.")
                    break
            if stream is None:
                stream, resolution = self._closest_to_720(list(yt.streams))
                if stream is None:  # raised, so that yt-dlp has a go
                    raise RuntimeError("no MP4 stream available at any resolution")
                logger.debug(f"{vid}: no preferred resolution; picked {resolution}.")

            logger.info(f"{vid}: download in {resolution} started with pytubefix.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            stream.download(temp_dir, filename=f"{vid}.mp4")
            self.video_title = yt.title
            return self._finish_download(temp_path, final_path, vid, resolution)

        except Exception as e:
            logger.error(f"{vid}: pytubefix download method failed: {e}")

        try:
            if self.update_package and monday:
                self.upgrade_package_if_needed("yt-dlp")
            resolution = "best<=720p"
            ydl_opts = {
                "remote_components": ["ejs:github"],  # as --remote-components ejs:github
                "noplaylist": True,
                "outtmpl": os.path.join(temp_dir, f"{vid}.%(ext)s"),
                # broad on purpose: the working CLI picked formats such as 299+251 from m3u8 information
                "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
                "merge_output_format": "mp4",
                "final_ext": "mp4",
                "verbose": False,  # True shows whether the cookies were loaded
                **({"js_runtimes": self.js_runtimes} if self.js_runtimes else {}),
            }
            logger.info(f"{vid}: download in {resolution} started with yt-dlp.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # the logins in turn, as --cookies / --cookies-from-browser, then none: a login YouTube has turned against
            # answers "Video unavailable" for public videos (lin4080's Chrome, 2026-10-15) and no login meets the
            # bot check, so when one fails the next has a go
            browser = self.cookies_browser or (None if self.cookies_file else "brave")
            logins = ([("the cookies file", {"cookiefile": self.cookies_file})] if self.cookies_file else []) \
                + ([(browser, {"cookiesfrombrowser": self._browser_cookies(browser)})] if browser else []) \
                + [("no login", {})]
            for i, (name, login) in enumerate(logins):
                try:
                    with yt_dlp.YoutubeDL({**ydl_opts, **login}) as ydl:  # pyright: ignore[reportArgumentType]
                        info = ydl.extract_info(youtube_url, download=True)
                    # an HLS (m3u8) stream can stop early and yt-dlp still reports success: 61 videos were cut off,
                    # one to 10 s of 43 min (2026-10-17); a file over 5 s shorter than YouTube says is a failure
                    want = info.get("duration") if isinstance(info, dict) else None
                    got = self._video_seconds(temp_path) if want else None
                    if want and got is not None and got < want - 5:
                        os.remove(temp_path)
                        raise yt_dlp.utils.DownloadError(f"cut off: {got:.0f} s of {want:.0f} s")
                    break
                except yt_dlp.utils.DownloadError as e:
                    if i == len(logins) - 1:
                        raise
                    logger.warning(f"{vid}: failed with {name} ({e}); trying {logins[i + 1][0]}.")
            self.download_error = None
            self.video_title = info.get("title") if isinstance(info, dict) else None
            return self._finish_download(temp_path, final_path, vid, resolution)

        except Exception as ytdlp_error:
            logger.error(f"{vid}: yt-dlp download method failed: {ytdlp_error}")
            self.download_error = str(ytdlp_error)  # why, for the downloader (tools/download_videos.py)
            return None

    def get_video_fps(self, video_file_path):
        """The FPS of a video file (OpenCV), rounded to a whole number (a float); None when it cannot be read."""
        try:
            video = cv2.VideoCapture(video_file_path)
            fps = video.get(cv2.CAP_PROP_FPS) if video.isOpened() else 0.0  # OpenCV gives 0.0 rather than raise
            video.release()
        except Exception as e:
            logger.error(f"Failed to retrieve FPS of {video_file_path}: {e}")
            return None
        if not fps > 0:  # also NaN
            logger.error(f"Failed to retrieve FPS of {video_file_path}.")
            return None
        return round(fps, 0)

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
        # Load the video and create a subclip using the provided start and end times, at most to a second before the
        # file's end: YouTube rounds a length to whole seconds, so a mapping's end can run past the downloaded file,
        # and MoviePy fails reading past it (Bsyh0B6MCt8: 4325 s for a 4321.3 s file failed every round, 2026-10-22)
        video_clip = VideoFileClip(input_path)
        video_clip = video_clip.subclip(start_time, min(end_time, max(video_clip.duration - 1, start_time + 1)))

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()

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

    @staticmethod
    def _frame_rows(boxes, frame_width, frame_height):
        """The detections of one frame as [yolo-id, x-center, y-center, width, height, unique-id, confidence] rows:
        the box normalised, it and the confidence rounded to 6 decimals as in YOLO's label files."""
        cls = boxes.cls.int().cpu().tolist() if getattr(boxes, "cls", None) is not None else []
        ids = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None else [-1] * len(cls)
        confs = boxes.conf.float().cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(cls)
        if getattr(boxes, "xywhn", None) is not None:
            xywhn = boxes.xywhn.cpu().tolist()
        else:
            xywh = boxes.xywh.cpu().tolist() if getattr(boxes, "xywh", None) is not None else []
            xywhn = [[x / frame_width, y / frame_height, w / frame_width, h / frame_height] for x, y, w, h in xywh]
        return [[int(c), *(float(f"{v:.6f}") for v in box), int(t), float(f"{conf:.6f}")]
                for c, box, t, conf in zip(cls, xywhn, ids, confs)]

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

    # the trimmed segments main.py writes (yt_VID_ST_ET_FPS_mod_HEX.mp4), and VID_mod.mp4 of older code
    MOD_VIDEO = re.compile(r"^(?:yt_[A-Za-z0-9_-]{11}_\d+_\d+_\d+_mod_[0-9a-f]{32}|[A-Za-z0-9_-]{11}_mod)\.mp4$")

    def delete_youtube_mod_videos(self, folders, min_age_s=3600):
        """
        Delete the trimmed segment videos a stopped run left in the given folders: only those not written to for
        min_age_s, as another process sharing the folders (Snellius tasks) may be writing one.
        """
        for folder in folders:
            if not os.path.exists(folder):
                logger.info(f"Skipping missing folder: {folder}")
                continue
            for filename in os.listdir(folder):
                if not self.MOD_VIDEO.match(filename):
                    continue
                file_path = os.path.join(folder, filename)
                try:
                    if time.time() - os.path.getmtime(file_path) < min_age_s:
                        continue
                    os.remove(file_path)
                    logger.info(f"Deleted: {file_path}")
                except OSError as e:
                    logger.info(f"Failed to delete {file_path}: {e}")

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

    @staticmethod
    def _write_mapping(df, path):
        """Write the mapping DataFrame to path whole (a temporary file swapped in): a reader never sees half."""
        df.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)

    def update_population_in_csv(self, data):
        """
        Set the population_country column (thousands, World Bank's latest) of the mapping DataFrame data, and of
        the mapping file: read again and rewritten under the mapping lock, so edits made meanwhile stay.
        """
        if "iso3" not in data.columns:
            raise KeyError("The CSV file does not have a 'iso3' column.")
        latest_population = self.get_latest_population()
        population_dict = dict(zip(latest_population['iso3'], latest_population['Population']))
        data["population_country"] = data["iso3"].map(population_dict)
        mapping_path = common.get_configs("mapping")
        with _mapping_lock(mapping_path):
            fresh = pd.read_csv(mapping_path)
            fresh["population_country"] = fresh["iso3"].map(population_dict)
            self._write_mapping(fresh, mapping_path)
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
        Fill the gini column of the mapping DataFrame df (kept as self.mapping) with World Bank's latest values, and
        of the mapping file: read again and rewritten under the mapping lock, so edits made meanwhile stay.
        """
        try:
            if 'gini' not in df.columns:
                logger.error("The required columns 'iso3' and/or 'gini' are missing from the file.")
                return
            geni_df = self.get_latest_gini_values()

            def with_gini(frame):
                merged = pd.merge(frame, geni_df, on='iso3', how='left', suffixes=('', '_new'))
                merged['gini'] = merged['gini_new'].combine_first(merged['gini'])
                return merged.drop(columns=['gini_new'])

            self.mapping = with_gini(df)
            mapping_path = common.get_configs("mapping")
            with _mapping_lock(mapping_path):
                self._write_mapping(with_gini(pd.read_csv(mapping_path)), mapping_path)
            logger.info("Mapping file updated successfully with GINI value.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def update_track_buffer_in_yaml(self, yaml_path, video_fps):
        """Set track_buffer of the tracker YAML file to the config's track_buffer_sec in frames of video_fps."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        config['track_buffer'] = common.get_configs("track_buffer_sec") * video_fps
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def tracking_mode(self, input_video_path, output_video_path, video_title,
                      video_fps, bbox_mode, flag=0, run_root='runs'):
        """
        Track objects in a video with YOLO (bbox_mode) and write the detections to run_root/detect/{video_title of
        this helper}.csv (BBOX_COLUMNS; no file when nothing was found). flag: also make the annotated video
        {video_title}_mod_bbox.mp4 in output_video_path from the tracked frames (save_tracked_img).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Multi-thread safety: all YOLO side-effects go to run_root (one per task), and the tracker YAML is a copy
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
                if not os.path.exists(bbox_tracker_path):
                    bbox_tracker_path = src

        if bbox_mode:
            bbox_model = YOLO(self.tracking_model)

        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        # Output paths
        bbox_annotated_frame_output_path = os.path.join(detect_root, "annotated_frames")
        bbox_tracked_frame_output_path = os.path.join(detect_root, "tracked_frame")
        bbox_txt_output_path = os.path.join(detect_root, "labels")
        bbox_display_video_output_path = os.path.join(detect_root, "display_video.mp4")
        bbox_output_csv_path = os.path.join(detect_root, f"{self.video_title}.csv")
        if bbox_mode:
            os.makedirs(bbox_annotated_frame_output_path, exist_ok=True)
            os.makedirs(bbox_tracked_frame_output_path, exist_ok=True)
            if not self.delete_labels:
                os.makedirs(bbox_txt_output_path, exist_ok=True)
        # plotting the boxes and tracks is CPU-expensive: only when a frame is saved or shown
        draw = self.save_annoted_img or self.save_tracked_img or self.display_frame_tracking

        bbox_video_writer = None
        if bbox_mode and self.display_frame_tracking:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            bbox_video_writer = cv2.VideoWriter(
                bbox_display_video_output_path, fourcc, video_fps, (frame_width, frame_height)  # type: ignore
            )

        # Progress bar
        if total_frames == 0:
            logger.warning("Warning: Could not determine total frames. Progress bar may not work correctly.")
            total_frames = None

        progress_bar = tqdm(total=total_frames, unit="frames", dynamic_ncols=True)
        frame_count = 0
        bbox_track_history = defaultdict(lambda: [])
        csv_file, csv_writer = None, None  # the CSV is opened at the first detection

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                has_boxes = False  # bbox mode found objects (and did not fail)
                boxes = None
                bbox_annotated_frame = frame

                if bbox_mode:
                    try:
                        bbox_results = bbox_model.track(  # type: ignore
                            frame,
                            tracker=bbox_tracker_path,
                            persist=True,
                            conf=0.0,
                            save=False,
                            save_txt=False,
                            project=detect_root,
                            name="track",
                            exist_ok=True,
                            line_width=LINE_THICKNESS,
                            show_labels=False,
                            show_conf=False,
                            show=False,
                            verbose=False,
                            device=device,
                        )
                        boxes = bbox_results[0].boxes
                        if boxes is not None and boxes.xywh.size(0) > 0:  # type: ignore
                            if draw:
                                bbox_annotated_frame = bbox_results[0].plot()
                            has_boxes = True
                        else:
                            logger.info(f"[Frame {frame_count}] BBox: No objects found. Using original frame.")
                    except Exception as e:
                        logger.error(f"[Frame {frame_count}] BBox failed: {e}. Using original frame.")

                progress_bar.update(1)

                if self.save_annoted_img and bbox_mode:
                    cv2.imwrite(os.path.join(bbox_annotated_frame_output_path, f"frame_{frame_count}.jpg"),
                                bbox_annotated_frame)

                if has_boxes:
                    try:
                        rows = self._frame_rows(boxes, frame_width, frame_height)
                    except Exception as e:
                        logger.warning(f"[Frame {frame_count}] BBox label write failed: {e}")
                        rows = []
                    if rows:
                        if csv_writer is None:
                            new = not os.path.exists(bbox_output_csv_path)
                            csv_file = open(bbox_output_csv_path, "a", newline="")
                            csv_writer = csv.writer(csv_file, lineterminator=os.linesep)  # as pandas' to_csv
                            if new:
                                csv_writer.writerow(BBOX_COLUMNS)
                        csv_writer.writerows(row + [frame_count] for row in rows)
                    if not self.delete_labels:  # YOLO's label file of the frame, as save_txt writes it
                        with open(os.path.join(bbox_txt_output_path, f"label_{frame_count}.txt"), "w") as f:
                            f.writelines(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {t} {conf:.6f}\n"
                                         for c, x, y, w, h, t, conf in rows)

                    if draw:  # the tracks of the last 30 frames
                        try:
                            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []  # type: ignore
                            for box, track_id in zip(boxes.xywh.cpu(), ids):  # type: ignore
                                x, y, w, h = box
                                track = bbox_track_history[track_id]
                                track.append((float(x), float(y)))
                                if len(track) > 30:
                                    track.pop(0)
                                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                cv2.polylines(bbox_annotated_frame, [points], isClosed=False,
                                              color=(230, 230, 230), thickness=LINE_THICKNESS * 5)
                        except Exception as e:
                            logger.warning(f"[Frame {frame_count}] Drawing the tracks failed: {e}")

                if self.display_frame_tracking and bbox_mode:
                    cv2.imshow("YOLOv11 Tracking", bbox_annotated_frame)
                    bbox_video_writer.write(bbox_annotated_frame)  # type: ignore

                if self.save_tracked_img and bbox_mode:
                    cv2.imwrite(os.path.join(bbox_tracked_frame_output_path, f"frame_tracked_{frame_count}.jpg"),
                                bbox_annotated_frame)

                # Break the loop if 'q' is pressed, only when display is enabled
                if self.display_frame_tracking:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if csv_file is not None:
                csv_file.close()
            if bbox_video_writer is not None:
                bbox_video_writer.release()
            if self.display_frame_tracking:
                cv2.destroyAllWindows()
            progress_bar.close()

        if flag and bbox_mode:
            self.create_video_from_images(
                image_folder=bbox_tracked_frame_output_path,
                output_path=output_video_path,
                video_title=video_title,
                bbox_mode=bbox_mode,
                frame_rate=video_fps,
            )
