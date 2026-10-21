# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
"""Refresh mapping_metadata.csv: fetch title, upload date, channel, views, description and chapters (pytubefix) of
every mapping video not in it yet, and again of those with a missing title, upload date, channel or views. Rows of
videos no longer in the mapping are dropped (the ids of the other rows stay)."""

import pandas as pd
from pytubefix import YouTube
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import socket
import sys
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from release import split_large_files  # noqa: E402
from utils.core.parsing import parse_videos  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

metadata_file = "mapping_metadata.csv"
csv_headers = [
    "id", "video", "title", "upload_date", "channel", "views", "description",
    "chapters", "segments", "date_updated"
]
SAVE_EVERY = 200  # videos between saves of the file, so an interrupted manual run keeps most of its work
socket.setdefaulttimeout(60)  # a stalled YouTube connection must not hang the weekly job forever


def clean_text(val):
    """A cell as one line of text: a list as its repr, NaN as ''."""
    if isinstance(val, list):
        return str(val)
    if pd.isna(val):
        return ""
    return str(val).replace("\n", " ").replace("\r", " ").strip()


def is_missing(value):
    """Return True if value is None, NaN, empty, or 'nan'."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip().lower()
    return s == "" or s == "nan"


def to_int_safe(x):
    """x as an int ('21092024.0' -> 21092024); 0 for missing or unparsable values."""
    try:
        if is_missing(x):
            return 0
        s = str(x).strip()
        return int(float(s)) if "." in s else int(s)
    except (TypeError, ValueError, OverflowError):
        return 0


def get_video_info(video_id):
    """The metadata fields of one video; empty strings for all of them when it cannot be fetched."""
    try:
        # OAuth asks for a sign-in in the terminal, which a scheduled run cannot answer
        yt = YouTube(
            f"https://www.youtube.com/watch?v={video_id}",
            use_oauth=sys.stdin.isatty(),
            allow_oauth_cache=True,
        )
        title = yt.title
        description = yt.description
        channel = yt.channel_id
        views = yt.views
        upload_date = yt.publish_date.strftime("%d%m%Y") if yt.publish_date else ""

        chapters = []
        for c in yt.chapters or []:
            try:
                chapters.append({
                    "title": c.title,
                    "timestamp": str(timedelta(seconds=c.start_seconds))
                })
            except Exception as e:
                logger.error(f"⚠️ failed to extract chapter for {video_id}: {e}")

        logger.info(
            "✅ fetched: {} | title: {} | upload: {} | channel: {} | views: {}",
            video_id, title, upload_date, channel, views
        )

        return {
            "video": video_id,
            "title": clean_text(title),
            "upload_date": upload_date,
            "channel": clean_text(channel),
            "views": views,
            "description": clean_text(description),
            "chapters": clean_text(chapters),
        }

    except Exception as e:  # unavailable, private, needs a login, network error: retried next run
        logger.info("❌ failed to fetch {}: {}", video_id, e)
        return {
            "video": video_id,
            "title": "",
            "upload_date": "",
            "channel": "",
            "views": "",
            "description": "",
            "chapters": "",
        }


def save(df):
    """Write df to the metadata file atomically (the weekly job pushes it); views as integers."""
    df["views"] = df["views"].apply(to_int_safe)
    tmp = f"{metadata_file}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, metadata_file)


# -----------------------------------------------------
# main
# -----------------------------------------------------

if __name__ == "__main__":

    df_map = pd.read_csv(common.get_configs("mapping"))

    # gather all video IDs found in mapping file
    all_video_ids = [vid for row in df_map["videos"] for vid in parse_videos(row)]
    video_count = pd.Series(all_video_ids).value_counts().to_dict()
    unique_video_ids = list(video_count.keys())

    # load metadata file
    split_large_files.join(metadata_file)
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file, dtype=str)
        existing_df["id"] = existing_df["id"].astype(int)  # id must stay numeric
        last_id = existing_df["id"].max() if not existing_df.empty else 0  # before dropping: ids are never reused
        # videos no longer in the mapping (removed, or merged away) leave the file
        gone = ~existing_df["video"].astype(str).isin(video_count)
        if gone.any():
            logger.info(f"🗑️ dropping {int(gone.sum())} videos no longer in the mapping")
            existing_df = existing_df[~gone].reset_index(drop=True)
        existing_ids = set(existing_df["video"].astype(str).tolist())

        logger.info(f"🗃️ found {len(existing_ids)} videos in existing metadata")

        # detect missing data
        critical_cols = ["title", "upload_date", "channel", "views"]
        failed_mask = existing_df[critical_cols].map(is_missing).any(axis=1)
        failed_ids = set(existing_df.loc[failed_mask, "video"].astype(str).tolist())

        logger.info(f"🔁 found {len(failed_ids)} videos with missing metadata")

    else:
        # no file → create an empty one
        with open(metadata_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
            w.writerow(csv_headers)

        existing_df = pd.DataFrame(columns=csv_headers)
        gone = pd.Series(dtype=bool)
        existing_ids = set()
        failed_ids = set()
        last_id = 0

        logger.info("📁 created new metadata file")

    # determine which videos to fetch
    missing_in_metadata = [vid for vid in unique_video_ids if vid not in existing_ids]
    retry_videos = list(failed_ids)
    videos_to_fetch = missing_in_metadata + retry_videos

    logger.info(
        f"🔍 total videos to fetch: {len(videos_to_fetch)} "
        f"({len(missing_in_metadata)} missing, {len(retry_videos)} need update)"
    )

    now = datetime.now().strftime("%d%m%Y")
    current_id = last_id

    # ensure correct dtypes
    existing_df["video"] = existing_df["video"].astype(str)

    # main loop
    for n, vid in enumerate(tqdm(videos_to_fetch), start=1):
        vid = str(vid)
        info = get_video_info(vid)

        info["segments"] = video_count.get(vid, 1)
        info["date_updated"] = now

        # update or insert row
        mask = existing_df["video"] == vid

        if mask.any():
            info["id"] = existing_df.loc[mask, "id"].iloc[0]
            for col, val in info.items():
                existing_df.loc[mask, col] = val
        else:
            current_id += 1
            info["id"] = current_id
            existing_df = pd.concat([existing_df, pd.DataFrame([info])], ignore_index=True)

        if n % SAVE_EVERY == 0 or n == len(videos_to_fetch):
            save(existing_df)
            logger.info(f"💾 saved metadata after {n} of {len(videos_to_fetch)} videos")

    if gone.any() and not videos_to_fetch:  # only rows dropped: save them too
        save(existing_df)
        logger.info("💾 saved metadata without the dropped videos")
    logger.info("✅ all videos processed and metadata file updated.")
