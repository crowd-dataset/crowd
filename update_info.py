# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import pandas as pd
import ast
from pytubefix import YouTube
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import csv
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

metadata_file = "mapping_metadata.csv"
csv_headers = [
    "id", "video", "title", "upload_date", "channel", "views", "description",
    "chapters", "segments", "date_updated"
]


# -----------------------------
# helpers
# -----------------------------

def safe_parse(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return []


def clean_text(val):
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


def get_video_info(video_id):
    try:
        yt = YouTube(
            f"https://www.youtube.com/watch?v={video_id}",
            use_oauth=True,
            allow_oauth_cache=True,
        )
        title = yt.title
        description = yt.description
        channel = yt.channel_id
        views = yt.views
        upload_date = yt.publish_date.strftime("%d%m%Y") if yt.publish_date else ""

        chapters = []
        if yt.chapters:
            for c in yt.chapters:
                try:
                    chapters.append({
                        "title": c.title,
                        "timestamp": str(timedelta(seconds=c.start_seconds))
                    })
                except Exception as e:
                    logger.error(f"⚠️ failed to extract chapter for {video_id}: {e}")

        safe_title = str(title).replace("{", "{{").replace("}", "}}")
        safe_channel = str(channel).replace("{", "{{").replace("}", "}}")

        logger.info(
            "✅ fetched: {} | title: {} | upload: {} | channel: {} | views: {}",
            video_id, safe_title, upload_date, safe_channel, views
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

    except Exception as e:
        logger.info(
            "❌ failed to fetch {}: {}",
            video_id, e
        )
        return {
            "video": video_id,
            "title": "",
            "upload_date": "",
            "channel": "",
            "views": "",
            "description": "",
            "chapters": "",
        }


# -----------------------------------------------------
# main
# -----------------------------------------------------

if __name__ == "__main__":

    df_map = pd.read_csv(common.get_configs("mapping"))

    # gather all video IDs found in mapping file
    all_video_ids = []
    for row in df_map["videos"]:
        vids = safe_parse(row)
        all_video_ids.extend(vids)

    video_count = pd.Series(all_video_ids).value_counts().to_dict()
    unique_video_ids = list(video_count.keys())

    # load metadata file
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file, dtype=str)
        existing_df["id"] = existing_df["id"].astype(int)  # id must stay numeric
        existing_ids = set(existing_df["video"].astype(str).tolist())
        last_id = existing_df["id"].max() if not existing_df.empty else 0

        logger.info(f"🗃️ found {len(existing_ids)} videos in existing metadata")

        # detect missing data
        critical_cols = ["title", "upload_date", "channel", "views"]
        failed_mask = existing_df[critical_cols].applymap(is_missing).any(axis=1)  # type: ignore
        failed_ids = set(existing_df.loc[failed_mask, "video"].astype(str).tolist())  # type: ignore

        logger.info(f"🔁 found {len(failed_ids)} videos with missing metadata")

    else:
        # no file → create an empty one
        with open(metadata_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
            w.writerow(csv_headers)

        existing_df = pd.DataFrame(columns=csv_headers)
        existing_ids = set()
        failed_ids = set()
        last_id = 0

        logger.info("📁 created new metadata file")

    # determine which videos to fetch
    missing_in_metadata = [vid for vid in unique_video_ids if vid not in existing_ids]
    retry_videos = [vid for vid in failed_ids]

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
    for vid in tqdm(videos_to_fetch):
        vid = str(vid)
        info = get_video_info(vid)

        info["segments"] = video_count.get(vid, 1)
        info["date_updated"] = now

        # update or insert row
        mask = existing_df["video"] == vid

        if mask.any():
            existing_id = existing_df.loc[mask, "id"].iloc[0]
            info["id"] = existing_id
            for col, val in info.items():
                existing_df.loc[mask, col] = val
        else:
            current_id += 1
            info["id"] = current_id
            existing_df = pd.concat([existing_df, pd.DataFrame([info])], ignore_index=True)

        # enforce integer types for selected columns
        int_columns = ["id", "upload_date", "date_updated", "views"]

        for col in int_columns:
            if col in existing_df.columns:
                def to_int_safe(x):
                    try:
                        if x is None:
                            return 0
                        s = str(x).strip()
                        if s == "" or s.lower() == "nan":
                            return 0
                        # handle floats stored as strings: "21092024.0"
                        if "." in s:
                            return int(float(s))
                        return int(s)
                    except Exception:
                        return 0

        existing_df[col] = existing_df[col].apply(to_int_safe)

        # write entire file safely
        existing_df.to_csv(metadata_file, index=False)

        logger.info(f"💾 updated metadata for {vid}")

    logger.info("✅ all videos processed and metadata file updated.")
