"""Fill in the missing upload dates of the mapping's videos from YouTube (pytubefix), written to mapping_updated.csv
(the mapping itself is not changed). Dates are DDMMYYYY; a row whose upload_date list does not match its videos gets
every date fetched again.

    python -m tools.update_upload_date
"""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
from pytubefix import YouTube
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402
from utils.core.parsing import parse_list  # noqa: E402

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def get_upload_date(video_id):
    """Upload date of a video as DDMMYYYY, None when YouTube gives none."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        yt_date = YouTube(url).publish_date
    except Exception as e:  # pytubefix raises many kinds (unavailable, bot check, network)
        logger.error(f"Error fetching date for {video_id}: {e}")
        return None
    if yt_date is None:  # logging its strftime first failed here, so these were reported as errors
        logger.error(f"No date for {url}")
        return None
    logger.info(f"Received date for {url} as {yt_date.strftime('%d%m%Y')}")
    return yt_date.strftime('%d%m%Y')


if __name__ == "__main__":
    # as text: the other cells are written back as they are
    df = pd.read_csv(common.get_configs("mapping"), dtype=str, keep_default_na=False)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_ids = parse_list(row['videos'])
        upload_dates = parse_list(row['upload_date'])
        if len(upload_dates) != len(video_ids):
            upload_dates = [None] * len(video_ids)
        updated_dates = [get_upload_date(vid) if date in [None, 'None', '', 'nan'] else date
                         for vid, date in zip(video_ids, upload_dates)]
        df.at[idx, 'upload_date'] = str(updated_dates)  # type: ignore

    updated_csv_file = 'mapping_updated.csv'
    df.to_csv(updated_csv_file, index=False)
    logger.info(f"Updated data saved to {updated_csv_file}")
