# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import ast
from pytubefix import YouTube
from tqdm import tqdm
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def safe_parse(val):
    if pd.isna(val) or not isinstance(val, str):
        return []
    val = val.strip()

    # Case 1: properly formatted Python list (e.g., "['a','b']")
    try:
        return ast.literal_eval(val)
    except Exception:
        pass

    # Case 2: malformed list like [abc,def] — parse with regex
    if val.startswith('[') and val.endswith(']'):
        inner = val[1:-1]
        return [item.strip().strip("'\"") for item in inner.split(',') if item.strip()]

    return []


def get_upload_date(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        yt_date = yt.publish_date
        logger.info(f"Received date for https://www.youtube.com/watch?v={video_id} as {yt_date.strftime('%d%m%Y')}")  # type: ignore # noqa:E501
        return yt_date.strftime('%d%m%Y') if yt_date else None
    except Exception as e:
        logger.error(f"Error fetching date for {video_id}: {e}")
        return None


if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv(common.get_configs("mapping"))

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_ids = safe_parse(row['videos'])
        upload_dates = safe_parse(row['upload_date'])

        if len(upload_dates) != len(video_ids):
            upload_dates = [None] * len(video_ids)  # fallback

        updated_dates = []
        for vid, date in zip(video_ids, upload_dates):
            if date in [None, 'None', '', 'nan']:
                correct_date = get_upload_date(vid)
                updated_dates.append(correct_date if correct_date else None)
            else:
                updated_dates.append(date)

        df.at[idx, 'upload_date'] = str(updated_dates)  # type: ignore

    # Save updated CSV
    updated_csv_file = 'mapping_updated.csv'
    df.to_csv(updated_csv_file, index=False)
    logger.info(f"Updated data saved to {updated_csv_file}")
