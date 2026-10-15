# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import csv
import ast
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# file path
filename = "mapping.csv"

# columns to check
videos_col = 'videos'
channels_col = 'channel'

nested_list_cols = ['time_of_day', 'start_time', 'end_time', 'vehicle_type']


def parse_simple_list(raw):
    raw = raw.strip()[1:-1]
    if not raw:
        return []
    return [item.strip() for item in raw.split(',')]


def parse_nested_list(raw):
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, list):
            return val
        return []
    except Exception:
        return []


if __name__ == "__main__":
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            try:
                videos = parse_simple_list(row[videos_col])
                channels = parse_simple_list(row[channels_col])
            except Exception as e:
                logger.error(f'error parsing row {row_num}: {e}')
                continue

            len_v = len(videos)
            len_c = len(channels)

            if len_v != len_c:
                logger.info(f'mismatch at row {row_num}: {len_v} videos vs {len_c} channels')

            # check each nested list column for consistent length with `videos`
            for col in nested_list_cols:
                nested = parse_nested_list(row[col])
                len_nested = len(nested)
                if len_nested != len_v:
                    logger.info(f'mismatch in {col} at row {row_num}: {len_nested} entries vs {len_v} videos')
