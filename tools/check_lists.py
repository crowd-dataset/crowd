"""Check that the list cells of each mapping row line up: as many channels, upload dates, vehicle types and
time_of_day/start_time/end_time entries as videos; logs each row where they do not.

    python -m tools.check_lists
"""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402
from utils.core.parsing import parse_list, parse_videos  # noqa: E402

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

filename = "mapping.csv"
# columns with one entry per video (nested lists: one inner list per video)
list_cols = ['channel', 'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date']

if __name__ == "__main__":
    with open(filename, newline='', encoding='utf-8') as f:
        for row_num, row in enumerate(csv.DictReader(f), start=1):
            len_v = len(parse_videos(row['videos']))
            for col in list_cols:
                n = len(parse_list(row[col]))
                if n != len_v:
                    logger.info(f'mismatch in {col} at row {row_num}: {n} entries vs {len_v} videos')
