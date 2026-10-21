"""Merge the contributors' mapping-NAME.csv files into mapping.csv (backed up to mapping_bkp.csv first): each video
goes to the row of the same locality, country, iso3 (and state, when given), replacing the values the mapping has
for a video it has already; a locality the mapping lacks becomes a new row. Ids are renumbered.

    python -m tools.merge_mapping

The review hub's merge (review_hub.merge_mapping_files) does this job now, keeping the mapping's values instead.
"""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import glob
import os
import shutil
import sys

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
from tools.mapping_io import locked, replace  # noqa: E402
from utils.core.parsing import parse_list  # noqa: E402

BASE_FILE, BACKUP_FILE = 'mapping.csv', 'mapping_bkp.csv'
LIST_COLUMNS = ['videos', 'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date', 'channel']
NESTED_LIST_COLUMNS = ['time_of_day', 'start_time', 'end_time']


def parse_nested_list(cell):
    """A nested list cell as a list of lists; also '["[0]", "[1]"]' (inner lists written as strings) -> [[0], [1]]."""
    return [parse_list(i) if isinstance(i, str) else i for i in parse_list(cell)]


def normalise(value):
    """value for matching localities: stripped, lower case."""
    return str(value).strip().lower()


def format_list(lst):
    """[a,b] as the mapping writes a flat list."""
    return f"[{','.join(str(x) for x in lst)}]"


def format_nested_list(lst):
    """[[0,10],[5]] as the mapping writes a nested list."""
    return format_list(format_list(sublist) for sublist in lst)


def read_lists(path):
    """The mapping file at path with its list columns parsed (every other cell kept as its text)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in LIST_COLUMNS:
        df[col] = df[col].apply(parse_nested_list if col in NESTED_LIST_COLUMNS else parse_list)
    return df


def merge():
    """Merge every mapping-NAME.csv into the mapping and write it."""
    shutil.copyfile(BASE_FILE, BACKUP_FILE)
    print(f"Backed up {BASE_FILE} -> {BACKUP_FILE}")
    main_df = read_lists(BASE_FILE)
    for file in glob.glob('mapping-*.csv'):
        print(f"Working with {file}")
        for _, new_row in read_lists(file).iterrows():
            match = main_df[
                (main_df['locality'].apply(normalise) == normalise(new_row['locality'])) &
                (main_df['country'].apply(normalise) == normalise(new_row['country'])) &
                (main_df['iso3'].apply(normalise) == normalise(new_row['iso3']))
            ]
            state = normalise(new_row['state'])
            if state:
                match = match[match['state'].apply(normalise) == state]
            if match.empty:
                main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)
                continue
            idx = match.index[0]
            existing_videos = main_df.at[idx, 'videos']
            for vid_idx, vid in enumerate(new_row['videos']):
                if vid in existing_videos:
                    existing_idx = existing_videos.index(vid)
                    for col in LIST_COLUMNS:
                        main_df.at[idx, col][existing_idx] = new_row[col][vid_idx]
                else:
                    for col in LIST_COLUMNS:
                        main_df.at[idx, col].append(new_row[col][vid_idx])

    main_df.reset_index(drop=True, inplace=True)
    main_df['id'] = main_df.index + 1
    for col in LIST_COLUMNS:
        main_df[col] = main_df[col].apply(format_nested_list if col in NESTED_LIST_COLUMNS else format_list)
    replace(BASE_FILE, lambda tmp: main_df.to_csv(tmp, index=False))
    print(f"Updated {BASE_FILE} with new videos from mapping-*.csv files.")


if __name__ == '__main__':
    with locked(BASE_FILE):
        merge()
