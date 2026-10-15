# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import glob
import shutil
import ast


def parse_nested_list(cell):
    """Converts a string like '["[0]", "[1]"]' into [[0], [1]]"""
    if pd.isna(cell):
        return []
    try:
        parsed = ast.literal_eval(cell)
        if all(isinstance(i, str) and i.startswith('[') for i in parsed):
            return [ast.literal_eval(i) for i in parsed]
        return parsed
    except Exception:
        return []


def parse_flat_list(cell):
    """Parses '["val1", "val2"]' or '[val1,val2]' into ['val1', 'val2']"""
    if pd.isna(cell):
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [item.strip(" '\"") for item in str(cell).strip("[]").split(',') if item.strip()]


def normalise(value):
    if pd.isna(value):
        return ''
    return str(value).strip().lower()


# filenames
base_file = 'mapping.csv'
backup_file = 'mapping_bkp.csv'

# create backup
shutil.copyfile(base_file, backup_file)
print(f"Backed up {base_file} -> {backup_file}")

# list-type columns
list_columns = ['videos', 'time_of_day', 'start_time', 'end_time', 'vehicle_type', 'upload_date',
                'channel']
nested_list_columns = ['time_of_day', 'start_time', 'end_time']

# load main mapping.csv
main_df = pd.read_csv(base_file)
for col in list_columns:
    if col in nested_list_columns:
        main_df[col] = main_df[col].apply(parse_nested_list)
    else:
        main_df[col] = main_df[col].apply(parse_flat_list)

# process each mapping-NAME.csv
for file in glob.glob('mapping-*.csv'):
    if file == base_file or file == backup_file:
        continue
    print(f"Working with {file}")
    new_df = pd.read_csv(file)
    for col in list_columns:
        if col in nested_list_columns:
            new_df[col] = new_df[col].apply(parse_nested_list)
        else:
            new_df[col] = new_df[col].apply(parse_flat_list)

    for _, new_row in new_df.iterrows():
        locality = normalise(new_row['locality'])
        country = normalise(new_row['country'])
        iso3 = normalise(new_row['iso3'])
        state = normalise(new_row['state'])

        match = main_df[
            (main_df['locality'].apply(normalise) == locality) &
            (main_df['country'].apply(normalise) == country) &
            (main_df['iso3'].apply(normalise) == iso3)
        ]

        if state:
            match = match[match['state'].apply(normalise) == state]

        if not match.empty:
            idx = match.index[0]
            existing_videos = main_df.at[idx, 'videos']  # type: ignore

            for vid_idx, vid in enumerate(new_row['videos']):
                if vid in existing_videos:
                    existing_idx = existing_videos.index(vid)  # type: ignore
                    for col in list_columns:
                        main_df.at[idx, col][existing_idx] = new_row[col][vid_idx]  # type: ignore
                else:
                    for col in list_columns:
                        value = new_row[col][vid_idx]
                        main_df.at[idx, col].append(value)  # type: ignore
        else:
            main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)

# reindex
main_df.reset_index(drop=True, inplace=True)
main_df['id'] = main_df.index + 1


# convert lists to string format
def format_list(lst):
    return f"[{','.join(str(x) for x in lst)}]"


def format_nested_list(lst):
    inner = [f"[{','.join(str(i) for i in sublist)}]" for sublist in lst]
    return f"[{','.join(inner)}]"


for col in list_columns:
    if col in nested_list_columns:
        main_df[col] = main_df[col].apply(format_nested_list)
    else:
        main_df[col] = main_df[col].apply(format_list)

# write to mapping.csv
main_df.to_csv(base_file, index=False)
print(f"Updated {base_file} with new videos from mapping-*.csv files.")
