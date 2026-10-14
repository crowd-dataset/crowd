# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import pandas as pd
import common

# get a list of already downloaded video files (assuming mp4 format)
downloaded_videos = set()
for folder in common.get_configs('videos'):
    if os.path.exists(folder):
        downloaded_videos.update(f.split('.')[0] for f in os.listdir(folder) if f.endswith('.mp4'))

# load mapping
df = pd.read_csv(common.get_configs('mapping'))


# function to check if a video id is already downloaded
def filter_videos(video_list):
    if not isinstance(video_list, str) or not video_list.strip():
        return []
    try:
        # remove surrounding quotes and brackets
        video_list = video_list.strip().strip('"[]')
        # split by comma and remove spaces
        video_ids = [vid.strip() for vid in video_list.split(',')]
        return [vid for vid in video_ids if vid and vid not in downloaded_videos]
    except Exception:
        return []  # return empty list if parsing fails


# filter rows where there are videos that haven't been downloaded
df['filtered_videos'] = df['videos'].apply(filter_videos)
df_filtered = df[df['filtered_videos'].map(len) > 0]

df_filtered = df_filtered[['id',
                           'locality',
                           'locality_aka',
                           'state',
                           'country',
                           'iso3',
                           'filtered_videos',
                           'start_time',
                           'end_time',
                           'vehicle_type',
                           'upload_date',
                           'channel',
                           'time_of_day']].rename(columns={'filtered_videos': 'videos'})
# save the filtered subset with correct list format
df_filtered['videos'] = df_filtered['videos'].apply(lambda x: f"[{','.join(x)}]" if isinstance(x, list) else "[]")

queue_file = 'mapping_queue.csv'
df_filtered.to_csv(queue_file, index=False)

# count total number of videos added
total_videos = df_filtered['videos'].apply(lambda x: len(x.strip('[]').split(',')) if x != "[]" else 0).sum()
print(f'{total_videos} videos added to {queue_file}.')
