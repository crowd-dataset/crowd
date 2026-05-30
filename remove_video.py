import csv
import re
import ast  # Still useful for parsing numeric or nested lists like time_of_day


def remove_video_data(input_file, output_file, video_ids_to_remove):
    """
    Removes specified video IDs and their associated data from a CSV mapping file.

    Args:
        input_file (str): Path to the input CSV mapping file.
        output_file (str): Path to save the modified CSV file.
        video_ids_to_remove (list): A list of video IDs (strings) to remove.
    """

    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        # Get the indices of the columns to process
        try:
            video_col_idx = header.index('videos')
            time_of_day_col_idx = header.index('time_of_day')
            start_time_col_idx = header.index('start_time')
            end_time_col_idx = header.index('end_time')
            vehicle_type_col_idx = header.index('vehicle_type')
            upload_date_col_idx = header.index('upload_date')
            channel_col_idx = header.index('channel')
        except ValueError as e:
            print(f"Error: Missing expected column in header - {e}")
            return

        for row in reader:
            original_row = list(row)  # Make a copy to modify

            # --- Parsing input data ---
            # Parse 'videos' using regex as it has unquoted elements
            videos_str = row[video_col_idx]
            videos = re.findall(r'[\w-]+', videos_str)

            # For other columns, attempt ast.literal_eval first.
            # If it fails, fall back to stripping brackets and splitting by comma.
            # This handles cases where items might be numbers, strings, or nested lists.

            # Helper function to parse list-like strings robustly
            def parse_list_string(s, is_string_list=False):
                s = s.strip()
                if not s or s == '[]':
                    return []
                # Attempt ast.literal_eval first for correctly formatted lists (e.g., nested lists, numbers)
                try:
                    return ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    # Fallback for unquoted items, like your 'videos' or 'upload_date'
                    # Remove outer brackets and split by comma
                    inner_content = s.strip('[]').strip()
                    if not inner_content:
                        return []
                    # Split by comma, ensuring we handle potential inner list structures.
                    # This simple split works well for flat lists of numbers or unquoted strings.
                    # For nested lists, ast.literal_eval is preferred if the format is strict Python.
                    if is_string_list:
                        return [item.strip() for item in inner_content.split(',') if item.strip()]
                    else:
                        # For numeric or potentially nested lists where ast.literal_eval failed,
                        # we try to parse each segment. This is more complex if true nested lists
                        # are malformed like the main 'videos' column. Given the example,
                        # this simple split and literal_eval for items should be sufficient.
                        # For lists like `[[0],[1],[0,1]]`, ast.literal_eval is needed.
                        # If a column like `time_of_day` is `[[0],[1,1]]` and it's failing,
                        # that's because the outer `[` and `]` are there but the inner lists
                        # might not be correctly quoted if they contained strings.
                        # However, for the provided data, `[[0],[1]]` is valid for ast.literal_eval.
                        # The `SyntaxError` was mainly due to `[XzwI2rGmf04, ...]`

                        # Let's revert to a more robust approach for time_of_day, start_time, etc.
                        # If the primary ast.literal_eval fails for these, it implies they are
                        # truly malformed or structured unexpectedly.
                        # The regex approach for single strings works for videos.
                        # For structured data like [[0],[1]], ast.literal_eval is the correct tool.
                        # The previous try-except with stripping was for *simple* lists of unquoted numbers.

                        return ast.literal_eval(s)  # Re-attempt literal_eval with the original string if the first parse_list_string call implies this path.  # noqa: E501

            # Applying parsing with the understanding that only 'videos', 'upload_date', 'channel'
            # might have unquoted string elements needing `re.findall` or custom string parsing.
            # Other columns (time_of_day, start_time, etc.) as per your example are valid Python lists for ast.literal_eval.  # noqa: E501

            time_of_day = ast.literal_eval(row[time_of_day_col_idx])
            start_time = ast.literal_eval(row[start_time_col_idx])
            end_time = ast.literal_eval(row[end_time_col_idx])
            vehicle_type = ast.literal_eval(row[vehicle_type_col_idx])

            # For upload_date and channel, they also contain unquoted strings.
            # We can use re.findall for a similar effect, or a more direct string manipulation.
            # Let's use re.findall for consistency with videos for these string lists.
            upload_date = re.findall(r'[\w-]+', row[upload_date_col_idx])
            channel = re.findall(r'[\w-]+', row[channel_col_idx])

            # Create lists to hold data that will be kept
            new_videos = []
            new_time_of_day = []
            new_start_time = []
            new_end_time = []
            new_vehicle_type = []
            new_upload_date = []
            new_channel = []

            for i, video_id in enumerate(videos):
                if video_id not in video_ids_to_remove:
                    new_videos.append(video_id)
                    # Check if the index exists before appending to avoid IndexError
                    if i < len(time_of_day):
                        new_time_of_day.append(time_of_day[i])
                    if i < len(start_time):
                        new_start_time.append(start_time[i])
                    if i < len(end_time):
                        new_end_time.append(end_time[i])
                    if i < len(vehicle_type):
                        new_vehicle_type.append(vehicle_type[i])
                    if i < len(upload_date):
                        new_upload_date.append(upload_date[i])
                    if i < len(channel):
                        new_channel.append(channel[i])

            # --- Formatting output data to match original style (no spaces after commas, no quotes for strings) ---
            def format_list_output(lst):
                if not lst:
                    return "[]"
                # If the list contains nested lists or numbers, convert each item to string
                # and join. Otherwise, join as is.
                # This handles cases like [[0],[1]] vs [30,30] vs [ID1,ID2]
                if any(isinstance(item, (list, int, float)) for item in lst):
                    # Special handling for nested lists like time_of_day and start_time
                    # where str(item) will give the correct representation like '[0]' or '[119, 5386]'
                    return '[' + ','.join(str(item).replace(' ', '') for item in lst) + ']'
                else:
                    # For lists of simple strings (like video IDs, upload dates, channel IDs)
                    # just join them directly without extra quotes or spaces.
                    return '[' + ','.join(str(item) for item in lst) + ']'

            # Update the row with the new formatted lists
            original_row[video_col_idx] = format_list_output(new_videos)
            original_row[time_of_day_col_idx] = format_list_output(new_time_of_day)
            original_row[start_time_col_idx] = format_list_output(new_start_time)
            original_row[end_time_col_idx] = format_list_output(new_end_time)
            original_row[vehicle_type_col_idx] = format_list_output(new_vehicle_type)
            original_row[upload_date_col_idx] = format_list_output(new_upload_date)
            original_row[channel_col_idx] = format_list_output(new_channel)

            writer.writerow(original_row)


# --- Configuration ---
input_csv_file = 'mapping.csv'
output_csv_file = 'mapping_cleaned.csv'

# List of video IDs to remove
videos_to_delete = [
    'UbnFsbQW-C4'
]

# Run the script
remove_video_data(input_csv_file, output_csv_file, videos_to_delete)

print(f"Processing complete. Modified data saved to '{output_csv_file}'")
