"""
Parsing of the mapping's list cells and the processed length of a segment, shared by the analysis and the figures.

The mapping stores lists as text: `videos` as bare IDs (`[abc,def]`), `start_time`, `end_time` and `time_of_day` as
nested numeric lists, one inner list per video (`[[0,100],[5]]`), and `vehicle_type`, `upload_date` and `channel` as
flat lists. The top-folder scripts (main.py, helper_script.py, add_video.py, ...) do not import this module: workers
get only those files on a sync, not utils/.
"""
import ast
import re

# The detector/tracker pipeline trims each segment to t_end - 1 second (main.py); durations in the analysis use the
# same processed endpoint.
ENDPOINT_ADJUSTMENT_SECONDS = 1

_VIDEO_ID = re.compile(r"[\w-]+")


def processed_segment_duration_seconds(start_time, end_time) -> int:
    """Processed length (s) of one segment: end - 1 s - start, or end - start where the shorter end would not be after
    the start (as main.py trims); 0 for values that are not numbers."""
    try:
        start, end = int(float(start_time)), int(float(end_time))
    except (TypeError, ValueError, OverflowError):  # also inf
        return 0
    processed_end = end - ENDPOINT_ADJUSTMENT_SECONDS
    if processed_end <= start:
        processed_end = end
    return max(0, processed_end - start)


def parse_list(cell) -> list:
    """A list cell as a Python list: numeric (possibly nested) lists via literal_eval, bare tokens (`[abc,def]`) split
    at commas; [] for empty, missing or unparsable cells. A list passes through unchanged."""
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        return []
    s = cell.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":  # a quoted cell
        s = s[1:-1].strip()
    if s in ("", "[]") or s.lower() in ("nan", "none", "null"):
        return []
    try:
        value = ast.literal_eval(s)
        return value if isinstance(value, list) else []
    except (ValueError, SyntaxError):
        if s.startswith("[") and s.endswith("]"):
            return [p.strip().strip("'\"") for p in s[1:-1].split(",") if p.strip().strip("'\"")]
        return []


def parse_nested(cell) -> list:
    """A nested list cell (one inner list per video) as a list of lists; a flat list is one inner list."""
    value = parse_list(cell)
    return [value] if value and not any(isinstance(v, list) for v in value) else value


def parse_videos(cell) -> list:
    """The video IDs of a `videos` cell, with or without quotes around the IDs."""
    return _VIDEO_ID.findall(cell) if isinstance(cell, str) else []


def total_processed_seconds(start_cell, end_cell) -> int:
    """Processed length (s) of all segments of one mapping row, from its start_time and end_time cells."""
    return sum(processed_segment_duration_seconds(s, e)
               for starts, ends in zip(parse_nested(start_cell), parse_nested(end_cell))
               for s, e in zip(starts, ends))


if __name__ == "__main__":
    assert processed_segment_duration_seconds(0, 61) == 60 and processed_segment_duration_seconds(5, 6) == 1
    assert processed_segment_duration_seconds("x", 3) == 0 and processed_segment_duration_seconds(0, float("inf")) == 0
    assert parse_list("[abc,def-_1]") == ["abc", "def-_1"] and parse_list('"[1, 2]"') == [1, 2]
    assert parse_list("") == [] and parse_list(None) == [] and parse_list("nan") == []
    assert parse_nested("[0, 10]") == [[0, 10]] and parse_nested("[[0],[5,6]]") == [[0], [5, 6]]
    assert parse_videos('["ab-c","d_e"]') == ["ab-c", "d_e"] and parse_videos("[]") == []
    assert total_processed_seconds("[[0,100],[5]]", "[[61,160],[35]]") == 60 + 59 + 29
    print("ok")
