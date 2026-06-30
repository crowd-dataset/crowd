from __future__ import annotations

import ast
import math
from typing import Any

import polars as pl

from utils.core.metadata import MetaData

metadata = MetaData()


class Dataset_Stats:
    # The detector/tracker pipeline trims each segment to t_end - 1 second.
    # All analysis durations should therefore use this processed endpoint.
    ENDPOINT_ADJUSTMENT_SECONDS = 1

    def __init__(self) -> None:
        pass

    @staticmethod
    def _processed_segment_duration_seconds(start_time, end_time) -> int:
        """Return processed segment duration using the pipeline endpoint adjustment.

        For normal segments, the processed endpoint is end_time - 1 second.
        If subtracting one second would make the endpoint non-positive relative
        to start_time, fall back to the original endpoint, matching main.py.
        """
        try:
            start = int(float(start_time))
            end = int(float(end_time))
        except Exception:
            return 0

        processed_end = end - Dataset_Stats.ENDPOINT_ADJUSTMENT_SECONDS
        if processed_end <= start:
            processed_end = end

        return max(0, processed_end - start)

    @staticmethod
    def _parse_nested_list(s: str | None) -> list:
        """
        Parse stringified Python list (possibly nested) into a Python list.
        Normalizes:
          [] -> []
          [0, 10] -> [[0, 10]]
          [[0, 10], [20, 30]] -> as-is
        """
        if not isinstance(s, str) or not s.strip():
            return []
        try:
            v = ast.literal_eval(s)
        except Exception:
            return []
        if not isinstance(v, list):
            return []
        if v and not any(isinstance(el, list) for el in v):
            return [v]
        return v

    @staticmethod
    def _sum_durations(start_times: list, end_times: list) -> int:
        """Sum processed durations across nested start/end lists."""
        total = 0
        for start, end in zip(start_times, end_times):
            start_list = start if isinstance(start, list) else [start]
            end_list = end if isinstance(end, list) else [end]
            for s, e in zip(start_list, end_list):
                total += Dataset_Stats._processed_segment_duration_seconds(s, e)
        return total

    @staticmethod
    def _videos_to_list_expr(col: str = "videos") -> pl.Expr:
        """
        Convert `videos` string column to a list of cleaned video IDs.
        Handles examples like:
          [D_LyZL4P3_k]
          "[ikOjqfFj-7o,GDXBW8LRFu4]"
          []
          null
        """
        videos_clean = (
            pl.col(col)
            .cast(pl.Utf8)
            .fill_null("")
            .str.strip_chars()
            .str.strip_chars("\"'")   # remove surrounding quotes if present
            .str.strip_chars("[]")    # remove brackets
            .str.strip_chars()
        )

        return (
            videos_clean
            .str.split(",")
            .list.eval(pl.element().str.strip_chars())
            .list.filter(pl.element() != "")
        )

    # ----------------------------
    # Public methods (Polars inputs)
    # ----------------------------
    def calculate_total_seconds_for_locality(self, df: pl.DataFrame, locality_name: str, state_name: str) -> int:
        """Calculates the total number of seconds of video for a given locality and state.

        Args:
            df (pl.DataFrame): Must include `locality`, `state`, `start_time`, `end_time`.
            locality_name (str): locality to match.
            state_name (str): State to match; if "unknown", matches null/empty/"NA".

        Returns:
            int: Total processed duration seconds for the first matching row; 0 if no match.
        """
        if state_name.lower() == "unknown":
            mask = (
                (pl.col("locality") == locality_name)
                & (
                    pl.col("state").is_null()
                    | (pl.col("state").cast(pl.Utf8).str.strip_chars() == "")
                    | (pl.col("state").cast(pl.Utf8) == "NA")
                )
            )
        else:
            mask = (pl.col("locality") == locality_name) & (pl.col("state") == state_name)

        row = df.filter(mask).select(["start_time", "end_time"]).head(1)
        if row.height == 0:
            return 0

        start_s = row["start_time"][0]
        end_s = row["end_time"][0]

        start_times = self._parse_nested_list(start_s)
        end_times = self._parse_nested_list(end_s)

        return self._sum_durations(start_times, end_times)

    def calculate_total_seconds(self, df: pl.DataFrame) -> int:
        """Calculates total processed video duration (seconds) across the entire mapping DataFrame.

        Args:
            df (pl.DataFrame): Must include `start_time` and `end_time`.

        Returns:
            int: Total processed duration seconds across all rows.
        """
        total = 0
        for start_s, end_s in df.select(["start_time", "end_time"]).iter_rows():
            start_times = self._parse_nested_list(start_s)
            end_times = self._parse_nested_list(end_s)
            total += self._sum_durations(start_times, end_times)
        return total

    def remove_columns_below_threshold(self, df: pl.DataFrame, threshold: int) -> pl.DataFrame:
        """Removes `start_time*`/`end_time*` column pairs where total recorded time is below a threshold.

        Args:
            df (pl.DataFrame): Contains start/end columns (stringified nested lists).
            threshold (int): Minimum total seconds required to retain a pair.

        Returns:
            pl.DataFrame: DataFrame with low-duration pairs removed.
        """
        cols = df.columns
        bases: set[str] = set()

        for c in cols:
            if c.startswith("start_time"):
                bases.add(c.replace("start_time", ""))
            elif c.startswith("end_time"):
                bases.add(c.replace("end_time", ""))

        cols_to_remove: list[str] = []

        for base in bases:
            start_col = f"start_time{base}"
            end_col = f"end_time{base}"
            if start_col not in cols or end_col not in cols:
                continue

            total_seconds = 0
            for start_s, end_s in df.select([start_col, end_col]).iter_rows():
                start_times = self._parse_nested_list(start_s)
                end_times = self._parse_nested_list(end_s)
                total_seconds += self._sum_durations(start_times, end_times)

            if total_seconds < threshold:
                cols_to_remove.extend([start_col, end_col])

        cols_to_remove = sorted(set(cols_to_remove))
        return df.drop(cols_to_remove) if cols_to_remove else df

    def calculate_total_videos(self, df: pl.DataFrame) -> int:
        """Counts total number of unique videos from a Polars mapping DataFrame.

        Args:
            df (pl.DataFrame): Must include `videos`.

        Returns:
            int: Unique video count across the dataset.
        """
        out = (
            df.select(self._videos_to_list_expr("videos").alias("_videos"))
            .explode("_videos")
            .select(pl.col("_videos").n_unique().alias("n_unique_videos"))
        )
        return int(out.item())

    def pedestrian_cross_per_locality(self, pedestrian_crossing_count: dict, df_mapping: pl.DataFrame) -> dict:
        """Aggregates pedestrian crossing counts per locality-condition key.

        Keeps your existing MetaData lookup:
            metadata.find_values_with_video_id(...)

        Note: df_mapping is Polars; it is converted once to pandas for the MetaData helper.

        Args:
            pedestrian_crossing_count (dict): {video_id: {"ids": [...]}, ...}
            df_mapping (pl.DataFrame): mapping data

        Returns:
            dict: { "{locality}_{lat}_{long}_{condition}": total_count }
        """
        final: dict[str, int] = {}

        # Count events per video
        count = {key: len(value.get("ids", [])) for key, value in pedestrian_crossing_count.items()}

        for video_id, total_events in count.items():
            result = metadata.find_values_with_video_id(df_mapping, video_id)

            if result is not None:
                condition = result[3]
                locality = result[4]
                lat = result[6]
                long = result[7]

                locality_time_key = f"{locality}_{lat}_{long}_{condition}"
                final[locality_time_key] = final.get(locality_time_key, 0) + total_events

        return final

    def pedestrian_cross_per_country(self, pedestrian_cross_locality: dict, df_mapping: pl.DataFrame) -> dict:
        """Aggregates pedestrian crossing counts from locality level to country level.

        Keeps your existing MetaData lookup:
            metadata.get_value(...)

        Note: df_mapping is Polars; it is converted once to pandas for the MetaData helper.

        Args:
            pedestrian_cross_locality (dict): { "{locality}_{lat}_{long}_{condition}": count }
            df_mapping (pl.DataFrame): mapping data

        Returns:
            dict: { "{country}_{condition}": total_count }
        """
        final: dict[str, int] = {}

        for locality_lat_long_cond, value in pedestrian_cross_locality.items():
            try:
                locality, lat, _lon, cond = locality_lat_long_cond.split("_")
                lat_f = float(lat)
            except Exception:
                continue

            country = metadata.get_value(df_mapping, "locality", locality, "lat", lat_f, "country")
            country_key = f"{country}_{cond}"
            final[country_key] = final.get(country_key, 0) + int(value)

        return final

    def safe_average(self, values: list[Any]) -> float:
        """Calculates the average of a list, ignoring None and NaN values."""
        valid_values = [
            v for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        return sum(valid_values) / len(valid_values) if valid_values else 0.0
