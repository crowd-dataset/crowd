"""Totals of the mapping (processed seconds, unique videos); list parsing is in utils/core/parsing.py."""
from __future__ import annotations

import polars as pl

from utils.core import parsing


class Dataset_Stats:
    ENDPOINT_ADJUSTMENT_SECONDS = parsing.ENDPOINT_ADJUSTMENT_SECONDS

    # Kept as static methods: analysis.py and utils/plotting call them by these names.
    _processed_segment_duration_seconds = staticmethod(parsing.processed_segment_duration_seconds)
    _parse_nested_list = staticmethod(parsing.parse_nested)

    @staticmethod
    def _sum_durations(start_times: list, end_times: list) -> int:
        """Processed seconds of parsed nested start/end lists (a scalar element is one segment)."""
        total = 0
        for start, end in zip(start_times, end_times):
            start_list = start if isinstance(start, list) else [start]
            end_list = end if isinstance(end, list) else [end]
            for s, e in zip(start_list, end_list):
                total += parsing.processed_segment_duration_seconds(s, e)
        return total

    def calculate_total_seconds(self, df: pl.DataFrame) -> int:
        """Processed seconds of all segments of the mapping (`start_time`, `end_time` columns)."""
        return sum(self._sum_durations(parsing.parse_nested(s), parsing.parse_nested(e))
                   for s, e in df.select(["start_time", "end_time"]).iter_rows())

    def calculate_total_videos(self, df: pl.DataFrame) -> int:
        """Number of unique video IDs in the `videos` column; empty and missing cells count none."""
        return len({v for cell in df.get_column("videos").to_list() for v in parsing.parse_videos(cell)})
