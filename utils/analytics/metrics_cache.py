"""YOLO class IDs used by the analysis, and unique (and duplicated) keys of the mapping."""
from typing import Optional, Sequence, Set, Tuple, Union

import polars as pl

from utils.core import parsing

YOLO_PERSON = 0
YOLO_BICYCLE = 1
YOLO_CAR = 2
YOLO_MOTORCYCLE = 3
YOLO_BUS = 5
YOLO_TRUCK = 7
YOLO_TRAFFIC_SIGN_IDS = (9, 11)
YOLO_CELLPHONE = 67

UniqueValue = Union[str, Tuple[str, ...]]
UniqueValues = Set[UniqueValue]


class MetricsCache:
    # Kept as a static method: analysis.py and utils/plotting call it by this name.
    _parse_videos_cell = staticmethod(parsing.parse_videos)

    def get_unique_values(
        self,
        df: pl.DataFrame,
        value: Union[str, Sequence[str]],
        null_placeholder: str = "__NULL__",
        return_duplicates: bool = False,
    ) -> Tuple[UniqueValues, int, Optional[pl.DataFrame]]:
        """(unique values, their number, duplicates report or None) of column `value` or of a composite key.

        A single column skips nulls; in a composite key a null (e.g., no state) is a value, shown as
        null_placeholder. The report lists keys on more than one row with their row indices."""
        cols = list(value) if isinstance(value, (list, tuple)) else [value]
        key_exprs = [pl.col(c).cast(pl.Utf8).fill_null(null_placeholder).alias(c) for c in cols]

        unique_values: UniqueValues = set()
        if len(cols) == 1:
            unique_values.update(df.get_column(cols[0]).cast(pl.Utf8).drop_nulls().unique().to_list())
        else:
            unique_values.update(tuple(str(v) for v in row) for row in df.select(key_exprs).unique().rows())

        dup_report: Optional[pl.DataFrame] = None
        if return_duplicates:
            dup_report = (
                df.with_row_index("row_index").with_columns(key_exprs)
                .group_by(cols)
                .agg(pl.len().alias("dup_count"), pl.col("row_index").alias("row_indices"))
                .filter(pl.col("dup_count") > 1)
                .sort("dup_count", descending=True)
            )

        return unique_values, len(unique_values), dup_report
