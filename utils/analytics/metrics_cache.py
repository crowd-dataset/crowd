"""
metrics_cache.py

Purpose
-------
Compute and cache per-video detection metrics derived from YOLO CSV outputs, then wrap/aggregate
those metrics via Grouping.locality_country_wrapper() using the mapping dataframe.

Key design points
-----------------
- Uses a class-level cache so multiple downstream calls do not re-scan the filesystem or re-read CSVs.
- Indexes CSV files once per compute run (fast lookup by filename/prefix).
- Computes *rates per minute* for most object classes (unique object IDs per minute).
- Computes a specialized "cellphones per person" normalized measure.

Compatibility / linting
-----------------------
- Avoids Python 3.10-only typing features such as:
    - typing.TypeAlias
    - PEP 604 unions: X | Y
  and avoids PEP 585 generics such as list[str], set[str], dict[str, str] which can trigger
  mypy/pylint issues depending on configured python_version.
"""

import ast
import os
import re
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from utils.core.grouping import Grouping
from utils.core.metadata import MetaData

# ------------------------------------------------------------------------------
# Constants (YOLO class IDs) used by the project
# ------------------------------------------------------------------------------
YOLO_PERSON = 0
YOLO_BICYCLE = 1
YOLO_CAR = 2
YOLO_MOTORCYCLE = 3
YOLO_BUS = 5
YOLO_TRUCK = 7
YOLO_TRAFFIC_SIGN_IDS = (9, 11)
YOLO_CELLPHONE = 67

# ------------------------------------------------------------------------------
# Type aliases (kept compatible with older Python tooling)
# ------------------------------------------------------------------------------
UniqueValue = Union[str, Tuple[str, ...]]
UniqueValues = Set[UniqueValue]

# ------------------------------------------------------------------------------
# Shared helpers (external dependencies)
# ------------------------------------------------------------------------------
_METADATA = MetaData()
_GROUPING = Grouping()
_LOGGER = CustomLogger(__name__)


class MetricsCache:
    # The detector/tracker pipeline trims each segment to t_end - 1 second.
    ENDPOINT_ADJUSTMENT_SECONDS = 1

    """
    Compute and cache metrics derived from YOLO detection CSVs.

    Public entrypoints
    ------------------
    - calculate_cellphones(df_mapping)
    - calculate_traffic_signs(df_mapping)
    - calculate_traffic(df_mapping, ...flags...)
    - get_unique_values(df, value, ...)
    - clear_cache()

    Cache format
    ------------
    _all_metrics_cache is a dictionary mapping metric name -> wrapped output from Grouping.locality_country_wrapper().
    Example keys:
      "cellphones", "traffic_signs", "vehicles", "bicycles", "cars", "motorcycles", "buses", "trucks", "persons"
    """

    # Class-level cache: avoids recomputation across calls within the same process.
    _all_metrics_cache: ClassVar[Dict[str, Any]] = {}

    # --------------------------------------------------------------------------
    # CSV indexing and parsing helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _processed_segment_duration_seconds(start_time: Any, end_time: Any) -> float:
        """Return processed segment duration using the pipeline endpoint adjustment."""
        try:
            start = float(start_time)
            end = float(end_time)
        except Exception:
            return 0.0

        processed_end = end - MetricsCache.ENDPOINT_ADJUSTMENT_SECONDS
        if processed_end <= start:
            processed_end = end

        return max(0.0, processed_end - start)

    @staticmethod
    def _parse_videos_cell(videos_cell: Optional[str]) -> List[str]:
        """
        Extract video IDs robustly from the mapping cell content.

        The mapping file may store values like:
          [abc]
          "[abc,def]"
          ["abc","def"]
          []
        We treat any token matching [A-Za-z0-9_-]+ as an ID component.
        """
        if not isinstance(videos_cell, str):
            return []
        return re.findall(r"[\w-]+", videos_cell)

    @staticmethod
    def _index_csv_files(data_folders: Sequence[str], subfolders: Sequence[str]) -> Dict[str, str]:
        """
        Build a filename -> full_path index for detection CSVs.

        This allows O(1) lookup once we know the exact filename, and quick prefix checks when only
        vid/start_time is known.
        """
        csv_index: Dict[str, str] = {}
        for folder_path in data_folders:
            for sub in subfolders:
                sub_path = os.path.join(folder_path, sub)
                if not os.path.exists(sub_path):
                    continue

                for fname in os.listdir(sub_path):
                    if fname.endswith(".csv"):
                        csv_index[fname] = os.path.join(sub_path, fname)

        return csv_index

    @staticmethod
    def _extract_fps_from_filename(vid: str, start_time: str, filename: str) -> Optional[int]:
        """
        Extract FPS from filenames matching:
            {vid}_{start_time}_{fps}.csv

        Returns:
            int FPS if pattern matches, otherwise None.
        """
        pattern = r"^%s_%s_(\d+)\.csv$" % (re.escape(str(vid)), re.escape(str(start_time)))
        match = re.match(pattern, filename)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _count_unique_objects(df: pl.DataFrame, yolo_ids: Iterable[int]) -> int:
        """
        Count unique 'unique-id' values where 'yolo-id' is in yolo_ids.

        Notes:
        - Casts columns defensively to handle string-typed CSV columns.
        - Returns 0 if required columns are missing or result is empty.
        """
        required = {"yolo-id", "unique-id"}
        if not required.issubset(set(df.columns)):
            return 0

        filtered = df.filter(pl.col("yolo-id").cast(pl.Int64, strict=False).is_in(list(yolo_ids)))
        if filtered.height == 0:
            return 0

        # Polars scalar extraction: select -> item()
        return int(filtered.select(pl.col("unique-id").n_unique()).item())

    # --------------------------------------------------------------------------
    # Core compute path
    # --------------------------------------------------------------------------
    @classmethod
    def _compute_all_metrics(cls, df_mapping: pl.DataFrame) -> None:
        """
        Compute and cache all metrics for the provided mapping DataFrame.

        Expected df_mapping columns (minimum):
          - videos
          - start_time
          - time_of_day

        In addition, MetaData.find_values_with_video_id() must be able to locate:
          - start timestamp (seconds)
          - end timestamp (seconds)
          - fps (frames per second)
        from the mapping for a given key: "{vid}_{start_time}_{fps}"
        """
        # 1) Build an index of available detection CSVs
        data_folders = common.get_configs("data")
        subfolders = common.get_configs("sub_domain")
        csv_files = cls._index_csv_files(data_folders, subfolders)

        # 2) Prepare metric layers as raw per-video dictionaries (video_key -> metric_value)
        cellphone_metric: Dict[str, float] = {}
        traffic_signs_metric: Dict[str, float] = {}
        vehicles_metric: Dict[str, float] = {}
        bicycles_metric: Dict[str, float] = {}
        cars_metric: Dict[str, float] = {}
        motorcycles_metric: Dict[str, float] = {}
        buses_metric: Dict[str, float] = {}
        trucks_metric: Dict[str, float] = {}
        persons_metric: Dict[str, float] = {}

        min_conf = float(common.get_configs("min_confidence"))

        # 3) Iterate mapping rows (Polars)
        mapping_iter = df_mapping.select(["videos", "start_time", "time_of_day"]).iter_rows(named=True)

        for row in tqdm(mapping_iter, total=df_mapping.height, desc="Analysing the csv files:"):
            videos_cell = row.get("videos")
            start_time_cell = row.get("start_time")
            time_of_day_cell = row.get("time_of_day")

            video_ids = cls._parse_videos_cell(videos_cell)

            # start_time/time_of_day appear to be stored as python-literal strings (lists of lists).
            try:
                start_times = ast.literal_eval(start_time_cell) if isinstance(start_time_cell, str) else None
                time_of_day = ast.literal_eval(time_of_day_cell) if isinstance(time_of_day_cell, str) else None
            except Exception:
                # Malformed cells should not crash the run; skip this row.
                continue

            if not (isinstance(start_times, list) and isinstance(time_of_day, list)):
                continue

            # Each mapping row may reference multiple vids; each vid has multiple segments.
            for vid, start_times_list, time_of_day_list in zip(video_ids, start_times, time_of_day):
                if not isinstance(start_times_list, list) or not isinstance(time_of_day_list, list):
                    continue

                for start_time, _tod in zip(start_times_list, time_of_day_list):
                    # Identify a CSV by prefix {vid}_{start_time}_ then parse fps from the filename.
                    prefix = "%s_%s_" % (vid, start_time)

                    matches = [fname for fname in csv_files.keys() if fname.startswith(prefix) and fname.endswith(".csv")]  # noqa: E501
                    if not matches:
                        _LOGGER.warning("[WARNING] File not found for prefix: %s", prefix)
                        continue

                    if len(matches) > 1:
                        _LOGGER.warning(
                            "[WARNING] Multiple files found for prefix: %s, using first: %s",
                            prefix,
                            matches[0],
                        )

                    fps = cls._extract_fps_from_filename(str(vid), str(start_time), matches[0])
                    if fps is None:
                        _LOGGER.error("[ERROR] Could not extract fps from filename: %s", matches[0])
                        continue

                    filename = "%s_%s_%s.csv" % (vid, start_time, fps)
                    file_path = csv_files.get(filename)
                    if not file_path:
                        continue

                    # 4) Use mapping metadata to compute segment duration
                    key_for_meta = "%s_%s_%s" % (vid, start_time, fps)
                    meta = _METADATA.find_values_with_video_id(df_mapping, key_for_meta)
                    if meta is None:
                        continue

                    # NOTE: These indices come from your existing MetaData contract.
                    # If you can change MetaData to return a dict/namedtuple, that will be safer.
                    start_sec = meta[1]
                    end_sec = meta[2]
                    fps_from_meta = meta[17]

                    # Prefer the FPS from metadata if present/valid.
                    try:
                        fps_final = int(fps_from_meta)
                    except Exception:
                        fps_final = fps

                    duration = cls._processed_segment_duration_seconds(start_sec, end_sec)

                    if duration <= 0:
                        continue

                    video_key = "%s_%s_%s" % (vid, start_time, fps_final)

                    # 5) Read detection CSV and filter by confidence
                    try:
                        df = pl.read_csv(file_path)
                    except Exception as exc:
                        _LOGGER.warning("[WARNING] Failed reading %s: %s", file_path, exc)
                        continue

                    required_cols = {"confidence", "yolo-id", "unique-id"}
                    if not required_cols.issubset(set(df.columns)):
                        continue

                    df = df.filter(pl.col("confidence").cast(pl.Float64, strict=False) >= min_conf)
                    if df.height == 0:
                        # Nothing above threshold; still record zeros for rate metrics.
                        traffic_signs_metric[video_key] = 0.0
                        vehicles_metric[video_key] = 0.0
                        bicycles_metric[video_key] = 0.0
                        cars_metric[video_key] = 0.0
                        motorcycles_metric[video_key] = 0.0
                        buses_metric[video_key] = 0.0
                        trucks_metric[video_key] = 0.0
                        persons_metric[video_key] = 0.0
                        continue

                    # 6) Compute unique counts
                    persons = cls._count_unique_objects(df, [YOLO_PERSON])
                    cellphones = cls._count_unique_objects(df, [YOLO_CELLPHONE])

                    traffic_signs = cls._count_unique_objects(df, YOLO_TRAFFIC_SIGN_IDS)
                    vehicles = cls._count_unique_objects(df, [YOLO_CAR, YOLO_MOTORCYCLE, YOLO_BUS, YOLO_TRUCK])
                    bicycles = cls._count_unique_objects(df, [YOLO_BICYCLE])
                    cars = cls._count_unique_objects(df, [YOLO_CAR])
                    motorcycles = cls._count_unique_objects(df, [YOLO_MOTORCYCLE])
                    buses = cls._count_unique_objects(df, [YOLO_BUS])
                    trucks = cls._count_unique_objects(df, [YOLO_TRUCK])

                    # 7) Convert to per-minute rates (unique objects per minute)
                    per_min = 60.0 / duration

                    traffic_signs_metric[video_key] = float(traffic_signs) * per_min
                    vehicles_metric[video_key] = float(vehicles) * per_min
                    bicycles_metric[video_key] = float(bicycles) * per_min
                    cars_metric[video_key] = float(cars) * per_min
                    motorcycles_metric[video_key] = float(motorcycles) * per_min
                    buses_metric[video_key] = float(buses) * per_min
                    trucks_metric[video_key] = float(trucks) * per_min
                    persons_metric[video_key] = float(persons) * per_min

                    # 8) Cellphones metric: per-person normalized measure
                    # Your original formula:
                    #   avg_cellphone = ((cellphones * 60) / duration / persons) * 1000
                    if persons > 0 and cellphones > 0:
                        cellphone_metric[video_key] = ((float(cellphones) * 60.0) / duration / float(persons)) * 1000.0

        # 9) Wrap metrics with grouping layer and store into the class cache
        metric_layers: List[Tuple[str, Dict[str, float]]] = [
            ("cellphones", cellphone_metric),
            ("traffic_signs", traffic_signs_metric),
            ("vehicles", vehicles_metric),
            ("bicycles", bicycles_metric),
            ("cars", cars_metric),
            ("motorcycles", motorcycles_metric),
            ("buses", buses_metric),
            ("trucks", trucks_metric),
            ("persons", persons_metric),
        ]

        cls._all_metrics_cache = {}

        for idx, (name, layer) in enumerate(metric_layers, start=1):
            _LOGGER.info("[%s/%s] Wrapping '%s' ...", idx, len(metric_layers), name)
            wrapped = _GROUPING.locality_country_wrapper(
                input_dict=layer,
                mapping=df_mapping,
                show_progress=True,
            )
            cls._all_metrics_cache[name] = wrapped

    @classmethod
    def _ensure_cache(cls, df_mapping: pl.DataFrame) -> None:
        """Compute metrics if cache is empty."""
        if not cls._all_metrics_cache:
            cls._compute_all_metrics(df_mapping)

    # --------------------------------------------------------------------------
    # Public API: metric getters
    # --------------------------------------------------------------------------
    @classmethod
    def calculate_cellphones(cls, df_mapping: pl.DataFrame) -> Any:
        """Return wrapped cellphone metric (computes cache on first call)."""
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["cellphones"]

    @classmethod
    def calculate_traffic_signs(cls, df_mapping: pl.DataFrame) -> Any:
        """Return wrapped traffic-sign metric (computes cache on first call)."""
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["traffic_signs"]

    @classmethod
    def calculate_traffic(
        cls,
        df_mapping: pl.DataFrame,
        person: bool = False,
        bicycle: bool = False,
        motorcycle: bool = False,
        car: bool = False,
        bus: bool = False,
        truck: bool = False,
    ) -> Any:
        """
        Return a requested traffic-related metric from the cache.

        Selection logic (kept consistent with your original intent):
        - If person=True: return persons
        - Else if bicycle=True: return bicycles
        - Else if motorcycle=car=bus=truck=True: return vehicles (aggregate)
        - Else return the first specific vehicle type requested
        - Else fallback: vehicles
        """
        cls._ensure_cache(df_mapping)

        if person:
            return cls._all_metrics_cache["persons"]
        if bicycle:
            return cls._all_metrics_cache["bicycles"]

        if motorcycle and car and bus and truck:
            return cls._all_metrics_cache["vehicles"]
        if car:
            return cls._all_metrics_cache["cars"]
        if motorcycle:
            return cls._all_metrics_cache["motorcycles"]
        if bus:
            return cls._all_metrics_cache["buses"]
        if truck:
            return cls._all_metrics_cache["trucks"]

        return cls._all_metrics_cache["vehicles"]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached metrics (useful if input data changes)."""
        cls._all_metrics_cache.clear()

    # --------------------------------------------------------------------------
    # Utility: unique value extraction + duplicate reporting
    # --------------------------------------------------------------------------
    def get_unique_values(
        self,
        df: pl.DataFrame,
        value: Union[str, Sequence[str]],
        null_placeholder: str = "__NULL__",
        return_duplicates: bool = False,
    ) -> Tuple[UniqueValues, int, Optional[pl.DataFrame]]:
        """
        Returns (unique_values, count, dup_report).

        unique_values:
          - set[str] for single-column keys
          - set[tuple[str, ...]] for composite keys
        """
        cols = list(value) if isinstance(value, (list, tuple)) else [value]

        key_exprs = [pl.col(c).cast(pl.Utf8).fill_null(null_placeholder).alias(c) for c in cols]  # type: ignore
        keys_only = df.select(key_exprs)

        # Build as UniqueValues explicitly so Pylance does not infer set[Any] unions.
        unique_values: UniqueValues = set()

        if len(cols) == 1:
            series = keys_only.get_column(cols[0])  # type: ignore
            for v in series.unique().to_list():
                unique_values.add(str(v))
        else:
            for row in keys_only.unique().rows():
                unique_values.add(tuple(str(v) for v in row))

        dup_report: Optional[pl.DataFrame] = None
        if return_duplicates:
            keyed = df.with_row_index("row_index").with_columns(key_exprs)
            dup_report = (
                keyed.group_by(cols)
                .agg(
                    pl.len().alias("dup_count"),
                    pl.col("row_index").alias("row_indices"),
                )
                .filter(pl.col("dup_count") > 1)
                .sort("dup_count", descending=True)
            )

        return unique_values, len(unique_values), dup_report
