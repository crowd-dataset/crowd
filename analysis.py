"""
CROWD dataset analysis: maps, rollup tables, figures and the README statistics from the mapping file, plus the YOLO
object counts when the detection CSVs (or their committed snapshot) are available.

Authors:
- Shadab Alam <md_shadab_alam@outlook.com>
- Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
"""

from __future__ import annotations

import functools
import glob
import gzip
import io
import json
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pycountry
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs
from utils.analytics.metrics_cache import (YOLO_BICYCLE, YOLO_BUS, YOLO_CAR, YOLO_MOTORCYCLE, YOLO_PERSON,
                                           YOLO_TRUCK, MetricsCache)
from utils.core.dataset_stats import Dataset_Stats
from utils.core.parsing import (ENDPOINT_ADJUSTMENT_SECONDS, parse_list, parse_nested, parse_videos,
                                processed_segment_duration_seconds)
from utils.analytics import contributors
from utils.plotting import dataset_figures, hover
from utils.plotting.maps import Maps

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

maps = Maps()
dataset_stats = Dataset_Stats()
metrics_cache = MetricsCache()


def _flag(alpha_2: str) -> str:
    """Emoji flag of a two-letter country code (its pair of regional indicator symbols)."""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in alpha_2)


# Emoji flag per ISO3 code; Kosovo (XKX, XK) is not in ISO 3166
ISO3_TO_FLAG = {c.alpha_3: _flag(c.alpha_2) for c in pycountry.countries} | {"XKX": _flag("XK")}
NO_FLAG = "🏳️"

VEHICLE_MAP = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Motorcycle",  # "Two-wheeler" in the released files (crowd_index.jsonl)
    4: "Bicycle",
    5: "Automated car",
    6: "Electric scooter",
    7: "Monowheel/unicycle",
    8: "Emergency vehicle",
    9: "Automated bus",
    10: "Automated truck",
    11: "Automated motorcycle",
    12: "Non-electric scooter",
    13: "Pedestrian"
}
TIME_MAP = {0: "Day", 1: "Night"}

# the names the code used before these were module constants
analysis_class = SimpleNamespace(iso3_to_flag=ISO3_TO_FLAG, vehicle_map=VEHICLE_MAP, time_map=TIME_MAP)

# ---------------------------------------------------------------------
# Parsing the mapping's list cells into rows, video positions and segments
# ---------------------------------------------------------------------
_INT_LOL = pl.List(pl.List(pl.Int64))
_FLOAT_LOL = pl.List(pl.List(pl.Float64))
_PLACE_COLS = ["id", "continent", "country", "locality", "state", "iso3", "lat", "lon"]


def _decode(df: pl.DataFrame, col: str, dtype: pl.DataType, parse) -> pl.Series:
    """Column `col` of list cells as `dtype`: JSON-decoded, or with `parse` (utils.core.parsing) when a cell is not
    JSON of that shape (polars would read a flat list as one inner list per element)."""
    s = df[col].cast(pl.Utf8)
    cells = s.drop_nulls().str.strip_chars()
    if dtype not in (_INT_LOL, _FLOAT_LOL) or (cells.str.starts_with("[[") | cells.is_in(["[]", ""])).all():
        try:
            return s.str.json_decode(dtype).alias(col)
        except pl.exceptions.PolarsError:
            pass
    return s.map_elements(parse, return_dtype=dtype).alias(col)


def _processed_seconds(start: pl.Expr, end: pl.Expr) -> pl.Expr:
    """processed_segment_duration_seconds as an expression (end - 1 s - start, or end - start where the shorter end
    would not be after the start); 0 where a value is missing."""
    s, e = start.cast(pl.Int64, strict=False), end.cast(pl.Int64, strict=False)
    shorter = e - ENDPOINT_ADJUSTMENT_SECONDS
    return (pl.when(shorter <= s).then(e).otherwise(shorter) - s).clip(lower_bound=0).fill_null(0)


def _parse(df_mapping: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The mapping as (rows, videos, segments).

    rows: the mapping (in its order) with `row` (position), the parsed list columns and per-row `upload_records`
    (videos), `segment_records` (time-of-day labels, else start/end pairs) and `footage_time_s` (processed seconds).
    videos: one row per position of the row's list cells (up to the longest), with that position's `video_id`, `tods`,
    `starts`, `ends` and `vehicle` (null where a list is shorter) and the row's place columns.
    segments: one row per start/end pair of each position, with its processed `duration_s`.
    """
    rows = df_mapping.with_row_index("row").with_columns(
        pl.col("videos").cast(pl.Utf8).str.extract_all(r"[\w-]+").fill_null([]).alias("videos_list"),  # parse_videos
        _decode(df_mapping, "time_of_day", _INT_LOL, parse_nested).alias("time_of_day_lol"),
        _decode(df_mapping, "start_time", _FLOAT_LOL, parse_nested).alias("start_time_lol"),
        _decode(df_mapping, "end_time", _FLOAT_LOL, parse_nested).alias("end_time_lol"),
        _decode(df_mapping, "vehicle_type", pl.List(pl.Int64), parse_list).alias("vehicle_list"),
    )
    positions = [
        rows.select("row", pl.col(src).alias(dst))
            .with_columns(pl.int_ranges(0, pl.col(dst).list.len()).alias("i"))
            .explode(dst, "i")
            .drop_nulls("i")
        for src, dst in [("videos_list", "video_id"), ("time_of_day_lol", "tods"), ("start_time_lol", "starts"),
                         ("end_time_lol", "ends"), ("vehicle_list", "vehicle")]
    ]
    videos = (
        functools.reduce(lambda a, b: a.join(b, on=["row", "i"], how="full", coalesce=True), positions)
        .join(rows.select("row", *_PLACE_COLS), on="row", how="left")
        .sort("row", "i")
    )
    pairs = pl.min_horizontal(pl.col("starts").list.len(), pl.col("ends").list.len())
    segments = (
        videos.filter(pl.col("starts").is_not_null() & pl.col("ends").is_not_null() & (pairs > 0))
              .with_columns(pl.col("starts").list.head(pairs).alias("start"),
                            pl.col("ends").list.head(pairs).alias("end"))
              .explode("start", "end")
              .with_columns(_processed_seconds(pl.col("start"), pl.col("end")).alias("duration_s"))
    )
    per_row = segments.group_by("row").agg(pl.len().cast(pl.Int64).alias("segment_records_se"),
                                           pl.sum("duration_s").alias("footage_time_s"))
    labels = pl.col("time_of_day_lol").list.eval(pl.element().list.len()).list.sum().fill_null(0).cast(pl.Int64)
    rows = (
        rows.join(per_row, on="row", how="left")
            .sort("row")
            .with_columns(pl.col("segment_records_se", "footage_time_s").fill_null(0),
                          pl.col("videos_list").list.len().cast(pl.Int64).alias("upload_records"),
                          labels.alias("segment_records_tod"))
            .with_columns(pl.when(pl.col("segment_records_tod") > 0).then(pl.col("segment_records_tod"))
                            .otherwise(pl.col("segment_records_se")).alias("segment_records"))
    )
    return rows, videos, segments


# ---------------------------------------------------------------------
# Rollup tables (log_rollups): helpers and one function per section
# ---------------------------------------------------------------------
CONTINENT_ORDER = ["Europe", "Asia", "North America", "Africa", "Oceania", "South America", "Total"]
DAYNIGHT_ORDER = ["only_day", "only_night", "both_day_night", "unknown"]
TOP_COUNTRIES = 10  # concentration tables (section I)
TOP_LOCALITIES = 20
CONCENTRATION_PERCENTILES = [1, 5, 10]


def _df_full_str(df: pl.DataFrame) -> str:
    """The whole DataFrame as text for the log (no `...` truncation)."""
    with pl.Config(tbl_rows=10_000, tbl_cols=1_000, tbl_width_chars=5_000, fmt_str_lengths=5_000):
        return df.__repr__()


def _log_table(title: str, df: pl.DataFrame) -> None:
    """Log a section title and its table, each on its own lines."""
    logger.info(f"\n=== [rollups] {title} ===")
    logger.info(f"\n{_df_full_str(df)}")


def _ensure_zero_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """`df` with the missing `cols` added as Int64 zeros (a category absent from a pivot)."""
    return df.with_columns(pl.lit(0).cast(pl.Int64).alias(c) for c in cols if c not in df.columns)


def _with_total(df: pl.DataFrame, total: dict) -> pl.DataFrame:
    """`df` with the row `total` appended, cast to `df`'s column types."""
    return pl.concat([df, pl.DataFrame([total]).select(df.columns).cast(df.schema, strict=False)])


def _sort_continent(df: pl.DataFrame, *then: str) -> pl.DataFrame:
    """`df` sorted by CONTINENT_ORDER of its `continent` column (unknown names last), then by the columns `then`."""
    order = {name: i for i, name in enumerate(CONTINENT_ORDER)}
    return df.sort(pl.col("continent").replace_strict(order, default=999, return_dtype=pl.Int64), *then)


def _pct(count: pl.Expr, total: int) -> pl.Expr:
    """`count` as a percentage of `total` (at least 1), 2 decimals."""
    return (count / pl.lit(max(total, 1)) * 100).round(2)


def _share(count: str, total: str) -> pl.Expr:
    """Column `count` as a percentage of column `total` per row (0 where the total is 0), 2 decimals."""
    return pl.when(pl.col(total) > 0).then((pl.col(count) / pl.col(total) * 100).round(2)).otherwise(pl.lit(0.0))


def _pct_py(count: int, total: int) -> float:
    """`count` as a percentage of `total` (at least 1), 2 decimals."""
    return round(count / max(total, 1) * 100, 2)


def _count_pct(count: str, pct: str) -> pl.Expr:
    """'<count> (<pct>%)' with the percentage at 2 decimals."""
    return pl.concat_str([pl.col(count).cast(pl.Int64).cast(pl.Utf8), pl.lit(" ("),
                          pl.col(pct).round(2).map_elements(lambda x: f"{float(x):.2f}", return_dtype=pl.Utf8),
                          pl.lit("%)")])


def _daynight(has_day: pl.Expr, has_night: pl.Expr) -> pl.Expr:
    """Category of a set of time-of-day labels: both_day_night, only_day, only_night or unknown."""
    return (pl.when(has_day & has_night).then(pl.lit("both_day_night"))
              .when(has_day).then(pl.lit("only_day"))
              .when(has_night).then(pl.lit("only_night"))
              .otherwise(pl.lit("unknown")).alias("daynight_category"))


def _has_continent() -> pl.Expr:
    """Rows with a continent."""
    return pl.col("continent").is_not_null() & (pl.col("continent") != "")


def _named(videos: pl.DataFrame) -> pl.DataFrame:
    """Video positions with a known vehicle type, one row per time-of-day label, with `vehicle_type_name` and
    `time_of_day_name`."""
    return (
        videos.filter(pl.col("tods").is_not_null())
              .explode("tods")
              .with_columns(pl.col("vehicle").replace_strict(VEHICLE_MAP, default=None, return_dtype=pl.Utf8)
                              .alias("vehicle_type_name"),
                            pl.col("tods").replace_strict(TIME_MAP, default=None, return_dtype=pl.Utf8)
                              .alias("time_of_day_name"))
              .filter(pl.col("vehicle_type_name").is_not_null() & pl.col("time_of_day_name").is_not_null())
    )


def _log_summary(rows: pl.DataFrame) -> int:
    """Log the dataset totals and alignment warnings; return the total segment records."""
    n = rows.height
    total_segment_records = int(rows["segment_records"].sum())
    logger.info("\n=== [rollups] Dataset summary ===")
    logger.info(f"rows={n} | upload_records(sum videos per row)={int(rows['upload_records'].sum())}")
    logger.info(f"segment_records(label-entries)={total_segment_records} | "
                f"duration_h={round(int(rows['footage_time_s'].sum()) / 3600, 2)}")
    n_slots = rows.filter(pl.col("upload_records") != pl.col("time_of_day_lol").list.len()).height
    if n_slots:
        logger.warning(f"[rollups] {n_slots}/{n} rows have len(videos_list) != len(time_of_day_lol). "
                       "Those rows may have parsing/alignment issues (zips will truncate).")
    n_seg = rows.filter((pl.col("segment_records_tod") > 0) & (pl.col("segment_records_se") > 0)
                        & (pl.col("segment_records_tod") != pl.col("segment_records_se"))).height
    if n_seg:
        logger.warning(f"[rollups] {n_seg}/{n} rows have segment_records_tod != segment_records_se (both present). "
                       "Using time_of_day-derived counts for segment_records to match paper definition.")
    n_none = rows.filter((pl.col("segment_records") == 0) & (pl.col("upload_records") > 0)).height
    if n_none:
        logger.warning(f"[rollups] {n_none}/{n} rows have videos but neither time-of-day labels nor start/end "
                       "pairs; they count 0 segment records.")
    return total_segment_records


def _log_continent_segments(rows: pl.DataFrame) -> None:
    """A) segment records and processed footage per continent, with shares and a Total row."""
    cont = (rows.filter(_has_continent()).group_by("continent")
                .agg(pl.sum("segment_records"), pl.sum("footage_time_s").alias("duration_s")))
    records, seconds = int(cont["segment_records"].sum()), int(cont["duration_s"].sum())
    cont = cont.with_columns(_pct(pl.col("segment_records"), records).alias("segment_share_pct"),
                             _pct(pl.col("duration_s"), seconds).alias("duration_share_pct"),
                             (pl.col("duration_s") / 3600).round(2).alias("duration_h"))
    total = {"continent": "Total", "segment_records": records, "duration_s": seconds, "segment_share_pct": 100.0,
             "duration_share_pct": 100.0, "duration_h": round(seconds / 3600, 2)}
    _log_table("A) Continent segment distribution (label-entries; paper)", _sort_continent(_with_total(cont, total)))


def _canonical_continents(videos: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    """(video_id, continent) with one continent per upload: the one most of its mapping rows are in, ties broken by
    the most processed footage there, then by name. Warns about uploads mapped to several continents."""
    seconds = segments.group_by("row", "i").agg(pl.sum("duration_s"))
    per_continent = (
        videos.filter(pl.col("video_id").is_not_null() & _has_continent())
              .join(seconds, on=["row", "i"], how="left")
              .group_by(["video_id", "continent"])
              .agg(pl.len().alias("n_rows"), pl.sum("duration_s"))
              .fill_null(0)
    )
    canon = (
        per_continent.sort(["video_id", "n_rows", "duration_s", "continent"], descending=[False, True, True, False])
                     .group_by("video_id", maintain_order=True)
                     .agg(pl.col("continent").first(), pl.col("continent").n_unique().alias("n_continents"))
    )
    n_multi = canon.filter(pl.col("n_continents") > 1).height
    if n_multi:
        logger.warning(
            f"[rollups] {n_multi} uploads appear in more than one continent in the mapping. Segment and duration "
            "rollups sum over mapping rows, so those uploads contribute to each mapped continent. Unique upload "
            "counts assign each upload to one continent using most frequent mapping (ties by mapped duration).")
    return canon.select("video_id", "continent")


def _log_unique_uploads(canon: pl.DataFrame) -> None:
    """B) unique uploads per (canonical) continent, as % of all unique uploads, with a Total row."""
    total = canon.height
    cont = canon.group_by("continent").agg(pl.col("video_id").n_unique().alias("unique_uploads"))
    cont = cont.with_columns(_pct(pl.col("unique_uploads"), total).alias("unique_uploads_pct"))
    _log_table("B) Continent unique uploads (canonical; % of GLOBAL unique uploads)",
               _sort_continent(_with_total(cont, {"continent": "Total", "unique_uploads": total,
                                                  "unique_uploads_pct": 100.0})))


def _log_vehicles(videos: pl.DataFrame) -> None:
    """C) unique videos per vehicle type (% of all unique uploads) and C2) vehicle x time-of-day label counts."""
    uploads = videos["video_id"].drop_nulls().n_unique()
    per_vehicle = (
        videos.with_columns(pl.col("vehicle").replace_strict(VEHICLE_MAP, default=None, return_dtype=pl.Utf8)
                              .alias("vehicle_type_name"))
              .filter(pl.col("video_id").is_not_null() & pl.col("vehicle_type_name").is_not_null())
              .unique(["video_id", "vehicle_type_name"])
              .group_by("vehicle_type_name").len().rename({"len": "unique_videos"})
              .with_columns(_pct(pl.col("unique_videos"), uploads).alias("unique_videos_pct"))
              .sort("vehicle_type_name")
    )
    logger.info("=== [rollups] C) Vehicle distribution by unique videos (% of GLOBAL unique uploads) ===")
    logger.info(_df_full_str(per_vehicle))

    veh_tod = _named(videos).group_by(["vehicle_type_name", "time_of_day_name"]).len().rename({"len": "count"})
    veh_tod = (veh_tod.with_columns(_pct(pl.col("count"), int(veh_tod["count"].sum())).alias("pct"))
                      .sort(["vehicle_type_name", "time_of_day_name"]))
    logger.info("=== [rollups] C2) Vehicle x time-of-day (pair counts + %) ===")
    logger.info(_df_full_str(veh_tod))


def _log_continent_time_of_day(videos: pl.DataFrame, total_segment_records: int) -> None:
    """D) Day and Night label entries per continent (counts and % within the continent), with a Total row."""
    wide = (
        videos.filter(_has_continent() & pl.col("tods").is_not_null())
              .explode("tods")
              .with_columns(pl.col("tods").replace_strict(TIME_MAP, default=None, return_dtype=pl.Utf8)
                              .alias("time_of_day_name"))
              .filter(pl.col("time_of_day_name").is_not_null())
              .group_by(["continent", "time_of_day_name"]).len().rename({"len": "entries"})
              .pivot(index="continent", on="time_of_day_name", values="entries", aggregate_function="first")
              .fill_null(0)
    )
    wide = (_ensure_zero_cols(wide, ["Day", "Night"]).select("continent", "Day", "Night")
            .with_columns((pl.col("Day") + pl.col("Night")).alias("total_entries"))
            .with_columns(_share("Day", "total_entries").alias("day_pct"),
                          _share("Night", "total_entries").alias("night_pct")))
    day, night = int(wide["Day"].sum()), int(wide["Night"].sum())
    wide = _sort_continent(_with_total(wide, {
        "continent": "Total", "Day": day, "Night": night, "total_entries": day + night,
        "day_pct": _pct_py(day, day + night), "night_pct": _pct_py(night, day + night)}))
    paper = wide.select("continent", _count_pct("Day", "day_pct").alias("day_entries"),
                        _count_pct("Night", "night_pct").alias("night_entries"), "total_entries")
    _log_table("D) Continent x time-of-day label entries (paper)", paper)
    _log_table("D) Continent x time-of-day label entries (raw wide; with pct)", wide)

    if day + night != total_segment_records:
        logger.warning(
            f"[rollups] Segment count sanity: total_entries_global={day + night} != "
            f"total_segment_records(df_base)={total_segment_records}. "
            "This can happen if time_of_day parsing differs between df_mapping and df_base.")
    else:
        logger.info("[rollups] Segment count sanity: total_entries_global matches total_segment_records "
                    f"({total_segment_records}).")


def _log_daynight(videos: pl.DataFrame, canon: pl.DataFrame) -> None:
    """E) uploads with day-only, night-only or both labels, and F) the same per canonical continent."""
    flags = (
        videos.filter(pl.col("video_id").is_not_null())
              .group_by("video_id")
              .agg(pl.col("tods").list.contains(0).fill_null(False).any().alias("has_day"),
                   pl.col("tods").list.contains(1).fill_null(False).any().alias("has_night"))
              .with_columns(_daynight(pl.col("has_day"), pl.col("has_night")))
    )
    order = {name: i for i, name in enumerate(DAYNIGHT_ORDER)}
    overall = (flags.group_by("daynight_category").len().rename({"len": "uploads"})
                    .sort(pl.col("daynight_category").replace_strict(order, default=99, return_dtype=pl.Int64)))
    uploads = int(overall["uploads"].sum())
    overall = overall.with_columns(_pct(pl.col("uploads"), uploads).alias("pct"))
    _log_table("E) Upload day/night composition (global; paper)",
               _with_total(overall, {"daynight_category": "Total", "uploads": uploads, "pct": 100.0}))

    canon_flags = flags.join(canon, on="video_id", how="left").filter(_has_continent())
    wide = (
        canon_flags.group_by(["continent", "daynight_category"]).len().rename({"len": "unique_uploads"})
                   .pivot(index="continent", on="daynight_category", values="unique_uploads",
                          aggregate_function="first")
                   .fill_null(0)
    )
    wide = (_ensure_zero_cols(wide, DAYNIGHT_ORDER).select("continent", *DAYNIGHT_ORDER)
            .with_columns(functools.reduce(lambda a, b: a + b, map(pl.col, DAYNIGHT_ORDER)).alias("unique_uploads"))
            .with_columns(_share(c, "unique_uploads").alias(f"{c}_pct") for c in DAYNIGHT_ORDER))
    totals = {c: canon_flags.filter(pl.col("daynight_category") == c).height for c in DAYNIGHT_ORDER}
    total = sum(totals.values())
    wide = _sort_continent(_with_total(wide, {"continent": "Total", **totals, "unique_uploads": total,
                                              **{f"{c}_pct": _pct_py(n, total) for c, n in totals.items()}}))
    paper = wide.select("continent", _count_pct("only_day", "only_day_pct").alias("day_only"),
                        _count_pct("only_night", "only_night_pct").alias("night_only"),
                        _count_pct("both_day_night", "both_day_night_pct").alias("both_day_night"),
                        _count_pct("unknown", "unknown_pct").alias("unknown"), "unique_uploads")
    _log_table("F) Upload day/night composition by continent (canonical; paper)", paper)
    _log_table("F) Upload day/night composition by continent (raw wide; with pct)", wide)
    if total != canon.height:
        logger.warning(f"[rollups] Canonical by-continent unique_uploads total={total} != global unique videos="
                       f"{canon.height} (unexpected).")
    else:
        logger.info(f"[rollups] Canonical by-continent totals match global unique videos {canon.height}.")


def _log_vehicle_types_per_continent(videos: pl.DataFrame) -> None:
    """G) vehicle types per continent seen by day only, night only or both, and vehicle x time-of-day label counts
    per continent."""
    pairs = _named(videos.filter(_has_continent()))
    day, night = pl.col("time_of_day_name") == "Day", pl.col("time_of_day_name") == "Night"
    presence = pairs.group_by(["continent", "vehicle_type_name"]).agg(
        day.any().alias("has_day"), night.any().alias("has_night"))
    presence = presence.with_columns(_daynight(pl.col("has_day"), pl.col("has_night")))
    types = presence.group_by("continent").agg(
        pl.col("vehicle_type_name").n_unique().alias("unique_vehicle_types"),
        (pl.col("daynight_category") == "only_day").sum().cast(pl.Int64).alias("day_only_types"),
        (pl.col("daynight_category") == "only_night").sum().cast(pl.Int64).alias("night_only_types"),
        (pl.col("daynight_category") == "both_day_night").sum().cast(pl.Int64).alias("types_in_both_day_night"),
    )
    _log_table("G) Vehicle types per continent (paper)", _sort_continent(types))
    per_continent = (
        pairs.group_by(["continent", "vehicle_type_name", "time_of_day_name"]).len().rename({"len": "count"})
             .with_columns((pl.col("count") / pl.sum("count").over("continent") * 100).round(2)
                           .alias("pct_within_continent"))
    )
    _log_table("G) Vehicle types per continent x time-of-day (pair counts + % within continent)",
               _sort_continent(per_continent, "vehicle_type_name", "time_of_day_name"))


def _rank(rows: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Uploads, segment records and processed footage per group of `keys`."""
    return (rows.group_by(keys)
                .agg(pl.sum("upload_records").alias("upload_count"), pl.sum("segment_records").alias("segment_count"),
                     pl.sum("footage_time_s"))
                .with_columns((pl.col("footage_time_s") / 3600).round(2).alias("footage_time_h")))


def _log_extremes(rows: pl.DataFrame) -> None:
    """H1) the locality with the most footage per continent, H) max/min locality and country by uploads, segments
    and footage. A locality is a mapping row (`id`): same-named places stay apart."""
    localities = _rank(rows.filter(pl.col("locality").is_not_null() & (pl.col("locality") != "")),
                       ["id", "continent", "country", "locality", "state", "iso3"]).drop("id")
    top = (
        localities.filter(pl.col("footage_time_s") > 0)
                  .join(localities.group_by("continent").agg(pl.max("footage_time_s").alias("max_s")),
                        on="continent")
                  .filter(pl.col("footage_time_s") == pl.col("max_s"))
                  .drop("max_s")
                  .sort(["continent", "footage_time_s", "segment_count", "upload_count", "locality"],
                        descending=[False, True, True, True, False])
    )
    _log_table("H1) Max locality by duration within each continent (including ties)", top)
    # one per continent: ties broken by segment_count, then upload_count, then locality name
    _log_table("H1) Max locality by duration within each continent (one per continent)",
               top.unique(subset=["continent"], keep="first").sort("continent"))

    countries = _rank(rows.filter(pl.col("country").is_not_null() & (pl.col("country") != "")),
                      ["continent", "country", "iso3"])
    # ties (many places have 1 upload) go to the first by name
    for name, df, names in [("locality", localities, ["country", "locality", "state"]),
                            ("COUNTRY", countries, ["country", "continent"])]:
        for col, what in [("upload_count", "upload_count"), ("segment_count", "segment_count"),
                          ("footage_time_s", "duration")]:
            most = df.sort([col, *names], descending=[True] + [False] * len(names)).head(1)
            least = df.filter(pl.col(col) > 0).sort([col, *names]).head(1)
            logger.info(f"\n=== [rollups] H) Max/Min {name} by {what} ===")
            logger.info(f"\nMAX:\n{_df_full_str(most)}\nMIN (non-zero):\n{_df_full_str(least)}")


def _clean_text(col: str) -> pl.Expr:
    """Column `col` as stripped text, "" where missing or nan/none/null."""
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    return pl.when(s.str.to_lowercase().is_in(["nan", "none", "null"])).then(pl.lit("")).otherwise(s) \
             .fill_null("").alias(col)


def _concentration_dir() -> Path:
    """_output/concentration next to the mapping file (in the working folder if the mapping is not configured)."""
    try:
        return Path(str(common.get_configs("mapping"))).expanduser().resolve().parent / "_output" / "concentration"
    except Exception as e:
        logger.warning(f"[rollups] I) No mapping path in config ({e!r}); writing to the working folder.")
        return Path(os.getcwd()) / "_output" / "concentration"


def _log_concentration(segments: pl.DataFrame) -> None:
    """I) how concentrated the processed footage is in few countries and localities; also written as CSV and JSON
    to _concentration_dir(). A locality is a mapping row with its place columns, so homonyms stay apart."""
    seg = (
        segments.filter(pl.col("video_id").is_not_null() & (pl.col("duration_s") > 0))
                .with_columns(_clean_text(c) for c in _PLACE_COLS)
                .with_columns(pl.concat_str(["id", "locality", "state", "country", "iso3", "continent", "lat", "lon"],
                                            separator="|").alias("locality_key"),
                              pl.concat_str(["iso3", "country"], separator="|").alias("country_key"))
    )
    if seg.height == 0:
        logger.warning("[rollups] I) Concentration analysis skipped because no valid segment rows were found.")
        return
    total_s = int(seg["duration_s"].sum())
    n_segments = seg.height
    n_uploads = seg["video_id"].n_unique()
    countries = (
        seg.filter(pl.col("country") != "")
           .group_by(["country_key", "country", "iso3"])
           .agg(pl.sum("duration_s"), pl.len().alias("segment_records"),
                pl.col("video_id").n_unique().alias("unique_uploads"),
                pl.col("locality_key").n_unique().alias("localities"))
           .with_columns((pl.col("duration_s") / 3600).round(2).alias("duration_h"),
                         _pct(pl.col("duration_s"), total_s).alias("duration_share_pct"))
           .sort(["duration_s", "country"], descending=[True, False])
           .with_row_index("rank", offset=1)
           .select("rank", "country", "iso3", "localities", "duration_s", "duration_h", "duration_share_pct",
                   "segment_records", "unique_uploads")
    )
    localities = (
        seg.filter(pl.col("locality") != "")
           .group_by(["locality_key", "locality", "state", "country", "iso3", "continent", "lat", "lon"])
           .agg(pl.sum("duration_s"), pl.len().alias("segment_records"),
                pl.col("video_id").n_unique().alias("unique_uploads"))
           .with_columns((pl.col("duration_s") / 3600).round(2).alias("duration_h"),
                         _pct(pl.col("duration_s"), total_s).alias("duration_share_pct"))
           .sort(["duration_s", "country", "locality", "state"], descending=[True, False, False, False])
           .with_row_index("rank", offset=1)
           .select("rank", "locality", "state", "country", "iso3", "continent", "duration_s", "duration_h",
                   "duration_share_pct", "segment_records", "unique_uploads")
    )
    n_localities = localities.height
    share_rows = []
    for percentile in CONCENTRATION_PERCENTILES:
        n_top = math.ceil(n_localities * percentile / 100)
        top_s = int(localities.head(n_top)["duration_s"].sum())
        share_rows.append({"top_locality_percent": percentile, "n_localities": n_top,
                           "duration_h": round(top_s / 3600, 2), "duration_share_pct": _pct_py(top_s, total_s)})
    shares = pl.DataFrame(share_rows)
    one_segment = localities.filter(pl.col("segment_records") == 1).height
    one_upload = localities.filter(pl.col("unique_uploads") == 1).height
    one_segment_pct, one_upload_pct = _pct_py(one_segment, n_localities), _pct_py(one_upload, n_localities)
    top_countries_share = _pct_py(int(countries.head(TOP_COUNTRIES)["duration_s"].sum()), total_s)

    _log_table("I) Country concentration: largest contributors by retained processed duration",
               countries.head(TOP_COUNTRIES))
    _log_table("I) Locality concentration: largest contributors by retained processed duration",
               localities.head(TOP_LOCALITIES))
    _log_table("I) Locality concentration: top 1%, 5%, and 10% by retained hours", shares)
    logger.info(f"[rollups] I) Long-tail locality counts: {one_segment:,} localities ({one_segment_pct:.2f}%) have "
                f"exactly one segment; {one_upload:,} localities ({one_upload_pct:.2f}%) have exactly one unique "
                "upload.")
    logger.info(f"[rollups] I) Concentration totals: countries={countries.height:,}, localities={n_localities:,}, "
                f"segments={n_segments:,}, unique_uploads={n_uploads:,}, duration_h={total_s / 3600:.2f}, "
                f"top_{TOP_COUNTRIES}_countries_duration_share={top_countries_share:.2f}%.")

    out = _concentration_dir()
    try:
        out.mkdir(parents=True, exist_ok=True)
        countries.write_csv(out / "country_concentration.csv")
        countries.head(TOP_COUNTRIES).write_csv(out / "country_concentration_top10.csv")
        localities.write_csv(out / "locality_concentration.csv")
        localities.head(TOP_LOCALITIES).write_csv(out / "locality_concentration_top20.csv")
        shares.write_csv(out / "locality_concentration_shares.csv")
        summary = {
            "total_segment_records": n_segments,
            "total_unique_uploads": n_uploads,
            "total_duration_s": total_s,
            "total_duration_h": round(total_s / 3600, 2),
            "total_countries": countries.height,
            "total_localities": n_localities,
            "top_country_n": TOP_COUNTRIES,
            "top_country_duration_share_pct": top_countries_share,
            "top_locality_n": TOP_LOCALITIES,
            "top_countries": countries.head(TOP_COUNTRIES).to_dicts(),
            "top_localities": localities.head(TOP_LOCALITIES).to_dicts(),
            "locality_concentration": shares.to_dicts(),
            "one_segment_localities": one_segment,
            "one_segment_localities_pct": one_segment_pct,
            "one_upload_localities": one_upload,
            "one_upload_localities_pct": one_upload_pct,
        }
        (out / "concentration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")
        logger.info(f"[rollups] I) Concentration output written to: {out}")
    except Exception as exc:
        logger.warning(f"[rollups] I) Could not write concentration outputs: {exc!r}")


def log_rollups(df_mapping: pl.DataFrame) -> None:
    """Log the dataset rollup tables (sections A-I) and write the concentration tables of section I.

    Segment records are time-of-day label entries ([0, 1] counts twice), so their total equals the Day + Night
    entries. Unique uploads per continent count each upload once, in its canonical continent (_canonical_continents).
    """
    rows, videos, segments = _parse(df_mapping)
    total_segment_records = _log_summary(rows)
    _log_continent_segments(rows)
    canon = _canonical_continents(videos, segments)
    _log_unique_uploads(canon)
    _log_vehicles(videos)
    _log_continent_time_of_day(videos, total_segment_records)
    _log_daynight(videos, canon)
    _log_vehicle_types_per_continent(videos)
    _log_extremes(rows)
    _log_concentration(segments)


# ---------------------------------------------------------------------
# README dataset statistics (regenerated on every run)
# ---------------------------------------------------------------------
README_FILE = os.path.join(common.root_dir, "README.md")
README_FIGURES_URL = "https://htmlpreview.github.io/?https://github.com/crowd-dataset/crowd/blob/main/figures/"
README_STATS_START = "<!-- dataset-stats:start -->"
README_STATS_END = "<!-- dataset-stats:end -->"
README_CONTRIBUTORS_START = "<!-- contributors:start -->"
README_CONTRIBUTORS_END = "<!-- contributors:end -->"
README_YOLO_START = "<!-- yolo-figures:start -->"
README_YOLO_END = "<!-- yolo-figures:end -->"
README_TOP_CITIES = 200

# README column name -> YOLO class ID
README_DETECTIONS = {
    "Persons": YOLO_PERSON,
    "Cars": YOLO_CAR,
    "Bicycles": YOLO_BICYCLE,
    "Motorcycles": YOLO_MOTORCYCLE,
    "Buses": YOLO_BUS,
    "Trucks": YOLO_TRUCK,
}

# per-segment counts of the bbox CSVs, committed so that runs without the CSVs (the GitHub Action) make the detection
# figures and README counts too; written by count_detections.py (see its docstring)
DETECTION_SNAPSHOT = common.data_path("detection_counts.csv.gz")
DETECTION_CACHE = os.path.join("_output", "detection_cache.json.gz")  # per-file counts, not committed


def _count_file(args) -> tuple:
    """Unique tracked objects per class with confidence >= min_conf in one bbox CSV: (path, {yolo id: count}) or
    (path, None) if unreadable."""
    path, min_conf, ids = args  # (a thread pool runs it: polars reads without holding the GIL)
    try:
        # explicit types: a CSV with a header only (nothing detected) would otherwise be read as strings
        df = pl.read_csv(path, columns=["yolo-id", "unique-id", "confidence"],
                         schema_overrides={"yolo-id": pl.Int64, "unique-id": pl.Float64, "confidence": pl.Float64})
    except Exception:
        return path, None
    per_class = (df.filter((pl.col("confidence") >= min_conf) & pl.col("yolo-id").is_in(ids))
                   .group_by("yolo-id").agg(pl.col("unique-id").drop_nulls().n_unique()))
    return path, dict(per_class.iter_rows())


def _segment_lookup(df_mapping: pl.DataFrame) -> dict:
    """Map "{vid}_{start_time}" -> (mapping row id, video, start, processed seconds) for every mapping segment; warns
    about keys of more than one segment (the last one is kept)."""
    segment_to_id, duplicates = {}, []
    for row in df_mapping.select(["id", "videos", "start_time", "end_time"]).iter_rows(named=True):
        for vid, starts, ends in zip(parse_videos(row["videos"]), parse_nested(row["start_time"]),
                                     parse_nested(row["end_time"])):
            for st, et in zip(starts, ends):
                key = f"{vid}_{int(st)}"
                if key in segment_to_id:
                    duplicates.append(key)
                segment_to_id[key] = (row["id"], vid, int(st), processed_segment_duration_seconds(st, et))
    if duplicates:
        logger.warning(f"{len(duplicates):,} segments have the video and start of an earlier segment of the mapping "
                       f"(e.g. {', '.join(duplicates[:5])}); their detections count for the last one only.")
    return segment_to_id


def load_detection_snapshot(df_mapping: pl.DataFrame) -> tuple[pl.DataFrame, int] | None:
    """Load the committed per-segment detection counts, for runs without access to the bbox CSVs (e.g. CI).

    The snapshot is keyed by (video, start) because mapping row ids change when rows are edited or renumbered; ids
    and processed seconds are refreshed from the current mapping, and segments no longer in the mapping are dropped.
    Classes missing from an older snapshot count 0. Returns (counts, number of segments) or None.
    """
    if not os.path.isfile(DETECTION_SNAPSHOT):
        return None
    try:
        with gzip.open(DETECTION_SNAPSHOT, "rb") as f:
            snap = pl.read_csv(f.read(), schema_overrides={"video": pl.Utf8, "start": pl.Int64})
    except Exception as e:
        logger.warning(f"Could not read {DETECTION_SNAPSHOT}: {e}")
        return None
    lookup = pl.DataFrame([(vid, start, i, s) for i, vid, start, s in _segment_lookup(df_mapping).values()],
                          schema={"video": pl.Utf8, "start": pl.Int64, "id": df_mapping.schema["id"],
                                  "detected_seconds": pl.Int64}, orient="row")
    df = (
        snap.with_columns(pl.lit(0).alias(c) for c in README_DETECTIONS if c not in snap.columns)
            .with_row_index("__order")
            .join(lookup, on=["video", "start"], how="inner")
            .sort("__order")
            .select("id", "video", "start", *[pl.col(c).cast(pl.Int64) for c in README_DETECTIONS],
                    "detected_seconds")
    )
    if df.height == 0:
        return None
    logger.info(f"Using the committed detection snapshot {DETECTION_SNAPSHOT} ({df.height:,} segments).")
    return df, df.height


def count_detections(df_mapping: pl.DataFrame, workers: int = 1, write_snapshot: bool = False) -> \
        tuple[pl.DataFrame, int] | None:
    """Count unique tracked objects per segment and YOLO class from the data/*/bbox CSVs.

    Returns (DataFrame with one row per segment with detections: `id`, `video`, `start` (as in
    dataset_figures.segments()), one column per README_DETECTIONS entry and `detected_seconds` (processed footage of
    the segment), number of segments counted (one CSV each)). Without a readable CSV: the committed snapshot
    (load_detection_snapshot), or None with `write_snapshot`. A segment found in several CSVs (e.g., in two data
    folders) is counted once. `workers` threads read the CSVs in parallel; with `write_snapshot`, the counts are saved
    as DETECTION_SNAPSHOT (count_detections.py does so after a complete count).
    """
    files = [f for d in common.get_configs("data") for f in glob.glob(os.path.join(d, "bbox", "*.csv"))]
    if not files:
        return None if write_snapshot else load_detection_snapshot(df_mapping)

    # CSV files are named {vid}_{start_time}_{fps}.csv
    segment_to_id = _segment_lookup(df_mapping)

    min_conf = float(common.get_configs("min_confidence"))
    id_to_class = {v: k for k, v in README_DETECTIONS.items()}
    counts: dict = {}
    todo, seen = [], set()  # one file per segment (a segment found in several data folders is counted once)
    for f in files:
        segment = segment_to_id.get(os.path.basename(f)[:-4].rsplit("_", 1)[0])
        if segment is not None and segment[:3] not in seen:
            seen.add(segment[:3])
            todo.append((f, segment))
    # per-file counts of earlier runs, keyed by file name, size and modification time: a weekly run reads only new or
    # changed CSVs
    cache: dict = {}
    if os.path.exists(DETECTION_CACHE):
        with gzip.open(DETECTION_CACHE, "rt") as f:
            stored = json.load(f)
        if stored.get("min_conf") == min_conf:
            cache = stored["files"]

    def key(path):
        st = os.stat(path)
        return f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"

    keys = [key(f) for f, _ in todo]
    jobs = [(f, min_conf, list(id_to_class)) for (f, _), k in zip(todo, keys) if k not in cache]
    logger.info(f"{len(todo) - len(jobs):,} of {len(todo):,} detection files counted before; reading {len(jobs):,}.")
    # threads, not processes: polars reads without holding the GIL, and forked processes deadlock in polars' own
    # thread pool
    pool = ThreadPoolExecutor(max(workers, 1))
    try:
        results = pool.map(_count_file, jobs)
        for (f, segment), k in tqdm(zip(todo, keys), total=len(todo), desc="Counting detections"):
            if k in cache:
                per_class = {int(i): n for i, n in cache[k].items()}
            else:
                path, per_class = next(results)
                if per_class is None:
                    logger.warning(f"Skipping unreadable detection file {path}")
                    continue
                cache[k] = per_class
            row = dict.fromkeys(README_DETECTIONS, 0) | {"detected_seconds": segment[3]}
            for yolo_id, n in per_class.items():
                row[id_to_class[yolo_id]] = n
            counts[segment[:3]] = row
    finally:
        pool.shutdown(cancel_futures=True)
    if jobs:
        os.makedirs(os.path.dirname(DETECTION_CACHE), exist_ok=True)
        with gzip.open(DETECTION_CACHE + ".tmp", "wt") as f:
            json.dump({"min_conf": min_conf, "files": cache}, f)
        os.replace(DETECTION_CACHE + ".tmp", DETECTION_CACHE)

    if not counts:
        return None if write_snapshot else load_detection_snapshot(df_mapping)
    logger.info(f"{len(counts):,} of {len(segment_to_id):,} segments of the mapping have detections.")
    result = pl.DataFrame(
        [{"id": k[0], "video": k[1], "start": k[2], **v} for k, v in counts.items()],
        schema={"id": df_mapping.schema["id"], "video": pl.Utf8, "start": pl.Int64,
                **dict.fromkeys([*README_DETECTIONS, "detected_seconds"], pl.Int64)},
    )
    if write_snapshot:  # committed, so that runs without the bbox CSVs (CI) still make the detection figures
        buf = io.BytesIO()
        result.select("video", "start", *README_DETECTIONS).sort("video", "start").write_csv(buf)
        with gzip.GzipFile(DETECTION_SNAPSHOT + ".tmp", "wb", compresslevel=9, mtime=0) as f:  # same counts, same file
            f.write(buf.getvalue())
        os.replace(DETECTION_SNAPSHOT + ".tmp", DETECTION_SNAPSHOT)
        logger.info(f"Wrote {DETECTION_SNAPSHOT} ({result.height:,} segments).")
    return result, len(counts)


def _now() -> str:
    """Date and time of an update for the README, in UTC (the clock of the GitHub Action too)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _md_table(df: pl.DataFrame) -> str:
    """Render a DataFrame as a numbered Markdown table, formatting numbers with thousands separators."""
    def fmt(v):
        if v is None:  # e.g. detections of footage that is not analysed yet
            return "–"
        return f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
    lines = ["| # | " + " | ".join(df.columns) + " |", "|" + "---|" * (len(df.columns) + 1)]
    lines += [f"| {i} | " + " | ".join(fmt(v) for v in row) + " |" for i, row in enumerate(df.iter_rows(), 1)]
    return "\n".join(lines)


def _flag_expr() -> pl.Expr:
    """Emoji flag of the `iso3` column (NO_FLAG for codes without one)."""
    return pl.col("iso3").replace_strict(ISO3_TO_FLAG, default=NO_FLAG, return_dtype=pl.Utf8)


def build_readme_stats(df_mapping: pl.DataFrame, detections: tuple[pl.DataFrame, int] | None) -> str:
    """Build the Markdown block with the dataset summary, country table and top-cities table."""
    rows, _, _ = _parse(df_mapping)
    df = rows.with_columns(
        pl.col("footage_time_s").alias("seconds"),
        pl.col("start_time_lol").list.eval(pl.element().list.len()).list.sum().fill_null(0).alias("segments"),
        pl.concat_str([_flag_expr(), pl.col("country")], separator=" ").alias("Country"),
        pl.col("videos_list").list.len().alias("Videos"),
    ).filter(pl.col("seconds") > 0)

    det_cols = []
    if detections is not None:
        df_det, n_segments = detections
        det_cols = list(README_DETECTIONS)
        per_row = df_det.group_by("id").agg(pl.col(det_cols).sum(), pl.sum("detected_seconds"))  # per segment
        df = df.join(per_row, on="id", how="left").with_columns(pl.col([*det_cols, "detected_seconds"]).fill_null(0))

    hours = (pl.col("seconds") / 3600).round(1).alias("Footage (h)")
    # how much of it YOLO has analysed; the counts of places with nothing analysed yet are "–", not 0
    analysed = [(pl.col("detected_seconds") / 3600).round(1).alias("Analysed (h)")] if det_cols else []
    det_counts = [pl.when(pl.col("detected_seconds") > 0).then(pl.col(c)).alias(c) for c in det_cols]
    countries = (
        df.group_by("Country")
          .agg(pl.len().alias("Cities"), pl.sum("Videos"), pl.sum("seconds"),
               *[pl.sum(c) for c in [*det_cols, *(["detected_seconds"] if det_cols else [])]])
          .sort(["seconds", "Country"], descending=[True, False])  # ties (Andorra and Nicaragua) by name
          .select("Country", "Cities", "Videos", hours, *analysed, *det_counts)
    )
    city = pl.concat_str([pl.col("locality"), pl.lit(", ") + pl.col("state")], ignore_nulls=True)
    cities = (
        df.sort("seconds", descending=True).head(README_TOP_CITIES)
          .select(city.alias("City"), "Country", "Videos", hours, *analysed, *det_counts)
    )

    total_s = int(df["seconds"].sum())
    summary = [
        f"- **Footage:** {total_s / 3600:,.1f} hours ({total_s / 86400:,.1f} days)",
        f"- **Videos:** {df['videos_list'].explode().drop_nulls().n_unique():,} unique videos "
        f"split into {int(df['segments'].sum()):,} segments",
        f"- **Localities (cities):** {df.height:,}",
        f"- **Countries and territories:** {df['iso3'].n_unique():,}",
        f"- **Continents:** {df['continent'].n_unique():,}",
    ]
    if det_cols:
        summary.append(f"- **YOLO detections:** {n_segments:,} processed segments (one detection CSV each)")
        summary.append("- **Detected objects** (unique tracked objects, confidence ≥ "
                       f"{common.get_configs('min_confidence')}): "
                       + ", ".join(f"{int(df[c].sum()):,} {c.lower()}" for c in det_cols))
    summary.append(f"- **Last updated:** {_now()}")

    return "\n".join([
        README_STATS_START,
        "<!-- Generated by analysis.py, do not edit by hand. -->",
        "## Dataset overview",
        *summary,
        "",
        f"<details><summary><b>All {countries.height} countries and territories</b></summary>",
        "",
        _md_table(countries),
        "",
        "</details>",
        "",
        f"<details><summary><b>Top {cities.height} cities by footage</b></summary>",
        "",
        _md_table(cities),
        "",
        "</details>",
        README_STATS_END,
    ])


def _replace_block(readme: str, start_marker: str, end_marker: str, content: str) -> str:
    """Replace the text between two markers (markers included) with `content`; unchanged if a marker is missing."""
    start, end = readme.find(start_marker), readme.find(end_marker)
    if start == -1 or end == -1:
        logger.warning(f"README markers {start_marker} / {end_marker} not found; that part is not updated.")
        return readme
    return readme[:start] + content + readme[end + len(end_marker):]


def update_readme(df_mapping: pl.DataFrame, detections: tuple[pl.DataFrame, int] | None,
                  detection_tables: list | None = None, contributors: pl.DataFrame | None = None) -> None:
    """Refresh the generated parts of README.md: dataset statistics and, when detections exist, the detection
    figures and tables (`detection_tables`: (title, DataFrame) from dataset_figures.detection_figures).

    Without detections the statistics are written without object counts and the detection section is left as it is
    (it is only regenerated where the detection CSVs are available).
    """
    with open(README_FILE, encoding="utf-8") as f:
        readme = f.read()
    readme = _replace_block(readme, README_STATS_START, README_STATS_END, build_readme_stats(df_mapping, detections))
    if contributors is not None:
        readme = _replace_block(readme, README_CONTRIBUTORS_START, README_CONTRIBUTORS_END, "\n".join([
            README_CONTRIBUTORS_START, "<!-- Generated by analysis.py, do not edit by hand. -->",
            _md_table(contributors), "",
            f"[![Footage in the dataset over time by who added it](figures/area_contributors_footage.png)]"
            f"({README_FIGURES_URL}area_contributors_footage.html)",
            "Footage in the dataset over time, by who added it (cumulative hours of the videos in the dataset now, by "
            "the date they were added).", README_CONTRIBUTORS_END]))
    if detections is not None:
        figures = [f"[![{caption}](figures/{name}.png)]({README_FIGURES_URL}{name}.html)\n{caption}\n"
                   for name, caption in dataset_figures.DETECTION_FIGURES.items()
                   if os.path.exists(os.path.join(common.root_dir, "figures", name + ".png"))]  # made with enough data
        tables = [f"<details><summary><b>{title}</b></summary>\n\n{_md_table(table)}\n\n</details>\n"
                  for title, table in detection_tables or [] if table.height]
        readme = _replace_block(readme, README_YOLO_START, README_YOLO_END, "\n".join(
            [README_YOLO_START, "<!-- Generated by analysis.py from YOLO detections, do not edit by hand. -->",
             f"Based on the YOLO detections in {detections[1]:,} processed segments, last updated {_now()}.\n",
             "### Tables", *tables, "### Figures", *figures, README_YOLO_END]))
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(readme)
    logger.info("Updated README.md.")


def make_maps(df_mapping: pl.DataFrame, seg: pl.DataFrame) -> None:
    """The city maps (all cities; bubbles by population, videos and footage), the world map with screenshots, the
    country maps of footage per continent and the map of US states (`seg`: dataset_figures.segments())."""
    rows, _, _ = _parse(df_mapping)
    flag = _flag_expr()
    # hover text on the city maps: names and hours, not raw columns and seconds
    df = (
        df_mapping.with_columns(pl.col("state").fill_null("NA"),
                                rows["upload_records"].alias("video_count"),
                                rows["footage_time_s"].alias("total_time"))
                  .sort(["continent", "locality"])  # point order of the maps, as before
                  .with_columns(pl.concat_str([flag, pl.col("locality").cast(pl.Utf8)], separator=" ")
                                  .alias("flag_locality"),
                                pl.concat_str([flag, pl.col("country").cast(pl.Utf8)], separator=" ")
                                  .alias("flag_country"))
                  .with_columns((pl.col("total_time") / 3600).round(1).alias("Footage (hours)"),
                                pl.col("video_count").alias("Videos"),
                                pl.col("population_locality").alias("Population"))
    )
    # the full popups shown on hover, the same as in the other figures
    locality_hover, country_hover = hover.popups(df_mapping, seg, ISO3_TO_FLAG)
    df = df.join(locality_hover, on="id", how="left").sort(["continent", "country"])

    pdf = df.to_pandas()
    maps.mapbox_map(df=pdf, hover_data=["country", "state", "Footage (hours)", "Videos", "Population"],
                    hover_name="flag_locality", marker_size=4, file_name='mapbox_map_all')
    maps.world_map_ss(df=pdf, df_mapping=df_mapping.to_pandas(), show_images=True, hover_data=["total_time"],
                      save_file=False, show_colorbar=True, colorbar_title="Footage (hours)")

    # maps with all cities as bubbles sized by population, number of videos and amount of footage
    df = df.sort(["country", "locality"])
    pdf = df.to_pandas()
    hover_data = ["country", "Footage (hours)", "Videos"]
    for col, title, factor, name in [("population_locality", "Population", 1, "mapbox_map_all_pop"),
                                     ("video_count", "Number of videos", 1, "mapbox_map_all_videos"),
                                     ("total_time", "Footage (hours)", 1 / 3600, "mapbox_map_all_time")]:
        maps.mapbox_map(df=pdf, hover_data=hover_data, hover_name='locality', value_col=col, value_title=title,
                        value_factor=factor, file_name=name)

    # maps of countries in each continent coloured by amount of footage
    # countries split across continents (e.g., Russia) are labelled at the part shown, not the country's centre
    split_countries = (df.group_by("iso3").agg(pl.col("continent").n_unique())
                         .filter(pl.col("continent") > 1)["iso3"].to_list())
    # one colour scale for all continents, so a colour means the same amount of footage on each map
    # countries on two continents are coloured by their whole footage on both maps: the map colours the whole
    # country, and a continent's part alone (Türkiye's 1.9 h in Europe) would be wrong for the rest of it
    whole = df.group_by("iso3").agg((pl.sum("total_time") / 3600).alias("whole_h"))
    per_country = df.group_by(["continent", "iso3"]).agg((pl.sum("total_time") / 3600).alias("h"))
    per_country = per_country.filter(pl.col("h") > 0)
    continent_range = (per_country["h"].min(), max(per_country["h"].max(), whole["whole_h"].max()))
    owner_points = {}
    for iso3, lon, lat in df.select("iso3", "lon", "lat").iter_rows():
        owner_points.setdefault(iso3, []).append((lon, lat))
    for continent in df["continent"].drop_nulls().unique().sort():
        maps.footage_map(df=df.filter(pl.col("continent") == continent)
                              .group_by(["iso3", "flag_country"])
                              .agg((pl.sum("total_time") / 3600).alias("footage_h"),
                                   pl.mean("lat"), pl.mean("lon"))  # anchors for outside labels
                              .join(whole, on="iso3")
                              .with_columns(
                                  pl.when(pl.col("iso3").is_in(split_countries)).then(pl.col("whole_h"))
                                    .otherwise(pl.col("footage_h")).alias("footage_h"),
                                  pl.when(pl.col("iso3").is_in(split_countries))
                                    .then(pl.col("flag_country") + " (whole country)")
                                    .otherwise(pl.col("flag_country")).alias("flag_country"))
                              .join(country_hover, on="iso3", how="left")
                              .to_pandas(),
                         location_col="iso3",
                         locationmode="ISO-3",
                         label_col="flag_country",
                         scope="world",
                         view=maps.CONTINENT_VIEWS[continent],
                         anchor_labels=split_countries,
                         value_range=continent_range,
                         owner_points=owner_points,
                         file_name=f"map_footage_{continent.lower().replace(' ', '_')}")

    # map of US states coloured by amount of footage
    maps.footage_map(df=df.filter(pl.col("iso3") == "USA")
                          .group_by("state")
                          .agg((pl.sum("total_time") / 3600).alias("footage_h"))
                          .to_pandas(),
                     location_col="state",
                     locationmode="USA-states",
                     scope="usa",
                     callouts=maps.US_STATE_CALLOUTS,
                     file_name="map_us_states_footage")


if __name__ == "__main__":
    logger.info("Analysis started.")
    df_mapping = pl.read_csv(common.get_configs("mapping"))
    countries_include = common.get_configs("countries_analyse")
    if countries_include:
        df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_include))
    seg = dataset_figures.segments(df_mapping, VEHICLE_MAP)

    make_maps(df_mapping, seg)

    total_duration = dataset_stats.calculate_total_seconds(df_mapping)
    logger.info(
        f"Duration of videos in seconds: {total_duration}, "
        f"in minutes: {total_duration/60:.2f}, "
        f"in hours: {total_duration/3600:.2f}, "
        f"in days: {total_duration/86400:.2f}, "
        f"in weeks: {total_duration/604800:.2f}, "
        f"in months: {total_duration/2629800:.2f}, "   # average month (30.44 days)
        f"in years: {total_duration/31557600:.2f}."    # average year (365.25 days)
    )
    logger.info("Total number of videos: {}.", dataset_stats.calculate_total_videos(df_mapping))
    _, number, _ = metrics_cache.get_unique_values(df_mapping, "iso3")
    logger.info(f"Total number of countries and territories: {number}.")
    _, number, dup_report = metrics_cache.get_unique_values(df_mapping, ["locality", "state", "iso3"],
                                                            return_duplicates=True)
    logger.info(f"Total number of unique locality+state+iso3 keys: {number}.")
    if dup_report is not None and dup_report.height > 0:
        logger.warning(f"Duplicated keys:\n{dup_report}")

    log_rollups(df_mapping)
    logger.info("Analysis complete.")

    # figures of the dataset, and of what is detected in it when the YOLO detection CSVs are available
    dataset_figures.dataset_figures(df_mapping, seg, ISO3_TO_FLAG)
    # who added the videos: credits for videos new since the last run, then the figure and README table
    contributors_table = dataset_figures.contributor_figures(seg, contributors.update())
    detections = count_detections(df_mapping)
    detection_tables = None
    if detections is None:
        logger.warning("No readable bbox CSVs in configured data folders; figures and README stats based on "
                       "detections are skipped.")
    else:
        detection_tables = dataset_figures.detection_figures(df_mapping, detections[0], list(README_DETECTIONS), seg,
                                                             ISO3_TO_FLAG)
    update_readme(df_mapping, detections, detection_tables, contributors_table)
