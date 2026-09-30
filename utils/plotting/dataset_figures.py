"""
Figures describing the dataset and, when YOLO detection CSVs are available, what is detected in it.

Dataset figures are built from the mapping alone, one row per segment (`segments()`). Detection figures take the
per-locality counts of unique tracked objects produced by `analysis.count_detections()`.
"""

import ast
import math

import numpy as np
import plotly.express as px
import polars as pl

import common
from utils.analytics.metrics_cache import MetricsCache
from utils.core.dataset_stats import Dataset_Stats
from utils.plotting.constants import CONTINENT_COLORS
from utils.plotting.io import IO

io = IO()

# Country-level columns of the mapping compared with the amount of footage and detection rates (column -> name).
INDICATORS = {
    "population_country": "Population (log10)",
    "traffic_mortality": "Road traffic deaths per 100k",
    "gini": "Gini index",
    "med_age": "Median age (years)",
    "literacy_rate": "Literacy rate (%)",
}
CONTINENT_ORDER = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
# the same order and colour for each continent in every figure
CONTINENT_STYLE = dict(category_orders={"continent": CONTINENT_ORDER}, color_discrete_map=CONTINENT_COLORS)
LABEL_TOP = 12  # scatter plots label only the largest points; the rest show on hover


def _style(fig, **layout):
    # log axes: label powers of ten only
    fig.for_each_xaxis(lambda a: a.update(dtick=1) if a.type == "log" else None)
    fig.for_each_yaxis(lambda a: a.update(dtick=1) if a.type == "log" else None)
    fig.update_layout(template=common.get_configs("plotly_template"),
                      font=dict(family=common.get_configs("font_family"), size=common.get_configs("font_size")),
                      **layout)
    return fig


def _save(fig, name):
    io.save_plotly_figure(fig, name, save_final=True)


def _literal_list(cell) -> list:
    try:
        value = ast.literal_eval(cell) if isinstance(cell, str) else None
    except (ValueError, SyntaxError):
        return []
    return value if isinstance(value, list) else []


def _item(lst, i):
    return lst[i] if i < len(lst) else None


def segments(df_mapping: pl.DataFrame, vehicle_map: dict) -> pl.DataFrame:
    """One row per segment: locality, video, processed seconds, night flag, vehicle type, upload year, channel."""
    rows = []
    for r in df_mapping.iter_rows(named=True):
        vids = MetricsCache._parse_videos_cell(r["videos"])
        starts = Dataset_Stats._parse_nested_list(r["start_time"])
        ends = Dataset_Stats._parse_nested_list(r["end_time"])
        tods = Dataset_Stats._parse_nested_list(r["time_of_day"])
        vehicles = _literal_list(r["vehicle_type"])
        dates = _literal_list(r["upload_date"])
        channels = MetricsCache._parse_videos_cell(r["channel"])
        for i, vid in enumerate(vids):
            date = _item(dates, i)
            for j, (start, end) in enumerate(zip(_item(starts, i) or [], _item(ends, i) or [])):
                rows.append(dict(
                    id=r["id"], locality=r["locality"], country=r["country"], iso3=r["iso3"],
                    continent=r["continent"], video=vid, start=int(start),
                    seconds=Dataset_Stats._processed_segment_duration_seconds(start, end),
                    night=_item(_item(tods, i) or [], j) == 1,
                    vehicle=vehicle_map.get(_item(vehicles, i)),
                    year=int(str(date)[-4:]) if date else None,  # upload dates are stored as DMMYYYY / DDMMYYYY
                    channel=_item(channels, i),
                ))
    return pl.DataFrame(rows).filter(pl.col("seconds") > 0)


def _country_indicators(df_mapping: pl.DataFrame) -> pl.DataFrame:
    """One row per country with its indicators; zeros are treated as missing."""
    return (df_mapping.group_by("iso3").agg(pl.col(list(INDICATORS)).filter(pl.col(list(INDICATORS)) > 0).first())
                      .with_columns(pl.col("population_country").log10()))


def _vs_indicators(df: pl.DataFrame, value: str, value_title: str, name: str, log_y: bool):
    """Scatter of `value` per country against each indicator, one panel per indicator."""
    long = df.unpivot(index=["country", "continent", value], on=list(INDICATORS), variable_name="indicator",
                      value_name="indicator_value").drop_nulls("indicator_value")
    long = long.with_columns(pl.col("indicator").replace(INDICATORS))
    fig = px.scatter(long.to_pandas(), x="indicator_value", y=value, color="continent", facet_col="indicator",
                     facet_col_wrap=3, facet_col_spacing=0.06, facet_row_spacing=0.12, log_y=log_y,
                     hover_name="country", **CONTINENT_STYLE, labels={value: value_title})
    fig.update_xaxes(matches=None, showticklabels=True, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _save(_style(fig, legend_title_text=""), name)


def _footage_vs_videos(df: pl.DataFrame, label: str, name: str):
    """Footage against number of videos (log-log), coloured by continent, the largest points labelled."""
    # rank rows, not names: same-named places (e.g., two Philadelphias) must not share a label
    df = df.with_columns(pl.when(pl.col("hours").rank("ordinal", descending=True) <= LABEL_TOP)
                         .then(pl.col(label)).otherwise(pl.lit("")).alias("text"))
    # SVG rendering: with many points plotly switches to WebGL, which cannot draw emoji flags
    fig = px.scatter(df.to_pandas(), x="hours", y="videos", color="continent", text="text", log_x=True, log_y=True,
                     render_mode="svg",
                     hover_name=label, hover_data={"text": False, "hours": ":,.1f"}, **CONTINENT_STYLE,
                     labels={"hours": "Footage (hours)", "videos": "Number of videos"})
    fig.update_traces(textposition="top center", textfont_size=12, marker=dict(size=8, opacity=0.75))
    _save(_style(fig, legend_title_text=""), name)


def dataset_figures(df_mapping: pl.DataFrame, seg: pl.DataFrame, flags: dict) -> None:
    """Figures based on the mapping only. `flags` maps ISO3 codes to emoji flags for labels."""
    hours = (pl.sum("seconds") / 3600).alias("hours")
    flag = pl.col("iso3").replace_strict(flags, default="🏳️", return_dtype=pl.Utf8)

    # footage against number of videos, per locality and per country
    city = (seg.group_by("id").agg(hours, pl.col("video").n_unique().alias("videos"))
               .join(df_mapping.select("id", "locality", "iso3", "continent"), on="id")
               .with_columns(pl.concat_str([flag, pl.col("locality")], separator=" ").alias("name")))
    _footage_vs_videos(city, "name", "scatter_all_total_time-video_count")
    country = (seg.group_by("iso3").agg(hours, pl.col("video").n_unique().alias("videos"), pl.first("continent"))
                  .with_columns(pl.concat_str([flag, pl.col("iso3")], separator=" ").alias("name")))
    _footage_vs_videos(country, "name", "scatter_all_country_total_time-video_count")

    # day and night footage per continent
    tod = (seg.with_columns(pl.when(pl.col("night")).then(pl.lit("Night")).otherwise(pl.lit("Day")).alias("time"))
              .group_by("continent", "time").agg(hours))
    fig = px.bar(tod.to_pandas(), x="continent", y="hours", color="time",
                 category_orders={"continent": CONTINENT_ORDER, "time": ["Day", "Night"]},
                 color_discrete_map={"Day": "#E69F00", "Night": "#0072B2"},
                 labels={"hours": "Footage (hours)", "continent": "", "time": ""})
    _save(_style(fig), "bar_continent_time_of_day")

    # vehicle the footage is filmed from: cars are ~90% of footage, so a log axis with the share written on each bar
    veh = (seg.drop_nulls("vehicle").group_by("vehicle")
              .agg(hours, (pl.col("seconds").filter(pl.col("night")).sum() / pl.sum("seconds") * 100).alias("night"))
              .with_columns((pl.col("hours") / pl.col("hours").sum() * 100).alias("share"))
              .sort("hours"))
    veh = veh.with_columns(pl.format("{}% of footage, {}% at night", pl.col("share").round(1),
                                     pl.col("night").round(0).cast(pl.Int64)).alias("text"))
    fig = px.bar(veh.to_pandas(), y="vehicle", x="hours", orientation="h", text="text", log_x=True,
                 labels={"hours": "Footage (hours, log scale)", "vehicle": ""})
    fig.update_traces(textposition="outside", marker_color="#0072B2", cliponaxis=False)
    _save(_style(fig, margin=dict(r=260)), "bar_vehicle_type_time_of_day")

    # 1) footage against city population: which large cities are under-sampled
    city = (seg.group_by("id").agg(hours)
               .join(df_mapping.select("id", "locality", "country", "continent", "population_locality"), on="id")
               .filter(pl.col("population_locality") > 0))
    fig = px.scatter(city.to_pandas(), x="population_locality", y="hours", color="continent", log_x=True, log_y=True,
                     hover_name="locality", hover_data=["country"], **CONTINENT_STYLE,
                     labels={"population_locality": "Population of locality", "hours": "Footage (hours)"})
    _save(_style(fig, legend_title_text=""), "scatter_population_footage")

    # 2) footage against country indicators: is the dataset biased towards some kinds of countries
    # countries split across continents (e.g., Russia) count as one country, shown with their first continent
    country = (seg.group_by("iso3", "country").agg(hours, pl.first("continent"))
                  .join(_country_indicators(df_mapping), on="iso3", how="left"))
    _vs_indicators(country, "hours", "Footage (hours)", "scatter_indicators_footage", log_y=True)

    # 3) share of night-time footage per country
    night = (seg.group_by("iso3", "country").agg((pl.col("seconds").filter(pl.col("night")).sum()
                                                  / pl.sum("seconds") * 100).alias("night_pct")))
    fig = px.choropleth(night.to_pandas(), locations="iso3", color="night_pct", hover_name="country",
                        color_continuous_scale="Blues", projection="natural earth",
                        labels={"night_pct": "Night footage (%)"})
    _save(_style(fig, margin=dict(l=0, r=0, t=0, b=0)), "map_night_share")

    # 4) type of vehicle the footage is filmed from, per continent (share of footage)
    veh = (seg.drop_nulls("vehicle").group_by("continent", "vehicle").agg(hours)
              .with_columns((pl.col("hours") / pl.col("hours").sum().over("continent") * 100).alias("share")))
    fig = px.bar(veh.sort("hours", descending=True).to_pandas(), x="continent", y="share", color="vehicle",
                 **CONTINENT_STYLE,
                 labels={"share": "Share of footage (%)", "continent": "", "vehicle": "Type of vehicle"})
    _save(_style(fig), "bar_vehicle_type_continent")

    # 5) upload year per continent
    years = (seg.drop_nulls("year").filter(pl.col("year").is_between(2005, 2100))  # YouTube started in 2005
                .group_by("year", "continent").agg(hours))
    fig = px.bar(years.sort("year").to_pandas(), x="year", y="hours", color="continent",
                 **CONTINENT_STYLE,
                 labels={"year": "Year of upload", "hours": "Footage (hours)"})
    _save(_style(fig, legend_title_text=""), "bar_upload_year_continent")

    # 6) how concentrated the footage is in a few YouTube channels
    chan = (seg.drop_nulls("channel").group_by("channel").agg(hours).sort("hours", descending=True)
               .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"),
                             (pl.col("hours").cum_sum() / pl.col("hours").sum() * 100).alias("cumulative")))
    fig = px.line(chan.to_pandas(), x="rank", y="cumulative", log_x=True, hover_data=["channel", "hours"],
                  labels={"rank": "Number of channels (largest first)", "cumulative": "Share of footage (%)"})
    for n in (10, 100):
        if chan.height >= n:
            share = chan["cumulative"][n - 1]
            fig.add_annotation(x=math.log10(n), y=share, text=f"top {n}: {share:.0f}%", showarrow=True, ax=40, ay=30)
    _save(_style(fig), "line_channel_concentration")

    # 7) length of segments
    minutes = seg.select((pl.col("seconds") / 60).alias("minutes"))
    edges = np.logspace(np.log10(max(minutes["minutes"].min(), 0.1)), np.log10(minutes["minutes"].max()), 40)
    counts, _ = np.histogram(minutes["minutes"], bins=edges)
    fig = px.bar(x=np.sqrt(edges[:-1] * edges[1:]), y=counts, log_x=True,
                 labels={"x": "Length of segment (minutes)", "y": "Number of segments"})
    fig.update_traces(width=np.diff(edges))
    _save(_style(fig), "hist_segment_length")


# Figures based on YOLO detections, with their README captions, in the order they appear in the README.
DETECTION_FIGURES = {
    "map_pedestrians_per_minute": "Pedestrians per minute of footage per country (unique tracked persons).",
    "box_pedestrians_per_minute_continent": "Pedestrians per minute of footage per locality, by continent.",
    "bar_road_user_mix": "Mix of detected road users in the 30 countries with the most analysed footage.",
    "scatter_indicators_pedestrians": "Pedestrians per minute of footage per country against country indicators.",
}


def detection_figures(df_mapping: pl.DataFrame, det: pl.DataFrame, classes: list) -> None:
    """
    Figures based on YOLO detections.

    Args:
        df_mapping: The mapping.
        det: Per mapping row: `id`, one count column per class in `classes` and `detected_seconds` (footage covered
            by the detection CSVs).
        classes: Detection count columns, the first one being persons.
    """
    person = classes[0]
    det = det.join(df_mapping.select("id", "locality", "country", "iso3", "continent"), on="id").filter(
        pl.col("detected_seconds") > 0)
    per_min = (pl.col(person) / (pl.col("detected_seconds") / 60)).alias("per_minute")

    country = det.group_by("iso3", "country").agg(pl.col(classes + ["detected_seconds"]).sum(), pl.first("continent"))
    fig = px.choropleth(country.with_columns(per_min).to_pandas(), locations="iso3", color="per_minute",
                        hover_name="country", color_continuous_scale="YlOrRd", projection="natural earth",
                        labels={"per_minute": "Pedestrians per minute"})
    _save(_style(fig, margin=dict(l=0, r=0, t=0, b=0)), "map_pedestrians_per_minute")

    fig = px.box(det.with_columns(per_min).to_pandas(), x="continent", y="per_minute", color="continent",
                 points="outliers",
                 hover_name="locality", **CONTINENT_STYLE,
                 labels={"per_minute": "Pedestrians per minute", "continent": ""})
    _save(_style(fig), "box_pedestrians_per_minute_continent")

    top = country.sort("detected_seconds", descending=True).head(30)
    mix = (top.unpivot(index="country", on=classes, variable_name="Road user", value_name="count")
              .with_columns((pl.col("count") / pl.col("count").sum().over("country") * 100).alias("share")))
    fig = px.bar(mix.to_pandas(), y="country", x="share", color="Road user", orientation="h",
                 category_orders={"country": top["country"].to_list()},
                 labels={"share": "Share of detected road users (%)", "country": ""})
    _save(_style(fig, height=900), "bar_road_user_mix")

    rates = country.with_columns(per_min).join(_country_indicators(df_mapping), on="iso3", how="left")
    _vs_indicators(rates, "per_minute", "Pedestrians per minute", "scatter_indicators_pedestrians", log_y=False)


if __name__ == "__main__":
    # self-check of the segment table on a two-video locality
    m = pl.DataFrame(dict(
        id=[1], locality=["A"], country=["B"], iso3=["BBB"], continent=["Europe"], videos=["[v1,v-2]"],
        start_time=["[[0,100],[5]]"], end_time=["[[61,160],[35]]"], time_of_day=["[[0,1],[0]]"],
        vehicle_type=["[0,1]"], upload_date=["[3042024,15012019]"], channel=["[c1,c2]"]))
    s = segments(m, {0: "Car", 1: "Bus"})
    assert s["seconds"].to_list() == [60, 59, 29], s
    assert s["night"].to_list() == [False, True, False] and s["vehicle"].to_list() == ["Car", "Car", "Bus"]
    assert s["year"].to_list() == [2024, 2024, 2019] and s["channel"].to_list() == ["c1", "c1", "c2"]
    print("ok")
