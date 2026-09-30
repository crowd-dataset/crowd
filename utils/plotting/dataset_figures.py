"""
Figures describing the dataset and, when YOLO detection CSVs are available, what is detected in it.

Dataset figures are built from the mapping alone, one row per segment (`segments()`). Detection figures take the
per-locality counts of unique tracked objects produced by `analysis.count_detections()`.
"""

import ast
import math
from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import common
from utils.analytics.metrics_cache import MetricsCache
from utils.core.dataset_stats import Dataset_Stats
from utils.plotting import map_labels
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
                    month=int(str(date)[-6:-4]) if date and len(str(date)) >= 7 else None,
                    channel=_item(channels, i),
                ))
    return pl.DataFrame(rows).filter(pl.col("seconds") > 0)


def _country_indicators(df_mapping: pl.DataFrame) -> pl.DataFrame:
    """One row per country with its indicators; zeros are treated as missing."""
    return (df_mapping.group_by("iso3").agg(pl.col(list(INDICATORS)).filter(pl.col(list(INDICATORS)) > 0).first())
                      .with_columns(pl.col("population_country").log10()))


def _vs_indicators(df: pl.DataFrame, value: str, value_title: str, name: str, log_y: bool):
    """Scatter of `value` per country against each indicator, one panel per indicator, each with its Spearman rank
    correlation, the number of countries and a least-squares trend line (on the log scale when `log_y`)."""
    cols = 3
    rows = math.ceil(len(INDICATORS) / cols)
    titles = []
    for col in INDICATORS:
        d = df.select(col, value).drop_nulls().filter(pl.col(value) > 0)
        rho = d.select(pl.corr(col, value, method="spearman")).item() if d.height > 2 else float("nan")
        titles.append(f"{INDICATORS[col]}<br><sup>Spearman ρ = {rho:.2f}, n = {d.height}</sup>")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.07, vertical_spacing=0.16)
    for i, col in enumerate(INDICATORS):
        r, c = i // cols + 1, i % cols + 1
        d = df.select("country", "continent", col, value).drop_nulls([col, value]).filter(pl.col(value) > 0)
        for continent in CONTINENT_ORDER:
            dc = d.filter(pl.col("continent") == continent)
            fig.add_trace(go.Scatter(x=dc[col], y=dc[value], mode="markers", name=continent,
                                     legendgroup=continent, showlegend=i == 0, text=dc["country"],
                                     marker=dict(color=CONTINENT_COLORS[continent], size=7, opacity=0.8),
                                     hovertemplate="%{text}<br>%{x:,.1f}, %{y:,.1f}<extra></extra>"), row=r, col=c)
        if d.height > 2:
            x = d[col].to_numpy()
            y = np.log10(d[value].to_numpy()) if log_y else d[value].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ys = slope * xs + intercept
            fig.add_trace(go.Scatter(x=xs, y=10 ** ys if log_y else ys, mode="lines", showlegend=False,
                                     line=dict(color="#555555", width=2, dash="dash"), hoverinfo="skip"),
                          row=r, col=c)
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text=value_title, col=1)
    _save(_style(fig, legend_title_text=""), name)


MIN_HOURS = 10  # countries with less footage are greyed out on share maps: a share of a few minutes is noise


def _country_map(df: pl.DataFrame, value: str, title: str, scale: str, name: str, fmt: str, log: bool = False):
    """World map of a per-country value; countries with less than MIN_HOURS of footage are grey."""
    ok = df.filter(pl.col("hours") >= MIN_HOURS) if "hours" in df.columns else df
    few = df.filter(pl.col("hours") < MIN_HOURS) if "hours" in df.columns else df.clear()
    d = ok.with_columns((pl.col(value).log10() if log else pl.col(value)).alias("_c"))
    fig = px.choropleth(d.to_pandas(), locations="iso3", color="_c", hover_name="country",
                        hover_data={value: fmt, "_c": False, "iso3": False}, color_continuous_scale=scale,
                        projection="natural earth", labels={value: title})
    if log:
        ticks = [10 ** p for p in range(math.floor(d["_c"].min()), math.ceil(d["_c"].max()) + 1)]
        fig.update_layout(coloraxis_colorbar=dict(tickvals=np.log10(ticks).tolist(),
                                                  ticktext=[f"{t:,g}" for t in ticks]))
    fig.update_layout(coloraxis_colorbar_title=title.replace(" (", "<br>("))
    if few.height:
        fig.add_trace(go.Choropleth(locations=few["iso3"], z=[0] * few.height, locationmode="ISO-3",
                                    colorscale=[[0, "#cfcfcf"], [1, "#cfcfcf"]], showscale=False,
                                    text=few["country"], hovertemplate=f"%{{text}}: under {MIN_HOURS} hours of "
                                    "footage<extra></extra>", marker_line_width=0.5))
        fig.add_annotation(text=f"Grey: under {MIN_HOURS} hours of footage", x=0.01, y=0.02, xref="paper",
                           yref="paper", showarrow=False, font=dict(size=14, color="#666666"))
    _save(_style(fig, margin=dict(l=0, r=0, t=0, b=0)), name)


def _footage_vs_videos(df: pl.DataFrame, label: str, name: str):
    """Footage against number of videos (log-log), coloured by continent, the largest points labelled without
    overlaps (next to the point, or with a leader line when there is no room)."""
    size, margin = (1600, 900), dict(l=90, r=30, t=30, b=80)
    x = np.log10(df["hours"].to_numpy())
    y = np.log10(df["videos"].to_numpy())
    x_range = (x.min() - 0.1, x.max() + 0.35)  # room on the right for the largest points' labels
    y_range = (y.min() - 0.15, y.max() + 0.2)
    fig = go.Figure()
    for continent in CONTINENT_ORDER:
        d = df.filter(pl.col("continent") == continent)
        fig.add_trace(go.Scatter(x=d["hours"], y=d["videos"], mode="markers", name=continent, text=d[label],
                                 marker=dict(color=CONTINENT_COLORS[continent], size=8, opacity=0.75),
                                 hovertemplate="%{text}<br>%{x:,.1f} hours, %{y:,} videos<extra></extra>"))
    # rank rows, not names: same-named places (e.g., two Philadelphias) must not share a label
    top = df.with_row_index("_i").sort("hours", descending=True).head(LABEL_TOP)
    proj = map_labels.AxisProjection(x_range, y_range, size[0] - margin["l"] - margin["r"],
                                     size[1] - margin["t"] - margin["b"])
    items = [dict(code=str(r["_i"]), lines=[r[label]], anchor=(math.log10(r["hours"]), math.log10(r["videos"])),
                  ct=None) for r in top.iter_rows(named=True)]
    placed = map_labels.place_labels(items, proj, {})
    names = {str(r["_i"]): r[label] for r in top.iter_rows(named=True)}
    font = dict(size=11, color="black")
    for code, p in placed.items():
        (ax, ay) = p[1]
        if p[0] == "dot":  # invisible marker so the text sits beside the point like on the maps
            fig.add_trace(go.Scatter(x=[10 ** ax], y=[10 ** ay], mode="markers+text", text=[names[code]],
                                     textposition=p[2], textfont=font, marker=dict(size=8, opacity=0),
                                     showlegend=False, hoverinfo="skip"))
        else:
            (lx, ly), pos = p[2], p[3]
            fig.add_trace(go.Scatter(x=[10 ** ax, 10 ** lx], y=[10 ** ay, 10 ** ly], mode="lines",
                                     line=dict(color="grey", width=1), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[10 ** lx], y=[10 ** ly], mode="text", text=[names[code]],
                                     textposition=pos, textfont=font, showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(type="log", range=x_range, title_text="Footage (hours)", automargin=False)
    fig.update_yaxes(type="log", range=y_range, title_text="Number of videos", automargin=False)
    _save(_style(fig, width=size[0], height=size[1], margin=margin,
                 legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)")), name)


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

    # 3) share of night-time footage per country; countries with too little footage are grey (share unreliable)
    night = (seg.group_by("iso3", "country").agg(hours, (pl.col("seconds").filter(pl.col("night")).sum()
                                                         / pl.sum("seconds") * 100).alias("night_pct")))
    _country_map(night, "night_pct", "Night footage (%)", "Blues", "map_night_share", ":.0f")

    # footage per million inhabitants: coverage relative to country size
    pop = df_mapping.group_by("iso3").agg(pl.col("population_country").filter(pl.col("population_country") > 0)
                                          .first())
    per_capita = (seg.group_by("iso3", "country").agg(hours).join(pop, on="iso3")
                     .with_columns((pl.col("hours") / pl.col("population_country") * 1e6).alias("per_million")))
    _country_map(per_capita, "per_million", "Footage (hours per million inhabitants)", "YlOrRd",
                 "map_footage_per_capita", ":,.1f", log=True)

    # share of each country's footage from its largest channel: where one uploader dominates the data
    by_channel = seg.drop_nulls("channel").group_by("iso3", "country", "channel").agg(pl.sum("seconds"))
    top_channel = (by_channel.group_by("iso3", "country")
                             .agg((pl.max("seconds") / pl.sum("seconds") * 100).alias("top_channel_pct"),
                                  (pl.sum("seconds") / 3600).alias("hours"), pl.len().alias("channels")))
    _country_map(top_channel, "top_channel_pct", "Footage from the largest channel (%)", "Purples",
                 "map_top_channel_share", ":.0f")

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

    # videos uploaded per quarter, by continent; the quarter still in progress is hatched
    videos = (seg.drop_nulls(["year", "month"]).filter(pl.col("year").is_between(2005, 2100),
                                                       pl.col("month").is_between(1, 12))
                 .group_by("video").agg(pl.first("year"), pl.first("month"), pl.first("continent"))
                 .with_columns(pl.date(pl.col("year"), (pl.col("month") - 1) // 3 * 3 + 1, 1).alias("quarter"))
                 .group_by("quarter", "continent").len().sort("quarter"))
    today = date.today()
    current = date(today.year, (today.month - 1) // 3 * 3 + 1, 1)
    fig = px.bar(videos.to_pandas(), x="quarter", y="len", color="continent", **CONTINENT_STYLE,
                 pattern_shape=[q == current for q in videos["quarter"]], pattern_shape_map={True: "/", False: ""},
                 labels={"quarter": "Quarter of upload", "len": "Number of videos"})
    # px makes one trace per continent and hatching; show each continent once in the legend
    fig.for_each_trace(lambda t: t.update(name=t.name.split(", ")[0], legendgroup=t.name.split(", ")[0],
                                          showlegend=t.name.endswith("False")))
    fig.update_layout(bargap=0.1)
    fig.add_annotation(text="Hatched: current quarter, not complete yet", x=0.01, y=0.98, xref="paper",
                       yref="paper", showarrow=False, xanchor="left", font=dict(size=14, color="#666666"))
    _save(_style(fig, legend_title_text=""), "hist_months")

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
