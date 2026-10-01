"""
Figures describing the dataset and, when YOLO detection CSVs are available, what is detected in it.

Dataset figures are built from the mapping alone, one row per segment (`segments()`). Detection figures take the
per-locality counts of unique tracked objects produced by `analysis.count_detections()`.
"""

import ast
import math
import os
import urllib.request
from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from PIL import Image
from plotly.subplots import make_subplots

import common
from utils.analytics.metrics_cache import MetricsCache
from utils.core.dataset_stats import Dataset_Stats
from utils.plotting import hover, map_labels
from utils.plotting.constants import CONTINENT_COLORS, MAP_TOP_SHARE, colorbar_top
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


def _save(fig, name, post_script=None, save_eps=True):
    # static images at the figure's own size where it sets one (e.g., a taller scatter), else 1600x900
    io.save_plotly_figure(fig, name, width=fig.layout.width or 1600, height=fig.layout.height or 900,
                          save_final=True, post_script=post_script, save_eps=save_eps)


def _hover_args(d: pl.DataFrame, line: str = "") -> dict:
    """Hover showing the shared locality or country popup (column `hover`, see hover.py), after an optional line
    with the figure's own value (plotly template syntax)."""
    return dict(hovertext=d["hover"], hovertemplate=(f"<b>{line}</b><br>" if line else "") + hover.TEMPLATE)


# Interactive scatter plots: when zoomed in to at most ZOOM_LABELS points, label every point in view (a hidden
# trace with all labels) instead of only the largest ones.
ZOOM_LABELS = 150
ZOOM_LABELS_JS = """
var gd = document.getElementById('{plot_id}');
function zoomLabels() {
  var all = gd.data.findIndex(function (t) { return t.meta === 'zoom-labels'; });
  if (all < 0) return;
  var t = gd.data[all], xr = gd._fullLayout.xaxis.range, yr = gd._fullLayout.yaxis.range, n = 0;
  for (var i = 0; i < t.x.length; i++) {
    var x = gd._fullLayout.xaxis.type === 'log' ? Math.log10(t.x[i]) : t.x[i], y = Math.log10(t.y[i]);
    if (x >= xr[0] && x <= xr[1] && y >= yr[0] && y <= yr[1]) n++;
  }
  var show = n <= %d;
  if (!!t.visible === show) return;
  var top = [];
  gd.data.forEach(function (d, i) { if (d.meta === 'top-labels') top.push(i); });
  Plotly.restyle(gd, {visible: show}, [all]);
  if (top.length) Plotly.restyle(gd, {visible: !show}, top);
}
gd.on('plotly_relayout', zoomLabels);
""" % ZOOM_LABELS


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


def _bar_totals(fig, df: pl.DataFrame, x: str, y: str, fmt: str = "{:,.0f}", vertical: bool = False,
                plot_width: float = 1400):
    """Write each stacked bar's total above it. Vertical labels (narrow bars) are as large as one bar's width allows
    (`plot_width`: width of the plot area in pixels)."""
    totals = df.group_by(x).agg(pl.sum(y)).sort(x)
    size = min(20, int(plot_width / totals.height * 0.85)) if vertical else 16
    for xv, yv in totals.iter_rows():
        fig.add_annotation(x=xv, y=yv, text=fmt.format(yv), showarrow=False, textangle=-90 if vertical else 0,
                           yanchor="bottom", yshift=2, font=dict(size=size))
    # room above the tallest bar for its label (about 4.5 characters of the label's font size)
    fig.update_yaxes(range=[0, totals[y].max() * ((1.08 + size * 0.012) if vertical else 1.08)])


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
        d = df.select("country", "continent", "hover", col, value).drop_nulls([col, value]).filter(pl.col(value) > 0)
        for continent in CONTINENT_ORDER:
            dc = d.filter(pl.col("continent") == continent)
            fig.add_trace(go.Scatter(x=dc[col], y=dc[value], mode="markers", name=continent,
                                     legendgroup=continent, showlegend=i == 0,
                                     marker=dict(color=CONTINENT_COLORS[continent], size=7, opacity=0.8),
                                     **_hover_args(dc, f"{INDICATORS[col]}: %{{x:,.1f}} · {value_title}: "
                                                       "%{y:,.2f}")), row=r, col=c)
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
    fig = px.choropleth(d.to_pandas(), locations="iso3", color="_c", hover_name="hover",
                        color_continuous_scale=scale, projection="natural earth", labels={value: title})
    fig.update_traces(hovertemplate=hover.TEMPLATE)
    if log:
        lo, hi = d["_c"].min(), d["_c"].max()
        # powers of ten, and 2 and 5 times them when the values span only a few of those
        steps = (1,) if hi - lo > 2 else (1, 2, 5)
        ticks = [s * 10 ** p for p in range(math.floor(lo), math.ceil(hi) + 1) for s in steps
                 if lo <= math.log10(s * 10 ** p) <= hi]
        fig.update_layout(coloraxis_colorbar=dict(tickvals=np.log10(ticks).tolist(),
                                                  ticktext=[f"{t:,g}" for t in ticks]))
    fig.update_layout(coloraxis_colorbar=colorbar_top(title))
    fig.update_geos(domain=dict(x=[0, 1], y=[0, MAP_TOP_SHARE]))
    if few.height:
        fig.add_trace(go.Choropleth(locations=few["iso3"], z=[0] * few.height, locationmode="ISO-3",
                                    colorscale=[[0, "#cfcfcf"], [1, "#cfcfcf"]], showscale=False,
                                    marker_line_width=0.5,
                                    **_hover_args(few, f"Under {MIN_HOURS} hours of footage: grey on this map")))
        fig.add_annotation(text=f"Grey: under {MIN_HOURS} hours of footage", x=0.01, y=0.02, xref="paper",
                           yref="paper", showarrow=False, font=dict(size=14, color="#666666"))
    _save(_style(fig, margin=dict(l=0, r=0, t=10, b=0)), name)


def _labelled_scatter(df: pl.DataFrame, x_col: str, y_col: str, label: str, labelled: pl.Expr, x_title: str,
                      y_title: str, name: str, size=(1600, 900), emphasis: pl.Expr = pl.lit(True), note: str = "",
                      log_x: bool = True):
    """Log-log scatter (linear x if not `log_x`) coloured by continent. Rows where `labelled` is true get a label
    placed without overlaps (next to the point, or with a leader line when there is no room); in the HTML, zooming in
    labels every point.
    Labels of rows where `emphasis` is false are smaller and grey, so the eye goes to the emphasised ones first."""
    margin = dict(l=90, r=30, t=30, b=80)
    fx = math.log10 if log_x else float  # data -> axis units (labels are placed in axis units)
    ix = (lambda v: 10 ** v) if log_x else float  # axis units -> data
    x = np.log10(df[x_col].to_numpy()) if log_x else df[x_col].to_numpy().astype(float)
    y = np.log10(df[y_col].to_numpy())
    # room on the right for the largest points' labels: in axis units, a third of a decade on log axes
    x_range = (x.min() - 0.1, x.max() + 0.35) if log_x else (x.min() - 0.03 * np.ptp(x), x.max() + 0.12 * np.ptp(x))
    y_range = (y.min() - 0.15, y.max() + 0.2)
    fig = go.Figure()
    for continent in CONTINENT_ORDER:
        d = df.filter(pl.col("continent") == continent)
        fig.add_trace(go.Scatter(x=d[x_col], y=d[y_col], mode="markers", name=continent,
                                 marker=dict(color=CONTINENT_COLORS[continent], size=8, opacity=0.75),
                                 **_hover_args(d)))
    # select rows, not names: same-named places (e.g., two Philadelphias) must not share a label
    top = df.with_row_index("_i").with_columns(emphasis.alias("_emphasis")).filter(labelled)
    proj = map_labels.AxisProjection(x_range, y_range, size[0] - margin["l"] - margin["r"],
                                     size[1] - margin["t"] - margin["b"])
    items = [dict(code=str(r["_i"]), lines=[r[label]], anchor=(fx(r[x_col]), math.log10(r[y_col])),
                  ct=None, scale=1 if r["_emphasis"] else 9 / 11)
             for r in top.sort(y_col, descending=True).iter_rows(named=True)]
    # labels with no free spot are left out of the static image; hover and zoom in the HTML still show them
    placed = map_labels.place_labels(items, proj, {}, stack_clusters=False, drop_unplaced=True)
    names = {str(r["_i"]): r[label] for r in top.iter_rows(named=True)}
    emphasised = {str(r["_i"]) for r in top.iter_rows(named=True) if r["_emphasis"]}
    strong, faint = dict(size=11, color="black"), dict(size=9, color="#8a8a8a")
    for code, p in placed.items():
        (ax, ay) = p[1]
        font = strong if code in emphasised else faint
        if p[0] == "dot":  # invisible marker so the text sits beside the point like on the maps
            fig.add_trace(go.Scatter(x=[ix(ax)], y=[10 ** ay], mode="markers+text", text=[names[code]],
                                     textposition=p[2], textfont=font, marker=dict(size=8, opacity=0),
                                     showlegend=False, hoverinfo="skip", meta="top-labels"))
        else:
            (lx, ly), pos = p[2], p[3]
            fig.add_trace(go.Scatter(x=[ix(ax), ix(lx)], y=[10 ** ay, 10 ** ly], mode="lines",
                                     line=dict(color="#bbbbbb" if font is faint else "grey", width=1),
                                     showlegend=False, hoverinfo="skip",
                                     meta="top-labels"))
            fig.add_trace(go.Scatter(x=[ix(lx)], y=[10 ** ly], mode="text", text=[names[code]],
                                     textposition=pos, textfont=font, showlegend=False, hoverinfo="skip",
                                     meta="top-labels"))
    # every point's label, shown in the HTML only when zoomed in (see ZOOM_LABELS_JS)
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="text", text=df[label], textposition="top center",
                             textfont=strong, visible=False, showlegend=False, hoverinfo="skip", meta="zoom-labels"))
    fig.update_xaxes(type="log" if log_x else "linear", range=x_range, title_text=x_title, automargin=False)
    fig.update_yaxes(type="log", range=y_range, title_text=y_title, automargin=False)
    if note:  # e.g., the correlation, in the bottom-right corner
        fig.add_annotation(text=note, x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right",
                           showarrow=False, font=dict(size=16), bgcolor="rgba(255,255,255,0.8)")
    _save(_style(fig, width=size[0], height=size[1], margin=margin,
                 legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)")), name, post_script=ZOOM_LABELS_JS)


def dataset_figures(df_mapping: pl.DataFrame, seg: pl.DataFrame, flags: dict) -> None:
    """Figures based on the mapping only. `flags` maps ISO3 codes to emoji flags for labels."""
    hours = (pl.sum("seconds") / 3600).alias("hours")
    flag = pl.col("iso3").replace_strict(flags, default="🏳️", return_dtype=pl.Utf8)
    # the popups shown on hover for each locality and country, the same in every figure
    loc_hover, cty_hover = hover.popups(df_mapping, seg, flags)

    # footage against number of videos, per locality and per country
    city = (seg.group_by("id").agg(hours, pl.col("video").n_unique().alias("videos"))
               .join(df_mapping.select("id", "locality", "iso3", "continent"), on="id").join(loc_hover, on="id")
               .with_columns(pl.concat_str([flag, pl.col("locality")], separator=" ").alias("name")))
    # the 40 localities with most footage labelled, the top ones in larger black text; zoom in the HTML for more
    rank = pl.col("hours").rank("ordinal", descending=True)
    _labelled_scatter(city, "hours", "videos", "name", rank <= 40, "Footage (hours)", "Number of videos",
                      "scatter_all_total_time-video_count", emphasis=rank <= LABEL_TOP)
    # every country labelled with flag and ISO3 code (tall, so the flags fit); the top 30 by footage in black
    country = (seg.group_by("iso3").agg(hours, pl.col("video").n_unique().alias("videos"), pl.first("continent"))
                  .with_columns(pl.concat_str([flag, pl.col("iso3")], separator=" ").alias("name"))
                  .join(cty_hover, on="iso3"))
    _labelled_scatter(country, "hours", "videos", "name", pl.lit(True), "Footage (hours)", "Number of videos",
                      "scatter_all_country_total_time-video_count", size=(1600, 1700),
                      emphasis=pl.col("hours").rank("ordinal", descending=True) <= 30)

    # day and night footage per continent
    tod = (seg.with_columns(pl.when(pl.col("night")).then(pl.lit("Night")).otherwise(pl.lit("Day")).alias("time"))
              .group_by("continent", "time").agg(hours))
    fig = px.bar(tod.to_pandas(), x="continent", y="hours", color="time",
                 category_orders={"continent": CONTINENT_ORDER, "time": ["Day", "Night"]},
                 color_discrete_map={"Day": "#E69F00", "Night": "#0072B2"},
                 labels={"hours": "Footage (hours)", "continent": "", "time": ""})
    _bar_totals(fig, tod, "continent", "hours")
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
    # labelled: the localities with the most footage and the largest ones by population
    city = (seg.group_by("id").agg(hours)
               .join(df_mapping.select("id", "locality", "iso3", "continent", "population_locality"), on="id")
               .join(loc_hover, on="id").filter(pl.col("population_locality") > 0)
               .with_columns(pl.concat_str([flag, pl.col("locality")], separator=" ").alias("name")))
    notable = ((pl.col("hours").rank("ordinal", descending=True) <= 8)
               | (pl.col("population_locality").rank("ordinal", descending=True) <= 8))
    _labelled_scatter(city, "population_locality", "hours", "name", notable, "Population of locality",
                      "Footage (hours)", "scatter_population_footage")

    # footage against the economy (GMP, known for ~200 large cities) and the congestion of each locality: do rich or
    # congested cities dominate the data? The traffic index (how much slower than free flow traffic is, %) is one
    # TomTom reading per locality; shown where it is above 0, as older zeros also stand for failed requests
    city = (seg.group_by("id").agg(hours)
               .join(df_mapping.select("id", "locality", "iso3", "continent", "gmp", "traffic_index"), on="id")
               .join(loc_hover, on="id")
               .with_columns(pl.concat_str([flag, pl.col("locality")], separator=" ").alias("name")))
    for col, x_title, name, top, log_x in [
            ("gmp", "Gross metropolitan product (billion USD)", "scatter_gmp_footage", 15, True),
            ("traffic_index", "Traffic index: how much slower than free flow (%, TomTom; localities above 0)",
             "scatter_traffic_index_footage", 12, False)]:
        d = city.filter(pl.col(col) > 0)
        rho = d.select(pl.corr(col, "hours", method="spearman")).item()
        notable = ((pl.col("hours").rank("ordinal", descending=True) <= top)
                   | (pl.col(col).rank("ordinal", descending=True) <= top))
        _labelled_scatter(d, col, "hours", "name", notable, x_title, "Footage (hours)", name, log_x=log_x,
                          note=f"Spearman ρ = {rho:.2f}, n = {d.height:,} localities")

    # 2) footage against country indicators: is the dataset biased towards some kinds of countries
    # countries split across continents (e.g., Russia) count as one country, shown with their first continent
    country = (seg.group_by("iso3", "country").agg(hours, pl.first("continent"))
                  .join(_country_indicators(df_mapping), on="iso3", how="left").join(cty_hover, on="iso3"))
    _vs_indicators(country, "hours", "Footage (hours)", "scatter_indicators_footage", log_y=True)

    # 3) share of night-time footage per country; countries with too little footage are grey (share unreliable)
    night = (seg.group_by("iso3", "country").agg(hours, (pl.col("seconds").filter(pl.col("night")).sum()
                                                         / pl.sum("seconds") * 100).alias("night_pct")))
    _country_map(night.join(cty_hover, on="iso3"), "night_pct", "Night footage (%)", "Blues", "map_night_share",
                 ":.0f")

    # footage per million inhabitants: coverage relative to country size
    pop = df_mapping.group_by("iso3").agg(pl.col("population_country").filter(pl.col("population_country") > 0)
                                          .first())
    per_capita = (seg.group_by("iso3", "country").agg(hours).join(pop, on="iso3")
                     .with_columns((pl.col("hours") / pl.col("population_country") * 1e6).alias("per_million")))
    _country_map(per_capita.join(cty_hover, on="iso3"), "per_million", "Hours per million people", "YlOrRd",
                 "map_footage_per_capita", ":,.1f", log=True)

    # share of each country's footage from its largest channel: where one uploader dominates the data
    by_channel = seg.drop_nulls("channel").group_by("iso3", "country", "channel").agg(pl.sum("seconds"))
    top_channel = (by_channel.group_by("iso3", "country")
                             .agg((pl.max("seconds") / pl.sum("seconds") * 100).alias("top_channel_pct"),
                                  (pl.sum("seconds") / 3600).alias("hours"), pl.len().alias("channels")))
    _country_map(top_channel.join(cty_hover, on="iso3"), "top_channel_pct", "From largest channel (%)", "Purples",
                 "map_top_channel_share", ":.0f")

    # 4) type of vehicle the footage is filmed from, per continent (share of footage)
    veh = (seg.drop_nulls("vehicle").group_by("continent", "vehicle").agg(hours)
              .with_columns((pl.col("hours") / pl.col("hours").sum().over("continent") * 100).alias("share")))
    fig = px.bar(veh.sort("hours", descending=True).to_pandas(), x="continent", y="share", color="vehicle",
                 **CONTINENT_STYLE,
                 labels={"share": "Share of footage (%)", "continent": "", "vehicle": "Type of vehicle"})
    _save(_style(fig), "bar_vehicle_type_continent")

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
    # legend inside the empty top-left corner, so the bars (and their labels) get the full width
    margin = dict(l=80, r=20, t=20, b=90)
    fig.update_layout(bargap=0.1, margin=margin, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"))
    fig.update_xaxes(title_text="Quarter of upload (hatched: the current quarter, not complete yet)")
    _bar_totals(fig, videos, "quarter", "len", vertical=True, plot_width=1600 - margin["l"] - margin["r"])
    # a tick every year and a grid in both directions, so each bar can be read off
    fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=-45, showgrid=True, gridcolor="#e5e5e5",
                     ticklabelmode="period")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
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

    # share of footage against share of population per country: which countries are over- or under-represented
    # (population share of all countries in the dataset); the 25 largest by either share
    shares = (seg.group_by("iso3", "country").agg(hours).join(pop, on="iso3")
                 .with_columns((pl.col("hours") / pl.col("hours").sum() * 100).alias("footage"),
                               (pl.col("population_country") / pl.col("population_country").sum() * 100)
                               .alias("population"),
                               pl.concat_str([flag, pl.col("country")], separator=" ").alias("name")))
    top_share = ((pl.col("footage").rank("ordinal", descending=True) <= 25)
                 | (pl.col("population").rank("ordinal", descending=True) <= 25))
    shares = shares.join(cty_hover, on="iso3").filter(top_share).sort("population")
    _dumbbell(shares, "name", {"footage": ("Footage", "#D55E00"), "population": ("Population", "#0072B2")},
              "Share (%, log scale)", "dumbbell_footage_population_share", log=True, fmt=":.2f")

    # footage by size of locality, per continent: what "urban" means in each part of the dataset
    bands = ["Under 10k", "10k–100k", "100k–1M", "1M–10M", "Over 10M"]
    size = (seg.join(df_mapping.select("id", "population_locality"), on="id")
               .with_columns(pl.when(pl.col("population_locality") > 0)
                               .then(pl.col("population_locality").cut([1e4, 1e5, 1e6, 1e7], labels=bands,
                                                                       left_closed=True).cast(pl.Utf8))
                               .otherwise(pl.lit("Unknown")).alias("band")))
    size = pl.concat([size, size.with_columns(pl.lit("All").alias("continent"))])
    size = (size.group_by("continent", "band").agg(hours)
                .with_columns((pl.col("hours") / pl.col("hours").sum().over("continent") * 100).alias("share")))
    fig = px.bar(size.to_pandas(), x="continent", y="share", color="band",
                 category_orders={"continent": CONTINENT_ORDER + ["All"], "band": bands + ["Unknown"]},
                 color_discrete_map={**dict(zip(bands, px.colors.sequential.Viridis_r[1::2])), "Unknown": "#cccccc"},
                 labels={"share": "Share of footage (%)", "continent": "", "band": "Population of locality"},
                 hover_data={"hours": ":,.0f", "share": ":.1f"})
    _save(_style(fig), "bar_footage_population_band")

    # effective number of channels (inverse Herfindahl index of channel shares): 1 means a single uploader,
    # n means as diverse as n channels with equal shares
    effective = (by_channel.with_columns((pl.col("seconds") / pl.col("seconds").sum().over("iso3")).alias("s"))
                           .group_by("iso3", "country")
                           .agg((1 / (pl.col("s") ** 2).sum()).alias("effective"),
                                (pl.sum("seconds") / 3600).alias("hours")))
    _country_map(effective.join(cty_hover, on="iso3"), "effective", "Effective number of channels", "Greens",
                 "map_effective_channels", ":.1f", log=True)

    # continent -> country -> locality, sized by footage and coloured by the share at night; the HTML drills down
    place = pl.concat_str([pl.col("locality"), pl.col("state")], separator=", ", ignore_nulls=True)
    tree = (seg.join(df_mapping.select("id", "state"), on="id")
               .with_columns(place.alias("place"), pl.concat_str([flag, pl.col("country")], separator=" ")
                             .alias("country"))
               .group_by("continent", "country", "place")  # same-named places in one state are merged
               .agg(hours, (pl.col("seconds").filter(pl.col("night")).sum() / pl.sum("seconds") * 100)
                    .alias("night"), pl.first("id"), pl.first("iso3"))
               .join(loc_hover, on="id"))
    fig = px.treemap(tree.to_pandas(), path=[px.Constant("All footage"), "continent", "country", "place"],
                     values="hours", color="night", color_continuous_scale="Blues", maxdepth=3,
                     labels={"night": "Night (%)", "hours": "Footage (hours)"})
    # the shared popups: localities and countries as in the other figures, continents with their totals
    popups = {f"All footage/{c}/{k}/{p}": h for c, k, p, h in tree.select("continent", "country", "place", "hover")
              .iter_rows()}
    popups |= {f"All footage/{c}/{k}": h for c, k, h in tree.unique(["continent", "country"])
               .join(cty_hover, on="iso3").select("continent", "country", "hover_right").iter_rows()}
    totals = tree.with_columns(night_h=pl.col("hours") * pl.col("night") / 100)
    for key, d in [("All footage", totals), *((f"All footage/{c}", d) for (c,), d in totals.group_by("continent"))]:
        popups[key] = (f"<b>{key.split('/')[-1]}</b><br><b>Footage:</b> {d['hours'].sum():,.0f} h in "
                       f"{d['country'].n_unique()} countries and {d.height:,} localities<br><b>Night:</b> "
                       f"{d['night_h'].sum() / d['hours'].sum() * 100:.0f}%")
    fig.update_traces(texttemplate="%{label}<br>%{value:,.0f} h", root_color="#f5f5f5",
                      hovertext=[popups.get(i, "") for i in fig.data[0].ids], hovertemplate=hover.TEMPLATE)
    fig.update_layout(coloraxis_colorbar=dict(title="Night (%)"))
    _save(_style(fig, margin=dict(l=5, r=5, t=5, b=5)), "treemap_footage")

    # every locality as a dot sized by footage (area proportional to hours), for print where the tile maps are weak
    night_hours = (pl.col("seconds").filter(pl.col("night")).sum() / 3600).alias("night_hours")
    dots = (seg.group_by("id").agg(hours, night_hours)
               .join(df_mapping.select("id", "locality", "country", "continent", "lat", "lon"), on="id")
               .join(loc_hover, on="id").drop_nulls(["lat", "lon"]).sort("hours", descending=True))
    fig = go.Figure(_dot_traces(dots))
    fig.update_geos(projection_type="natural earth", lataxis_range=[-57, 84], **GEO_STYLE)
    fig.add_annotation(text="Dot area proportional to hours of footage", x=0.01, y=0.02, xref="paper",
                       yref="paper", showarrow=False, font=dict(size=14, color="#666666"))
    _save(_style(fig, margin=dict(l=0, r=0, t=0, b=0),
                 legend=dict(x=0.01, y=0.35, itemsizing="constant", bgcolor="rgba(255,255,255,0.7)")),
          "map_localities_footage")

    # the same dots on a globe; the HTML spins until it is touched, then can be dragged
    fig = go.Figure(_dot_traces(dots))
    # a pale sea, so the edge of the globe shows, and a margin, so it does not touch the edges of the figure
    fig.update_geos(projection_type="orthographic", projection_rotation=dict(lon=10, lat=25),
                    showlakes=False, **{**GEO_STYLE, "oceancolor": "#e4edf5"})
    _save(_style(fig, width=1200, height=1000, margin=dict(l=30, r=190, t=30, b=30),
                 legend=dict(x=1.0, y=0.5, yanchor="middle", itemsizing="constant")),
          "globe_localities_footage", post_script=SPIN_GEO_JS, save_eps=False)

    # a 3D globe with a spike on each locality, its height proportional to the hours of footage; the HTML spins
    # until it is touched, then can be dragged
    # no EPS for the globes (the 3D one would be embedded in it as a ~10 MB picture): they are for the screen
    _save(_globe(dots), "globe_localities_footage_spikes", post_script=SPIN_JS, save_eps=False)

    # footprints of the channels with the most footage: where each one films (travel channels vs local drivers);
    # channels are numbered by footage; in the HTML each number links to the channel on YouTube
    per_channel = (seg.drop_nulls("channel").group_by("channel", "id").agg(hours)
                      .join(dots.select("id", "locality", "country", "continent", "lat", "lon", "hover"), on="id"))
    top = (per_channel.group_by("channel").agg(pl.sum("hours"), pl.col("country").n_unique().alias("countries"))
                      .sort("hours", descending=True).head(20))
    cols = 5
    rows = math.ceil(top.height / cols)
    titles = [f"<a href='https://www.youtube.com/channel/{r['channel']}' style='color:#0072B2'>#{i}</a> · "
              f"{r['countries']} {'country' if r['countries'] == 1 else 'countries'} · {r['hours']:,.0f} h"
              for i, r in enumerate(top.iter_rows(named=True), 1)]
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scattergeo"}] * cols] * rows,
                        subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.04)
    for i, r in enumerate(top.iter_rows(named=True)):
        d = per_channel.filter(pl.col("channel") == r["channel"]).sort("hours", descending=True)
        for t in _dot_traces(d, scale=0.45, line=f"#{i + 1} channel: %{{customdata:,.1f}} hours here"):
            fig.add_trace(t, row=i // cols + 1, col=i % cols + 1)
    seen = set()  # each continent once in the legend
    for t in fig.data:
        t.showlegend = t.legendgroup not in seen
        seen.add(t.legendgroup)
    fig.update_geos(projection_type="natural earth", lataxis_range=[-57, 84], **GEO_STYLE)
    fig.update_annotations(font_size=15)
    _save(_style(fig, width=1600, height=260 * rows + 80, margin=dict(l=5, r=5, t=40, b=5),
                 legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.01, yanchor="top",
                             itemsizing="constant")), "map_channel_footprints")

    # length of segments by continent and by type of vehicle, as boxes from precomputed quantiles (the HTML stays
    # small with ~100k segments); whiskers at the 5th and 95th percentiles
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                        column_widths=[0.45, 0.55], subplot_titles=["By continent", "By type of vehicle"])
    by_vehicle = seg.drop_nulls("vehicle").group_by("vehicle").agg(pl.sum("seconds")).sort("seconds", descending=True)
    for col, key, groups in [(1, "continent", CONTINENT_ORDER), (2, "vehicle", by_vehicle["vehicle"].to_list())]:
        for g in groups:
            m = seg.filter(pl.col(key) == g)["seconds"].to_numpy() / 60
            if not len(m):
                continue
            q = np.percentile(m, [5, 25, 50, 75, 95])
            color = CONTINENT_COLORS[g] if key == "continent" else "#0072B2"
            fig.add_trace(go.Box(x=[g], q1=[q[1]], median=[q[2]], q3=[q[3]], lowerfence=[q[0]], upperfence=[q[4]],
                                 name=g, marker_color=color, showlegend=False,
                                 hovertext=f"{g}: {len(m):,} segments"), row=1, col=col)
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Length of segment (minutes)", col=1)
    _save(_style(fig), "box_segment_length")


GEO_STYLE = dict(showland=True, landcolor="#eeeeee", showcountries=True, countrycolor="#cccccc", showocean=True,
                 oceancolor="white", showframe=False, coastlinecolor="#bbbbbb")
# Rotate a flat (orthographic geo) globe until the reader grabs it.
SPIN_GEO_JS = """
var gd = document.getElementById('{plot_id}'), lon = gd._fullLayout.geo.projection.rotation.lon, spinning = true;
var last = performance.now();
function spin() {  // 6 degrees a second, however long each redraw takes
  if (!spinning) return;
  var now = performance.now();
  lon = (lon + (now - last) * 0.006) % 360;
  last = now;
  Plotly.relayout(gd, {'geo.projection.rotation.lon': lon}).then(function () { requestAnimationFrame(spin); });
}
['mousedown', 'touchstart', 'wheel'].forEach(function (e) {
  gd.addEventListener(e, function () { spinning = false; }, {passive: true});
});
spin();
"""
# Turn the camera around a 3D globe (west to east, like the Earth) until the reader grabs it.
SPIN_JS = """
var gd = document.getElementById('{plot_id}'), eye = gd._fullLayout.scene.camera.eye, spinning = true;
var r = Math.hypot(eye.x, eye.y), a = Math.atan2(eye.y, eye.x), last = performance.now();
function spin() {  // 6 degrees a second, however long each redraw takes
  if (!spinning) return;
  var now = performance.now();
  a -= (now - last) * Math.PI / 30000;
  last = now;
  Plotly.relayout(gd, {'scene.camera.eye': {x: r * Math.cos(a), y: r * Math.sin(a), z: eye.z}})
    .then(function () { requestAnimationFrame(spin); });
}
['mousedown', 'touchstart', 'wheel'].forEach(function (e) {
  gd.addEventListener(e, function () { spinning = false; }, {passive: true});
});
spin();
"""


def _xyz(lon, lat, r=1.0):
    lon, lat = np.radians(np.asarray(lon, dtype=float)), np.radians(np.asarray(lat, dtype=float))
    # rounded to ~1 km on the globe: keeps the HTML small
    xyz = r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)
    return tuple(np.round(a, 4) for a in xyz)


GLOBE_MIN_HOURS = 0.1  # spikes start at six minutes of footage
# NASA Blue Marble (land_shallow_topo, public domain, 2048x1024, equirectangular), cached on first use
EARTH_IMAGE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg"


def _earth_mesh(width: int = 180, height: int = 90):
    """The Earth as a sphere mesh coloured with the Blue Marble image: one vertex per image pixel, with plotly blending
    the true colours between vertices."""
    path = os.path.join(common.output_dir, "blue_marble_2048.jpg")
    if not os.path.exists(path):
        os.makedirs(common.output_dir, exist_ok=True)
        urllib.request.urlretrieve(EARTH_IMAGE_URL, path)
    pixels = np.asarray(Image.open(path).convert("RGB").resize((width + 1, height + 1), Image.LANCZOS))
    lon, lat = np.meshgrid(np.linspace(-180, 180, width + 1), np.linspace(90, -90, height + 1))
    x, y, z = (a.ravel() for a in _xyz(lon, lat, 0.998))
    v = np.arange((height + 1) * (width + 1)).reshape(height + 1, width + 1)
    a, b, c, d = v[:-1, :-1].ravel(), v[:-1, 1:].ravel(), v[1:, 1:].ravel(), v[1:, :-1].ravel()  # cell corners
    return go.Mesh3d(x=x, y=y, z=z, i=np.concatenate([a, a]), j=np.concatenate([b, c]), k=np.concatenate([c, d]),
                     vertexcolor=[f"#{r:02x}{g:02x}{b_:02x}" for r, g, b_ in pixels.reshape(-1, 3)],
                     hoverinfo="skip", showscale=False, lighting=dict(ambient=1, diffuse=0, specular=0, fresnel=0))


def _globe(dots: pl.DataFrame, max_height: float = 0.25):
    """3D globe: the Blue Marble image with country borders and a spike on each locality of `dots` (lat, lon,
    locality, hours, night_hours). Spike height is on a log scale, so localities with little footage still show: zero
    at GLOBE_MIN_HOURS, `max_height` globe radii at the largest. Each spike is split by the share of day (below) and
    night (on top) footage."""
    fig = go.Figure(_earth_mesh())
    bx, by, bz = [], [], []
    for shape in map_labels.country_shapes().values():
        for ring in shape["rings"]:
            x, y, z = _xyz(*zip(*ring), r=1.001)
            bx += [*x, None]
            by += [*y, None]
            bz += [*z, None]
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(color="rgba(255,255,255,0.45)", width=1),
                               hoverinfo="skip", showlegend=False))
    lo, top = math.log10(GLOBE_MIN_HOURS), dots["hours"].max()
    h = np.log10(np.maximum(dots["hours"].to_numpy(), GLOBE_MIN_HOURS))
    r_top = 1.002 + max_height * (h - lo) / (math.log10(top) - lo)
    night = dots["night_hours"].to_numpy() / dots["hours"].to_numpy()
    r_mid = 1.002 + (r_top - 1.002) * (1 - night)  # day footage below, night footage on top
    nan = np.full(dots.height, np.nan)  # breaks between spikes
    # night in sky blue (not the darker blue of the other figures), so it shows against the dark ocean
    for name, r0, r1, color in [("Day", 1.002, r_mid, "#E69F00"), ("Night", r_mid, r_top, "#56B4E9")]:
        p0, p1 = _xyz(dots["lon"], dots["lat"], r0), _xyz(dots["lon"], dots["lat"], r1)
        x, y, z = (np.column_stack([a, b, nan]).ravel() for a, b in zip(p0, p1))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, line=dict(color=color, width=2.5),
                                   hoverinfo="skip"))
    # the popup on the spike tips; 3D traces take it as `text` (plotly leaves %{hovertext} empty in 3D)
    tips = _xyz(dots["lon"], dots["lat"], r_top)
    fig.add_trace(go.Scatter3d(x=tips[0], y=tips[1], z=tips[2], mode="markers", showlegend=False,
                               marker=dict(size=3, color=np.where(night > 0, "#56B4E9", "#E69F00")),
                               text=dots["hover"], hovertemplate=hover.GL_TEMPLATE))
    hidden = dict(visible=False, showbackground=False)
    eye = 1.45 * np.array(_xyz(15, 50))  # centred on Europe
    fig.update_layout(scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="data", dragmode="turntable",
                                 camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1))))
    fig.add_annotation(text=f"Spike height: hours of footage on a log scale (6 minutes to {top:,.0f} hours); "
                       "colours: share of day and night footage",
                       x=0.01, y=0.02, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color="#666666"))
    return _style(fig, width=1200, height=1000, margin=dict(l=0, r=0, t=0, b=0),
                  legend=dict(x=0.99, xanchor="right", y=0.5, yanchor="middle", bgcolor="rgba(255,255,255,0.7)"))


def _dot_traces(d: pl.DataFrame, scale: float = 1.3, legend: bool = True, line: str = "") -> list:
    """One Scattergeo trace per continent with a dot per locality in `d` (columns lat, lon, locality, continent,
    hours); dot area proportional to hours."""
    traces = []
    for continent in CONTINENT_ORDER:
        c = d.filter(pl.col("continent") == continent)
        if not c.height:
            continue
        traces.append(go.Scattergeo(
            lon=c["lon"], lat=c["lat"], name=continent, customdata=c["hours"], **_hover_args(c, line),
            legendgroup=continent, showlegend=legend, legendrank=CONTINENT_ORDER.index(continent),
            marker=dict(size=np.clip(2 + scale * np.sqrt(c["hours"].to_numpy()), 2, 32),
                        color=CONTINENT_COLORS[continent], opacity=0.6, line_width=0)))
    return traces


def _dumbbell(df: pl.DataFrame, label: str, values: dict, x_title: str, name: str, log: bool = False,
              fmt: str = ":.1f"):
    """One row per `label`, a dot per column of `values` (column -> (legend name, colour)) joined by a line;
    rows in the order of `df`, the first at the bottom."""
    cols = list(values)
    fig = go.Figure()
    xs, ys = [], []
    for r in df.iter_rows(named=True):
        xs += [r[cols[0]], r[cols[-1]], None]
        ys += [r[label], r[label], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#bbbbbb", width=2), showlegend=False,
                             hoverinfo="skip"))
    for col, (legend, color) in values.items():
        fig.add_trace(go.Scatter(x=df[col], y=df[label], mode="markers", name=legend,
                                 marker=dict(color=color, size=11),
                                 **_hover_args(df, f"{legend}: %{{x{fmt}}}")))
    fig.update_xaxes(type="log" if log else "linear", title_text=x_title, showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(categoryorder="array", categoryarray=df[label].to_list(), title_text="", dtick=1)
    _save(_style(fig, height=max(500, 24 * df.height + 150), margin=dict(l=10, r=20, t=20, b=70),
                 legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom")), name)


# Figures based on YOLO detections, with their README captions, in the order they appear in the README.
DETECTION_FIGURES = {
    "map_pedestrians_per_minute": "Pedestrians per minute of footage per country (unique tracked persons).",
    "box_pedestrians_per_minute_continent": "Pedestrians per minute of footage per locality, by continent.",
    "bar_road_user_mix": "Mix of detected road users in the 30 countries with the most analysed footage.",
    "scatter_indicators_pedestrians": "Pedestrians per minute of footage per country against country indicators.",
    "dot_pedestrians_per_minute": "Pedestrians per minute of footage per country, with a 95% bootstrap interval over "
                                  "its localities (countries with at least 10 hours in at least 3 localities).",
    "dumbbell_pedestrians_day_night": "Pedestrians per minute of footage by day and by night per country (countries "
                                      "with at least an hour of each).",
    "bar_road_user_mix_continent": "Mix of detected road users per continent.",
    "scatter_pedestrians_population": "Pedestrians per minute of footage per locality against its population.",
}


def detection_figures(df_mapping: pl.DataFrame, det: pl.DataFrame, classes: list, seg: pl.DataFrame,
                      flags: dict) -> None:
    """
    Figures based on YOLO detections.

    Args:
        df_mapping: The mapping.
        det: Per mapping row: `id`, one count column per class in `classes`, `detected_seconds` (footage covered
            by the detection CSVs), and `night_seconds` and `night_persons` (the part of those at night).
        classes: Detection count columns, the first one being persons.
        seg: The segment table (`segments()`), for the hover popups.
        flags: ISO3 code -> emoji flag, for the hover popups.
    """
    person = classes[0]
    loc_hover, cty_hover = hover.popups(df_mapping, seg, flags)
    det = det.join(df_mapping.select("id", "locality", "country", "iso3", "continent"), on="id").filter(
        pl.col("detected_seconds") > 0).join(loc_hover, on="id")
    per_min = (pl.col(person) / (pl.col("detected_seconds") / 60)).alias("per_minute")
    rate_line = "<b>Pedestrians per minute: %{PLACE:.2f}</b><br>"  # on top of the shared popup

    country = (det.group_by("iso3", "country").agg(pl.col(classes + ["detected_seconds"]).sum(), pl.first("continent"))
                  .join(cty_hover, on="iso3"))
    fig = px.choropleth(country.with_columns(per_min).to_pandas(), locations="iso3", color="per_minute",
                        hover_name="hover", color_continuous_scale="YlOrRd", projection="natural earth",
                        labels={"per_minute": "Pedestrians per minute"})
    fig.update_traces(hovertemplate=rate_line.replace("PLACE", "z") + hover.TEMPLATE)
    fig.update_layout(coloraxis_colorbar=colorbar_top("Pedestrians per minute"))
    fig.update_geos(domain=dict(x=[0, 1], y=[0, MAP_TOP_SHARE]))
    _save(_style(fig, margin=dict(l=0, r=0, t=10, b=0)), "map_pedestrians_per_minute")

    fig = px.box(det.with_columns(per_min).to_pandas(), x="continent", y="per_minute", color="continent",
                 points="outliers", hover_name="hover", **CONTINENT_STYLE,
                 labels={"per_minute": "Pedestrians per minute", "continent": ""})
    fig.update_traces(hoveron="points", hovertemplate=rate_line.replace("PLACE", "y") + hover.TEMPLATE)
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

    # pedestrians per minute per country with a 95% bootstrap interval over its localities; countries with at least
    # MIN_HOURS of analysed footage in at least 3 localities
    rng = np.random.default_rng(0)
    rows = []
    for (iso3, name), d in det.group_by("iso3", "country"):
        if d.height < 3 or d["detected_seconds"].sum() < MIN_HOURS * 3600:
            continue
        continent = d["continent"][0]  # countries split across continents are shown once
        p, m = d[person].to_numpy(), d["detected_seconds"].to_numpy() / 60
        i = rng.integers(0, d.height, (1000, d.height))
        lo, hi = np.percentile(p[i].sum(1) / m[i].sum(1), [2.5, 97.5])
        rows.append(dict(iso3=iso3, country=name, continent=continent, rate=p.sum() / m.sum(), lo=lo, hi=hi))
    if rows:
        ci = pl.DataFrame(rows).join(cty_hover, on="iso3").sort("rate")
        fig = go.Figure()
        for continent in CONTINENT_ORDER:
            d = ci.filter(pl.col("continent") == continent)
            fig.add_trace(go.Scatter(x=d["rate"], y=d["country"], mode="markers", name=continent,
                                     marker=dict(color=CONTINENT_COLORS[continent], size=9),
                                     error_x=dict(type="data", symmetric=False, array=d["hi"] - d["rate"],
                                                  arrayminus=d["rate"] - d["lo"], color="#999999", thickness=1.5),
                                     customdata=np.column_stack([d["lo"], d["hi"]]),
                                     **_hover_args(d, "Pedestrians per minute: %{x:.2f} (95% interval "
                                                      "%{customdata[0]:.2f}–%{customdata[1]:.2f})")))
        fig.update_xaxes(title_text="Pedestrians per minute (95% bootstrap interval over localities)",
                         showgrid=True, gridcolor="#e5e5e5", rangemode="tozero")
        fig.update_yaxes(categoryorder="array", categoryarray=ci["country"].to_list(), dtick=1)
        _save(_style(fig, height=max(500, 22 * ci.height + 150), margin=dict(l=10, r=20, t=20, b=70),
                     legend=dict(x=0.99, xanchor="right", y=0.01, yanchor="bottom")), "dot_pedestrians_per_minute")

    # pedestrians per minute by day and by night, in countries with at least an hour of analysed footage of each
    day_night = (country.join(det.group_by("iso3").agg(pl.sum("night_seconds", "night_persons")), on="iso3")
                        .with_columns(((pl.col(person) - pl.col("night_persons"))
                                       / ((pl.col("detected_seconds") - pl.col("night_seconds")) / 60)).alias("day"),
                                      (pl.col("night_persons") / (pl.col("night_seconds") / 60)).alias("night"))
                        .filter(pl.col("night_seconds") >= 3600,
                                pl.col("detected_seconds") - pl.col("night_seconds") >= 3600)
                        .sort("day"))
    if day_night.height:
        _dumbbell(day_night, "country", {"day": ("Day", "#E69F00"), "night": ("Night", "#0072B2")},
                  "Pedestrians per minute", "dumbbell_pedestrians_day_night", fmt=":.2f")

    # mix of detected road users per continent and overall
    continent = det.group_by("continent").agg(pl.col(classes).sum())
    continent = pl.concat([continent, continent.select(pl.lit("All").alias("continent"), pl.col(classes).sum())])
    mix = (continent.unpivot(index="continent", on=classes, variable_name="Road user", value_name="count")
                    .with_columns((pl.col("count") / pl.col("count").sum().over("continent") * 100).alias("share")))
    fig = px.bar(mix.to_pandas(), x="continent", y="share", color="Road user",
                 category_orders={"continent": CONTINENT_ORDER + ["All"], "Road user": classes},
                 labels={"share": "Share of detected road users (%)", "continent": ""},
                 hover_data={"count": ":,", "share": ":.1f"})
    _save(_style(fig), "bar_road_user_mix_continent")

    # pedestrians per minute per locality against its population: whether more pedestrians just means a larger city
    loc = (det.join(df_mapping.select("id", "population_locality"), on="id").with_columns(per_min)
              .filter(pl.col("population_locality") > 0, pl.col("per_minute") > 0))
    if loc.height > 2:
        rho = loc.select(pl.corr("population_locality", "per_minute", method="spearman")).item()
        fig = px.scatter(loc.to_pandas(), x="population_locality", y="per_minute", color="continent",
                         hover_name="hover", log_x=True, log_y=True, opacity=0.7, **CONTINENT_STYLE,
                         labels={"population_locality": "Population of locality",
                                 "per_minute": "Pedestrians per minute", "continent": ""})
        fig.update_traces(hovertemplate=rate_line.replace("PLACE", "y") + hover.TEMPLATE)
        fig.add_annotation(text=f"Spearman ρ = {rho:.2f}, n = {loc.height:,} localities", x=0.99, y=0.02,
                           xref="paper", yref="paper", xanchor="right", showarrow=False, font=dict(size=16))
        _save(_style(fig), "scatter_pedestrians_population")


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
