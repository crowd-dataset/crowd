"""
Figures describing the dataset and, when YOLO detection CSVs are available, what is detected in it.

Dataset figures are built from the mapping alone, one row per segment (`segments()`). Detection figures take the
per-locality counts of unique tracked objects produced by `analysis.count_detections()`.
"""

import ast
import json
import math
import os
import time
import unicodedata
from fractions import Fraction
import urllib.request
from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import requests
from PIL import Image
from plotly.subplots import make_subplots

import common
from custom_logger import CustomLogger
from utils.analytics.metrics_cache import MetricsCache
from utils.core.dataset_stats import Dataset_Stats
from utils.plotting import area_figures, hover, map_labels
from utils.plotting.constants import CONTINENT_COLORS, MAP_TOP_SHARE, colorbar_top
from utils.plotting.io import IO

io = IO()

# Country-level columns of the mapping compared with the amount of footage and detection rates (column -> name).
logger = CustomLogger(__name__)

INDICATORS = {
    "population_country": "Population (log10)",
    "traffic_mortality": "Road traffic deaths per 100k",
    "gini": "Gini index",
    "med_age": "Median age (years)",
    "literacy_rate": "Literacy rate (%)",
}
# road users detected by YOLO: the same colour in every figure (Okabe-Ito)
CLASS_COLOURS = {"Persons": "#E69F00", "Cars": "#0072B2", "Bicycles": "#009E73", "Motorcycles": "#D55E00",
                 "Buses": "#CC79A7", "Trucks": "#56B4E9"}
# the vehicle the footage is filmed from: cars in the cars' blue; the rare types in greys
VEHICLE_COLOURS = {"Car": "#0072B2", "Bus": "#CC79A7", "Truck": "#56B4E9", "Motorcycle": "#D55E00",
                   "Bicycle": "#009E73", "Electric scooter": "#E69F00", "Non-electric scooter": "#F0E442",
                   "Automated car": "#555555", "Emergency vehicle": "#999999", "Monowheel/unicycle": "#BBBBBB"}
# log axes of rates per minute: 1, 2 and 5 times the powers of ten (the default labels only the powers)
LOG_TICKS = [m * 10.0 ** p for p in range(-2, 4) for m in (1, 2, 5)]
CONTINENT_ORDER = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
# the same order and colour for each continent in every figure
CONTINENT_STYLE = dict(category_orders={"continent": CONTINENT_ORDER}, color_discrete_map=CONTINENT_COLORS)
LABEL_TOP = 12  # scatter plots label only the largest points; the rest show on hover


def _style(fig, **layout):
    # log axes: label powers of ten only, unless a figure set its own ticks (e.g., for a narrow range)
    fig.for_each_xaxis(lambda a: a.update(dtick=1) if a.type == "log" and a.tickvals is None else None)
    fig.for_each_yaxis(lambda a: a.update(dtick=1) if a.type == "log" and a.tickvals is None else None)
    fig.update_layout(template=common.get_configs("plotly_template"),
                      font=dict(family=common.get_configs("font_family"), size=common.get_configs("font_size")),
                      **layout)
    return fig


def _save(fig, name, post_script=None, save_eps=True, html_fig=None):
    # static images at the figure's own size where it sets one (e.g., a taller scatter), else 1600x900
    io.save_plotly_figure(fig, name, width=fig.layout.width or 1600, height=fig.layout.height or 900,
                          save_final=True, post_script=post_script, save_eps=save_eps, html_fig=html_fig)


def _hover_args(d: pl.DataFrame, line: str = "") -> dict:
    """Hover showing the shared locality or country popup (column `hover`, see hover.py), after an optional line
    with the figure's own value (plotly template syntax)."""
    return dict(hovertext=d["hover"], hovertemplate=(f"<b>{line}</b><br>" if line else "") + hover.TEMPLATE)


# Interactive scatter plots: when zoomed in to at most ZOOM_LABELS points, label every point in view (a hidden
# trace with all labels) instead of only the largest ones.
ZOOM_LABELS = 150
ZOOM_LABELS_JS = """
var gd = document.getElementById('{plot_id}');
var home = [gd._fullLayout.xaxis.range.slice(), gd._fullLayout.yaxis.range.slice()];  // the view as published
function zoomLabels() {
  var all = gd.data.findIndex(function (t) { return t.meta === 'zoom-labels'; });
  if (all < 0) return;
  var t = gd.data[all], xr = gd._fullLayout.xaxis.range, yr = gd._fullLayout.yaxis.range, n = 0;
  for (var i = 0; i < t.x.length; i++) {
    var x = gd._fullLayout.xaxis.type === 'log' ? Math.log10(t.x[i]) : t.x[i];
    var y = gd._fullLayout.yaxis.type === 'log' ? Math.log10(t.y[i]) : t.y[i];
    if (x >= xr[0] && x <= xr[1] && y >= yr[0] && y <= yr[1]) n++;
  }
  var zoomed = [xr, yr].some(function (r, a) {
    return Math.abs(r[0] - home[a][0]) + Math.abs(r[1] - home[a][1]) > 1e-9;
  });
  var show = zoomed && n <= MAX_LABELS;
  if (!!t.visible === show) return;
  var top = [];
  gd.data.forEach(function (d, i) { if (d.meta === 'top-labels') top.push(i); });
  Plotly.restyle(gd, {visible: show}, [all]);
  if (top.length) Plotly.restyle(gd, {visible: !show}, top);
}
gd.on('plotly_relayout', zoomLabels);
"""


# HTML of scatters with a country legend: each country's points carry their own labels (`meta.top`: the labelled
# ones; `meta.all`: every point), so a country hidden in the legend hides its labels too; zoomed in to at most
# MAX_LABELS visible points, every point in view is labelled.
COUNTRY_ZOOM_JS = """
var gd = document.getElementById('{plot_id}');
var home = [gd._fullLayout.xaxis.range.slice(), gd._fullLayout.yaxis.range.slice()];
function zoomLabels() {
  var xr = gd._fullLayout.xaxis.range, yr = gd._fullLayout.yaxis.range, n = 0, idx = [];
  var lx = gd._fullLayout.xaxis.type === 'log', ly = gd._fullLayout.yaxis.type === 'log';
  gd.data.forEach(function (t, i) {
    if (!t.meta || !t.meta.all) return;
    idx.push(i);
    if (t.visible === 'legendonly' || t.visible === false) return;
    for (var j = 0; j < t.x.length; j++) {
      var x = lx ? Math.log10(t.x[j]) : t.x[j], y = ly ? Math.log10(t.y[j]) : t.y[j];
      if (x >= xr[0] && x <= xr[1] && y >= yr[0] && y <= yr[1]) n++;
    }
  });
  var zoomed = [xr, yr].some(function (r, a) {
    return Math.abs(r[0] - home[a][0]) + Math.abs(r[1] - home[a][1]) > 1e-9;
  });
  var key = zoomed && n <= MAX_LABELS ? 'all' : 'top';
  if (gd._labelKey === key) return;
  gd._labelKey = key;
  Plotly.restyle(gd, {text: idx.map(function (i) { return gd.data[i].meta[key]; })}, idx);
}
gd._labelKey = 'top';
gd.on('plotly_relayout', zoomLabels);
gd.on('plotly_restyle', zoomLabels);  // a country hidden or shown in the legend
"""


def _literal_list(cell) -> list:
    try:
        value = ast.literal_eval(cell) if isinstance(cell, str) else None
    except (ValueError, SyntaxError):
        return []
    return value if isinstance(value, list) else []


def _item(lst, i):
    return lst[i] if i < len(lst) else None


def _place(flag: pl.Expr) -> pl.Expr:
    """Flag and locality, with the state for US places (several US localities share a name, e.g. Philadelphia, PA
    and Philadelphia, MS). Needs columns locality, state and iso3."""
    locality = (pl.when((pl.col("iso3") == "USA") & pl.col("state").is_not_null())
                  .then(pl.concat_str([pl.col("locality"), pl.col("state")], separator=", "))
                  .otherwise(pl.col("locality")))
    return pl.concat_str([flag, locality], separator=" ")


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


def _vs_indicators(df: pl.DataFrame, value: str, value_title: str, name: str, log_y: bool,
                   indicators: dict = INDICATORS, min_hours: float = 0):
    """Scatter of `value` per country against each indicator (column -> title), one panel per indicator, each with its
    Spearman rank correlation, the number of countries and a least-squares trend line (on the log scale when
    `log_y`). With `min_hours`, only countries with at least that many hours (column hours)."""
    if min_hours:
        df = df.filter(pl.col("hours") >= min_hours)
    cols = 3 if len(indicators) <= 9 else 4
    rows = math.ceil(len(indicators) / cols)
    titles = []
    for col in indicators:
        d = df.select(col, value).drop_nulls().filter(pl.col(value) > 0)
        rho = d.select(pl.corr(col, value, method="spearman")).item() if d.height > 2 else float("nan")
        titles.append(f"{indicators[col]}<br><sup>Spearman ρ = {rho:.2f}, n = {d.height}</sup>")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.07,
                        vertical_spacing=0.32 / rows)
    for i, col in enumerate(indicators):
        r, c = i // cols + 1, i % cols + 1
        d = df.select("country", "continent", "hover", col, value).drop_nulls([col, value]).filter(pl.col(value) > 0)
        for continent in CONTINENT_ORDER:
            dc = d.filter(pl.col("continent") == continent)
            fig.add_trace(go.Scatter(x=dc[col], y=dc[value], mode="markers", name=continent,
                                     legendgroup=continent, showlegend=i == 0,
                                     marker=dict(color=CONTINENT_COLORS[continent], size=7, opacity=0.8),
                                     **_hover_args(dc, f"{indicators[col]}: %{{x:,.1f}} · {value_title}: "
                                                       "%{y:,.2f}")), row=r, col=c)
        if d.height > 5:
            # the median of equal-sized groups of countries along x: a monotone trend like Spearman's ρ shows,
            # where a straight line would be pulled by a few outliers (and could run below 0)
            order = np.argsort(d[col].to_numpy())
            x, y = d[col].to_numpy()[order], d[value].to_numpy()[order]
            groups = np.array_split(np.arange(len(x)), max(3, min(8, len(x) // 15)))
            fig.add_trace(go.Scatter(x=[np.median(x[g]) for g in groups], y=[np.median(y[g]) for g in groups],
                                     mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color="#222222", width=2.5, dash="dash")), row=r, col=c)
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text=value_title, col=1)
    legend = {}
    if rows * cols > len(indicators):  # the legend in the first empty slot (bottom row), not beside the grid
        legend = dict(legend=dict(x=(len(indicators) % cols + 0.15) / cols, y=0.8 / rows, xanchor="left",
                                  yanchor="top"))
    _save(_style(fig, legend_title_text="", height=max(900, 300 * rows), **legend), name)


MIN_HOURS = 10  # countries with less footage are greyed out on share maps: a share of a few minutes is noise


def _country_map(df: pl.DataFrame, value: str, title: str, scale: str, name: str, fmt: str, log: bool = False):
    """World map of a per-country value; countries with less than MIN_HOURS of footage are grey."""
    ok = df.filter(pl.col("hours") >= MIN_HOURS) if "hours" in df.columns else df
    few = df.filter(pl.col("hours") < MIN_HOURS) if "hours" in df.columns else df.clear()
    d = ok.with_columns((pl.col(value).log10() if log else pl.col(value)).alias("_c"))
    fig = px.choropleth(d.to_pandas(), locations="iso3", color="_c", hover_name="hover",
                        color_continuous_scale=area_figures.visible_scale(scale), projection="natural earth",
                        labels={value: title})
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


GDP_PER_CAPITA = "NY.GDP.PCAP.PP.CD"  # World Bank: GDP per person at purchasing-power parity, current int. $
WEALTH_COLOURS = {"above": "#C8504A", "near": "#BDBDB0", "below": "#6C79C9"}  # as in The Economist's chart


def _cached_json(name: str, url: str, params: dict, max_age_days: int):
    """JSON downloaded from `url`, cached in .wb_cache (like update_params.py) and refreshed after `max_age_days`;
    the old copy when the download fails, None when there is none."""
    path = os.path.join(".wb_cache", name + ".json")
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > max_age_days * 86400:
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            os.makedirs(".wb_cache", exist_ok=True)
            with open(path, "w") as f:
                f.write(r.text)
        except requests.RequestException:
            if not os.path.exists(path):
                return None
    with open(path) as f:
        return json.load(f)


def gdp_per_capita(max_age_days: int = 30) -> pl.DataFrame | None:
    """GDP per person at PPP ($'000) in each country's latest year, from the World Bank; None when it cannot be
    downloaded."""
    data = _cached_json(GDP_PER_CAPITA, f"https://api.worldbank.org/v2/country/all/indicator/{GDP_PER_CAPITA}",
                        {"format": "json", "mrnev": 1, "per_page": 1000}, max_age_days)
    if data is None:
        return None
    return pl.DataFrame([{"iso3": r["countryiso3code"], "gdp_pc": r["value"] / 1000} for r in data[1]
                         if r.get("value") and r.get("countryiso3code")])


def _world_bank(code: str, max_age_days: int = 30) -> dict:
    """ISO3 -> a World Bank indicator in each country's latest year ({} when it cannot be downloaded)."""
    data = _cached_json(code, f"https://api.worldbank.org/v2/country/all/indicator/{code}",
                        {"format": "json", "mrnev": 1, "per_page": 1000}, max_age_days)
    return {r["countryiso3code"]: r["value"] for r in (data or [None, []])[1]
            if r.get("value") is not None and r.get("countryiso3code")}


def _who(code: str, dim1: str | None = None, max_age_days: int = 180) -> dict:
    """ISO3 -> a WHO Global Health Observatory indicator in each country's latest year (with `dim1`, only that
    breakdown, e.g. both sexes); {} when it cannot be downloaded."""
    data = _cached_json(f"who_{code}" + (f"_{dim1}" if dim1 else ""), f"https://ghoapi.azureedge.net/api/{code}",
                        {"$filter": f"Dim1 eq '{dim1}'"} if dim1 else {}, max_age_days)
    latest = {}
    for r in (data or {}).get("value", []):
        if r.get("SpatialDimType") == "COUNTRY" and r.get("NumericValue") is not None:
            latest[r["SpatialDim"]] = max(latest.get(r["SpatialDim"], (0, 0)), (r["TimeDim"], r["NumericValue"]))
    return {k: v for k, (_, v) in latest.items()}


def _climate(variable: str, max_age_days: int = 365) -> dict:
    """ISO3 -> a 1991-2020 annual climatology (ERA5) from the World Bank's Climate Change Knowledge Portal: "tas" mean
    temperature (°C), "pr" precipitation (mm a year)."""
    data = _cached_json(f"cckp_{variable}_1991-2020", "https://cckpapi.worldbank.org/cckp/v1/era5-x0.25_climatology_"
                        f"{variable}_climatology_annual_1991-2020_mean_historical_era5_x0.25_mean/all_countries",
                        {"_format": "json"}, max_age_days)
    return {iso3: next(iter(v.values())) for iso3, v in (data or {}).get("data", {}).items() if v}


# indicators of the pedestrians figure beyond the mapping's own (public sources, cached in .wb_cache)
PEDESTRIAN_INDICATORS = INDICATORS | {
    "gdp_log": "GDP per person ($'000 PPP, log10)",
    "pedestrian_share": "Pedestrians among road deaths (%)",
    "traffic_index": "TomTom traffic index (footage-weighted)",
    "vehicles": "Registered vehicles per 1,000 people",
    "urban": "Urban population (%)",
    "density_log": "People per km² (log10)",
    "inactive": "Adults with too little physical activity (%)",
    "obese": "Adults with obesity (%)",
    "temperature": "Mean temperature, 1991-2020 (°C)",
    "rainfall": "Rainfall, 1991-2020 (mm a year)",
    "agriculture": "Jobs in agriculture (%)",
}


def _public_indicators(df_mapping: pl.DataFrame, seg: pl.DataFrame) -> pl.DataFrame:
    """One row per country of the mapping with the PEDESTRIAN_INDICATORS that are not mapping columns: World Bank
    (GDP per person, urban population, density, jobs in agriculture), WHO (pedestrians among road deaths, 2016;
    registered vehicles, 2016, per 1,000 of the mapping's population; physical inactivity; obesity) and ERA5
    temperature and rainfall; and the localities' TomTom traffic index averaged over the country, weighted by their
    footage."""
    gdp, urban, density, agriculture = (_world_bank(c) for c in (GDP_PER_CAPITA, "SP.URB.TOTL.IN.ZS", "EN.POP.DNST",
                                                                 "SL.AGR.EMPL.ZS"))
    deaths = pedestrian_death_share()
    share = dict(deaths.iter_rows()) if deaths is not None else {}
    vehicles = _who("RS_194")
    inactive, obese = _who("NCD_PAA", "SEX_BTSX"), _who("NCD_BMI_30A", "SEX_BTSX")
    temperature, rainfall = _climate("tas"), _climate("pr")
    pop = df_mapping.group_by("iso3").agg(pl.col("population_country").filter(pl.col("population_country") > 0)
                                          .first())
    traffic = (seg.group_by("id").agg(pl.sum("seconds"))
                  .join(df_mapping.select("id", "iso3", "traffic_index").filter(pl.col("traffic_index") > 0), on="id")
                  .group_by("iso3").agg(((pl.col("traffic_index") * pl.col("seconds")).sum() / pl.sum("seconds"))
                                        .alias("traffic_index")))
    lookup = {"gdp_log": gdp, "pedestrian_share": share, "urban": urban, "density_log": density,
              "inactive": inactive, "obese": obese, "temperature": temperature, "rainfall": rainfall,
              "agriculture": agriculture, "vehicles": vehicles}
    out = pop.with_columns(pl.col("iso3").replace_strict(d, default=None, return_dtype=pl.Float64).alias(k)
                           for k, d in lookup.items())
    return (out.with_columns((pl.col("gdp_log") / 1000).log10(), pl.col("density_log").log10(),
                             (pl.col("vehicles") / pl.col("population_country") * 1000).alias("vehicles"))
               .join(traffic, on="iso3", how="left").drop("population_country"))


WHO_ROAD_USER_DEATHS = "RS_246"  # WHO: distribution of road traffic deaths by type of road user (%), 2016


def pedestrian_death_share(max_age_days: int = 180) -> pl.DataFrame | None:
    """Share of each country's road traffic deaths that are pedestrians (%), latest year, from the WHO's Global Health
    Observatory; None when it cannot be downloaded."""
    share = _who(WHO_ROAD_USER_DEATHS, "ROADUSERTYPE_RS-DDC-PEDESTRIANS", max_age_days)
    # not where most deaths are of an "other" or unknown road user (the Philippines: 94%, pedestrians 1%)
    other = _who(WHO_ROAD_USER_DEATHS, "ROADUSERTYPE_RS-DDC-OTHER", max_age_days)
    share = {k: v for k, v in share.items() if other.get(k, 0) <= 50}
    return pl.DataFrame({"iso3": list(share), "pedestrian_share": list(share.values())}) if share else None


def _vs_trend(df: pl.DataFrame, x_col: str, x_title: str, value: str, title: str, name: str, ratio: float,
              what: str, basis: str, bubble: bool = True, labelled: pl.Expr | None = None):
    """Scatter of `value` against `x_col` (both log scales) with a least-squares line through all points, coloured by
    whether a point is over `ratio` times, near, or under 1/`ratio` of what the line predicts from its `basis` (as
    The Economist's charts: "above/below expectations"). Countries are bubbles sized by population
    (population_country); with `bubble` False, plain dots (localities). Columns: name, hover, `x_col`, `value`, and
    population_country for bubbles. `labelled`: the rows to label (default: the 15 most populous and the 6 furthest
    from the line on either side)."""
    x, y = np.log10(df[x_col].to_numpy()), np.log10(df[value].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    gap = y - (slope * x + intercept)  # log10 of value / predicted
    frac = Fraction(1 / ratio).limit_denominator(10)
    labels = {"above": f"More than {ratio:g}x what its {basis} predicts", "near": f"Near what its {basis} predicts",
              "below": f"Less than {frac.numerator}/{frac.denominator} of what its {basis} predicts"}
    k = math.log10(ratio)
    df = df.with_columns(pl.Series("_gap", gap),
                         pl.Series("_group", [labels["above" if g > k else "below" if g < -k else "near"]
                                              for g in gap]))
    if bubble:
        df = df.with_columns((8 + 70 * (pl.col("population_country") / pl.col("population_country").max()).sqrt())
                             .alias("_size"))
    if labelled is None:
        labelled = ((pl.col("population_country").rank("ordinal", descending=True) <= 15)
                    | (pl.col("_gap").rank("ordinal", descending=True) <= 6) | (pl.col("_gap").rank("ordinal") <= 6))
    xs = np.linspace(x.min(), x.max(), 50)
    size = "Circle size: population · " if bubble else ""
    _labelled_scatter(df, x_col, value, "name", labelled, x_title, title, name,
                      groups=("_group", {labels[g]: WEALTH_COLOURS[g] for g in labels}),
                      bubble="_size" if bubble else None, trend=(10 ** xs, 10 ** (slope * xs + intercept)),
                      zoom_labels=df.height if df.height <= 300 else ZOOM_LABELS,
                      note=f"{size}dotted line: {what} predicted from {basis} · n = {df.height:,}")


def _vs_wealth(df: pl.DataFrame, value: str, title: str, name: str, ratio: float, what: str):
    """`value` per country against GDP per person: see _vs_trend (columns also gdp_pc)."""
    _vs_trend(df, "gdp_pc", "GDP per person ($'000 at purchasing-power parity, log scale)", value, title, name,
              ratio, what, "wealth")


def _labelled_scatter(df: pl.DataFrame, x_col: str, y_col: str, label: str, labelled: pl.Expr, x_title: str,
                      y_title: str, name: str, size=(1600, 900), emphasis: pl.Expr = pl.lit(True), note: str = "",
                      log_x: bool = True, log_y: bool = True, zoom_labels: int = ZOOM_LABELS,
                      country_legend: str | None = None, groups: tuple | None = None, bubble: str | None = None,
                      trend: tuple | None = None, decorate=None):
    """Log-log scatter (linear axes if not `log_x` / `log_y`) coloured by continent; with `country_legend` (a column
    with each row's flag and country), the legend lists the countries instead, beside the plot. Rows where `labelled`
    is true get a label placed without overlaps (next to the point, or with a leader line when there is no room); in
    the HTML, zooming in labels every point in view once at most `zoom_labels` points are in it.
    Labels of rows where `emphasis` is false are smaller and grey, so the eye goes to the emphasised ones first.
    `groups` (column, {value: colour} in legend order) colours by that column instead of continents; `bubble` is a
    column with each point's diameter (px); `trend` is (x values, y values) of a line drawn under the points;
    `decorate(fig)` adds anything else (e.g., reference lines)."""
    margin = dict(l=90, r=320 if country_legend else 30, t=30 if country_legend else 70, b=110, autoexpand=False)
    fx, fy = (math.log10 if log else float for log in (log_x, log_y))  # data -> axis units (labels are placed in
    ix, iy = ((lambda v: 10 ** v) if log else float for log in (log_x, log_y))  # axis units) and back
    x = np.log10(df[x_col].to_numpy()) if log_x else df[x_col].to_numpy().astype(float)
    y = np.log10(df[y_col].to_numpy()) if log_y else df[y_col].to_numpy().astype(float)
    # room on the right for the largest points' labels: in axis units, a third of a decade on log axes
    x_range = (x.min() - 0.1, x.max() + 0.35) if log_x else (x.min() - 0.03 * np.ptp(x), x.max() + 0.12 * np.ptp(x))
    y_range = (y.min() - 0.15, y.max() + 0.2) if log_y else (y.min() - 0.03 * np.ptp(y), y.max() + 0.06 * np.ptp(y))
    fig = go.Figure()
    html_fig = None
    legend = dict(orientation="h", x=0, y=1.01, yanchor="bottom")  # above the plot, clear of the points
    if country_legend:
        # one legend entry per country (clicking one hides or shows it), in alphabetical order; the points keep
        # their continent's colour. The legend is a column beside the plot, scrolling when it does not fit (the
        # static image shows its top)
        order = df.group_by(country_legend).agg(pl.first("country"))
        # alphabetical ignoring accents: Åland Islands with the As, Türkiye before Turkmenistan
        order = order.with_columns(pl.col("country").map_elements(
            lambda c: unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode().lower(),
            return_dtype=pl.Utf8).alias("_key")).sort("_key")
        for country in order[country_legend]:
            d = df.filter(pl.col(country_legend) == country)
            fig.add_trace(go.Scatter(x=d[x_col], y=d[y_col], mode="markers", name=country,
                                     marker=dict(color=[CONTINENT_COLORS.get(c, "#999999") for c in d["continent"]],
                                                 size=8, opacity=0.75), **_hover_args(d)))
        legend = dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top", font=dict(size=11),
                      title=dict(text="Countries<br>(colours: continents)", side="top"))
        # the HTML: the same points, each country's labels on its own points (no leader lines), so they hide with
        # the country
        html_fig = go.Figure()
        marked = df.with_columns(labelled.alias("_lab"))
        for country in order[country_legend]:
            d = marked.filter(pl.col(country_legend) == country)
            top_text = [n if lab else "" for n, lab in zip(d[label], d["_lab"])]
            html_fig.add_trace(go.Scatter(x=d[x_col], y=d[y_col], mode="markers+text", name=country,
                                          text=top_text, textposition="middle right", textfont=dict(size=10),
                                          meta={"top": top_text, "all": d[label].to_list()},
                                          marker=dict(color=[CONTINENT_COLORS.get(c, "#999999")
                                                             for c in d["continent"]], size=8, opacity=0.75),
                                          **_hover_args(d)))
    else:
        column, colours = groups or ("continent", {c: CONTINENT_COLORS[c] for c in CONTINENT_ORDER})
        if trend is not None:
            fig.add_trace(go.Scatter(x=trend[0], y=trend[1], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color="#888888", width=1.5, dash="dot")))
        for group, colour in colours.items():
            d = df.filter(pl.col(column) == group)
            if bubble:  # the largest bubbles first, so the smaller ones stay visible on top
                d = d.sort(bubble, descending=True)
            fig.add_trace(go.Scatter(x=d[x_col], y=d[y_col], mode="markers", name=group,
                                     marker=dict(color=colour, size=d[bubble] if bubble else 8,
                                                 opacity=0.6 if bubble else 0.75,
                                                 line=dict(width=0.8, color="white") if bubble else None),
                                     **_hover_args(d)))
    # select rows, not names: same-named places (e.g., two Philadelphias) must not share a label
    top = df.with_row_index("_i").with_columns(emphasis.alias("_emphasis")).filter(labelled)
    proj = map_labels.AxisProjection(x_range, y_range, size[0] - margin["l"] - margin["r"],
                                     size[1] - margin["t"] - margin["b"])
    items = [dict(code=str(r["_i"]), lines=[r[label]], anchor=(fx(r[x_col]), fy(r[y_col])),
                  ct=None, scale=1 if r["_emphasis"] else 9 / 11, dot=r[bubble] if bubble else 8)
             for r in top.sort(y_col, descending=True).iter_rows(named=True)]
    dots = {item["code"]: item["dot"] for item in items}
    # labels with no free spot are left out of the static image; hover and zoom in the HTML still show them
    placed = map_labels.place_labels(items, proj, {}, stack_clusters=False, drop_unplaced=True)
    names = {str(r["_i"]): r[label] for r in top.iter_rows(named=True)}
    emphasised = {str(r["_i"]) for r in top.iter_rows(named=True) if r["_emphasis"]}
    strong, faint = dict(size=11, color="black"), dict(size=9, color="#8a8a8a")
    for code, p in placed.items():
        (ax, ay) = p[1]
        font = strong if code in emphasised else faint
        if p[0] == "dot":  # invisible marker so the text sits beside the point like on the maps
            fig.add_trace(go.Scatter(x=[ix(ax)], y=[iy(ay)], mode="markers+text", text=[names[code]],
                                     textposition=p[2], textfont=font, marker=dict(size=dots[code], opacity=0),
                                     showlegend=False, hoverinfo="skip", meta="top-labels"))
        else:
            (lx, ly), pos = p[2], p[3]
            fig.add_trace(go.Scatter(x=[ix(ax), ix(lx)], y=[iy(ay), iy(ly)], mode="lines",
                                     line=dict(color="#bbbbbb" if font is faint else "grey", width=1),
                                     showlegend=False, hoverinfo="skip",
                                     meta="top-labels"))
            fig.add_trace(go.Scatter(x=[ix(lx)], y=[iy(ly)], mode="text", text=[names[code]],
                                     textposition=pos, textfont=font, showlegend=False, hoverinfo="skip",
                                     meta="top-labels"))
    # every point's label, shown in the HTML only when zoomed in (see ZOOM_LABELS_JS)
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="text", text=df[label], textposition="top center",
                             textfont=strong, visible=False, showlegend=False, hoverinfo="skip", meta="zoom-labels"))
    fig.update_xaxes(type="log" if log_x else "linear", range=x_range, title_text=x_title, automargin=False)
    fig.update_yaxes(type="log" if log_y else "linear", range=y_range, title_text=y_title, automargin=False)
    for axis, log, r in ((fig.layout.xaxis, log_x, x_range), (fig.layout.yaxis, log_y, y_range)):
        if log and r[1] - r[0] < 2:  # under two decades: ticks at 1, 2 and 5 times the powers of ten
            ticks = [m * 10 ** p for p in range(math.floor(r[0]), math.ceil(r[1]) + 1) for m in (1, 2, 5)
                     if r[0] <= math.log10(m * 10 ** p) <= r[1]]
            axis.update(tickvals=ticks, ticktext=[f"{t:,g}" for t in ticks], dtick=None, tickmode="array")
    if note:  # e.g., the correlation, below the plot on the right (inside, it covered points and labels)
        fig.add_annotation(text=note, x=1, y=-0.105, xref="paper", yref="paper", xanchor="right", yanchor="top",
                           showarrow=False, font=dict(size=15))
    if decorate is not None:
        decorate(fig)
    if html_fig is not None:
        html_fig.update_layout(xaxis=fig.layout.xaxis, yaxis=fig.layout.yaxis)
        html_fig = _style(html_fig, margin=margin, legend=legend)
    _save(_style(fig, width=size[0], height=size[1], margin=margin, legend=legend), name,
          post_script=(COUNTRY_ZOOM_JS if html_fig is not None else ZOOM_LABELS_JS).replace("MAX_LABELS",
                                                                                            str(zoom_labels)),
          html_fig=html_fig)


def dataset_figures(df_mapping: pl.DataFrame, seg: pl.DataFrame, flags: dict) -> None:
    """Figures based on the mapping only. `flags` maps ISO3 codes to emoji flags for labels."""
    hours = (pl.sum("seconds") / 3600).alias("hours")
    flag = pl.col("iso3").replace_strict(flags, default="🏳️", return_dtype=pl.Utf8)
    # the popups shown on hover for each locality and country, the same in every figure
    loc_hover, cty_hover = hover.popups(df_mapping, seg, flags)

    # footage against number of videos, per locality and per country
    city = (seg.group_by("id").agg(hours, pl.col("video").n_unique().alias("videos"))
               .join(df_mapping.select("id", "locality", "state", "country", "iso3", "continent"), on="id")
               .join(loc_hover, on="id")
               .with_columns(_place(flag).alias("name"),
                             pl.format("{} {} ({})", flag, pl.col("country"), pl.col("iso3")).alias("flag_country")))
    # the 40 localities with most footage labelled, the top ones in larger black text; zoom in the HTML for more
    rank = pl.col("hours").rank("ordinal", descending=True)
    _labelled_scatter(city, "hours", "videos", "name", rank <= 40, "Footage (hours)", "Number of videos",
                      "scatter_all_total_time-video_count", emphasis=rank <= LABEL_TOP, country_legend="flag_country")
    # every country labelled with flag and ISO3 code (tall, so the flags fit); the top 30 by footage in black
    country = (seg.group_by("iso3").agg(hours, pl.col("video").n_unique().alias("videos"), pl.first("continent"))
                  .with_columns(pl.concat_str([flag, pl.col("iso3")], separator=" ").alias("name"))
                  .join(cty_hover, on="iso3"))
    _labelled_scatter(country, "hours", "videos", "name", pl.lit(True), "Footage (hours)", "Number of videos",
                      "scatter_all_country_total_time-video_count", size=(1600, 1700),
                      emphasis=pl.col("hours").rank("ordinal", descending=True) <= 30, zoom_labels=country.height)
    # the same on linear scales: how far the largest localities and countries lead; the 40 localities and 30
    # countries with the most footage are labelled where there is room, and zooming in the HTML labels every point
    _labelled_scatter(city, "hours", "videos", "name", rank <= 40, "Footage (hours)", "Number of videos",
                      "scatter_all_total_time-video_count_linear", emphasis=rank <= LABEL_TOP, log_x=False,
                      log_y=False, country_legend="flag_country", zoom_labels=city.height)  # any zoom: all labels
    top30 = pl.col("hours").rank("ordinal", descending=True) <= 30
    _labelled_scatter(country, "hours", "videos", "name", top30, "Footage (hours)", "Number of videos",
                      "scatter_all_country_total_time-video_count_linear", size=(1600, 1000), log_x=False,
                      log_y=False, zoom_labels=country.height)

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
    share = pl.when(pl.col("share") < 0.1).then(pl.lit("<0.1")).otherwise(pl.col("share").round(1).cast(pl.Utf8))
    veh = veh.with_columns(pl.format("{}% of footage, {}% at night", share,
                                     pl.col("night").round(0).cast(pl.Int64)).alias("text"))
    fig = px.bar(veh.to_pandas(), y="vehicle", x="hours", orientation="h", text="text", log_x=True,
                 labels={"hours": "Footage (hours, log scale)", "vehicle": ""})
    fig.update_traces(textposition="outside", marker_color="#0072B2", cliponaxis=False)
    _save(_style(fig, margin=dict(r=260)), "bar_vehicle_type_time_of_day")

    # 1) footage against city population: which large cities are under-sampled
    # labelled: the localities with the most footage and the largest ones by population
    city = (seg.group_by("id").agg(hours)
               .join(df_mapping.select("id", "locality", "state", "iso3", "continent", "population_locality"), on="id")
               .join(loc_hover, on="id").filter(pl.col("population_locality") > 0)
               .with_columns(_place(flag).alias("name")))
    notable = ((pl.col("hours").rank("ordinal", descending=True) <= 8)
               | (pl.col("population_locality").rank("ordinal", descending=True) <= 8))
    _labelled_scatter(city, "population_locality", "hours", "name", notable, "Population of locality",
                      "Footage (hours)", "scatter_population_footage")
    # localities with more or less footage than their size predicts: the large cities to collect next; labelled:
    # the most populous, the most footage and the most under-covered cities of over a million
    under = pl.when(pl.col("population_locality") >= 1e6).then(pl.col("_gap")).rank("ordinal") <= 12
    _vs_trend(city, "population_locality", "Population of locality (log scale)", "hours", "Footage (hours, log scale)",
              "scatter_locality_footage_vs_population", ratio=3, what="footage", basis="population", bubble=False,
              labelled=((pl.col("population_locality").rank("ordinal", descending=True) <= 10)
                        | (pl.col("hours").rank("ordinal", descending=True) <= 8) | under))

    # footage against the economy (GMP, known for ~200 large cities) and the congestion of each locality: do rich or
    # congested cities dominate the data? The traffic index (how much slower than free flow traffic is, %) is one
    # TomTom reading per locality; shown where it is above 0, as older zeros also stand for failed requests
    city = (seg.group_by("id").agg(hours)
               .join(df_mapping.select("id", "locality", "state", "iso3", "continent", "gmp", "traffic_index"),
                     on="id")
               .join(loc_hover, on="id")
               .with_columns(_place(flag).alias("name")))
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

    # footage per person against wealth, as The Economist's innovation chart: bubbles sized by population, coloured
    # by how far each country is from what its wealth predicts (a least-squares line on the log scales)
    gdp = gdp_per_capita()
    if gdp is not None:
        wealth = (per_capita.join(gdp, on="iso3").join(cty_hover, on="iso3")
                  .with_columns(pl.concat_str([flag, pl.col("country")], separator=" ").alias("name"))
                  .filter(pl.col("per_million") > 0))
        _vs_wealth(wealth, "per_million", "Footage (hours per million people, log scale)",
                   "bubble_footage_per_capita_gdp", ratio=2, what="footage")

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
                 category_orders={"continent": CONTINENT_ORDER}, color_discrete_map=VEHICLE_COLOURS,
                 labels={"share": "Share of footage (%)", "continent": "", "vehicle": "Type of vehicle"})
    _save(_style(fig, legend_traceorder="reversed"), "bar_vehicle_type_continent")  # legend in stacking order

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
                 labels={"x": "Length of segment (minutes, log scale)", "y": "Number of segments"})
    fig.update_traces(width=np.diff(edges), marker_color="#0072B2")
    # a little room left of the first bar, so the 0.1 label is clear of the y axis's 0
    fig.update_xaxes(range=[math.log10(edges[0]) - 0.12, math.log10(edges[-1]) + 0.03])
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
    _save(_style(fig, legend_traceorder="reversed"), "bar_footage_population_band")  # legend in stacking order

    # effective number of channels (inverse Herfindahl index of channel shares): 1 means a single uploader,
    # n means as diverse as n channels with equal shares
    effective = (by_channel.with_columns((pl.col("seconds") / pl.col("seconds").sum().over("iso3")).alias("s"))
                           .group_by("iso3", "country")
                           .agg((1 / (pl.col("s") ** 2).sum()).alias("effective"),
                                (pl.sum("seconds") / 3600).alias("hours")))
    _country_map(effective.join(cty_hover, on="iso3"), "effective", "Effective number of channels", "Greens",
                 "map_effective_channels", ":.1f", log=True)
    # fewer uploaders than a country's amount of footage predicts: comparisons rest on few channels' routes and cameras
    channels = (effective.join(pop, on="iso3").join(cty_hover, on="iso3").drop_nulls("population_country")
                .with_columns(pl.concat_str([flag, pl.col("country")], separator=" ").alias("name"))
                .filter(pl.col("hours") > 0, pl.col("effective") > 0))
    _vs_trend(channels, "hours", "Footage (hours, log scale)", "effective", "Effective number of channels (log scale)",
              "bubble_channels_vs_footage", ratio=2, what="channels", basis="amount of footage")

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
                     values="hours", color="night", color_continuous_scale=area_figures.visible_scale("Blues"),
                     range_color=(0, tree["night"].quantile(0.95)), maxdepth=3,  # a few places are all night
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
    fig.update_layout(coloraxis_colorbar=dict(title=dict(text="Night (%)", side="right")))
    _save(_style(fig, margin=dict(l=5, r=5, t=5, b=5)), "treemap_footage")

    # every locality as a dot sized by footage (area proportional to hours), for print where the tile maps are weak
    night_hours = (pl.col("seconds").filter(pl.col("night")).sum() / 3600).alias("night_hours")
    dots = (seg.group_by("id").agg(hours, night_hours)
               .join(df_mapping.select("id", "locality", "state", "country", "continent", "lat", "lon"), on="id")
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
    # a spike per locality (thin, so dense areas stay readable); in the HTML, a button merges neighbouring
    # localities into one spike per area of about 200 km
    areas = _globe_cells(dots.join(df_mapping.select("id", "iso3"), on="id").with_columns(
        _place(flag).alias("name")))
    _save(_globe([("Each locality", "One spike per locality", dots, 1.2),
                  ("Areas of ~200 km", "One spike per ~200 km area", areas, 2)]),
          "globe_localities_footage_spikes", post_script=SPIN_JS, save_eps=False)
    _widget_globe()

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

    # county and region maps, spike maps and smoothed maps of the footage
    area_figures.dataset_area_figures(df_mapping, seg, loc_hover, flags)


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
  gd.addEventListener(e, function (ev) {  // grabbing the globe stops it; the buttons above it do not
    if (!(ev.target.closest && ev.target.closest('.updatemenu-container'))) spinning = false;
  }, {passive: true});
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
  gd.addEventListener(e, function (ev) {  // grabbing the globe stops it; the buttons above it do not
    if (!(ev.target.closest && ev.target.closest('.updatemenu-container'))) spinning = false;
  }, {passive: true});
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


# the CROWD dashboards' globe widget (dashboards/globe), as a README figure: htmlpreview.github.io runs only scripts
# hosted on GitHub, so its d3 and topojson-client are saved next to the figures (once), as plotly.min.js is
WIDGET_GLOBE = os.path.join(common.root_dir, "dashboards", "globe", "widget_globe.html")
WIDGET_SCRIPTS = {"d3.min.js": "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
                  "topojson-client.min.js": "https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/"
                                            "topojson-client.min.js"}
README_URL = "https://github.com/crowd-dataset/crowd#readme"


def _mapping_on_github(ago_s: int) -> str:
    """mapping.csv on GitHub's main as it was ago_s seconds ago ('' when it cannot be fetched), as the globe on
    bazilinskyy.github.io does: the history of mapping.csv from GitHub's API (a CI checkout has no history)."""
    until = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - ago_s))
    token = os.environ.get("GITHUB_TOKEN")  # the Action's: a higher API rate limit
    try:
        r = requests.get("https://api.github.com/repos/crowd-dataset/crowd/commits", timeout=60,
                         params=dict(sha="main", path="mapping.csv", until=until, per_page=1),
                         headers={"Authorization": f"Bearer {token}"} if token else {})
        r.raise_for_status()
        if not r.json():
            return ""
        r = requests.get(f"https://raw.githubusercontent.com/crowd-dataset/crowd/{r.json()[0]['sha']}/mapping.csv",
                         timeout=300)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        logger.warning(f"Could not get mapping.csv as it was {ago_s // 86400} days ago ({e}); the globe widget "
                       "figure shows no recent places.")
        return ""


def _widget_globe():
    """The globe widget with every locality built in (a square each, its size growing with the hours of footage) and
    the places with segments added in the last week (red) and month (amber), as on bazilinskyy.github.io, as
    globe_localities_widget.html; a still of it (plotly, in the widget's colours) as the PNG."""
    from dashboards.globe import globe_data  # the widget's own comparison of two mappings
    folder = os.path.join(common.root_dir, "figures")
    html = open(WIDGET_GLOBE, encoding="utf-8").read()
    for name, url in WIDGET_SCRIPTS.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Could not download {url} ({e}); globe_localities_widget is skipped.")
                return
            with open(path, "wb") as f:
                f.write(r.content)
        html = html.replace(url, name)
    with open(common.get_configs("mapping"), encoding="utf-8") as f:
        now = globe_data.segments(f.read())
    week, month = (globe_data.segments(t) if t else {} for t in map(_mapping_on_github, (7 * 86400, 30 * 86400)))
    points, totals = globe_data.compare(now, week, month)  # [lat, lon, hours, new in a month, new in a week, name]
    data = dict(date=time.strftime("%Y-%m-%d", time.gmtime()), localities=len(points),
                hours=round(sum(p[2] for p in points)), **totals, points=points)
    built_in = "<script>window.GLOBE = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";</script>\n"
    html = (html.replace("__REFRESH__", "3600").replace("__DATASET__", README_URL)
                .replace('<script src="dock.js"></script>', "")  # the widget server's window placement
                .replace("<script>\nconst LAPTOP", built_in + "<script>\nconst LAPTOP", 1))
    with open(os.path.join(folder, "globe_localities_widget.html"), "w", encoding="utf-8") as f:
        f.write(html)
    # the still: the widget's first view (centred on 10° E, 22° N) and colours; older places as squares, the last
    # month's and week's as circles on top
    lat, lon, hours, month_new, week_new = (np.array([p[k] for p in points]) for k in range(5))
    fig = go.Figure()
    for name, sel, marker in [
            (f"{len(points):,} localities · {data['hours']:,} h", (month_new == 0),
             dict(symbol="square", color="#2563eb", opacity=0.6,
                  size=2.4 * (1.2 + np.minimum(3.8, np.sqrt(hours) / 5)))),
            (f"last month: {totals['recent_places']:,} places, +{totals['recent']:,} segments",
             (month_new > 0) & (week_new == 0), dict(color="#d97706", size=9)),
            (f"last week: {totals['day_places']:,} places, +{totals['day']:,} segments", week_new > 0,
             dict(color="#e11d48", size=10))]:
        size = marker.pop("size")
        fig.add_trace(go.Scattergeo(lat=lat[sel], lon=lon[sel], mode="markers", name=name, hoverinfo="skip",
                                    marker=dict(size=size[sel] if isinstance(size, np.ndarray) else size,
                                                line=dict(width=0 if marker.get("symbol") else 1, color="white"),
                                                **marker)))
    fig.update_geos(projection_type="orthographic", projection_rotation=dict(lon=10, lat=22), showland=True,
                    landcolor="#f8fafc", showocean=True, oceancolor="#dfeaf6", showcountries=True,
                    countrycolor="#cbd5e1", coastlinecolor="#cbd5e1", showlakes=False, showframe=False,
                    lataxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.06)"),
                    lonaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.06)"))
    fig = _style(fig, width=1000, height=1000, margin=dict(l=20, r=20, t=20, b=20),
                 legend=dict(x=0.01, y=0.01, yanchor="bottom", itemsizing="constant", bgcolor="rgba(255,255,255,0.8)",
                             title=dict(text=f"As of {data['date']}")))
    for out in (folder, common.output_dir):
        fig.write_image(os.path.join(out, "globe_localities_widget.png"))


def _globe_cells(dots: pl.DataFrame, deg: float = 2.0) -> pl.DataFrame:
    """Localities merged into cells of `deg` degrees (about 200 km), one spike each, so that spikes of neighbouring
    localities do not overlap: the cell's total and night hours, at the footage-weighted centre of its localities.
    The hover text is the locality's popup for a cell with one locality, else a list of its localities."""
    cells = (dots.with_columns((pl.col("lat") / deg).round().alias("_r"), (pl.col("lon") / deg).round().alias("_c"))
                 .sort("hours", descending=True)
                 .group_by("_r", "_c")
                 .agg(pl.sum("hours", "night_hours"), ((pl.col("lat") * pl.col("hours")).sum() / pl.sum("hours"))
                      .alias("lat"), ((pl.col("lon") * pl.col("hours")).sum() / pl.sum("hours")).alias("lon"),
                      pl.col("name"), pl.col("hours").alias("each"), pl.first("hover"), pl.len().alias("n")))

    def summary(r):
        if r["n"] == 1:
            return r["hover"]
        rows = [f"{n}: {h:,.1f} h" for n, h in zip(r["name"][:10], r["each"][:10])]
        more = f"<br>and {r['n'] - 10} more" if r["n"] > 10 else ""
        return (f"<b>{r['n']} localities around {r['name'][0]}</b><br>Footage: {r['hours']:,.1f} h, "
                f"{r['night_hours'] / r['hours'] * 100:.0f}% at night<br>──────────<br>" + "<br>".join(rows) + more)
    return cells.with_columns(pl.Series("hover", [summary(r) for r in cells.iter_rows(named=True)])).select(
        "lat", "lon", "hours", "night_hours", "hover")


def _globe(views: list, max_height: float = 0.25):
    """3D globe: the Blue Marble image with country borders and spikes. `views`: (button label, note, dots, line
    width) for each set of spikes, the first shown (and in the static image), the others behind buttons in the HTML;
    dots: lat, lon, hours, night_hours, hover. Spike height is on a log scale, so places with little footage still
    show: zero at GLOBE_MIN_HOURS, `max_height` globe radii at the largest. Each spike is split by the share of day
    (below) and night (on top) footage."""
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
    lo, top = math.log10(GLOBE_MIN_HOURS), max(d["hours"].max() for _, _, d, _ in views)
    for v, (_, _, dots, width) in enumerate(views):
        h = np.log10(np.maximum(dots["hours"].to_numpy(), GLOBE_MIN_HOURS))
        r_top = 1.002 + max_height * (h - lo) / (math.log10(top) - lo)
        night = dots["night_hours"].to_numpy() / dots["hours"].to_numpy()
        r_mid = 1.002 + (r_top - 1.002) * (1 - night)  # day footage below, night footage on top
        nan = np.full(dots.height, np.nan)  # breaks between spikes
        # night in sky blue (not the darker blue of the other figures), so it shows against the dark ocean
        for name, r0, r1, color in [("Day", 1.002, r_mid, "#E69F00"), ("Night", r_mid, r_top, "#56B4E9")]:
            p0, p1 = _xyz(dots["lon"], dots["lat"], r0), _xyz(dots["lon"], dots["lat"], r1)
            x, y, z = (np.column_stack([a, b, nan]).ravel() for a, b in zip(p0, p1))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, legendgroup=name, showlegend=v == 0,
                                       visible=v == 0, line=dict(color=color, width=width), hoverinfo="skip"))
        # the popup on the spike tips; 3D traces take it as `text` (plotly leaves %{hovertext} empty in 3D)
        tips = _xyz(dots["lon"], dots["lat"], r_top)
        fig.add_trace(go.Scatter3d(x=tips[0], y=tips[1], z=tips[2], mode="markers", showlegend=False, visible=v == 0,
                                   marker=dict(size=width + 0.5, color=np.where(night > 0, "#56B4E9", "#E69F00")),
                                   text=dots["hover"], hovertemplate=hover.GL_TEMPLATE))
    notes = [f"{note}: hours of footage on a log scale (6 minutes to {top:,.0f} h); colours: share of day and night "
             "footage" for _, note, _, _ in views]
    if len(views) > 1:  # buttons in the HTML (static images leave them out)
        buttons = [dict(label=label, method="update",
                        args=[{"visible": [True, True] + [w == v for w in range(len(views)) for _ in range(3)]},
                              {"annotations[0].text": notes[v]}])
                   for v, (label, _, _, _) in enumerate(views)]
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0.01, y=0.99, xanchor="left",
                                            yanchor="top", showactive=True, buttons=buttons)])
    hidden = dict(visible=False, showbackground=False)
    eye = 1.45 * np.array(_xyz(15, 50))  # centred on Europe
    fig.update_layout(scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="data", dragmode="turntable",
                                 camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1))))
    fig.add_annotation(text=notes[0],
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
    # range fixed to the rows: autorange leaves several empty rows above and below
    fig.update_yaxes(categoryorder="array", categoryarray=df[label].to_list(), title_text="", dtick=1,
                     range=[-0.6, df.height - 0.4])
    _save(_style(fig, height=max(500, 24 * df.height + 110), margin=dict(l=10, r=20, t=40, b=60),
                 legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom")), name)


def contributor_figures(seg: pl.DataFrame, credits: dict) -> pl.DataFrame:
    """Who added the videos in the dataset (credits: video -> (contributor, added_utc), see
    utils/analytics/contributors.py): cumulative footage over time per contributor, and the README table."""
    who = pl.DataFrame([(v, c, t) for v, (c, t, *_) in credits.items()], schema=["video", "contributor", "added"],
                       orient="row").with_columns(pl.col("added").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"))
    videos = (seg.group_by("video").agg((pl.sum("seconds") / 3600).alias("hours"), pl.first("id"), pl.first("iso3"))
                 .join(who, on="video", how="left")
                 .with_columns(pl.col("contributor").fill_null("Unknown")))
    totals = videos.group_by("contributor").agg(pl.sum("hours")).sort("hours", descending=True)
    order = totals["contributor"].to_list()
    colors = dict(zip(order, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
                              "#999999", "#000000", "#882255"] * 2))
    daily = (videos.drop_nulls("added").with_columns(pl.col("added").dt.date().alias("day"))
                   .group_by("contributor", "day").agg(pl.sum("hours")).sort("day"))
    days = pl.DataFrame({"day": pl.date_range(daily["day"].min(), daily["day"].max(), eager=True)})
    fig = go.Figure()
    for c in order[::-1]:  # the largest at the bottom of the stack
        d = days.join(daily.filter(pl.col("contributor") == c), on="day", how="left").with_columns(
            pl.col("hours").fill_null(0).cum_sum().alias("cumulative"))
        total = totals.filter(pl.col("contributor") == c)["hours"][0]
        fig.add_trace(go.Scatter(x=d["day"], y=d["cumulative"], name=f"{c} ({total:,.0f} h)", mode="lines",
                                 stackgroup="all",
                                 line=dict(width=0.5, color=colors[c]),
                                 hovertemplate=f"{c}: %{{y:,.0f}} hours by %{{x|%d %b %Y}}<extra></extra>"))
    fig.update_xaxes(title_text="Date added to the dataset", dtick="M12", tickformat="%Y", showgrid=True,
                     gridcolor="#e5e5e5")
    fig.update_yaxes(title_text="Footage (hours, cumulative)", showgrid=True, gridcolor="#e5e5e5")
    _save(_style(fig, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)", traceorder="reversed")),
          "area_contributors_footage")
    return (videos.group_by("contributor")
                  .agg(pl.len().alias("Videos"), pl.sum("hours").round(1).alias("Footage (h)"),
                       pl.col("id").n_unique().alias("Localities"), pl.col("iso3").n_unique().alias("Countries"),
                       pl.col("added").min().dt.strftime("%Y-%m-%d").alias("First added"),
                       pl.col("added").max().dt.strftime("%Y-%m-%d").alias("Last added"))
                  .sort("Footage (h)", descending=True).rename({"contributor": "Contributor"}))


# Figures based on YOLO detections, with their README captions, in the order they appear in the README.
DETECTION_FIGURES = {
    "map_detection_coverage": "Share of each country's footage that has been analysed with YOLO (countries with less "
                              "than 10 hours of footage are grey).",
    "map_pedestrians_per_minute": "Pedestrians per minute of footage per country (unique tracked persons; countries "
                                  "with less than 10 analysed hours are grey).",
    "dot_pedestrians_per_minute": "Pedestrians per minute of footage per country, with a 95% bootstrap interval over "
                                  "its localities (countries with at least 10 analysed hours in at least 3 "
                                  "localities).",
    "scatter_interval_width_hours": "How much footage a stable estimate of a locality needs: the width of its 95% "
                                    "bootstrap interval of pedestrians per minute over its segments (as a share of "
                                    "the rate) against its analysed hours (localities with at least 5 segments; "
                                    "segments of one drive are alike, so the true intervals are somewhat wider).",
    "scatter_walkable_car_dominated": "Pedestrians against cars per minute per locality (at least an analysed hour), "
                                      "split at the medians into walkable, busy, car-dominated and quiet streets.",
    "bar_pedestrians_city_size": "Pedestrians per minute by population of the locality, per continent, with a 95% "
                                 "bootstrap interval over localities (groups with at least 3 localities and an "
                                 "analysed hour).",
    "waffle_road_user_mix": "Road-user mix of the 30 localities with the most analysed footage, each as 100 squares "
                            "(ordered by the share of pedestrians).",
    "heatmap_road_users_per_minute": "Detected road users per minute of footage, per class, in the 40 countries with "
                                     "the most analysed footage (log colour scale).",
    "box_pedestrians_per_minute_continent": "Pedestrians per minute of footage per locality, by continent.",
    "hist_pedestrians_per_minute_segment": "Distribution of pedestrians per minute over segments of at least a "
                                           "minute (log scale).",
    "bar_road_user_mix_continent": "Mix of detected road users per continent.",
    "bar_road_user_mix": "Mix of detected road users in the 30 countries with the most analysed footage.",
    "map_two_wheeler_share": "Share of two-wheelers (motorcycles and bicycles) among the detected vehicles per "
                             "country (countries with less than 10 analysed hours are grey).",
    "bar_road_users_day_night": "Detected road users per minute by day and by night, per class and continent (log "
                                "scale).",
    "dumbbell_pedestrians_day_night": "Pedestrians per minute of footage by day and by night per country (countries "
                                      "with at least an hour of each).",
    "bar_pedestrians_vehicle_type": "Pedestrians per minute by the type of vehicle the footage is filmed from: the "
                                    "viewpoint changes what the detector sees.",
    "line_road_users_upload_year": "Detected road users per minute by year of upload (years with at least 10 analysed "
                                   "hours): newer cameras and higher resolutions can change what the detector sees.",
    "scatter_indicators_pedestrians": "Pedestrians per minute of footage per country (at least 10 analysed hours) "
                                      "against country indicators: population, road deaths, Gini index, median age "
                                      "and literacy (from the mapping), GDP per person, urban population, density "
                                      "and jobs in agriculture (World Bank), pedestrians among road deaths, "
                                      "registered vehicles, physical inactivity and obesity (WHO), mean temperature "
                                      "and rainfall (ERA5, World Bank climate portal) and the footage-weighted "
                                      "TomTom traffic index. Dashed lines: medians of equal-sized groups of "
                                      "countries along each indicator.",
    "bubble_pedestrians_gdp": "Pedestrians per minute of footage against GDP per person (countries with at least 10 "
                              "analysed hours; circle size: population), coloured by whether a country has more "
                              "than 1.5 times, about, or less than two-thirds of the pedestrians its wealth "
                              "predicts.",
    "bubble_motorcycles_gdp": "Motorcycles as a share of detected vehicles against GDP per person, coloured by "
                              "whether a country has more than twice, about, or less than half the share its "
                              "wealth predicts.",
    "bubble_bicycles_gdp": "Bicycles per minute against GDP per person: cycling cultures stand out above the line.",
    "bubble_cars_gdp": "Cars per minute against GDP per person: motorisation on the street, relative to wealth.",
    "bubble_road_deaths_vs_pedestrians": "Road traffic deaths per 100,000 people against pedestrians per minute: "
                                         "countries with more or fewer deaths than their number of pedestrians on "
                                         "the street predicts.",
    "bubble_pedestrian_deaths_vs_pedestrians": "Pedestrian road deaths per 100,000 people (road deaths times "
                                               "the WHO's pedestrian share of them, 2016) against pedestrians per "
                                               "minute: where walking is more or less deadly than the number of "
                                               "pedestrians on the street predicts (not countries where most road "
                                               "deaths are of an unknown road user, e.g. the Philippines).",
    "map_us_counties_pedestrians": "Pedestrians per minute per US county (counties with at least an hour of "
                                   "analysed footage).",
    "map_eu_regions_pedestrians": "Pedestrians per minute per European region (NUTS 2, and first-level regions of "
                                  "European Russia, Ukraine, Belarus and Bosnia and Herzegovina; regions with at "
                                  "least an hour of analysed footage).",
    "map_eu_regions_bicycles": "Bicycles per minute per European region: cycling regions stand out.",
    "map_pedestrians_smoothed_europe": "Pedestrians per minute in Europe, smoothed over about 50 km; grey where there "
                                       "is less than an analysed hour within 100 km.",
    "map_walkable_minutes_europe": "Walkable street minutes in Europe: the share of analysed minutes with at least 2 "
                                   "pedestrians and at most 10 cars per minute, smoothed over about 50 km.",
    "map_motorcycle_share_europe": "Motorcycles as a share of detected vehicles in Europe, smoothed over about 50 km.",
    "map_bicycles_smoothed_europe": "Bicycles per minute in Europe, smoothed over about 50 km.",
    "scatter_locality_pedestrians_vs_population": "Pedestrians per minute per locality against its population "
                                                  "(localities with at least an hour of analysed footage): busy and "
                                                  "quiet streets for a city's size.",
    "scatter_pedestrians_population": "Pedestrians per minute of footage per locality against its population.",
}
# the HTML fills the window: squares sized to the cells of a 10 x 10 grid, on load and after every resize
FIT_SQUARES_JS = """
var gd = document.getElementById('{plot_id}');
function fitSquares() {
    var x = gd._fullLayout.xaxis, y = gd._fullLayout.yaxis;
    Plotly.restyle(gd, {'marker.size': 0.85 * Math.min(x._length, y._length) / 10.2});
}
fitSquares();
window.addEventListener('resize', function () { setTimeout(fitSquares, 200); });
"""
TWO_WHEELERS, VEHICLES = ("Bicycles", "Motorcycles"), ("Cars", "Bicycles", "Motorcycles", "Buses", "Trucks")


def detection_figures(df_mapping: pl.DataFrame, det: pl.DataFrame, classes: list, seg: pl.DataFrame,
                      flags: dict) -> list:
    """
    Figures and README tables based on YOLO detections.

    Args:
        df_mapping: The mapping.
        det: One row per segment with detections: `id`, `video`, `start` (as in `segments()`) and one count column
            per class in `classes` (unique tracked objects).
        classes: Detection count columns, the first one being persons.
        seg: The segment table (`segments()`): time of day, vehicle, upload year, and the hover popups.
        flags: ISO3 code -> emoji flag.

    Returns:
        README tables as (title, DataFrame).
    """
    person = classes[0]
    loc_hover, cty_hover = hover.popups(df_mapping, seg, flags)
    flag = pl.col("iso3").replace_strict(flags, default="🏳️", return_dtype=pl.Utf8)
    d = (det.select("id", "video", "start", *classes).join(seg, on=["id", "video", "start"])
            .with_columns((pl.col("seconds") / 60).alias("minutes")))

    def rates(df: pl.DataFrame, *by) -> pl.DataFrame:
        """Detections per minute of each class per group, with the analysed hours, segments and localities."""
        return df.group_by(*by).agg(*[(pl.col(c).sum() / pl.sum("minutes")).alias(c) for c in classes],
                                    (pl.sum("minutes") / 60).alias("hours"), pl.len().alias("segments"),
                                    pl.col("id").n_unique().alias("localities"))

    country = (rates(d, "iso3").join(seg.group_by("iso3").agg(pl.first("country"), pl.first("continent")), on="iso3")
                               .join(cty_hover, on="iso3")
                               .with_columns(pl.concat_str([flag, pl.col("country")], separator=" ").alias("name")))
    locality = (rates(d, "id").join(df_mapping.select("id", "locality", "state", "iso3", "continent",
                                                      "population_locality"),
                                    on="id").join(loc_hover, on="id"))
    rate_line = "<b>Pedestrians per minute: %{PLACE:.2f}</b><br>"  # on top of the shared popup

    # how much of each country's footage has been analysed
    total = seg.group_by("iso3", "country").agg((pl.sum("seconds") / 3600).alias("hours"))
    coverage = (total.join(country.select("iso3", pl.col("hours").alias("analysed")), on="iso3", how="left")
                     .with_columns((pl.col("analysed").fill_null(0) / pl.col("hours") * 100).alias("analysed_pct"))
                     .join(cty_hover, on="iso3"))
    _country_map(coverage, "analysed_pct", "Footage analysed (%)", "Greens", "map_detection_coverage", ":.0f")

    # countries with under MIN_HOURS of analysed footage grey: a rate from a few minutes is noise
    _country_map(country, person, "Pedestrians per minute", "YlOrRd", "map_pedestrians_per_minute", ":.2f")

    # pedestrians per minute per country with a 95% bootstrap interval over its localities; countries with at least
    # MIN_HOURS of analysed footage in at least 3 localities
    rng = np.random.default_rng(0)

    def boot(df: pl.DataFrame, *by) -> pl.DataFrame:
        """Pedestrians per minute per group of `by` with a 95% bootstrap interval over its localities (`rate`, `lo`,
        `hi`; groups with at least 3 localities)."""
        rows = []
        per_loc = df.group_by(*by, "id").agg(pl.col(person).sum(), pl.sum("minutes"))
        for key, g in per_loc.group_by(*by, maintain_order=True):  # in order: the same intervals on every run
            if g.height < 3:
                continue
            p, m = g[person].to_numpy(), g["minutes"].to_numpy()
            i = rng.integers(0, g.height, (1000, g.height))
            lo, hi = np.percentile(p[i].sum(1) / m[i].sum(1), [2.5, 97.5])
            rows.append(dict(zip(by, key)) | dict(rate=p.sum() / m.sum(), lo=lo, hi=hi))
        return pl.DataFrame(rows, schema={**{b: df.schema[b] for b in by}, "rate": pl.Float64, "lo": pl.Float64,
                                          "hi": pl.Float64})

    intervals = boot(d.sort("iso3", "id"), "iso3").join(
        country.select("iso3", "name", "continent", "hover", "hours", "localities"), on="iso3")
    ci = intervals.filter(pl.col("hours") >= MIN_HOURS).sort("rate")
    if ci.height:
        fig = go.Figure()
        for continent in CONTINENT_ORDER:
            c = ci.filter(pl.col("continent") == continent)
            fig.add_trace(go.Scatter(x=c["rate"], y=c["name"], mode="markers", name=continent,
                                     marker=dict(color=CONTINENT_COLORS[continent], size=9),
                                     error_x=dict(type="data", symmetric=False, array=c["hi"] - c["rate"],
                                                  arrayminus=c["rate"] - c["lo"], color="#999999", thickness=1.5),
                                     customdata=np.column_stack([c["lo"], c["hi"]]),
                                     **_hover_args(c, "Pedestrians per minute: %{x:.2f} (95% interval "
                                                      "%{customdata[0]:.2f}–%{customdata[1]:.2f})")))
        fig.update_xaxes(title_text="Pedestrians per minute (95% bootstrap interval over localities)",
                         showgrid=True, gridcolor="#e5e5e5", rangemode="tozero")
        # range fixed to the countries: autorange leaves several empty rows above and below
        fig.update_yaxes(categoryorder="array", categoryarray=ci["name"].to_list(), dtick=1,
                         range=[-0.6, ci.height - 0.4])
        _save(_style(fig, height=max(500, 22 * ci.height + 90), margin=dict(l=10, r=20, t=10, b=60),
                     legend=dict(x=0.99, xanchor="right", y=0.005, yanchor="bottom")), "dot_pedestrians_per_minute")

    # how much footage a stable estimate of a locality needs: the width of its 95% interval (bootstrap over its
    # segments) against its analysed hours
    widths = []
    for (lid,), g in d.select("id", person, "minutes").sort("id").group_by("id", maintain_order=True):
        p, m = g[person].to_numpy(), g["minutes"].to_numpy()
        if g.height < 5 or p.sum() == 0:
            continue
        i = rng.integers(0, g.height, (500, g.height))
        lo, hi = np.percentile(p[i].sum(1) / m[i].sum(1), [2.5, 97.5])
        widths.append(dict(id=lid, width=(hi - lo) / (p.sum() / m.sum()) * 100))
    w = (pl.DataFrame(widths, schema={"id": d.schema["id"], "width": pl.Float64}).filter(pl.col("width") > 0)
           .join(locality, on="id")
           .with_columns(_place(flag).alias("name")))
    if w.height > 2:
        x, y = np.log10(w["hours"].to_numpy()), np.log10(w["width"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        need = (f"; about {10 ** ((math.log10(50) - intercept) / slope):,.0f} hours for ±25%" if slope < 0 else "")
        _labelled_scatter(w, "hours", "width", "name",
                          pl.col("hours").rank("ordinal", descending=True) <= 15,
                          "Analysed footage (hours, log scale)",
                          "Width of the 95% interval (% of the rate, log scale)",
                          "scatter_interval_width_hours", trend=(10 ** xs, 10 ** (slope * xs + intercept)),
                          note=f"dotted line: width ~ hours^{slope:.2f}{need} · n = {w.height:,} localities")

    # walkable or car-dominated: pedestrians and cars per minute per locality, split at the medians
    quad = (locality.filter(pl.col("hours") >= 1, pl.col(person) > 0, pl.col("Cars") > 0)
            .with_columns(_place(flag).alias("name"))
            if "Cars" in classes else locality.clear())
    if quad.height > 3:
        mx, my = quad["Cars"].median(), quad[person].median()
        corners = {"Walkable: many pedestrians, few cars": "#009E73", "Busy: many pedestrians and cars": "#E69F00",
                   "Car-dominated: few pedestrians, many cars": "#D55E00",
                   "Quiet: few pedestrians and cars": "#56B4E9"}
        walk, busy, car, quiet = corners
        many_p, many_c = pl.col(person) >= my, pl.col("Cars") >= mx
        quad = quad.with_columns(pl.when(many_p & ~many_c).then(pl.lit(walk)).when(many_p).then(pl.lit(busy))
                                   .when(many_c).then(pl.lit(car)).otherwise(pl.lit(quiet)).alias("_q"),
                                 (pl.col(person) / pl.col("Cars")).alias("_ratio"))

        def medians(fig):
            for k, v in (("x", mx), ("y", my)):
                fig.add_shape(type="line", xref="x" if k == "x" else "paper", yref="paper" if k == "x" else "y",
                              x0=v if k == "x" else 0, x1=v if k == "x" else 1, y0=0 if k == "x" else v,
                              y1=1 if k == "x" else v, line=dict(color="#999999", width=1, dash="dash"), layer="below")

        _labelled_scatter(quad, "Cars", person, "name",
                          (pl.col("hours").rank("ordinal", descending=True) <= 15)
                          | (pl.col("_ratio").rank("ordinal", descending=True) <= 6)
                          | (pl.col("_ratio").rank("ordinal") <= 6),
                          "Cars per minute (log scale)", "Pedestrians per minute (log scale)",
                          "scatter_walkable_car_dominated", groups=("_q", corners), decorate=medians,
                          note=f"dashed lines: medians ({mx:.1f} cars, {my:.1f} pedestrians per minute) · "
                               f"n = {quad.height:,} localities with at least an hour")

    # small towns and big cities: pedestrians per minute by population of the locality, per continent
    bands = ["Under 100,000", "100,000–1 million", "1–5 million", "Over 5 million"]
    sized = (d.join(df_mapping.select("id", "population_locality"), on="id")
              .filter(pl.col("population_locality") > 0)
              .with_columns(pl.col("population_locality").cut([1e5, 1e6, 5e6], labels=bands, left_closed=True)
                            .cast(pl.Utf8).alias("band")))
    by_size = (boot(sized.sort("continent", "band", "id"), "continent", "band")
               .join(rates(sized, "continent", "band").select("continent", "band", "hours", "localities"),
                     on=["continent", "band"])
               .filter(pl.col("hours") >= 1)
               .with_columns((pl.col("hi") - pl.col("rate")).alias("plus"), (pl.col("rate") - pl.col("lo"))
                             .alias("minus")))
    if by_size.height:
        fig = px.bar(by_size.to_pandas(), x="continent", y="rate", color="band", barmode="group",
                     error_y="plus", error_y_minus="minus",
                     category_orders={"continent": CONTINENT_ORDER, "band": bands},
                     color_discrete_sequence=["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
                     hover_data={"hours": ":,.0f", "localities": True, "lo": ":.2f", "hi": ":.2f",
                                 "plus": False, "minus": False},
                     labels={"rate": "Pedestrians per minute (95% interval over localities)", "continent": "",
                             "band": "Population of locality"})
        fig.update_traces(error_y=dict(color="#555555", thickness=1.2))
        _save(_style(fig, legend=dict(x=0.99, xanchor="right", y=0.99)), "bar_pedestrians_city_size")

    # road-user mix of the localities with the most analysed footage, each as 100 squares
    top_loc = (locality.sort("hours", descending=True).head(30)
                       .with_columns((pl.col(person) / pl.sum_horizontal(classes)).alias("_p")).sort("_p",
                                                                                                     descending=True))
    if top_loc.height:
        colours = CLASS_COLOURS
        cols = 5
        rows_n = math.ceil(top_loc.height / cols)
        titles = [f"{flags.get(r['iso3'], '')} {r['locality']}" for r in top_loc.iter_rows(named=True)]
        row_h, gap = 350, 55  # px per row and between rows: cells about as tall as wide
        fig = make_subplots(rows=rows_n, cols=cols, subplot_titles=titles, horizontal_spacing=0.02,
                            vertical_spacing=gap / (row_h * rows_n))
        for n, r in enumerate(top_loc.iter_rows(named=True)):
            v = np.array([r[c] for c in classes])
            share = v / v.sum() * 100
            squares = np.floor(share).astype(int)  # largest remainders get the leftover squares: 100 in all
            squares[np.argsort(squares - share)[:100 - squares.sum()]] += 1
            cell = np.repeat(np.arange(len(classes)), squares)
            for k, c in enumerate(classes):
                idx = np.flatnonzero(cell == k)
                fig.add_trace(go.Scatter(x=idx % 10, y=9 - idx // 10, mode="markers", name=c, legendgroup=c,
                                         showlegend=n == 0, marker=dict(symbol="square", size=25, color=colours[c]),
                                         hovertemplate=f"{titles[n]}<br>{c}: {share[k]:.1f}% ({r[c]:.2f} per "
                                                       "minute)<extra></extra>"),
                              row=n // cols + 1, col=n % cols + 1)
        fig.update_xaxes(visible=False, range=[-0.6, 9.6], fixedrange=True)
        fig.update_yaxes(visible=False, range=[-0.6, 9.6], fixedrange=True)
        fig.update_annotations(font=dict(size=15))
        _save(_style(fig, width=1600, height=row_h * rows_n + 110, margin=dict(l=10, r=10, t=100, b=10),
                     legend=dict(orientation="h", x=0.5, xanchor="center", y=0.995, yanchor="top", yref="container",
                                 font=dict(size=16))), "waffle_road_user_mix", post_script=FIT_SQUARES_JS)

    # every class in the countries with the most analysed footage, on one log colour scale
    top = country.sort("hours", descending=True).head(40).sort("hours")
    z = top.select(classes).to_numpy()
    fig = go.Figure(go.Heatmap(z=np.log10(np.where(z > 0, z, np.nan)), x=classes, y=top["name"],
                               text=[[f"{v:.1f}" if v >= 0.1 else f"{v:.2f}" for v in row] for row in z],
                               texttemplate="%{text}",
                               colorscale="YlOrRd",
                               customdata=z, hovertemplate="%{y}<br>%{x}: %{customdata:.2f} per minute"
                                                           "<extra></extra>",
                               colorbar=dict(title="Per minute", tickvals=[-2, -1, 0, 1],
                                             ticktext=["0.01", "0.1", "1", "10"])))
    fig.update_yaxes(dtick=1)
    _save(_style(fig, height=max(600, 24 * top.height + 120), margin=dict(l=10, r=20, t=20, b=40)),
          "heatmap_road_users_per_minute")

    fig = px.box(locality.to_pandas(), x="continent", y=person, color="continent", points="outliers",
                 hover_name="hover", **CONTINENT_STYLE, labels={person: "Pedestrians per minute", "continent": ""})
    fig.update_traces(hoveron="points", hovertemplate=rate_line.replace("PLACE", "y") + hover.TEMPLATE)
    _save(_style(fig, showlegend=False), "box_pedestrians_per_minute_continent")

    # pedestrians per minute over segments of at least a minute: how much it varies within the data
    per_seg = d.filter(pl.col("minutes") >= 1).select((pl.col(person) / pl.col("minutes")).alias("rate"))
    per_seg = per_seg.filter(pl.col("rate") > 0)["rate"].to_numpy()
    if len(per_seg):
        edges = np.logspace(np.log10(per_seg.min()), np.log10(per_seg.max()), 40)
        counts, _ = np.histogram(per_seg, bins=edges)
        fig = px.bar(x=np.sqrt(edges[:-1] * edges[1:]), y=counts, log_x=True,
                     labels={"x": "Pedestrians per minute (per segment)", "y": "Number of segments"})
        fig.update_traces(width=np.diff(edges), marker_color="#0072B2")
        _save(_style(fig), "hist_pedestrians_per_minute_segment")

    # mix of detected road users per continent and overall, and in the countries with the most analysed footage
    by_continent = pl.concat([rates(d, "continent"), rates(d.with_columns(continent=pl.lit("All")), "continent")])
    for df, key, order, name in [(by_continent, "continent", CONTINENT_ORDER + ["All"], "bar_road_user_mix_continent"),
                                 (country.sort("hours", descending=True).head(30), "name", None, "bar_road_user_mix")]:
        mix = (df.unpivot(index=key, on=classes, variable_name="Road user", value_name="rate")
                 .with_columns((pl.col("rate") / pl.col("rate").sum().over(key) * 100).alias("share")))
        horizontal = key == "name"
        fig = px.bar(mix.to_pandas(), x="share" if horizontal else key, y=key if horizontal else "share",
                     color="Road user", orientation="h" if horizontal else "v",
                     category_orders={key: order or df[key].to_list(), "Road user": classes},
                     color_discrete_map=CLASS_COLOURS, labels={"share": "Share of detected road users (%)", key: ""},
                     hover_data={"rate": ":.2f", "share": ":.1f"})
        _save(_style(fig, height=900 if horizontal else None), name)

    # two-wheelers among the detected vehicles: motorcycle cultures stand out
    if all(c in classes for c in VEHICLES):
        share = country.with_columns((pl.sum_horizontal(TWO_WHEELERS) / pl.sum_horizontal(VEHICLES) * 100)
                                     .alias("two_wheelers"))
        _country_map(share, "two_wheelers", "Two-wheelers among vehicles (%)", "Purples", "map_two_wheeler_share",
                     ":.0f")

    # by day and by night: every class, per continent
    tod = (rates(d.with_columns(pl.when(pl.col("night")).then(pl.lit("Night")).otherwise(pl.lit("Day"))
                                .alias("time")), "continent", "time")
           .unpivot(index=["continent", "time", "hours"], on=classes, variable_name="Road user", value_name="rate"))
    fig = px.bar(tod.to_pandas(), x="Road user", y="rate", color="time", barmode="group", facet_col="continent",
                 facet_col_wrap=3, log_y=True, category_orders={"continent": CONTINENT_ORDER, "time": ["Day", "Night"],
                                                                "Road user": classes},
                 color_discrete_map={"Day": "#E69F00", "Night": "#0072B2"}, hover_data={"hours": ":,.0f"},
                 labels={"rate": "Per minute (log scale)", "Road user": "", "time": ""})
    fig.update_yaxes(tickvals=LOG_TICKS, ticktext=[f"{t:g}" for t in LOG_TICKS])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _save(_style(fig, height=900), "bar_road_users_day_night")

    night = rates(d, "iso3", "night").with_columns(pl.col(person).alias("rate"))
    day = night.filter(~pl.col("night")).select("iso3", pl.col("rate").alias("day"), pl.col("hours").alias("dh"))
    dark = night.filter(pl.col("night")).select("iso3", pl.col("rate").alias("night"), pl.col("hours").alias("nh"))
    day_night = (day.join(dark, on="iso3").filter(pl.col("dh") >= 1, pl.col("nh") >= 1)
                    .join(country.select("iso3", "name", "hover"), on="iso3").sort("day"))
    if day_night.height:
        _dumbbell(day_night, "name", {"day": ("Day", "#E69F00"), "night": ("Night", "#0072B2")},
                  "Pedestrians per minute", "dumbbell_pedestrians_day_night", fmt=":.2f")

    # by the vehicle the footage is filmed from, and by year of upload: what the detector sees depends on both
    vehicle = rates(d.drop_nulls("vehicle"), "vehicle").filter(pl.col("hours") >= 1).sort(person)  # rates need data
    fig = px.bar(vehicle.to_pandas(), x=person, y="vehicle", orientation="h",
                 text=[f"{h:,.0f} h analysed" for h in vehicle["hours"]],
                 labels={person: "Pedestrians per minute", "vehicle": ""})
    fig.update_traces(textposition="outside", marker_color="#0072B2", cliponaxis=False)
    _save(_style(fig, margin=dict(r=160)), "bar_pedestrians_vehicle_type")

    year = (rates(d.drop_nulls("year"), "year").filter(pl.col("hours") >= MIN_HOURS).sort("year")
            .unpivot(index=["year", "hours"], on=classes, variable_name="Road user", value_name="rate"))
    fig = px.line(year.to_pandas(), x="year", y="rate", color="Road user", markers=True, log_y=True,
                  category_orders={"Road user": classes}, hover_data={"hours": ":,.0f"},
                  color_discrete_map=CLASS_COLOURS,
                  labels={"rate": "Per minute (log scale)", "year": "Year of upload"})
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickvals=LOG_TICKS, ticktext=[f"{t:g}" for t in LOG_TICKS])
    _save(_style(fig), "line_road_users_upload_year")

    rates_ind = (country.join(_country_indicators(df_mapping), on="iso3", how="left")
                        .join(_public_indicators(df_mapping, seg), on="iso3", how="left"))
    _vs_indicators(rates_ind, person, "Pedestrians per minute", "scatter_indicators_pedestrians", log_y=False,
                   indicators=PEDESTRIAN_INDICATORS, min_hours=MIN_HOURS)

    # road users against wealth and road safety, as The Economist's charts: more or fewer than predicted (countries
    # with at least MIN_HOURS of analysed footage)
    gdp = gdp_per_capita()
    pop = df_mapping.group_by("iso3").agg(pl.col("population_country").filter(pl.col("population_country") > 0)
                                          .first())
    ctry = (country.filter(pl.col("hours") >= MIN_HOURS).join(pop, on="iso3").drop_nulls("population_country")
                   .join(_country_indicators(df_mapping).select("iso3", "traffic_mortality"), on="iso3", how="left"))
    if "Motorcycles" in classes and all(c in classes for c in VEHICLES):
        ctry = ctry.with_columns((pl.col("Motorcycles") / pl.sum_horizontal(VEHICLES) * 100).alias("motorcycle_share"))
    if gdp is not None:
        ctry = ctry.join(gdp, on="iso3", how="left")
        for col, title, name, ratio, what in [
                (person, "Pedestrians per minute (log scale)", "bubble_pedestrians_gdp", 1.5, "pedestrians"),
                ("motorcycle_share", "Motorcycles among detected vehicles (%, log scale)", "bubble_motorcycles_gdp", 2,
                 "motorcycle share"),
                ("Bicycles", "Bicycles per minute (log scale)", "bubble_bicycles_gdp", 2, "bicycles"),
                ("Cars", "Cars per minute (log scale)", "bubble_cars_gdp", 1.5, "cars")]:
            w = ctry.drop_nulls("gdp_pc").filter(pl.col(col) > 0) if col in ctry.columns else ctry.clear()
            if w.height > 2:
                _vs_wealth(w, col, title, name, ratio=ratio, what=what)
    # road deaths against pedestrians on the street: busy streets with few deaths, or quiet streets with many
    deaths = ctry.filter(pl.col("traffic_mortality") > 0, pl.col(person) > 0)
    if deaths.height > 2:
        _vs_trend(deaths, person, "Pedestrians per minute (log scale)", "traffic_mortality",
                  "Road traffic deaths per 100,000 people (log scale)", "bubble_road_deaths_vs_pedestrians", ratio=1.5,
                  what="road deaths", basis="pedestrian numbers")
    # pedestrian deaths alone (road deaths x the WHO's pedestrian share of them): how dangerous the street is for the
    # pedestrians on it
    share = pedestrian_death_share()
    if share is not None:
        ped_deaths = (ctry.join(share, on="iso3")
                          .with_columns((pl.col("traffic_mortality") * pl.col("pedestrian_share") / 100)
                                        .alias("pedestrian_deaths"))
                          .filter(pl.col("pedestrian_deaths") > 0, pl.col(person) > 0))
        if ped_deaths.height > 2:
            _vs_trend(ped_deaths, person, "Pedestrians per minute (log scale)", "pedestrian_deaths",
                      "Pedestrian road deaths per 100,000 people (log scale)",
                      "bubble_pedestrian_deaths_vs_pedestrians", ratio=1.5, what="pedestrian deaths",
                      basis="pedestrian numbers")

    # pedestrians per minute per locality against its population: whether more pedestrians just means a larger city
    loc = locality.filter(pl.col("population_locality") > 0, pl.col(person) > 0)
    if loc.height > 2:
        rho = loc.select(pl.corr("population_locality", person, method="spearman")).item()
        fig = px.scatter(loc.to_pandas(), x="population_locality", y=person, color="continent", hover_name="hover",
                         log_x=True, log_y=True, opacity=0.7, **CONTINENT_STYLE,
                         labels={"population_locality": "Population of locality (log scale)",
                                 person: "Pedestrians per minute (log scale)", "continent": ""})
        fig.update_traces(hovertemplate=rate_line.replace("PLACE", "y") + hover.TEMPLATE)
        fig.add_annotation(text=f"Spearman ρ = {rho:.2f}, n = {loc.height:,} localities", x=0.99, y=0.02,
                           xref="paper", yref="paper", xanchor="right", showarrow=False, font=dict(size=16))
        _save(_style(fig), "scatter_pedestrians_population")
        # localities with more or fewer pedestrians than their size predicts (at least an hour of analysed footage)
        loc = loc.filter(pl.col("hours") >= 1).with_columns(
            _place(flag).alias("name"))
        if loc.height > 2:
            _vs_trend(loc, "population_locality", "Population of locality (log scale)", person,
                      "Pedestrians per minute (log scale)", "scatter_locality_pedestrians_vs_population", ratio=2,
                      what="pedestrians", basis="population", bubble=False,
                      labelled=((pl.col("population_locality").rank("ordinal", descending=True) <= 10)
                                | (pl.col("_gap").rank("ordinal", descending=True) <= 8)
                                | (pl.col("_gap").rank("ordinal") <= 8)))

    # road users per US county and European region, and smoothed maps
    area_figures.detection_area_figures(df_mapping, d, classes, loc_hover, flags)

    # README tables
    two = dict(function=lambda v: f"{v:.2f}", return_dtype=pl.Utf8)  # rates with two decimals in the tables
    per_min = [pl.col(c).map_elements(**two).alias(f"{c} / min") for c in classes]
    footage = pl.concat([total.join(seg.group_by("iso3").agg(pl.first("continent")), on="iso3")
                              .group_by("continent").agg(pl.sum("hours")),
                         pl.DataFrame({"continent": ["All"], "hours": [total["hours"].sum()]})])
    t_continent = (by_continent.join(footage.rename({"hours": "total"}), on="continent")
                   .with_columns(pl.col("continent").replace_strict(
                       {c: i for i, c in enumerate(CONTINENT_ORDER + ["All"])}, default=None,
                       return_dtype=pl.Int64).alias("_o"))
                   .sort("_o")
                   .select(pl.col("continent").alias("Continent"), pl.col("hours").round(1).alias("Analysed (h)"),
                           (pl.col("hours") / pl.col("total") * 100).round(1).alias("Of footage (%)"),
                           pl.col("localities").alias("Localities"), *per_min))
    t_country = (ci.sort("rate", descending=True).head(20)
                   .select(pl.col("name").alias("Country"),
                           pl.col("rate").map_elements(**two).alias("Pedestrians / min"),
                           pl.format("{}–{}", pl.col("lo").map_elements(**two), pl.col("hi").map_elements(**two))
                           .alias("95% interval"),
                           pl.col("hours").round(1).alias("Analysed (h)"), pl.col("localities").alias("Localities")))
    t_locality = (locality.filter(pl.col("hours") >= 1).sort(person, descending=True).head(20)
                          .select(_place(flag).alias("Locality"),
                                  pl.col(person).map_elements(**two).alias("Pedestrians / min"),
                                  pl.col("hours").round(1).alias("Analysed (h)")))
    return [("Detections per continent", t_continent),
            (f"Top {t_country.height} countries by pedestrians per minute (at least {MIN_HOURS} analysed hours in at "
             "least 3 localities)", t_country),
            (f"Top {t_locality.height} localities by pedestrians per minute (at least 1 analysed hour)", t_locality)]


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
