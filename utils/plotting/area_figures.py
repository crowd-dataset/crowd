"""
Maps by area and 3D views of where the footage (and, with YOLO detections, the road users) is.

- County and region maps: localities are assigned to US counties and European NUTS-2 regions by their coordinates
  (point in polygon), and each area is coloured by its footage, footage per million people or share of night
  footage (or detections per minute).
- Spike maps: a country as a tilted flat map, one spike per locality.
- Smoothed maps: a value per locality spread over the land, with callouts at the highest and lowest places.

Outlines and populations are public downloads, cached in _output/geo_cache (refreshed every half year): US
counties (plotly's datasets) and their populations (US Census Bureau estimates), NUTS-2 regions (Eurostat GISCO)
and their populations (Eurostat). A figure whose download fails is skipped.
"""

import json
import math
import os
import time
import urllib.parse

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import polars as pl
import requests

import common
from custom_logger import CustomLogger
from utils.plotting import hover, map_labels
from utils.plotting.constants import MAP_TOP_SHARE, colorbar_top

logger = CustomLogger(__name__)

CACHE = os.path.join(common.output_dir, "geo_cache")
SOURCES = {
    "us_counties.json": "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    "us_county_population.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/"
                                "co-est2024-alldata.csv",
    "nuts2.json": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
                  "NUTS_RG_20M_2021_4326_LEVL_2.geojson",
    # every year since 2014: each region's latest (the UK's is 2019; some countries report the latest year late)
    "nuts2_population_since2014.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
                                       "demo_r_d2jan?sex=T&age=TOTAL&sinceTimePeriod=2014&format=JSON&lang=EN",
}
# European countries without NUTS regions, from geoBoundaries (CC BY 4.0): first-level regions where they are about
# the size of NUTS 2 regions, the whole country where the first level is much finer (Moldova, Kosovo) or the country
# is small; Russia's regions west of the Urals only (centre west of 60° E)
GEOBOUNDARIES = {"RUS": "ADM1", "UKR": "ADM1", "BLR": "ADM1", "BIH": "ADM1", "MDA": "ADM0", "XKX": "ADM0",
                 "FRO": "ADM0", "GGY": "ADM0", "IMN": "ADM0", "AND": "ADM0", "SMR": "ADM0", "MCO": "ADM0",
                 "GIB": "ADM0"}
SOURCES |= {f"gb_{iso3}_{level}.json": f"https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
                                       f"{iso3}/{level}/geoBoundaries-{iso3}-{level}_simplified.geojson"
            for iso3, level in GEOBOUNDARIES.items()}
# their regions' populations, by ISO 3166-2 code (each region's latest figure)
WIKIDATA_POPULATION = """SELECT ?code ?pop ?date WHERE { ?item wdt:P300 ?code; p:P1082 ?st. ?st ps:P1082 ?pop.
OPTIONAL { ?st pq:P585 ?date. } FILTER(STRSTARTS(?code, "RU-") || STRSTARTS(?code, "UA-") || STRSTARTS(?code, "BY-")
|| STRSTARTS(?code, "BA-")) }"""
SOURCES["region_population_wikidata.json"] = ("https://query.wikidata.org/sparql?format=json&query="
                                              + urllib.parse.quote(WIKIDATA_POPULATION))
EUROPE_SEL = (pl.col("continent") == "Europe") | pl.col("iso3").is_in(["RUS", "TUR"])  # their European regions
EUROPE_NOTE = ("Regions: Eurostat NUTS 2 (EU, EFTA, UK and candidate countries); geoBoundaries for European Russia, "
               "Ukraine, Belarus and Bosnia and Herzegovina, and whole countries for Moldova, Kosovo and small ones")
EUROPE_VIEW = dict(projection_type="natural earth", lonaxis_range=[-25, 62], lataxis_range=[34, 71.5])
# (lon min, lon max, lat min, lat max) of the spike maps
SPIKE_BOXES = {"USA": (-125, -66.5, 24, 49.5), "RUS": (19, 180, 41, 78), "JPN": (128.5, 146, 30, 46)}
MIN_HOURS_AREA = 1  # areas with less footage (or analysed footage) are left out of share and rate maps


def _fetch(name: str, max_age_days: int = 180) -> str | None:
    """Path of a cached download (see SOURCES); None when it cannot be downloaded."""
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_days * 86400:
        return path
    try:
        r = requests.get(SOURCES[name], timeout=300, headers={"User-Agent": "crowd-dataset figures"})
        r.raise_for_status()
        os.makedirs(CACHE, exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            f.write(r.content)
        os.replace(path + ".tmp", path)
    except requests.RequestException as e:
        logger.warning(f"Could not download {SOURCES[name]} ({e}); figures that need it are skipped.")
    return path if os.path.exists(path) else None


# --- areas ----------------------------------------------------------------------------------------------------------

def _rings(geometry: dict) -> list:
    """Outer rings of a GeoJSON polygon or multipolygon (holes are ignored)."""
    if geometry["type"] == "Polygon":
        return [[tuple(p) for p in geometry["coordinates"][0]]]
    return [[tuple(p) for p in polygon[0]] for polygon in geometry["coordinates"]]


def _clockwise(geometry: dict) -> dict:
    """The polygon(s) with clockwise outer rings, as in the NUTS file: plotly's maps fill a counter-clockwise ring as
    the whole globe except it (geoBoundaries draws them counter-clockwise)."""
    def fix(polygon):
        r = polygon[0]
        ccw = sum(x1 * y2 - x2 * y1 for (x1, y1, *_), (x2, y2, *_) in zip(r, r[1:])) > 0
        return [ring[::-1] for ring in polygon] if ccw else polygon
    coords = geometry["coordinates"]
    return dict(type=geometry["type"],
                coordinates=fix(coords) if geometry["type"] == "Polygon" else [fix(p) for p in coords])


def _areas(kind: str) -> tuple | None:
    """(features as {id: dict(name, rings, geometry)}, population by id) of US counties ("us") or NUTS-2 regions
    ("eu"); None when the outlines cannot be downloaded."""
    if kind == "us":
        path, pop_path = _fetch("us_counties.json"), _fetch("us_county_population.csv")
        if not path:
            return None
        with open(path, encoding="utf-8") as f:
            geo = json.load(f)
        pop, state = {}, {}
        if pop_path:
            with open(pop_path, "rb") as f:
                p = pl.read_csv(f.read(), encoding="latin1", infer_schema_length=0)
            p = p.filter(pl.col("SUMLEV") == "050")
            fips = p["STATE"] + p["COUNTY"]
            pop = dict(zip(fips, p["POPESTIMATE2024"].cast(pl.Int64)))
            state = dict(zip(fips, p["STNAME"]))
        feats = {g["id"]: dict(name=f"{g['properties']['NAME']} {g['properties'].get('LSAD', '')}".strip()
                               + (f", {state[g['id']]}" if g["id"] in state else ""),
                               rings=_rings(g["geometry"]), geometry=g["geometry"]) for g in geo["features"]}
        return feats, pop
    path, pop_path = _fetch("nuts2.json"), _fetch("nuts2_population_since2014.json")
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        geo = json.load(f)
    pop = {}
    if pop_path:
        with open(pop_path, encoding="utf-8") as f:
            d = json.load(f)
        years, values = d["size"][-1], d["value"]  # JSON-stat: geo, then time, are the last dimensions
        for code, i in d["dimension"]["geo"]["category"]["index"].items():
            latest = [values[k] for k in (str(i * years + t) for t in range(years)) if values.get(k)]
            if latest:
                pop[code] = float(latest[-1])
    feats = {g["properties"]["NUTS_ID"]: dict(name=f"{g['properties']['NAME_LATN']} ({g['properties']['CNTR_CODE']})",
                                              rings=_rings(g["geometry"]), geometry=g["geometry"])
             for g in geo["features"]}
    latest = {}  # ISO 3166-2 code -> (date, population)
    if wd := _fetch("region_population_wikidata.json"):
        with open(wd, encoding="utf-8") as f:
            for b in json.load(f)["results"]["bindings"]:
                row = (b.get("date", {}).get("value", ""), float(b["pop"]["value"]))
                latest[b["code"]["value"]] = max(latest.get(b["code"]["value"], row), row)
    for iso3, level in GEOBOUNDARIES.items():
        gb = _fetch(f"gb_{iso3}_{level}.json")
        if not gb:
            continue
        with open(gb, encoding="utf-8") as f:
            for g in json.load(f)["features"]:
                rings = _rings(g["geometry"])
                if iso3 == "RUS" and np.mean([min(x for x, _ in r) + max(x for x, _ in r) for r in rings]) / 2 >= 60:
                    continue  # east of the Urals
                # whole countries by their ISO3 code (their population comes from the mapping)
                area = iso3 if level == "ADM0" else f"{iso3}-{g['properties']['shapeName']}"
                feats[area] = dict(name=f"{g['properties']['shapeName']} ({iso3})", rings=rings,
                                   geometry=_clockwise(g["geometry"]))
                if g["properties"].get("shapeISO") in latest:
                    pop[area] = latest[g["properties"]["shapeISO"]][1]
    return feats, pop


def _assign(lon: np.ndarray, lat: np.ndarray, feats: dict) -> list:
    """The area id each point lies in (None outside all areas): bounding boxes first, then point in polygon."""
    ids = list(feats)
    boxes = []
    for i in ids:
        pts = np.array([p for r in feats[i]["rings"] for p in r])
        boxes.append([pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()])
    boxes = np.array(boxes)
    out = []
    for x, y in zip(lon, lat):
        hit = None
        for k in np.nonzero((boxes[:, 0] <= x) & (x <= boxes[:, 1]) & (boxes[:, 2] <= y) & (y <= boxes[:, 3]))[0]:
            if map_labels._inside(x, y, feats[ids[k]]["rings"]):
                hit = ids[k]
                break
        out.append(hit)
    return out


def _geojson(feats: dict, ids) -> dict:
    """GeoJSON of the given areas only, coordinates rounded to ~100 m, so the HTML stays small."""
    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 3), round(c[1], 3)]
    features = []
    for i in ids:
        geometry = feats[i]["geometry"]
        features.append(dict(type="Feature", id=i, properties={},
                             geometry=dict(type=geometry["type"], coordinates=rnd(geometry["coordinates"]))))
    return dict(type="FeatureCollection", features=features)


def _area_table(seg: pl.DataFrame, area_of: dict, df_mapping: pl.DataFrame, flags: dict) -> pl.DataFrame:
    """Footage per area: area, hours, night (%), localities, and the largest localities (for the hover)."""
    loc = (df_mapping.select("id", "locality", "iso3").with_columns(
        pl.col("id").replace_strict(area_of, default=None, return_dtype=pl.Utf8).alias("area"))
        .drop_nulls("area"))
    per_loc = (seg.join(loc, on="id").group_by("area", "id", "locality", "iso3")
                  .agg((pl.sum("seconds") / 3600).alias("h"),
                       (pl.col("seconds").filter(pl.col("night")).sum() / 3600).alias("nh")))
    per_loc = per_loc.with_columns(pl.format("{} {}: {} h", pl.col("iso3").replace_strict(flags, default="🏳️"),
                                             pl.col("locality"), pl.col("h").round(1)).alias("line"))
    return (per_loc.sort("h", descending=True).group_by("area")
                   .agg(pl.sum("h").alias("hours"), (pl.sum("nh") / pl.sum("h") * 100).alias("night"),
                        pl.len().alias("localities"), pl.col("line").head(6).alias("top")))


def _area_map(table: pl.DataFrame, feats: dict, pop: dict, value: str, title: str, scale: str, name: str,
              log: bool, fmt: str, us: bool, note: str = ""):
    """A map of the areas in `table` coloured by `value` (log colour scale if `log`)."""
    from utils.plotting import dataset_figures as dfig  # lazy: dataset_figures imports this module
    t = table.filter(pl.col(value) > 0) if log else table.drop_nulls(value)
    if not t.height:
        return
    z = np.log10(t[value].to_numpy()) if log else t[value].to_numpy()
    text = [f"<b>{feats[a]['name']}</b><br>{title}: {format(v, fmt)}<br>Footage: {h:,.1f} h in {n:,} "
            f"{'locality' if n == 1 else 'localities'}, {night:.0f}% at night"
            + (f"<br>Population: {pop[a]:,.0f}" if pop.get(a) else "") + "<br>──────────<br>" + "<br>".join(top)
            for a, v, h, n, night, top in t.select("area", pl.col(value).alias("_v"), "hours", "localities", "night",
                                                   "top").iter_rows()]
    colorbar = colorbar_top(title)
    if log:
        steps = (1,) if z.max() - z.min() > 2.5 else (1, 2, 5)  # powers of ten; with a narrow range also 2x, 5x
        ticks = [m * 10 ** p for p in range(math.floor(z.min()), math.ceil(z.max()) + 1) for m in steps
                 if z.min() <= math.log10(m * 10 ** p) <= z.max()]
        colorbar.update(tickvals=np.log10(ticks).tolist(), ticktext=[f"{t_:,g}" for t_ in ticks])
    fig = go.Figure(go.Choropleth(geojson=_geojson(feats, t["area"]), locations=t["area"], z=z, featureidkey="id",
                                  colorscale=scale, marker_line_width=0.2, marker_line_color="#888888",
                                  colorbar=colorbar, hovertext=text, hovertemplate="%{hovertext}<extra></extra>"))
    if us:
        fig.update_geos(scope="usa", showsubunits=True, subunitcolor="#999999", showland=True, landcolor="#f2f2f2")
    else:
        fig.update_geos(showcountries=True, countrycolor="#999999", showland=True, landcolor="#f2f2f2",
                        showocean=True, oceancolor="#ffffff", **EUROPE_VIEW)
    fig.update_geos(domain=dict(x=[0, 1], y=[0, MAP_TOP_SHARE]), showframe=False)
    if note:
        fig.add_annotation(text=note, x=0.01, y=0.01, xref="paper", yref="paper", showarrow=False,
                           font=dict(size=14, color="#666666"), xanchor="left", bgcolor="rgba(255,255,255,0.85)")
    dfig._save(dfig._style(fig, margin=dict(l=0, r=0, t=10, b=0)), name)


def _country_population(pop: dict, df_mapping: pl.DataFrame) -> dict:
    """`pop` with the mapping's population of the countries that are one area (GEOBOUNDARIES at ADM0)."""
    whole = [iso3 for iso3, level in GEOBOUNDARIES.items() if level == "ADM0"]
    return pop | dict(df_mapping.filter(pl.col("iso3").is_in(whole), pl.col("population_country").cast(pl.Float64) > 0)
                      .group_by("iso3").agg(pl.col("population_country").cast(pl.Float64).first()).iter_rows())


def _area_maps(seg: pl.DataFrame, df_mapping: pl.DataFrame, flags: dict):
    """Footage, footage per million people and night share per US county and per European region."""
    for kind, prefix, sel in [("us", "map_us_counties", pl.col("iso3") == "USA"),
                              ("eu", "map_eu_regions", EUROPE_SEL)]:
        areas = _areas(kind)
        if areas is None:
            continue
        feats, pop = areas
        pop = pop if kind == "us" else _country_population(pop, df_mapping)
        pts = df_mapping.filter(sel).drop_nulls(["lat", "lon"])
        area_of = dict(zip(pts["id"], _assign(pts["lon"].to_numpy(), pts["lat"].to_numpy(), feats)))
        table = _area_table(seg, {k: v for k, v in area_of.items() if v}, df_mapping, flags)
        table = table.with_columns(pl.col("area").replace_strict(pop, default=None, return_dtype=pl.Float64)
                                   .alias("population"))
        table = table.with_columns((pl.col("hours") / pl.col("population") * 1e6).alias("per_million"))
        us = kind == "us"
        what = "county" if us else "region"
        _area_map(table, feats, pop, "hours", "Footage (hours)", "YlOrRd", f"{prefix}_footage", True, ",.1f", us,
                  note="" if us else EUROPE_NOTE)
        _area_map(table, feats, pop, "per_million", "Hours per million people", "YlGnBu",
                  f"{prefix}_footage_per_capita", True, ",.1f", us,
                  note="Population: US Census Bureau estimates, 2024" if us else
                  "Population: Eurostat, each region's latest year (UK: 2019); Russia, Ukraine, Belarus and Bosnia "
                  "and Herzegovina: Wikidata, latest figure; whole countries: their population")
        # muted purples without the palest step: near-white regions would look like water
        _area_map(table.filter(pl.col("hours") >= MIN_HOURS_AREA), feats, pop, "night", "Night footage (%)",
                  plotly.colors.sequential.Purp[1:], f"{prefix}_night_share", False, ".0f", us,
                  note=f"Each {what} with at least {MIN_HOURS_AREA} hour of footage")


def _detection_area_maps(d: pl.DataFrame, df_mapping: pl.DataFrame, flags: dict, classes: list):
    """Pedestrians per minute per US county and European region, and bicycles per minute per European region
    (`d`: one row per analysed segment with `id`, `minutes` and a count column per class)."""
    person = classes[0]
    for kind, prefix, sel, cols in [("us", "map_us_counties", pl.col("iso3") == "USA", [person]),
                                    ("eu", "map_eu_regions", EUROPE_SEL,
                                     [person] + (["Bicycles"] if "Bicycles" in classes else []))]:
        areas = _areas(kind)
        if areas is None:
            continue
        feats, pop = areas
        pop = pop if kind == "us" else _country_population(pop, df_mapping)
        pts = df_mapping.filter(sel).drop_nulls(["lat", "lon"])
        area_of = {k: v for k, v in zip(pts["id"], _assign(pts["lon"].to_numpy(), pts["lat"].to_numpy(), feats)) if v}
        dd = d.with_columns(pl.col("id").replace_strict(area_of, default=None, return_dtype=pl.Utf8).alias("area"))
        rates = (dd.drop_nulls("area").group_by("area")
                   .agg(*[(pl.col(c).sum() / pl.sum("minutes")).alias(c) for c in cols],
                        (pl.sum("minutes") / 60).alias("analysed")))
        table = _area_table(d.select("id", "seconds", "night"), area_of, df_mapping, flags).join(rates, on="area")
        table = table.filter(pl.col("analysed") >= MIN_HOURS_AREA)
        us = kind == "us"
        for c, word in zip(cols, ["pedestrians", "bicycles"]):
            _area_map(table, feats, pop, c, f"{word.capitalize()} per minute", "YlOrRd", f"{prefix}_{word}", False,
                      ".2f", us, note=f"Each {'county' if us else 'region'} with at least {MIN_HOURS_AREA} analysed "
                                      "hour")


# --- spikes -----------------------------------------------------------------------------------------------

def _blur(grid: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    """Gaussian blur (sigmas in cells), with numpy only."""
    for axis, sigma in ((0, sigma_y), (1, sigma_x)):
        r = max(1, int(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        grid = np.apply_along_axis(lambda v: np.convolve(v, k / k.sum(), mode="same"), axis, grid)
    return grid


def _surface(lon, lat, w, box, cell: float, sigma_km: float = 40) -> tuple:
    """(lon centres, lat centres, grid) of the smoothed sum of weights `w` per cell."""
    nx, ny = int((box[1] - box[0]) / cell), int((box[3] - box[2]) / cell)
    rng = [[box[2], box[3]], [box[0], box[1]]]
    sy = sigma_km / 111 / cell
    sx = sy / max(math.cos(math.radians((box[2] + box[3]) / 2)), 0.2)
    weight, _, _ = np.histogram2d(lat, lon, bins=[ny, nx], range=rng, weights=w)
    grid = _blur(weight, sy, sx)
    return (np.linspace(box[0], box[1], nx + 1)[:-1] + cell / 2, np.linspace(box[2], box[3], ny + 1)[:-1] + cell / 2,
            grid)


def _borders(box) -> tuple:
    """Country borders within the box as one line (x, y lists with None between rings)."""
    xs, ys = [], []
    for shape in map_labels.country_shapes().values():
        for ring in shape["rings"]:
            pts = [(x, y) for x, y in ring if box[0] - 1 <= x <= box[1] + 1 and box[2] - 1 <= y <= box[3] + 1]
            if len(pts) > 1:
                xs += [p[0] for p in pts] + [None]
                ys += [p[1] for p in pts] + [None]
    return xs, ys


def _scene(box, z_max: float, distance: float = 1.0) -> dict:
    """A tilted 3D view of a flat map: the map's proportions, low relief, seen from the south (`distance`: of the
    camera, relative to the default)."""
    w = (box[1] - box[0]) * math.cos(math.radians((box[2] + box[3]) / 2))
    h = box[3] - box[2]
    m = max(w, h)
    hidden = dict(visible=False, showbackground=False, showspikes=False)
    return dict(xaxis=dict(range=box[:2], **hidden), yaxis=dict(range=box[2:], **hidden),
                zaxis=dict(range=[0, z_max * 1.05], **hidden), aspectmode="manual",
                aspectratio=dict(x=w / m * 1.6, y=h / m * 1.6, z=0.45), dragmode="turntable",
                camera=dict(eye=dict(x=0.08 * distance, y=-0.95 * distance, z=0.62 * distance),
                            center=dict(x=0, y=0.02, z=-0.18),
                            up=dict(x=0, y=0, z=1)))


def _in_box(df: pl.DataFrame, box) -> pl.DataFrame:
    return df.filter(pl.col("lon").is_between(box[0], box[1]), pl.col("lat").is_between(box[2], box[3]))


def _spike_map(locs: pl.DataFrame, iso3: str, name: str, country: str):
    """A country as a tilted flat map with a spike per locality, its height (log scale) and colour the hours."""
    from utils.plotting import dataset_figures as dfig
    box = SPIKE_BOXES[iso3]
    p = _in_box(locs.filter(pl.col("iso3") == iso3, pl.col("hours") > 0), box).sort("hours")
    if not p.height:
        return
    lo, top = math.log10(0.1), math.log10(p["hours"].max())
    h = (np.log10(np.maximum(p["hours"].to_numpy(), 0.1)) - lo) / (top - lo)
    n = p.height
    x = np.column_stack([p["lon"], p["lon"], np.full(n, np.nan)]).ravel()
    y = np.column_stack([p["lat"], p["lat"], np.full(n, np.nan)]).ravel()
    z = np.column_stack([np.zeros(n), h, np.full(n, np.nan)]).ravel()
    c = np.column_stack([h, h, h]).ravel()
    bx, by = _borders(box)
    ticks = [t for t in (0.1, 1, 10, 100, 1000) if lo <= math.log10(t) <= top]
    fig = go.Figure([
        go.Surface(x=list(box[:2]), y=list(box[2:]), z=[[0, 0], [0, 0]], colorscale=[[0, "#e9eef2"], [1, "#e9eef2"]],
                   showscale=False, hoverinfo="skip"),
        go.Scatter3d(x=bx, y=by, z=[0.002 if v is not None else None for v in bx], mode="lines",
                     line=dict(color="#666666", width=1.5), hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip", showlegend=False,
                     line=dict(color=c, colorscale="YlOrRd", cmin=0, cmax=1, width=3, showscale=True,
                               colorbar=dict(title=dict(text="Footage (hours)", side="top"), orientation="h", x=0.5,
                                             y=0.93, len=0.4, thickness=14,
                                             tickvals=[(math.log10(t) - lo) / (top - lo) for t in ticks],
                                             ticktext=[f"{t:g}" for t in ticks]))),
        go.Scatter3d(x=p["lon"], y=p["lat"], z=h, mode="markers", showlegend=False,
                     marker=dict(size=2, color=h, colorscale="YlOrRd", cmin=0, cmax=1),
                     text=p["hover"], hovertemplate=hover.GL_TEMPLATE)])
    big = p.sort("hours", descending=True).head(10)
    fig.add_trace(go.Scatter3d(x=big["lon"], y=big["lat"], z=(np.log10(big["hours"].to_numpy()) - lo) / (top - lo)
                               + 0.04, mode="text", text=big["name"], textfont=dict(size=12, color="#222222"),
                               hoverinfo="skip", showlegend=False))
    fig.update_layout(scene=_scene(box, 1.0))
    fig.add_annotation(text=f"{country}: one spike per locality, height and colour the hours of footage (log scale)",
                       x=0.01, y=0.01, xref="paper", yref="paper", showarrow=False, xanchor="left",
                       font=dict(size=15, color="#555555"))
    dfig._save(dfig._style(fig, width=1600, height=1000, margin=dict(l=0, r=0, t=0, b=0)), name, save_eps=False)


# --- smoothed maps --------------------------------------------------------------------------------------------------
# A value per locality spread over a fine grid of land (as maps of "comfortable weather days"), with country names
# and callouts at the highest and lowest places. Rates and shares are only shown where there is enough nearby
# footage; elsewhere the land is grey rather than filled in from distant localities.

FIELD_CELL = {"Europe": 0.2, "North America": 0.3, "Asia": 0.35}
FIELD_BOXES = {"Europe": (-25, 45, 34, 71.5), "North America": (-128, -60, 14, 58), "Asia": (60, 150, -10, 55)}
LAND_GREY = "#d9d9d9"


def _land_mask(box, cell: float) -> np.ndarray:
    """Boolean grid (rows: latitude upwards) of the cells on land, from the country outlines."""
    from PIL import Image, ImageDraw
    nx, ny = int((box[1] - box[0]) / cell), int((box[3] - box[2]) / cell)
    img = Image.new("1", (nx, ny), 0)
    draw = ImageDraw.Draw(img)
    for shape in map_labels.country_shapes().values():
        for ring in shape["rings"]:
            xs = [p[0] for p in ring]
            if max(xs) - min(xs) > 180:  # crosses the antimeridian (e.g., Russia's far east): unwrap to the east
                ring = [(x + 360 if x < 0 else x, y) for x, y in ring]
                xs = [p[0] for p in ring]
            if max(xs) < box[0] or min(xs) > box[1]:
                continue
            draw.polygon([((x - box[0]) / cell, (y - box[2]) / cell) for x, y in ring], fill=1)
    return np.array(img, dtype=bool)


def _distance_grid(lons, lats, pts_lon, pts_lat) -> np.ndarray:
    """Great-circle distance (km) from each grid cell to the nearest point, in chunks (numpy only)."""
    gx, gy = np.meshgrid(np.radians(lons), np.radians(lats))
    cell = np.column_stack([np.cos(gy.ravel()) * np.cos(gx.ravel()), np.cos(gy.ravel()) * np.sin(gx.ravel()),
                            np.sin(gy.ravel())])
    px, py = np.radians(np.asarray(pts_lon)), np.radians(np.asarray(pts_lat))
    pts = np.column_stack([np.cos(py) * np.cos(px), np.cos(py) * np.sin(px), np.sin(py)])
    best = np.empty(len(cell))
    for i in range(0, len(cell), 4000):
        best[i:i + 4000] = (cell[i:i + 4000] @ pts.T).max(axis=1)
    return (6371 * np.arccos(np.clip(best, -1, 1))).reshape(gx.shape)


def _country_names(box) -> tuple:
    """Names and label points of the larger countries inside the box."""
    import pycountry
    names, xs, ys = [], [], []
    for iso3, shape in map_labels.country_shapes().items():
        ct = shape.get("ct")
        if not ct or not (box[0] + 1 < ct[0] < box[1] - 1 and box[2] + 1 < ct[1] < box[3] - 1):
            continue
        ring_pts = [p for r in shape["rings"] for p in r]
        span = max(p[0] for p in ring_pts) - min(p[0] for p in ring_pts)
        if span < (box[1] - box[0]) / 25:  # too small for a name at this scale
            continue
        c = pycountry.countries.get(alpha_3=iso3)
        if c:
            names.append(getattr(c, "common_name", None) or c.name)
            xs.append(ct[0])
            ys.append(ct[1])
    return names, xs, ys


def _field_map(lons, lats, grid, shown, land, box, title: str, colorscale, name: str, note: str, callouts: list,
               log: bool = False, fmt: str = ".1f", zrange=None):
    """Draw a smoothed field: `grid` where `shown`, grey elsewhere on land, white sea; borders, country names and
    `callouts` (lon, lat, text) with arrows."""
    from utils.plotting import dataset_figures as dfig
    z = np.where(shown, np.log10(np.maximum(grid, 1e-9)) if log else grid, np.nan)
    grey = np.where(land & ~shown, 1.0, np.nan)
    colorbar = dict(title=dict(text=title, side="top"), orientation="h", x=0.5, xanchor="center", y=1.0,
                    yanchor="bottom", len=0.5, thickness=14)
    if log:
        lo, hi = np.nanmin(z), np.nanmax(z)
        ticks = [m * 10 ** p for p in range(math.floor(lo), math.ceil(hi) + 1) for m in (1, 3)
                 if lo <= math.log10(m * 10 ** p) <= hi]
        colorbar.update(tickvals=np.log10(ticks).tolist(), ticktext=[f"{t:,g}" for t in ticks])
    customdata = np.where(shown, grid, np.nan)
    fig = go.Figure([
        go.Heatmap(x=lons, y=lats, z=grey, colorscale=[[0, LAND_GREY], [1, LAND_GREY]], showscale=False,
                   hoverinfo="skip"),
        go.Heatmap(x=lons, y=lats, z=z, colorscale=colorscale, colorbar=colorbar, customdata=customdata,
                   zmin=zrange[0] if zrange else None, zmax=zrange[1] if zrange else None,
                   hovertemplate=f"{title}: %{{customdata:{fmt}}}<br>%{{y:.1f}}°, %{{x:.1f}}°<extra></extra>")])
    bx, by = _borders(box)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color="#555555", width=0.6), hoverinfo="skip",
                             showlegend=False))
    names, xs, ys = _country_names(box)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=names, hoverinfo="skip", showlegend=False,
                             textfont=dict(size=11, color="#222222")))
    for lon, lat, text in callouts:  # the box points inwards, so it stays clear of the colour bar and the edges
        fig.add_annotation(x=lon, y=lat, text=text, showarrow=True, arrowhead=2, arrowwidth=1.2, arrowcolor="#333333",
                           ax=70 if lon < (box[0] + box[1]) / 2 else -70,
                           ay=50 if lat > box[2] + 0.7 * (box[3] - box[2]) else -50,
                           font=dict(size=12, color="#111111"),
                           bgcolor="rgba(255,255,255,0.85)", bordercolor="#888888", borderwidth=0.5, borderpad=3)
    if note:
        fig.add_annotation(text=note, x=0.0, y=-0.02, xref="paper", yref="paper", showarrow=False, xanchor="left",
                           yanchor="top", font=dict(size=13, color="#666666"))
    mid = math.cos(math.radians((box[2] + box[3]) / 2))
    fig.update_xaxes(range=box[:2], visible=False, constrain="domain")
    fig.update_yaxes(range=box[2:], visible=False, scaleanchor="x", scaleratio=1 / mid, constrain="domain")
    dfig._save(dfig._style(fig, width=1400, height=1200, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=110, b=50)), name, save_eps=False)


def _extremes(lons, lats, grid, shown, pts: pl.DataFrame, n_high: int, n_low: int, fmt, min_sep: float = 6,
              words=("Highest", "Lowest")) -> list:
    """Callouts at the highest and lowest cells that are shown, at least `min_sep` degrees apart, each named after
    the locality with the most footage within ~100 km."""
    out, used = [], []
    order = np.argsort(np.where(shown, grid, np.nan), axis=None)
    valid = order[~np.isnan(np.where(shown, grid, np.nan).ravel()[order])]
    for idx_list, n, word in ((valid[::-1], n_high, "high"), (valid, n_low, "low")):
        found = 0
        for idx in idx_list:
            if found >= n:
                break
            i, j = np.unravel_index(idx, grid.shape)
            lon, lat = lons[j], lats[i]
            if any(abs(lon - u) < min_sep and abs(lat - v) < min_sep / 2 for u, v in used):
                continue
            near = pts.filter((pl.col("lon") - lon).abs() < 1.2, (pl.col("lat") - lat).abs() < 0.9)
            if not near.height:
                continue
            place = near.sort("weight", descending=True)["name"][0]
            used.append((lon, lat))
            out.append((lon, lat, f"{words[0] if word == 'high' else words[1]} around {place}: {fmt(grid[i, j])}"))
            found += 1
    return out


def _smoothed(pts: pl.DataFrame, value: str | None, box, cell, sigma_km=50, min_weight=1.0, max_km=100) -> tuple:
    """(lons, lats, grid, shown): with `value`, the weighted mean of value (weights `weight`), shown within `max_km`
    of a locality with at least `min_weight`; without, the density of `weight` per 10,000 km² where there is any."""
    p = _in_box(pts, box)
    lons, lats, weight = _surface(p["lon"].to_numpy(), p["lat"].to_numpy(), p["weight"].to_numpy(), box, cell,
                                  sigma_km=sigma_km)
    if value is None:
        area = (cell * 111) ** 2 * np.cos(np.radians(lats))[:, None]  # km² per cell
        density = weight / area * 1e4
        return lons, lats, density, density > density.max() * 1e-4
    _, _, num = _surface(p["lon"].to_numpy(), p["lat"].to_numpy(), (p["weight"] * p[value]).to_numpy(), box, cell,
                         sigma_km=sigma_km)
    mean = num / np.maximum(weight, 1e-12)
    solid = p.filter(pl.col("weight") >= min_weight)
    if not solid.height:
        return lons, lats, mean, np.zeros_like(mean, dtype=bool)
    return lons, lats, mean, _distance_grid(lons, lats, solid["lon"], solid["lat"]) <= max_km


def _field_maps(locs: pl.DataFrame):
    """Footage density and distance to footage (Europe, North America, Asia), night share and upload year (Europe)
    as smoothed maps. locs: lon, lat, name, hours, night_hours, year (footage-weighted mean upload year)."""
    pts = locs.with_columns(pl.col("hours").alias("weight"))
    for region, box in FIELD_BOXES.items():
        cell = FIELD_CELL[region]
        slug = region.lower().replace(" ", "_")
        lons, lats, dens, shown = _smoothed(pts, None, box, cell)
        land = _land_mask(box, cell)
        shown &= land
        if shown.any():
            calls = _extremes(lons, lats, dens, shown, pts, 3, 0, lambda v: f"{v:,.0f} h per 10,000 km²", 8)
            _field_map(lons, lats, dens, shown, land, box, "Footage (hours per 10,000 km², smoothed over ~50 km)",
                       "YlOrRd", f"map_density_footage_{slug}", f"{region}: where the footage is", calls, log=True,
                       fmt=",.1f")
        with_footage = _in_box(pts.filter(pl.col("hours") >= 1), box)
        if with_footage.height:
            dist = _distance_grid(lons, lats, with_footage["lon"], with_footage["lat"])
            far = np.where(land, dist, -1)
            calls = []
            for _ in range(3):  # the furthest places from any footage, one per area
                i, j = np.unravel_index(np.argmax(far), far.shape)
                if far[i, j] <= 0:
                    break
                calls.append((lons[j], lats[i], f"{far[i, j]:,.0f} km from any footage"))
                far[max(0, i - int(8 / cell)):i + int(8 / cell), max(0, j - int(12 / cell)):j + int(12 / cell)] = -1
            _field_map(lons, lats, np.minimum(dist, 500), land, land, box,
                       "Distance to the nearest locality with at least an hour of footage (km)",
                       [[0, "#1a9850"], [0.2, "#91cf60"], [0.4, "#fee08b"], [0.7, "#fc8d59"], [1, "#d73027"]],
                       f"map_distance_to_footage_{slug}", f"{region}: blind spots of the dataset (distances over "
                       "500 km shown as 500)", calls, fmt=",.0f", zrange=(0, 500))
    box, cell = FIELD_BOXES["Europe"], FIELD_CELL["Europe"]
    for value, title, scale, name, fmt, lows, words in [
            ("night_pct", "Footage at night (%)", "Blues", "map_night_share_smoothed_europe", lambda v: f"{v:.0f}%",
             0, ("Most night footage", "")),
            ("year", "Average year of upload", "Viridis", "map_upload_year_europe", lambda v: f"{v:.0f}", 2,
             ("Newest footage", "Oldest footage"))]:
        p = pts.drop_nulls(value) if value in pts.columns else pts.clear()
        if not p.height:
            continue
        lons, lats, mean, shown = _smoothed(p, value, box, cell)
        land = _land_mask(box, cell)
        shown &= land
        calls = _extremes(lons, lats, mean, shown, p, 2, lows, fmt, words=words)
        _field_map(lons, lats, mean, shown, land, box, f"{title} (smoothed over ~50 km, weighted by footage)", scale,
                   name, "Europe: grey where there is too little footage nearby", calls, fmt=".1f")


def _detection_field_maps(d: pl.DataFrame, df_mapping: pl.DataFrame, flags: dict, classes: list):
    """Pedestrians per minute, walkable street minutes, motorcycle share and bicycles per minute as smoothed maps
    of Europe (`d`: one row per analysed segment with id, minutes and a count column per class)."""
    person = classes[0]
    seg = d.with_columns((pl.col(person) / pl.col("minutes")).alias("_p"),
                         (pl.col("Cars") / pl.col("minutes")).alias("_c") if "Cars" in classes else pl.lit(0.0)
                         .alias("_c"))
    # walkable street minutes: segments with at least 2 pedestrians and at most 10 cars per minute
    walkable = (pl.col("_p") >= 2) & (pl.col("_c") <= 10)
    vehicles = [c for c in ("Cars", "Bicycles", "Motorcycles", "Buses", "Trucks") if c in classes]
    per_loc = (seg.group_by("id").agg(
        pl.sum("minutes").alias("weight"), (pl.col(person).sum() / pl.sum("minutes")).alias("pedestrians"),
        (pl.col("minutes").filter(walkable).sum() / pl.sum("minutes") * 100).alias("walkable"),
        *([(pl.col("Motorcycles").sum() / pl.sum_horizontal(*[pl.col(c).sum() for c in vehicles]) * 100)
           .alias("motorcycles")] if "Motorcycles" in classes else []),
        *([(pl.col("Bicycles").sum() / pl.sum("minutes")).alias("bicycles")] if "Bicycles" in classes else []))
        .join(df_mapping.select("id", "locality", "iso3", "lat", "lon"), on="id").drop_nulls(["lat", "lon"])
        .with_columns(pl.concat_str([pl.col("iso3").replace_strict(flags, default="🏳️"), pl.col("locality")],
                                    separator=" ").alias("name")))
    box, cell = FIELD_BOXES["Europe"], FIELD_CELL["Europe"]
    for value, title, scale, name, fmt in [
            ("pedestrians", "Pedestrians per minute", "YlOrRd", "map_pedestrians_smoothed_europe",
             lambda v: f"{v:.2f} per minute"),
            ("walkable", "Walkable street minutes (%: ≥2 pedestrians and ≤10 cars per minute)", "YlGn",
             "map_walkable_minutes_europe", lambda v: f"{v:.0f}%"),
            ("motorcycles", "Motorcycles among detected vehicles (%)", "Purples", "map_motorcycle_share_europe",
             lambda v: f"{v:.1f}%"),
            ("bicycles", "Bicycles per minute", "Greens", "map_bicycles_smoothed_europe",
             lambda v: f"{v:.2f} per minute")]:
        if value not in per_loc.columns:
            continue
        p = per_loc.drop_nulls(value).filter(pl.col("weight") > 0)
        if p.height < 3:
            continue
        lons, lats, mean, shown = _smoothed(p, value, box, cell, min_weight=60)  # an analysed hour
        land = _land_mask(box, cell)
        shown &= land
        calls = _extremes(lons, lats, mean, shown, p, 2, 2, fmt)
        _field_map(lons, lats, mean, shown, land, box, f"{title}, smoothed over ~50 km", scale, name,
                   "Europe: grey where there is too little analysed footage nearby", calls, fmt=".2f")


# --- entry points ---------------------------------------------------------------------------------------------------

def _localities(seg: pl.DataFrame, df_mapping: pl.DataFrame, loc_hover: pl.DataFrame, flags: dict) -> pl.DataFrame:
    year = pl.col("year").filter(pl.col("year").is_between(2005, 2100))
    weights = pl.col("seconds").filter(pl.col("year").is_between(2005, 2100))
    return (seg.group_by("id").agg((pl.sum("seconds") / 3600).alias("hours"),
                                   (pl.col("seconds").filter(pl.col("night")).sum() / pl.sum("seconds") * 100)
                                   .alias("night_pct"),
                                   ((year * weights).sum() / weights.sum()).alias("year"))
               .join(df_mapping.select("id", "locality", "iso3", "lat", "lon"), on="id").join(loc_hover, on="id")
               .drop_nulls(["lat", "lon"])
               .with_columns(pl.concat_str([pl.col("iso3").replace_strict(flags, default="🏳️"), pl.col("locality")],
                                           separator=" ").alias("name")))


def dataset_area_figures(df_mapping: pl.DataFrame, seg: pl.DataFrame, loc_hover: pl.DataFrame, flags: dict):
    """County and region maps, spike maps and smoothed maps of the footage (from the mapping)."""
    _area_maps(seg, df_mapping, flags)
    locs = _localities(seg, df_mapping, loc_hover, flags)
    for iso3, country in [("USA", "United States"), ("RUS", "Russia"), ("JPN", "Japan")]:
        _spike_map(locs, iso3, f"spikes_footage_{iso3.lower()}", country)
    _field_maps(locs)


def detection_area_figures(df_mapping: pl.DataFrame, d: pl.DataFrame, classes: list, loc_hover: pl.DataFrame,
                           flags: dict):
    """County and region maps and smoothed maps of road users (from YOLO detections; `d`: one row per analysed
    segment with id, seconds, minutes, night and a count column per class)."""
    _detection_area_maps(d, df_mapping, flags, classes)
    _detection_field_maps(d, df_mapping, flags, classes)
