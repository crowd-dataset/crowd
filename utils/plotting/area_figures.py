"""
Maps by area and 3D views of where the footage (and, with YOLO detections, the road users) is.

- County and region maps: localities are assigned to US counties and European NUTS-2 regions by their coordinates
  (point in polygon), and each area is coloured by its footage, footage per million people or share of night
  footage (or detections per minute).
- Terrain views: a smoothed surface of footage, population or detections per minute over a tilted flat map, as in
  population-density "mountain" maps; and flat spike maps of a country, one spike per locality.

Outlines and populations are public downloads, cached in _output/geo_cache (refreshed every half year): US
counties (plotly's datasets) and their populations (US Census Bureau estimates), NUTS-2 regions (Eurostat GISCO)
and their populations (Eurostat), and cities of over 15,000 people (GeoNames) for population terrains. A figure
whose download fails is skipped.
"""

import io
import json
import math
import os
import time
import zipfile

import numpy as np
import plotly.graph_objects as go
import polars as pl
import requests
from plotly.subplots import make_subplots

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
    "nuts2_population.json": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_r_d2jan"
                             "?sex=T&age=TOTAL&lastTimePeriod=1&format=JSON&lang=EN",
    "cities15000.zip": "https://download.geonames.org/export/dump/cities15000.zip",
}
EUROPE_VIEW = dict(projection_type="natural earth", lonaxis_range=[-25, 45], lataxis_range=[34, 71.5])
# (lon min, lon max, lat min, lat max) of the terrain views
BOXES = {"Europe": (-11, 42, 34.5, 71), "North America": (-128, -60, 14, 58), "Asia": (60, 150, -10, 55)}
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
    path, pop_path = _fetch("nuts2.json"), _fetch("nuts2_population.json")
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        geo = json.load(f)
    pop = {}
    if pop_path:
        with open(pop_path, encoding="utf-8") as f:
            d = json.load(f)
        index = d["dimension"]["geo"]["category"]["index"]
        pop = {code: d["value"].get(str(i)) for code, i in index.items() if d["value"].get(str(i))}
    feats = {g["properties"]["NUTS_ID"]: dict(name=f"{g['properties']['NAME_LATN']} ({g['properties']['CNTR_CODE']})",
                                              rings=_rings(g["geometry"]), geometry=g["geometry"])
             for g in geo["features"]}
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
                           font=dict(size=14, color="#666666"), xanchor="left")
    dfig._save(dfig._style(fig, margin=dict(l=0, r=0, t=10, b=0)), name)


def _area_maps(seg: pl.DataFrame, df_mapping: pl.DataFrame, flags: dict):
    """Footage, footage per million people and night share per US county and per European NUTS-2 region."""
    for kind, prefix, sel in [("us", "map_us_counties", pl.col("iso3") == "USA"),
                              ("eu", "map_eu_regions", pl.col("continent") == "Europe")]:
        areas = _areas(kind)
        if areas is None:
            continue
        feats, pop = areas
        pts = df_mapping.filter(sel).drop_nulls(["lat", "lon"])
        area_of = dict(zip(pts["id"], _assign(pts["lon"].to_numpy(), pts["lat"].to_numpy(), feats)))
        table = _area_table(seg, {k: v for k, v in area_of.items() if v}, df_mapping, flags)
        table = table.with_columns(pl.col("area").replace_strict(pop, default=None, return_dtype=pl.Float64)
                                   .alias("population"))
        table = table.with_columns((pl.col("hours") / pl.col("population") * 1e6).alias("per_million"))
        us = kind == "us"
        what = "county" if us else "region"
        uk = "" if us else ("Regions: Eurostat NUTS 2 (EU, EFTA and candidate countries; not the United Kingdom, "
                            "Ukraine or Russia)")
        _area_map(table, feats, pop, "hours", "Footage (hours)", "YlOrRd", f"{prefix}_footage", True, ",.1f", us,
                  note=uk)
        _area_map(table, feats, pop, "per_million", "Hours per million people", "YlGnBu",
                  f"{prefix}_footage_per_capita", True, ",.1f", us,
                  note=f"Population: {'US Census Bureau estimates, 2024' if us else 'Eurostat, latest year'}")
        _area_map(table.filter(pl.col("hours") >= MIN_HOURS_AREA), feats, pop, "night", "Night footage (%)", "Blues",
                  f"{prefix}_night_share", False, ".0f", us,
                  note=f"Each {what} with at least {MIN_HOURS_AREA} hour of footage")


def _detection_area_maps(d: pl.DataFrame, df_mapping: pl.DataFrame, flags: dict, classes: list):
    """Pedestrians per minute per US county and European region, and bicycles per minute per European region
    (`d`: one row per analysed segment with `id`, `minutes` and a count column per class)."""
    person = classes[0]
    for kind, prefix, sel, cols in [("us", "map_us_counties", pl.col("iso3") == "USA", [person]),
                                    ("eu", "map_eu_regions", pl.col("continent") == "Europe",
                                     [person] + (["Bicycles"] if "Bicycles" in classes else []))]:
        areas = _areas(kind)
        if areas is None:
            continue
        feats, pop = areas
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


# --- terrain and spikes -----------------------------------------------------------------------------------------

def _blur(grid: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    """Gaussian blur (sigmas in cells), with numpy only."""
    for axis, sigma in ((0, sigma_y), (1, sigma_x)):
        r = max(1, int(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        grid = np.apply_along_axis(lambda v: np.convolve(v, k / k.sum(), mode="same"), axis, grid)
    return grid


def _surface(lon, lat, w, box, cell: float, sigma_km: float = 40, rate=None) -> tuple:
    """(lon centres, lat centres, grid) of the smoothed sum of weights `w` per cell, or with `rate` the smoothed
    weighted mean of `rate` (weights `w`), 0 where there is too little weight."""
    nx, ny = int((box[1] - box[0]) / cell), int((box[3] - box[2]) / cell)
    rng = [[box[2], box[3]], [box[0], box[1]]]
    sy = sigma_km / 111 / cell
    sx = sy / max(math.cos(math.radians((box[2] + box[3]) / 2)), 0.2)
    weight, _, _ = np.histogram2d(lat, lon, bins=[ny, nx], range=rng, weights=w)
    weight = _blur(weight, sy, sx)
    if rate is not None:
        num, _, _ = np.histogram2d(lat, lon, bins=[ny, nx], range=rng, weights=w * rate)
        num = _blur(num, sy, sx)
        mean = num / np.maximum(weight, 1e-12)
        # fade out where there is little footage, so places without data are flat rather than cut off by cliffs
        k = np.percentile(weight[weight > 0], 50) if (weight > 0).any() else 1
        grid = mean * weight / (weight + k)
    else:
        grid = weight
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


def _terrain_traces(lons, lats, grid, box, colorscale, labels: pl.DataFrame | None, hover_pts: pl.DataFrame | None):
    """Surface, borders, peak labels and hover points of one terrain."""
    z = grid / grid.max() if grid.max() > 0 else grid
    traces = [go.Surface(x=lons, y=lats, z=np.round(z, 4), surfacecolor=z, colorscale=colorscale, cmin=0, cmax=1,
                         showscale=False, hoverinfo="skip", opacity=1,
                         lighting=dict(ambient=0.75, diffuse=0.6, specular=0.05, roughness=0.9))]
    bx, by = _borders(box)
    traces.append(go.Scatter3d(x=bx, y=by, z=[0.003 if v is not None else None for v in bx], mode="lines",
                               line=dict(color="#555555", width=1.5), hoverinfo="skip", showlegend=False))

    def height(lon, lat):  # the top of the peak around a point (a few cells either way)
        i = int(np.clip(np.searchsorted(lats, lat), 0, len(lats) - 1))
        j = int(np.clip(np.searchsorted(lons, lon), 0, len(lons) - 1))
        return z[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4].max()
    if labels is not None and labels.height:
        traces.append(go.Scatter3d(x=labels["lon"], y=labels["lat"],
                                   z=[height(x, y) + 0.04 for x, y in zip(labels["lon"], labels["lat"])],
                                   mode="text", text=labels["name"], textfont=dict(size=12, color="#222222"),
                                   hoverinfo="skip", showlegend=False))
    if hover_pts is not None and hover_pts.height:
        traces.append(go.Scatter3d(x=hover_pts["lon"], y=hover_pts["lat"], z=[0.006] * hover_pts.height,
                                   mode="markers", marker=dict(size=1.5, color="#555555", opacity=0.3),
                                   text=hover_pts["hover"], hovertemplate=hover.GL_TEMPLATE, showlegend=False))
    return traces


def _in_box(df: pl.DataFrame, box) -> pl.DataFrame:
    return df.filter(pl.col("lon").is_between(box[0], box[1]), pl.col("lat").is_between(box[2], box[3]))


def _terrains(pts: pl.DataFrame, value: str, name: str, colorscale, note: str, rate: bool = False):
    """Terrain views of `value` per locality for each region in BOXES (pts: lon, lat, name, hover, `value`, and
    minutes with `rate`, the weights of a smoothed mean)."""
    from utils.plotting import dataset_figures as dfig
    for region, box in BOXES.items():
        p = _in_box(pts, box).filter(pl.col(value) > 0)
        if p.height < 3:
            continue
        cell = 0.25 if region == "Europe" else 0.35
        if rate:
            lons, lats, grid = _surface(p["lon"].to_numpy(), p["lat"].to_numpy(), p["minutes"].to_numpy(), box,
                                        cell, rate=p[value].to_numpy())
        else:
            lons, lats, grid = _surface(p["lon"].to_numpy(), p["lat"].to_numpy(), p[value].to_numpy(), box, cell)
        # labelled: the localities with the most footage (with `rate`, the most analysed footage)
        labels = p.sort("minutes" if rate else value, descending=True).head(10)
        fig = go.Figure(_terrain_traces(lons, lats, grid, box, colorscale, labels, p))
        fig.update_layout(scene=_scene(box, 1.0))
        fig.add_annotation(text=f"{region}: {note}", x=0.01, y=0.01, xref="paper", yref="paper", showarrow=False,
                           xanchor="left", font=dict(size=15, color="#555555"))
        dfig._save(dfig._style(fig, width=1600, height=1000, margin=dict(l=0, r=0, t=0, b=0)),
                   f"{name}_{region.lower().replace(' ', '_')}", save_eps=False)


def _cities(iso2: str) -> pl.DataFrame | None:
    """Cities of over 15,000 people in a country (GeoNames): lon, lat, name, population."""
    path = _fetch("cities15000.zip")
    if not path:
        return None
    with zipfile.ZipFile(path) as z:
        raw = z.read("cities15000.txt")
    df = pl.read_csv(io.BytesIO(raw), separator="\t", has_header=False, quote_char=None, infer_schema_length=0)
    df = df.select(pl.nth(1).alias("name"), pl.nth(4).cast(pl.Float64).alias("lat"),
                   pl.nth(5).cast(pl.Float64).alias("lon"), pl.nth(8).alias("iso2"),
                   pl.nth(14).cast(pl.Float64).alias("population"))
    return df.filter(pl.col("iso2") == iso2)


def _population_vs_footage(locs: pl.DataFrame, iso3: str, iso2: str, country: str, name: str):
    """A country's population terrain (GeoNames cities) next to its footage terrain."""
    from utils.plotting import dataset_figures as dfig
    shape = map_labels.country_shapes().get(iso3)
    cities = _cities(iso2)
    if shape is None or cities is None or not cities.height:
        return
    xs = [x for r in shape["rings"] for x, _ in r]
    ys = [y for r in shape["rings"] for _, y in r]
    box = (min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5)
    cell = max(box[1] - box[0], box[3] - box[2]) / 160
    foot = locs.filter(pl.col("iso3") == iso3, pl.col("hours") > 0)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.0,
                        subplot_titles=["Population (cities of over 15,000 people)",
                                        f"Footage in the dataset ({foot['hours'].sum():,.0f} hours)"])
    big = cities.sort("population", descending=True).head(8)
    for col, (pts, weights, labels, hov) in enumerate([
            (cities, "population", big, None),
            (foot, "hours", foot.sort("hours", descending=True).head(8), foot)], 1):
        lons, lats, grid = _surface(pts["lon"].to_numpy(), pts["lat"].to_numpy(), pts[weights].to_numpy(), box,
                                    cell, sigma_km=25)
        for t in _terrain_traces(lons, lats, grid, box, "YlOrRd", labels, hov):
            fig.add_trace(t, row=1, col=col)
    fig.update_layout(scene=_scene(box, 1.0, distance=1.45), scene2=_scene(box, 1.0, distance=1.45))
    fig.update_annotations(font_size=18)
    fig.add_annotation(text=f"{country}: each terrain scaled to its own highest peak", x=0.5, y=0.955, xref="paper",
                       yref="paper", showarrow=False, font=dict(size=14, color="#555555"))
    dfig._save(dfig._style(fig, width=1600, height=850, margin=dict(l=0, r=0, t=40, b=0)), name, save_eps=False)


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


# --- entry points ---------------------------------------------------------------------------------------------------

def _localities(seg: pl.DataFrame, df_mapping: pl.DataFrame, loc_hover: pl.DataFrame, flags: dict) -> pl.DataFrame:
    return (seg.group_by("id").agg((pl.sum("seconds") / 3600).alias("hours"))
               .join(df_mapping.select("id", "locality", "iso3", "lat", "lon"), on="id").join(loc_hover, on="id")
               .drop_nulls(["lat", "lon"])
               .with_columns(pl.concat_str([pl.col("iso3").replace_strict(flags, default="🏳️"), pl.col("locality")],
                                           separator=" ").alias("name")))


def dataset_area_figures(df_mapping: pl.DataFrame, seg: pl.DataFrame, loc_hover: pl.DataFrame, flags: dict):
    """County and region maps, footage terrains, population against footage and spike maps (from the mapping)."""
    _area_maps(seg, df_mapping, flags)
    locs = _localities(seg, df_mapping, loc_hover, flags)
    _terrains(locs, "hours", "terrain_footage", "YlOrRd", "height and colour show the footage (hours), smoothed over "
              "~40 km")
    for iso3, iso2, country in [("IND", "IN", "India"), ("NGA", "NG", "Nigeria")]:
        _population_vs_footage(locs, iso3, iso2, country, f"terrain_population_footage_{country.lower()}")
    for iso3, country in [("USA", "United States"), ("RUS", "Russia"), ("JPN", "Japan")]:
        _spike_map(locs, iso3, f"spikes_footage_{iso3.lower()}", country)


def detection_area_figures(df_mapping: pl.DataFrame, d: pl.DataFrame, classes: list, loc_hover: pl.DataFrame,
                           flags: dict):
    """Pedestrian and night-life terrains and county and region maps of road users (from YOLO detections; `d`: one
    row per analysed segment with id, seconds, minutes, night and a count column per class)."""
    person = classes[0]
    pts = (d.group_by("id").agg((pl.col(person).sum() / pl.sum("minutes")).alias("rate"), pl.sum("minutes"))
            .join(df_mapping.select("id", "locality", "iso3", "lat", "lon"), on="id").join(loc_hover, on="id")
            .drop_nulls(["lat", "lon"])
            .with_columns(pl.concat_str([pl.col("iso3").replace_strict(flags, default="🏳️"), pl.col("locality")],
                                        separator=" ").alias("name")))
    _terrains(pts, "rate", "terrain_pedestrians", "YlOrRd", "height and colour show pedestrians per minute, "
              "smoothed over ~40 km (weighted by analysed footage)", rate=True)
    night = (d.filter(pl.col("night")).group_by("id")
              .agg((pl.col(person).sum() / pl.sum("minutes")).alias("rate"), pl.sum("minutes"))
              .join(pts.select("id", "locality", "iso3", "lat", "lon", "hover", "name"), on="id"))
    _terrains(night, "rate", "terrain_pedestrians_night", "Purples", "height and colour show pedestrians per minute "
              "at night, smoothed over ~40 km", rate=True)
    _detection_area_maps(d, df_mapping, flags, classes)
