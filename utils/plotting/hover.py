"""
Hover popups shared by all figures: one complete, formatted popup per locality and per country.

Figures put the popup text on each point (`hovertext`, or `customdata` with plotly express) and show it with
TEMPLATE, so a locality or country looks the same in every map and scatter plot.
"""

import ast

import polars as pl

TEMPLATE = "%{hovertext}<extra></extra>"
PX_TEMPLATE = "%{customdata[0]}<extra></extra>"  # plotly express figures: custom_data=["hover"]

INDICATORS = [  # country column, label, format
    ("population_country", "Population", "{:,.0f}"),
    ("traffic_mortality", "Road deaths per 100k", "{:.1f}"),
    ("literacy_rate", "Literacy rate", "{:.1f}%"),
    ("gini", "Gini index", "{:.1f}"),
    ("med_age", "Median age", "{:.1f} years"),
    ("avg_height", "Average height", "{:.1f} cm"),
]


def _rule():
    return "──────────"  # in the text colour, which plotly picks to contrast with the point's colour


def _hours(h: float) -> str:
    return f"{h:,.1f} h" if h >= 1 else f"{h * 60:,.0f} min"


def _vehicles(shares: dict) -> str:
    top = [(v, s) for v, s in sorted(shares.items(), key=lambda kv: -kv[1])[:3] if s >= 0.5]
    return ", ".join(f"{v} {s:.0f}%" for v, s in top)


def _years(lo, hi) -> str:
    if lo is None:
        return "unknown"
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _footage_lines(r: dict) -> list:
    return [
        f"<b>Footage:</b> {_hours(r['hours'])} in {r['n_segments']:,} segments of {r['n_videos']:,} "
        f"{'video' if r['n_videos'] == 1 else 'videos'}",
        f"<b>Night:</b> {r['night']:.0f}%   <b>Channels:</b> {r['n_channels']:,}",
        f"<b>Vehicles:</b> {_vehicles(r['vehicle_shares'])}",
        f"<b>Uploaded:</b> {_years(r['first_year'], r['last_year'])}",
    ]


def _summary(seg: pl.DataFrame, key: str) -> pl.DataFrame:
    """Footage statistics per `key` (id or iso3) from the segment table."""
    vehicles = (seg.drop_nulls("vehicle").group_by(key, "vehicle").agg(pl.sum("seconds"))
                   .with_columns((pl.col("seconds") / pl.col("seconds").sum().over(key) * 100).alias("share"))
                   .group_by(key).agg(pl.struct("vehicle", "share").alias("vehicle_shares")))
    return (seg.group_by(key)
               .agg((pl.sum("seconds") / 3600).alias("hours"), pl.len().alias("n_segments"),
                    pl.col("video").n_unique().alias("n_videos"),
                    pl.col("channel").drop_nulls().n_unique().alias("n_channels"),
                    (pl.col("seconds").filter(pl.col("night")).sum() / pl.sum("seconds") * 100).alias("night"),
                    pl.col("year").filter(pl.col("year").is_between(2005, 2100)).min().alias("first_year"),
                    pl.col("year").filter(pl.col("year").is_between(2005, 2100)).max().alias("last_year"))
               .join(vehicles, on=key, how="left"))


def _shares(row: dict) -> dict:
    return {v["vehicle"]: v["share"] for v in row.get("vehicle_shares") or []}


def _aka(cell) -> list:
    try:
        names = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip("[] ") else []
    except (ValueError, SyntaxError):  # unquoted lists such as [İstanbul]
        names = [n.strip(" '\"") for n in str(cell).strip("[]").split(",")]
    return [str(n) for n in names if str(n).strip()]


def locality_hover(df_mapping: pl.DataFrame, seg: pl.DataFrame, flags: dict) -> pl.DataFrame:
    """One popup per mapping row: `id`, `hover`."""
    rows = df_mapping.join(_summary(seg, "id"), on="id", how="left")
    out = []
    for r in rows.iter_rows(named=True):
        flag = flags.get(r["iso3"], "🏳️")
        lines = [f"<b>{flag} {r['locality']}</b>"]
        aka = _aka(r.get("locality_aka"))
        if aka:
            lines.append(f"<i>also {', '.join(aka[:3])}</i>")
        place = ", ".join(p for p in (r.get("state"), r["country"]) if p)
        lines.append(f"{place} · {r['continent']}")
        lines.append(_rule())
        if r.get("hours"):
            r["vehicle_shares"] = _shares(r)
            lines += _footage_lines(r)
        else:
            lines.append("No footage")
        lines.append(_rule())
        if r.get("population_locality"):
            lines.append(f"<b>Population:</b> {r['population_locality']:,.0f}")
        if r.get("gmp"):
            lines.append(f"<b>Gross metropolitan product:</b> ${r['gmp']:,.0f} billion")
        if r.get("traffic_index") is not None:
            lines.append(f"<b>Traffic index (TomTom):</b> {r['traffic_index']:,.1f}")
        if r.get("lat") is not None:
            lines.append(f"<b>Location:</b> {r['lat']:.3f}, {r['lon']:.3f}")
        lines.append(f"<b>{r['country']}:</b> " + _country_indicators(r))
        out.append(dict(id=r["id"], hover="<br>".join(lines)))
    return pl.DataFrame(out, schema={"id": df_mapping.schema["id"], "hover": pl.Utf8})


def _country_indicators(r: dict) -> str:
    """The country's indicators on two lines (zeros mean no data)."""
    parts = [f"{label} {fmt.format(r[col])}" for col, label, fmt in INDICATORS if r.get(col)]
    if not parts:
        return "no indicators"
    return " · ".join(parts[:3]) + ("<br>" + " · ".join(parts[3:]) if parts[3:] else "")


def country_hover(df_mapping: pl.DataFrame, seg: pl.DataFrame, flags: dict) -> pl.DataFrame:
    """One popup per country: `iso3`, `hover`."""
    first = (pl.col(c).filter(pl.col(c) > 0).first() for c, _, _ in INDICATORS)
    countries = (df_mapping.group_by("iso3").agg(pl.first("country"), pl.col("continent").unique().sort(),
                                                 pl.len().alias("n_localities"), *first)
                           .join(_summary(seg, "iso3"), on="iso3", how="left"))
    by_channel = seg.drop_nulls("channel").group_by("iso3", "channel").agg(pl.sum("seconds"))
    channels = (by_channel.with_columns((pl.col("seconds") / pl.col("seconds").sum().over("iso3")).alias("s"))
                          .group_by("iso3").agg((1 / (pl.col("s") ** 2).sum()).alias("effective"),
                                                (pl.max("s") * 100).alias("top_channel")))
    countries = countries.join(channels, on="iso3", how="left")
    out = []
    for r in countries.iter_rows(named=True):
        lines = [f"<b>{flags.get(r['iso3'], '🏳️')} {r['country']}</b>", " · ".join(r["continent"]), _rule()]
        if r.get("hours"):
            r["vehicle_shares"] = _shares(r)
            lines += _footage_lines(r)
            lines.insert(4, f"<b>Localities:</b> {r['n_localities']:,}")
            if r.get("population_country"):
                lines.append(f"<b>Per million people:</b> {r['hours'] / r['population_country'] * 1e6:,.1f} h")
            if r.get("effective"):
                lines.append(f"<b>Largest channel:</b> {r['top_channel']:.0f}% of footage; "
                             f"effective number of channels {r['effective']:.1f}")
        else:
            lines.append("No footage")
        lines.append(_rule())
        for col, label, fmt in INDICATORS:
            if r.get(col):
                lines.append(f"<b>{label}:</b> {fmt.format(r[col])}")
        out.append(dict(iso3=r["iso3"], hover="<br>".join(lines)))
    return pl.DataFrame(out, schema={"iso3": pl.Utf8, "hover": pl.Utf8})


if __name__ == "__main__":
    # self-check on one locality with two videos (one at night)
    m = pl.DataFrame(dict(id=[1], locality=["A"], locality_aka=["['Aa']"], state=["S"], country=["B"], iso3=["BBB"],
                          continent=["Europe"], lat=[1.0], lon=[2.0], gmp=[None], population_locality=[1000],
                          population_country=[5e6], traffic_mortality=[3.5], literacy_rate=[0.0], gini=[30.0],
                          med_age=[40.0], avg_height=[170.0], traffic_index=[0.0]))
    s = pl.DataFrame(dict(id=[1, 1], iso3=["BBB"] * 2, video=["v1", "v2"], seconds=[3600, 1800], night=[False, True],
                          vehicle=["Car", "Bus"], year=[2020, 2024], channel=["c1", "c1"]))
    h = locality_hover(m, s, {"BBB": "🇧🇧"})["hover"][0]
    assert "🇧🇧 A" in h and "also Aa" in h and "1.5 h in 2 segments of 2 videos" in h, h
    assert "Night:</b> 33%" in h and "Car 67%, Bus 33%" in h and "2020–2024" in h and "Traffic index" in h, h
    assert "literacy" not in h, h  # zero literacy means no data
    c = country_hover(m, s, {"BBB": "🇧🇧"})["hover"][0]
    assert "Localities:</b> 1" in c and "Per million people:</b> 0.3 h" in c and "Largest channel:</b> 100%" in c, c
    print("ok")
