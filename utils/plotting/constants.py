"""
Plotting constants used across utils.plotting.

Keep this module free of Plotly/Pandas imports to avoid heavy import side effects.
"""

from __future__ import annotations

from typing import Final

# Country outlines of every plotly map (HTML, PNG/EPS and map_labels): plotly's world map on UN boundaries, which
# draws Crimea as part of Ukraine (plotly.js 2's default, https://cdn.plot.ly/world_110m.json, puts it in Russia).
# plotly.js 3 uses these by default.
TOPOJSON_URL: Final[str] = "https://cdn.plot.ly/un/"

# Continent colours used in every figure (Okabe-Ito palette, distinguishable with colour-blindness)
CONTINENT_COLORS: Final[dict] = {
    "Africa": "#E69F00",
    "Asia": "#D55E00",
    "Europe": "#0072B2",
    "North America": "#009E73",
    "Oceania": "#CC79A7",
    "South America": "#56B4E9",
}

# Colour bars of maps: horizontal, centred above the map
MAP_TOP_SHARE: Final[float] = 0.88  # maps use the lower 88% of the figure height; the colour bar sits above


def colorbar_top(title: str) -> dict:
    """Plotly colour bar layout: horizontal, centred above the map, title above it."""
    return dict(orientation="h", x=0.5, xanchor="center", y=MAP_TOP_SHARE + 0.02, yanchor="bottom", len=0.5,
                thickness=14, title=dict(text=title, side="top"))
