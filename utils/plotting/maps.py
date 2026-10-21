import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import common
from utils.plotting.io import IO
from utils.plotting import hover, map_labels
from utils.plotting.constants import CONTINENT_COLORS, MAP_TOP_SHARE, colorbar_top
import warnings
from custom_logger import CustomLogger

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logger = CustomLogger(__name__)  # use custom logger

io_class = IO()


class Maps:
    _CARTO_TILE = ("https://basemaps.cartocdn.com/rastertiles/light_all"  # io.py uses light_nolabels for images
                   "/{{z}}/{{x}}/{{y}}.png?key={key}")

    def __init__(self) -> None:
        self._carto_key = common.get_secrets("carto_api_key")

    def _apply_carto_key(self, fig) -> None:
        """Replace built-in CARTO style with a keyed raster layer to remove the watermark."""
        if not self._carto_key:
            return
        tile_url = self._CARTO_TILE.format(key=self._carto_key)
        layer = dict(below="traces", sourcetype="raster", source=[tile_url])
        # scatter_map uses 'map' layout; density_mapbox uses 'mapbox' layout
        if fig.data and "mapbox" in fig.data[0].type:
            fig.update_layout(mapbox=dict(style="white-bg", layers=[layer]))
        else:
            fig.update_layout(map=dict(style="white-bg", layers=[layer]))

    # world_map_ss: localities with more footage than this (s) are drawn as red squares above the black dots
    HIGHLIGHT_SECONDS = 100_000

    # world_map_ss with show_images: screenshot tiles (paper coordinates; approx_lon/lat is the tile end of the
    # connector line to the locality) with their caption and video ID
    SS_IMAGES = [
        {"locality": "Tokyo", "country": "Japan", "file": "tokyo.png", "x": 0.92, "y": 0.62,
         "approx_lon": 165.2, "approx_lat": 17.2, "label": "Tokyo, Japan", "x_label": 0.97, "y_label": 0.693,
         "video": "-YNLQlmPnqA", "x_video": 0.91719, "y_video": 0.560},
        {"locality": "Cape Town", "country": "South Africa", "file": "cape_town.jpg", "x": 0.64, "y": 0.19,
         "approx_lon": 44.0, "approx_lat": -48.0, "label": "Cape Town, South Africa", "x_label": 0.6385,
         "y_label": 0.24, "video": "0xP7JgDiBb8", "x_video": 0.69, "y_video": 0.12},
        {"locality": "San Francisco", "country": "United States", "file": "san_francisco.png", "x": 0.13, "y": 0.56,
         "approx_lon": -123.5, "approx_lat": 8.0, "label": "San Francisco, CA, USA", "x_label": 0.08,
         "y_label": 0.618, "video": "HZrm3s4UsgU", "x_video": 0.131, "y_video": 0.503},
        {"locality": "London", "country": "United Kingdom", "file": "london.png", "x": 0.395, "y": 0.66,
         "approx_lon": -36.0, "approx_lat": 34.0, "label": "London, UK", "x_label": 0.36845, "y_label": 0.733,
         "video": "Bs3MZ4wWMQs", "x_video": 0.418, "y_video": 0.60},
        {"locality": "Sao Paulo", "country": "Brazil", "file": "sao_paulo.jpg", "x": 0.43, "y": 0.27,
         "approx_lon": -41.0, "approx_lat": -36.0, "label": "Sao Paulo, Brazil", "x_label": 0.414, "y_label": 0.32,
         "video": "Ic2ERD7kt4o", "x_video": 0.458, "y_video": 0.20},
        {"locality": "Perth", "country": "Australia", "file": "perth.png", "x": 0.72, "y": 0.33,
         "approx_lon": 72.0, "approx_lat": -25.0, "label": "Perth, WA, Australia", "x_label": 0.74959,
         "y_label": 0.392, "video": "xTDUhnnj3q4", "x_video": 0.77, "y_video": 0.26},
    ]
    # the larger YOLO example tile in the bottom-left corner (no connector line)
    SS_YOLO = {"file": "toronto_yolo.jpg", "x": 0.12, "y": 0.158,
               "label": "Example of YOLO output (Toronto, ON, Canada)", "x_label": 0.00, "y_label": 0.278,
               "video": "3ai7SUaPoHM", "x_video": 0.1925, "y_video": 0.019}

    def world_map_ss(self, df, *, title=None, projection="natural earth", df_mapping=None, show_images=False,
                     hover_data=None, save_file=False, save_final=False, name_file=None, show_colorbar=True,
                     colorbar_title="Footage (hours)", color_scale="Turbo", marker_size=3, filter_zero=True):
        """World map of localities as black dots (alpha 0.35), with red squares on top for those with more than
        HIGHLIGHT_SECONDS of footage, and with show_images the SS_IMAGES/SS_YOLO screenshots; saved as name_file.

        df: one row per locality with locality, country, footage_hours or total_time (s), and lat/lon (or pass
        df_mapping to look them up); rows with the same id (if present) are summed. hover_data: extra columns
        shown on hover (total_time is left out: the footage is already shown in hours). save_file, show_colorbar,
        colorbar_title and color_scale are unused (the dots have a fixed colour; the figure is always saved).
        """
        palette = {
            "ocean": "#D6ECFA",
            "land": "#F3E2CC",
            "country": "#D8D2C7",
            "coast": "#DDD8CF",
            "connector": "#3B3B3B",
            "label_bg": "rgba(255,255,255,0.88)",
            "label_border": "#6B7280",
            "label_text": "#1F2937",
            "paper_bg": "white",
        }

        df_plot = df.copy()

        if "locality" not in df_plot.columns or "country" not in df_plot.columns:
            raise ValueError("`df` must contain 'locality' and 'country' columns.")

        # Footage in hours and in seconds, whichever is missing
        if "footage_hours" not in df_plot.columns:
            if "total_time" not in df_plot.columns:
                raise ValueError("`df` must contain either 'footage_hours' or 'total_time'.")
            df_plot["footage_hours"] = pd.to_numeric(df_plot["total_time"], errors="coerce") / 3600.0
        elif "total_time" not in df_plot.columns:
            df_plot["total_time"] = pd.to_numeric(df_plot["footage_hours"], errors="coerce") * 3600.0

        # Add coordinates from df_mapping if lat lon are missing
        if not {"lat", "lon"}.issubset(df_plot.columns):
            if df_mapping is None:
                raise ValueError("`df` must contain 'lat' and 'lon', or you must pass `df_mapping`.")
            # the two tables may spell Türkiye differently
            df_plot["country"] = df_plot["country"].replace({"Türkiye": "Turkey"})
            df_mapping_local = df_mapping.copy()
            df_mapping_local["country"] = df_mapping_local["country"].replace({"Türkiye": "Turkey"})

            df_plot = df_plot.merge(
                df_mapping_local[["locality", "country", "lat", "lon"]].drop_duplicates(),
                on=["locality", "country"],
                how="left",
            )

        # Keep only rows with valid coordinates and footage
        df_plot["lat"] = pd.to_numeric(df_plot["lat"], errors="coerce")
        df_plot["lon"] = pd.to_numeric(df_plot["lon"], errors="coerce")
        df_plot["footage_hours"] = pd.to_numeric(df_plot["footage_hours"], errors="coerce")
        has_footage = df_plot["footage_hours"] > 0 if filter_zero else df_plot["footage_hours"] >= 0
        df_plot = df_plot[has_footage & df_plot["lat"].notna() & df_plot["lon"].notna()].copy()

        if df_plot.empty:
            raise ValueError("No valid locality rows remain after filtering.")

        # One row per locality: by id, as different cities can share a name and country (e.g. Portland, USA)
        keys = ["id"] if "id" in df_plot.columns else ["locality", "country"]
        agg_cols = {c: "first" for c in ["locality", "country"] if c not in keys}
        agg_cols.update({"lat": "first", "lon": "first", "footage_hours": "sum", "total_time": "sum"})

        extra_hover_cols = []
        for col in (hover_data or []):
            if col in df_plot.columns and col not in {"locality", "country", "lat", "lon", "footage_hours",
                                                      "total_time"}:
                agg_cols[col] = "sum" if pd.api.types.is_numeric_dtype(df_plot[col]) else "first"
                extra_hover_cols.append(col)

        df_plot = df_plot.groupby(keys, as_index=False).agg(agg_cols)

        fig = go.Figure()

        hover_lines = [
            "<b>%{text}</b>",
            "Country: %{customdata[0]}",
            "Footage: %{customdata[1]:.2f} hours",
        ]
        for i, col in enumerate(extra_hover_cols, start=2):
            hover_lines.append(f"{col}: " + "%{customdata[" + str(i) + "]}")
        hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

        # Two traces so the red markers always render on top
        df_normal = df_plot[df_plot["total_time"] <= self.HIGHLIGHT_SECONDS]
        df_red = df_plot[df_plot["total_time"] > self.HIGHLIGHT_SECONDS]
        logger.info(f"Cities with more than {self.HIGHLIGHT_SECONDS:,} s of footage: {len(df_red)}.")

        for part, marker, name in [
                (df_normal, dict(size=marker_size, color="rgba(0,0,0,0.35)", symbol="circle", line=dict(width=0)),
                 "cities"),
                (df_red, dict(size=marker_size + 2, color="rgba(255,0,0,1.0)", symbol="square"),
                 "cities_highlight")]:
            if part.empty:
                continue
            fig.add_trace(
                go.Scattergeo(
                    lon=part["lon"],
                    lat=part["lat"],
                    text=part["locality"],
                    customdata=np.column_stack([part[c].to_numpy()
                                                for c in ["country", "footage_hours"] + extra_hover_cols]),
                    mode="markers",
                    hovertemplate=hovertemplate,
                    marker=dict(**marker, showscale=False),
                    showlegend=False,
                    name=name,
                )
            )

        if title:
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor="center",
                )
            )

        fig.update_layout(
            font=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size"),
                color=palette["label_text"],
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor=palette["paper_bg"],
            plot_bgcolor=palette["paper_bg"],
        )

        fig.update_geos(
            projection_type=projection,
            showland=True,
            landcolor=palette["land"],
            showcountries=True,
            countrycolor=palette["country"],
            countrywidth=0.7,
            showocean=True,
            oceancolor=palette["ocean"],
            showlakes=True,
            lakecolor=palette["ocean"],
            showcoastlines=True,
            coastlinecolor=palette["coast"],
            coastlinewidth=0.6,
            showframe=False,
            bgcolor=palette["paper_bg"],
        )

        if show_images:
            path_screenshots = os.path.join(common.root_dir, "screenshots")

            def _add_box_annotation(text, x, y, font_size=12):
                fig.add_annotation(
                    text=text,
                    x=x,
                    y=y,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=font_size, color=palette["label_text"]),
                    align="center",
                    bgcolor=palette["label_bg"],
                    bordercolor=palette["label_border"],
                    borderwidth=1,
                    borderpad=2,
                )

            def _add_image(item, size):
                """Add a screenshot with its caption and video ID (nothing if the file is missing)."""
                try:
                    img = Image.open(os.path.join(path_screenshots, item["file"]))
                except FileNotFoundError:
                    return
                fig.add_layout_image(dict(source=img, xref="paper", yref="paper", x=item["x"], y=item["y"],
                                          sizex=size, sizey=size, xanchor="center", yanchor="middle",
                                          layer="above"))
                _add_box_annotation(text=item["label"], x=item["x_label"], y=item["y_label"], font_size=12)
                _add_box_annotation(text=item["video"], x=item["x_video"], y=item["y_video"], font_size=10)

            for item in self.SS_IMAGES:
                _add_image(item, 0.1)

            # Connector lines from screenshot area to actual locality
            for item in self.SS_IMAGES:
                row = df_plot[(df_plot["locality"].str.lower() == item["locality"].lower())
                              & (df_plot["country"].str.lower() == item["country"].lower())]
                if row.empty:
                    continue

                fig.add_trace(
                    go.Scattergeo(
                        lon=[item["approx_lon"], row["lon"].iloc[0]],
                        lat=[item["approx_lat"], row["lat"].iloc[0]],
                        mode="lines",
                        line=dict(width=1.8, color=palette["connector"]),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            _add_image(self.SS_YOLO, 0.24)

        if name_file is None:
            name_file = "world_map_locality_footage"

        io_class.save_plotly_figure(fig, name_file, save_final=save_final)

    def mapbox_map(self, df, value_col=None, value_title=None, value_factor=1, hover_data=None, hover_name=None,
                   marker_size=5, file_name="mapbox_map", save_final=True):
        """Generates a world map of cities using Mapbox, with optional bubbles sized by a value.

        This method can create either:
            1. A simple scatter map showing locality locations colored by continent.
            2. A bubble map: each locality a circle whose area and (log) colour show `value_col`.

        Args:
            df (pandas.DataFrame): DataFrame containing mapping information.
                Required columns: "lat", "lon", "locality", "continent".
            value_col (str, optional): Column with the value shown by the bubbles. Defaults to None (scatter map).
            value_title (str, optional): Name of the value in the colour bar and hover. Defaults to `value_col`.
            value_factor (float, optional): Factor applied to the value, e.g. 1 / 3600 for seconds to hours.
            hover_data (list, optional): List of additional DataFrame columns to display when hovering.
                Defaults to None.
            hover_name (list, optional): title on top of hover popup.
            marker_size (int, optional): size of markers.
            file_name (str, optional): Name of the saved file (without extension). Defaults to "mapbox_map".
            save_final (bool, optional): If True, saves the figure. Defaults to True.

        Returns:
            None: The Plotly figure is created, displayed, and optionally saved.
        """
        # Draw scatter map if no value column is provided
        if not value_col:
            fig = px.scatter_map(
                df,
                lat="lat",
                lon="lon",
                hover_data=hover_data,
                hover_name="hover" if "hover" in df.columns else hover_name,
                color=df["continent"],
                color_discrete_map=CONTINENT_COLORS,
                zoom=1.3  # pyright: ignore[reportArgumentType]
            )

            # Apply marker size
            fig.update_traces(marker=dict(size=marker_size))

        # Draw bubble map if a value column is provided. A density heatmap on a linear scale shows only the few
        # largest values, so the value is shown by circle area and on a log colour scale instead.
        else:
            value_title = value_title or value_col
            df = df.assign(**{value_title: df[value_col] * value_factor})
            df = df[df[value_title] > 0].sort_values(value_title, ascending=False)  # small circles on top
            if df.empty:
                logger.warning(f"{file_name}: no positive {value_col}, map not made.")
                return
            df["log_value"] = np.log10(df[value_title])
            fig = px.scatter_map(
                df,
                lat="lat",
                lon="lon",
                size=value_title,
                size_max=30,
                color="log_value",
                # YlOrRd without its three palest steps: pale yellow vanishes on the white land when zoomed in
                color_continuous_scale=px.colors.sequential.YlOrRd[3:],
                hover_name="hover" if "hover" in df.columns else hover_name,
                hover_data={**{c: True for c in hover_data or [] if c in df.columns}, value_title: ":,.1f",
                            "log_value": False},
                zoom=1.3  # pyright: ignore[reportArgumentType]
            )
            fig.update_traces(marker=dict(sizemin=2, opacity=0.85))
            fig.update_layout(coloraxis_colorbar=dict(title=value_title,
                                                      **self._log_ticks(df["log_value"].min(), df["log_value"].max())))

        if "hover" in df.columns:  # the full locality popup shared by all figures
            fig.update_traces(hovertemplate=hover.TEMPLATE)

        # Update map layout to improve appearance
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size')
            ),
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.6)',  # Semi-transparent white background
                bordercolor='rgba(0,0,0,0.1)',   # Light border
                borderwidth=1
            ),
            legend_title_text=""  # Remove legend title
        )

        # Save the figure if requested (static images drop the basemap's place labels, see IO._static_basemap)
        fig.update_layout(map_style="carto-positron")
        self._apply_carto_key(fig)
        # the world is 512 * 2 ** 1.3 = 1,260 px wide at zoom 1.3: a wider image shows Japan, Australia and
        # Alaska twice
        io_class.save_plotly_figure(fig, file_name, width=1260, save_final=save_final)

    # State -> ((lon, lat) inside the state, (lon, lat) of its label off the coast),
    # for states whose label does not fit inside the state
    US_STATE_CALLOUTS = {
        "VT": ((-72.7, 44.0), (-75.0, 46.6)),
        "NH": ((-71.6, 43.6), (-69.2, 43.2)),
        "MA": ((-71.9, 42.3), (-68.6, 42.0)),
        "RI": ((-71.5, 41.7), (-69.2, 40.9)),
        "CT": ((-72.7, 41.6), (-69.8, 39.9)),
        "NJ": ((-74.5, 40.1), (-70.6, 38.9)),
        "DE": ((-75.5, 39.0), (-71.4, 37.9)),
        "MD": ((-76.6, 39.2), (-72.2, 36.9)),
        "DC": ((-77.0, 38.9), (-73.0, 35.9)),
        "FL": ((-81.4, 27.6), (-78.6, 26.4)),
        "HI": ((-155.5, 19.6), (-159.8, 19.2)),
    }

    # Continent -> map extent (lon/lat ranges, optional projection centre for views crossing 180°)
    CONTINENT_VIEWS = {
        "Africa": dict(lon=(-30, 65), lat=(-36, 38)),
        "Asia": dict(lon=(10, 150), lat=(-11, 75)),
        "Europe": dict(lon=(-25, 45), lat=(34, 71)),
        "North America": dict(lon=(-170, -20), lat=(3, 80)),
        "Oceania": dict(lon=(110, 235), lat=(-48, 20), rotation=170),
        "South America": dict(lon=(-90, -30), lat=(-56, 16)),
    }
    MAP_SIZE = (1600, 900)  # size of saved maps (px); labels are laid out for it

    @staticmethod
    def _log_ticks(lo, hi):
        """Colour bar ticks at the powers of ten from log10 value lo to hi, labelled with the values."""
        ticks = [10 ** p for p in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)]
        return dict(tickvals=np.log10(ticks).tolist(), ticktext=[f"{t:,}" for t in ticks])

    @staticmethod
    def _fmt_value(v):
        """Map label of a value: whole numbers from 10, one decimal below."""
        return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"

    def footage_map(self, df, *, location_col, locationmode, scope, label_col=None, value_col="footage_h",
                    callouts=None, view=None, anchor_labels=None, colorbar_title="Footage (hours)",
                    color_scale="YlOrRd", value_range=None, owner_points=None, file_name="map_footage",
                    save_final=True, log=True, fmt=None):
        """
        Choropleth of regions (US states or countries) on a log colour scale (or a linear one), labelled with each
        region's name and value.

        With `view` (country maps), labels are placed automatically without overlaps: inside the country if it fits,
        otherwise next to a dot coloured like the country, or with a leader line if there is no room next to the dot.
        Without `view` (US states), labels are centred on each region and `callouts` move the ones that do not fit.

        Args:
            df (pd.DataFrame): One row per region; for country maps also `lon`/`lat` (a point inside the region,
                used for countries plotly does not draw and for `anchor_labels`).
            location_col (str): Column with region codes matching `locationmode` (e.g., 'CA', 'DEU').
            locationmode (str): Plotly location mode, e.g. 'USA-states' or 'ISO-3'.
            scope (str): Plotly geo scope ('usa', 'world', ...).
            label_col (str|None): Column with the label text; defaults to `location_col`.
            value_col (str): Column with positive values to colour by.
            callouts (dict|None): Without `view`: region code -> ((lon, lat) inside the region, (lon, lat) of its
                label) for regions too small to hold their label.
            view (dict|None): Map extent as dict(lon=(min, max), lat=(min, max), rotation=lon of projection centre).
            anchor_labels (list|None): Codes labelled at their `lon`/`lat` instead of the country's centre, e.g.
                countries split across continents whose part shown is far from the country's centre.
            colorbar_title (str): Title of the colour bar and of the value in the hover text.
            color_scale (str|list): Plotly colour scale.
            owner_points (dict|None): ISO3 -> [(lon, lat)] of all localities, used to colour territories plotly draws
                only as part of another country (e.g., French Guiana), see map_labels.territory_ring.
            value_range (tuple|None): (min, max) of the colour scale; pass the same range to maps that are compared
                with each other (e.g., all continents) so a colour means the same value on each. Default: this map's.
            file_name (str): Name of the saved file (without extension).
            save_final (bool): Passed to `save_plotly_figure`.
            log (bool): Log colour scale (only positive values shown); False: linear (shares, rates).
            fmt (callable|None): Value -> label text; default `_fmt_value`.
        """
        label_col = label_col or location_col
        df = df[df[value_col] > 0 if log else df[value_col].notna()].sort_values(value_col, ascending=False).copy()
        if df.empty:
            logger.warning(f"{file_name}: no {'positive ' if log else ''}{value_col}, map not made.")
            return
        # Log colour scale: a few regions hold most of the footage, which would wash out the rest. The column keeps
        # its name on a linear scale.
        df["log_value"] = np.log10(df[value_col]) if log else df[value_col]
        lo, hi = ((np.log10(value_range) if log else value_range) if value_range
                  else (df["log_value"].min(), df["log_value"].max()))
        df["value_text"] = [(fmt or self._fmt_value)(v) for v in df[value_col]]
        rows = df.set_index(location_col, drop=False)

        fig = px.choropleth(df, locations=location_col, locationmode=locationmode, scope=scope, color="log_value",
                            color_continuous_scale=color_scale, range_color=(lo, hi),
                            hover_name="hover" if "hover" in df.columns else label_col,
                            hover_data={value_col: ":,.1f", "log_value": False, location_col: False},
                            labels={value_col: colorbar_title})
        if "hover" in df.columns:  # the full country popup shared by all figures
            fig.update_traces(hovertemplate=hover.TEMPLATE)
        # black text with a white halo stays readable on any fill and on the sea
        halo = "1px 1px 1px white, -1px -1px 1px white, 1px -1px 1px white, -1px 1px 1px white"
        font = dict(size=11, color="black", shadow=halo)

        def add_text(lon, lat, text, position="middle center"):
            fig.add_trace(go.Scattergeo(lon=lon, lat=lat, text=text, mode="text", textposition=position,
                                        textfont=font, hoverinfo="skip"))

        def add_lines(segments):
            fig.add_trace(go.Scattergeo(lon=[v for a, b in segments for v in (a[0], b[0], None)],
                                        lat=[v for a, b in segments for v in (a[1], b[1], None)],
                                        mode="lines", line=dict(width=1, color="grey"), hoverinfo="skip"))

        def add_dots(codes, points, text=None, positions="middle right"):
            fig.add_trace(go.Scattergeo(
                lon=[p[0] for p in points], lat=[p[1] for p in points], mode="markers+text",
                text=text or [""] * len(codes), textposition=positions, textfont=font,
                marker=dict(size=6, color=rows.loc[codes, "log_value"], coloraxis="coloraxis",
                            line=dict(width=0.5, color="grey")),
                hovertext=[rows.at[c, "hover"] if "hover" in rows.columns
                           else f"{rows.at[c, label_col]} {rows.at[c, 'value_text']}" for c in codes],
                hoverinfo="text",
            ))

        if view:
            width, height = self.MAP_SIZE
            proj = map_labels.Projection(view, width, height * MAP_TOP_SHARE)
            # territories plotly draws only as part of another country (e.g., French Guiana) or without an ISO code
            # (e.g., Kosovo): colour their own outline
            shapes = map_labels.country_shapes()
            territories = {c: map_labels.territory_ring((r["lon"], r["lat"]), shapes, owner_points or {})
                           for c, r in rows.iterrows() if c not in shapes}
            territories = {c: ring for c, ring in territories.items() if ring}
            if territories:
                fig.add_trace(go.Choropleth(
                    geojson=dict(type="FeatureCollection", features=[
                        dict(type="Feature", id=c,
                             geometry=dict(type="Polygon", coordinates=[[list(p) for p in ring]]))
                        for c, ring in territories.items()]),
                    locations=list(territories), z=rows.loc[list(territories), "log_value"], coloraxis="coloraxis",
                    marker_line=dict(width=0.5, color="#444444"),
                    hovertext=[rows.at[c, "hover"] if "hover" in rows.columns else rows.at[c, label_col]
                               for c in territories], hoverinfo="text"))
            placed = map_labels.layout_countries(
                [(code, [r[label_col], r["value_text"]], (r["lon"], r["lat"])) for code, r in rows.iterrows()],
                proj, split=anchor_labels)
            inside = [c for c, p in placed.items() if p[0] == "inside"]
            dots = [c for c, p in placed.items() if p[0] == "dot"]
            lined = [c for c, p in placed.items() if p[0] == "line"]
            add_text([placed[c][1][0] for c in inside], [placed[c][1][1] for c in inside],
                     [f"{rows.at[c, label_col]}<br>{rows.at[c, 'value_text']}" for c in inside])
            if lined:
                add_lines([(placed[c][1], placed[c][2]) for c in lined])
                add_text([placed[c][2][0] for c in lined], [placed[c][2][1] for c in lined],
                         [f"{rows.at[c, label_col]} {rows.at[c, 'value_text']}" for c in lined],
                         [placed[c][3] for c in lined])
            if dots or lined:
                add_dots(dots + lined, [placed[c][1] for c in dots + lined],
                         [f"{rows.at[c, label_col]}{'<br>' if placed[c][3] else ' '}{rows.at[c, 'value_text']}"
                          for c in dots] + [""] * len(lined),
                         [placed[c][2] for c in dots] + ["middle right"] * len(lined))
            fig.update_geos(domain=dict(x=[0, 1], y=[0, MAP_TOP_SHARE]),
                            lonaxis_range=view["lon"], lataxis_range=view["lat"],
                            projection_rotation_lon=view.get("rotation", sum(view["lon"]) / 2))
            fig.update_layout(width=width, height=height)
        else:
            callouts = {c: ab for c, ab in (callouts or {}).items() if c in rows.index}
            inside = rows.index.difference(list(callouts))
            fig.add_trace(go.Scattergeo(
                locations=inside, locationmode=locationmode, mode="text", hoverinfo="skip", textfont=font,
                text=[f"{rows.at[c, label_col]}<br>{rows.at[c, 'value_text']}" for c in inside]))
            if callouts:
                codes = list(callouts)
                add_lines([callouts[c] for c in codes])
                add_text([callouts[c][1][0] for c in codes], [callouts[c][1][1] for c in codes],
                         [f"{rows.at[c, label_col]} {rows.at[c, 'value_text']}" for c in codes],
                         ["middle right" if callouts[c][1][0] > callouts[c][0][0] else "middle left" for c in codes])
                add_dots(codes, [callouts[c][0] for c in codes])
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(family=common.get_configs('font_family'), size=common.get_configs('font_size')),
        )
        if not view:
            fig.update_geos(domain=dict(x=[0, 1], y=[0, MAP_TOP_SHARE]))
        fig.update_layout(coloraxis_colorbar=dict(**colorbar_top(colorbar_title),
                                                  **(self._log_ticks(lo, hi) if log else {})))
        io_class.save_plotly_figure(fig, file_name, save_final=save_final)
