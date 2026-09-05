import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageFont, ImageDraw, ImageColor
import common
from utils.plotting.io import IO
import warnings
from custom_logger import CustomLogger

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logger = CustomLogger(__name__)  # use custom logger

io_class = IO()


class Maps:
    _CARTO_TILE = ("https://basemaps.cartocdn.com/rastertiles/light_all"
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

    @staticmethod
    def _default_locality_image_items() -> list[dict]:
        """Default overlay pack used for screenshot tiles and arrows.

        Notes
        -----
        * approx_lon/approx_lat place the screenshot tile on the map.
        * locality/country are used to draw the connector line to the actual locality.
        * label_dlon/label_dlat and video_dlon/video_dlat control text offsets
          from the tile centre.
        * file names are assumed to match the screenshots directory.
        """

        return [
            {
                "locality": "Cape Town",
                "country": "South Africa",
                "file": "cape_town.jpg",
                "approx_lon": 0.0,
                "approx_lat": -48.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Cape Town, South Africa",
                "label_dlon": -10.0,
                "label_dlat": 9.5,
                "label_font_size": 78,
                "label_extra_width_px": 40,
                "label_extra_height_px": 16,
                "video": "0xP7JgDiBb8",
                "video_dlon": 10.0,
                "video_dlat": -9.5,
                "video_font_size": 78,
                "video_text_position": "middle right",
                "video_text_pad_x_deg": 0.8,
                "video_box_width_deg": 14.2,
                "video_box_height_deg": 1.75,
                "video_box_gap_deg": 0.00,
                "video_bottom_offset_deg": 0.0,
                "video_right_offset_deg": 0.0,
                "line_control_point_lon": 0.0,
                "line_control_point_lat": -35.0,
            },
            {
                "locality": "Tokyo",
                "country": "Japan",
                "file": "seoul.jpg",
                "approx_lon": 174.0,
                "approx_lat": 20.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Tokyo, Japan",
                "label_dlon": -6.0,
                "label_dlat": 9.5,
                "label_font_size": 78,
                "label_extra_width_px": 40,
                "label_extra_height_px": 16,
                "video": "qOx5CwCrN9k",
                "video_dlon": 8.0,
                "video_dlat": -9.5,
                "video_font_size": 78,
                "video_use_full_image_lon_bounds": True,
                "video_text_position": "middle right",
                "video_text_pad_x_deg": 0.8,
                "video_box_width_deg": 14.0,
                "video_box_height_deg": 1.95,
                "video_box_gap_deg": 0.00,
                "video_bottom_offset_deg": 0.0,
                "video_right_offset_deg": 0.0,
            },
            {
                "locality": "London",
                "country": "United Kingdom",
                "file": "london.jpg",
                "approx_lon": -32.0,
                "approx_lat": 50.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "London, UK",
                "label_dlon": -10.0,
                "label_dlat": 7.0,
                "label_font_size": 78,
                "label_extra_width_px": 40,
                "label_extra_height_px": 16,
                "video": "QI4_dGvZ5yE",
                "video_dlon": 10.0,
                "video_dlat": -7.0,
                "video_font_size": 78,
                "video_text_position": "middle right",
                "video_box_width_deg": 13.2,
                "video_box_height_deg": 1.55,
                "video_box_gap_deg": 0.00,
                "video_bottom_offset_deg": 0.0,
                "video_right_offset_deg": 0.0,
            },
            {
                "locality": "New York",
                "country": "United States",
                "file": "new_york.jpg",
                "approx_lon": -42.0,
                "approx_lat": 28.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "New York, US",
                "label_dlon": -9.0,
                "label_dlat": 9.0,
                "label_font_size": 78,
                "label_extra_width_px": 40,
                "label_extra_height_px": 16,
                "video": "_Wyg213IZDI",
                "video_dlon": 9.0,
                "video_dlat": -9.0,
                "video_font_size": 78,
                "video_text_position": "middle right",
                "video_box_width_deg": 13.2,
                "video_box_height_deg": 1.95,
                "video_box_gap_deg": 0.00,
                "video_bottom_offset_deg": 0.0,
                "video_right_offset_deg": 0.0,
            },
            {
                "locality": "Perth",
                "country": "Australia",
                "file": "sydney.jpg",
                "approx_lon": 80.0,
                "approx_lat": -30.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Perth, Australia",
                "label_dlon": -6.0,
                "label_dlat": 9.5,
                "label_font_size": 78,
                "label_extra_width_px": 40,
                "label_extra_height_px": 16,
                "video": "_Wyg213IZDI",
                "video_dlon": 8.0,
                "video_dlat": -9.5,
                "video_font_size": 78,
                "video_text_position": "middle right",
                "video_text_pad_x_deg": 0.8,
                "video_box_width_deg": 14.4,
                "video_box_height_deg": 1.75,
                "video_box_gap_deg": 0.00,
                "video_bottom_offset_deg": 0.0,
                "video_right_offset_deg": 0.0,
            },
            {
                "locality": "Sao Paulo",
                "country": "Brazil",
                "file": "sao_paulo.jpg",
                "approx_lon": -20.0,
                "approx_lat": -28.0,
                "img_half_width_deg": 16.0,
                "img_half_height_deg": 8.0,
                "label": "Sao Paulo, Brazil",
                "label_dlon": -7.5,
                "label_dlat": 8.5,
                "label_font_size": 220,
                "label_box_scale": 1.15,
                "label_extra_width_px": 40,
                "label_extra_height_px": 20,
                "label_box_height_deg": 1.8,
                "label_box_gap_deg": 0.12,
                "label_scale_font_with_image_px": False,
                "label_line_width": 10,
                "label_scale_line_width_with_box": True,
                "label_render_scale": 4,
                "label_max_render_font_px": 320,
                "video": "Ic2ERD7kt4o",
                "video_dlon": 10.0,
                "video_dlat": -8.5,
                "video_font_size": 220,
                "video_box_scale": 1.15,
                "video_scale_font_with_image_px": False,
                "video_line_width": 10,
                "video_scale_line_width_with_box": True,
                "video_render_scale": 4,
                "video_max_render_font_px": 320,
                "video_text_position": "middle right",
                "video_text_pad_x_deg": 0.8,
                "video_box_width_deg": 12.0,
                "video_box_height_deg": 2.4,
                "video_box_gap_deg": 0.00,
                "video_bottom_offset_deg": 0.0,
                "video_right_offset_deg": 0.0,
            },
        ]

    @staticmethod
    def _safe_open_image(path: str):
        """Open an image for Plotly embedding, return None if missing."""
        try:
            return Image.open(path)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    @staticmethod
    def _rect_corners(center_lon: float,
                      center_lat: float,
                      half_width_deg: float,
                      half_height_deg: float):
        """Return geographic corners for a rectangular image layer."""
        return [
            [center_lon - half_width_deg, center_lat + half_height_deg],
            [center_lon + half_width_deg, center_lat + half_height_deg],
            [center_lon + half_width_deg, center_lat - half_height_deg],
            [center_lon - half_width_deg, center_lat - half_height_deg],
        ]

    @staticmethod
    def _mercator_lat_to_y(lat: float) -> float:
        """Convert latitude in degrees to Web Mercator y."""
        lat_clamped = max(min(float(lat), 85.05112878), -85.05112878)
        lat_rad = np.deg2rad(lat_clamped)
        return float(np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0)))

    @staticmethod
    def _mercator_y_to_lat(y: float) -> float:
        """Convert Web Mercator y back to latitude in degrees."""
        lat_rad = 2.0 * np.arctan(np.exp(float(y))) - (np.pi / 2.0)
        lat_deg = np.rad2deg(lat_rad)
        return float(max(min(lat_deg, 85.05112878), -85.05112878))

    @staticmethod
    def _constant_screen_geo_bounds(center_lon: float,
                                    center_lat: float,
                                    half_width_deg: float,
                                    half_height_deg: float,
                                    keep_screen_size_constant: bool = True) -> tuple[float, float, float, float]:
        """Return geographic bounds for a tile.

        When keep_screen_size_constant is True, the longitude span remains fixed and
        the latitude span is adjusted in Web Mercator space so the tile keeps the
        same rendered screen height regardless of its latitude.
        """
        left = float(center_lon) - float(half_width_deg)
        right = float(center_lon) + float(half_width_deg)

        if not keep_screen_size_constant:
            top = float(center_lat) + float(half_height_deg)
            bottom = float(center_lat) - float(half_height_deg)
            return left, right, top, bottom

        nominal_half_height = max(float(half_height_deg), 0.0)
        half_y = abs(Maps._mercator_lat_to_y(nominal_half_height) - Maps._mercator_lat_to_y(0.0))
        center_y = Maps._mercator_lat_to_y(center_lat)
        top = Maps._mercator_y_to_lat(center_y + half_y)
        bottom = Maps._mercator_y_to_lat(center_y - half_y)
        return left, right, top, bottom

    @staticmethod
    def _tile_edge_point_from_bounds(
        center_lon: float,
        center_lat: float,
        target_lon: float,
        target_lat: float,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> tuple[float, float]:
        """Return the point where the line from the tile centre exits a rectangle."""
        dx = float(target_lon) - float(center_lon)
        dy = float(target_lat) - float(center_lat)

        if dx == 0 and dy == 0:
            return float(center_lon), float(center_lat)

        candidates: list[tuple[float, float, float]] = []

        if dx > 0:
            t = (float(right) - float(center_lon)) / dx
            y = float(center_lat) + t * dy
            if t >= 0 and float(bottom) - 1e-9 <= y <= float(top) + 1e-9:
                candidates.append((t, float(right), y))
        elif dx < 0:
            t = (float(left) - float(center_lon)) / dx
            y = float(center_lat) + t * dy
            if t >= 0 and float(bottom) - 1e-9 <= y <= float(top) + 1e-9:
                candidates.append((t, float(left), y))

        if dy > 0:
            t = (float(top) - float(center_lat)) / dy
            x = float(center_lon) + t * dx
            if t >= 0 and float(left) - 1e-9 <= x <= float(right) + 1e-9:
                candidates.append((t, x, float(top)))
        elif dy < 0:
            t = (float(bottom) - float(center_lat)) / dy
            x = float(center_lon) + t * dx
            if t >= 0 and float(left) - 1e-9 <= x <= float(right) + 1e-9:
                candidates.append((t, x, float(bottom)))

        if not candidates:
            return float(center_lon), float(center_lat)

        _, edge_lon, edge_lat = min(candidates, key=lambda row: row[0])
        return float(edge_lon), float(edge_lat)

    @staticmethod
    def _point_geojson(lon: float, lat: float, text: str, property_name: str = "label") -> dict:
        """Create a tiny GeoJSON point feature collection for a map symbol layer."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {
                        property_name: text,
                    },
                }
            ],
        }

    @staticmethod
    def _symbol_text_layer(*,
                           lon: float,
                           lat: float,
                           text: str,
                           font_size: int = 16,
                           text_color: str = "black",
                           property_name: str = "label") -> dict | None:
        """Return a crisp map-attached text layer using a GeoJSON symbol layer."""
        if not text:
            return None

        return dict(
            sourcetype="geojson",
            source=Maps._point_geojson(lon, lat, text, property_name=property_name),
            type="symbol",
            symbol=dict(
                text="{" + property_name + "}",
                placement="point",
                textfont=dict(
                    size=int(font_size),
                    color=text_color,
                ),
            ),
        )

    @staticmethod
    def _load_text_font(font_size: int):
        """Load a scalable font with sensible cross platform fallbacks."""
        size = max(int(font_size), 1)

        candidates = [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/ARIAL.TTF",
        ]

        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

        raise RuntimeError(
            "No scalable TrueType font found. Install DejaVu Sans or Arial, or add a valid font path."
        )

    @staticmethod
    def _measure_text_pixels(text: str, font_size: int) -> tuple[int, int]:
        """Measure text width and height in pixels."""
        text_value = str(text) if text is not None else ""
        if not text_value:
            return 1, max(int(font_size), 1)

        font = Maps._load_text_font(font_size)
        dummy_img = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text_value, font=font)
        width_px = max(int(np.ceil(bbox[2] - bbox[0])), 1)
        height_px = max(int(np.ceil(bbox[3] - bbox[1])), 1)
        return width_px, height_px

    @staticmethod
    def _effective_text_font_size(*,
                                  base_font_size: int,
                                  img_width_px: int | None = None,
                                  img_height_px: int | None = None,
                                  scale_with_image_px: bool = False,
                                  reference_width_px: float = 1280.0,
                                  reference_height_px: float | None = 720.0) -> int:
        """Optionally scale text size to compensate for very high resolution overlay images."""
        size = max(int(base_font_size), 1)
        if not scale_with_image_px:
            return size

        if img_width_px is None or int(img_width_px) <= 0:
            return size

        scale = float(img_width_px) / max(float(reference_width_px), 1.0)

        if (
            reference_height_px is not None
            and img_height_px is not None
            and float(reference_height_px) > 0
            and int(img_height_px) > 0
        ):
            scale = max(scale, float(img_height_px) / float(reference_height_px))

        scale = max(scale, 1.0)
        return max(int(round(size * scale)), 1)

    @staticmethod
    def _deg_per_px(*, lat: float, zoom: float = 1.3) -> tuple[float, float]:
        """Approximate geographic degrees per rendered pixel at a latitude."""
        world_px = 512.0 * (2.0 ** float(zoom))
        deg_per_px_lon = 360.0 / world_px
        deg_per_px_lat = deg_per_px_lon * max(float(np.cos(np.deg2rad(lat))), 0.2)
        return deg_per_px_lon, deg_per_px_lat

    @staticmethod
    def _text_box_total_pixel_size(*,
                                   text: str,
                                   font_size: int,
                                   horizontal_padding_px: float = 12.0,
                                   vertical_padding_px: float = 8.0) -> tuple[float, float]:
        """Return total text box width and height in pixels including padding."""
        width_px, height_px = Maps._measure_text_pixels(text=text, font_size=font_size)
        total_width_px = width_px + 2.0 * float(horizontal_padding_px)
        total_height_px = height_px + 2.0 * float(vertical_padding_px)
        return float(total_width_px), float(total_height_px)

    @staticmethod
    def _text_box_half_sizes(*,
                             text: str,
                             font_size: int,
                             lat: float,
                             zoom: float = 1.3,
                             horizontal_padding_px: float = 12.0,
                             vertical_padding_px: float = 8.0,
                             min_half_width_deg: float = 2.0,
                             min_half_height_deg: float = 0.9,
                             img_half_width_deg: float | None = None,
                             img_half_height_deg: float | None = None,
                             img_width_px: int | None = None,
                             img_height_px: int | None = None) -> tuple[float, float]:
        """Estimate geographic half sizes for a text box from measured text size."""
        total_width_px, total_height_px = Maps._text_box_total_pixel_size(
            text=text,
            font_size=font_size,
            horizontal_padding_px=horizontal_padding_px,
            vertical_padding_px=vertical_padding_px,
        )

        if (
            img_half_width_deg is not None and img_half_height_deg is not None
            and img_width_px is not None and img_height_px is not None
            and img_half_width_deg > 0 and img_half_height_deg > 0
            and img_width_px > 0 and img_height_px > 0
        ):
            deg_per_px_lon = (2.0 * float(img_half_width_deg)) / float(img_width_px)
            deg_per_px_lat = (2.0 * float(img_half_height_deg)) / float(img_height_px)
        else:
            deg_per_px_lon, deg_per_px_lat = Maps._deg_per_px(lat=lat, zoom=zoom)

        half_width_deg = max(float(min_half_width_deg), 0.5 * total_width_px * deg_per_px_lon)
        half_height_deg = max(float(min_half_height_deg), 0.5 * total_height_px * deg_per_px_lat)
        return half_width_deg, half_height_deg

    @staticmethod
    def _text_box_height_deg_from_pixels(*,
                                         total_height_px: float,
                                         lat: float,
                                         zoom: float = 1.3,
                                         min_half_height_deg: float = 0.0,
                                         img_half_height_deg: float | None = None,
                                         img_height_px: int | None = None) -> float:
        """Convert a target text box pixel height into geographic degrees."""
        if (
            img_half_height_deg is not None
            and img_height_px is not None
            and img_half_height_deg > 0
            and img_height_px > 0
        ):
            deg_per_px_lat = (2.0 * float(img_half_height_deg)) / float(img_height_px)
        else:
            _, deg_per_px_lat = Maps._deg_per_px(lat=lat, zoom=zoom)

        half_height_deg = max(float(min_half_height_deg), 0.5 * float(total_height_px) * deg_per_px_lat)
        return 2.0 * half_height_deg

    @staticmethod
    def _parse_text_position(text_position: str = "top left") -> tuple[str, str]:
        """Normalise a text position string into vertical and horizontal anchors."""
        pos = str(text_position).strip().lower().replace("_", " ")
        parts = pos.split()
        vertical = "top"
        horizontal = "left"
        if len(parts) == 1:
            if parts[0] in {"left", "center", "right"}:
                horizontal = "center" if parts[0] == "center" else parts[0]
                vertical = "middle"
            elif parts[0] in {"top", "middle", "center", "bottom"}:
                vertical = "middle" if parts[0] == "center" else parts[0]
                horizontal = "center"
        elif len(parts) >= 2:
            vertical = "middle" if parts[0] == "center" else parts[0]
            horizontal = "center" if parts[1] == "center" else parts[1]
        return vertical, horizontal

    @staticmethod
    def _box_pixel_size(*,
                        box_width_deg: float,
                        box_height_deg: float,
                        img_half_width_deg: float | None = None,
                        img_half_height_deg: float | None = None,
                        img_width_px: int | None = None,
                        img_height_px: int | None = None,
                        lat: float = 0.0,
                        zoom: float = 1.3,
                        min_width_px: int = 80,
                        min_height_px: int = 36) -> tuple[int, int]:
        """Convert geographic box dimensions into raster pixel dimensions."""
        if (
            img_half_width_deg is not None and img_half_height_deg is not None
            and img_width_px is not None and img_height_px is not None
            and img_half_width_deg > 0 and img_half_height_deg > 0
            and img_width_px > 0 and img_height_px > 0
        ):
            px_per_deg_x = float(img_width_px) / (2.0 * float(img_half_width_deg))
            px_per_deg_y = float(img_height_px) / (2.0 * float(img_half_height_deg))
        else:
            deg_per_px_lon, deg_per_px_lat = Maps._deg_per_px(lat=lat, zoom=zoom)
            px_per_deg_x = 1.0 / max(deg_per_px_lon, 1e-9)
            px_per_deg_y = 1.0 / max(deg_per_px_lat, 1e-9)

        width_px = max(int(round(float(box_width_deg) * px_per_deg_x)), int(min_width_px))
        height_px = max(int(round(float(box_height_deg) * px_per_deg_y)), int(min_height_px))
        return width_px, height_px

    @staticmethod
    def _effective_box_line_width(*,
                                  line_width: int,
                                  box_width_px: int,
                                  box_height_px: int,
                                  reference_min_dim_px: float = 120.0,
                                  max_scale: float = 3.0) -> int:
        """Scale border thickness with box size so oversized boxes do not look thinner."""
        base_line_width = max(int(line_width), 1)
        min_dim_px = max(min(int(box_width_px), int(box_height_px)), 1)
        scale = max(float(min_dim_px) / max(float(reference_min_dim_px), 1.0), 1.0)
        scale = min(scale, max(float(max_scale), 1.0))
        return max(int(round(base_line_width * scale)), 1)

    @staticmethod
    def _wrap_longitude_interval(left: float, right: float) -> tuple[float, float]:
        """Shift a longitude interval near the visible wrapped world span."""
        left_wrapped = float(left)
        right_wrapped = float(right)

        if left_wrapped > right_wrapped:
            left_wrapped, right_wrapped = right_wrapped, left_wrapped

        while left_wrapped < -180.0 and right_wrapped < -180.0:
            left_wrapped += 360.0
            right_wrapped += 360.0

        while left_wrapped > 180.0 and right_wrapped > 180.0:
            left_wrapped -= 360.0
            right_wrapped -= 360.0

        return left_wrapped, right_wrapped

    @staticmethod
    def _split_longitude_interval_for_map(left: float,
                                          right: float) -> list[tuple[float, float, float, float]]:
        """Split a longitude interval into one or two visible map segments.

        Returns tuples of:
            (segment_left, segment_right, source_x_start_frac, source_x_end_frac)

        This preserves overlays that cross the antimeridian by splitting them into
        two map aligned pieces and cropping the image source accordingly.
        """
        left_wrapped, right_wrapped = Maps._wrap_longitude_interval(left=left, right=right)
        width = float(right_wrapped) - float(left_wrapped)

        if width <= 0:
            return []

        if width >= 360.0:
            return [(-180.0, 180.0, 0.0, 1.0)]

        if -180.0 <= left_wrapped and right_wrapped <= 180.0:
            return [(left_wrapped, right_wrapped, 0.0, 1.0)]

        if right_wrapped > 180.0:
            split_frac = (180.0 - left_wrapped) / width
            return [
                (left_wrapped, 180.0, 0.0, split_frac),
                (-180.0, right_wrapped - 360.0, split_frac, 1.0),
            ]

        if left_wrapped < -180.0:
            split_frac = (-180.0 - left_wrapped) / width
            return [
                (left_wrapped + 360.0, 180.0, 0.0, split_frac),
                (-180.0, right_wrapped, split_frac, 1.0),
            ]

        return [(left_wrapped, right_wrapped, 0.0, 1.0)]

    @staticmethod
    def _crop_image_source_horizontally(source,
                                        start_frac: float,
                                        end_frac: float):
        """Crop a PIL like image source horizontally using fractional x bounds."""
        start = max(0.0, min(float(start_frac), 1.0))
        end = max(start, min(float(end_frac), 1.0))

        if start <= 0.0 and end >= 1.0:
            return source

        if not hasattr(source, "crop") or not hasattr(source, "size"):
            return None

        width_px, height_px = source.size
        if width_px <= 1 or height_px <= 0:
            return source

        left_px = max(int(np.floor(start * width_px)), 0)
        right_px = min(int(np.ceil(end * width_px)), int(width_px))
        right_px = max(right_px, left_px + 1)
        return source.crop((left_px, 0, right_px, int(height_px)))

    @staticmethod
    def _append_wrapped_rect_image_layers(*,
                                          map_layers: list,
                                          source,
                                          left: float,
                                          right: float,
                                          top: float,
                                          bottom: float,
                                          opacity: float = 1.0,
                                          below: str = "traces") -> None:
        """Append one or two image layers so rectangles crossing the antimeridian stay visible."""
        segments = Maps._split_longitude_interval_for_map(left=left, right=right)
        for seg_left, seg_right, src_start_frac, src_end_frac in segments:
            seg_source = Maps._crop_image_source_horizontally(
                source=source,
                start_frac=src_start_frac,
                end_frac=src_end_frac,
            )
            if seg_source is None:
                continue

            map_layers.append(
                dict(
                    sourcetype="image",
                    source=seg_source,
                    coordinates=[
                        [float(seg_left), float(top)],
                        [float(seg_right), float(top)],
                        [float(seg_right), float(bottom)],
                        [float(seg_left), float(bottom)],
                    ],
                    opacity=float(opacity),
                    below=below,
                )
            )

    @staticmethod
    def _parse_rgba_color(color, default_alpha: int = 255):
        """Parse a colour string or tuple into an RGBA tuple."""
        if color is None:
            return (255, 255, 255, default_alpha)

        if isinstance(color, tuple):
            if len(color) == 4:
                return tuple(int(v) for v in color)
            if len(color) == 3:
                return tuple(int(v) for v in color) + (default_alpha,)

        value = str(color).strip()
        value_lower = value.lower()
        if value_lower.startswith("rgba(") and value.endswith(")"):
            parts = [p.strip() for p in value[5:-1].split(",")]
            if len(parts) == 4:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                a_raw = float(parts[3])
                a = int(round(255.0 * a_raw)) if a_raw <= 1.0 else int(round(a_raw))
                return (r, g, b, max(0, min(a, 255)))

        if value_lower.startswith("rgb(") and value.endswith(")"):
            parts = [p.strip() for p in value[4:-1].split(",")]
            if len(parts) == 3:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                return (r, g, b, default_alpha)

        rgb = ImageColor.getrgb(value)
        if len(rgb) == 4:
            return tuple(int(v) for v in rgb)
        return tuple(int(v) for v in rgb) + (default_alpha,)

    @staticmethod
    def _render_text_box_image(*,
                               text: str,
                               font_size: int,
                               box_width_px: int,
                               box_height_px: int,
                               text_position: str = "top left",
                               text_pad_x_px: int = 8,
                               text_pad_y_px: int = 6,
                               text_color: str = "black",
                               fill_color: str = "rgba(255,255,255,0.0)",
                               line_color: str = "black",
                               line_width: int = 2,
                               render_scale: int = 6,
                               max_render_font_px: int | None = 320):
        """Render a bordered text box as a high resolution PIL image.

        The visible style stays the same, but the rasterised font size is capped
        so static PNG export does not turn large labels into black bars.
        """
        requested_scale = max(float(render_scale), 1.0)

        width_px = max(int(box_width_px), 2)
        height_px = max(int(box_height_px), 2)

        text_value = str(text) if text is not None else ""
        base_font_size = max(int(font_size), 1)

        scaled_font_size = max(int(round(base_font_size * requested_scale)), 1)
        if max_render_font_px is not None:
            scaled_font_size = min(scaled_font_size, max(int(max_render_font_px), 1))

        # Keep box, padding, and border scaling aligned with the actual raster font size
        effective_scale = max(float(scaled_font_size) / float(base_font_size), 1.0)

        canvas_width_px = max(int(round(width_px * effective_scale)), 2)
        canvas_height_px = max(int(round(height_px * effective_scale)), 2)

        img = Image.new("RGBA", (canvas_width_px, canvas_height_px), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        fill_rgba = Maps._parse_rgba_color(fill_color, default_alpha=0)
        line_rgba = Maps._parse_rgba_color(line_color)
        text_rgba = Maps._parse_rgba_color(text_color)

        scaled_line_width = max(int(round(int(line_width) * effective_scale)), 1)
        draw.rectangle(
            [(0, 0), (canvas_width_px - 1, canvas_height_px - 1)],
            outline=line_rgba,
            fill=fill_rgba,
            width=scaled_line_width,
        )

        if text_value:
            font = Maps._load_text_font(scaled_font_size)
            bbox = draw.textbbox((0, 0), text_value, font=font)

            bbox_left = int(np.floor(bbox[0]))
            bbox_top = int(np.floor(bbox[1]))
            text_w = max(int(np.ceil(bbox[2] - bbox[0])), 1)
            text_h = max(int(np.ceil(bbox[3] - bbox[1])), 1)

            vertical, horizontal = Maps._parse_text_position(text_position)

            scaled_text_pad_x_px = max(int(round(int(text_pad_x_px) * effective_scale)), 0)
            scaled_text_pad_y_px = max(int(round(int(text_pad_y_px) * effective_scale)), 0)

            if horizontal == "left":
                x_box = scaled_text_pad_x_px
            elif horizontal == "right":
                x_box = int(canvas_width_px - scaled_text_pad_x_px - text_w)
            else:
                x_box = int(round((canvas_width_px - text_w) / 2.0))

            if vertical == "top":
                y_box = scaled_text_pad_y_px
            elif vertical == "bottom":
                y_box = int(canvas_height_px - scaled_text_pad_y_px - text_h)
            else:
                y_box = int(round((canvas_height_px - text_h) / 2.0))

            x_box = max(0, min(x_box, max(canvas_width_px - text_w, 0)))
            y_box = max(0, min(y_box, max(canvas_height_px - text_h, 0)))

            draw_x = x_box - bbox_left
            draw_y = y_box - bbox_top

            stroke_width = max(int(round(effective_scale)), 1)
            stroke_fill = fill_rgba if fill_rgba[3] > 0 else (255, 255, 255, 230)

            draw.text(
                (draw_x, draw_y),
                text_value,
                font=font,
                fill=text_rgba,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )

        return img

    def _add_text_box_layer_from_bounds(self, fig, *,
                                        map_layers: list,
                                        left: float,
                                        right: float,
                                        top: float,
                                        bottom: float,
                                        text: str,
                                        font_size: int,
                                        text_position: str = "top left",
                                        text_pad_x_deg: float = 0.5,
                                        text_pad_y_deg: float = 0.3,
                                        text_color: str = "black",
                                        fill_color: str = "rgba(255,255,255,0.0)",
                                        line_color: str = "black",
                                        line_width: int = 10,
                                        render_scale: int = 4,
                                        max_render_font_px: int | None = 320,
                                        scale_line_width_with_box: bool = True,
                                        line_width_reference_px: float = 120.0,
                                        line_width_max_scale: float = 3.0,
                                        img_half_width_deg: float | None = None,
                                        img_half_height_deg: float | None = None,
                                        img_width_px: int | None = None,
                                        img_height_px: int | None = None):
        """Render a bordered text box as a map image layer."""
        if not text:
            return

        zoom = float(getattr(fig.layout.map, "zoom", 1.3) or 1.3)
        center_lat = 0.5 * (float(top) + float(bottom))
        box_width_deg = float(right) - float(left)
        box_height_deg = float(top) - float(bottom)
        box_width_px, box_height_px = self._box_pixel_size(
            box_width_deg=box_width_deg,
            box_height_deg=box_height_deg,
            img_half_width_deg=img_half_width_deg,
            img_half_height_deg=img_half_height_deg,
            img_width_px=img_width_px,
            img_height_px=img_height_px,
            lat=center_lat,
            zoom=zoom,
        )

        text_pad_x_px = 8
        text_pad_y_px = 6
        if box_width_deg > 0:
            text_pad_x_px = max(int(round(float(text_pad_x_deg) * box_width_px / box_width_deg)), 0)
        if box_height_deg > 0:
            text_pad_y_px = max(int(round(float(text_pad_y_deg) * box_height_px / box_height_deg)), 0)

        effective_line_width = max(int(line_width), 1)
        if bool(scale_line_width_with_box):
            effective_line_width = self._effective_box_line_width(
                line_width=effective_line_width,
                box_width_px=box_width_px,
                box_height_px=box_height_px,
                reference_min_dim_px=float(line_width_reference_px),
                max_scale=float(line_width_max_scale),
            )

        box_img = self._render_text_box_image(
            text=str(text),
            font_size=int(font_size),
            box_width_px=box_width_px,
            box_height_px=box_height_px,
            text_position=text_position,
            text_pad_x_px=text_pad_x_px,
            text_pad_y_px=text_pad_y_px,
            text_color=text_color,
            fill_color=fill_color,
            line_color=line_color,
            line_width=effective_line_width,
            render_scale=int(render_scale),
            max_render_font_px=max_render_font_px,
        )

        self._append_wrapped_rect_image_layers(
            map_layers=map_layers,
            source=box_img,
            left=float(left),
            right=float(right),
            top=float(top),
            bottom=float(bottom),
            opacity=1.0,
            below="traces",
        )

    def mapbox_map_footage(self, df: pd.DataFrame, *,
                           footage_col: str = "total_time",
                           hover_data=None,
                           hover_name: str | None = None,
                           marker_size: int = 3,
                           color_scale: str | list = "Viridis_r",
                           show_colorbar: bool = True,
                           colorbar_title: str | None = None,
                           log_colour: bool = True,
                           cap_quantile: float = 0.99,
                           range_q_low: float = 0.02,
                           range_q_high: float = 0.98,
                           show_images: bool = False,
                           uniform_label_box_height: bool = True,
                           uniform_video_box_height: bool = True,
                           label_box_height_scale: float = 0.5,
                           video_box_height_scale: float = 0.5,
                           image_items: list[dict] | None = None,
                           screenshots_dir: str | None = None,
                           file_name: str = "mapbox_map_footage",
                           save_final: bool = True):
        """Scatter map of cities coloured by footage amount using a continuous hue based scale."""

        if df is None or len(df) == 0:
            return

        if screenshots_dir is None:
            screenshots_dir = os.path.join(common.root_dir, "screenshots")

        df_plot = df.copy()
        if footage_col not in df_plot.columns:
            raise KeyError(f"Missing column: {footage_col}")

        df_plot = df_plot.dropna(subset=["lat", "lon"]).copy()

        df_plot["footage_hours"] = pd.to_numeric(df_plot[footage_col], errors="coerce") / 3600.0
        df_plot["footage_hours"] = df_plot["footage_hours"].fillna(0.0)

        x = df_plot["footage_hours"].clip(lower=0.0)

        cap = x.quantile(cap_quantile) if len(x) else 0.0
        if pd.isna(cap) or cap <= 0:
            cap = float(x.max()) if len(x) else 0.0
        x_capped = x.clip(upper=cap)

        df_plot["footage_log_capped"] = np.log1p(x_capped)

        colour_col = "footage_log_capped" if log_colour else "footage_hours"

        if log_colour:
            q = df_plot[colour_col].quantile([range_q_low, range_q_high])
            cmin = float(q.iloc[0]) if len(q) else float(df_plot[colour_col].min())
            cmax = float(q.iloc[1]) if len(q) else float(df_plot[colour_col].max())
            if (not np.isfinite(cmin)) or (not np.isfinite(cmax)) or cmin == cmax:
                cmin = float(df_plot[colour_col].min())
                cmax = float(df_plot[colour_col].max())
        else:
            cmin = float(df_plot[colour_col].min())
            cmax = float(df_plot[colour_col].max())

        if cmin == cmax:
            cmax = cmin + 1e-9

        hover_data_local = hover_data
        if hover_data_local is None:
            hover_data_local = ["footage_hours"]
        elif isinstance(hover_data_local, list):
            if "footage_hours" not in hover_data_local:
                hover_data_local = list(hover_data_local) + ["footage_hours"]
        elif isinstance(hover_data_local, dict):
            hover_data_local = dict(hover_data_local)
            hover_data_local.setdefault("footage_hours", True)

        if log_colour:
            if isinstance(hover_data_local, list):
                if "footage_log_capped" not in hover_data_local:
                    hover_data_local = list(hover_data_local) + ["footage_log_capped"]
            elif isinstance(hover_data_local, dict):
                hover_data_local.setdefault("footage_log_capped", True)

        if colorbar_title is None:
            colorbar_title = "" if log_colour else "Footage (hours)"

        fig = px.scatter_map(
            df_plot,
            lat="lat",
            lon="lon",
            hover_data=hover_data_local,
            hover_name=hover_name,
            color=colour_col,
            color_continuous_scale=color_scale,
            range_color=(cmin, cmax),
            zoom=1.3,  # type: ignore
            map_style="carto-positron",
        )

        fig.update_traces(marker=dict(size=marker_size, opacity=0.95), selector=dict(type="scattermap"))

        halo_size = max(marker_size + 2, 2)
        outline_trace = go.Scattermap(
            lat=df_plot["lat"],
            lon=df_plot["lon"],
            mode="markers",
            marker=dict(size=halo_size, color="black", opacity=0.75),
            hoverinfo="skip",
            showlegend=False,
            subplot="map",
            name="_outline",
        )
        fig.add_trace(outline_trace)

        data_list = list(fig.data)
        data_list = [data_list[-1]] + data_list[:-1]
        fig.data = tuple(data_list)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size"),
            ),
            showlegend=False,
        )

        fig.update_coloraxes(
            showscale=show_colorbar,
            colorbar=dict(
                title=colorbar_title,
                x=0.5,
                xanchor="center",
                y=0.02 if not show_images else 0.01,
                yanchor="bottom",
                orientation="h",
                len=0.55,
                thickness=10,
                bgcolor="rgba(255,255,255,0.85)",
            ) if show_colorbar else {}
        )

        if show_images:
            if image_items is None:
                image_items = self._default_locality_image_items()

            def _item_anchor_lon(item: dict) -> float | None:
                value = item.get("img_lon", item.get("approx_lon"))
                return float(value) if value is not None else None

            def _item_anchor_lat(item: dict) -> float | None:
                value = item.get("img_lat", item.get("approx_lat"))
                return float(value) if value is not None else None

            def _item_image_bounds(item: dict):
                lon0 = _item_anchor_lon(item)
                lat0 = _item_anchor_lat(item)
                if lon0 is None or lat0 is None:
                    return None

                half_width = float(item.get("img_half_width_deg", 10.0))
                half_height = float(item.get("img_half_height_deg", 5.0))
                keep_screen_size_constant = bool(item.get("keep_screen_size_constant", True))
                return self._constant_screen_geo_bounds(
                    center_lon=lon0,
                    center_lat=lat0,
                    half_width_deg=half_width,
                    half_height_deg=half_height,
                    keep_screen_size_constant=keep_screen_size_constant,
                )

            def _item_image_corners(item: dict):
                bounds = _item_image_bounds(item)
                if bounds is None:
                    return None

                left, right, top, bottom = bounds
                return [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ]

            map_layers = list(fig.layout.map.layers) if getattr(fig.layout.map, "layers", None) else []

            for item in image_items:
                file_name_img = item.get("file")
                corners = _item_image_corners(item)

                if file_name_img and corners is not None:
                    img_path = os.path.join(screenshots_dir, file_name_img)
                    img = self._safe_open_image(img_path)
                    if img is not None:
                        item["_img_size_px"] = tuple(img.size)
                        self._append_wrapped_rect_image_layers(
                            map_layers=map_layers,
                            source=img,
                            left=float(corners[0][0]),
                            right=float(corners[1][0]),
                            top=float(corners[0][1]),
                            bottom=float(corners[2][1]),
                            opacity=float(item.get("opacity", 1.0)),
                            below=item.get("below", "traces"),
                        )
                    elif item.get("source") is not None:
                        source_obj = item["source"]
                        if hasattr(source_obj, "size"):
                            item["_img_size_px"] = tuple(source_obj.size)
                        self._append_wrapped_rect_image_layers(
                            map_layers=map_layers,
                            source=source_obj,
                            left=float(corners[0][0]),
                            right=float(corners[1][0]),
                            top=float(corners[0][1]),
                            bottom=float(corners[2][1]),
                            opacity=float(item.get("opacity", 1.0)),
                            below=item.get("below", "traces"),
                        )

            uniform_label_box_height_px = 72 if bool(uniform_label_box_height) else None
            uniform_video_box_height_px = 72 if bool(uniform_video_box_height) else None

            for item in image_items:
                label = item.get("label")
                anchor_lon = _item_anchor_lon(item)
                anchor_lat = _item_anchor_lat(item)

                if anchor_lon is not None and anchor_lat is not None:
                    bounds = _item_image_bounds(item)
                    if bounds is None:
                        continue

                    img_left, img_right, img_top, img_bottom = bounds
                    img_geo_half_width_deg = 0.5 * (img_right - img_left)
                    img_geo_half_height_deg = 0.5 * (img_top - img_bottom)

                    img_size = item.get("_img_size_px")
                    img_width_px = None
                    img_height_px = None
                    if isinstance(img_size, tuple) and len(img_size) == 2:
                        img_width_px = int(img_size[0])
                        img_height_px = int(img_size[1])

                    visible_img_left = max(img_left, -180.0)
                    visible_img_right = min(img_right, 180.0)

                    label_use_full_image_lon_bounds = bool(item.get("label_use_full_image_lon_bounds", False))
                    video_use_full_image_lon_bounds = bool(item.get("video_use_full_image_lon_bounds", False))

                    label_left_bound = img_left if label_use_full_image_lon_bounds else visible_img_left
                    label_right_bound = img_right if label_use_full_image_lon_bounds else visible_img_right
                    video_left_bound = img_left if video_use_full_image_lon_bounds else visible_img_left
                    video_right_bound = img_right if video_use_full_image_lon_bounds else visible_img_right

                    if label:
                        label_font_size = self._effective_text_font_size(
                            base_font_size=int(item.get("label_font_size", 60)),
                            img_width_px=img_width_px,
                            img_height_px=img_height_px,
                            scale_with_image_px=bool(item.get("label_scale_font_with_image_px", False)),
                            reference_width_px=float(item.get("label_reference_width_px", 1280.0)),
                            reference_height_px=float(item.get("label_reference_height_px", 720.0)),
                        )

                        label_half_w_auto, label_half_h_auto = self._text_box_half_sizes(
                            text=str(label),
                            font_size=label_font_size,
                            lat=img_top,
                            img_half_width_deg=img_geo_half_width_deg,
                            img_half_height_deg=img_geo_half_height_deg,
                            img_width_px=img_width_px,
                            img_height_px=img_height_px,
                        )

                        label_box_scale = float(item.get("label_box_scale", 1.0))
                        label_box_width_deg = float(
                            item.get("label_box_width_deg", 2.0 * label_half_w_auto * label_box_scale)
                        )

                        label_extra_width_px = float(item.get("label_extra_width_px", 0.0))
                        if label_extra_width_px > 0 and img_width_px is not None and img_width_px > 0:
                            deg_per_px_lon = (2.0 * float(img_geo_half_width_deg)) / float(img_width_px)
                            label_box_width_deg += label_extra_width_px * deg_per_px_lon

                        if "label_box_height_deg" in item:
                            label_box_height_deg = float(item["label_box_height_deg"])
                        elif uniform_label_box_height_px is not None:
                            label_box_height_deg = self._text_box_height_deg_from_pixels(
                                total_height_px=uniform_label_box_height_px,
                                lat=img_top,
                                img_half_height_deg=img_geo_half_height_deg,
                                img_height_px=img_height_px,
                                min_half_height_deg=0.0,
                            )
                        else:
                            label_box_height_deg = float(
                                2.0 * label_half_h_auto * label_box_scale * float(label_box_height_scale)
                            )

                        label_extra_height_px = float(item.get("label_extra_height_px", 0.0))
                        if label_extra_height_px > 0:
                            label_box_height_deg += self._text_box_height_deg_from_pixels(
                                total_height_px=label_extra_height_px,
                                lat=img_top,
                                img_half_height_deg=img_geo_half_height_deg,
                                img_height_px=img_height_px,
                                min_half_height_deg=0.0,
                            )

                        label_gap_deg = float(item.get("label_box_gap_deg", 0.0))
                        label_left_offset_deg = float(item.get("label_box_left_offset_deg", 0.0))
                        label_top_offset_deg = float(item.get("label_box_top_offset_deg", 0.0))
                        label_text_position = str(item.get("label_text_position", "top left"))
                        label_text_pad_x_deg = float(item.get("label_text_pad_x_deg", 0.5))
                        label_text_pad_y_deg = float(item.get("label_text_pad_y_deg", 0.3))

                        label_left = max(img_left + label_left_offset_deg, label_left_bound)
                        label_right = label_left + label_box_width_deg
                        if label_right > label_right_bound:
                            label_right = label_right_bound
                            label_left = max(label_right - label_box_width_deg, label_left_bound)

                        label_top = img_top + label_gap_deg + label_box_height_deg + label_top_offset_deg

                        self._add_text_box_layer_from_bounds(
                            fig,
                            map_layers=map_layers,
                            left=label_left,
                            right=label_right,
                            top=label_top,
                            bottom=label_top - label_box_height_deg,
                            text=str(label),
                            font_size=label_font_size,
                            text_position=label_text_position,
                            text_pad_x_deg=label_text_pad_x_deg,
                            text_pad_y_deg=label_text_pad_y_deg,
                            line_width=int(item.get("label_line_width", 10)),
                            render_scale=int(item.get("label_render_scale", 4)),
                            scale_line_width_with_box=bool(item.get("label_scale_line_width_with_box", True)),
                            line_width_reference_px=float(item.get("label_line_width_reference_px", 120.0)),
                            line_width_max_scale=float(item.get("label_line_width_max_scale", 3.0)),
                            img_half_width_deg=img_geo_half_width_deg,
                            img_half_height_deg=img_geo_half_height_deg,
                            img_width_px=img_width_px,
                            img_height_px=img_height_px,
                        )

                    video = item.get("video")
                    if video:
                        video_font_size = self._effective_text_font_size(
                            base_font_size=int(item.get("video_font_size", 78)),
                            img_width_px=img_width_px,
                            img_height_px=img_height_px,
                            scale_with_image_px=bool(item.get("video_scale_font_with_image_px", False)),
                            reference_width_px=float(item.get("video_reference_width_px", 1280.0)),
                            reference_height_px=float(item.get("video_reference_height_px", 720.0)),
                        )

                        video_half_w_auto, video_half_h_auto = self._text_box_half_sizes(
                            text=str(video),
                            font_size=video_font_size,
                            lat=img_bottom,
                            img_half_width_deg=img_geo_half_width_deg,
                            img_half_height_deg=img_geo_half_height_deg,
                            img_width_px=img_width_px,
                            img_height_px=img_height_px,
                        )

                        video_box_scale = float(item.get("video_box_scale", 1.0))
                        video_box_width_deg = float(
                            item.get("video_box_width_deg", 2.0 * video_half_w_auto * video_box_scale)
                        )

                        video_extra_width_px = float(item.get("video_extra_width_px", 0.0))
                        if video_extra_width_px > 0 and img_width_px is not None and img_width_px > 0:
                            deg_per_px_lon = (2.0 * float(img_geo_half_width_deg)) / float(img_width_px)
                            video_box_width_deg += video_extra_width_px * deg_per_px_lon

                        if "video_box_height_deg" in item:
                            video_box_height_deg = float(item["video_box_height_deg"])
                        elif uniform_video_box_height_px is not None:
                            video_box_height_deg = self._text_box_height_deg_from_pixels(
                                total_height_px=uniform_video_box_height_px,
                                lat=img_bottom,
                                img_half_height_deg=img_geo_half_height_deg,
                                img_height_px=img_height_px,
                                min_half_height_deg=0.0,
                            )
                        else:
                            video_box_height_deg = float(
                                2.0 * video_half_h_auto * video_box_scale * float(video_box_height_scale)
                            )

                        video_gap_deg = float(item.get("video_box_gap_deg", 0.0))
                        video_right_offset_deg = float(item.get("video_box_right_offset_deg", 0.0))
                        video_bottom_offset_deg = float(item.get("video_box_bottom_offset_deg", 0.0))
                        video_text_position = str(item.get("video_text_position", "middle center"))
                        video_text_pad_x_deg = float(item.get("video_text_pad_x_deg", 0.5))
                        video_text_pad_y_deg = float(item.get("video_text_pad_y_deg", 0.3))

                        video_right = min(img_right - video_right_offset_deg, video_right_bound)
                        video_left = video_right - video_box_width_deg
                        if video_left < video_left_bound:
                            video_left = video_left_bound
                            video_right = min(video_left + video_box_width_deg, video_right_bound)

                        video_bottom = img_bottom - video_gap_deg - video_box_height_deg - video_bottom_offset_deg

                        self._add_text_box_layer_from_bounds(
                            fig,
                            map_layers=map_layers,
                            left=video_left,
                            right=video_right,
                            top=video_bottom + video_box_height_deg,
                            bottom=video_bottom,
                            text=str(video),
                            font_size=video_font_size,
                            text_position=video_text_position,
                            text_pad_x_deg=video_text_pad_x_deg,
                            text_pad_y_deg=video_text_pad_y_deg,
                            line_width=int(item.get("video_line_width", 10)),
                            render_scale=int(item.get("video_render_scale", 4)),
                            scale_line_width_with_box=bool(item.get("video_scale_line_width_with_box", True)),
                            line_width_reference_px=float(item.get("video_line_width_reference_px", 120.0)),
                            line_width_max_scale=float(item.get("video_line_width_max_scale", 3.0)),
                            img_half_width_deg=img_geo_half_width_deg,
                            img_half_height_deg=img_geo_half_height_deg,
                            img_width_px=img_width_px,
                            img_height_px=img_height_px,
                        )

            if map_layers:
                fig.update_layout(map=dict(layers=map_layers))

            if {"locality", "country", "lat", "lon"}.issubset(df_plot.columns):
                for item in image_items:
                    if not ("locality" in item and "country" in item):
                        continue

                    anchor_lon = _item_anchor_lon(item)
                    anchor_lat = _item_anchor_lat(item)
                    if anchor_lon is None or anchor_lat is None:
                        continue

                    locality = str(item["locality"]).strip().lower()
                    country = str(item["country"]).strip().lower()

                    rows = df_plot[
                        (df_plot["locality"].astype(str).str.strip().str.lower() == locality)
                        & (df_plot["country"].astype(str).str.strip().str.lower() == country)
                    ]
                    if rows.empty:
                        continue

                    lon_locality = float(rows["lon"].iloc[0])
                    lat_locality = float(rows["lat"].iloc[0])

                    bounds = _item_image_bounds(item)
                    if bounds is None:
                        continue
                    img_left, img_right, img_top, img_bottom = bounds

                    edge_lon, edge_lat = self._tile_edge_point_from_bounds(
                        center_lon=anchor_lon,
                        center_lat=anchor_lat,
                        target_lon=lon_locality,
                        target_lat=lat_locality,
                        left=img_left,
                        right=img_right,
                        top=img_top,
                        bottom=img_bottom,
                    )

                    fig.add_trace(
                        go.Scattermap(
                            lon=[edge_lon, lon_locality],
                            lat=[edge_lat, lat_locality],
                            mode="lines",
                            line=dict(width=2, color="black"),
                            hoverinfo="skip",
                            showlegend=False,
                            subplot="map",
                        )
                    )

        self._apply_carto_key(fig)
        io_class.save_plotly_figure(fig, file_name, save_final=save_final)

    def world_map_ss(self, df, *, title=None, projection="natural earth", df_mapping=None, show_images=False,
                     hover_data=None, save_file=False, save_final=False, name_file=None, show_colorbar=True,
                     colorbar_title="Footage (hours)", color_scale="Turbo", marker_size=3, filter_zero=True):
        """
        World map with locality dots shown in a single muted red colour.

        Cities with lower footage hours are more transparent.
        Cities with higher footage hours are less transparent.

        Expected input:
            df must be locality level data and contain:
                - locality
                - country
                - either:
                    a) lat, lon
                    b) or df_mapping with locality, country, lat, lon
                - either:
                    a) footage_hours
                    b) or total_time in seconds

        Notes:
            - No choropleth is drawn.
            - All markers use the same muted red colour.
            - Marker transparency depends on raw footage_hours.
            - Hover text shows raw footage hours.
            - No colourbar is shown because colour is fixed.
        """

        def _hex_to_rgb(hex_colour):
            hex_colour = hex_colour.lstrip("#")
            return tuple(int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))

        palette = {
            "ocean": "#D6ECFA",
            "land": "#F3E2CC",
            "country": "#D8D2C7",
            "coast": "#DDD8CF",
            "dot": "#000000",
            "connector": "#3B3B3B",
            "label_bg": "rgba(255,255,255,0.88)",
            "label_border": "#6B7280",
            "label_text": "#1F2937",
            "paper_bg": "white",
        }

        df_plot = df.copy()

        if "locality" not in df_plot.columns or "country" not in df_plot.columns:
            raise ValueError("`df` must contain 'locality' and 'country' columns.")

        country_name_map = {"Türkiye": "Turkey"}
        df_plot["country"] = df_plot["country"].replace(country_name_map)

        # Build raw footage hours from total_time if needed
        if "footage_hours" not in df_plot.columns:
            if "total_time" not in df_plot.columns:
                raise ValueError("`df` must contain either 'footage_hours' or 'total_time'.")
            df_plot["footage_hours"] = pd.to_numeric(df_plot["total_time"], errors="coerce") / 3600.0

        # Add coordinates from df_mapping if lat lon are missing
        if not {"lat", "lon"}.issubset(df_plot.columns):
            if df_mapping is None:
                raise ValueError("`df` must contain 'lat' and 'lon', or you must pass `df_mapping`.")
            df_mapping_local = df_mapping.copy()
            df_mapping_local["country"] = df_mapping_local["country"].replace(country_name_map)

            df_plot = df_plot.merge(
                df_mapping_local[["locality", "country", "lat", "lon"]].drop_duplicates(),
                on=["locality", "country"],
                how="left",
            )

        # Keep only rows with valid coordinates and footage
        df_plot["lat"] = pd.to_numeric(df_plot["lat"], errors="coerce")
        df_plot["lon"] = pd.to_numeric(df_plot["lon"], errors="coerce")
        df_plot["footage_hours"] = pd.to_numeric(df_plot["footage_hours"], errors="coerce")

        if filter_zero:
            df_plot = df_plot[
                df_plot["footage_hours"].notna()
                & (df_plot["footage_hours"] > 0)
                & df_plot["lat"].notna()
                & df_plot["lon"].notna()
            ].copy()
        else:
            df_plot = df_plot[
                df_plot["footage_hours"].notna()
                & (df_plot["footage_hours"] >= 0)
                & df_plot["lat"].notna()
                & df_plot["lon"].notna()
            ].copy()

        if df_plot.empty:
            raise ValueError("No valid locality rows remain after filtering.")

        # If there are duplicate locality rows, aggregate to one row per locality
        agg_cols = {
            "lat": "first",
            "lon": "first",
            "footage_hours": "sum",
            "total_time": "sum"
        }

        extra_hover_cols = []
        for col in (hover_data or []):
            if col in df_plot.columns and col not in {"locality", "country", "lat", "lon", "footage_hours"}:
                if pd.api.types.is_numeric_dtype(df_plot[col]):
                    agg_cols[col] = "sum"
                else:
                    agg_cols[col] = "first"
                extra_hover_cols.append(col)

        df_plot = (
            df_plot.groupby(["locality", "country"], as_index=False)
            .agg(agg_cols)
            .copy()
        )

        # Keep only non negative hours
        df_plot = df_plot[df_plot["footage_hours"] >= 0].copy()

        if df_plot.empty:
            raise ValueError("No valid non negative 'footage_hours' values remain.")

        # # Transparency based on raw footage hours, but keep the upper bound lower
        # # so dense regions do not become visually overwhelming.
        # hours_min = float(df_plot["footage_hours"].min())
        # hours_max = float(df_plot["footage_hours"].max())

        # if hours_max == hours_min:
        #     df_plot["marker_opacity"] = 0.55
        # else:
        #     norm = (df_plot["footage_hours"] - hours_min) / (hours_max - hours_min)
        #     df_plot["marker_opacity"] = 0.18 + 0.42 * np.sqrt(norm)

        # # Build per point rgba colours instead of using a global marker opacity.
        # # Cities with > 100 footage hours are drawn in red; all others use the default dot colour.
        # dot_r, dot_g, dot_b = _hex_to_rgb(palette["dot"])
        # red_r, red_g, red_b = _hex_to_rgb("#FF0000")

        # def _make_color(row):
        #     a = float(row["marker_opacity"])
        #     if row["total_time"] > 100000:  # more than 100k s of footage
        #         return f"rgba({red_r},{red_g},{red_b},1.0)"
        #     return f"rgba({dot_r},{dot_g},{dot_b},{a:.4f})"

        # df_plot["marker_color"] = df_plot.apply(_make_color, axis=1)

        fig = go.Figure()

        # Build hover text
        customdata_parts = [
            df_plot["country"].to_numpy(),
            df_plot["total_time"].to_numpy(),
        ]

        for col in extra_hover_cols:
            customdata_parts.append(df_plot[col].to_numpy())

        customdata = np.column_stack(customdata_parts)  # noqa: F841

        hover_lines = [
            "<b>%{text}</b>",
            "Country: %{customdata[0]}",
            "Footage: %{customdata[1]:.2f} hours",
        ]
        for i, col in enumerate(extra_hover_cols, start=2):
            hover_lines.append(f"{col}: " + "%{customdata[" + str(i) + "]}")
        hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

        # Split into two dataframes so red markers always render on top
        df_normal = df_plot[df_plot["total_time"] <= 100000]
        df_red = df_plot[df_plot["total_time"] > 100000]
        logger.info(f"Cities with more than 100,000 s of footage: {len(df_red)}.")

        dot_r, dot_g, dot_b = _hex_to_rgb(palette["dot"])

        # Normal cities — bottom layer
        if not df_normal.empty:
            customdata_normal = np.column_stack(
                [df_normal["country"].to_numpy(), df_normal["total_time"].to_numpy()]
                + [df_normal[col].to_numpy() for col in extra_hover_cols]
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=df_normal["lon"],
                    lat=df_normal["lat"],
                    text=df_normal["locality"],
                    customdata=customdata_normal,
                    mode="markers",
                    hovertemplate=hovertemplate,
                    marker=dict(
                        size=marker_size,
                        color=f"rgba({dot_r},{dot_g},{dot_b},0.35)",
                        symbol="circle",
                        line=dict(width=0),
                        showscale=False,
                    ),
                    showlegend=False,
                    name="cities",
                )
            )

        # High-footage cities — top layer, always rendered above normal markers
        if not df_red.empty:
            customdata_red = np.column_stack(
                [df_red["country"].to_numpy(), df_red["total_time"].to_numpy()]
                + [df_red[col].to_numpy() for col in extra_hover_cols]
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=df_red["lon"],
                    lat=df_red["lat"],
                    text=df_red["locality"],
                    customdata=customdata_red,
                    mode="markers",
                    hovertemplate=hovertemplate,
                    marker=dict(
                        size=marker_size + 2,
                        color="rgba(255,0,0,1.0)",
                        symbol="square",
                        # line=dict(
                        #     width=1.5,
                        #     color="rgba(80,0,0,1.0)",
                        # ),
                        showscale=False,
                    ),
                    showlegend=False,
                    name="cities_highlight",
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
            locality_images = [
                {
                    "locality": "Tokyo",
                    "country": "Japan",
                    "file": "tokyo.png",
                    "x": 0.92, "y": 0.62,
                    "approx_lon": 165.2, "approx_lat": 17.2,
                    "label": "Tokyo, Japan",
                    "x_label": 0.97, "y_label": 0.693,
                    "video": "-YNLQlmPnqA",
                    "x_video": 0.91719, "y_video": 0.560,
                },
                {
                    "locality": "Cape Town",
                    "country": "South Africa",
                    "file": "cape_town.jpg",
                    "x": 0.64, "y": 0.19,
                    "approx_lon": 44.0, "approx_lat": -48.0,
                    "label": "Cape Town, South Africa",
                    "x_label": 0.6385, "y_label": 0.24,
                    "video": "0xP7JgDiBb8",
                    "x_video": 0.69, "y_video": 0.12,
                },
                {
                    "locality": "San Francisco",
                    "country": "United States",
                    "file": "san_francisco.png",
                    "x": 0.13, "y": 0.56,
                    "approx_lon": -123.5, "approx_lat": 8.0,
                    "label": "San Francisco, CA, USA",
                    "x_label": 0.08, "y_label": 0.618,
                    "video": "HZrm3s4UsgU",
                    "x_video": 0.131, "y_video": 0.503,
                },
                {
                    "locality": "London",
                    "country": "United Kingdom",
                    "file": "london.png",
                    "x": 0.395, "y": 0.66,
                    "approx_lon": -36.0, "approx_lat": 34.0,
                    "label": "London, UK",
                    "x_label": 0.36845, "y_label": 0.733,
                    "video": "Bs3MZ4wWMQs",
                    "x_video": 0.418, "y_video": 0.60,
                },
                {
                    "locality": "Sao Paulo",
                    "country": "Brazil",
                    "file": "sao_paulo.jpg",
                    "x": 0.43, "y": 0.27,
                    "approx_lon": -41.0, "approx_lat": -36.0,
                    "label": "Sao Paulo, Brazil",
                    "x_label": 0.414, "y_label": 0.32,
                    "video": "Ic2ERD7kt4o",
                    "x_video": 0.458, "y_video": 0.20,
                },
                {
                    "locality": "Perth",
                    "country": "Australia",
                    "file": "perth.png",
                    "x": 0.72, "y": 0.33,
                    "approx_lon": 72.0, "approx_lat": -25.0,
                    "label": "Perth, WA, Australia",
                    "x_label": 0.74959, "y_label": 0.392,
                    "video": "xTDUhnnj3q4",
                    "x_video": 0.77, "y_video": 0.26,
                },
            ]

            path_screenshots = os.path.join(common.root_dir, "screenshots")

            def _img(path):
                try:
                    return Image.open(path)
                except FileNotFoundError:
                    return None

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

            for item in locality_images:
                img_path = os.path.join(path_screenshots, item["file"])
                img = _img(img_path)
                if img is None:
                    continue

                fig.add_layout_image(
                    dict(
                        source=img,
                        xref="paper",
                        yref="paper",
                        x=item["x"],
                        y=item["y"],
                        sizex=0.1,
                        sizey=0.1,
                        xanchor="center",
                        yanchor="middle",
                        layer="above",
                    )
                )

                _add_box_annotation(
                    text=item["label"],
                    x=item["x_label"],
                    y=item["y_label"],
                    font_size=12,
                )

                _add_box_annotation(
                    text=item["video"],
                    x=item["x_video"],
                    y=item["y_video"],
                    font_size=10,
                )

            # Connector lines from screenshot area to actual locality
            locality_lookup = df_plot.copy()
            locality_lookup["locality_lower"] = locality_lookup["locality"].str.lower()
            locality_lookup["country_lower"] = locality_lookup["country"].str.lower()

            for item in locality_images:
                row = locality_lookup[
                    (locality_lookup["locality_lower"] == item["locality"].lower())
                    & (locality_lookup["country_lower"] == item["country"].lower())
                ]
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

            yolo_img = _img(os.path.join(path_screenshots, "toronto_yolo.jpg"))
            if yolo_img is not None:
                fig.add_layout_image(
                    dict(
                        source=yolo_img,
                        xref="paper",
                        yref="paper",
                        x=0.12,
                        y=0.158,
                        sizex=0.24,
                        sizey=0.24,
                        xanchor="center",
                        yanchor="middle",
                        layer="above",
                    )
                )

                _add_box_annotation(
                    text="Example of YOLO output (Toronto, ON, Canada)",
                    x=0.00,
                    y=0.278,
                    font_size=12,
                )

                _add_box_annotation(
                    text="3ai7SUaPoHM",
                    x=0.1925,
                    y=0.019,
                    font_size=10,
                )

        if name_file is None:
            name_file = "world_map_locality_footage"

        io_class.save_plotly_figure(fig, name_file, save_final=save_final)

    def mapbox_map(self, df, density_col=None, density_radius=30, hover_data=None, hover_name=None,
                   marker_size=5, file_name="mapbox_map", save_final=True):
        """Generates a world map of cities using Mapbox, with optional density visualization.

        This method can create either:
            1. A simple scatter map showing locality locations colored by continent.
            2. A density map showing intensity values based on a specified column.

        Args:
            df (pandas.DataFrame): DataFrame containing mapping information.
                Required columns: "lat", "lon", "locality", "continent".
            density_col (str, optional): Column name for density values.
                If provided, a density map is generated. Defaults to None.
            density_radius (int, optional): The pixel radius for density spread. Defaults to 30.
            hover_data (list, optional): List of additional DataFrame columns to display when hovering.
                Defaults to None.
            hover_name (list, optional): title on top of hover popup.
            marker_size (int, optional): size of markers.
            file_name (str, optional): Name of the saved file (without extension). Defaults to "mapbox_map".
            save_final (bool, optional): If True, saves the figure. Defaults to True.

        Returns:
            None: The Plotly figure is created, displayed, and optionally saved.
        """
        # Draw scatter map if no density column is provided
        if not density_col:
            fig = px.scatter_map(
                df,
                lat="lat",
                lon="lon",
                hover_data=hover_data,
                hover_name=hover_name,
                color=df["continent"],
                zoom=1.3  # pyright: ignore[reportArgumentType]
            )

            # Apply marker size
            fig.update_traces(marker=dict(size=marker_size))

        # Draw density map if density column is provided
        else:
            fig = px.density_mapbox(
                df,
                lat="lat",
                lon="lon",
                z=density_col,  # Use density column for intensity
                radius=density_radius,  # Control the spread of density
                zoom=2.5,  # Initial zoom level for density view # pyright: ignore[reportArgumentType]
                center=dict(
                    lat=df["lat"].mean(),
                    lon=df["lon"].mean()
                ),  # Center map on mean coordinates
                mapbox_style="carto-positron",  # Light and clean map style
                hover_data=hover_data,
                hover_name=hover_name
            )

        # Update map layout to improve appearance
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
            mapbox=dict(zoom=1.3),
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

        # Save the figure if requested
        self._apply_carto_key(fig)
        io_class.save_plotly_figure(fig, file_name, save_final=save_final)

    def world_map(self, df_mapping):
        """
        Generate a world map with highlighted countries and red markers for cities using Plotly.

        - Highlights countries based on the cities present in the dataset.
        - Adds scatter points for each locality with detailed socioeconomic and traffic-related hover info.
        - Adjusts map appearance to improve clarity and remove irrelevant regions like Antarctica.

        Args:
            df_mapping (pd.DataFrame): A DataFrame with columns:
                ['locality', 'state', 'country', 'lat', 'lon', 'continent',
                 'gmp', 'population_locality', 'population_country',
                 'traffic_mortality', 'literacy_rate', 'avg_height',
                 'gini', 'traffic_index']

        Returns:
            None. Saves and displays the interactive map to disk.
        """
        cities = df_mapping["locality"]
        states = df_mapping["state"]
        countries = df_mapping["country"]
        coords_lat = df_mapping["lat"]
        coords_lon = df_mapping["lon"]

        # Create the country list to highlight in the choropleth map
        countries_set = set(countries)  # Use set to avoid duplicates
        # if "Denmark" in countries_set:
        #     countries_set.add('Greenland')
        if "Türkiye" in countries_set:
            countries_set.add('Turkey')
        if "Somalia" in countries_set:
            countries_set.add('Somaliland')

        # Create a DataFrame for highlighted countries with a value (same for all to have the same color)
        df = pd.DataFrame({'country': list(countries_set), 'value': 1})

        # Create a choropleth map using Plotly with grey color for countries
        fig = px.choropleth(df, locations="country", locationmode="country names",
                            color="value", hover_name="country", hover_data={'value': False, 'country': False},
                            color_continuous_scale=["rgb(242, 186, 78)", "rgb(242, 186, 78)"],
                            labels={'value': 'Highlighted'})

        # Update layout to remove Antarctica, Easter Island, remove the color bar, and set ocean color
        fig.update_layout(
            coloraxis_showscale=False,  # Remove color bar
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="black",  # Set coastline color
                showcountries=True,  # Show country borders
                countrycolor="black",  # Set border color
                projection_type='equirectangular',
                showlakes=True,
                lakecolor='rgb(173, 216, 230)',  # Light blue for lakes
                projection_scale=1,
                center=dict(lat=20, lon=0),  # Center map to remove Antarctica
                bgcolor='rgb(173, 216, 230)',  # Light blue for ocean
                resolution=50
            ),
            margin=dict(l=0, r=0, t=0, b=0),  # Remove the margins
            paper_bgcolor='rgb(173, 216, 230)'  # Set the paper background to match the ocean color
        )

        # Process each locality and its corresponding country
        locality_coords = []
        for i, (locality, state, lat, lon) in enumerate(tqdm(zip(cities, states, coords_lat, coords_lon), total=len(cities))):  # noqa: E501
            if not state or str(state).lower() == 'nan':
                state = 'N/A'
            if lat and lon:
                locality_coords.append({
                    'locality': locality,
                    'State': state,
                    'Country': df_mapping["country"].iloc[i],
                    'Continent': df_mapping["continent"].iloc[i],
                    'lat': lat,
                    'lon': lon,
                    'GDP (Billion USD)': df_mapping["gmp"].iloc[i],
                    'locality population (thousands)': df_mapping["population_locality"].iloc[i] / 1000.0,
                    'Country population (thousands)': df_mapping["population_country"].iloc[i] / 1000.0,
                    'Traffic mortality rate (per 100,000)': df_mapping["traffic_mortality"].iloc[i],
                    'Literacy rate': df_mapping["literacy_rate"].iloc[i],
                    'Average height (cm)': df_mapping["avg_height"].iloc[i],
                    'Gini coefficient': df_mapping["gini"].iloc[i],
                    'Traffic index': df_mapping["traffic_index"].iloc[i],
                })

        if locality_coords:
            locality_df = pd.DataFrame(locality_coords)
            # locality_df["locality"] = locality_df["locality"]  # Format locality name with "locality:"
            locality_trace = px.scatter_geo(
                locality_df, lat='lat', lon='lon',
                hover_data={
                    'locality': True,
                    'State': True,
                    'Country': True,
                    'Continent': True,
                    'GDP (Billion USD)': True,
                    'locality population (thousands)': True,
                    'Country population (thousands)': True,
                    'Traffic mortality rate (per 100,000)': True,
                    'Literacy rate': True,
                    'Average height (cm)': True,
                    'Gini coefficient': True,
                    'Traffic index': True,
                    'lat': False,
                    'lon': False  # Hide lat and lon
                }
            )
            # Update the locality markers to be red and adjust size
            locality_trace.update_traces(marker=dict(color="red", size=5))

            # Add the scatter_geo trace to the choropleth map
            fig.add_trace(locality_trace.data[0])

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # Save and display the figure
        io_class.save_plotly_figure(fig, "world_map", save_final=True)

    def map_world(self, df, *, color, title=None, projection="natural earth", hover_name="country", hover_data=None,
                  show_colorbar=False, colorbar_title=None, colorbar_kwargs=None, color_scale="YlOrRd",
                  show_cities=False, df_cities=None, locality_marker_size=3, show_images=False, image_items=None,
                  denmark_greenland=False, save_file=False, save_final=False, file_name="map",
                  filter_zero_nan=True, country_name_map=None):
        """
        Unified world choropleth with optional cities and annotations.

        Args:
            df (pd.DataFrame): Must include 'country' and the column referenced by `color`.
            color (str): Column in df for choropleth coloring.
            title (str|None): Optional figure title.
            projection (str): Plotly geo projection (e.g., 'natural earth').
            hover_name (str): Column used for hover name (defaults to 'country').
            hover_data (list|dict|None): Extra hover fields.
            show_colorbar (bool): Whether to show a color bar.
            colorbar_title (str|None): Title for color bar (defaults to `color`).
            colorbar_kwargs (dict|None): Extra layout props for the color bar (merged with sensible defaults).
            color_scale (str|list): Plotly color scale.
            show_cities (bool): Add locality markers from `df_cities`.
            df_cities (pd.DataFrame|None): Needs columns 'lat' and 'lon'; optional 'locality','country'.
            locality_marker_size (int|float): Marker size for cities.
            show_images (bool): Add overlay images/arrows/labels from `image_items`.
            image_items (list[dict]|None): Same schema you used before.
            denmark_greenland (bool): If True, duplicate Denmark’s value for Greenland.
            save_file (bool): If True, save HTML (and optionally final image) via your helper.
            save_final (bool): Passed to your `save_plotly_figure`.
            file_name (str): Base filename without extension.
            filter_zero_nan (bool): Filter rows where `color` is 0 or NaN (your old map() behavior).
            country_name_map (dict|None): Extra name normalization; default includes {'Türkiye': 'Turkey'}.
        """

        # --- prep ---
        df = df.copy()
        if country_name_map is None:
            country_name_map = {"Türkiye": "Turkey"}
        df["country"] = df["country"].replace(country_name_map)

        # Optional Greenland-from-Denmark duplication
        if denmark_greenland and "country" in df and color in df:
            denmark_vals = df.loc[df["country"] == "Denmark", color].values
            if len(denmark_vals) > 0:
                # build row with same columns
                greenland_row = {col: (None if col not in ("country", color) else
                                       ("Greenland" if col == "country" else denmark_vals[0]))
                                 for col in df.columns}
                df = pd.concat([df, pd.DataFrame([greenland_row])], ignore_index=True)

        # Optional filter like old map()
        if filter_zero_nan:
            df = df[df[color].fillna(0) != 0].copy()

        # Default colorbar title
        if colorbar_title is None:
            colorbar_title = color

        # --- figure ---
        fig = px.choropleth(
            df,
            locations="country",
            locationmode="country names",
            color=color,
            hover_name=hover_name,
            hover_data=hover_data,
            projection=projection,
            color_continuous_scale=color_scale,
            title=title,
        )

        # Fonts from your config
        fig.update_layout(
            font=dict(
                family=common.get_configs('font_family'),
                size=common.get_configs('font_size'),
            )
        )

        # --- optional cities ---
        if show_cities and df_cities is not None and {"lat", "lon"}.issubset(df_cities.columns):
            fig.add_trace(go.Scattergeo(
                lon=df_cities["lon"],
                lat=df_cities["lat"],
                text=df_cities.get("locality", None),
                mode="markers",
                hoverinfo="skip",
                marker=dict(size=locality_marker_size, color="black", opacity=0.7, symbol="circle"),
                name="cities",
            ))

        # --- optional image overlays/arrows/labels (your schema preserved) ---
        if show_images and image_items:
            # base path reused from your code
            path_screenshots = os.path.join(common.root_dir, "readme")

            # images and labels
            for item in image_items:
                if "file" in item:
                    fig.add_layout_image(dict(
                        source=os.path.join(path_screenshots, item["file"]),
                        xref="paper", yref="paper",
                        x=item.get("x", 0.5), y=item.get("y", 0.5),
                        sizex=item.get("sizex", 0.1), sizey=item.get("sizey", 0.1),
                        xanchor=item.get("xanchor", "center"),
                        yanchor=item.get("yanchor", "middle"),
                        layer=item.get("layer", "above"),
                    ))
                if "label" in item:
                    fig.add_annotation(
                        text=item["label"],
                        x=item.get("x_label", item.get("x", 0.5)),
                        y=item.get("y_label", item.get("y", 0.5) + 0.1),
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                    )

            # lines to actual locality coords (if df_cities available)
            if df_cities is not None and {"lat", "lon", "locality", "country"}.issubset(df_cities.columns):
                for item in image_items:
                    if "locality" in item and "country" in item and \
                       "approx_lon" in item and "approx_lat" in item:
                        row = df_cities[
                            (df_cities["locality"].str.lower() == item["locality"].lower()) &
                            (df_cities["country"].str.lower() == item["country"].lower())
                        ]
                        if not row.empty:
                            fig.add_trace(go.Scattergeo(
                                lon=[item["approx_lon"], row["lon"].values[0]],
                                lat=[item["approx_lat"], row["lat"].values[0]],
                                mode="lines",
                                line=dict(width=2, color="black"),
                                showlegend=False,
                                geo="geo",
                                hoverinfo="skip"
                            ))
                    # optional small “video code” tag
                    if "video" in item:
                        fig.add_annotation(dict(
                            text=item["video"],
                            x=item.get("x_video", item.get("x", 0.5)),
                            y=item.get("y_video", item.get("y", 0.5) - 0.1),
                            xref="paper", yref="paper",
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            align="center",
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="black",
                            borderwidth=1,
                        ))

        # --- colorbar handling (merged defaults + overrides) ---
        base_colorbar = dict(
            title=colorbar_title,
            orientation="h",
            x=0.5,
            y=0.06,
            xanchor="center",
            yanchor="bottom",
            len=0.5,
            thickness=10,
            bgcolor="rgba(255,255,255,0.7)",
            tickfont=dict(size=max(common.get_configs('font_size') - 5, 8)),
        )
        if colorbar_kwargs:
            # shallow merge, keys in colorbar_kwargs win
            base_colorbar.update(colorbar_kwargs)

        fig.update_coloraxes(
            showscale=show_colorbar,
            colorbar=(base_colorbar if show_colorbar else {})
        )

        # --- layout ---
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )

        # --- save/show ---
        if save_file:
            # tiny padding for saved version
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            # save via your utility
            io_class.save_plotly_figure(fig, file_name, save_final=save_final)
        else:
            fig.show()

        return fig
