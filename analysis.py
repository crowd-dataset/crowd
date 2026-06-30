"""
Project module setup.

Authors:
- Shadab Alam <md_shadab_alam@outlook.com>
- Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
"""

from __future__ import annotations

import ast
import math
import os
import pickle
import warnings
from typing import Set

import polars as pl

import common
from custom_logger import CustomLogger
from logmod import logs
from utils.analytics.metrics_cache import MetricsCache
from utils.core.dataset_stats import Dataset_Stats
from utils.plotting.bivariate import Bivariate
from utils.plotting.distributions import Distributions
from utils.plotting.maps import Maps

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

# Suppress a specific FutureWarning emitted by plotly.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# ---------------------------------------------------------------------
# Class instances (singletons for this module)
# ---------------------------------------------------------------------
maps = Maps()
bivariate = Bivariate()
distribution = Distributions()

dataset_stats = Dataset_Stats()
metrics_cache = MetricsCache()

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# File to store the locality coordinates.
file_results: str = "results.pickle"

video_paths = common.get_configs("videos")

# Common junk files/folders to ignore.
MISC_FILES: Set[str] = {"DS_Store", "seg", "bbox"}

# The detector/tracker pipeline trims each segment to t_end - 1 second.
# Keep analysis duration calculations aligned with that processed endpoint.
ENDPOINT_ADJUSTMENT_SECONDS = 1


def processed_segment_duration_seconds(start_time, end_time) -> int:
    """Return processed duration for one segment using t_end - 1 second.

    This mirrors main.py: if subtracting one second would make the adjusted
    endpoint less than or equal to the start time, the original endpoint is used.
    """
    try:
        start = int(float(start_time))
        end = int(float(end_time))
    except Exception:
        return 0

    processed_end = end - ENDPOINT_ADJUSTMENT_SECONDS
    if processed_end <= start:
        processed_end = end

    return max(0, processed_end - start)


class Analysis():

    def __init__(self) -> None:
        pass

    # Emoji flag mapping for ISO3 codes
    iso3_to_flag = {
        'ABW': '🇦🇼',  # Aruba
        'AFG': '🇦🇫',  # Afghanistan
        'AGO': '🇦🇴',  # Angola
        'AIA': '🇦🇮',  # Anguilla
        'ALA': '🇦🇽',  # Åland Islands
        'ALB': '🇦🇱',  # Albania
        'AND': '🇦🇩',  # Andorra
        'ARE': '🇦🇪',  # United Arab Emirates
        'ARG': '🇦🇷',  # Argentina
        'ARM': '🇦🇲',  # Armenia
        'ASM': '🇦🇸',  # American Samoa
        'ATA': '🇦🇶',  # Antarctica
        'ATF': '🏳️',  # French Southern Territories (no Unicode flag)
        'ATG': '🇦🇬',  # Antigua and Barbuda
        'AUS': '🇦🇺',  # Australia
        'AUT': '🇦🇹',  # Austria
        'AZE': '🇦🇿',  # Azerbaijan
        'BDI': '🇧🇮',  # Burundi
        'BEL': '🇧🇪',  # Belgium
        'BEN': '🇧🇯',  # Benin
        'BES': '🇧🇶',  # Bonaire, Sint Eustatius and Saba
        'BFA': '🇧🇫',  # Burkina Faso
        'BGD': '🇧🇩',  # Bangladesh
        'BGR': '🇧🇬',  # Bulgaria
        'BHR': '🇧🇭',  # Bahrain
        'BHS': '🇧🇸',  # Bahamas
        'BIH': '🇧🇦',  # Bosnia and Herzegovina
        'BLM': '🇧🇱',  # Saint Barthélemy
        'BLR': '🇧🇾',  # Belarus
        'BLZ': '🇧🇿',  # Belize
        'BMU': '🇧🇲',  # Bermuda
        'BOL': '🇧🇴',  # Bolivia
        'BRA': '🇧🇷',  # Brazil
        'BRB': '🇧🇧',  # Barbados
        'BRN': '🇧🇳',  # Brunei
        'BTN': '🇧🇹',  # Bhutan
        'BVT': '🏳️',  # Bouvet Island (no Unicode flag)
        'BWA': '🇧🇼',  # Botswana
        'CAF': '🇨🇫',  # Central African Republic
        'CAN': '🇨🇦',  # Canada
        'CCK': '🇨🇨',  # Cocos (Keeling) Islands
        'CHE': '🇨🇭',  # Switzerland
        'CHL': '🇨🇱',  # Chile
        'CHN': '🇨🇳',  # China
        'CIV': '🇨🇮',  # Côte d'Ivoire
        'CMR': '🇨🇲',  # Cameroon
        'COD': '🇨🇩',  # DR Congo
        'COG': '🇨🇬',  # Congo
        'COK': '🇨🇰',  # Cook Islands
        'COL': '🇨🇴',  # Colombia
        'COM': '🇰🇲',  # Comoros
        'CPV': '🇨🇻',  # Cape Verde
        'CRI': '🇨🇷',  # Costa Rica
        'CUB': '🇨🇺',  # Cuba
        'CUW': '🇨🇼',  # Curaçao
        'CXR': '🇨🇽',  # Christmas Island
        'CYM': '🇰🇾',  # Cayman Islands
        'CYP': '🇨🇾',  # Cyprus
        'CZE': '🇨🇿',  # Czechia
        'DEU': '🇩🇪',  # Germany
        'DJI': '🇩🇯',  # Djibouti
        'DMA': '🇩🇲',  # Dominica
        'DNK': '🇩🇰',  # Denmark
        'DOM': '🇩🇴',  # Dominican Republic
        'DZA': '🇩🇿',  # Algeria
        'ECU': '🇪🇨',  # Ecuador
        'EGY': '🇪🇬',  # Egypt
        'ERI': '🇪🇷',  # Eritrea
        'ESH': '🇪🇭',  # Western Sahara
        'ESP': '🇪🇸',  # Spain
        'EST': '🇪🇪',  # Estonia
        'ETH': '🇪🇹',  # Ethiopia
        'FIN': '🇫🇮',  # Finland
        'FJI': '🇫🇯',  # Fiji
        'FLK': '🇫🇰',  # Falkland Islands
        'FRA': '🇫🇷',  # France
        'FRO': '🇫🇴',  # Faroe Islands
        'FSM': '🇫🇲',  # Micronesia
        'GAB': '🇬🇦',  # Gabon
        'GBR': '🇬🇧',  # United Kingdom
        'GEO': '🇬🇪',  # Georgia
        'GGY': '🇬🇬',  # Guernsey
        'GHA': '🇬🇭',  # Ghana
        'GIB': '🇬🇮',  # Gibraltar
        'GIN': '🇬🇳',  # Guinea
        'GLP': '🇬🇵',  # Guadeloupe
        'GMB': '🇬🇲',  # Gambia
        'GNB': '🇬🇼',  # Guinea-Bissau
        'GNQ': '🇬🇶',  # Equatorial Guinea
        'GRC': '🇬🇷',  # Greece
        'GRD': '🇬🇩',  # Grenada
        'GRL': '🇬🇱',  # Greenland
        'GTM': '🇬🇹',  # Guatemala
        'GUF': '🇬🇫',  # French Guiana
        'GUM': '🇬🇺',  # Guam
        'GUY': '🇬🇾',  # Guyana
        'HKG': '🇭🇰',  # Hong Kong
        'HMD': '🇭🇲',  # Heard Island and McDonald Islands
        'HND': '🇭🇳',  # Honduras
        'HRV': '🇭🇷',  # Croatia
        'HTI': '🇭🇹',  # Haiti
        'HUN': '🇭🇺',  # Hungary
        'IDN': '🇮🇩',  # Indonesia
        'IMN': '🇮🇲',  # Isle of Man
        'IND': '🇮🇳',  # India
        'IOT': '🇮🇴',  # British Indian Ocean Territory
        'IRL': '🇮🇪',  # Ireland
        'IRN': '🇮🇷',  # Iran
        'IRQ': '🇮🇶',  # Iraq
        'ISL': '🇮🇸',  # Iceland
        'ISR': '🇮🇱',  # Israel
        'ITA': '🇮🇹',  # Italy
        'JAM': '🇯🇲',  # Jamaica
        'JEY': '🇯🇪',  # Jersey
        'JOR': '🇯🇴',  # Jordan
        'JPN': '🇯🇵',  # Japan
        'KAZ': '🇰🇿',  # Kazakhstan
        'KEN': '🇰🇪',  # Kenya
        'KGZ': '🇰🇬',  # Kyrgyzstan
        'KHM': '🇰🇭',  # Cambodia
        'KIR': '🇰🇮',  # Kiribati
        'KNA': '🇰🇳',  # Saint Kitts and Nevis
        'KOR': '🇰🇷',  # South Korea
        'KWT': '🇰🇼',  # Kuwait
        'LAO': '🇱🇦',  # Laos
        'LBN': '🇱🇧',  # Lebanon
        'LBR': '🇱🇷',  # Liberia
        'LBY': '🇱🇾',  # Libya
        'LCA': '🇱🇨',  # Saint Lucia
        'LIE': '🇱🇮',  # Liechtenstein
        'LKA': '🇱🇰',  # Sri Lanka
        'LSO': '🇱🇸',  # Lesotho
        'LTU': '🇱🇹',  # Lithuania
        'LUX': '🇱🇺',  # Luxembourg
        'LVA': '🇱🇻',  # Latvia
        'MAC': '🇲🇴',  # Macao
        'MAF': '🇲🇫',  # Saint Martin
        'MAR': '🇲🇦',  # Morocco
        'MCO': '🇲🇨',  # Monaco
        'MDA': '🇲🇩',  # Moldova
        'MDG': '🇲🇬',  # Madagascar
        'MDV': '🇲🇻',  # Maldives
        'MEX': '🇲🇽',  # Mexico
        'MHL': '🇲🇭',  # Marshall Islands
        'MKD': '🇲🇰',  # North Macedonia
        'MLI': '🇲🇱',  # Mali
        'MLT': '🇲🇹',  # Malta
        'MMR': '🇲🇲',  # Myanmar
        'MNE': '🇲🇪',  # Montenegro
        'MNG': '🇲🇳',  # Mongolia
        'MNP': '🇲🇵',  # Northern Mariana Islands
        'MOZ': '🇲🇿',  # Mozambique
        'MRT': '🇲🇷',  # Mauritania
        'MSR': '🇲🇸',  # Montserrat
        'MTQ': '🇲🇶',  # Martinique
        'MUS': '🇲🇺',  # Mauritius
        'MWI': '🇲🇼',  # Malawi
        'MYS': '🇲🇾',  # Malaysia
        'MYT': '🇾🇹',  # Mayotte
        'NAM': '🇳🇦',  # Namibia
        'NCL': '🇳🇨',  # New Caledonia
        'NER': '🇳🇪',  # Niger
        'NFK': '🇳🇫',  # Norfolk Island
        'NGA': '🇳🇬',  # Nigeria
        'NIC': '🇳🇮',  # Nicaragua
        'NIU': '🇳🇺',  # Niue
        'NLD': '🇳🇱',  # Netherlands
        'NOR': '🇳🇴',  # Norway
        'NPL': '🇳🇵',  # Nepal
        'NRU': '🇳🇷',  # Nauru
        'NZL': '🇳🇿',  # New Zealand
        'OMN': '🇴🇲',  # Oman
        'PAK': '🇵🇰',  # Pakistan
        'PAN': '🇵🇦',  # Panama
        'PCN': '🇵🇳',  # Pitcairn Islands
        'PER': '🇵🇪',  # Peru
        'PHL': '🇵🇭',  # Philippines
        'PLW': '🇵🇼',  # Palau
        'PNG': '🇵🇬',  # Papua New Guinea
        'POL': '🇵🇱',  # Poland
        'PRI': '🇵🇷',  # Puerto Rico
        'PRK': '🇰🇵',  # North Korea
        'PRT': '🇵🇹',  # Portugal
        'PRY': '🇵🇾',  # Paraguay
        'PSE': '🇵🇸',  # Palestine
        'PYF': '🇵🇫',  # French Polynesia
        'QAT': '🇶🇦',  # Qatar
        'REU': '🇷🇪',  # Réunion
        'ROU': '🇷🇴',  # Romania
        'RUS': '🇷🇺',  # Russia
        'RWA': '🇷🇼',  # Rwanda
        'SAU': '🇸🇦',  # Saudi Arabia
        'SDN': '🇸🇩',  # Sudan
        'SEN': '🇸🇳',  # Senegal
        'SGP': '🇸🇬',  # Singapore
        'SGS': '🏳️',  # South Georgia & South Sandwich Islands (no Unicode flag)
        'SHN': '🇸🇭',  # Saint Helena
        'SJM': '🏳️',  # Svalbard and Jan Mayen (no Unicode flag)
        'SLB': '🇸🇧',  # Solomon Islands
        'SLE': '🇸🇱',  # Sierra Leone
        'SLV': '🇸🇻',  # El Salvador
        'SMR': '🇸🇲',  # San Marino
        'SOM': '🇸🇴',  # Somalia
        'SPM': '🇵🇲',  # Saint Pierre and Miquelon
        'SRB': '🇷🇸',  # Serbia
        'SSD': '🇸🇸',  # South Sudan
        'STP': '🇸🇹',  # São Tomé and Príncipe
        'SUR': '🇸🇷',  # Suriname
        'SVK': '🇸🇰',  # Slovakia
        'SVN': '🇸🇮',  # Slovenia
        'SWE': '🇸🇪',  # Sweden
        'SWZ': '🇸🇿',  # Eswatini
        'SXM': '🇸🇽',  # Sint Maarten
        'SYC': '🇸🇨',  # Seychelles
        'SYR': '🇸🇾',  # Syria
        'TCA': '🇹🇨',  # Turks and Caicos Islands
        'TCD': '🇹🇩',  # Chad
        'TGO': '🇹🇬',  # Togo
        'THA': '🇹🇭',  # Thailand
        'TJK': '🇹🇯',  # Tajikistan
        'TKL': '🇹🇰',  # Tokelau
        'TKM': '🇹🇲',  # Turkmenistan
        'TLS': '🇹🇱',  # Timor-Leste
        'TON': '🇹🇴',  # Tonga
        'TTO': '🇹🇹',  # Trinidad and Tobago
        'TUN': '🇹🇳',  # Tunisia
        'TUR': '🇹🇷',  # Turkey
        'TUV': '🇹🇻',  # Tuvalu
        'TWN': '🇹🇼',  # Taiwan
        'TZA': '🇹🇿',  # Tanzania
        'UGA': '🇺🇬',  # Uganda
        'UKR': '🇺🇦',  # Ukraine
        'UMI': '🇺🇲',  # U.S. Minor Outlying Islands
        'URY': '🇺🇾',  # Uruguay
        'USA': '🇺🇸',  # United States
        'UZB': '🇺🇿',  # Uzbekistan
        'VAT': '🇻🇦',  # Vatican City
        'VCT': '🇻🇨',  # Saint Vincent and the Grenadines
        'VEN': '🇻🇪',  # Venezuela
        'VGB': '🇻🇬',  # British Virgin Islands
        'VIR': '🇻🇮',  # U.S. Virgin Islands
        'VNM': '🇻🇳',  # Vietnam
        'VUT': '🇻🇺',  # Vanuatu
        'WLF': '🇼🇫',  # Wallis and Futuna
        'WSM': '🇼🇸',  # Samoa
        'XKX': '🇽🇰',  # Kosovo
        'YEM': '🇾🇪',  # Yemen
        'ZAF': '🇿🇦',  # South Africa
        'ZMB': '🇿🇲',  # Zambia
        'ZWE': '🇿🇼',  # Zimbabwe
    }

    # vehicle type mapping
    vehicle_map = {
        0: "Car",
        1: "Bus",
        2: "Truck",
        3: "Two-wheeler",
        4: "Bicycle",
        5: "Automated car",
        6: "Electric scooter",
        7: "Monowheel/unicycle",
        8: "Emergency vehicle",
        9: "Automated bus",
        10: "Automated truck",
        11: "Automated two-wheeler",
        12: "Non-electric scooter",
        13: "Pedestrian"
    }

    # time of day mapping
    time_map = {0: "Day", 1: "Night"}


analysis_class = Analysis()

# ---------------------------------------------------------------------
# Extra rollups / summaries (continent, day-night, vehicle, max/min)
# ---------------------------------------------------------------------


def log_rollups(df_mapping: "pl.DataFrame") -> None:
    """
    Rollups + logging (paper-aligned) with COUNT-CONSISTENCY fixes.

    Key fixes vs your previous version
    ----------------------------------
    1) "Segment records" now means *segment-level label entries* (time_of_day entries),
       consistent with the paper definition:
          - time_of_day entry [0,1] contributes 2 segment records (Day + Night)
       This makes:
          - Total segment_records == Total time-of-day label entries (the "total_entries" table)
       rather than counting only videos.

    2) Unique uploads by continent uses a *canonical continent per upload* to avoid double-counting,
       so the "Total" unique uploads equals the global unique upload count:
          - pick the most frequent continent among mapped rows for that upload
          - if tied, pick the continent with the greatest total mapped duration (seconds)

    Assumptions / dependencies
    --------------------------
    - Requires: `import polars as pl`, `import ast`, and a `logger`.
    - Uses `analysis_class.vehicle_map` and `analysis_class.time_map` if present (same as your code).
      If you don't have `analysis_class`, replace those with your own dicts (vehicle_map/time_map).
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    def _df_full_str(
        df: "pl.DataFrame",
        *,
        rows: int = 10_000,
        cols: int = 1_000,
        width_chars: int = 5_000,
        str_len: int = 5_000,
    ) -> str:
        """Render a Polars DataFrame as a FULL string for logging (no `...` truncation)."""
        with pl.Config(
            tbl_rows=rows,
            tbl_cols=cols,
            tbl_width_chars=width_chars,
            fmt_str_lengths=str_len,
        ):
            return df.__repr__()

    def _ensure_zero_cols(df: "pl.DataFrame", cols: list[str], dtype: "pl.DataType") -> "pl.DataFrame":
        """Add missing columns with zeros (useful after pivots when a category is absent)."""
        for c in cols:
            if c not in df.columns:
                df = df.with_columns(pl.lit(0).cast(dtype).alias(c))
        return df

    def _vstack_like(df_base: "pl.DataFrame", df_to_add: "pl.DataFrame") -> "pl.DataFrame":
        """
        Concatenate vertically, aligning df_to_add to df_base schema:
          - add missing columns as null
          - drop extra columns
          - reorder columns to match df_base
          - cast to df_base dtypes (strict=False)
        Prevents Polars schema/order mismatches.
        """
        base_cols = df_base.columns
        base_schema = df_base.schema

        for c in base_cols:
            if c not in df_to_add.columns:
                df_to_add = df_to_add.with_columns(pl.lit(None).alias(c))

        df_to_add = df_to_add.select(base_cols)

        cast_exprs = []
        for c in base_cols:
            target_dtype = base_schema.get(c)
            if target_dtype is not None:
                cast_exprs.append(pl.col(c).cast(target_dtype, strict=False).alias(c))
        if cast_exprs:
            df_to_add = df_to_add.with_columns(cast_exprs)

        return pl.concat([df_base, df_to_add], how="vertical_relaxed")

    CONTINENT_ORDER = ["Europe", "Asia", "North America", "Africa", "Oceania", "South America", "Total"]
    _continent_order_map = {name: i for i, name in enumerate(CONTINENT_ORDER)}

    def _sort_continent(df: "pl.DataFrame") -> "pl.DataFrame":
        if "continent" not in df.columns:
            return df
        return (
            df.with_columns(
                pl.col("continent")
                .map_elements(lambda x: _continent_order_map.get(str(x), 999), return_dtype=pl.Int64)
                .alias("__k")
            )
            .sort("__k")
            .drop("__k")
        )

    def _pct_2dp_str(expr: "pl.Expr") -> "pl.Expr":
        return expr.map_elements(lambda x: f"{float(x):.2f}", return_dtype=pl.Utf8)

    def _count_pct_fmt(count_expr: "pl.Expr", pct_expr: "pl.Expr") -> "pl.Expr":
        """Format as '<count> (<pct>%)' with pct at 2 decimals."""
        return pl.concat_str(
            [
                count_expr.cast(pl.Int64).cast(pl.Utf8),
                pl.lit(" ("),
                _pct_2dp_str(pct_expr.round(2)),
                pl.lit("%)"),
            ]
        )

    def _safe_eval_list(x):
        """
        Parse list-like cells robustly:
          - None -> []
          - list -> list
          - string -> ast.literal_eval when possible, else best-effort fallback
        """
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if not isinstance(x, str):
            return []

        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return []

        # strip wrapping quotes if the entire cell is quoted
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in {"'", '"'}):
            s = s[1:-1].strip()

        if s == "" or s == "[]":
            return []

        # Try python literal eval first (works for numeric nested lists)
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            # Fallback: if looks like [a,b,c] with bare tokens (common for video ids)
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                if inner == "":
                    return []
                parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
                return [p for p in parts if p != ""]
            return []

    def _safe_num_to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            try:
                return int(float(str(v).strip()))
            except Exception:
                return None

    def _to_list_of_ints(x) -> list[int]:
        """Scalar/list -> list[int] (dropping unparseables)."""
        if x is None:
            return []
        if isinstance(x, list):
            out = []
            for t in x:
                iv = _safe_num_to_int(t)
                if iv is not None:
                    out.append(iv)
            return out
        iv = _safe_num_to_int(x)
        return [iv] if iv is not None else []

    def _norm_list_of_lists_int(x) -> list[list[int]]:
        """
        Ensure time_of_day / similar becomes list[list[int]]:
          - [[0],[0,1]] stays same (coerced to ints)
          - [0,1] becomes [[0],[1]]
          - [] stays []
        """
        lst = _safe_eval_list(x)
        out: list[list[int]] = []
        for item in lst:
            if isinstance(item, list):
                out.append(_to_list_of_ints(item))
            else:
                out.append(_to_list_of_ints(item))
        # Drop empties at inner level (but keep structure mostly clean)
        return [a for a in out if len(a) > 0]

    def _norm_list_of_lists_float(x) -> list[list[float]]:
        """Ensure start_time/end_time becomes list[list[float]] (best effort)."""
        lst = _safe_eval_list(x)
        out: list[list[float]] = []
        for item in lst:
            if isinstance(item, list):
                arr = []
                for t in item:
                    try:
                        fv = float(t)
                        if math.isfinite(fv):
                            arr.append(fv)
                    except Exception:
                        continue
                if arr:
                    out.append(arr)
            else:
                try:
                    fv = float(item)
                    if math.isfinite(fv):
                        out.append([fv])
                except Exception:
                    continue
        return out

    def _count_label_entries(tods_lol: list[list[int]]) -> int:
        """Segment record count = number of time_of_day label entries (sum of inner lengths)."""
        if not isinstance(tods_lol, list):
            return 0
        total = 0
        for item in tods_lol:
            if isinstance(item, list):
                total += len(item)
            else:
                total += 1
        return int(total)

    def _count_segments_from_start_end(st_lol: list[list[float]], en_lol: list[list[float]]) -> int:
        """Count segments as total zipped pairs across per-video lists."""
        if not (isinstance(st_lol, list) and isinstance(en_lol, list)):
            return 0
        c = 0
        for st_i, en_i in zip(st_lol, en_lol):
            st_list = st_i if isinstance(st_i, list) else [st_i]
            en_list = en_i if isinstance(en_i, list) else [en_i]
            c += min(len(st_list), len(en_list))
        return int(c)

    def _compute_footage_time_s_from_lol(st_lol: list[list[float]], en_lol: list[list[float]]) -> int:
        """Sum processed durations across per-video aligned start/end lists."""
        if not (isinstance(st_lol, list) and isinstance(en_lol, list)):
            return 0
        total = 0
        for st_i, en_i in zip(st_lol, en_lol):
            st_list = st_i if isinstance(st_i, list) else [st_i]
            en_list = en_i if isinstance(en_i, list) else [en_i]
            for start, end in zip(st_list, en_list):
                total += processed_segment_duration_seconds(start, end)
        return int(total)

    def _map_with_fallback(dct: dict, v):
        if v is None:
            return None
        if v in dct:
            return dct[v]
        sv = v.strip() if isinstance(v, str) else str(v).strip()
        if sv in dct:
            return dct[sv]
        try:
            iv = int(sv)
            if iv in dct:
                return dct[iv]
        except Exception:
            try:
                iv = int(float(sv))
                if iv in dct:
                    return dct[iv]
            except Exception:
                pass
        try:
            fv = float(sv)
            if fv in dct:
                return dct[fv]
        except Exception:
            pass
        return None

    # -----------------------------
    # Required columns check
    # -----------------------------
    required_cols = {"continent", "country", "locality", "iso3", "videos", "start_time", "end_time"}
    missing_cols = sorted(list(required_cols - set(df_mapping.columns)))
    if missing_cols:
        logger.warning(f"[rollups] Missing columns in df_mapping: {missing_cols}. Some rollups may be incomplete.")

    # -----------------------------
    # Parse videos robustly (string-split fallback)
    # -----------------------------
    videos_raw = pl.col("videos").cast(pl.Utf8).fill_null("").str.strip_chars().str.strip_chars("\"'")
    is_list = videos_raw.str.starts_with("[") & videos_raw.str.ends_with("]")

    videos_inner = (
        pl.when(is_list)
        .then(videos_raw.str.strip_chars("[]"))
        .otherwise(videos_raw)
        .fill_null("")
        .str.replace_all(r"[\"']", "")
        .str.strip_chars()
    )

    videos_list_expr = (
        pl.when(videos_inner == "")
        .then(pl.lit([]).cast(pl.List(pl.Utf8)))
        .otherwise(
            videos_inner
            .str.split(",")
            .list.eval(pl.element().str.strip_chars())
            .list.filter(pl.element() != "")
        )
        .alias("videos_list")
    )

    # -----------------------------
    # Parse time_of_day / start_time / end_time into list-of-lists
    # -----------------------------
    tod_lol_dtype = pl.List(pl.List(pl.Int64))
    se_lol_dtype = pl.List(pl.List(pl.Float64))

    has_tod = "time_of_day" in df_mapping.columns
    has_st = "start_time" in df_mapping.columns
    has_en = "end_time" in df_mapping.columns

    df_base = df_mapping.with_columns([videos_list_expr])

    if has_tod:
        df_base = df_base.with_columns(
            pl.col("time_of_day")
            .map_elements(_norm_list_of_lists_int, return_dtype=tod_lol_dtype)
            .alias("time_of_day_lol")
        )
    else:
        df_base = df_base.with_columns(pl.lit([]).cast(tod_lol_dtype).alias("time_of_day_lol"))

    if has_st:
        df_base = df_base.with_columns(
            pl.col("start_time")
            .map_elements(_norm_list_of_lists_float, return_dtype=se_lol_dtype)
            .alias("start_time_lol")
        )
    else:
        df_base = df_base.with_columns(pl.lit([]).cast(se_lol_dtype).alias("start_time_lol"))

    if has_en:
        df_base = df_base.with_columns(
            pl.col("end_time")
            .map_elements(_norm_list_of_lists_float, return_dtype=se_lol_dtype)
            .alias("end_time_lol")
        )
    else:
        df_base = df_base.with_columns(pl.lit([]).cast(se_lol_dtype).alias("end_time_lol"))

    # -----------------------------
    # Compute: upload_records, segment_records (COUNT-CONSISTENT), duration
    # -----------------------------
    df_base = df_base.with_columns(
        pl.col("videos_list").list.len().fill_null(0).cast(pl.Int64).alias("upload_records")
    )

    # segment_records = label-entry count from time_of_day_lol; fallback to start/end count; fallback to uploads
    df_base = df_base.with_columns(
        pl.col("time_of_day_lol")
        .map_elements(_count_label_entries, return_dtype=pl.Int64)
        .alias("segment_records_tod")
    ).with_columns(
        pl.struct(["start_time_lol", "end_time_lol"])
        .map_elements(lambda r: _count_segments_from_start_end(r["start_time_lol"],
                                                               r["end_time_lol"]), return_dtype=pl.Int64)
        .alias("segment_records_se")
    ).with_columns(
        pl.when(pl.col("segment_records_tod") > 0)
        .then(pl.col("segment_records_tod"))
        .when(pl.col("segment_records_se") > 0)
        .then(pl.col("segment_records_se"))
        .otherwise(pl.col("upload_records"))
        .alias("segment_records")
    )

    df_base = df_base.with_columns(
        pl.struct(["start_time_lol", "end_time_lol"])
        .map_elements(lambda r: _compute_footage_time_s_from_lol(r["start_time_lol"],
                                                                 r["end_time_lol"]), return_dtype=pl.Int64)
        .alias("footage_time_s")
    ).with_columns((pl.col("footage_time_s") / 3600).round(2).alias("footage_time_h"))

    # -----------------------------
    # High-level consistency diagnostics
    # -----------------------------
    total_rows = df_base.height

    # top-level alignment: videos vs time_of_day (per-video list length)
    # NOTE: time_of_day_lol is list-of-lists; its length should equal upload_records ideally.
    df_diag = df_base.select(
        [
            pl.col("upload_records"),
            pl.col("time_of_day_lol").list.len().cast(pl.Int64).alias("tod_video_slots"),
            pl.col("segment_records_tod"),
            pl.col("segment_records_se"),
        ]
    )

    n_mismatch_slots = int(
        df_diag.filter(pl.col("upload_records") != pl.col("tod_video_slots")).height
    )
    n_mismatch_seg = int(
        df_diag.filter(
            (pl.col("segment_records_tod") > 0)
            & (pl.col("segment_records_se") > 0)
            & (pl.col("segment_records_tod") != pl.col("segment_records_se"))
        ).height
    )

    total_upload_records = int(df_base.select(pl.sum("upload_records")).item() or 0)
    total_segment_records = int(df_base.select(pl.sum("segment_records")).item() or 0)
    total_dur_s = int(df_base.select(pl.sum("footage_time_s")).item() or 0)

    logger.info("\n=== [rollups] Dataset summary ===")
    logger.info(
        f"rows={total_rows} | upload_records(sum videos per row)={total_upload_records}",
    )
    logger.info(
        f"segment_records(label-entries)={total_segment_records} | duration_h={round(total_dur_s / 3600, 2)}",
    )
    if n_mismatch_slots > 0:
        logger.warning(
            f"[rollups] {n_mismatch_slots}/{total_rows} rows have len(videos_list) != len(time_of_day_lol). "
            "Those rows may have parsing/alignment issues (zips will truncate).",
        )
    if n_mismatch_seg > 0:
        logger.warning(
            f"[rollups] {n_mismatch_seg}/{total_rows} rows have segment_records_tod != segment_records_se (both present)."  # noqa: E501
            "Using time_of_day-derived counts for segment_records to match paper definition.",
        )

    # =========================================================================
    # A) Continent segment distribution (segment label-entries, shares, duration) + Total
    # =========================================================================
    cont = (
        df_base
        .filter(pl.col("continent").is_not_null() & (pl.col("continent") != ""))
        .group_by("continent")
        .agg(
            [
                pl.sum("segment_records").alias("segment_records"),
                pl.sum("footage_time_s").alias("duration_s"),
            ]
        )
    )

    tot_records = int(cont.select(pl.sum("segment_records")).item() or 0)
    tot_dur_s2 = int(cont.select(pl.sum("duration_s")).item() or 0)
    denom_r = max(tot_records, 1)
    denom_d = max(tot_dur_s2, 1)

    cont = cont.with_columns(
        [
            (pl.col("segment_records") / pl.lit(denom_r) * 100).round(2).alias("segment_share_pct"),
            (pl.col("duration_s") / pl.lit(denom_d) * 100).round(2).alias("duration_share_pct"),
            (pl.col("duration_s") / 3600).round(2).alias("duration_h"),
        ]
    )

    cont_total = pl.DataFrame(
        {
            "continent": ["Total"],
            "segment_records": [tot_records],
            "duration_s": [tot_dur_s2],
            "segment_share_pct": [100.0],
            "duration_share_pct": [100.0],
            "duration_h": [round(tot_dur_s2 / 3600, 2)],
        }
    )
    cont_table = _sort_continent(_vstack_like(cont, cont_total))

    logger.info("\n=== [rollups] A) Continent segment distribution (label-entries; paper) ===")
    logger.info(f"\n{_df_full_str(cont_table)}")

    # =========================================================================
    # Build a canonical (video_id -> continent) mapping to avoid double counting
    #
    # Note:
    # - Segment and duration totals in (A) are computed by summing mapping rows within each continent,
    #   so uploads mapped to multiple continents contribute to every continent to which they are mapped.
    # - For counts of unique uploads by continent, each upload is assigned to a single continent:
    #     1) most frequent continent among its mapped rows
    #     2) if tied, continent with greatest total mapped duration (seconds) for that upload
    # =========================================================================

    vid_cont_row_dtype = pl.List(
        pl.Struct(
            [
                pl.Field("video_id", pl.Utf8),
                pl.Field("continent", pl.Utf8),
                pl.Field("duration_s", pl.Int64),
            ]
        )
    )

    def _expand_vid_cont_dur(row: dict) -> list[dict]:
        contv = "" if row.get("continent") is None else str(row.get("continent"))
        vids = row.get("videos_list") or []
        st_lol = row.get("start_time_lol") or []
        en_lol = row.get("end_time_lol") or []

        out: list[dict] = []
        for i, vid in enumerate(vids):
            if vid is None:
                continue
            vid_s = str(vid).strip()
            if vid_s == "":
                continue

            dur = 0.0
            if i < len(st_lol) and i < len(en_lol):
                st_i = st_lol[i] if isinstance(st_lol[i], list) else [st_lol[i]]
                en_i = en_lol[i] if isinstance(en_lol[i], list) else [en_lol[i]]
                for start, end in zip(st_i, en_i):
                    dur += processed_segment_duration_seconds(start, end)

            out.append({"video_id": vid_s, "continent": contv, "duration_s": int(dur)})

        return out

    df_vid_rows = (
        df_base
        .select(["continent", "videos_list", "start_time_lol", "end_time_lol"])
        .filter(pl.col("continent").is_not_null() & (pl.col("continent") != ""))
        .with_columns(
            pl.struct(["continent", "videos_list", "start_time_lol", "end_time_lol"])
            .map_elements(_expand_vid_cont_dur, return_dtype=vid_cont_row_dtype)
            .alias("vid_rows")
        )
        .explode("vid_rows")
        .with_columns(
            [
                pl.col("vid_rows").struct.field("video_id").alias("video_id"),
                pl.col("vid_rows").struct.field("continent").alias("continent"),
                pl.col("vid_rows").struct.field("duration_s").alias("duration_s"),
            ]
        )
        .drop(["vid_rows", "videos_list", "start_time_lol", "end_time_lol"])
        .filter(pl.col("video_id").is_not_null() & (pl.col("video_id") != ""))
        .filter(pl.col("continent").is_not_null() & (pl.col("continent") != ""))
    )

    # For each upload and continent: row frequency + total mapped duration
    df_vid_per_cont = (
        df_vid_rows
        .group_by(["video_id", "continent"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.sum("duration_s").alias("duration_s"),
            ]
        )
        .fill_null(0)
    )

    # Canonical continent selection:
    # - max n_rows (most frequent)
    # - tie break by max duration_s
    # - final deterministic tie break by continent name
    df_vid_canon_full = (
        df_vid_per_cont
        .sort(["video_id", "n_rows", "duration_s", "continent"], descending=[False, True, True, False])
        .group_by("video_id", maintain_order=True)
        .agg(
            [
                pl.col("continent").first().alias("continent"),
                pl.col("continent").n_unique().alias("n_continents"),
            ]
        )
    )

    n_multi_cont = int(df_vid_canon_full.filter(pl.col("n_continents") > 1).height)
    df_vid_canon = df_vid_canon_full.select(["video_id", "continent"])

    total_unique_global = int(df_vid_canon.select(pl.col("video_id").n_unique()).item() or 0)

    if n_multi_cont > 0:
        logger.warning(
            f"[rollups] {n_multi_cont} uploads appear in more than one continent in the mapping. "
            "Segment and duration rollups sum over mapping rows, so those uploads contribute to each mapped continent."  # noqa: E501
            "Unique upload counts assign each upload to one continent using most frequent mapping (ties by mapped duration).",  # noqa: E501
        )

    # =========================================================================
    # B) Unique uploads by continent (+% over global unique uploads) + Total (GLOBAL)
    # =========================================================================
    cont_unique = (
        df_vid_canon
        .group_by("continent")
        .agg(pl.col("video_id").n_unique().alias("unique_uploads"))
    )
    denom_u = max(total_unique_global, 1)

    cont_unique = cont_unique.with_columns(
        (pl.col("unique_uploads") / pl.lit(denom_u) * 100).round(2).alias("unique_uploads_pct")
    )

    cont_unique_total = pl.DataFrame(
        {"continent": ["Total"], "unique_uploads": [total_unique_global], "unique_uploads_pct": [100.0]}
    )
    cont_unique_table = _sort_continent(_vstack_like(cont_unique, cont_unique_total))

    logger.info("\n=== [rollups] B) Continent unique uploads (canonical; % of GLOBAL unique uploads) ===")
    logger.info(f"\n{_df_full_str(cont_unique_table)}")

    # =========================================================================
    # C) Vehicle distribution by unique videos (+% over GLOBAL unique uploads)
    # =========================================================================
    if {"videos", "vehicle_type"} <= set(df_mapping.columns):

        video_vehicle_dtype = pl.List(
            pl.Struct(
                [
                    pl.Field("video_id", pl.Utf8),
                    pl.Field("vehicle_type", pl.Utf8),
                ]
            )
        )

        def _expand_video_vehicle(row: dict) -> list[dict]:
            try:
                vids = _safe_eval_list(row.get("videos"))
                vts = _safe_eval_list(row.get("vehicle_type"))
                if not (isinstance(vids, list) and isinstance(vts, list)):
                    return []
            except Exception:
                return []
            out = []
            for vid, vt in zip(vids, vts):
                vid_s = "" if vid is None else str(vid).strip()
                if vid_s == "":
                    continue
                vt_list = vt if isinstance(vt, list) else [vt]
                for v in vt_list:
                    out.append({"video_id": vid_s, "vehicle_type": str(v)})
            return out

        df_video_vehicle = (
            df_mapping
            .select(["videos", "vehicle_type"])
            .with_columns(
                pl.struct(["videos", "vehicle_type"])
                .map_elements(lambda r: _expand_video_vehicle(r), return_dtype=video_vehicle_dtype)
                .alias("pairs")
            )
            .explode("pairs")
            .with_columns(
                [
                    pl.col("pairs").struct.field("video_id").alias("video_id"),
                    pl.col("pairs").struct.field("vehicle_type").alias("vehicle_type_raw"),
                ]
            )
            .drop("pairs")
        )

        vehicle_map = getattr(analysis_class, "vehicle_map", {})

        df_video_vehicle = (
            df_video_vehicle
            .with_columns(
                pl.col("vehicle_type_raw")
                .map_elements(lambda x: _map_with_fallback(vehicle_map, x), return_dtype=pl.Utf8)
                .alias("vehicle_type_name")
            )
            .filter(
                pl.col("video_id").is_not_null()
                & (pl.col("video_id") != "")
                & pl.col("vehicle_type_name").is_not_null()
            )
            .unique(["video_id", "vehicle_type_name"])
        )

        veh_video = (
            df_video_vehicle
            .group_by("vehicle_type_name")
            .len()
            .rename({"len": "unique_videos"})
        )

        denom_v = max(total_unique_global, 1)

        veh_video = (
            veh_video
            .with_columns(
                (pl.col("unique_videos") / pl.lit(denom_v) * 100).round(2).alias("unique_videos_pct")
            )
            .sort(["vehicle_type_name"])
        )

        logger.info("=== [rollups] C) Vehicle distribution by unique videos (% of GLOBAL unique uploads) ===")
        logger.info(f"{_df_full_str(veh_video)}")

    # =========================================================================
    # C2) Vehicle x time-of-day (pair counts + %)
    # =========================================================================
    if {"vehicle_type", "time_of_day"} <= set(df_mapping.columns):

        pair_dtype = pl.List(
            pl.Struct(
                [
                    pl.Field("vehicle_type", pl.Utf8),
                    pl.Field("time_of_day", pl.Utf8),
                ]
            )
        )

        def _expand_vehicle_tod(row: dict) -> list[dict]:
            try:
                vts = _safe_eval_list(row.get("vehicle_type"))
                tods = _safe_eval_list(row.get("time_of_day"))
                if not (isinstance(vts, list) and isinstance(tods, list)):
                    return []
            except Exception:
                return []
            out = []
            for vt, tod in zip(vts, tods):
                vt_list = vt if isinstance(vt, list) else [vt]
                tod_list = tod if isinstance(tod, list) else [tod]
                for v in vt_list:
                    for t in tod_list:
                        out.append({"vehicle_type": str(v), "time_of_day": str(t)})
            return out

        df_pairs = (
            df_mapping
            .select(["vehicle_type", "time_of_day"])
            .with_columns(
                pl.struct(["vehicle_type", "time_of_day"])
                .map_elements(lambda r: _expand_vehicle_tod(r), return_dtype=pair_dtype)
                .alias("pairs")
            )
            .explode("pairs")
            .with_columns(
                [
                    pl.col("pairs").struct.field("vehicle_type").alias("vehicle_type_raw"),
                    pl.col("pairs").struct.field("time_of_day").alias("time_of_day_raw"),
                ]
            )
            .drop("pairs")
        )

        # maps (keep your existing objects)
        vehicle_map = getattr(analysis_class, "vehicle_map", {})
        time_map = getattr(analysis_class, "time_map", {})

        df_pairs = (
            df_pairs
            .with_columns(
                [
                    pl.col("vehicle_type_raw")
                    .map_elements(lambda x: _map_with_fallback(vehicle_map, x), return_dtype=pl.Utf8)
                    .alias("vehicle_type_name"),
                    pl.col("time_of_day_raw")
                    .map_elements(lambda x: _map_with_fallback(time_map, x), return_dtype=pl.Utf8)
                    .alias("time_of_day_name"),
                ]
            )
            .filter(pl.col("vehicle_type_name").is_not_null() & pl.col("time_of_day_name").is_not_null())
        )

        veh_tod = (
            df_pairs
            .group_by(["vehicle_type_name", "time_of_day_name"])
            .len()
            .rename({"len": "count"})
        )

        tot_pairs = int(veh_tod.select(pl.sum("count")).item() or 0)
        denom_p = max(tot_pairs, 1)

        veh_tod = (
            veh_tod
            .with_columns((pl.col("count") / pl.lit(denom_p) * 100).round(2).alias("pct"))
            .sort(["vehicle_type_name", "time_of_day_name"])
        )

        logger.info("=== [rollups] C2) Vehicle x time-of-day (pair counts + %) ===")
        logger.info(f"{_df_full_str(veh_tod)}")

    # =========================================================================
    # D) Continent x time-of-day LABEL-ENTRY distribution (Day/Night entries + % within continent) + Total
    # =========================================================================
    if {"continent", "time_of_day"} <= set(df_mapping.columns):

        cont_pair_dtype = pl.List(
            pl.Struct(
                [
                    pl.Field("continent", pl.Utf8),
                    pl.Field("time_of_day", pl.Utf8),
                ]
            )
        )

        def _expand_continent_tod(row: dict) -> list[dict]:
            contv = "" if row.get("continent") is None else str(row.get("continent"))
            tods = _safe_eval_list(row.get("time_of_day"))
            out = []
            for tod in tods:
                tod_list = tod if isinstance(tod, list) else [tod]
                for t in tod_list:
                    out.append({"continent": contv, "time_of_day": str(t)})
            return out

        df_ct = (
            df_mapping
            .select(["continent", "time_of_day"])
            .with_columns(
                pl.struct(["continent", "time_of_day"])
                .map_elements(lambda r: _expand_continent_tod(r), return_dtype=cont_pair_dtype)
                .alias("pairs")
            )
            .explode("pairs")
            .with_columns(
                [
                    pl.col("pairs").struct.field("continent").alias("continent"),
                    pl.col("pairs").struct.field("time_of_day").alias("time_of_day_raw"),
                ]
            )
            .drop("pairs")
        )

        time_map = getattr(analysis_class, "time_map", {})

        df_ct = (
            df_ct
            .with_columns(
                pl.col("time_of_day_raw")
                .map_elements(lambda x: _map_with_fallback(time_map, x), return_dtype=pl.Utf8)
                .alias("time_of_day_name")
            )
            .filter(
                pl.col("continent").is_not_null()
                & (pl.col("continent") != "")
                & pl.col("time_of_day_name").is_not_null()
            )
        )

        cont_tod_long = (
            df_ct
            .group_by(["continent", "time_of_day_name"])
            .len()
            .rename({"len": "entries"})
        )

        cont_tod_wide = (
            cont_tod_long
            .pivot(index="continent", on="time_of_day_name", values="entries", aggregate_function="first")
            .fill_null(0)
        )
        cont_tod_wide = _ensure_zero_cols(cont_tod_wide, ["Day", "Night"], dtype=pl.Int64)  # type: ignore

        cont_tod_wide = cont_tod_wide.with_columns((pl.col("Day") + pl.col("Night")).alias("total_entries"))

        cont_tod_wide = cont_tod_wide.with_columns(
            [
                pl.when(pl.col("total_entries") > 0)
                .then((pl.col("Day") / pl.col("total_entries") * 100).round(2))
                .otherwise(pl.lit(0.0))
                .alias("day_pct"),
                pl.when(pl.col("total_entries") > 0)
                .then((pl.col("Night") / pl.col("total_entries") * 100).round(2))
                .otherwise(pl.lit(0.0))
                .alias("night_pct"),
            ]
        )

        tot_day = int(cont_tod_wide.select(pl.sum("Day")).item() or 0)
        tot_night = int(cont_tod_wide.select(pl.sum("Night")).item() or 0)
        tot_all = max(tot_day + tot_night, 1)

        cont_tod_total = pl.DataFrame(
            {
                "continent": ["Total"],
                "Day": [tot_day],
                "Night": [tot_night],
                "total_entries": [tot_day + tot_night],
                "day_pct": [round(tot_day / tot_all * 100, 2)],
                "night_pct": [round(tot_night / tot_all * 100, 2)],
            }
        )

        cont_tod_wide_total = _sort_continent(_vstack_like(cont_tod_wide, cont_tod_total))

        cont_tod_paper = (
            cont_tod_wide_total
            .with_columns(
                [
                    _count_pct_fmt(pl.col("Day"), pl.col("day_pct")).alias("day_entries"),
                    _count_pct_fmt(pl.col("Night"), pl.col("night_pct")).alias("night_entries"),
                ]
            )
            .select(["continent", "day_entries", "night_entries", "total_entries"])
        )

        logger.info("\n=== [rollups] D) Continent x time-of-day label entries (paper) ===")
        logger.info(f"\n{_df_full_str(cont_tod_paper)}")

        logger.info("\n=== [rollups] D) Continent x time-of-day label entries (raw wide; with pct) ===")
        logger.info(f"\n{_df_full_str(cont_tod_wide_total)}")

        # sanity check vs segment_records total (should match globally, if same mapping)
        total_entries_global = int(cont_tod_wide_total.filter(pl.col("continent") == "Total")
                                   .select("total_entries").item() or 0)
        if total_entries_global != total_segment_records:
            logger.warning(
                f"[rollups] Segment count sanity: total_entries_global={total_entries_global} != total_segment_records(df_base)={total_segment_records}."  # noqa: E501
                "This can happen if time_of_day parsing differs between df_mapping and df_base.",
            )
        else:
            logger.info(
                f"[rollups] Segment count sanity: total_entries_global matches total_segment_records ({total_segment_records}).",  # noqa: E501
            )

    # =========================================================================
    # E) Upload day/night composition (global): day-only / night-only / both (+%) + Total
    # =========================================================================
    if {"videos", "time_of_day"} <= set(df_mapping.columns):

        vid_flag_dtype = pl.List(
            pl.Struct(
                [
                    pl.Field("video_id", pl.Utf8),
                    pl.Field("has_day", pl.Boolean),
                    pl.Field("has_night", pl.Boolean),
                ]
            )
        )

        def _video_daynight_pairs(row: dict) -> list[dict]:
            vids = row.get("videos_list", [])
            tods = row.get("time_of_day_lol", [])
            out = []
            for i, vid in enumerate(vids):
                vid_str = str(vid).strip()
                if not vid_str:
                    continue
                tod = tods[i] if (isinstance(tods, list) and i < len(tods)) else []
                flags = set()
                for t in (tod if isinstance(tod, list) else [tod]):
                    iv = _safe_num_to_int(t)
                    if iv in (0, 1):
                        flags.add(iv)
                out.append({"video_id": vid_str, "has_day": (0 in flags), "has_night": (1 in flags)})
            return out

        df_video_flags = (
            df_base
            .select(["videos_list", "time_of_day_lol"])
            .with_columns(
                pl.struct(["videos_list", "time_of_day_lol"])
                .map_elements(lambda r: _video_daynight_pairs(r), return_dtype=vid_flag_dtype)
                .alias("video_flags")
            )
            .explode("video_flags")
            .with_columns(
                [
                    pl.col("video_flags").struct.field("video_id").alias("video_id"),
                    pl.col("video_flags").struct.field("has_day").alias("has_day"),
                    pl.col("video_flags").struct.field("has_night").alias("has_night"),
                ]
            )
            .drop("video_flags")
            .filter(pl.col("video_id").is_not_null() & (pl.col("video_id") != ""))
            .group_by("video_id")
            .agg(
                [
                    pl.any("has_day").alias("has_day"),  # type: ignore
                    pl.any("has_night").alias("has_night"),  # type: ignore
                ]
            )
            .with_columns(
                pl.when(pl.col("has_day") & pl.col("has_night")).then(pl.lit("both_day_night"))
                .when(pl.col("has_day")).then(pl.lit("only_day"))
                .when(pl.col("has_night")).then(pl.lit("only_night"))
                .otherwise(pl.lit("unknown"))
                .alias("daynight_category")
            )
        )

        daynight_global = (
            df_video_flags
            .group_by("daynight_category")
            .len()
            .rename({"len": "uploads"})
        )

        order_map = {"only_day": 0, "only_night": 1, "both_day_night": 2, "unknown": 3}
        daynight_global = (
            daynight_global
            .with_columns(
                pl.col("daynight_category")
                .map_elements(lambda x: order_map.get(str(x), 99), return_dtype=pl.Int64)
                .alias("__k")
            )
            .sort("__k")
            .drop("__k")
        )

        tot_uploads = int(daynight_global.select(pl.sum("uploads")).item() or 0)
        denom = max(tot_uploads, 1)
        daynight_global = daynight_global.with_columns((pl.col("uploads") / pl.lit(denom) * 100).round(2).alias("pct"))

        daynight_total = pl.DataFrame({"daynight_category": ["Total"], "uploads": [tot_uploads], "pct": [100.0]})
        daynight_global_table = _vstack_like(daynight_global, daynight_total)

        logger.info("\n=== [rollups] E) Upload day/night composition (global; paper) ===")
        logger.info(f"\n{_df_full_str(daynight_global_table)}")

    # =========================================================================
    # F) Upload day/night composition by continent (CANONICAL continent; totals consistent)
    # =========================================================================
    if {"continent", "videos", "time_of_day"} <= set(df_mapping.columns):

        # join canonical continent onto global video flags
        df_flags_canon = (
            df_video_flags
            .join(df_vid_canon, on="video_id", how="left")
            .filter(pl.col("continent").is_not_null() & (pl.col("continent") != ""))
        )

        cont_daynight_long = (
            df_flags_canon
            .group_by(["continent", "daynight_category"])
            .len()
            .rename({"len": "unique_uploads"})
        )

        cont_daynight_wide = (
            cont_daynight_long
            .pivot(index="continent", on="daynight_category", values="unique_uploads", aggregate_function="first")
            .fill_null(0)
        )
        cont_daynight_wide = _ensure_zero_cols(
            cont_daynight_wide,
            ["only_day", "only_night", "both_day_night", "unknown"], dtype=pl.Int64,  # type: ignore
        )

        cont_daynight_wide = cont_daynight_wide.with_columns(
            (pl.col("only_day") + pl.col("only_night") + pl.col("both_day_night") + pl.col("unknown")).
            alias("unique_uploads")
        )

        cont_daynight_wide = cont_daynight_wide.with_columns(
            [
                pl.when(pl.col("unique_uploads") > 0)
                .then((pl.col("only_day") / pl.col("unique_uploads") * 100).round(2))
                .otherwise(pl.lit(0.0))
                .alias("only_day_pct"),
                pl.when(pl.col("unique_uploads") > 0)
                .then((pl.col("only_night") / pl.col("unique_uploads") * 100).round(2))
                .otherwise(pl.lit(0.0))
                .alias("only_night_pct"),
                pl.when(pl.col("unique_uploads") > 0)
                .then((pl.col("both_day_night") / pl.col("unique_uploads") * 100).round(2))
                .otherwise(pl.lit(0.0))
                .alias("both_day_night_pct"),
                pl.when(pl.col("unique_uploads") > 0)
                .then((pl.col("unknown") / pl.col("unique_uploads") * 100).round(2))
                .otherwise(pl.lit(0.0))
                .alias("unknown_pct"),
            ]
        )

        # add Total row (global, consistent)
        tot_only_day = int(df_flags_canon.filter(pl.col("daynight_category") == "only_day").height)
        tot_only_night = int(df_flags_canon.filter(pl.col("daynight_category") == "only_night").height)
        tot_both = int(df_flags_canon.filter(pl.col("daynight_category") == "both_day_night").height)
        tot_unknown = int(df_flags_canon.filter(pl.col("daynight_category") == "unknown").height)
        tot_all = max(tot_only_day + tot_only_night + tot_both + tot_unknown, 1)

        cont_dn_total = pl.DataFrame(
            {
                "continent": ["Total"],
                "only_day": [tot_only_day],
                "only_night": [tot_only_night],
                "both_day_night": [tot_both],
                "unknown": [tot_unknown],
                "unique_uploads": [tot_only_day + tot_only_night + tot_both + tot_unknown],
                "only_day_pct": [round(tot_only_day / tot_all * 100, 2)],
                "only_night_pct": [round(tot_only_night / tot_all * 100, 2)],
                "both_day_night_pct": [round(tot_both / tot_all * 100, 2)],
                "unknown_pct": [round(tot_unknown / tot_all * 100, 2)],
            }
        )
        cont_daynight_wide_total = _sort_continent(_vstack_like(cont_daynight_wide, cont_dn_total))

        cont_daynight_paper = (
            cont_daynight_wide_total
            .with_columns(
                [
                    _count_pct_fmt(pl.col("only_day"), pl.col("only_day_pct")).alias("day_only"),
                    _count_pct_fmt(pl.col("only_night"), pl.col("only_night_pct")).alias("night_only"),
                    _count_pct_fmt(pl.col("both_day_night"), pl.col("both_day_night_pct")).alias("both_day_night"),
                    _count_pct_fmt(pl.col("unknown"), pl.col("unknown_pct")).alias("unknown"),
                ]
            )
            .select(["continent", "day_only", "night_only", "both_day_night", "unknown", "unique_uploads"])
        )

        logger.info("\n=== [rollups] F) Upload day/night composition by continent (canonical; paper) ===")
        logger.info(f"\n{_df_full_str(cont_daynight_paper)}")

        logger.info("\n=== [rollups] F) Upload day/night composition by continent (raw wide; with pct) ===")
        logger.info(f"\n{_df_full_str(cont_daynight_wide_total)}")

        # ensure totals consistent with global unique
        tot_u = int(cont_dn_total.select("unique_uploads").item() or 0)
        if tot_u != total_unique_global:
            logger.warning(
                f"[rollups] Canonical by-continent unique_uploads total={tot_u} != global unique videos={total_unique_global} (unexpected).",  # noqa: E501
            )
        else:
            logger.info(f"[rollups] Canonical by-continent totals match global unique videos {total_unique_global}.")

    # =========================================================================
    # G) Vehicle types per continent summary (paper table) + per-continent x time-of-day pairs
    # =========================================================================
    if {"continent", "vehicle_type", "time_of_day"} <= set(df_mapping.columns):

        cont_vehicle_pair_dtype = pl.List(
            pl.Struct(
                [
                    pl.Field("continent", pl.Utf8),
                    pl.Field("vehicle_type", pl.Utf8),
                    pl.Field("time_of_day", pl.Utf8),
                ]
            )
        )

        def _expand_continent_vehicle_tod(row: dict) -> list[dict]:
            contv = "" if row.get("continent") is None else str(row.get("continent"))
            try:
                vts = _safe_eval_list(row.get("vehicle_type"))
                tods = _safe_eval_list(row.get("time_of_day"))
                if not (isinstance(vts, list) and isinstance(tods, list)):
                    return []
            except Exception:
                return []
            out = []
            for vt, tod in zip(vts, tods):
                vt_list = vt if isinstance(vt, list) else [vt]
                tod_list = tod if isinstance(tod, list) else [tod]
                for v in vt_list:
                    for t in tod_list:
                        out.append({"continent": contv, "vehicle_type": str(v), "time_of_day": str(t)})
            return out

        df_cont_veh_pairs = (
            df_mapping
            .select(["continent", "vehicle_type", "time_of_day"])
            .with_columns(
                pl.struct(["continent", "vehicle_type", "time_of_day"])
                .map_elements(lambda r: _expand_continent_vehicle_tod(r), return_dtype=cont_vehicle_pair_dtype)
                .alias("pairs")
            )
            .explode("pairs")
            .with_columns(
                [
                    pl.col("pairs").struct.field("continent").alias("continent"),
                    pl.col("pairs").struct.field("vehicle_type").alias("vehicle_type_raw"),
                    pl.col("pairs").struct.field("time_of_day").alias("time_of_day_raw"),
                ]
            )
            .drop("pairs")
            .filter(pl.col("continent").is_not_null() & (pl.col("continent") != ""))
        )

        vehicle_map = getattr(analysis_class, "vehicle_map", {})
        time_map = getattr(analysis_class, "time_map", {})

        df_cont_veh_pairs = (
            df_cont_veh_pairs
            .with_columns(
                [
                    pl.col("vehicle_type_raw")
                    .map_elements(lambda x: _map_with_fallback(vehicle_map, x), return_dtype=pl.Utf8)
                    .alias("vehicle_type_name"),
                    pl.col("time_of_day_raw")
                    .map_elements(lambda x: _map_with_fallback(time_map, x), return_dtype=pl.Utf8)
                    .alias("time_of_day_name"),
                ]
            )
            .filter(pl.col("vehicle_type_name").is_not_null() & pl.col("time_of_day_name").is_not_null())
        )

        cont_vehicle_presence = (
            df_cont_veh_pairs
            .with_columns(
                [
                    (pl.col("time_of_day_name") == "Day").alias("has_day"),
                    (pl.col("time_of_day_name") == "Night").alias("has_night"),
                ]
            )
            .group_by(["continent", "vehicle_type_name"])
            .agg([pl.any("has_day").alias("has_day"), pl.any("has_night").alias("has_night")])  # type: ignore
            .with_columns(
                pl.when(pl.col("has_day") & pl.col("has_night")).then(pl.lit("both_day_night"))
                .when(pl.col("has_day")).then(pl.lit("only_day"))
                .when(pl.col("has_night")).then(pl.lit("only_night"))
                .otherwise(pl.lit("unknown"))
                .alias("daynight_category")
            )
        )

        vehicle_types_continent = (
            cont_vehicle_presence
            .group_by("continent")
            .agg(
                [
                    pl.col("vehicle_type_name").n_unique().alias("unique_vehicle_types"),
                    (pl.col("daynight_category") == "only_day").sum().cast(pl.Int64).alias("day_only_types"),
                    (pl.col("daynight_category") == "only_night").sum().cast(pl.Int64).alias("night_only_types"),
                    (pl.col("daynight_category") == "both_day_night").sum().cast(
                        pl.Int64).alias("both_day_night_types"),
                ]
            )
        )

        vehicle_types_paper = (
            vehicle_types_continent
            .select(["continent", "unique_vehicle_types", "day_only_types",
                     "night_only_types", "both_day_night_types"])
            .rename({"both_day_night_types": "types_in_both_day_night"})
        )
        vehicle_types_paper = _sort_continent(vehicle_types_paper)

        logger.info("\n=== [rollups] G) Vehicle types per continent (paper) ===")
        logger.info(f"\n{_df_full_str(vehicle_types_paper)}")

        cont_veh_tod = (
            df_cont_veh_pairs
            .group_by(["continent", "vehicle_type_name", "time_of_day_name"])
            .len()
            .rename({"len": "count"})
            .with_columns((pl.col("count") / pl.sum("count").
                           over("continent") * 100).round(2).alias("pct_within_continent"))
        )
        cont_veh_tod = _sort_continent(cont_veh_tod)

        logger.info("\n=== [rollups] G) Vehicle types per continent x time-of-day (pair counts + % within continent) ===")  # noqa: E501
        logger.info(f"\n{_df_full_str(cont_veh_tod)}")

    # =========================================================================
    # H) Coverage extremes: max/min locality and country by uploads, segments, and duration
    # =========================================================================
    locality_rank = (
        df_base
        .filter(pl.col("locality").is_not_null() & (pl.col("locality") != ""))
        .group_by(["continent", "country", "locality", "iso3"])
        .agg(
            [
                pl.sum("upload_records").alias("upload_count"),
                pl.sum("segment_records").alias("segment_count"),
                pl.sum("footage_time_s").alias("footage_time_s"),
            ]
        )
        .with_columns((pl.col("footage_time_s") / 3600).round(2).alias("footage_time_h"))
    )

    # =========================================================================
    # H1) Max locality by duration within each continent
    # =========================================================================
    max_locality_per_continent_all_ties = (
        locality_rank
        .filter(pl.col("footage_time_s") > 0)
        .join(
            locality_rank
            .group_by("continent")
            .agg(pl.max("footage_time_s").alias("max_footage_time_s")),
            on="continent",
            how="inner",
        )
        .filter(pl.col("footage_time_s") == pl.col("max_footage_time_s"))
        .select(
            [
                "continent",
                "country",
                "locality",
                "iso3",
                "upload_count",
                "segment_count",
                "footage_time_s",
                "footage_time_h",
            ]
        )
        .sort(
            ["continent", "footage_time_s", "segment_count", "upload_count", "locality"],
            descending=[False, True, True, True, False],
        )
    )

    logger.info("\n=== [rollups] H1) Max locality by duration within each continent (including ties) ===")
    logger.info(f"\n{_df_full_str(max_locality_per_continent_all_ties)}")

    # Exactly one locality per continent (tie break by segment_count, then upload_count, then locality name)
    max_locality_per_continent_one = (
        max_locality_per_continent_all_ties
        .unique(subset=["continent"], keep="first")
        .sort("continent")
    )

    logger.info("\n=== [rollups] H1) Max locality by duration within each continent (one per continent) ===")
    logger.info(f"\n{_df_full_str(max_locality_per_continent_one)}")

    country_rank = (
        df_base
        .filter(pl.col("country").is_not_null() & (pl.col("country") != ""))
        .group_by(["continent", "country", "iso3"])
        .agg(
            [
                pl.sum("upload_records").alias("upload_count"),
                pl.sum("segment_records").alias("segment_count"),
                pl.sum("footage_time_s").alias("footage_time_s"),
            ]
        )
        .with_columns((pl.col("footage_time_s") / 3600).round(2).alias("footage_time_h"))
    )

    def _top_bottom(df: "pl.DataFrame", col: str):
        top = df.sort(col, descending=True).head(1)
        bot = df.filter(pl.col(col) > 0).sort(col).head(1)
        return top, bot

    top_locality_u, bot_locality_u = _top_bottom(locality_rank, "upload_count")
    top_locality_s, bot_locality_s = _top_bottom(locality_rank, "segment_count")
    top_locality_t, bot_locality_t = _top_bottom(locality_rank, "footage_time_s")

    top_ctry_u, bot_ctry_u = _top_bottom(country_rank, "upload_count")
    top_ctry_s, bot_ctry_s = _top_bottom(country_rank, "segment_count")
    top_ctry_t, bot_ctry_t = _top_bottom(country_rank, "footage_time_s")

    logger.info("\n=== [rollups] H) Max/Min locality by upload_count ===")
    logger.info(f"\nMAX:\n{_df_full_str(top_locality_u)}\nMIN (non-zero):\n{_df_full_str(bot_locality_u)}")

    logger.info("\n=== [rollups] H) Max/Min locality by segment_count ===")
    logger.info(f"\nMAX:\n{_df_full_str(top_locality_s)}\nMIN (non-zero):\n{_df_full_str(bot_locality_s)}")

    logger.info("\n=== [rollups] H) Max/Min locality by duration ===")
    logger.info(f"\nMAX:\n{_df_full_str(top_locality_t)}\nMIN (non-zero):\n{_df_full_str(bot_locality_t)}")

    logger.info("\n=== [rollups] H) Max/Min COUNTRY by upload_count ===")
    logger.info(f"\nMAX:\n{_df_full_str(top_ctry_u)}\nMIN (non-zero):\n{_df_full_str(bot_ctry_u)}")

    logger.info("\n=== [rollups] H) Max/Min COUNTRY by segment_count ===")
    logger.info(f"\nMAX:\n{_df_full_str(top_ctry_s)}\nMIN (non-zero):\n{_df_full_str(bot_ctry_s)}")

    logger.info("\n=== [rollups] H) Max/Min COUNTRY by duration ===")
    logger.info(f"\nMAX:\n{_df_full_str(top_ctry_t)}\nMIN (non-zero):\n{_df_full_str(bot_ctry_t)}")


# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    if os.path.exists(file_results) and not common.get_configs('always_analyse'):
        # Load the data from the pickle file
        with open(file_results, 'rb') as file:
            (data,                                          # 0
             person_counter,                                # 1
             bicycle_counter,                               # 2
             car_counter,                                   # 3
             motorcycle_counter,                            # 4
             bus_counter,                                   # 5
             truck_counter,                                 # 6
             cellphone_counter,                             # 7
             traffic_light_counter,                         # 8
             stop_sign_counter,                             # 9
             pedestrian_cross_locality,                         # 10
             pedestrian_crossing_count,                     # 11
             person_locality,                                   # 12
             bicycle_locality,                                  # 13
             car_locality,                                      # 14
             motorcycle_locality,                               # 15
             bus_locality,                                      # 16
             truck_locality,                                    # 17
             cross_evnt_locality,                               # 18
             vehicle_locality,                                  # 19
             cellphone_locality,                                # 20
             traffic_sign_locality,                             # 21
             all_speed,                                     # 22
             all_time,                                      # 23
             avg_time_locality,                                 # 24
             avg_speed_locality,                                # 25
             df_mapping,                                    # 26
             avg_speed_country,                             # 27
             avg_time_country,                              # 28
             crossings_with_traffic_equipment_locality,         # 29
             crossings_without_traffic_equipment_locality,      # 30
             crossings_with_traffic_equipment_country,      # 31
             crossings_without_traffic_equipment_country,   # 32
             min_max_speed,                                 # 33
             min_max_time,                                  # 34
             pedestrian_cross_country,                      # 35
             all_speed_locality,                                # 36
             all_time_locality,                                 # 37
             all_speed_country,                             # 38
             all_time_country,                              # 39
             df_mapping_raw,                                # 40
             pedestrian_cross_locality_all,                     # 41
             pedestrian_cross_country_all                   # 42
             ) = pickle.load(file)

        logger.info("Loaded analysis results from pickle file.")
        log_rollups(df_mapping)
    else:
        # Store the mapping file
        df_mapping = pl.read_csv(common.get_configs("mapping"))

        # Produce map with all data
        df = df_mapping.clone()  # copy df to manipulate for output
        df = df.with_columns(pl.col("state").fill_null("NA").alias("state"))

        # Sort by continent and locality, both in ascending order
        df = df.sort(by=["continent", "locality"])

        # Count of videos (handles: [id], "[id1,id2]", and [] -> 0)
        videos_clean = (
            pl.col("videos")
              .cast(pl.Utf8)
              .str.strip_chars("\"'")   # remove surrounding quotes if present
              .str.strip_chars("[]")     # remove surrounding brackets
              .str.strip_chars()         # trim whitespace
        )

        df = df.with_columns(
            pl.when(pl.col("videos").is_null() | (videos_clean == ""))
              .then(0)
              .otherwise(
                  videos_clean
                  .str.split(",")
                  .list.eval(pl.element().str.strip_chars())  # trim each item
                  .list.filter(pl.element() != "")            # drop empties (so [] -> 0)
                  .list.len()
              ).alias("video_count")
        )

        # Total amount of seconds in segments
        def flatten(lst):
            """Flattens nested lists like [[1, 2], [3, 4]] -> [1, 2, 3, 4]"""
            out = []
            for sub in lst:
                if isinstance(sub, list):
                    out.extend(sub)
                else:
                    out.append(sub)
            return out

        def compute_total_time(row: dict) -> int:
            try:
                start_raw = row.get("start_time")
                end_raw = row.get("end_time")

                start_times = flatten(ast.literal_eval(start_raw)) if start_raw is not None else []
                end_times = flatten(ast.literal_eval(end_raw)) if end_raw is not None else []

                return int(sum(processed_segment_duration_seconds(s, e) for s, e in zip(start_times, end_times)))
            except Exception as e:
                logger.error(f"Error in row {row.get('id')}: {e}")
                return 0

        df = df.with_columns(
            pl.struct(["id", "start_time", "end_time"])
              .map_elements(compute_total_time, return_dtype=pl.Int64)
              .alias("total_time")
        )

        # create flag_locality column
        flag_expr = pl.col("iso3").map_elements(
            lambda x: analysis_class.iso3_to_flag.get(x, "🏳️"),
            return_dtype=pl.Utf8,
        )

        # Create a new country label with emoji flag + country name
        df = df.with_columns([
            pl.concat_str([flag_expr, pl.col("locality").cast(pl.Utf8)], separator=" ").alias("flag_locality"),
            pl.concat_str([flag_expr, pl.col("country").cast(pl.Utf8)], separator=" ").alias("flag_country"),
        ])

        # Data to avoid showing on hover in scatter plots
        columns_remove = ['videos', 'time_of_day', 'start_time', 'end_time', 'upload_date', 'vehicle_type', 'channel',
                          'display_label', 'flag_locality', 'flag_country']

        hover_data = sorted(list(set(df.columns) - set(columns_remove)))

        # Sort by continent and locality, both in ascending order
        df = df.sort(["continent", "country"])

        # map with all cities
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        hover_name="flag_locality",
                        marker_size=4,
                        file_name='mapbox_map_all')

        # # map with all cities coloured by footage amount (continuous hue scale) + optional screenshot overlays
        # maps.mapbox_map_footage(df=df.to_pandas(),
        #                         footage_col="total_time",
        #                         hover_data=hover_data,
        #                         hover_name="flag_locality",
        #                         marker_size=3,
        #                         log_colour=True,
        #                         show_images=True,
        #                         file_name='mapbox_map_all_footage')

        maps.world_map_ss(
            df=df.to_pandas(),
            df_mapping=df_mapping.to_pandas(),
            show_images=True,
            hover_data=["total_time"],
            save_file=False,
            show_colorbar=True,
            colorbar_title="Footage (hours)",
        )

        # Sort by continent and locality, both in ascending order
        df = df.sort(["country", "locality"])

        # scatter plot for cities with number of videos over total time
        bivariate.scatter(df=df,
                          x="total_time",
                          y="video_count",
                          color="flag_country",
                          text="flag_locality",
                          xaxis_title='Total time of footage (s)',
                          yaxis_title='Number of videos',
                          pretty_text=False,
                          marker_size=10,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="flag_locality",
                          legend_title="",
                          # legend_x=0.01,
                          # legend_y=1.0,
                          label_distance_factor=5.0,
                          marginal_x=None,  # type: ignore
                          marginal_y=None,  # type: ignore
                          file_name='scatter_all_total_time-video_count')  # type: ignore
        # scatter plot for countries with number of videos over total time

        # Reuse the already computed locality-level values.
        # total_time was computed above by summing processed segment durations:
        # processed duration = (end_time - 1 second) - start_time, with fallback
        # to the original end_time for very short segments.
        df = df.with_columns([
            pl.col("video_count").cast(pl.Int64).alias("locality_video_count"),
            pl.col("total_time").cast(pl.Int64).alias("locality_total_time"),
        ])

        # ---------- Aggregate to country level ----------
        df_country = (
            df.group_by(["country", "iso3", "continent"])
              .agg([
                  pl.col("locality_total_time").sum().alias("total_time"),
                  pl.col("locality_video_count").sum().alias("video_count"),
              ])
        )

        # add flag + iso3 label
        df_country = df_country.with_columns(
            pl.concat_str(
                [
                    pl.col("iso3").map_elements(
                        lambda x: analysis_class.iso3_to_flag.get(x, "🏳️"),
                        return_dtype=pl.Utf8,
                    ),
                    pl.col("iso3").cast(pl.Utf8),
                ],
                separator=" ",
            ).alias("flag_country")
        )

        # sort for readability
        df_country = df_country.sort(["continent", "country"])

        # define hover data
        hover_data = ["country", "continent", "total_time", "video_count"]

        # plot (convert at plotting boundary)
        bivariate.scatter(df=df_country,
                          x="total_time",
                          y="video_count",
                          color="continent",
                          text="flag_country",
                          xaxis_title="Total time of footage (s)",
                          yaxis_title="Number of videos",
                          pretty_text=False,
                          marker_size=12,
                          save_file=True,
                          hover_data=hover_data,
                          hover_name="flag_country",
                          legend_title="",
                          label_distance_factor=0.1,
                          marginal_x=None,  # type: ignore
                          marginal_y=None,  # type: ignore
                          file_name="scatter_all_country_total_time-video_count")

        # histogram of dates of videos
        distribution.video_histogram_by_month(df=df.to_pandas(),
                                              video_count_col='video_count',
                                              upload_date_col='upload_date',
                                              xaxis_title='Year',
                                              yaxis_title='Number of videos',
                                              save_file=True)

        # maps with all cities and population heatmap
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        density_col='population_locality',
                        density_radius=10,
                        file_name='mapbox_map_all_pop')

        # maps with all cities and video count heatmap
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        density_col='video_count',
                        density_radius=10,
                        file_name='mapbox_map_all_videos')

        # maps with all cities and total time heatmap
        maps.mapbox_map(df=df.to_pandas(),
                        hover_data=hover_data,
                        density_col='total_time',
                        density_radius=10,
                        file_name='mapbox_map_all_time')

        # Type of vehicle over time of day
        df = df_mapping.clone()  # copy df to manipulate for output

        # --- expand rows so each video becomes one row ---
        # Return type: List[Struct{vehicle_type: Utf8, time_of_day: Utf8}]
        pair_dtype = pl.List(
            pl.Struct([
                pl.Field("vehicle_type", pl.Utf8),
                pl.Field("time_of_day", pl.Utf8),
            ])
        )

        def expand_pairs(vs: str | None, ts: str | None) -> list[dict]:
            """Parse stringified lists (possibly nested) and emit expanded (vehicle_type,
               time_of_day) pairs as strings."""
            try:
                vehicle_types = ast.literal_eval(vs) if isinstance(vs, str) else None
                times_of_day = ast.literal_eval(ts) if isinstance(ts, str) else None
                if not (isinstance(vehicle_types, list) and isinstance(times_of_day, list)):
                    return []
            except Exception:
                return []

            out: list[dict] = []
            for v_type, tod in zip(vehicle_types, times_of_day):
                v_list = v_type if isinstance(v_type, list) else [v_type]
                t_list = tod if isinstance(tod, list) else [tod]
                for vt in v_list:
                    for t in t_list:
                        out.append({"vehicle_type": str(vt), "time_of_day": str(t)})
            return out

        def map_with_fallback(dct: dict, v):
            """
            Robust dict lookup for values that may arrive as str/int/float (or numeric strings).
            Tries:
              1) direct key
              2) string key (stripped)
              3) int key (from int(v) or int(float(v)) for "1.0")
              4) float key (rare, but safe)
            Returns None if no match.
            """
            if v is None:
                return None

            # 1) direct key
            if v in dct:
                return dct[v]

            # Normalize string form
            sv = v.strip() if isinstance(v, str) else str(v).strip()

            # 2) string key
            if sv in dct:
                return dct[sv]

            # 3) int key (handle "1" and "1.0")
            try:
                iv = int(sv)
                if iv in dct:
                    return dct[iv]
            except Exception:
                try:
                    iv = int(float(sv))
                    if iv in dct:
                        return dct[iv]
                except Exception:
                    pass

            # 4) float key (less common, but harmless)
            try:
                fv = float(sv)
                if fv in dct:
                    return dct[fv]
            except Exception:
                pass

            return None

        # --- expand rows ---
        df_expanded = (
            df.select(["vehicle_type", "time_of_day"])
              .with_columns(
                  pl.struct(["vehicle_type", "time_of_day"])
                  .map_elements(
                        lambda r: expand_pairs(r["vehicle_type"], r["time_of_day"]),
                        return_dtype=pair_dtype,
                    ).alias("pairs")
              ).select("pairs")               # avoid duplicate column name collisions
               .explode("pairs")
               .with_columns([
                  pl.col("pairs").struct.field("vehicle_type").alias("vehicle_type"),
                  pl.col("pairs").struct.field("time_of_day").alias("time_of_day"),
                  ]).drop("pairs")
        )

        # --- map to human-readable labels ---
        df_expanded = df_expanded.with_columns([
            pl.col("vehicle_type").map_elements(
                lambda x: map_with_fallback(analysis_class.vehicle_map, x),
                return_dtype=pl.Utf8,
            ).alias("vehicle_type_name"),
            pl.col("time_of_day").map_elements(
                lambda x: map_with_fallback(analysis_class.time_map, x),
                return_dtype=pl.Utf8,
            ).alias("time_of_day_name"),
        ])

        # drop rows where mapping failed
        df_expanded = df_expanded.filter(
            pl.col("vehicle_type_name").is_not_null() & pl.col("time_of_day_name").is_not_null()
        )

        # --- aggregate counts ---
        df_summary = (
            df_expanded
            .group_by(["vehicle_type_name", "time_of_day_name"])
            .len()
            .rename({"len": "count"})
        )

        # --- pivot into wide format for stacked bar plot ---
        df_pivot = df_summary.pivot(
            index="vehicle_type_name",
            on="time_of_day_name",      # renamed from `columns`
            values="count",
            aggregate_function="first",
        ).fill_null(0)

        # ensure consistent order of vehicle types
        vehicle_order = [
            "Car", "Bus", "Truck", "Two-wheeler", "Bicycle", "Automated car", "Automated bus", "Automated truck",
            "Automated two-wheeler", "Electric scooter"
        ]
        order_map = {name: i for i, name in enumerate(vehicle_order)}

        df_pivot = (
            df_pivot
            .with_columns(
                pl.col("vehicle_type_name")
                  .map_elements(lambda x: order_map.get(x, 10**9), return_dtype=pl.Int64)
                  .alias("_order")
            )
            .sort("_order")
            .drop("_order")
        )
        # --- plot ---
        distribution.bar(
            df=df_pivot.to_pandas(),
            x=df_pivot["vehicle_type_name"],
            y=[col for col in ["Day", "Night"] if col in df_pivot.columns],
            y_legend=["Day", "Night"],
            stacked=True,
            pretty_text=False,
            orientation="v",
            xaxis_title="Type of vehicle",
            yaxis_title="Number of segments",
            show_text_labels=False,
            save_file=True,
            save_final=True,
            name_file="bar_vehicle_type_time_of_day"
        )

        # Continent over time of day
        df = df_mapping.clone()  # copy df to manipulate for output

        # --- expand rows so each video becomes one row ---
        pair_dtype = pl.List(
            pl.Struct([
                pl.Field("continent", pl.Utf8),
                pl.Field("time_of_day", pl.Utf8),
            ])
        )

        def expand_continent_tod(continent: str | None, ts: str | None) -> list[dict]:
            try:
                times_of_day = ast.literal_eval(ts) if isinstance(ts, str) else None
                if not isinstance(times_of_day, list):
                    return []
            except Exception:
                return []

            cont = "" if continent is None else str(continent)

            out: list[dict] = []
            for tod in times_of_day:
                t_list = tod if isinstance(tod, list) else [tod]
                for t in t_list:
                    out.append({"continent": cont, "time_of_day": str(t)})
            return out

        # --- expand rows so each time-of-day entry becomes one row ---
        df_expanded = (
            df.select(["continent", "time_of_day"])
              .with_columns(
                  pl.struct(["continent", "time_of_day"])
                  .map_elements(
                        lambda r: expand_continent_tod(r["continent"], r["time_of_day"]),
                        return_dtype=pair_dtype,
                    ).alias("pairs")
              ).select("pairs").explode("pairs").with_columns([
                  pl.col("pairs").struct.field("continent").alias("continent"),
                  pl.col("pairs").struct.field("time_of_day").alias("time_of_day"),
              ]).drop("pairs")
        )

        # --- map to human-readable labels ---
        df_expanded = df_expanded.with_columns(
            pl.col("time_of_day").map_elements(
                lambda x: map_with_fallback(analysis_class.time_map, x),
                return_dtype=pl.Utf8,
            ).alias("time_of_day_name")
        )

        # drop rows where mapping failed
        df_expanded = df_expanded.filter(
            pl.col("time_of_day_name").is_not_null() & pl.col("continent").is_not_null() & (pl.col("continent") != "")
        )

        # --- aggregate counts ---
        df_summary = (
            df_expanded
            .group_by(["continent", "time_of_day_name"])
            .len()
            .rename({"len": "count"})
        )

        # --- pivot into wide format for stacked bar plot ---
        df_pivot = (
            df_summary
            .pivot(
                index="continent",
                on="time_of_day_name",   # Polars >= 1.0.0 uses `on` (not `columns`)
                values="count",
                aggregate_function="first",
            )
            .fill_null(0)
        )

        # ensure only expected columns (and ensure they exist)
        for col in ["Day", "Night"]:
            if col not in df_pivot.columns:
                df_pivot = df_pivot.with_columns(pl.lit(0).alias(col))

        # --- enforce alphabetical continent order ---
        df_pivot = df_pivot.sort("continent")

        time_columns = [col for col in ["Day", "Night"] if col in df_pivot.columns]

        # --- plot ---
        distribution.bar(
            df=df_pivot.to_pandas(),
            x=df_pivot["continent"],
            y=time_columns,
            y_legend=time_columns,
            stacked=True,
            pretty_text=False,
            orientation="v",
            xaxis_title="Continent",
            yaxis_title="Number of segments",
            show_text_labels=False,
            save_file=True,
            save_final=True,
            name_file="bar_continent_time_of_day"
        )

        total_duration = dataset_stats.calculate_total_seconds(df_mapping)

        # Displays values before applying filters
        logger.info(
            f"Duration of videos in seconds: {total_duration}, "
            f"in minutes: {total_duration/60:.2f}, "
            f"in hours: {total_duration/3600:.2f}, "
            f"in days: {total_duration/86400:.2f}, "
            f"in weeks: {total_duration/604800:.2f}, "
            f"in months: {total_duration/2629800:.2f}, "   # average month (30.44 days)
            f"in years: {total_duration/31557600:.2f}."    # average year (365.25 days)
        )
        logger.info("Total number of videos: {}.",
                    dataset_stats.calculate_total_videos(df_mapping))

        country, number, _ = metrics_cache.get_unique_values(df_mapping, "iso3")
        logger.info(f"Total number of countries and territories: {number}.")

        locality_state_iso3, number, dup_report = metrics_cache.get_unique_values(
            df_mapping,
            ["locality", "state", "iso3"],
            return_duplicates=True,
        )

        logger.info(f"Total number of unique locality+state+iso3 keys: {number}.")

        if dup_report is not None and dup_report.height > 0:
            logger.warning(f"Duplicated keys:\n{dup_report}")

        # Limit countries if required
        countries_include = common.get_configs("countries_analyse")
        if countries_include:
            df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_include))
        log_rollups(df_mapping)

        logger.info("Analysis complete.")
