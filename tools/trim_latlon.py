"""lat and lon of mapping.csv rounded to 7 decimals (about 1 cm) and without surrounding spaces; the other cells stay
as they are.

    python -m tools.trim_latlon
"""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import sys

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402
from tools.mapping_io import locked, replace  # noqa: E402

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

MAPPING = "mapping.csv"

if __name__ == "__main__":
    with locked(MAPPING):
        df = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
        for col in ("lat", "lon"):
            df[col] = df[col].str.strip().replace("", "nan").astype(float).round(7)
        replace(MAPPING, lambda tmp: df.to_csv(tmp, index=False))
    logger.info("mapping.csv normalised.")
