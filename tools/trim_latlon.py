# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# file path
filename = "mapping.csv"


# Load CSV
df = pd.read_csv(filename)

# Clean latitude
df["lat"] = (
    df["lat"]
    .astype(str)
    .str.strip()
    .astype(float)
    .round(7)
)

# Clean longitude
df["lon"] = (
    df["lon"]
    .astype(str)
    .str.strip()
    .astype(float)
    .round(7)
)

# Save back to CSV
df.to_csv(filename, index=False)

logger.info("mapping.csv normalised.")
