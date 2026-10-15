# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs


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
