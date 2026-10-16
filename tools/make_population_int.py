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

# load mapping file
df = pd.read_csv(common.get_configs("mapping"))

for col in ["population_country", "population_locality"]:
    if col in df.columns:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)   # replace missing/invalid with 0
            .astype(int)
        )

# save back
df.to_csv(common.get_configs("mapping"), index=False)
print("✅ population_country and population_locality converted to int and saved.")
