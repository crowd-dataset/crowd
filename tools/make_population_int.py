# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs

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
