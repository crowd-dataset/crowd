import os
import io
import re
import time
import zipfile
import pandas as pd
import requests

mapping_path = "mapping.csv"
height_path = "height_data.csv"
countries_path = "countries.csv"
OUT_PATH = "mapping_updated.csv"

WB_INDICATORS = {
    "SH.STA.TRAF.P5": "traffic_mortality",
    "SE.ADT.LITR.MA.ZS": "literacy_rate",
    "SP.POP.TOTL": "population_country",
    "SI.POV.GINI": "gini",
}


def robust_get(url, params=None, timeout=(20, 180), tries=5):
    last = None
    for i in range(tries):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "mapping-script/1.0"},
            )
            r.raise_for_status()
            return r
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as e:
            last = e
            time.sleep(min(2 ** i, 15))
    raise last  # type: ignore


def wb_download_indicator_latest(
    indicator_code: str,
    cache_dir: str = ".wb_cache",
) -> pd.DataFrame:
    """
    Download a World Bank ZIP CSV for an indicator and return the latest
    non empty value per ISO3 country code.

    Returned columns:
    iso3, year, value
    """
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, f"{indicator_code}.zip")

    if not os.path.exists(zip_path):
        url = (
            "https://api.worldbank.org/v2/country/all/indicator/"
            f"{indicator_code}"
        )
        r = robust_get(url, params={"downloadformat": "csv"})
        with open(zip_path, "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        data_name = None
        for name in z.namelist():
            base = os.path.basename(name)
            if base.startswith("API_") and base.lower().endswith(".csv"):
                data_name = name
                break

        if data_name is None:
            raise RuntimeError(
                f"Could not find API_*.csv inside {zip_path}"
            )

        raw = z.read(data_name)

    df = pd.read_csv(io.BytesIO(raw), skiprows=4)

    if "Country Code" not in df.columns:
        raise RuntimeError(
            f"Unexpected CSV format for {indicator_code}"
        )

    year_cols = [
        column
        for column in df.columns
        if re.fullmatch(r"\d{4}", str(column))
    ]

    long = df.melt(
        id_vars=["Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    long["iso3"] = (
        long["Country Code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long.dropna(subset=["year", "value"])

    latest = (
        long.sort_values("year")
        .groupby("iso3", as_index=False)
        .tail(1)
    )

    return latest[["iso3", "year", "value"]]


# 1) Load mapping and update avg_height
mapping = pd.read_csv(mapping_path)
height = pd.read_csv(height_path)

mapping["iso3"] = (
    mapping["iso3"]
    .astype("string")
    .str.strip()
    .str.upper()
)

height["cca3"] = (
    height["cca3"]
    .astype("string")
    .str.strip()
    .str.upper()
)
height["cca2"] = (
    height["cca2"]
    .astype("string")
    .str.strip()
    .str.upper()
)
height["year"] = pd.to_numeric(height["year"], errors="coerce")
height["meanHeightMale"] = pd.to_numeric(
    height["meanHeightMale"],
    errors="coerce",
)
height["meanHeightFemale"] = pd.to_numeric(
    height["meanHeightFemale"],
    errors="coerce",
)

height["avg_height_new"] = (
    (height["meanHeightMale"] + height["meanHeightFemale"]) / 2
).round(2)

height_last = (
    height.sort_values(["cca3", "year"])
    .groupby("cca3", as_index=False)
    .tail(1)[["cca3", "year", "avg_height_new"]]
)

cca2_map = (
    height.sort_values(["cca3", "year"])
    .groupby("cca3")["cca2"]
    .apply(
        lambda values: (
            values.dropna().iloc[-1]
            if len(values.dropna())
            else pd.NA
        )
    )
    .rename("cca2")
    .reset_index()
)

height_latest = height_last.merge(
    cca2_map,
    on="cca3",
    how="left",
)

out = mapping.merge(
    height_latest,
    left_on="iso3",
    right_on="cca3",
    how="left",
)

# Keep the old avg_height only when the new height is missing.
if "avg_height" not in out.columns:
    out["avg_height"] = pd.NA

out["avg_height"] = pd.to_numeric(
    out["avg_height"],
    errors="coerce",
)

out["avg_height"] = out["avg_height_new"].where(
    out["avg_height_new"].notna(),
    out["avg_height"],
).round(2)

missing_height = (
    out[out["avg_height_new"].isna()][["iso3", "country"]]
    .drop_duplicates()
    .sort_values(["country", "iso3"])
)

print("Missing avg_height (no match in height_data.csv):")
print(
    "None"
    if len(missing_height) == 0
    else missing_height.to_string(index=False)
)


# 2) Update med_age from countries.csv using ISO2 lookup
countries = pd.read_csv(countries_path, dtype="string")
countries = countries[countries["country"].ne("country")].copy()

countries["iso2"] = (
    countries["iso2"]
    .astype("string")
    .str.strip()
    .str.upper()
)
countries["median_age"] = pd.to_numeric(
    countries["median_age"],
    errors="coerce",
)

iso2_age = (
    countries.sort_values("median_age", na_position="last")
    .drop_duplicates(subset=["iso2"], keep="first")
    .set_index("iso2")["median_age"]
)

out["iso2_from_height"] = (
    out["cca2"]
    .astype("string")
    .str.strip()
    .str.upper()
)
out["median_age_new"] = out["iso2_from_height"].map(iso2_age)

if "med_age" not in out.columns:
    out["med_age"] = pd.NA

out["med_age"] = out["median_age_new"].where(
    out["median_age_new"].notna(),
    out["med_age"],
)

missing_med_age = (
    out[out["med_age"].isna()][
        ["iso3", "country", "iso2_from_height"]
    ]
    .drop_duplicates()
    .sort_values(["country", "iso3"])
)

print(
    "\nMissing med_age "
    "(no match or empty median_age in countries.csv):"
)
print(
    "None"
    if len(missing_med_age) == 0
    else missing_med_age.to_string(index=False)
)


# 3) World Bank updates
for ind_code, col_name in WB_INDICATORS.items():
    latest = wb_download_indicator_latest(ind_code).rename(
        columns={
            "value": f"{col_name}_new",
            "year": f"{col_name}_year",
        }
    )

    out = out.merge(latest, on="iso3", how="left")

    values = pd.to_numeric(
        out[f"{col_name}_new"],
        errors="coerce",
    )

    if col_name == "population_country":
        # SP.POP.TOTL already provides absolute population counts.
        # Pandas nullable Int64 keeps missing values without writing .0.
        out[col_name] = values.round().astype("Int64")
    elif col_name == "literacy_rate":
        out[col_name] = values.round(2)
    else:
        out[col_name] = values

    missing = (
        out[out[col_name].isna()][["iso3", "country"]]
        .drop_duplicates()
        .sort_values(["country", "iso3"])
    )

    print(f"\nMissing {col_name} (World Bank {ind_code}):")
    print(
        "None"
        if len(missing) == 0
        else missing.to_string(index=False)
    )

    out = out.drop(
        columns=[
            f"{col_name}_new",
            f"{col_name}_year",
        ],
        errors="ignore",
    )


# Ensure the requested output formats before saving.
out["avg_height"] = pd.to_numeric(
    out["avg_height"],
    errors="coerce",
).round(2)

out["literacy_rate"] = pd.to_numeric(
    out["literacy_rate"],
    errors="coerce",
).round(2)

out["population_country"] = pd.to_numeric(
    out["population_country"],
    errors="coerce",
).round().astype("Int64")


# Remove temporary columns and save.
out = out.drop(
    columns=[
        "cca3",
        "avg_height_new",
        "median_age_new",
        "iso2_from_height",
        "cca2",
    ],
    errors="ignore",
)

out = out.drop(
    columns=["year", "year_x", "year_y"],
    errors="ignore",
)

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")