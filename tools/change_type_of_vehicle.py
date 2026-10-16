# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import ast
import csv

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: common, logmod, ...
os.chdir(sys.path[0])  # files such as mapping.csv are opened relative to the top folder
import common  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

INPUT_PATH = "mapping.csv"
OUTPUT_PATH = "mapping_cleaned.csv"


def parse_list(raw: str):
    """Parse a stringified list from a CSV cell.
    Falls back to splitting on commas for lists of unquoted strings
    (e.g. channel IDs) that ast.literal_eval cannot handle."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        # Strip surrounding brackets and split — handles unquoted string lists
        return [item.strip() for item in raw.strip("[]").split(",") if item.strip()]


def remap_vehicle_type(channel: str, vehicle_type: int) -> None:
    """Set vehicle_type to `vehicle_type` for every video belonging to `channel`."""
    with open(INPUT_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = 0
    for row in rows:
        channels = parse_list(row["channel"])
        vtypes = parse_list(row["vehicle_type"])

        patched = [vehicle_type if ch == channel else vt
                   for ch, vt in zip(channels, vtypes)]

        if patched != vtypes:
            row["vehicle_type"] = "[" + ",".join(str(v) for v in patched) + "]"
            changed += 1

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Done. {changed} row(s) updated → {OUTPUT_PATH}")


if __name__ == "__main__":
    remap_vehicle_type(channel="UCIEYAtvhx3RjGYgy_TZr2Jw", vehicle_type=2)
