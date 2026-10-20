"""Count the YOLO detections in the bbox CSVs and save the per-segment counts as analysis_data/detection_counts.csv.gz.

Only this step of analysis.py: no figures and none of the other analyses of the detections, which read the CSVs in
far more detail. Run it where the bbox CSVs are (for example on the machine their drive is plugged into), then
commit detection_counts.csv.gz: analysis.py and the GitHub Action make the detection figures and README counts from
it when the CSVs are not available.

    python count_detections.py --data /path/to/data [/path/to/other/data ...] [--workers 8]

Each --data folder holds a bbox/ folder of CSVs named {video}_{start}_{fps}.csv (default: the `data` folders in
config). Segments are matched to the mapping file in config.
"""
import argparse
import os
import time

import polars as pl

import analysis
import common


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", help="data folders with a bbox/ folder (default: `data` in config)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel reading threads")
    args = ap.parse_args()
    if args.data:
        configured = common.get_configs
        common.get_configs = lambda key: args.data if key == "data" else configured(key)
    df_mapping = pl.read_csv(common.get_configs("mapping"))
    t = time.time()
    result = analysis.count_detections(df_mapping, workers=args.workers, write_snapshot=True)
    if result is None:
        raise SystemExit("No readable bbox CSV matched a segment of the mapping; nothing written.")
    det, _ = result
    print(f"{det.height:,} segments with detections (one CSV each) counted in {(time.time() - t) / 60:.1f} min; "
          f"written to {analysis.DETECTION_SNAPSHOT}")


if __name__ == "__main__":
    main()
