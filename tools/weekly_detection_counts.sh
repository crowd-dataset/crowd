#!/bin/bash
# Count the YOLO detections of new bbox CSVs and push detection_counts.csv.gz if it changed. Run weekly by cron on
# the machine the data drive is plugged into (lin4080), on Sunday before the GitHub Action that makes the detection
# figures from it (21:00 UTC). Only new or changed CSVs are read (per-file cache in _output/), so it takes minutes.
#
#   0 20 * * 0 /usr/bin/flock -n /tmp/crowd-counts.lock ~/crowd-counts/tools/weekly_detection_counts.sh \
#       >> ~/crowd-counts/weekly.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
data=${CROWD_DATA:-/media/salam/crowd-tue-3/pedestrians_in-youtube/data}
echo "== $(date -Is)"
[ -d "$data/bbox" ] || { echo "No $data/bbox: drive not mounted?"; exit 1; }
git pull -q --rebase
~/.local/bin/uv run --no-project --python 3.12 -q --with polars --with pyarrow --with pandas==2.2.3 \
    --with plotly==5.24.1 --with kaleido==0.2.1 --with pillow --with tqdm --with numpy --with pycountry \
    --with requests python count_detections.py --data "$data" 2>&1 | tr '\r' '\n' | grep -v '%|'
git add analysis_data/detection_counts.csv.gz
if git diff --staged --quiet; then
    echo "Counts unchanged."
else
    git commit -qm "Update detection counts"
    git pull -q --rebase
    git push -q
    echo "Pushed new counts."
fi
