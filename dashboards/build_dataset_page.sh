#!/bin/zsh
# Nightly (cron 04:00): the dataset page (port 8772, README) from the review mapping, unpushed reviews included.
# 1. a copy of ~/repos/crowd (its mapping) with GitHub's main (its code) in
#    ~/dataset-page/work with the GitHub Action's settings (no data or video folders, empty keys),
#    never the review repo itself: analysis.py rewrites README.md and figures/ where it runs
# 2. analysis.py there (its own env "dataset", the Action's package versions; kaleido needs the env on PATH)
# 3. dataset_page.py: the README's dataset sections and the figures as one page in ~/dataset-page/site
# Log ~/Library/Logs/dataset_page.log. EPS exports fail (no pdftops); the page uses html and png only.
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=$HOME/dataset-page/work; mkdir -p $W
rsync -a --delete --exclude .git --exclude _output --exclude videos --exclude '*.pt' --exclude secret \
  --exclude cookies.txt --exclude config --exclude '__pycache__' --exclude 'mapping-*.csv' "$REPO/" $W/ || exit 1
# the code, README and detection counts (analysis_data/detection_counts.csv.gz: the YOLO tables and figures) from GitHub's main,
# which the review repo does not pull; only the mapping stays the review repo's (the Action ignores mapping-NAME too)
git -C "$REPO" fetch -q origin && git -C "$REPO" archive origin/main | \
  tar -xf - -C $W --exclude mapping.csv --exclude 'mapping-*.csv' || exit 1
# the history, for the credits of new videos (utils/analytics/contributors.py): the copy as a repository at GitHub's
# main, sharing the review repo's objects; only the unpushed reviews in mapping.csv show as changes
[ -d $W/.git ] || { git init -q $W && echo "$REPO/.git/objects" > $W/.git/objects/info/alternates; }
git -C $W fetch -q "$REPO" +refs/remotes/origin/main:refs/heads/crowd-main && git -C $W reset -q crowd-main || exit 1
cd $W || exit 1
export PATH=$HOME/micromamba/envs/dataset/bin:$PATH
python - <<'PY'
import json
c = json.load(open("default.config")); c.update(data=[], videos=[], save_eps=False)
json.dump(c, open("config", "w"), indent=2)
json.dump({k: "" for k in json.load(open("default.secret"))}, open("secret", "w"), indent=2)
PY
echo "== $(date '+%F %T') analysis"
python analysis.py 2>&1 | grep -vE "EPS format|pdftops" | tail -3
python "$REPO/dashboards/dataset_page.py"
