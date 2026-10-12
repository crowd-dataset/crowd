#!/bin/bash
# Weekly metadata update: refresh mapping_metadata.csv (update_info.py) and crowd_index.jsonl
# (make_crowd_jsonl.py), split either into NAME_partN files when over 99 MB, commit and push.
# Runs from cron on the mini (Mondays 02:00) in a clone used only by this job:
#   0 2 * * 1 $HOME/repos/crowd/weekly_metadata.sh >> $HOME/repos/crowd-metadata.log 2>&1
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")"
PY=${PY:-$HOME/micromamba/envs/crowd/bin/python}
MAX_BYTES=${SPLIT_MAX_BYTES:-99000000}  # also read by split_large_files.py
FILES=(mapping_metadata.csv crowd_index.jsonl)

LOCK=/tmp/crowd-weekly-metadata.lock
mkdir "$LOCK" 2>/dev/null || { echo "$(date '+%F %T') already running"; exit 0; }
trap 'rmdir "$LOCK"' EXIT
echo "== $(date '+%F %T') start"

git fetch -q --depth 1 origin main
git reset -q --hard origin/main  # nothing local to keep: this clone only runs this job
git gc -q --prune=now
[ -f config ] || cp default.config config

"$PY" update_info.py < /dev/null
"$PY" make_crowd_jsonl.py

for f in "${FILES[@]}"; do
    "$PY" split_large_files.py split "$f"
    stem=${f%.*} ext=${f##*.}
    tracked_parts=$(git ls-files -- "${stem}_part*.${ext}")
    [ -z "$tracked_parts" ] || git rm -q --cached -- $tracked_parts  # current ones are added again below
    parts=("${stem}"_part*."$ext")
    if [ ${#parts[@]} -gt 0 ]; then
        git add -f -- "${parts[@]}"
        git rm -q --cached --ignore-unmatch -- "$f"
    else
        git add -f -- "$f"
    fi
done

too_big=""
for p in $(git diff --cached --name-only --diff-filter=AM); do
    [ "$(wc -c < "$p")" -le $MAX_BYTES ] || too_big="$too_big $p"
done
if [ -n "$too_big" ]; then
    echo "not pushing, over $MAX_BYTES bytes:$too_big"
    git reset -q
    exit 1
fi

if git diff --cached --quiet; then
    echo "no changes"
    exit 0
fi
git commit -q -m "Weekly update of video metadata"

for attempt in 1 2 3; do
    if git push -q origin HEAD:main; then
        echo "== $(date '+%F %T') pushed $(git rev-parse --short HEAD)"
        exit 0
    fi
    # someone pushed meanwhile: replay this commit on top, keeping its version of conflicting lines
    git fetch -q origin main
    git rebase -q -X theirs origin/main || { git rebase --abort; break; }
done
echo "push failed"
exit 1
