"""
Who added each video to the dataset, from the mapping file it first appeared in.

Each contributor adds videos to their own mapping file: mapping.csv is Pavlo's, and mapping-NAME.csv is NAME's (for
example mapping-olena.csv is Olena's; a trailing number is ignored, so mapping-epfl2.csv is EPFL's). A video is
credited to the file it appeared in first; when it appeared in mapping.csv and another file in the same commit, it is
credited to mapping.csv, since contributors' files are refreshed with copies of mapping.csv.

The credits are kept in video_contributors.csv (video, contributor, added_utc):
- `python -m utils.analytics.contributors` builds it from the git history (needs the full history; run once);
- update(), run by analysis.py, credits videos that are new since, from the mapping files as they are now.
"""

import csv
import glob
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import common

FILE = os.path.join(common.root_dir, "video_contributors.csv")
MAIN = "Pavlo"  # contributor of mapping.csv
# a bracketed, comma-separated list of 11-character YouTube video IDs, e.g. [abcdEFGH123,ZYXW-_98765]
VIDEO_LIST = re.compile(r"\[((?:[A-Za-z0-9_-]{11},?)+)\]")
VIDEO_URL = re.compile(r"(?:watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})")  # the first versions stored YouTube links
ACRONYMS = {"epfl"}


def contributor(path: str) -> str | None:
    """mapping.csv -> Pavlo, mapping-olena.csv -> Olena, mapping-epfl2.csv -> EPFL; None for other files."""
    name = os.path.basename(path)
    if name == "mapping.csv":
        return MAIN
    m = re.fullmatch(r"mapping-([A-Za-z]+?)\d*\.csv", name)
    if not m:
        return None
    who = m.group(1)
    return who.upper() if who.lower() in ACRONYMS else who.capitalize()  # EPFL, Olena


def video_ids(text: str) -> set:
    """Video IDs in lines of a mapping file."""
    ids = set(VIDEO_URL.findall(text))
    for match in VIDEO_LIST.finditer(text):
        # an 11-letter name in another list (e.g., other names of a locality) is read as an ID too: harmless, as it
        # never matches a video in the dataset, while real IDs can be letters only (e.g., Gmfidadvlbo)
        ids.update(v for v in match.group(1).split(",") if v)
    return ids


def load() -> dict:
    """video -> (contributor, added_utc)."""
    if not os.path.exists(FILE):
        return {}
    with open(FILE, newline="", encoding="utf-8") as f:
        return {r["video"]: (r["contributor"], r["added_utc"]) for r in csv.DictReader(f)}


def save(credits: dict) -> None:
    with open(FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "contributor", "added_utc"])
        for video, (who, when) in sorted(credits.items(), key=lambda kv: (kv[1][1], kv[0])):
            w.writerow([video, who, when])


def backfill_from_git() -> dict:
    """Credit every video in the history of the mapping files to the file it first appeared in. Reads the history as
    a stream: commits that replace a whole mapping file have diffs of several MB."""
    proc = subprocess.Popen(
        ["git", "log", "--reverse", "--date-order", "-p", "--unified=0", "--no-color", "--format=@@COMMIT %ct",
         "--", "mapping.csv", "mapping-*.csv"],
        cwd=common.root_dir, stdout=subprocess.PIPE, text=True, errors="replace")
    credits: dict = {}
    added: dict = {}  # contributor -> new video IDs in the current commit
    stamp, who = None, None

    def flush():
        # mapping.csv first: a video added to it and to a contributor's copy in the same commit is Pavlo's
        for w in sorted(added, key=lambda c: c != MAIN):
            for video in added[w] - credits.keys():
                credits[video] = (w, stamp)
        added.clear()

    for line in proc.stdout:
        if line.startswith("@@COMMIT "):
            flush()
            stamp = datetime.fromtimestamp(int(line.split()[1]), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif line.startswith("diff --git "):
            who = contributor(line.rstrip("\n").split(" b/", 1)[-1])
        elif who and line.startswith("+") and not line.startswith("+++"):
            ids = video_ids(line)
            if ids:
                added.setdefault(who, set()).update(ids)
    flush()
    proc.wait()
    return credits


def update() -> dict:
    """Credit videos not in video_contributors.csv yet, from the mapping files as they are now, and save. Returns
    video -> (contributor, added_utc)."""
    credits = load()
    files = {path: contributor(path) for path in glob.glob(os.path.join(common.root_dir, "mapping*.csv"))}
    current = {}
    for path, who in files.items():
        if who:
            with open(path, encoding="utf-8", errors="replace") as f:
                current[who] = video_ids(f.read())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = 0
    for who in sorted(current, key=lambda w: w != MAIN):  # in mapping.csv and a copy: Pavlo's
        for video in current[who] - credits.keys():
            credits[video] = (who, now)
            new += 1
    if new or not os.path.exists(FILE):
        save(credits)
    return credits


if __name__ == "__main__":
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=common.root_dir,
                             capture_output=True, text=True).stdout.strip()
    if shallow == "true":
        sys.exit("The repository has no full history (shallow clone): run `git fetch --unshallow` first.")
    assert contributor("mapping.csv") == "Pavlo" and contributor("x/mapping-olena.csv") == "Olena"
    assert contributor("mapping-epfl2.csv") == "EPFL" and contributor("mapping_remaining.csv") is None
    assert contributor("mapping-faye.csv") == "Faye" and video_ids("x,https://www.youtube.com/watch?v=G1I_PlmL_YA,0") \
        == {"G1I_PlmL_YA"}
    assert video_ids("1,A,[abcdEFGH123,ZYXW-_98765],[Gmfidadvlbo],[UC20vyRWEaC2GIS8DkmeRQaA]") == {
        "abcdEFGH123", "ZYXW-_98765", "Gmfidadvlbo"}
    credits = backfill_from_git()
    save(credits)
    counts: dict = {}
    for who, _ in credits.values():
        counts[who] = counts.get(who, 0) + 1
    print(f"{len(credits):,} videos credited:", ", ".join(f"{w} {n:,}" for w, n in sorted(counts.items(),
                                                                                          key=lambda kv: -kv[1])))
