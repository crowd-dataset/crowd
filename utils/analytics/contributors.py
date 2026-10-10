"""
Who added each video to the dataset, from the mapping file it first appeared in.

Each contributor adds videos to their own mapping file: mapping.csv is Pavlo's, and mapping-NAME.csv is NAME's (for
example mapping-olena.csv is Olena's; a trailing number is ignored), except files in FILE_OWNERS (mapping-epfl*.csv:
Shadab's). Videos added to
mapping.csv in commits from the account of another contributor (ACCOUNTS, e.g., Shadab) are credited to them. A video
is credited to the file it appeared in first; when it appeared in mapping.csv and another file in the same commit, it
is credited to mapping.csv, since contributors' files are refreshed with copies of mapping.csv.

The credits are kept in video_contributors.csv (video, contributor, added_utc: the time of the commit that added
the video, file: the mapping file it first appeared in). It is the record of who added what once the history of the
mapping files is gone, so keep it when rewriting the history:
- `python -m utils.analytics.contributors` builds it from the git history (needs the full history; run once);
- update(), run by analysis.py, credits videos that are new since, from the mapping files as they are now, with
  the time of the last commit that changed the file (or the current time while it has uncommitted changes).
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
FILE_OWNERS = {"epfl": "Shadab"}  # mapping-NAME.csv files kept by another contributor
# contributors who also commit to mapping.csv from their own account: a word in the commit author's name or email
ACCOUNTS = {"shadab": "Shadab", "fayefang": "Faye", "ying.fayeda": "Faye"}


def account_owner(author: str) -> str:
    """Contributor of videos added to mapping.csv in a commit by `author` ("name <email>"): Pavlo unless the author
    is in ACCOUNTS."""
    return next((who for key, who in ACCOUNTS.items() if key in author.lower()), MAIN)


def contributor(path: str) -> str | None:
    """mapping.csv -> Pavlo, mapping-olena.csv -> Olena, mapping-epfl2.csv -> Shadab; None for other files."""
    name = os.path.basename(path)
    if name == "mapping.csv":
        return MAIN
    m = re.fullmatch(r"mapping-([A-Za-z]+?)\d*\.csv", name)
    if not m:
        return None
    who = m.group(1)
    return FILE_OWNERS.get(who.lower(), who.capitalize())


def video_ids(text: str) -> set:
    """Video IDs in lines of a mapping file."""
    ids = set(VIDEO_URL.findall(text))
    for match in VIDEO_LIST.finditer(text):
        # an 11-letter name in another list (e.g., other names of a locality) is read as an ID too: harmless, as it
        # never matches a video in the dataset, while real IDs can be letters only (e.g., Gmfidadvlbo)
        ids.update(v for v in match.group(1).split(",") if v)
    return ids


def load() -> dict:
    """video -> (contributor, added_utc, file)."""
    if not os.path.exists(FILE):
        return {}
    with open(FILE, newline="", encoding="utf-8") as f:
        return {r["video"]: (r["contributor"], r["added_utc"], r.get("file", "")) for r in csv.DictReader(f)}


def save(credits: dict) -> None:
    tmp = FILE + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "contributor", "added_utc", "file"])
        for video, (who, when, path) in sorted(credits.items(), key=lambda kv: (kv[1][1], kv[0])):
            w.writerow([video, who, when, path])
    os.replace(tmp, FILE)


def backfill_from_git() -> dict:
    """Credit every video in the history of the mapping files to the file it first appeared in. Reads the history as
    a stream: commits that replace a whole mapping file have diffs of several MB."""
    proc = subprocess.Popen(
        ["git", "log", "--reverse", "--date-order", "-p", "--unified=0", "--no-color",
         "--format=@@COMMIT %ct %an <%ae>",
         "--", "mapping.csv", "mapping-*.csv"],
        cwd=common.root_dir, stdout=subprocess.PIPE, text=True, errors="replace")
    credits: dict = {}
    added: dict = {}  # (contributor, file) -> new video IDs in the current commit
    stamp, who, owner, path = None, None, MAIN, None

    def flush():
        # mapping.csv first: a video added to it and to a contributor's copy in the same commit is a copy; videos added
        # to mapping.csv go to the owner of the commit's account (Pavlo unless in ACCOUNTS)
        for w, f in sorted(added, key=lambda k: k[0] != MAIN):
            for video in added[(w, f)] - credits.keys():
                credits[video] = (owner if w == MAIN else w, stamp, f)
        added.clear()

    for line in proc.stdout:
        if line.startswith("@@COMMIT "):
            flush()
            _, timestamp, author = line.rstrip("\n").split(" ", 2)
            stamp = datetime.fromtimestamp(int(timestamp), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            owner = account_owner(author)
        elif line.startswith("diff --git "):
            path = line.rstrip("\n").split(" b/", 1)[-1]
            who = contributor(path)
        elif who and line.startswith("+") and not line.startswith("+++"):
            ids = video_ids(line)
            if ids:
                added.setdefault((who, path), set()).update(ids)
    flush()
    proc.wait()
    return credits


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=common.root_dir, capture_output=True, text=True).stdout.strip()


def _author_of_mapping() -> str:
    """Who added the latest changes to mapping.csv: the local git user while it has uncommitted changes, else the
    author of the last commit that changed it (also in a shallow clone, as in the GitHub Action)."""
    if _git("status", "--porcelain", "--", "mapping.csv"):
        return f"{_git('config', 'user.name')} <{_git('config', 'user.email')}>"
    return _git("log", "-1", "--format=%an <%ae>", "--", "mapping.csv")


def _added_time(path: str) -> str:
    """When the videos new in a mapping file were added: the time of the last commit that changed it, or now while
    it has uncommitted changes (or no commit: a file not in git)."""
    name = os.path.relpath(path, common.root_dir)
    if not _git("status", "--porcelain", "--", name):
        stamp = _git("log", "-1", "--format=%ct", "--", name)
        if stamp:
            return datetime.fromtimestamp(int(stamp), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def update() -> dict:
    """Credit videos not in video_contributors.csv yet, from the mapping files as they are now, and save. Returns
    video -> (contributor, added_utc, file)."""
    credits = load()
    files = {path: contributor(path) for path in glob.glob(os.path.join(common.root_dir, "mapping*.csv"))}
    current = []  # (contributor, file name, time it was added, video IDs)
    for path, who in files.items():
        if who:
            with open(path, encoding="utf-8", errors="replace") as f:
                current.append((who, os.path.basename(path), _added_time(path), video_ids(f.read())))
    owner = account_owner(_author_of_mapping())
    new = 0
    for who, name, when, ids in sorted(current, key=lambda c: c[0] != MAIN):  # in mapping.csv and a copy: credited
        for video in ids - credits.keys():                                    # for mapping.csv
            credits[video] = (owner if who == MAIN else who, when, name)
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
    assert contributor("mapping-epfl2.csv") == "Shadab" and contributor("mapping_remaining.csv") is None
    assert account_owner("MD SHADAB ALAM <88769183+Shaadalam9@users.noreply.github.com>") == "Shadab"
    assert account_owner("Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>") == "Pavlo"
    assert account_owner("FayeFang-creator <ying.fayeda@gmail.com>") == "Faye"
    assert contributor("mapping-faye.csv") == "Faye" and video_ids("x,https://www.youtube.com/watch?v=G1I_PlmL_YA,0") \
        == {"G1I_PlmL_YA"}
    assert video_ids("1,A,[abcdEFGH123,ZYXW-_98765],[Gmfidadvlbo],[UC20vyRWEaC2GIS8DkmeRQaA]") == {
        "abcdEFGH123", "ZYXW-_98765", "Gmfidadvlbo"}
    credits = backfill_from_git()
    save(credits)
    counts: dict = {}
    for who, *_ in credits.values():
        counts[who] = counts.get(who, 0) + 1
    print(f"{len(credits):,} videos credited:", ", ".join(f"{w} {n:,}" for w, n in sorted(counts.items(),
                                                                                          key=lambda kv: -kv[1])))
