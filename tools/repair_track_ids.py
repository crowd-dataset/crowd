"""Repair the track ids (unique-id) of the detection CSV files that were written while segments ran in threads.

    python -m tools.repair_track_ids FOLDER [FOLDER ...]                 # dry run: only counts, writes the report
    python -m tools.repair_track_ids FOLDER --apply --report report.csv  # rewrite the files

Why. Ultralytics hands out track ids from one counter that every tracker of the process shares, and creating a tracker
sets that counter back to 0 (BaseTrack._count, BYTETracker.__init__ -> reset_id). With max_workers > 1 the segments run
in threads of one process, so ids were issued again that an earlier object of the same segment had used. One id then
stands for several objects, hundreds of seconds apart. The tracker itself forgets a lost object after track_buffer
frames (2 s), so an id that is silent for longer than that cannot be the same object.

What it does. In each file, every id whose rows are separated by a gap of more than track_buffer_sec * fps + 2 frames
is split: the first run of rows keeps the id, each later run gets a new id, numbered after the highest id of the file
in the order of the runs' first frames. Only the unique-id field of those rows changes; every other byte of the file is
kept. The result depends on the content of the file only, so identical files give identical results on any machine,
and running it again changes nothing. It cannot repair a collision between two objects that are visible at the same
time or within the gap (the report counts the frames where an id occurs twice).

Safety. A file that was changed in the last --min-age seconds is skipped (a job may still write it). Each repaired file
is written next to the original and moved over it in one step, and a file that changed while it was repaired is left
alone. For every changed file a small file in --patches (default FOLDER/../bbox_id_repair) lists the rows with their
old and new id (row, frame, old_id, new_id; gzipped), which is all that is needed to undo the repair (--undo). The
report has one row per file, with the SHA-256 of the file before and after, so that two copies can be compared.
"""
import argparse
import csv
import gzip
import hashlib
import io
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BBOX_NAME = re.compile(r"^(.+)_(\d+)_(\d+(?:\.\d+)?)\.csv$")
TRACK_BUFFER_SEC = 2.0  # "track_buffer_sec" of the project config
SLACK = 2  # frames: a lost track can still be matched in the frame where it is removed, and in the one after it
REPORT_FIELDS = ["file", "rows", "ids", "ids_split", "rows_changed", "frames_with_a_repeated_id", "status",
                 "sha256_before", "sha256_after"]


def gap_limit(fps, buffer_sec=TRACK_BUFFER_SEC):
    """The largest gap (in frames) between two rows of one id that the tracker can have bridged."""
    return int(buffer_sec * float(fps)) + SLACK


def new_ids(ids, frames, limit):
    """The id of every row after the repair, as an array like `ids`; also the number of ids that were split.
    Rows without a track (id < 0 or missing) are not touched."""
    ids = np.asarray(ids, dtype=float)
    frames = np.asarray(frames, dtype=np.int64)
    out = ids.copy()
    valid = np.where(np.isfinite(ids) & (ids >= 0))[0]
    if not len(valid):
        return out.astype(np.int64) if np.isfinite(out).all() else out, 0
    order = valid[np.lexsort((frames[valid], ids[valid]))]  # by id, then frame (stable for equal frames)
    sid, sfr = ids[order], frames[order]
    new_run = np.ones(len(order), dtype=bool)
    new_run[1:] = (sid[1:] != sid[:-1]) | ((sfr[1:] - sfr[:-1]) > limit)
    first_of_id = np.ones(len(order), dtype=bool)
    first_of_id[1:] = sid[1:] != sid[:-1]
    later = new_run & ~first_of_id  # the start of a second, third ... run of an id
    split_ids = len(np.unique(sid[later]))
    if not later.any():
        return out, 0
    run_no = np.cumsum(new_run) - 1  # a number for every run
    starts = np.where(later)[0]
    key = np.lexsort((starts, sid[starts], sfr[starts]))  # first frame, then id, then position
    label = np.full(run_no[-1] + 1, -1, dtype=np.int64)
    label[run_no[starts[key]]] = int(ids[valid].max()) + 1 + np.arange(len(starts))
    out[order] = np.where(label[run_no] >= 0, label[run_no], ids[order])
    return out, split_ids


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def repair_file(args):
    """Repair one file. args = (path, apply, min_age, patches_dir). Returns a report row (dict)."""
    path, apply, min_age, patches = args
    name = os.path.basename(path)
    row = {"file": name, "rows": 0, "ids": 0, "ids_split": 0, "rows_changed": 0, "frames_with_a_repeated_id": 0,
           "status": "", "sha256_before": "", "sha256_after": ""}
    try:
        st = os.stat(path)
        if time.time() - st.st_mtime < min_age:
            row["status"] = "skipped: changed in the last %d s" % min_age
            return row
        fps = BBOX_NAME.match(name).group(3)
        raw = Path(path).read_bytes()
        row["sha256_before"] = hashlib.sha256(raw).hexdigest()
        lines = raw.split(b"\n")
        header = lines[0].rstrip(b"\r").decode().split(",")
        if "unique-id" not in header or "frame-count" not in header:
            row["status"] = "skipped: no unique-id / frame-count column"
            return row
        c_id, c_fr = header.index("unique-id"), header.index("frame-count")
        body = lines[1:]
        rows = [i for i, ln in enumerate(body) if ln.strip()]
        row["rows"] = len(rows)
        table = pd.read_csv(io.BytesIO(raw), usecols=[c_id, c_fr], dtype=np.float64)
        ids = table.iloc[:, list(table.columns).index("unique-id")].to_numpy()
        frames = table.iloc[:, list(table.columns).index("frame-count")].to_numpy().astype(np.int64)
        assert len(ids) == len(rows), "blank or broken lines"
        good = np.isfinite(ids) & (ids >= 0)
        row["ids"] = len(np.unique(ids[good]))
        pairs = pd.Series(ids[good]).astype(np.int64).astype(str) + ":" + pd.Series(frames[good]).astype(str)
        row["frames_with_a_repeated_id"] = int(pairs.duplicated().sum())
        fixed, split = new_ids(ids, frames, gap_limit(fps))
        changed = np.where(good & (fixed != ids))[0]
        row["ids_split"], row["rows_changed"] = split, len(changed)
        if not len(changed):
            row["status"] = "ok: nothing to repair"
            row["sha256_after"] = row["sha256_before"]
            return row
        row["status"] = "to repair" if not apply else ""
        if not apply:
            return row
        patch = []
        for n in changed:
            i = rows[n]
            f = body[i].split(b",")
            f[c_id] = str(int(fixed[n])).encode()
            patch.append((i, int(frames[n]), int(ids[n]), int(fixed[n])))
            body[i] = b",".join(f)
        new = b"\n".join([lines[0], *body])
        tmp = Path(path).with_name("." + name + ".repairing")
        tmp.write_bytes(new)
        if os.stat(path).st_mtime != st.st_mtime or os.stat(path).st_size != st.st_size:
            tmp.unlink()
            row["status"] = "skipped: changed while it was repaired"
            return row
        Path(patches).mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(patches) / (name + ".changes.csv.gz"), "wt", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["row", "frame", "old_id", "new_id"])
            w.writerows(patch)
        os.replace(tmp, path)
        row["sha256_after"] = hashlib.sha256(new).hexdigest()
        row["status"] = "repaired"
    except Exception as error:  # report it, go on with the other files
        row["status"] = f"error: {type(error).__name__}: {error}"
    return row


def undo_file(args):
    """Put the old ids back, from the file of changes. Returns (file, status)."""
    path, patches = args
    name = os.path.basename(path)
    changes = Path(patches) / (name + ".changes.csv.gz")
    if not changes.is_file():
        return name, "no changes file"
    lines = Path(path).read_bytes().split(b"\n")
    c_id = lines[0].rstrip(b"\r").decode().split(",").index("unique-id")
    with gzip.open(changes, "rt", newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["row"]) + 1
            fields = lines[i].split(b",")
            if int(float(fields[c_id])) != int(r["new_id"]):
                return name, f"row {r['row']} is not as it was after the repair: not undone"
            fields[c_id] = r["old_id"].encode()
            lines[i] = b",".join(fields)
    tmp = Path(path).with_name("." + name + ".undoing")
    tmp.write_bytes(b"\n".join(lines))
    os.replace(tmp, path)
    return name, "undone"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", help="folders with the detection CSV files {video}_{start}_{fps}.csv")
    p.add_argument("--apply", action="store_true", help="rewrite the files (without it: a dry run)")
    p.add_argument("--undo", action="store_true", help="put the old ids back, from the files of changes")
    p.add_argument("--report", default="track_id_repair_report.csv")
    p.add_argument("--patches", help="folder for the files of changes (default: FOLDER/../bbox_id_repair)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-age", type=int, default=600, help="skip files changed in the last N seconds")
    p.add_argument("--limit", type=int, help="only the first N files (to try it)")
    a = p.parse_args()
    jobs = []
    for folder in a.folders:
        patches = a.patches or str(Path(folder).resolve().parent / "bbox_id_repair")
        with os.scandir(folder) as entries:
            names = sorted(e.name for e in entries if e.is_file() and BBOX_NAME.match(e.name))
        jobs += [(os.path.join(folder, n), patches) for n in names]
    jobs = jobs[:a.limit] if a.limit else jobs
    print(f"{len(jobs)} files", flush=True)
    with Pool(a.workers) as pool:
        if a.undo:
            for n, (name, status) in enumerate(pool.imap_unordered(undo_file, jobs, chunksize=4), 1):
                if status != "no changes file":
                    print(name, status)
            return
        work = [(path, a.apply, a.min_age, patches) for path, patches in jobs]
        with open(a.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, lineterminator="\n")
            w.writeheader()
            totals = {"rows": 0, "ids": 0, "ids_split": 0, "rows_changed": 0, "files_to_repair": 0, "errors": 0,
                      "repeated": 0}
            for n, r in enumerate(pool.imap_unordered(repair_file, work, chunksize=2), 1):
                w.writerow(r)
                totals["rows"] += r["rows"]
                totals["ids"] += r["ids"]
                totals["ids_split"] += r["ids_split"]
                totals["rows_changed"] += r["rows_changed"]
                totals["repeated"] += r["frames_with_a_repeated_id"]
                totals["files_to_repair"] += r["rows_changed"] > 0
                totals["errors"] += r["status"].startswith("error")
                if n % 500 == 0:
                    f.flush()
                    print(f"{n}/{len(work)} files, {totals}", flush=True)
    print("done", totals)


if __name__ == "__main__":
    sys.exit(main())
