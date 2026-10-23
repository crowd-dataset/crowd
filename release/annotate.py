"""Local web tool for the manual audits of CROWD (the samples of release.sample_audit).

    python -m release.annotate --videos /disk1/videos /disk2/videos --bbox /disk1/data/bbox /disk2/data/bbox
    python -m release.annotate --config /path/to/config     # the "videos" and "data" folders of the project config

then open http://127.0.0.1:8773. Three tasks:

    labels   re-watch a segment on YouTube and record whether its boundaries are right and what its time of day and
             platform are (the dataset's own labels are hidden, so they cannot bias the answer)
    frames   draw boxes around the people, bicycles, cars, motorcycles, buses and trucks in a frame (taken from the
             video file, so the videos must be on this machine; a missing one can be downloaded with yt-dlp)
    clips    play a 10 s clip with the dataset's tracks drawn over it and count the tracking errors

Progress is saved after every item, so the tool can be stopped and resumed. Results are written to the output folder:
audit_labels_filled.csv (for release.audit_stats), annotations.csv (for release.evaluate_detections) and
clip_taxonomy.csv. Only the local machine can reach the tool (127.0.0.1).
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder: release
from release.evaluate_detections import CLASSES as COCO_CLASSES  # noqa: E402
from release.make_crowd_jsonl import VEHICLE_TYPE_NAMES  # noqa: E402

CLASS_NAMES = list(COCO_CLASSES.values())
PLATFORMS = [v for k, v in sorted(VEHICLE_TYPE_NAMES.items(), key=lambda kv: str(kv[0])) if isinstance(k, int)]
CLIP_S = 10
THRESHOLD = 0.7  # the confidence the analyses of the dataset use
LABEL_FIELDS = ["boundary_ok", "correct_start_s", "correct_end_s", "tod_audit", "platform_audit", "notes"]
CLIP_FIELDS = ["n_objects", "missed_objects", "false_tracks", "class_confusions", "id_switches", "fragmentations",
               "notes"]
ANNOTATION_FIELDS = ["sample_id", "class", "x_center", "y_center", "width", "height", "frame_k", "mode", "annotator"]

app = Flask(__name__)
CFG = {}
LOCK = threading.Lock()
TASKS = {}  # task -> list of sample rows (dicts)
STATE = {}  # task -> id -> saved answer
INFO = {}   # video id -> {fps, width, height} read from the file
MISSING = []  # sample files that were not found


# ------------------------------------------------------------------------------------------------ files
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def item_id(task, row):
    return row["segment_id"] if task == "labels" else row["sample_id"]


def load(samples, out):
    """Read the three sample files and the saved state."""
    CFG["out"] = Path(out)
    CFG["out"].mkdir(parents=True, exist_ok=True)
    for task, name in (("labels", "audit_labels.csv"), ("frames", "audit_frames.csv"), ("clips", "audit_clips.csv")):
        path = Path(samples) / name
        TASKS[task] = read_csv(path) if path.is_file() else []
        if not path.is_file():
            MISSING.append(str(path.resolve()))
        state_file = CFG["out"] / f"{task}.json"
        STATE[task] = json.load(open(state_file, encoding="utf-8")) if state_file.is_file() else {}


def save_state(task):
    """Write the state of a task and the derived CSV file (atomically)."""
    tmp = CFG["out"] / f"{task}.json.tmp"
    json.dump(STATE[task], open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, CFG["out"] / f"{task}.json")
    rows = TASKS[task]
    if task == "labels":
        out = []
        for r in rows:
            s = STATE["labels"].get(r["segment_id"], {})
            filled = {k: (s.get(k, "") if s.get("status") == "done" else "") for k in LABEL_FIELDS}
            out.append({**r, **filled, "annotator": s.get("annotator", ""), "status": s.get("status", "")})
        fields = list(rows[0]) + ["annotator", "status"] if rows else []
        write_csv(CFG["out"] / "audit_labels_filled.csv", fields, out)
    elif task == "frames":
        out = []
        for r in rows:
            s = STATE["frames"].get(r["sample_id"])
            if not s or s.get("status") != "done":
                continue
            base = {"sample_id": r["sample_id"], "frame_k": s.get("k", ""), "mode": s.get("mode", "blind"),
                    "annotator": s.get("annotator", "")}
            boxes = s.get("boxes") or [{"c": "none", "x": "", "y": "", "w": "", "h": ""}]
            out += [{**base, "class": b["c"], "x_center": b["x"], "y_center": b["y"], "width": b["w"],
                     "height": b["h"]} for b in boxes]
        write_csv(CFG["out"] / "annotations.csv", ANNOTATION_FIELDS, out)
    else:
        out = []
        for r in rows:
            s = STATE["clips"].get(r["sample_id"])
            if s and s.get("status") == "done":
                out.append({"sample_id": r["sample_id"], "continent": r["continent"], "tod": r["tod"],
                            "annotator": s.get("annotator", ""), **{k: s.get(k, "") for k in CLIP_FIELDS}})
        write_csv(CFG["out"] / "clip_taxonomy.csv", ["sample_id", "continent", "tod", "annotator", *CLIP_FIELDS], out)


# ------------------------------------------------------------------------------------------------ videos
def video_path(vid):
    """The local file of a video, or None."""
    for d in [*CFG.get("videos", []), CFG["cache"]]:
        for ext in ("mp4", "mkv", "webm"):
            p = Path(d) / f"{vid}.{ext}"
            if p.is_file():
                return p
    return None


def video_info(vid):
    """fps, width and height read from the file (the true frame rate, not the rounded one of the file names)."""
    if vid not in INFO:
        path = video_path(vid)
        if path is None:
            return None
        cap = cv2.VideoCapture(str(path))
        INFO[vid] = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
    return INFO[vid]


def frame_number(row, fps):
    """The 1-based frame number of the sample within its segment (as in the detection file)."""
    t = float(row["timestamp_s"] if "timestamp_s" in row else row["clip_start_s"])
    return round((t - float(row["start_s"])) * fps) + 1


BBOX_NAME = re.compile(r"^(.+)_(\d+)_(\d+(?:\.\d+)?)\.csv$")
BBOX = {"index": None, "time": 0.0}


def scan_bbox():
    """Index the detection files of all folders (and of their bbox subfolders) by (video id, start second)."""
    index = {}
    for d in CFG.get("bbox", []):
        for folder in (Path(d), Path(d) / "bbox"):
            if not folder.is_dir():
                continue
            with os.scandir(folder) as entries:
                for e in entries:
                    m = BBOX_NAME.match(e.name)
                    if m and e.is_file():
                        index.setdefault((m.group(1), int(m.group(2))), e.path)
    BBOX.update(index=index, time=time.time())
    return index


def detection_file(row):
    """The detection file of a sample's segment in any of the folders, or None (the folders are rescanned at most
    once a minute when a file is missing, as files can be added while the tool runs)."""
    key = (row["video_id"], int(float(row["start_s"])))
    index = BBOX["index"] if BBOX["index"] is not None else scan_bbox()
    if key not in index and time.time() - BBOX["time"] > 60:
        index = scan_bbox()
    return index.get(key)


def detections(row, k0, k1, threshold=THRESHOLD):
    """{frame number: [[track id, class, x, y, w, h, confidence], ...]} of a detection file for frames k0..k1."""
    path = detection_file(row)
    if path is None:
        return None
    cols = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
    df = pd.read_csv(path, usecols=cols)[cols]
    df = df[(df["frame-count"] >= k0) & (df["frame-count"] <= k1) & (df["confidence"] >= threshold)
            & df["yolo-id"].isin(COCO_CLASSES)]
    out = {}
    for r in df.itertuples(index=False):
        out.setdefault(int(r[7]), []).append([int(r[5]) if r[5] == r[5] else -1, COCO_CLASSES[int(r[0])],
                                              round(r[1], 4), round(r[2], 4), round(r[3], 4), round(r[4], 4),
                                              round(r[6], 3)])
    return out


def sample(task, sid):
    for r in TASKS[task]:
        if item_id(task, r) == sid:
            return r
    abort(404)


# ------------------------------------------------------------------------------------------------ API
@app.get("/api/<task>")
def api_items(task):
    if task not in TASKS:
        abort(404)
    items = []
    for r in TASKS[task]:
        sid = item_id(task, r)
        items.append({"id": sid, "status": STATE[task].get(sid, {}).get("status", "todo"), "continent": r["continent"],
                      "tod": r["tod"]})
    return jsonify(items)


@app.get("/api/<task>/<sid>")
def api_item(task, sid):
    if task not in TASKS:
        abort(404)
    r = sample(task, sid)
    saved = STATE[task].get(sid, {})
    if task == "labels":
        hidden = {"platform", "tod", "continent"}  # the dataset's labels stay hidden while the auditor decides
        data = {k: v for k, v in r.items() if k not in hidden and k not in LABEL_FIELDS and k != "contributor"}
        data["contributor"] = r["contributor"]
        return jsonify({"row": data, "saved": saved, "platforms": PLATFORMS, "annotator": CFG.get("annotator", "")})
    info = video_info(r["video_id"])
    meta = {"row": r, "saved": saved, "available": info is not None, "classes": CLASS_NAMES}
    if info:
        k = frame_number(r, info["fps"])
        meta.update({"fps": info["fps"], "width": info["width"], "height": info["height"], "k": k})
        if task == "clips":
            k1 = k + round(CLIP_S * info["fps"])
            meta.update({"k1": k1, "clip_start": float(r["clip_start_s"]), "segment_start": float(r["start_s"]),
                         "tracks": detections(r, k, k1)})
    return jsonify(meta)


@app.post("/api/<task>/<sid>")
def api_save(task, sid):
    if task not in TASKS:
        abort(404)
    sample(task, sid)
    data = request.get_json(force=True)
    with LOCK:
        STATE[task][sid] = {**data, "annotator": CFG.get("annotator", "")}
        save_state(task)
    return jsonify({"ok": True})


@app.get("/api/frames/<sid>/yolo")
def api_yolo(sid):
    """The dataset's boxes of the frame (confidence >= 0.7) as a starting point (pre-label mode)."""
    r = sample("frames", sid)
    info = video_info(r["video_id"])
    if info is None:
        return jsonify([])
    k = frame_number(r, info["fps"])
    found = (detections(r, k, k) or {}).get(k, [])
    return jsonify([{"c": c, "x": x, "y": y, "w": w, "h": h} for _, c, x, y, w, h, _ in found])


@app.get("/frame/<sid>.jpg")
def frame(sid):
    r = sample("frames", sid)
    cache = Path(CFG["cache"]) / "frames" / f"{sid}.jpg"
    if not cache.is_file():
        info = video_info(r["video_id"])
        if info is None:
            abort(404)
        index = round(float(r["start_s"]) * info["fps"]) + frame_number(r, info["fps"]) - 1
        cap = cv2.VideoCapture(str(video_path(r["video_id"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
        cap.release()
        if not ok:
            abort(404)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return send_file(cache, mimetype="image/jpeg")


@app.get("/video/<vid>")
def video(vid):
    path = video_path(vid)
    if path is None:
        abort(404)
    return send_file(path, conditional=True)  # conditional: the browser seeks with range requests


@app.post("/api/download/<vid>")
def api_download(vid):
    """Fetch a missing video (at most 720p) into the cache folder with yt-dlp."""
    if video_path(vid):
        return jsonify({"ok": True})
    Path(CFG["cache"]).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/bv*[height<=720]",
           "-o", str(Path(CFG["cache"]) / f"{vid}.%(ext)s"), "--no-playlist", f"https://www.youtube.com/watch?v={vid}"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    ok = done.returncode == 0 and video_path(vid) is not None
    return jsonify({"ok": ok, "error": "" if ok else (done.stderr or done.stdout)[-400:]})


def count(task, status):
    return sum(1 for s in STATE[task].values() if s.get("status") == status)


@app.get("/")
def index():
    counts = "".join(
        f'<li><a href="/{t}">{t}</a>: {count(t, "done")} of {len(TASKS[t])} done, {count(t, "skipped")} skipped</li>'
        for t in TASKS)
    items = "".join(f"<li>{m}</li>" for m in MISSING)
    warn = (
        '<div class="panel" style="border-color:#b00020;color:#b00020"><b>Sample files not found</b>, so there is '
        "nothing to work on. Copy audit_labels.csv, audit_frames.csv and audit_clips.csv to the folder below, or "
        f"start the tool with --samples FOLDER (the folder with the three files):<ul>{items}</ul></div>"
    ) if MISSING else ""
    return page("CROWD audits", f"{warn}<h1>CROWD audits</h1>"
                                f"<p>Annotator: <b>{CFG.get('annotator') or 'not set'}</b></p>"
                                f"<ul>{counts}</ul><p>Results: <code>{CFG['out']}</code></p>")


# ------------------------------------------------------------------------------------------------ pages
UI = Path(__file__).parent / "annotate_ui"  # the HTML, JavaScript and CSS of the pages


def asset(name):
    return (UI / name).read_text(encoding="utf-8")


def page(title, body, script=""):
    return Response(f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>{asset("style.css")}</style></head><body><nav><a href="/">CROWD audits</a><a href="/labels">labels</a>
<a href="/frames">frames</a><a href="/clips">clips</a></nav><main>{body}</main>
<script>{asset("common.js")}{script}</script></body></html>""", mimetype="text/html")


def task_page(task, title):
    return page(title, asset("head.html") + asset(f"{task}.html"), asset(f"{task}.js"))


@app.get("/labels")
def labels_page():
    return task_page("labels", "Label audit")


@app.get("/frames")
def frames_page():
    return task_page("frames", "Box annotation")


@app.get("/clips")
def clips_page():
    return task_page("clips", "Tracking errors")


def report():
    """Say how many of the samples can be worked on with the folders given."""
    for task in ("frames", "clips"):
        rows = TASKS[task]
        have = sum(1 for r in rows if video_path(r["video_id"]))
        tracks = sum(1 for r in rows if detection_file(r)) if task == "clips" else None
        print(f"{task}: {len(rows)} samples, video found for {have}"
              + (f", detection file found for {tracks}" if tracks is not None else "")
              + (" (a missing video can be downloaded from the page)" if have < len(rows) else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="paper/response", help="folder with audit_labels.csv, audit_frames.csv, "
                                                               "audit_clips.csv")
    p.add_argument("--out", default="paper/response/annotations", help="folder for the saved results")
    p.add_argument("--videos", nargs="*", default=[], help="any number of folders with {video_id}.mp4")
    p.add_argument("--bbox", nargs="*", default=[], help="any number of folders with the detection CSV files")
    p.add_argument("--config", help="the project config file: its videos and data folders are added (data/bbox)")
    p.add_argument("--cache", default="_output/annotation_cache", help="folder for extracted frames and downloads")
    p.add_argument("--annotator", default=os.environ.get("USER", ""), help="your name, saved with every answer")
    p.add_argument("--port", type=int, default=8773)
    a = p.parse_args()
    videos, bbox = list(a.videos), list(a.bbox)
    if a.config:
        conf = json.load(open(os.path.expanduser(a.config), encoding="utf-8"))
        videos += conf.get("videos") or []
        bbox += [os.path.join(d, "bbox") for d in conf.get("data") or []]
    videos, bbox = [os.path.expanduser(d) for d in videos], [os.path.expanduser(d) for d in bbox]
    CFG.update(videos=videos, bbox=bbox, cache=a.cache, annotator=a.annotator)
    for d in videos + bbox:
        if not os.path.isdir(d):
            print(f"warning: folder not found: {d}")
    load(a.samples, a.out)
    for m in MISSING:
        print(f"warning: sample file not found: {m}\n  (started in {os.getcwd()}; use --samples FOLDER)")
    report()
    print(f"Open http://127.0.0.1:{a.port}  (results in {a.out})")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
