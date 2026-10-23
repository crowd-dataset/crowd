"""Precision and recall of the released YOLO11x detections on the audit frames, from a manual review or manual boxes.

    python -m release.evaluate_detections box_review.csv --frames audit_frames.csv --out detection_eval.csv
    python -m release.evaluate_detections annotations.csv --frames audit_frames.csv \
        --bbox /disk1/data/bbox /disk2/data/bbox --out detection_eval.csv

Two kinds of input, told apart by the header of the file:

REVIEW (box_review.csv of release.annotate, the "review" page). The reviewer saw the detector's boxes
(confidence >= 0.7) and marked each as correct, a false positive or the wrong class (with the true class), and drew
the objects it missed. A correct box is a true positive; a false positive and a wrong class are false positives of the
predicted class; a missed object and a wrong class are false negatives of the true class. Precision, recall and F1
are given per class, overall and per time of day and continent, with a 95% bootstrap interval over frames. No AP is
possible, since only the boxes above the threshold were shown. Seeing the detector's boxes can make a reviewer accept
them too readily, so the precision can be optimistic: check it on a subset with the manual boxes below.

MANUAL BOXES (annotations.csv of the "draw" page, boxes drawn without seeing the detector).
annotations.csv has one row per manual box: sample_id (as in audit_frames.csv), class (person, bicycle, car,
motorcycle, bus or truck; "none" marks an annotated frame without any of them), x_center, y_center, width, height
(normalised to the frame, as in the detection files). A frame without a row is not annotated and is skipped.

The prediction for a frame is read from the detection file {video}_{start}_{fps}.csv of its segment: frame number
round((timestamp_s - start_s) * fps) + 1. Predictions are matched to the boxes of the same class, in order of
confidence, at IoU >= 0.5. Precision, recall and F1 are given at the confidence threshold that the analyses of
the dataset use (0.7, THRESHOLD) and AP50 over all confidences, overall and per time of day, continent and class,
with a 95% bootstrap interval over frames. The classes of the COCO index are in CLASSES.
"""
import argparse
import csv
import glob
import os
import random
import re
import sys
from collections import defaultdict

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
THRESHOLD = 0.7
IOU = 0.5
BOOTSTRAP = 1000
SEED = 2026


def iou(a, b):
    """IoU of two boxes (x_center, y_center, width, height)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr=IOU):
    """Greedy matching in order of confidence; preds are (confidence, box), gts boxes of one class in one frame.
    Returns [(confidence, is_true_positive)] for every prediction."""
    free = list(gts)
    out = []
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        best, best_iou = None, thr
        for i, g in enumerate(free):
            v = iou(box, g)
            if v >= best_iou:
                best, best_iou = i, v
        if best is not None:
            free.pop(best)
        out.append((conf, best is not None))
    return out


def average_precision(scored, n_gt):
    """AP (all-point interpolation) of [(confidence, is_tp)] over n_gt ground-truth boxes."""
    if n_gt == 0:
        return float("nan")
    tp = fp = 0
    points = []
    for _, hit in sorted(scored, key=lambda s: -s[0]):
        tp, fp = tp + hit, fp + (not hit)
        points.append((tp / n_gt, tp / (tp + fp)))
    ap, prev_recall = 0.0, 0.0
    for i, (recall, _) in enumerate(points):
        precision = max(p for _, p in points[i:])
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def frame_counts(scored, n_gt, threshold=THRESHOLD):
    """(true positives, false positives, ground-truth boxes) of one frame and class at a confidence threshold."""
    kept = [hit for conf, hit in scored if conf >= threshold]
    return sum(kept), len(kept) - sum(kept), n_gt


def prf(counts):
    """(precision, recall, F1) from a list of (tp, fp, n_gt) of frames."""
    tp, fp, n_gt = (sum(c[i] for c in counts) for i in range(3))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / n_gt if n_gt else float("nan")
    f = 2 * p * r / (p + r) if tp else (0.0 if n_gt else float("nan"))
    return p, r, f


def bootstrap(counts, rng, n=BOOTSTRAP):
    """95% interval of precision, recall and F1, resampling frames."""
    runs = [prf([counts[rng.randrange(len(counts))] for _ in counts]) for _ in range(n)]
    out = []
    for i in range(3):
        v = sorted(x[i] for x in runs if x[i] == x[i])
        out.append((v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float("nan"),) * 2)
    return out


def review_counts(rows):
    """{sample_id: {class: (true positives, false positives, ground-truth boxes)}} from box_review.csv rows."""
    per = {}
    for r in rows:
        sid = r["sample_id"]
        per.setdefault(sid, {c: [0, 0, 0] for c in CLASSES.values()})
        if r["kind"] == "pred":
            c, v, t = r["class"], r["verdict"], r.get("true_class", "")
            if v == "correct":
                per[sid][c][0] += 1
                per[sid][c][2] += 1
            else:
                per[sid][c][1] += 1
                if v == "wrong_class" and t in per[sid]:
                    per[sid][t][2] += 1  # the object was there, as another class: a miss for that class
        elif r["kind"] == "missed" and r["class"] in per[sid]:
            per[sid][r["class"]][2] += 1
    return {sid: {c: tuple(v) for c, v in d.items()} for sid, d in per.items()}


def summarise_counts(frames, per, groupers):
    """Rows of precision, recall and F1 (no AP) from per-frame, per-class counts, by group."""
    rng = random.Random(SEED)
    rows = []
    for gname, key in groupers:
        members = defaultdict(list)
        for sid in per:
            members[key(frames[sid])].append(sid)
        for value, sids in sorted(members.items()):
            for cls in (*CLASSES.values(), "all"):
                use = list(CLASSES.values()) if cls == "all" else [cls]
                counts = [tuple(map(sum, zip(*(per[s][c] for c in use)))) for s in sids]
                n_gt = sum(c[2] for c in counts)
                if n_gt == 0 and sum(c[1] for c in counts) == 0:  # no object and no detection of the class
                    continue
                (p, r, f), ci = prf(counts), bootstrap(counts, rng)
                rows.append({"group": gname, "value": value, "class": cls, "frames": len(sids), "boxes": n_gt,
                             "precision": p, "precision_lo": ci[0][0], "precision_hi": ci[0][1],
                             "recall": r, "recall_lo": ci[1][0], "recall_hi": ci[1][1], "f1": f,
                             "f1_lo": ci[2][0], "f1_hi": ci[2][1]})
    return rows


def predictions(row, bbox_dirs):
    """[(class name, confidence, box)] of the detection file frame that a sample row points to. bbox_dirs is a folder
    or a list of folders (each is searched, and its bbox subfolder); None if no folder has the file."""
    pattern = f"{row['video_id']}_{int(float(row['start_s']))}_*.csv"
    dirs = [bbox_dirs] if isinstance(bbox_dirs, str) else bbox_dirs
    files = [f for d in dirs for sub in (d, os.path.join(d, "bbox")) for f in glob.glob(os.path.join(sub, pattern))]
    if not files:
        return None
    if row.get("frame_k"):  # the frame number the annotation tool took the frame at (with the true frame rate)
        frame = int(float(row["frame_k"]))
    else:
        fps = float(re.search(r"_(\d+(?:\.\d+)?)\.csv$", files[0]).group(1))
        frame = round((float(row["timestamp_s"]) - float(row["start_s"])) * fps) + 1
    out = []
    with open(files[0], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["frame-count"]) == frame and int(float(r["yolo-id"])) in CLASSES:
                box = tuple(float(r[k]) for k in ("x-center", "y-center", "width", "height"))
                out.append((CLASSES[int(float(r["yolo-id"]))], float(r["confidence"]), box))
    return out


def evaluate(frames, gts, preds, groupers):
    """Rows of metrics. frames: sample_id -> row dict; gts, preds: sample_id -> [(class, ..., box)]."""
    per = {}  # sample_id -> class -> (scored, n_gt)
    for sid in frames:
        if sid not in gts or preds.get(sid) is None:
            continue
        per[sid] = {}
        for cls in CLASSES.values():
            g = [b for c, b in gts[sid] if c == cls]
            p = [(conf, box) for c, conf, box in preds[sid] if c == cls]
            per[sid][cls] = (match(p, g), len(g))
    rng = random.Random(SEED)
    rows = []
    for gname, key in groupers:
        members = defaultdict(list)
        for sid in per:
            members[key(frames[sid])].append(sid)
        for value, sids in sorted(members.items()):
            for cls in (*CLASSES.values(), "all"):
                use = CLASSES.values() if cls == "all" else [cls]
                counts = [tuple(map(sum, zip(*(frame_counts(*per[s][c]) for c in use)))) for s in sids]
                scored = [x for s in sids for c in use for x in per[s][c][0]]
                n_gt = sum(per[s][c][1] for s in sids for c in use)
                if n_gt == 0:
                    continue
                (p, r, f), ci = prf(counts), bootstrap(counts, rng)
                rows.append({"group": gname, "value": value, "class": cls, "frames": len(sids), "boxes": n_gt,
                             "precision": p, "precision_lo": ci[0][0], "precision_hi": ci[0][1],
                             "recall": r, "recall_lo": ci[1][0], "recall_hi": ci[1][1], "f1": f,
                             "f1_lo": ci[2][0], "f1_hi": ci[2][1], "ap50": average_precision(scored, n_gt)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("annotations", help="annotations.csv written by release.annotate (or made by hand)")
    p.add_argument("--frames", default="paper/response/audit_frames.csv", help="the sample of frames")
    p.add_argument("--bbox", nargs="+", default=[], help="any number of folders with the detection CSV files "
                                                         "(only for manual boxes: a review has the boxes in its file)")
    p.add_argument("--out", default="detection_eval.csv")
    a = p.parse_args()
    bbox_dirs = [os.path.expanduser(d) for d in a.bbox]
    with open(a.frames, newline="", encoding="utf-8") as f:
        frames = {r["sample_id"]: r for r in csv.DictReader(f)}
    groupers = [("all", lambda r: "all"), ("tod", lambda r: r["tod"]), ("continent", lambda r: r["continent"])]
    with open(a.annotations, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "verdict" in (reader.fieldnames or []):  # a review of the detector's boxes
            rows = summarise_counts(frames, review_counts(list(reader)), groupers)
            return write_and_print(rows, a.out)
    if not bbox_dirs:
        sys.exit("manual boxes are scored against the detection files: give --bbox")
    gts = defaultdict(list)
    with open(a.annotations, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("frame_k") and r["sample_id"] in frames:
                frames[r["sample_id"]]["frame_k"] = r["frame_k"]
            if r["class"] in CLASSES.values():
                gts[r["sample_id"]].append(
                    (r["class"], tuple(float(r[k]) for k in ("x_center", "y_center", "width", "height"))))
            else:
                gts.setdefault(r["sample_id"], [])
    preds = {}
    for sid, row in frames.items():
        if sid in gts:
            preds[sid] = predictions(row, bbox_dirs)
            if preds[sid] is None:
                print(f"no detection file for {sid} ({row['video_id']}_{row['start_s']})")
    write_and_print(evaluate(frames, gts, preds, groupers), a.out)


def write_and_print(rows, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for r in rows:
        if r["group"] == "all":
            ap = f"  AP50 {r['ap50']:.3f}" if "ap50" in r else ""
            print(f"{r['class']:10} boxes {r['boxes']:5}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
                  f"F1 {r['f1']:.3f}{ap}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
