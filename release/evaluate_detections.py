"""Precision and recall of the released YOLO11x detections on the audit frames, from the manual review of the boxes.

    python -m release.evaluate_detections box_review.csv --frames audit_frames.csv --out detection_eval.csv

box_review.csv is written by release.annotate (the "review" page). The reviewer saw the detector's boxes (confidence
>= 0.7) and marked each as correct, a false positive or the wrong class (with the true class), and drew the objects it
missed. A correct box is a true positive; a false positive and a wrong class are false positives of the predicted
class; a missed object and a wrong class are false negatives of the true class. Precision, recall and F1 are given per
class, overall and per time of day and continent, with a 95% bootstrap interval over frames. No average precision is
possible, since only the boxes above the threshold were shown. Seeing the detector's boxes can make a reviewer accept
them too readily, so the precision can be optimistic.
"""
import argparse
import csv
import random
from collections import defaultdict

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light",
           11: "stop sign"}  # COCO ids; the first six are the road users of the analyses
BOOTSTRAP = 1000
SEED = 2026


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
    """{sample_id: {class: (true positives, false positives, ground-truth objects)}} from box_review.csv rows."""
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
    """Rows of precision, recall and F1 from per-frame, per-class counts, by group."""
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
                rows.append({"group": gname, "value": value, "class": cls, "frames": len(sids), "objects": n_gt,
                             "precision": p, "precision_lo": ci[0][0], "precision_hi": ci[0][1],
                             "recall": r, "recall_lo": ci[1][0], "recall_hi": ci[1][1], "f1": f,
                             "f1_lo": ci[2][0], "f1_hi": ci[2][1]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("review", help="box_review.csv written by release.annotate")
    p.add_argument("--frames", default="paper/response/audit_frames.csv", help="the sample of frames")
    p.add_argument("--out", default="detection_eval.csv")
    a = p.parse_args()
    with open(a.frames, newline="", encoding="utf-8") as f:
        frames = {r["sample_id"]: r for r in csv.DictReader(f)}
    with open(a.review, newline="", encoding="utf-8") as f:
        per = review_counts(list(csv.DictReader(f)))
    groupers = [("all", lambda r: "all"), ("tod", lambda r: r["tod"]), ("continent", lambda r: r["continent"])]
    rows = summarise_counts(frames, per, groupers)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for r in rows:
        if r["group"] == "all":
            print(f"{r['class']:10} objects {r['objects']:5}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
                  f"F1 {r['f1']:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
