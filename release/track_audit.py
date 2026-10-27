"""Track-by-track audit of the identity errors of the tracker (BoT-SORT) in the released detection files.

    python -m release.track_audit track_review.csv      # summary of the answers of release.annotate (page "tracks")

The reviewer sees one track at a time: a short video (up to three pieces of 5 s: the start, the middle and the end of
the track) zoomed on the object, with only the track's own box drawn. One question: is the box on one and the same
object all the time? An identity switch (the box jumps to another object) makes it "no". Tracks are drawn from the
segments of the clip sample (audit_clips.csv), two per segment, uniformly among the tracks of road users that last at
least 2 s (confidence >= 0.7), and listed in random order, so that stopping early still leaves a random sample.
Reported: the share of tracks with an identity switch, and the share that are not a road user, with 95% Wilson
intervals. This is a taxonomy per track, not a standard tracking score (IDF1, MOTA), since identities were not
annotated, and it does not measure fragmentation (one object split over several tracks).
"""
import csv
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

ROAD_USER_IDS = (0, 1, 2, 3, 5, 7)
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLUMNS = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
SEED = 2026
PER_CLIP = 2
PIECE_S = 5.0  # a piece of the track that is shown
MIN_S = 2.0  # a track shorter than this is too brief to judge
THRESHOLD = 0.7
FIELDS = ["sample_id", "continent", "tod", "segment_id", "video_id", "locality", "contributor", "start_s",
          "processed_end_s", "track_id", "class", "first_k", "last_k", "n_frames", "seg_first_k", "seg_last_k"]
ONE = {"one": "Yes: the same object all the time", "switch": "No: the box jumps to another object",
       "false": "The box is not on a road user", "unsure": "Cannot tell"}


# ------------------------------------------------------------------------------------------------ the tracks
def read_tracks(path, threshold=THRESHOLD):
    """(road-user rows with a track id and confidence >= threshold, first and last frame of the file)."""
    raw = pd.read_csv(path, usecols=COLUMNS)[COLUMNS]
    raw.columns = ["cls", "x", "y", "w", "h", "id", "conf", "k"]
    df = raw[(raw["conf"] >= threshold) & raw["cls"].isin(ROAD_USER_IDS) & (raw["id"] >= 0)].copy()  # -1: no track
    df["id"] = df["id"].astype(int)
    df["k"] = df["k"].astype(int)
    df["cls"] = df["cls"].astype(int)
    return df, int(raw["k"].min()), int(raw["k"].max())


def choose_tracks(df, rng, n, min_frames):
    """n tracks of df chosen uniformly among those with at least min_frames frames: dicts of id, class, frames."""
    counts = df.groupby("id").size()
    ids = sorted(counts[counts >= min_frames].index)
    rng.shuffle(ids)
    out = []
    for i in ids[:n]:
        t = df[df["id"] == i]
        out.append({"track_id": int(i), "class": NAMES[int(t["cls"].mode().iloc[0])], "first_k": int(t["k"].min()),
                    "last_k": int(t["k"].max()), "n_frames": len(t)})
    return sorted(out, key=lambda t: t["first_k"])


def draw_tracks(clips, existing, find, per_clip=PER_CLIP):
    """New sample rows for the clip-sample rows that have none in `existing`. find(row) -> (detection file, fps)
    or None. The draw is seeded by the clip id, so it is repeatable."""
    have = {r["segment_id"] for r in existing}
    rows, skipped = [], 0
    for c in clips:
        if c["segment_id"] in have:
            continue
        found = find(c)
        if found is None:
            skipped += 1
            continue
        path, fps = found
        df, k0, k1 = read_tracks(path)
        rng = random.Random(f"{SEED}-{c['sample_id']}")
        for t in choose_tracks(df, rng, per_clip, int(MIN_S * fps)):
            sid = f"trk_{c['sample_id'][5:]}_t{t['track_id']}"
            rows.append({"sample_id": sid, **{k: c[k] for k in FIELDS[1:9]}, **t, "seg_first_k": k0, "seg_last_k": k1})
    random.Random(SEED).shuffle(rows)  # random order: any prefix of the list is a random sample too
    return rows, skipped


def pieces(df, track_id, fps, n=3):
    """The parts of a track to show: (first frame, last frame) of up to n pieces of PIECE_S seconds (the start, the
    middle and the end of the track; the whole track if it is short), and the box of each frame {k: [x, y, w, h]}."""
    t = df[df["id"] == track_id].sort_values("k")
    ks = [int(k) for k in t["k"]]
    boxes = {int(r.k): [round(r.x, 4), round(r.y, 4), round(r.w, 4), round(r.h, 4)] for r in t.itertuples()}
    length = round(PIECE_S * fps)
    if ks[-1] - ks[0] + 1 <= n * length:
        return [(ks[0], ks[-1])], boxes
    starts = [ks[0], ks[len(ks) // 2] - length // 2, ks[-1] - length + 1][:n]
    return [(max(k0, ks[0]), min(k0 + length - 1, ks[-1])) for k0 in starts], boxes


# ------------------------------------------------------------------------------------------------ the numbers
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return centre - half, centre + half


def summarise(rows):
    """Counts and Wilson intervals over the answered tracks of track_review.csv rows."""
    rows = [r for r in rows if r.get("one_object")]
    real = [r for r in rows if r["one_object"] != "false"]
    real = [r for r in real if r["one_object"] != "unsure"]

    def share(sub, pick):
        k = sum(pick(r) for r in sub)
        return k, len(sub), *wilson(k, len(sub))

    return {"tracks": len(rows), "false_tracks": share(rows, lambda r: r["one_object"] == "false"),
            "identity_switch": share(real, lambda r: r["one_object"] == "switch")}


def fmt(t):
    k, n, lo, hi = t
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f} to {100 * hi:.1f}%)" if n else "no rows"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/response/annotations/track_review.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups["All"].append(r)
        groups[r["continent"]].append(r)
        groups[r["tod"]].append(r)
    for name in ("All", *sorted(set(groups) - {"All"})):
        s = summarise(groups[name])
        print(f"{name}: {s['tracks']} tracks audited")
        print(f"  not a road user: {fmt(s['false_tracks'])}")
        print(f"  follows more than one object (identity switch): {fmt(s['identity_switch'])}")
    print(Counter(r["one_object"] for r in rows))


if __name__ == "__main__":
    main()
