import csv

import cv2
import numpy as np
import pytest

from release import annotate as a
from release import evaluate_detections as e

VID = "TESTVIDEO01"
FPS = 30


def make_video(path, frames=120):
    """Frame i is a uniform grey of level 2*i, so the frame that was extracted can be read off its brightness."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(frames):
        w.write(np.full((48, 64, 3), 2 * i, np.uint8))
    w.release()


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def client(tmp_path):
    videos_a, videos_b = tmp_path / "disk1" / "videos", tmp_path / "disk2" / "videos"
    bbox_a, bbox_b = tmp_path / "disk1" / "data", tmp_path / "disk2" / "data" / "bbox"
    for d in (videos_a, videos_b, bbox_a, bbox_b):
        d.mkdir(parents=True)
    make_video(videos_b / f"{VID}.mp4")  # the video is in the second folder only
    (bbox_a / "bbox").mkdir()  # the detection file is in a bbox subfolder of the first folder
    rows = [  # frame 1 of the segment is original frame 30 (start 1 s); a person at confidence 0.9 and a car at 0.3
        {"yolo-id": 0, "x-center": 0.5, "y-center": 0.5, "width": 0.2, "height": 0.4, "unique-id": 7,
         "confidence": 0.9, "frame-count": 16},
        {"yolo-id": 2, "x-center": 0.2, "y-center": 0.2, "width": 0.1, "height": 0.1, "unique-id": 8,
         "confidence": 0.3, "frame-count": 16},
        {"yolo-id": 0, "x-center": 0.6, "y-center": 0.5, "width": 0.2, "height": 0.4, "unique-id": 7,
         "confidence": 0.9, "frame-count": 17}]
    write(bbox_a / "bbox" / f"{VID}_1_30.csv", list(rows[0]), rows)
    common = {"continent": "Asia", "tod": "day", "video_id": VID, "locality": "X", "contributor": "Pavlo",
              "start_s": "1.0", "processed_end_s": "3.0"}
    write(tmp_path / "audit_labels.csv",
          ["continent", "tod", "segment_id", "video_id", "locality", "contributor", "start_s", "end_s", "platform",
           "url_start", "url_end", "boundary_ok", "correct_start_s", "correct_end_s", "tod_audit", "platform_audit",
           "notes"],
          [{**common, "segment_id": f"{VID}_1_4", "end_s": "4.0", "platform": "Car", "url_start": "u", "url_end": "v"},
           {**common, "segment_id": f"{VID}_9_9", "end_s": "9.0", "platform": "Bus", "url_start": "u",
            "url_end": "v"}])
    fields = ["sample_id", "continent", "tod", "segment_id", "video_id", "locality", "contributor", "start_s",
              "processed_end_s"]
    write(tmp_path / "audit_frames.csv", [*fields, "timestamp_s", "url"],
          [{**common, "sample_id": "frame_ASI_day_001", "timestamp_s": "1.5", "url": "u"},
           {**common, "sample_id": "frame_ASI_day_002", "timestamp_s": "2.0", "url": "u"}])
    write(tmp_path / "audit_clips.csv", [*fields, "clip_start_s", "url"],
          [{**common, "sample_id": "clip_ASI_day_001", "clip_start_s": "1.5", "url": "u"}])
    a.TASKS.clear(), a.STATE.clear(), a.INFO.clear(), a.MISSING.clear()
    a.BBOX.update(index=None, time=0.0)
    a.CFG.update(videos=[str(videos_a), str(videos_b)], bbox=[str(bbox_a), str(bbox_b)], cache=str(tmp_path / "cache"),
                 annotator="Tester")
    a.load(tmp_path, tmp_path / "out")
    a.app.config["TESTING"] = True
    return a.app.test_client(), tmp_path


def test_labels_hide_the_dataset_labels_and_write_a_file_for_the_audit_statistics(client):
    c, tmp = client
    row = c.get(f"/api/labels/{VID}_1_4").get_json()["row"]
    assert "platform" not in row and "tod" not in row and row["start_s"] == "1.0"
    done = {"status": "done", "boundary_ok": "n", "correct_start_s": "2", "correct_end_s": "3", "tod_audit": "night",
            "platform_audit": "Bus", "notes": ""}
    assert c.post(f"/api/labels/{VID}_1_4", json=done).get_json()["ok"]
    c.post(f"/api/labels/{VID}_9_9", json={"status": "skipped"})
    filled = list(csv.DictReader(open(tmp / "out" / "audit_labels_filled.csv")))
    assert (filled[0]["boundary_ok"], filled[0]["tod_audit"], filled[0]["annotator"]) == ("n", "night", "Tester")
    assert filled[1]["boundary_ok"] == "" and filled[1]["status"] == "skipped"  # a skipped row is not audited


def test_frame_is_the_one_the_detection_file_refers_to(client):
    c, _ = client
    meta = c.get("/api/frames/frame_ASI_day_001").get_json()
    assert meta["available"] and meta["k"] == 16 and abs(meta["fps"] - FPS) < 0.1
    img = cv2.imdecode(np.frombuffer(c.get("/frame/frame_ASI_day_001.jpg").data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert abs(float(img.mean()) - 2 * 45) < 8  # original frame index 30 + 15 = 45
    boxes = c.get("/api/frames/frame_ASI_day_001/yolo").get_json()
    assert [b["c"] for b in boxes] == ["person"]  # the car at confidence 0.3 is below the threshold


def test_saved_boxes_can_be_scored(client):
    c, tmp = client
    c.post("/api/frames/frame_ASI_day_001",
           json={"status": "done", "k": 16, "mode": "blind",
                 "boxes": [{"c": "person", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.4}]})
    c.post("/api/frames/frame_ASI_day_002", json={"status": "done", "k": 31, "boxes": []})
    rows = list(csv.DictReader(open(tmp / "out" / "annotations.csv")))
    assert [(r["sample_id"], r["class"], r["frame_k"]) for r in rows] == [
        ("frame_ASI_day_001", "person", "16"), ("frame_ASI_day_002", "none", "31")]
    frames = {r["sample_id"]: r for r in csv.DictReader(open(tmp / "audit_frames.csv"))}
    for r in rows:
        frames[r["sample_id"]]["frame_k"] = r["frame_k"]
    assert e.predictions(frames["frame_ASI_day_001"], str(tmp / "disk1" / "data" / "bbox"))[0][:2] == ("person", 0.9)
    assert e.predictions(frames["frame_ASI_day_002"], str(tmp / "disk1" / "data" / "bbox")) == []


def test_clip_shows_the_tracks_of_its_frames_and_saves_the_taxonomy(client):
    c, tmp = client
    meta = c.get("/api/clips/clip_ASI_day_001").get_json()
    assert meta["k"] == 16 and meta["k1"] == 16 + 10 * FPS
    assert sorted(meta["tracks"]) == ["16", "17"] and meta["tracks"]["16"][0][:2] == [7, "person"]
    c.post("/api/clips/clip_ASI_day_001", json={"status": "done", "n_objects": 5, "id_switches": 2, "notes": ""})
    row = list(csv.DictReader(open(tmp / "out" / "clip_taxonomy.csv")))[0]
    assert (row["n_objects"], row["id_switches"], row["continent"]) == ("5", "2", "Asia")


def test_missing_video_is_reported_not_crashed(client):
    c, _ = client
    a.CFG["videos"] = []
    a.INFO.clear()
    assert c.get("/api/frames/frame_ASI_day_001").get_json()["available"] is False
    assert c.get("/frame/frame_ASI_day_002.jpg").status_code == 404
    assert all(c.get(f"/{page}").status_code == 200 for page in ("labels", "frames", "clips"))


def test_scoring_finds_detection_files_in_several_folders(client):
    _, tmp = client
    row = {"video_id": VID, "start_s": "1.0", "timestamp_s": "1.5", "frame_k": "16"}
    # the file is in a bbox subfolder of the second folder
    folders = [str(tmp / "disk2" / "data"), str(tmp / "disk1" / "data")]
    assert e.predictions(row, folders)[0][:2] == ("person", 0.9)
    assert e.predictions(row, [str(tmp / "disk2" / "data")]) is None


def test_missing_sample_files_are_reported(client, tmp_path):
    c, _ = client
    a.MISSING.clear()
    a.load(tmp_path / "nowhere", tmp_path / "out2")
    assert len(a.MISSING) == 3
    assert b"Sample files not found" in c.get("/").data and b"audit_frames.csv" in c.get("/").data


def test_frame_beyond_the_end_of_the_video_says_why(client):
    c, _ = client
    a.TASKS["frames"].append({"sample_id": "far", "continent": "Asia", "tod": "day", "video_id": VID, "start_s": "1.0",
                              "timestamp_s": "100.0", "contributor": "x", "locality": "X", "processed_end_s": "3.0"})
    r = c.get("/frame/far.jpg")
    assert r.status_code == 422 and b"could read frame" in r.data and VID.encode() in r.data


def test_ffmpeg_reads_the_frame_when_opencv_cannot(client, monkeypatch):
    c, _ = client

    class Broken:  # OpenCV that cannot decode the file
        def __init__(self, *args):
            pass

        def set(self, *args):
            return False

        def read(self):
            return False, None

        def release(self):
            pass

        def get(self, *args):
            return 0.0

    monkeypatch.setattr(a.cv2, "VideoCapture", Broken)
    a.INFO[VID] = {"fps": float(FPS), "width": 64, "height": 48}
    r = c.get("/frame/frame_ASI_day_001.jpg")
    assert r.status_code == 200
    img = cv2.imdecode(np.frombuffer(r.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert abs(float(img.mean()) - 2 * 45) < 12  # frame 45, as with OpenCV


def test_relative_folders_work(client, monkeypatch):
    """Started without --cache, the cache folder is a relative path; Flask must not look for it in release/."""
    c, tmp = client
    monkeypatch.chdir(tmp)
    a.CFG["cache"] = "rel_cache"
    r = c.get("/frame/frame_ASI_day_001.jpg")
    assert r.status_code == 200 and (tmp / "rel_cache" / "frames" / "frame_ASI_day_001.jpg").is_file()


def test_unexpected_errors_are_shown_as_a_reason_not_a_500(client, monkeypatch):
    c, _ = client

    def boom(*args):
        raise RuntimeError("something odd")

    monkeypatch.setattr(a, "extract_frame", boom)
    r = c.get("/frame/frame_ASI_day_002.jpg")
    assert r.status_code == 422 and b"RuntimeError: something odd" in r.data


def test_review_page_gives_the_detectors_boxes_and_the_verdicts_are_scored(client):
    c, tmp = client
    meta = c.get("/api/review/frame_ASI_day_001").get_json()
    assert meta["has_detections"] and meta["k"] == 16
    assert [(b["c"], b["v"]) for b in meta["preds"]] == [("person", "correct")]  # the car at 0.3 is not shown
    box = meta["preds"][0]
    wrong = {**box, "v": "wrong_class", "t": "bicycle"}
    c.post("/api/review/frame_ASI_day_001", json={"status": "done", "k": 16, "preds": [wrong],
                                                  "missed": [{"c": "car", "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1}]})
    c.post("/api/review/frame_ASI_day_002", json={"status": "done", "k": 31, "preds": [], "missed": []})
    rows = list(csv.DictReader(open(tmp / "out" / "box_review.csv")))
    assert [(r["kind"], r["class"], r["verdict"], r["true_class"]) for r in rows] == [
        ("pred", "person", "wrong_class", "bicycle"), ("missed", "car", "missed", "car"), ("none", "none", "none", "")]
    per = e.review_counts(rows)["frame_ASI_day_001"]
    assert per["person"] == (0, 1, 0)       # a false positive for the predicted class
    assert per["bicycle"] == (0, 0, 1)      # and a miss for the true class
    assert per["car"] == (0, 0, 1)          # the missed car
    frames = {r["sample_id"]: r for r in csv.DictReader(open(tmp / "audit_frames.csv"))}
    out = e.summarise_counts(frames, e.review_counts(rows), [("all", lambda r: "all")])
    person = next(r for r in out if r["class"] == "person")
    assert person["precision"] == 0.0 and person["recall"] != person["recall"] and "ap50" not in person  # recall: nan
