import csv

from release import sample_audit as sa

HEADER = ["id", "locality", "locality_aka", "state", "country", "iso3", "continent", "lat", "lon", "videos",
          "time_of_day", "start_time", "end_time", "vehicle_type", "upload_date", "channel"]
VIDS = ["v0000000001", "v0000000002", "v0000000003", "v0000000004"]


def mapping(tmp_path):
    row = ["1", "Aville", "[]", "", "Atland", "ATL", "Europe", "1.0", "1.0", f"[{','.join(VIDS)}]",
           "[[0],[0],[0],[0]]", "[[0],[0],[0],[0]]", "[[600],[600],[600],[600]]", "[0,0,0,0]",
           "[1012024,1012024,1012024,1012024]", "[UCa,UCb,UCc,UCd]"]
    path = tmp_path / "mapping.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([HEADER, row])
    return path


def folders(tmp_path, with_detections, with_videos):
    bbox, videos = tmp_path / "data" / "bbox", tmp_path / "videos"
    bbox.mkdir(parents=True)
    videos.mkdir()
    for v in with_detections:
        (bbox / f"{v}_0_30.csv").write_text("x\n")
    for v in with_videos:
        (videos / f"{v}.mp4").write_text("x")
    return [str(videos)], [str(tmp_path / "data")]


def read(path):
    return list(csv.DictReader(open(path, newline="")))


def test_only_segments_that_exist_here_are_drawn(tmp_path):
    out = tmp_path / "out"
    videos, bbox = folders(tmp_path, VIDS[:2], VIDS[:3])  # v1, v2 have both; v3 has no detection file; v4 nothing
    sa.draw_samples(mapping(tmp_path), out, videos, bbox, frames_per=5, clips_per=5)
    for name in ("audit_frames", "audit_clips"):
        assert sorted(r["video_id"] for r in read(out / f"{name}.csv")) == VIDS[:2]  # one per upload, only these
    assert len(read(out / "audit_labels.csv")) == 4  # the label sample needs no local files


def test_refill_keeps_what_works_replaces_the_rest_and_never_reuses_an_id(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    common = {"continent": "Europe", "tod": "day", "segment_id": "s", "locality": "Aville", "contributor": "x",
              "start_s": "0.0", "processed_end_s": "599.0", "url": "u", "timestamp_s": "100.0"}
    fields = ["sample_id", "continent", "tod", "segment_id", "video_id", "locality", "contributor", "start_s",
              "processed_end_s", "timestamp_s", "url"]
    with open(out / "audit_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({**common, "sample_id": "frame_EUR_day_001", "video_id": VIDS[3]})  # not here: dropped
        w.writerow({**common, "sample_id": "frame_EUR_day_002", "video_id": VIDS[0]})  # here: kept
    videos, bbox = folders(tmp_path, VIDS[:3], VIDS[:3])
    sa.draw_samples(mapping(tmp_path), out, videos, bbox, frames_per=2, clips_per=1)
    rows = read(out / "audit_frames.csv")
    assert [r["sample_id"] for r in rows] == ["frame_EUR_day_002", "frame_EUR_day_003"]  # 001 is never reused
    assert rows[0]["video_id"] == VIDS[0] and rows[1]["video_id"] in VIDS[1:3] and rows[1]["video_id"] != VIDS[3]


def test_existing_label_sample_is_never_overwritten(tmp_path):
    out = tmp_path / "out"
    videos, bbox = folders(tmp_path, VIDS, VIDS)
    sa.draw_samples(mapping(tmp_path), out, videos, bbox, frames_per=1, clips_per=1)
    (out / "audit_labels.csv").write_text("kept\n")
    sa.draw_samples(mapping(tmp_path), out, videos, bbox, frames_per=1, clips_per=1)
    assert (out / "audit_labels.csv").read_text() == "kept\n"
