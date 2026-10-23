import pandas as pd

from release import update_info as u

V1, V2, V3 = "v0000000001", "v0000000002", "v0000000003"


def mapping(*videos):
    return pd.DataFrame({"videos": [f"[{','.join(videos)}]"]})


def info(video, **fields):
    row = {"video": video, "title": "t " + video, "upload_date": "01012024", "channel": "UCx", "views": 5,
           "description": "d", "chapters": "[]"}
    return {**row, **fields}


def rows(path):
    return pd.read_csv(path, dtype=str).fillna("")


def test_each_video_is_in_the_file_before_the_next_is_fetched(tmp_path):
    path = str(tmp_path / "meta.csv")
    seen = []

    def fetch(video):
        seen.append(len(rows(path)) if (tmp_path / "meta.csv").exists() else 0)
        return info(video)

    assert u.update(mapping(V1, V2, V3), path, fetch) == (3, 0, 0)
    assert seen == [0, 1, 2]  # the rows of the earlier videos were already written
    df = rows(path)
    assert list(df["id"]) == ["1", "2", "3"] and list(df["video"]) == [V1, V2, V3] and set(df["views"]) == {"5"}


def test_failed_fetch_of_a_new_video_is_written_empty_and_retried_next_time(tmp_path):
    path = str(tmp_path / "meta.csv")
    u.update(mapping(V1), path, lambda v: info(v, title="", upload_date="", channel="", views=""))
    assert rows(path).loc[0, "title"] == ""
    assert u.update(mapping(V1), path, lambda v: info(v, title="back")) == (0, 1, 0)
    df = rows(path)
    assert len(df) == 1 and df.loc[0, "title"] == "back" and df.loc[0, "id"] == "1"


def test_failed_retry_does_not_blank_the_row(tmp_path):
    path = str(tmp_path / "meta.csv")
    u.update(mapping(V1), path, lambda v: info(v, channel=""))  # channel missing: the row will be retried
    assert u.update(mapping(V1), path, lambda v: info(v, title="", upload_date="", channel="", views="")) == (0, 0, 1)
    row = rows(path).loc[0]
    assert row["title"] == "t " + V1 and row["views"] == "5"  # what the row had stays


def test_successful_retry_keeps_the_rows_added_in_the_same_run(tmp_path):
    path = str(tmp_path / "meta.csv")
    u.update(mapping(V1), path, lambda v: info(v, channel=""))
    assert u.update(mapping(V1, V2), path, lambda v: info(v, title="new " + v)) == (1, 1, 0)
    df = rows(path)
    assert sorted(df["video"]) == [V1, V2] and set(df["title"]) == {"new " + V1, "new " + V2}


def test_half_written_last_row_is_cut_and_fetched_again(tmp_path):
    path = tmp_path / "meta.csv"
    u.update(mapping(V1, V2), str(path), lambda v: info(v))
    path.write_text(path.read_text().rstrip("\n")[:-15], encoding="utf-8")  # a run killed while writing the last row
    assert u.update(mapping(V1, V2), str(path), lambda v: info(v)) == (1, 0, 0)
    assert sorted(rows(str(path))["video"]) == [V1, V2]


def test_videos_that_left_the_mapping_are_dropped_and_ids_are_not_reused(tmp_path):
    path = str(tmp_path / "meta.csv")
    u.update(mapping(V1, V2), path, lambda v: info(v))
    u.update(mapping(V1, V3), path, lambda v: info(v))
    df = rows(path)
    assert sorted(df["video"]) == [V1, V3] and df.set_index("video").loc[V3, "id"] == "3"
