import os
import time

from tools import repair_track_ids as r

HEADER = "yolo-id,x-center,y-center,width,height,unique-id,confidence,frame-count"


def write(path, rows, eol="\n"):
    lines = [HEADER] + [f"{c},0.5,0.5,0.1,0.1,{i},0.9,{k}" for c, i, k in rows]
    path.write_bytes((eol.join(lines) + eol).encode())
    os.utime(path, (time.time() - 3600, time.time() - 3600))


def test_an_id_that_is_silent_longer_than_the_tracker_remembers_is_split():
    ids = [7, 7, 7, 8, 7, 7, 9]
    frames = [1, 2, 3, 3, 500, 501, 600]
    fixed, split = r.new_ids(ids, frames, r.gap_limit(30))
    assert fixed.tolist() == [7, 7, 7, 8, 10, 10, 9] and split == 1  # the later run of 7 gets max id + 1


def test_gaps_the_tracker_can_bridge_and_rows_without_a_track_are_left_alone():
    fixed, split = r.new_ids([5, 5, -1, -1, 5], [1, 62, 1, 900, 124], r.gap_limit(30))  # 62 frames: still bridged
    assert split == 0 and fixed.tolist() == [5, 5, -1, -1, 5]
    fixed, split = r.new_ids([5, 5], [1, 65], r.gap_limit(30))
    assert split == 1 and fixed.tolist() == [5, 6]


def test_the_result_does_not_depend_on_the_order_of_the_rows():
    ids = [7, 7, 7, 8, 7, 7, 9]
    frames = [1, 2, 3, 3, 500, 501, 600]
    a, _ = r.new_ids(ids, frames, 62)
    perm = [6, 2, 4, 0, 5, 1, 3]
    b, _ = r.new_ids([ids[i] for i in perm], [frames[i] for i in perm], 62)
    assert [a[i] for i in perm] == b.tolist()


def test_file_repair_changes_only_the_ids_is_repeatable_and_can_be_undone(tmp_path):
    rows = [(0, 7, 1), (0, 7, 2), (2, 8, 2), (0, 7, 800), (0, 7, 801), (2, 9, 802)]
    for name, eol in (("a_0_30.csv", "\n"), ("b_0_30.csv", "\r\n")):
        write(tmp_path / name, rows, eol)
    original = {n: (tmp_path / n).read_bytes() for n in ("a_0_30.csv", "b_0_30.csv")}
    patches = tmp_path / "patches"
    dry = r.repair_file((str(tmp_path / "a_0_30.csv"), False, 600, str(patches)))
    assert dry["status"] == "to repair" and (tmp_path / "a_0_30.csv").read_bytes() == original["a_0_30.csv"]
    for name in original:
        done = r.repair_file((str(tmp_path / name), True, 600, str(patches)))
        assert done["status"] == "repaired" and done["rows_changed"] == 2 and done["ids_split"] == 1
        new = (tmp_path / name).read_bytes().decode().replace("\r\n", "\n").split("\n")
        old = original[name].decode().replace("\r\n", "\n").split("\n")
        diff = [(o, n) for o, n in zip(old, new) if o != n]
        assert [(o.split(",")[5], n.split(",")[5]) for o, n in diff] == [("7", "10"), ("7", "10")]
        assert all(o.split(",")[:5] == n.split(",")[:5] and o.split(",")[6:] == n.split(",")[6:] for o, n in diff)
        assert r.repair_file((str(tmp_path / name), True, 0, str(patches)))["status"].startswith("ok")  # again: no-op
        assert r.undo_file((str(tmp_path / name), str(patches)))[1] == "undone"
        assert (tmp_path / name).read_bytes() == original[name]


def test_identical_files_give_identical_results_and_recent_files_are_skipped(tmp_path):
    rows = [(0, 7, 1), (0, 7, 700), (2, 8, 5)]
    for d in ("one", "two"):
        (tmp_path / d).mkdir()
        write(tmp_path / d / "v_0_30.csv", rows)
        r.repair_file((str(tmp_path / d / "v_0_30.csv"), True, 600, str(tmp_path / d / "p")))
    assert (tmp_path / "one" / "v_0_30.csv").read_bytes() == (tmp_path / "two" / "v_0_30.csv").read_bytes()
    fresh = tmp_path / "w_0_30.csv"
    write(fresh, rows)
    os.utime(fresh, None)
    assert r.repair_file((str(fresh), True, 600, str(tmp_path / "p")))["status"].startswith("skipped")
