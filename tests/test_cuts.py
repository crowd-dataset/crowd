from tools import apply_cuts as c


def cut(t):
    return {"raw_cut_time": t, "previous_end_time": int(t // 1), "next_start_time": int(-(-t // 1))}


def test_a_cut_splits_into_floor_and_ceil():
    # the example of the docstring: a segment 43..200 with cuts at 143.533 and 154.367
    assert c.make_split_segments(43, 200, [cut(143.533), cut(154.367)]) == [
        {"start": 43, "end": 143}, {"start": 144, "end": 154}, {"start": 155, "end": 200}]


def test_short_parts_are_dropped():
    # a cut in the first second leaves no first part, and the segment then starts at ceil(cut)
    assert c.make_split_segments(43, 200, [cut(43.5)]) == [{"start": 44, "end": 200}]
    # a cut in the last second leaves less than MIN_SEGMENT_LENGTH_SEC for the last part
    assert c.make_split_segments(43, 100, [cut(99.5)]) == [{"start": 43, "end": 99}]


def test_cuts_outside_the_processed_part_are_ignored():
    cuts = [cut(43.0), cut(99.5), cut(70.2)]
    assert [x["raw_cut_time"] for x in c.get_valid_cuts_for_segment(cuts, 43, 100)] == [70.2]
