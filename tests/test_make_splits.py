import csv
from collections import Counter, defaultdict

from release import make_splits as m

HEADER = ["id", "locality", "locality_aka", "state", "country", "iso3", "continent", "lat", "lon", "videos",
          "time_of_day", "start_time", "end_time", "vehicle_type", "upload_date", "channel"]
# one upload in two localities, an upload without a known channel, and a second upload of the first channel
ROWS = [
    ["1", "Aville", "[]", "", "Atland", "ATL", "Europe", "1.0", "1.0", "[v0000000001,v0000000002]", "[[0],[0,1]]",
     "[[0],[10,500]]", "[[700],[400,900]]", "[0,0]", "[1012024,2012024]", "[UCchannelA,UCchannelB]"],
    ["2", "Bville", "[]", "", "Atland", "ATL", "Europe", "2.0", "2.0", "[v0000000001,v0000000003]", "[[0],[0]]",
     "[[1000],[0]]", "[[1600],[800]]", "[0,0]", "[1012024,3012024]", "[UCchannelA,None]"],
]


def test_split_is_stable_and_roughly_70_15_15():
    names = [f"channel:UC{i}" for i in range(20000)]
    assert [m.split_of(n) for n in names[:50]] == [m.split_of(n) for n in names[:50]]
    shares = Counter(m.split_of(n) for n in names)
    assert abs(shares["train"] / 20000 - 0.70) < 0.02 and abs(shares["val"] / 20000 - 0.15) < 0.02


def test_groups_never_cross_splits_and_weights_sum_to_one(tmp_path):
    mapping = tmp_path / "mapping.csv"
    with open(mapping, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([HEADER, *ROWS])
    rows = m.build_rows(mapping)
    assert len(rows) == 5 and len({r["segment_id"] for r in rows}) == 5
    for key, split in (("channel_group", "split_source"), ("video_id", "split_source"),
                       ("locality_group", "split_locality")):
        seen = defaultdict(set)
        for r in rows:
            seen[r[key]].add(r[split])
        assert all(len(v) == 1 for v in seen.values()), key
    assert {r["channel_group"] for r in rows if r["video_id"] == "v0000000003"} == {"video:v0000000003"}
    for key, weight in (("locality_group", "weight_locality"), ("iso3", "weight_country")):
        total = defaultdict(float)
        for r in rows:
            total[r[key]] += r[weight]
        assert all(abs(v - 1) < 1e-9 for v in total.values())
