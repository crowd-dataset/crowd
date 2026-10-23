import math

from release import audit_stats as a
from release import evaluate_detections as e

BOX = (0.5, 0.5, 0.2, 0.2)


def test_iou_and_matching():
    assert abs(e.iou(BOX, BOX) - 1.0) < 1e-9
    assert e.iou(BOX, (0.9, 0.9, 0.1, 0.1)) == 0.0
    scored = e.match([(0.9, BOX), (0.8, BOX), (0.4, (0.1, 0.1, 0.1, 0.1))], [BOX])
    assert scored == [(0.9, True), (0.8, False), (0.4, False)]  # one box can only be matched once


def test_perfect_and_imperfect_predictions():
    frames = {"f1": {"tod": "day", "continent": "Asia"}, "f2": {"tod": "night", "continent": "Asia"}}
    gts = {"f1": [("car", BOX)], "f2": []}
    perfect = {"f1": [("car", 0.9, BOX)], "f2": []}
    rows = e.evaluate(frames, gts, perfect, [("all", lambda r: "all")])
    car = next(r for r in rows if r["class"] == "car")
    assert (car["precision"], car["recall"], car["ap50"]) == (1.0, 1.0, 1.0)
    noisy = {"f1": [("car", 0.9, BOX)], "f2": [("car", 0.95, BOX), ("car", 0.3, BOX)]}  # a false positive above 0.7
    rows = e.evaluate(frames, gts, noisy, [("all", lambda r: "all")])
    car = next(r for r in rows if r["class"] == "car")
    assert car["precision"] == 0.5 and car["recall"] == 1.0 and car["ap50"] < 1.0


def test_average_precision():
    assert e.average_precision([(0.9, True), (0.8, True)], 2) == 1.0
    assert e.average_precision([(0.9, False), (0.8, True)], 1) == 0.5
    assert math.isnan(e.average_precision([], 0))


def test_audit_statistics():
    lo, hi = a.wilson(5, 10)
    assert lo < 0.5 < hi and a.wilson(0, 0)[0] != a.wilson(0, 0)[0]
    assert a.cohen_kappa([("day", "day"), ("night", "night"), ("day", "day"), ("night", "night")]) == 1.0
    rows = [{"boundary_ok": "y", "tod": "day", "tod_audit": "day", "platform": "Car", "platform_audit": "car",
             "continent": "Asia"},
            {"boundary_ok": "n", "tod": "day", "tod_audit": "night", "platform": "Car", "platform_audit": "bus",
             "continent": "Asia"},
            {"boundary_ok": "", "tod": "day", "tod_audit": "", "platform": "Car", "platform_audit": "",
             "continent": "Asia"}]
    s = a.summarise(rows)
    assert s["audited"] == 2 and s["boundary_corrected"][:2] == (1, 2) and s["tod_disagree"][:2] == (1, 2)
    assert s["platform_disagree"][:2] == (1, 2)
