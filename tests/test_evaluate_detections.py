from release import audit_stats as a


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
