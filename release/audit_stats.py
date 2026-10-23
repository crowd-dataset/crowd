"""Agreement statistics of the manual label audit (release.sample_audit's audit_labels.csv after it is filled in).

    python -m release.audit_stats [audit_labels.csv]

Reports, overall and per continent, the number of audited segments, the share with a boundary that had to be
corrected, and the agreement with the audited time-of-day and platform labels (with Cohen's kappa for time of day),
each with a 95% Wilson interval. Rows with an empty boundary_ok are not counted as audited.

Columns to fill in: boundary_ok (y or n), correct_start_s and correct_end_s (when n), tod_audit (day or night),
platform_audit (the platform label name, as in the platform column).
"""
import csv
import math
import sys
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    """(low, high) Wilson score interval of a share k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def cohen_kappa(pairs):
    """Cohen's kappa of (label A, label B) pairs; nan if undefined."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def summarise(rows):
    """Dict of counts and shares for a list of audited rows."""
    audited = [r for r in rows if r["boundary_ok"].strip().lower() in ("y", "n")]
    wrong = sum(r["boundary_ok"].strip().lower() == "n" for r in audited)
    tod = [(r["tod"].strip().lower(), r["tod_audit"].strip().lower()) for r in audited if r["tod_audit"].strip()]
    plat = [(r["platform"].strip().lower(), r["platform_audit"].strip().lower())
            for r in audited if r["platform_audit"].strip()]
    return {
        "audited": len(audited),
        "boundary_corrected": (wrong, len(audited), *wilson(wrong, len(audited))),
        "tod_disagree": (sum(a != b for a, b in tod), len(tod), *wilson(sum(a != b for a, b in tod), len(tod))),
        "tod_kappa": cohen_kappa(tod),
        "platform_disagree": (sum(a != b for a, b in plat), len(plat),
                              *wilson(sum(a != b for a, b in plat), len(plat))),
    }


def fmt(t):
    k, n, lo, hi = t
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f} to {100 * hi:.1f}%)" if n else "no rows"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/response/audit_labels.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups["All"].append(r)
        groups[r["continent"]].append(r)
    for name in ("All", *sorted(set(groups) - {"All"})):
        s = summarise(groups[name])
        print(f"{name}: {s['audited']} audited")
        print(f"  boundary corrected: {fmt(s['boundary_corrected'])}")
        print(f"  time of day differs: {fmt(s['tod_disagree'])}, kappa {s['tod_kappa']:.3f}")
        print(f"  platform differs: {fmt(s['platform_disagree'])}")


if __name__ == "__main__":
    main()
