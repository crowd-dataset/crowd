"""
Re-query the TomTom traffic index for localities that still need a local reading, a batch at a time.

The traffic index (how much slower than free flow traffic is on the road nearest to the locality, %) is one TomTom
reading per locality. Pending: localities with an index of 0 (older entries also stored failed requests as 0) or
whose last reading was taken at local night. Only localities where it is daytime now (08:00-20:00 local solar time)
are queried, so run it in the morning (Asia and Oceania in daytime) and in the evening (Europe, Africa and the
Americas), Central European time. At most --limit requests per run: TomTom's free tier for this API (Traffic
Flow segment data) allows 20,000 requests a month.

Each reading is appended to traffic_index_log.csv, which records what is done; mapping.csv gets the new values
(empty where TomTom has no road data near the point). Run until it reports that nothing is pending.

Usage: python update_traffic_index.py [--limit 1200] [--dry-run]
"""

import argparse
import csv
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

import common

URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
LOG = "traffic_index_log.csv"
DONE = {"ok", "no coverage"}  # statuses that need no further query; "night" and errors are queried again
DAY = (8, 20)  # local solar hours counted as daytime


def local_hour(lon: float, now: datetime) -> float:
    return (now.hour + now.minute / 60 + lon / 15) % 24


def query(key: str, lat: float, lon: float) -> tuple:
    """(status, value): ("ok", index), ("no coverage", None), or an error status."""
    status = "error"
    for attempt in range(4):
        try:
            r = requests.get(URL, params={"key": key, "point": f"{lat},{lon}"}, timeout=20)
        except requests.RequestException as e:
            status = f"error {type(e).__name__}"
            time.sleep(2)
            continue
        if r.status_code == 429 or r.status_code >= 500:  # rate limit or server error: wait and retry
            status = f"http {r.status_code}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 400 and "too far" in r.text.lower():  # no road segment near the point
            return "no coverage", None
        if r.status_code == 403:
            return "http 403 (quota or key)", None
        if r.status_code != 200:
            return f"http {r.status_code}", None
        flow = r.json().get("flowSegmentData") or {}
        if not flow.get("freeFlowSpeed"):
            return "no coverage", None
        return "ok", round((1 - flow["currentSpeed"] / flow["freeFlowSpeed"]) * 100, 2)
    return status, None


def read_log() -> dict:
    """id -> status of its latest reading."""
    if not os.path.exists(LOG):
        return {}
    with open(LOG, newline="", encoding="utf-8") as f:
        return {r["id"]: r["status"] for r in csv.DictReader(f)}


def pending(rows: list, header: list, log: dict, now: datetime) -> tuple:
    """(pending rows in daytime now, all pending rows)."""
    i_id, i_t, i_lon = header.index("id"), header.index("traffic_index"), header.index("lon")
    todo = []
    for r in rows:
        status = log.get(r[i_id])
        if status in DONE:
            continue
        if status is not None or r[i_t] in ("0", "0.0"):  # zero, or a reading still to be repeated (night, error)
            todo.append(r)
    day = [r for r in todo if r[i_lon] and DAY[0] <= local_hour(float(r[i_lon]), now) < DAY[1]]
    return day, todo


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=1200, help="maximum number of requests in this run")
    parser.add_argument("--dry-run", action="store_true", help="only show what is pending")
    args = parser.parse_args()

    mapping = common.get_configs("mapping")
    raw = open(mapping, newline="", encoding="utf-8").read()
    rows = list(csv.reader(io.StringIO(raw)))
    header, body = rows[0], rows[1:]
    now = datetime.now(timezone.utc)
    day, todo = pending(body, header, read_log(), now)
    batch = day[:args.limit]
    print(f"Pending: {len(todo)} localities; in daytime now: {len(day)}; querying {len(batch)}.")
    if args.dry_run or not batch:
        if not todo:
            print("Nothing pending: all localities have a traffic index reading. Done.")
        return

    key = common.get_secrets("tomtom_api_key")
    i_id, i_t, i_lat, i_lon = (header.index(c) for c in ("id", "traffic_index", "lat", "lon"))
    with ThreadPoolExecutor(4) as pool:  # TomTom allows a few requests per second
        results = list(pool.map(lambda r: query(key, r[i_lat], r[i_lon]), batch))

    stamp = now.isoformat(timespec="seconds")
    new_log = not os.path.exists(LOG)
    new_values = {}
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["id", "queried_utc", "status", "value"])
        for r, (status, value) in zip(batch, results):
            w.writerow([r[i_id], stamp, status, "" if value is None else value])
            if status == "ok":
                new_values[r[i_id]] = str(float(value))
            elif status == "no coverage":
                new_values[r[i_id]] = ""  # no data, not free-flowing traffic

    # re-read: the channel routine may have saved the mapping while the requests ran
    raw = open(mapping, newline="", encoding="utf-8").read()
    rows = list(csv.reader(io.StringIO(raw)))
    for r in rows[1:]:
        if r[i_id] in new_values:
            r[i_t] = new_values[r[i_id]]
    out = io.StringIO()
    csv.writer(out, lineterminator="\r\n" if "\r\n" in raw[:5000] else "\n").writerows(rows)
    with open(mapping, "w", newline="", encoding="utf-8") as f:
        f.write(out.getvalue())

    counts = {}
    for status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    print("Results:", ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    if any(s.startswith("http 403") for s, _ in results):
        print("TomTom refused requests (403): the daily quota is probably used up. Run again tomorrow.")
    print(f"Still pending: {len(todo) - sum(s in DONE for s, _ in results)}. Updated {mapping} and {LOG}.")


if __name__ == "__main__":
    main()
