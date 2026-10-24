"""Check that GPS positions count for the city whose suburbs they are in (Pavlo, 2026-10-24): within metro_km past
its boundary, the nearest boundary first, and a locality without a boundary by its _town_km disc. Boundaries are
given, so nothing is looked up. Run from the repo with the crowd env's python."""
import json

import pandas as pd

import propose_segments as ps

assert [round(float(ps.metro_km(p))) for p in (5_231_147, 1_700_000, 100_000, 10_000, 0)] == [60, 43, 11, 3, 2]

df = pd.DataFrame({'locality': ['Sydney', 'Campbelltown', 'Auckland'], 'state': ['NSW', 'NSW', None],
                   'country': ['Australia', 'Australia', 'New Zealand'], 'lat': [-33.87, -34.065, -36.85],
                   'lon': [151.21, 150.82, 174.76], 'population_locality': [5_231_147, 13_193, 1_547_200]})
fences = {
    json.dumps(['Sydney', 'NSW', 'Australia']): [[(-33.92, 151.15), (-33.92, 151.25), (-33.82, 151.25),
                                                  (-33.82, 151.15)]],
    json.dumps(['Campbelltown', 'NSW', 'Australia']): [[(-34.09, 150.79), (-34.09, 150.85), (-34.04, 150.85),
                                                        (-34.04, 150.79)]],
    json.dumps(['Auckland', None, 'New Zealand']): [],  # no boundary on OpenStreetMap
}
cases = {
    (-33.87, 151.20): 'Sydney',          # inside
    (-33.60, 150.75): 'Sydney',          # a suburb ~50 km past the boundary (Richmond's way)
    (-34.10, 150.82): 'Campbelltown',    # just past the smaller town's boundary, nearer than Sydney's
    (-34.60, 150.20): None,              # ~100 km out: rural
    (-36.70, 174.70): 'Auckland',        # 17 km from its middle: its disc reaches 25, then 41 more
    (-35.50, 174.30): None,              # 155 km out
}
for (lat, lon), want in cases.items():
    got = ps.metro_locality(lat, lon, df, fences)
    assert (got[0] if got else None) == want, (lat, lon, got, want)
print('ok')
