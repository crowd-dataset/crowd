import pandas as pd
from math import radians, sin, cos, sqrt, atan2

RADIUS_KM = 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


df = pd.read_csv('mapping.csv')
df = df.dropna(subset=['lat', 'lon'])

rows = df[['id', 'locality', 'country', 'lat', 'lon']].values.tolist()
pairs = []
nearby_ids = set()

for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        id1, loc1, country1, lat1, lon1 = rows[i]
        id2, loc2, country2, lat2, lon2 = rows[j]
        dist = haversine_km(lat1, lon1, lat2, lon2)
        if dist <= RADIUS_KM:
            pairs.append({
                'id_1': int(id1), 'locality_1': loc1, 'country_1': country1,
                'lat_1': lat1, 'lon_1': lon1,
                'id_2': int(id2), 'locality_2': loc2, 'country_2': country2,
                'lat_2': lat2, 'lon_2': lon2,
                'distance_km': round(dist, 3)
            })
            nearby_ids.update([int(id1), int(id2)])

# CSV output
out = pd.DataFrame(pairs)
csv_file = f'nearby_pairs_{RADIUS_KM}km.csv'
out.to_csv(csv_file, index=False)
print(f"Found {len(pairs)} pairs within {RADIUS_KM}km -> {csv_file}")

# Collect unique cities involved in any nearby pair
nearby_cities = (
    df[df['id'].isin(nearby_ids)][['id', 'locality', 'country', 'lat', 'lon']]
    .drop_duplicates()
)

# Build JS separately to avoid f-string issues with large content
js_lines = []
js_lines.append('const markerStyle = {radius:7,fillColor:"#ef4444",color:"#fff",weight:2,opacity:1,fillOpacity:0.9};')

for _, row in nearby_cities.iterrows():
    loc = row['locality'].replace("'", "\\'")
    country = row['country'].replace("'", "\\'")
    js_lines.append(
        f"L.circleMarker([{row['lat']},{row['lon']}],markerStyle)"
        f".addTo(map)"
        f".bindTooltip('{loc}',{{permanent:true,direction:'top',className:'city-label'}})"
        f".bindPopup('<b>{loc}</b><br>{country}<br>ID:{int(row['id'])}<br>{row['lat']:.5f},{row['lon']:.5f}');"
    )

for p in pairs:
    loc1 = p['locality_1'].replace("'", "\\'")
    loc2 = p['locality_2'].replace("'", "\\'")
    dist = p['distance_km']
    js_lines.append(
        f"L.polyline([[{p['lat_1']},{p['lon_1']}],[{p['lat_2']},{p['lon_2']}]],"
        f"{{color:'#ef4444',weight:1.5,opacity:0.5,dashArray:'4 4'}})"
        f".addTo(map)"
        f".bindPopup('{loc1} &harr; {loc2}: {dist} km');"
    )

js_file = f'nearby_pairs_{RADIUS_KM}km.js'
with open(js_file, 'w') as f:
    f.write('\n'.join(js_lines))
print(f"JS saved -> {js_file}")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nearby Cities (within {RADIUS_KM} km)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
  #header {{
    padding: 14px 20px;
    background: #1e293b;
    border-bottom: 1px solid #334155;
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  #header h1 {{ font-size: 15px; font-weight: 600; color: #f1f5f9; }}
  #header .badge {{
    background: #ef4444;
    color: white;
    font-size: 12px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
  }}
  #map {{ height: calc(100vh - 49px); width: 100%; }}
  .city-label {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 11px;
    font-weight: 600;
    color: #1e293b;
    white-space: nowrap;
    text-shadow: 0 0 3px #fff, 0 0 3px #fff;
  }}
</style>
</head>
<body>
<div id="header">
  <h1>Nearby city pairs — within {RADIUS_KM} km</h1>
  <span class="badge">{len(pairs)} pairs &middot; {len(nearby_ids)} cities</span>
</div>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([20, 0], 2);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 19
  }}).addTo(map);
</script>
<script src="{js_file}"></script>
</body>
</html>"""

html_file = f'nearby_pairs_{RADIUS_KM}km.html'
with open(html_file, 'w') as f:
    f.write(html)
print(f"Map saved -> {html_file}")
