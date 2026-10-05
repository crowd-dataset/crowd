"""Drag the localities added to the mapping since a commit to where they really are, then write those coordinates.

    python fix_coords.py --country "Puerto Rico"           # page at http://127.0.0.1:8771: drag the markers
    python fix_coords.py --country "Puerto Rico" --apply   # write the dragged coordinates to mapping.csv

Each drag is kept in _output/coords_fixes.json; --apply changes only lat and lon of those rows.
"""
import argparse
import csv
import fcntl
import io
import json
import os
import subprocess

from flask import Flask, jsonify, render_template_string, request

MAPPING = 'mapping.csv'
FIXES = '_output/coords_fixes.json'


def key(r):
    return f"{r['locality']}|{r['state']}|{r['country']}"


def new_rows(since, country):
    """Rows of the mapping that the commit since does not have, in the country when one is given."""
    old = subprocess.run(['git', 'show', f'{since}:{MAPPING}'], capture_output=True, text=True, check=True).stdout
    known = {key(r) for r in csv.DictReader(io.StringIO(old))}
    with open(MAPPING, newline='') as f:
        return [r for r in csv.DictReader(f) if key(r) not in known and (not country or r['country'] == country)]


def load_fixes():
    try:
        with open(FIXES) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def apply(rows, fixes):
    """Write the dragged coordinates into the mapping, changing nothing else in its lines. Holds the lock the
    review page's Apply takes, and swaps the file in whole."""
    with open(MAPPING + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _apply(rows, fixes)


def _apply(rows, fixes):
    with open(MAPPING, newline='') as f:
        lines = f.read().split('\n')
    header = next(csv.reader([lines[0]]))
    lat_i, lon_i = header.index('lat'), header.index('lon')
    wanted = {key(r) for r in rows}
    done = []
    for i, line in enumerate(lines[1:], 1):
        if not line:
            continue
        cells = next(csv.reader([line]))
        k = key(dict(zip(header, cells)))
        if k in wanted and k in fixes:
            cells[lat_i], cells[lon_i] = str(fixes[k][0]), str(fixes[k][1])
            buf = io.StringIO()
            csv.writer(buf, lineterminator='').writerow(cells)
            lines[i] = buf.getvalue()
            done.append(k.split('|')[0])
    with open(MAPPING + '.tmp', 'w', newline='') as f:
        f.write('\n'.join(lines))
    os.replace(MAPPING + '.tmp', MAPPING)
    return done


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Fix coordinates</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body { margin: 0; font-family: Arial, sans-serif; display: flex; height: 100vh; background: #fff; color: #222; }
:root { color-scheme: light; }
#list { background: #fff; width: 300px; overflow-y: auto; border-right: 1px solid #ccc; font-size: 14px; }
#list div { padding: 6px 10px; border-bottom: 1px solid #eee; cursor: pointer; }
#list div:hover { background: #f3f3f3; }
#list .moved { background: #e8f5e9; }
#list small { color: #666; display: block; }
#map { flex: 1; }
h1 { font-size: 15px; margin: 10px; }
</style></head><body>
<div id="list"><h1>{{ rows | length }} localities: drag each marker to the town centre (saved at once)</h1></div>
<div id="map"></div>
<script>
const rows = {{ rows | tojson }}, fixes = {{ fixes | tojson }};
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(map);
const list = document.getElementById('list'), bounds = [];
for (const r of rows) {
  const k = `${r.locality}|${r.state}|${r.country}`, at = fixes[k] || [+r.lat, +r.lon];
  const m = L.marker(at, { draggable: true }).addTo(map).bindTooltip(r.locality, { permanent: true, direction: 'right' });
  const item = document.createElement('div');
  const show = (lat, lon) => item.innerHTML = `${r.locality}<small>${lat.toFixed(5)}, ${lon.toFixed(5)}` +
    (fixes[k] ? ` (moved; was ${(+r.lat).toFixed(5)}, ${(+r.lon).toFixed(5)})` : '') + '</small>';
  item.classList.toggle('moved', !!fixes[k]); show(at[0], at[1]);
  item.onclick = () => map.setView(m.getLatLng(), 15);
  m.on('dragend', async () => {
    const { lat, lng } = m.getLatLng();
    const res = await fetch('fix', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: k, lat, lon: lng }) });
    if (!res.ok) { alert('not saved'); return; }
    fixes[k] = [lat, lng]; item.classList.add('moved'); show(lat, lng);
  });
  list.appendChild(item); bounds.push(at);
}
// the size is known only once the page is laid out; fitting before that zooms all the way in
window.addEventListener('load', () => { map.invalidateSize(); map.fitBounds(bounds, { padding: [30, 30] }); });
map.setView(bounds[0], 10);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--country', help='only localities of this country, as the mapping writes it')
    ap.add_argument('--since', default='HEAD', help='git commit the new localities are not in (default: HEAD)')
    ap.add_argument('--port', type=int, default=8771)
    ap.add_argument('--apply', action='store_true', help='write the dragged coordinates to mapping.csv')
    args = ap.parse_args()
    rows = new_rows(args.since, args.country)
    if args.apply:
        done = apply(rows, load_fixes())
        print(f'{len(done)} localities moved in {MAPPING}: {", ".join(done) or "none"}')
        return
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(PAGE, rows=rows, fixes=load_fixes())

    @app.route('/fix', methods=['POST'])
    def fix():
        d = request.get_json()
        fixes = load_fixes()
        fixes[d['key']] = [round(float(d['lat']), 7), round(float(d['lon']), 7)]
        os.makedirs(os.path.dirname(FIXES), exist_ok=True)
        with open(FIXES + '.tmp', 'w') as f:
            json.dump(fixes, f, indent=1, ensure_ascii=False)
        os.replace(FIXES + '.tmp', FIXES)
        return jsonify(ok=True)

    print(f'{len(rows)} localities: http://127.0.0.1:{args.port}')
    app.run(port=args.port)


if __name__ == '__main__':
    main()
