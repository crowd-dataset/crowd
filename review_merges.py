"""Decide which towns of the mapping to merge into the big city whose metropolitan area they lie in, then merge.

    python review_merges.py           # page at http://127.0.0.1:8772: town and city maps side by side, Y / N
    python review_merges.py --apply   # merge the towns accepted on the page into their city

Candidates: towns under METRO_TOWN_POP within METRO_KM of a city of METRO_POP or more, in the same country and
state (Jersey City, NJ is not New York City). Nothing is merged unless it was accepted on the page; decisions are
kept in _output/metro_merge_decisions.json.
"""
import argparse
import ast
import csv
import fcntl
import io
import json
import math
import os

from flask import Flask, jsonify, render_template_string, request

from propose_segments import METRO_KM, METRO_POP, METRO_TOWN_POP

MAPPING = 'mapping.csv'
DECISIONS = '_output/metro_merge_decisions.json'


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    h = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(h))


def key(r):
    return f"{r['locality']}|{r['state']}|{r['country']}"


def candidates():
    """[{town, city, km}] of the mapping, nearest city first per town."""
    with open(MAPPING, newline='') as f:
        rows = list(csv.DictReader(f))
    big = [r for r in rows if (num(r['population_locality']) or 0) >= METRO_POP and num(r['lat']) is not None]
    out = []
    for r in rows:
        pop, lat, lon = num(r['population_locality']), num(r['lat']), num(r['lon'])
        if lat is None or (pop or 0) >= METRO_TOWN_POP:
            continue
        near = [(km(lat, lon, float(b['lat']), float(b['lon'])), b) for b in big
                if b['country'] == r['country'] and b['state'] == r['state'] and b is not r]
        near = [(d, b) for d, b in near if d <= METRO_KM]
        if not near:
            continue
        d, b = min(near, key=lambda x: x[0])
        out.append({'town': side(r), 'city': side(b), 'km': round(d, 1)})
    return sorted(out, key=lambda c: (c['town']['country'], c['city']['name'], c['town']['name']))


def side(x):
    return {'key': key(x), 'name': x['locality'], 'state': x['state'], 'country': x['country'],
            'lat': float(x['lat']), 'lon': float(x['lon']), 'pop': int(num(x['population_locality']) or 0),
            'videos': len([v for v in x['videos'].strip('[]').split(',') if v])}


def load_decisions():
    try:
        with open(DECISIONS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def merge(accepted):
    """Merge each accepted town row into its city row: its videos and their segments move over, its name becomes
    an alternative name of the city, the town row goes and the ids are renumbered. Holds the mapping lock."""
    with open(MAPPING + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(MAPPING, newline='') as f:
            lines = f.read().split('\n')
        header = next(csv.reader([lines[0]]))
        rows = {i: dict(zip(header, next(csv.reader([ln])))) for i, ln in enumerate(lines[1:], 1) if ln}
        by_key = {key(r): i for i, r in rows.items()}
        flat = ('videos', 'vehicle_type', 'upload_date', 'channel')
        nested = ('time_of_day', 'start_time', 'end_time')
        done = []
        for town_key, city_key in accepted:
            if town_key not in by_key or city_key not in by_key:
                continue
            t, c = rows[by_key[town_key]], rows[by_key[city_key]]
            items = lambda r, col: [x for x in r[col].strip('[]').split(',') if x]  # noqa: E731
            lists = lambda r, col: ast.literal_eval(r[col]) if r[col].strip() not in ('', '[]') else []  # noqa: E731
            cv = {col: items(c, col) for col in flat}
            cn = {col: lists(c, col) for col in nested}
            tv = {col: items(t, col) for col in flat}
            tn = {col: lists(t, col) for col in nested}
            for j, video in enumerate(tv['videos']):
                if video in cv['videos']:  # in both: its segments join the city's entry of that video
                    k = cv['videos'].index(video)
                    for col in nested:
                        cn[col][k] = cn[col][k] + tn[col][j]
                else:
                    for col in flat:
                        cv[col].append(tv[col][j])
                    for col in nested:
                        cn[col].append(tn[col][j])
            for col in flat:
                c[col] = '[' + ','.join(cv[col]) + ']'
            for col in nested:
                c[col] = json.dumps(cn[col], separators=(',', ':'))
            aka = items(c, 'locality_aka')
            c['locality_aka'] = '[' + ','.join(aka + [t['locality']] * (t['locality'] not in aka)) + ']'
            del rows[by_key[town_key]]
            done.append(f"{t['locality']} -> {c['locality']}")
        out = [lines[0]]
        for n, (_, r) in enumerate(sorted(rows.items()), 1):
            r['id'] = str(n)
            buf = io.StringIO()
            csv.writer(buf, lineterminator='').writerow([r[h] for h in header])
            out.append(buf.getvalue())
        with open(MAPPING + '.tmp', 'w', newline='') as f:
            f.write('\n'.join(out) + ('\n' if lines[-1] == '' else ''))
        os.replace(MAPPING + '.tmp', MAPPING)
    return done


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Merge towns into cities</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root { color-scheme: light; }
body { margin: 0; font-family: Arial, sans-serif; background: #fff; color: #222; display: flex; height: 100vh; }
#list { width: 280px; overflow-y: auto; border-right: 1px solid #ccc; font-size: 13px; background: #fff; }
#list div { padding: 4px 8px; border-bottom: 1px solid #eee; cursor: pointer; }
#list .cur { background: #e3f2fd; }
#list .yes { color: #1b5e20; font-weight: bold; }
#main { flex: 1; display: flex; flex-direction: column; padding: 10px; gap: 8px; }
#maps { flex: 1; display: flex; gap: 10px; }
#maps > div { flex: 1; display: flex; flex-direction: column; }
.map { flex: 1; border: 1px solid #ccc; }
h2 { font-size: 17px; margin: 4px 0; }
button { font-size: 15px; padding: 8px 16px; margin-right: 8px; cursor: pointer; }
#merge { background: #2e7d32; color: #fff; border: 0; }
</style></head><body>
<div id="list"></div>
<div id="main">
  <div><b id="pos"></b> · <span id="dist"></span> · nothing is merged unless you press Merge (keys: Y merge, N keep
    separate, ← →)</div>
  <div id="maps">
    <div><h2 id="tname"></h2><div id="tmap" class="map"></div></div>
    <div><h2 id="cname"></h2><div id="cmap" class="map"></div></div>
  </div>
  <div><button id="merge" onclick="decide(true)">Merge into the city (Y)</button>
       <button onclick="decide(false)">Keep separate (N)</button><span id="state"></span></div>
</div>
<script>
const cands = {{ cands | tojson }}, dec = {{ decisions | tojson }};
let i = 0;
const tiles = m => L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                               { maxZoom: 18, attribution: '&copy; OpenStreetMap contributors' }).addTo(m);
const tmap = L.map('tmap'), cmap = L.map('cmap'); tiles(tmap); tiles(cmap);
const tm = L.marker([0, 0]).addTo(tmap), cm = L.marker([0, 0]).addTo(cmap);
const tOnC = L.circleMarker([0, 0], { radius: 6, color: '#c0392b' }).addTo(cmap);
const fmt = n => n.toLocaleString('en');
const list = document.getElementById('list');
cands.forEach((c, j) => {
  const d = document.createElement('div');
  d.onclick = () => show(j); list.appendChild(d);
});
function label(j) {
  const c = cands[j], d = list.children[j], yes = dec[c.town.key] === c.city.key;
  d.textContent = `${yes ? '✓ ' : ''}${c.town.name} → ${c.city.name} (${c.town.country})`;
  d.className = (j === i ? 'cur ' : '') + (yes ? 'yes' : '');
}
function show(j) {
  i = Math.max(0, Math.min(cands.length - 1, j));
  const c = cands[i], s = x => x.state ? `, ${x.state}` : '';
  document.getElementById('pos').textContent = `${i + 1} / ${cands.length}`;
  document.getElementById('dist').textContent = `${c.km} km apart`;
  document.getElementById('tname').textContent =
    `Town: ${c.town.name}${s(c.town)}, ${c.town.country} — ${fmt(c.town.pop)} inhabitants, ${c.town.videos} videos`;
  document.getElementById('cname').textContent =
    `City: ${c.city.name}${s(c.city)} — ${fmt(c.city.pop)} inhabitants (the town is the red dot)`;
  tm.setLatLng([c.town.lat, c.town.lon]); tmap.setView([c.town.lat, c.town.lon], 12);
  cm.setLatLng([c.city.lat, c.city.lon]); tOnC.setLatLng([c.town.lat, c.town.lon]);
  cmap.fitBounds([[c.town.lat, c.town.lon], [c.city.lat, c.city.lon]], { padding: [40, 40], maxZoom: 11 });
  document.getElementById('state').textContent = dec[c.town.key] === c.city.key ? 'merge accepted' : 'kept separate';
  cands.forEach((_, j) => label(j));
  list.children[i].scrollIntoView({ block: 'nearest' });
}
async function decide(yes) {
  const c = cands[i];
  let res;
  try {
    res = await fetch('decide', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ town: c.town.key, city: yes ? c.city.key : null }) });
  } catch (e) { res = null; }
  if (!res || !res.ok) {
    alert('Not saved: the page server (review_merges.py) is not running. Start it again.');
    return;
  }
  if (yes) dec[c.town.key] = c.city.key; else delete dec[c.town.key];
  show(i + 1);
}
document.addEventListener('keydown', e => {
  if (e.key === 'y' || e.key === 'Y') decide(true);
  else if (e.key === 'n' || e.key === 'N') decide(false);
  else if (e.key === 'ArrowRight') show(i + 1);
  else if (e.key === 'ArrowLeft') show(i - 1);
});
window.addEventListener('load', () => { tmap.invalidateSize(); cmap.invalidateSize(); show(0); });
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8772)
    ap.add_argument('--apply', action='store_true', help='merge the towns accepted on the page into their city')
    args = ap.parse_args()
    cands = candidates()
    if args.apply:
        dec = load_decisions()
        accepted = [(c['town']['key'], c['city']['key']) for c in cands
                    if dec.get(c['town']['key']) == c['city']['key']]
        done = merge(accepted)
        print(f'{len(done)} towns merged: {", ".join(done) or "none"}')
        return
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(PAGE, cands=cands, decisions=load_decisions())

    @app.route('/decide', methods=['POST'])
    def decide():
        d = request.get_json()
        dec = load_decisions()
        if d.get('city'):
            dec[d['town']] = d['city']
        else:
            dec.pop(d['town'], None)
        os.makedirs(os.path.dirname(DECISIONS), exist_ok=True)
        with open(DECISIONS + '.tmp', 'w') as f:
            json.dump(dec, f, indent=1, ensure_ascii=False)
        os.replace(DECISIONS + '.tmp', DECISIONS)
        return jsonify(ok=True)

    print(f'{len(cands)} candidates: http://127.0.0.1:{args.port}')
    app.run(port=args.port)


if __name__ == '__main__':
    main()
