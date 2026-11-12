"""Mapping coordinates that GeoNames does not confirm (tools/coords_check.py), to check one by one on the status page
(/coords): each with a map of the row's point, the GeoNames place it was matched to and other places of that name.

    python dashboards/coords_review.py           # (re)build _output/proposals/coords_review.json (~2 min)
    python dashboards/coords_review.py --apply   # write the decided coordinates into mapping.csv and push it

Run from the repo's top folder.

A case is a mapping row with an issue other than "ok". The suggestion is to move the point to the matched GeoNames
place, except where the country looks wrong (other_country) or GeoNames has no place of that name (no_name): there
it is to keep it. The page saves a decision per row in the same file: move (to the matched place), custom (a point
clicked on the map, e.g. one of the other places of the name), keep (the coordinates are right) or other (something
else is wrong, such as the name, state or country: for a person, with a note). A rebuild keeps the decisions of rows
whose coordinates did not change.

--apply: sets lat and lon (7 decimals) of each row decided move or custom in mapping.csv, under the review pages'
lock, if the row still has the coordinates the case was made with (else it is left for the next build), then
pushes the mapping as the hub's "Push mapping" does.
"""
import contextlib
import csv
import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/, add_video, review_hub: the top folder
from tools import coords_check  # noqa: E402

FILE = '_output/proposals/coords_review.json'
DECISIONS = {'move': 'move it to the GeoNames place', 'custom': 'move it to the point I clicked',
             'keep': 'the coordinates are right', 'other': 'something else is wrong (note)'}
WHY = {'other_country': 'nearer to places of {nearest_country} than of its own country',
       'wrong_state': 'the {match} of {match_admin1} is {match_km} km away; the point lies in {coord_admin1}',
       'far': 'the nearest {match} ({match_admin1}) is {match_km} km away',
       'no_name': 'GeoNames has no place of this name in the country; nearest place: {nearest_own} at '
                  '{nearest_own_km} km'}
_lock = threading.Lock()


@contextlib.contextmanager
def locked():
    """One writer at a time: the page (decide), a build and --apply run in separate processes."""
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with _lock, open(FILE + '.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def load():
    if not os.path.exists(FILE):
        return dict(built=None, cases=[])
    with open(FILE) as f:
        return json.load(f)


def save(data):
    with open(FILE + '.tmp', 'w') as f:
        json.dump(data, f, indent=1)
    os.replace(FILE + '.tmp', FILE)


def key(c):
    """A row as the cases know it: the ids are renumbered when the mapping is pushed (review_hub.push_mapping), so a
    case finds its row by name, state and country (and its coordinates), never by id (2026-10-25: 67 of 72 decisions
    were skipped after a renumbering)."""
    return c['locality'], c['state'], c['country']


def same_point(a, b):
    return abs(float(a[0]) - float(b[0])) < 1e-6 and abs(float(a[1]) - float(b[1])) < 1e-6


def suggest(c):
    if c['issue'] in ('far', 'wrong_state') and c.get('match_lat') is not None:
        return 'move'
    return 'other' if c['issue'] == 'other_country' else 'keep'


def case_of(res):
    c = {k: res.get(k) for k in ('id', 'locality', 'locality_aka', 'state', 'country', 'iso3', 'population', 'lat',
                                 'lon', 'issue', 'match', 'match_admin1', 'match_pop', 'match_lat', 'match_lon',
                                 'match_km', 'coord_admin1', 'same_name_places', 'nearest_own', 'nearest_own_km',
                                 'nearest_place', 'nearest_country', 'nearest_km')}
    c['alternatives'] = res.get('alternatives', [])
    c['why'] = WHY[c['issue']].format(**{k: ('?' if v is None else v) for k, v in c.items()})
    if res.get('other_states'):  # never offered as a move (coords_check, 2026-10-25)
        c['why'] += f" (the name is only in other states: {', '.join(res['other_states'])})"
    c['suggestion'] = suggest(c)
    return c


def carried(old, case):
    """The decision fields kept from the last build: those of the same row with the same coordinates, or, once
    applied, of the row now at the applied point."""
    if not old:
        return {}
    point = (case['lat'], case['lon'])
    if same_point((old['lat'], old['lon']), point) or (old.get('applied') and old.get('to')
                                                       and same_point(old['to'], point)):
        return {k: old[k] for k in ('decision', 'note', 'to', 'applied') if k in old}
    return {}


def build():
    with open('mapping.csv.lock', 'w') as lk:  # the review pages write mapping.csv under this lock
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open('mapping.csv', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    videos = {r['id']: r['videos'].strip('"[] ').split(',')[0].strip() for r in rows}
    result = coords_check.check(rows)
    old = {key(c): c for c in load()['cases']}
    cases = []
    for res in result:
        if res['issue'] == 'ok':
            continue
        c = case_of(res)
        c['video'] = videos.get(c['id'], '')
        c.update(carried(old.get(key(c)), c))
        cases.append(c)
    cases.sort(key=lambda c: (coords_check.ISSUES.index(c['issue']), -float(c.get('match_km') or 0)))
    with locked():  # the check takes minutes: keep a decision made on the page meanwhile
        latest = {key(c): c for c in load()['cases']}
        for c in cases:
            now, then = latest.get(key(c), {}), old.get(key(c), {})
            if any(now.get(k) != then.get(k) for k in ('decision', 'note', 'to')):
                c.update(carried(now, c))
        save(dict(built=time.time(), counts={k: sum(r['issue'] == k for r in result) for k in coords_check.ISSUES},
                  cases=cases))
    return cases


def decide(row_id, decision, note, to=None):
    """Saves the decision (None to undo it), note and, for custom, the point [lat, lon] for one row; returns
    dict(to=the point it would move to), or None for an unknown row."""
    if decision not in DECISIONS and decision is not None:
        raise ValueError(decision)
    if decision == 'custom':
        lat, lon = float(to[0]), float(to[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(to)
        to = [round(lat, 7), round(lon, 7)]
    with locked():
        data = load()
        for c in data['cases']:
            if c['id'] == str(row_id):
                c['decision'], c['note'] = decision, str(note or '')[:500]
                c['to'] = (to if decision == 'custom' else [c['match_lat'], c['match_lon']] if decision == 'move'
                           else None)
                c.pop('applied', None)
                save(data)
                return dict(to=c['to'])
    return None


def apply():
    import add_video  # noqa: E402: the review pages' modules, needed here only
    import review_hub
    data = load()
    todo = {key(c): c for c in data['cases'] if c.get('decision') in ('move', 'custom') and c.get('to')
            and not c.get('applied')}
    if not todo:
        return 'nothing to apply'
    moved, changed_since = [], []
    with review_hub.ps.plan_lock(add_video.FILE_PATH):  # not halfway through an Apply
        with open(add_video.FILE_PATH, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            c = todo.get(key(row))
            if not c:
                continue
            if not same_point((row['lat'], row['lon']), (c['lat'], c['lon'])):
                changed_since.append(c)
                continue
            row['lat'], row['lon'] = f"{float(c['to'][0]):.7f}".rstrip('0'), f"{float(c['to'][1]):.7f}".rstrip('0')
            moved.append(c)
        if moved:
            tmp = add_video.FILE_PATH + f'.{os.getpid()}.tmp'
            with open(tmp, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fields, lineterminator='\n')
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, add_video.FILE_PATH)
    now = time.time()
    with locked():
        latest = load()
        for c in latest['cases']:
            if key(c) in {key(m) for m in moved}:
                c['applied'] = now
            elif key(c) in {key(m) for m in changed_since}:
                c['note'] = (c.get('note') or '') + ' [not applied: the row has other coordinates now]'
        save(latest)
    missing = len(todo) - len(moved) - len(changed_since)
    msg = review_hub.push_mapping('Update mapping.csv: coordinates checked against GeoNames') if moved else ''
    return (f'{len(moved)} rows moved' + (f', {len(changed_since)} left (coordinates changed since the check)'
                                          if changed_since else '') + (f', {missing} not found (row renamed or merged)'
                                                                       if missing else '') + f'. {msg}')


def _check():
    c = case_of(dict(id='7', locality='X', lat=1.0, lon=2.0, issue='far', match='X', match_admin1='A', match_km=50.0,
                     match_lat=1.5, match_lon=2.5))
    assert c['suggestion'] == 'move' and '50.0 km' in c['why']
    assert case_of(dict(id='8', lat=0, lon=0, issue='no_name', nearest_own='Y',
                        nearest_own_km=3))['suggestion'] == 'keep'
    old = dict(lat=1.0, lon=2.0, decision='keep', note='n')
    assert carried(old, dict(lat=1.0, lon=2.0)) == dict(decision='keep', note='n')
    assert carried(old, dict(lat=1.1, lon=2.0)) == {}  # moved since: a new case
    applied = dict(lat=1.0, lon=2.0, decision='custom', to=[3.0, 4.0], applied=5)
    assert carried(applied, dict(lat=3.0, lon=4.0))['applied'] == 5  # flagged again at the applied point: decided


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Coordinates to check</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
<text y=%22.9em%22 font-size=%2290%22>📍</text></svg>">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#667085;--line:#e4e7ec;--link:#2563eb;--ok:#16a34a;--warn:#d97706;
 --bad:#dc2626}
:root[data-theme=dark]{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--muted:#8b93a1;--line:#272c36;--link:#60a5fa;
 --ok:#4ade80;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
 font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1200px}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}h1{font-size:20px;margin:0 0 4px}
.muted{color:var(--muted)}.small{font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0;
 display:grid;grid-template-columns:minmax(0,1fr);gap:12px}
.card.done{opacity:.65}
.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.maps.two{grid-template-columns:repeat(2,minmax(0,1fr))}
@media (max-width:820px){.maps,.maps.two{grid-template-columns:1fr}}
.map,.zmap{height:250px;border-radius:8px;border:1px solid var(--line)}.zcap{font-size:12px;margin:0 0 3px}
.issue{font-weight:600}.issue.other_country,.issue.wrong_state{color:var(--bad)}.issue.far{color:var(--warn)}
.issue.no_name{color:var(--muted)}
label{margin-right:12px;white-space:nowrap;cursor:pointer;display:inline-block}
input[type=text]{width:100%;padding:3px 6px;margin-top:6px;background:var(--bg);color:var(--fg);
 border:1px solid var(--line);border-radius:6px}
.saved{color:var(--ok);font-size:12px;margin-left:6px}.to{font-variant-numeric:tabular-nums}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}
button,select{padding:5px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);
 cursor:pointer}
.key span{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle}
#theme{position:fixed;right:12px;bottom:12px;z-index:9999;padding:6px 10px;border:1px solid var(--line);
border-radius:16px;background:var(--bg);color:var(--fg);font-size:13px;cursor:pointer}
</style>
{{ theme|safe }}
</head><body>
<h1>📍 Coordinates to check</h1>
<div class="muted"><a href=".">pipeline status</a> · {{ cases|length }} rows ·
 {{ cases|selectattr('decision')|list|length }} decided · {{ cases|selectattr('applied')|list|length }} applied ·
 built {{ hm(built) if built else 'never' }} from mapping.csv and GeoNames
 {% if counts %}({{ counts.ok }} rows confirmed){% endif %} · choices are saved as you click; writing them into the
 mapping is a separate step</div>
<div class="bar">
 <select id="show" onchange="filter()"><option value="open" data-label="undecided">undecided</option>
  <option value="sugg" data-label="undecided, with a suggested move">undecided, with a suggested move</option>
  <option value="all" data-label="all">all</option>
  {% for k in issues %}<option value="{{ k }}" data-label="{{ k.replace('_', ' ') }}">{{ k }}</option>{% endfor %}
 </select>
 <span class="small key"><span style="background:#dc2626"></span>row now<span
 style="background:#2563eb"></span>GeoNames
  match<span style="background:#9ca3af"></span>other places of the name<span style="background:#16a34a"></span>your
  point (click the map)</span>
</div>
{% for c in cases %}<div class="card {{ 'done' if c.decision }}" id="r{{ c.id }}" data-issue="{{ c.issue }}"
 data-sugg="{{ c.suggestion }}">
 <div>
  <div><b>{{ c.locality }}</b>{% if c.state %}, {{ c.state }}{% endif %}, {{ c.country }}
   <span class="muted small">row {{ c.id }}{% if c.locality_aka and c.locality_aka != '[]' %} ·
   aka {{ c.locality_aka }}{% endif %}{% if c.population %} · {{ '{:,}'.format(c.population|float|int) }}
   people{% endif %}</span>
   {% if c.applied %}<span class="saved">applied {{ hm(c.applied) }}</span>{% endif %}</div>
  <div class="small" style="margin:4px 0"><span class="issue {{ c.issue }}">{{ c.issue.replace('_', ' ') }}</span>:
   {{ c.why }}</div>
  <div class="small">now <a href="https://www.google.com/maps?q={{ c.lat }},{{ c.lon }}" target="_blank">{{ c.lat }},
   {{ c.lon }}</a>{% if c.match_lat is not none %} · GeoNames
   <a href="https://www.google.com/maps?q={{ c.match_lat }},{{
   c.match_lon }}" target="_blank">{{ c.match }}, {{ c.match_admin1 }}</a>{% if c.match_pop %} ({{
   '{:,}'.format(c.match_pop) }} people){% endif %}{% endif %}{% if c.same_name_places and c.same_name_places > 1 %} ·
   {{ c.same_name_places }} places of this name{% endif %}{% if c.video %} ·
   <a href="https://youtu.be/{{ c.video }}" target="_blank">a video of the row ↗</a>{% endif %}</div>
  <form onchange="send(this)" data-id="{{ c.id }}" data-suggestion="{{ c.suggestion }}" style="margin-top:8px">
   {% for k, label in decisions.items() %}{% if k != 'move' or c.match_lat is not none %}<label><input type="radio"
    name="d" value="{{ k }}" {{ 'checked' if c.decision == k }}> {{ label }}{% if k == c.suggestion %}
    <span class="muted small">(suggested)</span>{% endif %}</label>{% endif %}{% endfor %}
   <div class="small to">{% if c.to %}to {{ c.to[0] }}, {{ c.to[1] }}{% endif %}</div>
   <input type="text" name="note" placeholder="note (optional)" value="{{ c.note or '' }}"><span class="saved"></span>
  </form>
 </div>
 <div class="maps{{ '' if c.match_lat is not none else ' two' }}">
  <div><div class="zcap">overview{% if c.match_lat is none %} <span class="muted">(no GeoNames place of this
   name)</span>{% endif %}</div><div class="map" data-case='{{ c|tojson|forceescape }}'></div></div>
  <div><div class="zcap"><b style="color:#dc2626">●</b> row now, close up</div>
  <div class="zmap" data-which="now"></div></div>
  {% if c.match_lat is not none %}<div><div class="zcap"><b style="color:#2563eb">●</b> GeoNames {{ c.match }},
   {{ c.match_admin1 }}, close up</div><div class="zmap" data-which="match"></div></div>{% endif %}
 </div>
</div>{% endfor %}
<script>
const dot = (color) => ({radius: 7, color: '#fff', weight: 2, fillColor: color, fillOpacity: 1});
function draw(el) {
  const c = JSON.parse(el.dataset.case), card = el.closest('.card');
  const map = L.map(el, {scrollWheelZoom: false});
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
              {maxZoom: 18, attribution: '© OpenStreetMap'}).addTo(map);
  const pts = [[c.lat, c.lon]];
  (c.alternatives || []).forEach(a => {
    L.circleMarker([a[2], a[3]], dot('#9ca3af')).addTo(map).bindTooltip(`${a[0]}, ${a[1]} (${a[4]} km)`)
      .on('click', () => pick(card, [a[2], a[3]]));
    pts.push([a[2], a[3]]);
  });
  if (c.match_lat !== null && c.match_lat !== undefined) {
    L.polyline([[c.lat, c.lon], [c.match_lat, c.match_lon]], {color: '#2563eb', weight: 1, dashArray: '4'}).addTo(map);
    L.circleMarker([c.match_lat, c.match_lon], dot('#2563eb')).addTo(map)
      .bindTooltip(`GeoNames: ${c.match}, ${c.match_admin1}`);
    pts.push([c.match_lat, c.match_lon]);
  }
  L.circleMarker([c.lat, c.lon], dot('#dc2626')).addTo(map).bindTooltip('row now');
  card._mine = null;
  if (c.decision === 'custom' && c.to) card._mine = L.circleMarker(c.to, dot('#16a34a')).addTo(map);
  card._map = map;
  map.on('click', e => pick(card, [e.latlng.lat, e.latlng.lng]));
  if (pts.length > 1) map.fitBounds(pts, {padding: [25, 25], maxZoom: 13}); else map.setView(pts[0], 10);
  card._zooms = [];
  card.querySelectorAll('.zmap').forEach(z => {  // close-ups of the row's point and of the GeoNames place
    const at = z.dataset.which === 'now' ? [c.lat, c.lon] : [c.match_lat, c.match_lon];
    const zm = L.map(z, {scrollWheelZoom: false}).setView(at, 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                {maxZoom: 19, attribution: '© OpenStreetMap'}).addTo(zm);
    L.circleMarker([c.lat, c.lon], dot('#dc2626')).addTo(zm).bindTooltip('row now');
    if (c.match_lat !== null && c.match_lat !== undefined)
      L.circleMarker([c.match_lat, c.match_lon], dot('#2563eb')).addTo(zm).bindTooltip(`GeoNames: ${c.match}`);
    if (c.decision === 'custom' && c.to) zm._mine = L.circleMarker(c.to, dot('#16a34a')).addTo(zm);
    zm.on('click', e => pick(card, [e.latlng.lat, e.latlng.lng]));
    card._zooms.push(zm);
  });
}
function pick(card, p) {
  const map = card._map;
  if (card._mine) card._mine.setLatLng(p); else card._mine = L.circleMarker(p, dot('#16a34a')).addTo(map);
  (card._zooms || []).forEach(zm => {
    if (zm._mine) zm._mine.setLatLng(p); else zm._mine = L.circleMarker(p, dot('#16a34a')).addTo(zm);
  });
  const form = card.querySelector('form');
  form.querySelector('input[value=custom]').checked = true;
  form._to = [+p[0].toFixed(7), +p[1].toFixed(7)];
  send(form);
}
async function send(form) {
  const d = form.querySelector('input[name=d]:checked');
  const body = {id: form.dataset.id, decision: d ? d.value : null, note: form.note.value};
  if (d && d.value === 'custom') {
    if (!form._to) { form.querySelector('.saved').textContent = 'click the map for the point'; return; }
    body.to = form._to;
  }
  const r = await fetch('coords/decide', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify(body)});
  const j = r.ok ? await r.json() : null;
  form.querySelector('.saved').textContent = r.ok ? 'saved' : 'not saved: ' + r.status;
  form.querySelector('.to').textContent = j && j.to ? `to ${j.to[0]}, ${j.to[1]}` : '';
  if (r.ok) form.closest('.card').classList.toggle('done', !!d);
  counts();
}
function counts() {  // the numbers in the list, kept current as decisions are clicked
  const cards = [...document.querySelectorAll('.card')], open = c => !c.classList.contains('done');
  document.querySelectorAll('#show option').forEach(o => {
    const v = o.value, label = o.dataset.label;
    if (v === 'open') o.textContent = `${label} (${cards.filter(open).length})`;
    else if (v === 'sugg')
      o.textContent = `${label} (${cards.filter(c => open(c) && c.dataset.sugg === 'move').length})`;
    else if (v === 'all') o.textContent = `${label} (${cards.length}, ${cards.filter(c => !open(c)).length} decided)`;
    else { const k = cards.filter(c => c.dataset.issue === v);
           o.textContent = `${label} (${k.length}, ${k.filter(open).length} undecided)`; }
  });
}
function filter() {
  const v = document.getElementById('show').value;
  document.querySelectorAll('.card').forEach(c => {
    const open = !c.classList.contains('done');
    c.style.display = (v === 'all' || (v === 'sugg' && open && c.dataset.sugg === 'move') || (v === 'open' && open)
                       || c.dataset.issue === v) ? '' : 'none';
  });
}
const seen = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { seen.unobserve(e.target); draw(e.target); }
}), {rootMargin: '300px'});
document.querySelectorAll('.map').forEach(m => seen.observe(m));
counts();
filter();
</script></body></html>"""


if __name__ == '__main__':
    _check()
    coords_check._check()
    if '--apply' in sys.argv:
        print(apply())
    else:
        built = build()
        print(len(built), 'rows to check;', sum(c['suggestion'] == 'move' for c in built), 'with a suggested move')
