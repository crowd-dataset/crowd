"""
Collision-free placement of country labels on footage choropleth maps.

A label goes inside its country when it fits there; otherwise it goes next to a dot marking the country, and only
when no spot next to the dot is free, further away with a leader line. Label sizes are estimated from the text and
positions are computed in pixels of the equirectangular projection plotly uses for these maps, so the layout holds for
the saved 1600x900 figures.
"""

import json
import math
import os
import urllib.request

import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

# Country outlines and label points (`ct`) used by plotly to draw world maps.
TOPOJSON_URL = "https://cdn.plot.ly/world_110m.json"

# Label metrics for 11px Open Sans (measured in the browser).
CHAR_W = 6.2      # average character width
FLAG_W = 17       # emoji flag plus the space after it
LINE_H = 14       # line height
PAD = 5           # gap between a dot and its label
DOT = 8           # size of the box kept free around a dot

GAP = 3           # minimum space between labels
CROWD = 40        # dots closer than this (px) form a cluster
STACK = 8         # clusters with at least this many dots get their labels stacked in a column beside them

# Positions next to a dot in order of preference, with the direction each one points to (screen y grows down).
NEAR = {"middle right": (1, 0), "middle left": (-1, 0), "top center": (0, -1), "bottom center": (0, 1),
        "top right": (0.7, -0.7), "bottom right": (0.7, 0.7), "top left": (-0.7, -0.7), "bottom left": (-0.7, 0.7)}


def _load_topojson() -> dict:
    path = os.path.join(common.output_dir, "world_110m.json")
    if not os.path.exists(path):
        os.makedirs(common.output_dir, exist_ok=True)
        urllib.request.urlretrieve(TOPOJSON_URL, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def country_shapes() -> dict:
    """ISO3 -> dict(ct=(lon, lat) of plotly's label point, rings=[outer rings as [(lon, lat), ...]])."""
    topo = _load_topojson()
    (sx, sy), (tx, ty) = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:  # delta-encoded, quantised coordinates
        x = y = 0
        pts = []
        for dx, dy in arc:
            x, y = x + dx, y + dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)

    def ring(indices):
        pts = []
        for i in indices:
            a = arcs[i] if i >= 0 else arcs[~i][::-1]
            pts.extend(a if not pts else a[1:])
        return pts

    shapes = {}
    for g in topo["objects"]["countries"]["geometries"]:
        if "id" not in g or "properties" not in g:  # a few unnamed polygons
            continue
        polygons = [g["arcs"]] if g["type"] == "Polygon" else g.get("arcs", [])
        rings = [ring(p[0]) for p in polygons if p]  # outer rings only, holes ignored
        if rings:
            shapes[g["id"]] = dict(ct=tuple(g["properties"]["ct"]), rings=rings)
    return shapes


class Projection:
    """Equirectangular projection of a lon/lat view fitted and centred in a width x height plot area (as plotly)."""

    def __init__(self, view: dict, width: float, height: float):
        (lon0, lon1), (lat0, lat1) = view["lon"], view["lat"]
        self.w, self.h = width, height
        self.centre = view.get("rotation", (lon0 + lon1) / 2)
        self.mid_lon, self.mid_lat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        self.s = min(width / (lon1 - lon0), height / (lat1 - lat0))  # pixels per degree
        # pixel box of the map, centred in the plot area
        self.frame = (width / 2 - (lon1 - lon0) * self.s / 2, height / 2 - (lat1 - lat0) * self.s / 2,
                      width / 2 + (lon1 - lon0) * self.s / 2, height / 2 + (lat1 - lat0) * self.s / 2)

    def __call__(self, lon, lat):
        dlon = (lon - self.centre + 180) % 360 - 180
        return (self.w / 2 + (dlon - (self.mid_lon - self.centre)) * self.s,
                self.h / 2 - (lat - self.mid_lat) * self.s)

    def invert(self, x, y):
        return (self.centre + (self.mid_lon - self.centre) + (x - self.w / 2) / self.s,
                self.mid_lat - (y - self.h / 2) / self.s)


class AxisProjection:
    """Linear pixel mapping of a chart's axes (in axis units, e.g. log10 values), for labels on scatter plots.

    `x_range`/`y_range` are the axes' fixed ranges and `width`/`height` the plot area in pixels.
    """

    def __init__(self, x_range, y_range, width: float, height: float):
        (self.x0, self.x1), (self.y0, self.y1) = x_range, y_range
        self.w, self.h = width, height
        self.frame = (0, 0, width, height)

    def __call__(self, x, y):
        return ((x - self.x0) / (self.x1 - self.x0) * self.w, (self.y1 - y) / (self.y1 - self.y0) * self.h)

    def invert(self, px, py):
        return (self.x0 + px / self.w * (self.x1 - self.x0), self.y1 - py / self.h * (self.y1 - self.y0))


def text_width(text: str) -> float:
    """Estimated width of one line of label text; an emoji flag is a pair of regional indicator characters."""
    flags = sum(1 for c in text if 0x1F1E6 <= ord(c) <= 0x1F1FF) // 2
    rest = "".join(c for c in text if not 0x1F1E6 <= ord(c) <= 0x1F1FF).strip()
    return (FLAG_W * flags + CHAR_W * len(rest)) * 1.12 + 2  # rendered widths vary by up to ~12%


def _box(x, y, w, h, position, pad=PAD):
    """Box (x0, y0, x1, y1) of a w x h label placed at (x, y) with plotly `position`, e.g. 'top right'.
    `pad` is plotly's gap between the point and the text: ~PAD next to a marker, ~0 for text-only points."""
    v, _, hz = position.partition(" ")
    x0 = {"left": x - pad - w, "center": x - w / 2, "right": x + pad}[hz]
    y0 = {"top": y - pad - h, "middle": y - h / 2, "bottom": y + pad}[v]
    return x0, y0, x0 + w, y0 + h


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _cross(a, b, c, d):
    """Whether segments ab and cd intersect."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _inside(x, y, rings):
    hit = False
    for ring in rings:
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
    return hit


def place_labels(items: list, proj: Projection, shapes: dict) -> dict:
    """
    Place labels without overlaps.

    Args:
        items: dicts with `code`, `lines` (label as [name, value]), `anchor` ((lon, lat) inside the part of the
            country shown) and `ct` ((lon, lat) where plotly would centre the label, or None if plotly does not draw
            the country), in order of priority.
        proj: Projection of the map.
        shapes: Output of `country_shapes()`.

    Returns:
        code -> ("inside", (lon, lat)) | ("dot", (lon, lat) of the dot, textposition, value on its own line)
                | ("line", (lon, lat) of the dot, (lon, lat) of the label, textposition)
    """
    labels = []  # boxes of placed labels
    dots = []  # boxes kept free around dots
    lines = []  # leader lines as (start, end, points along them)
    bounds = (proj.frame[0] + 2, proj.frame[1] + 2, proj.frame[2] - 2, proj.frame[3] - 2)
    result = {}

    def free(box):
        grown = (box[0] - GAP, box[1] - GAP, box[2] + GAP, box[3] + GAP)
        return (bounds[0] <= box[0] and box[2] <= bounds[2] and bounds[1] <= box[1] and box[3] <= bounds[3]
                and not any(_overlaps(grown, t) for t in labels) and not any(_overlaps(box, d) for d in dots)
                and not any(box[0] <= px <= box[2] and box[1] <= py <= box[3] for *_, pts in lines for px, py in pts))

    def line_free(start, end):
        """A leader line may not cross other lines or labels; it may pass over other countries' dots."""
        n = max(int(math.dist(start, end) / 3), 1)
        pts = [(start[0] + (end[0] - start[0]) * i / n, start[1] + (end[1] - start[1]) * i / n) for i in range(n + 1)]
        pts = [p for p in pts if math.dist(p, start) > DOT / 2 and math.dist(p, end) > PAD]
        if any(t[0] - GAP <= px <= t[2] + GAP and t[1] - GAP <= py <= t[3] + GAP for t in labels for px, py in pts):
            return None
        if any(_cross(start, end, a, b) for a, b, _ in lines):
            return None
        return pts

    def dot_box(item):
        x, y = proj(*item["anchor"])
        return x - DOT / 2, y - DOT / 2, x + DOT / 2, y + DOT / 2

    # 1) labels inside countries large enough to hold them, largest countries first
    def pixel_area(item):
        rings = shapes.get(item["code"], {}).get("rings", [])
        return sum(abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1]))) for r in rings)

    def inside_spots(item):
        """Label boxes (closest to the country's label point first) whose core lies inside the country."""
        shape = shapes.get(item["code"])
        if not shape or item["ct"] is None:
            return []
        rings = [[proj(lon, lat) for lon, lat in r] for r in shape["rings"]]
        w, h = max(text_width(t) for t in item["lines"]), 2 * LINE_H
        cx, cy = proj(*item["ct"])
        spots = []
        for dx, dy in sorted(((dx, dy) for dx in range(-40, 41, 10) for dy in range(-30, 31, 10)),
                             key=lambda d: math.hypot(*d)):
            x, y = cx + dx, cy + dy
            # the label may spill a little over the border; its core must be inside the country
            core = [(x + fx * w * 0.4, y + fy * h * 0.35) for fx in (-1, 0, 1) for fy in (-1, 0, 1)]
            if all(_inside(px, py, rings) for px, py in core):
                spots.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        return spots

    spots = {item["code"]: inside_spots(item) for item in items}
    # countries whose label can never go inside get a dot; reserve their spots before any label is placed
    dots.extend(dot_box(item) for item in items if not spots[item["code"]])
    for item in sorted(items, key=pixel_area, reverse=True):
        for box in spots[item["code"]]:
            if free(box):
                labels.append(box)
                result[item["code"]] = ("inside", proj.invert((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                break

    # 2) dots for countries whose inside spots were all taken, kept free of labels
    rest = [item for item in items if item["code"] not in result]
    dots.extend(dot_box(item) for item in rest if spots[item["code"]])

    # 3) their labels: next to the dot if possible, else as close as possible with a leader line.
    # Labels are placed in order of priority. In a cluster of dots, labels prefer the side facing away from the
    # cluster and leader lines point outwards, so labels rarely wall in their neighbours' dots.
    points = {item["code"]: proj(*item["anchor"]) for item in rest}

    def outward(code):
        x, y = points[code]
        near = [p for c, p in points.items() if c != code and math.dist(p, (x, y)) < CROWD]
        if not near:
            return None
        dx, dy = x - sum(p[0] for p in near) / len(near), y - sum(p[1] for p in near) / len(near)
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 1 else (1, 0)

    out = {item["code"]: outward(item["code"]) for item in rest}

    # dense clusters (e.g., the Lesser Antilles): labels stacked in a column beside the cluster, ordered by latitude
    # so leader lines do not cross; the column moves away from the cluster until it is clear
    clusters, seen = [], set()
    for item in rest:
        if item["code"] in seen:
            continue
        group, todo = [], [item["code"]]
        while todo:
            c = todo.pop()
            if c in seen:
                continue
            seen.add(c)
            group.append(c)
            todo += [d for d in points if d not in seen and math.dist(points[c], points[d]) < CROWD]
        if len(group) >= STACK:
            clusters.append(sorted(group, key=lambda c: points[c][1]))
    by_code = {item["code"]: item for item in rest}

    def stack(group, east, shift):
        """Place the group's labels in a column `shift` px beside it; False (nothing placed) if it does not fit."""
        xs = [points[c][0] for c in group]
        lx = max(xs) + shift if east else min(xs) - shift
        step = LINE_H + GAP
        y0 = sum(points[c][1] for c in group) / len(group) - step * (len(group) - 1) / 2
        y0 = min(max(y0, bounds[1] + LINE_H), bounds[3] - LINE_H - step * (len(group) - 1))
        slot = {c: (lx, y0 + i * step) for i, c in enumerate(group)}
        for _ in range(100):  # swap the slots of crossing lines; each swap shortens the lines, so this ends
            pairs = [(a, b) for i, a in enumerate(group) for b in group[i + 1:]
                     if _cross(points[a], slot[a], points[b], slot[b])]
            if not pairs:
                break
            a, b = pairs[0]
            slot[a], slot[b] = slot[b], slot[a]
        position = "middle right" if east else "middle left"
        n_labels, n_lines = len(labels), len(lines)
        for c in group:
            box = _box(*slot[c], text_width(" ".join(by_code[c]["lines"])), LINE_H, position, pad=1)
            pts = line_free(points[c], slot[c]) if free(box) else None
            if pts is None:
                del labels[n_labels:], lines[n_lines:]
                return False
            labels.append(box)
            lines.append((points[c], slot[c], pts))
        for c in group:
            result[c] = ("line", by_code[c]["anchor"], proj.invert(*slot[c]), position)
        return True

    for group in sorted(clusters, key=len, reverse=True):  # largest first: they need the most room
        prefer_east = sum(points[c][0] for c in group) / len(group) >= (proj.frame[0] + proj.frame[2]) / 2
        any(stack(group, east, shift) for east in (prefer_east, not prefer_east) for shift in range(30, 300, 15))

    # most constrained first (fewest free spots next to the dot), then by priority
    def free_spots(item):
        x, y = points[item["code"]]
        w = text_width(" ".join(item["lines"]))
        return sum(free(_box(x, y, w, LINE_H, position)) for position in NEAR)

    todo = [item for item in rest if item["code"] not in result]
    order = {item["code"]: i for i, item in enumerate(todo)}
    todo.sort(key=lambda item: (free_spots(item), order[item["code"]]))

    def place_rest(todo):
        """Place labels next to dots or with leader lines; returns the codes that found no free spot."""
        failed = []
        for item in todo:
            x, y = points[item["code"]]
            o = out[item["code"]]
            # one line ("name value"), or if that does not fit anywhere next to the dot, two narrower lines
            one = (text_width(" ".join(item["lines"])), LINE_H, False)
            two = (max(text_width(t) for t in item["lines"]), 2 * LINE_H, True)
            near = sorted(NEAR, key=lambda pos: -(NEAR[pos][0] * o[0] + NEAR[pos][1] * o[1])) if o else NEAR
            w, h = one[:2]
            for (bw, bh, stacked), position in ((size, pos) for size in (one, two) for pos in near):
                box = _box(x, y, bw, bh, position)
                if free(box):
                    labels.append(box)
                    result[item["code"]] = ("dot", item["anchor"], position, stacked)
                    break
            else:
                # angles (counter-clockwise from east) closest to the outward direction first
                angles = sorted((math.pi * k / 16 for k in range(32)),
                                key=lambda a: -(math.cos(a) * o[0] - math.sin(a) * o[1]) if o else 0)
                for r in range(30, 600, 15):
                    for angle in angles:
                        lx, ly = x + r * math.cos(angle), y - r * math.sin(angle)
                        position = "middle right" if math.cos(angle) >= 0 else "middle left"
                        box = _box(lx, ly, w, h, position, pad=1)
                        pts = line_free((x, y), (lx, ly)) if free(box) else None
                        if pts is not None:
                            labels.append(box)
                            lines.append(((x, y), (lx, ly), pts))
                            result[item["code"]] = ("line", item["anchor"], proj.invert(lx, ly), position)
                            break
                    if item["code"] in result:
                        break
                else:  # nowhere free: keep the label next to its dot
                    failed.append(item["code"])
        return failed

    # rip-up and retry: labels that found no spot go first in the next round
    state = (labels[:], lines[:], dict(result))
    for attempt in range(5):
        if attempt:
            labels[:], lines[:], result = state[0][:], state[1][:], dict(state[2])
        failed = place_rest(todo)
        if not failed:
            break
        todo = sorted(todo, key=lambda item: item["code"] not in failed)
    for code in failed:  # nowhere free even so: keep the label next to its dot
        logger.warning(f"No free spot for the label of {code}; it may overlap other labels.")
        result[code] = ("dot", next(i["anchor"] for i in todo if i["code"] == code), "middle right", False)
    return result


def layout_countries(countries: list, proj: Projection, split=None) -> dict:
    """
    Place country labels on a world-scope map (see `place_labels` for the result).

    Args:
        countries: (ISO3, [label, value], (lon, lat) of the country's cities) in order of priority.
        proj: Projection of the map.
        split: ISO3 codes of countries split across maps (e.g., continents), anchored at their cities.
    """
    shapes = country_shapes()
    split = set(split or [])

    def in_view(lon, lat):
        x, y = proj(lon, lat)
        return proj.frame[0] <= x <= proj.frame[2] and proj.frame[1] <= y <= proj.frame[3]

    items = []
    for code, lines, cities in countries:
        if not in_view(*cities):  # e.g., Svalbard on a map of Europe: its cities are not on the map
            continue
        drawn = code in shapes
        # countries plotly does not draw, split across maps or centred outside the view are anchored at their cities
        anchor = cities if not drawn or code in split or not in_view(*shapes[code]["ct"]) else shapes[code]["ct"]
        items.append(dict(code=code, lines=lines, anchor=anchor, ct=anchor if drawn else None))
    return place_labels(items, proj, shapes)


if __name__ == "__main__":
    # self-check: projection round trip, box geometry and collision-free placement of three crowded labels
    p = Projection(dict(lon=(0, 20), lat=(0, 10)), 400, 300)
    assert p(10, 5) == (200, 150) and p.invert(*p(3, 7)) == (3, 7)
    assert _box(100, 100, 20, 10, "middle right") == (105, 95, 125, 105)
    items = [dict(code=c, lines=[c, "1"], anchor=(10, 5), ct=None) for c in ("AAA", "BBB", "CCC")]
    placed = place_labels(items, p, {})
    boxes = [_box(*p(*r[1]), text_width("AAA 1"), LINE_H * (1 + r[3]), r[2]) if r[0] == "dot" else
             _box(*p(*r[2]), text_width("AAA 1"), LINE_H, r[3], pad=1) for r in placed.values()]
    assert all(not _overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), boxes
    print("ok", placed)
