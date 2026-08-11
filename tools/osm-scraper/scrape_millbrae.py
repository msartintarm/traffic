#!/usr/bin/env python3
"""Scrape a drivable road graph from OpenStreetMap (via Overpass) and emit the
JSON the engine's `sim::map::OsmMap` consumes.

The output schema is the contract between this tool and the Rust engine:

    {
      "meta":  {"place", "bbox", "origin"},
      "nodes": [{"osm_id", "x", "y", "control", "signal"?}],
      "links": [{"from_osm", "to_osm", "lanes", "speed_limit", "road_class", "turn_lanes"?}]
    }

`x`/`y` are metres in a local equirectangular projection about the bbox centre
(engine geometry is planar); `control` is one of uncontrolled|signal|stop|yield;
`turn_lanes` (optional) is the OSM `turn:lanes` string for that direction;
`links` are already directed (two-way streets are emitted as two links), matching
`LinkSpec`. Ways are split at every intersection node so a link spans exactly one
block, which is what the signal/movement model expects.

The bounding box is an input, never checked in. Supply it one of three ways
(highest precedence first): `--bbox S W N E`, `--bbox-file PATH` (a gitignored
file containing `S W N E`), or the `TRAFFIC_BBOX="S W N E"` environment variable.

Usage:
    python3 scrape_millbrae.py --bbox-file bbox.local --out millbrae.json
"""

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
RETRYABLE_STATUS = {429, 502, 503, 504}

DRIVABLE = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link", "living_street",
}

# `--highways-only`: just the grade-separated freeway network and its ramps (the
# on/off-ramps are the "exits"). Everything a peninsula freeway scenario needs and
# nothing else — a tiny download even over a large bbox.
FREEWAY = {"motorway", "motorway_link", "trunk", "trunk_link"}

DEFAULT_SPEED_MPH = {
    "motorway": 65, "trunk": 55, "primary": 35, "secondary": 35,
    "tertiary": 30, "residential": 25, "unclassified": 25, "living_street": 15,
}

# Bend-point simplification tolerance (metres). A whole-city drivable scrape is dominated by
# way geometry (hundreds of thousands of bend points), most of them redundant on near-straight
# runs. Douglas–Peucker at half a lane width thins those with no perceptible loss — the driven
# path and the drawn ribbon are unchanged — and cuts both the file size and the engine's
# per-link geometry cost. Junction endpoints are kept exactly (they carry the topology).
SIMPLIFY_TOL_M = 0.5


def overpass_query(bbox, classes=None):
    s, w, n, e = bbox
    # Restricting to specific highway classes server-side keeps a large-area scrape
    # (e.g. the whole peninsula's freeways) a small download instead of every street.
    if classes:
        selector = f'way["highway"~"^({"|".join(sorted(classes))})$"]({s},{w},{n},{e});'
    else:
        selector = f'way["highway"]({s},{w},{n},{e});'
    return f"""
    [out:json][timeout:90];
    (
      {selector}
    );
    (._;>;);
    out body;
    """


def fetch(bbox, classes=None, attempts=6):
    return fetch_query(overpass_query(bbox, classes), attempts)


def fetch_query(query, attempts=6):
    body = urllib.parse.urlencode({"data": query}).encode()
    headers = {
        "User-Agent": "traffic-sim-osm-scraper/0.1 (github traffic sim)",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    last = None
    for attempt in range(attempts):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except urllib.error.HTTPError as err:
            last = err
            if err.code not in RETRYABLE_STATUS:
                raise
        except (urllib.error.URLError, TimeoutError) as err:
            last = err
        wait = min(2 ** attempt, 30)
        print(f"overpass {url} failed ({last}); retry {attempt + 1}/{attempts} in {wait}s")
        time.sleep(wait)
    raise SystemExit(f"overpass unreachable after {attempts} attempts: {last}")


def parse_speed_mps(tags, highway):
    raw = tags.get("maxspeed")
    mph = DEFAULT_SPEED_MPH.get(highway, 25)
    if raw:
        token = raw.split()[0]
        try:
            v = float(token)
            mph = v if "mph" in raw else v / 1.60934
        except ValueError:
            pass
    return round(mph * 0.44704, 2)


def parse_layer(tags):
    raw = tags.get("layer")
    if raw is not None:
        try:
            return int(float(raw))
        except ValueError:
            pass
    if tags.get("bridge") in ("yes", "viaduct", "true", "1"):
        return 1
    if tags.get("tunnel") in ("yes", "true", "1"):
        return -1
    return 0


def parse_lanes(tags, oneway):
    try:
        lanes = int(tags.get("lanes", ""))
    except ValueError:
        lanes = 0
    if lanes <= 0:
        return 1
    return lanes if oneway else max(1, lanes // 2)


def project(lat, lon, lat0, lon0):
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * 6371000.0
    y = math.radians(lat - lat0) * 6371000.0
    return round(x, 2), round(y, 2)


# --- land use (`--landuse`) -------------------------------------------------
# A second, lightweight Overpass pass: land-use polygons and point-of-interest
# nodes, rasterized onto a coarse grid, weight each link's trip *production*
# (res_weight — homes) and *attraction* (attr_weight — shops, jobs, campuses).
# The engine's demand generator reads these to place origins in residential
# fabric and pull destinations toward activity centres instead of a uniform
# scatter. Both default to neutral when the pass is skipped.

LANDUSE_CELL_M = 150.0
ATTR_LANDUSE = {"commercial": 1.0, "retail": 1.0, "industrial": 0.5}
POI_AMENITIES = (
    "restaurant|cafe|fast_food|bar|bank|school|college|university|hospital|"
    "clinic|pharmacy|cinema|theatre|library|townhall|marketplace|place_of_worship"
)


def landuse_query(bbox):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:90];
    (
      way["landuse"~"^(residential|commercial|retail|industrial)$"]({s},{w},{n},{e});
      node["shop"]({s},{w},{n},{e});
      node["amenity"~"^({POI_AMENITIES})$"]({s},{w},{n},{e});
      node["office"]({s},{w},{n},{e});
    );
    (._;>;);
    out body;
    """


def point_in_ring(x, y, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class LandUseGrid:
    """Residential / attraction signals on a coarse metre grid: polygon interiors
    painted by point-in-ring over the cells the polygon's bbox covers, POIs
    accumulated into their cell. Reads are 3x3-smoothed so a street bordering a
    zone still feels it."""

    def __init__(self, raw, lat0, lon0):
        pts = {
            e["id"]: project(e["lat"], e["lon"], lat0, lon0)
            for e in raw["elements"] if e["type"] == "node"
        }
        self.res, self.attr = defaultdict(float), defaultdict(float)
        cs = LANDUSE_CELL_M
        for e in raw["elements"]:
            tags = e.get("tags", {})
            if e["type"] == "way":
                lu = tags.get("landuse")
                ring = [pts[n] for n in e["nodes"] if n in pts]
                if lu is None or len(ring) < 3:
                    continue
                xs, ys = [p[0] for p in ring], [p[1] for p in ring]
                ci0, ci1 = int(min(xs) // cs), int(max(xs) // cs)
                cj0, cj1 = int(min(ys) // cs), int(max(ys) // cs)
                for ci in range(ci0, ci1 + 1):
                    for cj in range(cj0, cj1 + 1):
                        centre = ((ci + 0.5) * cs, (cj + 0.5) * cs)
                        if not point_in_ring(*centre, ring):
                            continue
                        if lu == "residential":
                            self.res[ci, cj] = 1.0
                        else:
                            self.attr[ci, cj] = max(self.attr[ci, cj], ATTR_LANDUSE[lu])
            elif e["type"] == "node" and any(k in tags for k in ("shop", "amenity", "office")):
                x, y = pts[e["id"]]
                self.attr[int(x // cs), int(y // cs)] += 0.25

    def _smooth(self, grid, x, y):
        ci, cj = int(x // LANDUSE_CELL_M), int(y // LANDUSE_CELL_M)
        cells = [(ci + di, cj + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        return sum(grid[c] for c in cells if c in grid) / 9.0

    def weights(self, x, y):
        res = self._smooth(self.res, x, y)
        attr = min(self._smooth(self.attr, x, y), 3.0)
        return round(0.3 + 1.4 * res, 2), round(0.3 + 1.2 * attr, 2)


def build(raw, bbox, place, drivable=DRIVABLE, landuse=None):
    nodes = {e["id"]: e for e in raw["elements"] if e["type"] == "node"}
    ways = [
        e for e in raw["elements"]
        if e["type"] == "way" and e.get("tags", {}).get("highway") in drivable
    ]

    usage = defaultdict(int)
    for way in ways:
        for nid in way["nodes"]:
            usage[nid] += 1
    for way in ways:
        for nid in (way["nodes"][0], way["nodes"][-1]):
            usage[nid] += 1

    def is_junction(nid):
        tags = nodes[nid].get("tags", {})
        # A level crossing must survive as a first-class node (like a signal):
        # the engine gates traffic there on the train timetable.
        return usage[nid] >= 2 or tags.get("highway") == "traffic_signals" or tags.get("railway") == "level_crossing"

    lat0 = (bbox[0] + bbox[2]) / 2
    lon0 = (bbox[1] + bbox[3]) / 2

    out_nodes, out_links, emitted = {}, [], set()

    def emit_node(nid):
        if nid in out_nodes:
            return
        n = nodes[nid]
        tags = n.get("tags", {})
        control = "uncontrolled"
        signal = None
        if tags.get("highway") == "traffic_signals":
            control, signal = "signal", {"green_secs": 25.0, "yellow_secs": 4.0, "offset": 0.0}
        elif tags.get("highway") == "stop":
            control = "stop"
        elif tags.get("highway") == "give_way":
            control = "yield"
        x, y = project(n["lat"], n["lon"], lat0, lon0)
        out_nodes[nid] = {"osm_id": nid, "x": x, "y": y, "control": control}
        if signal:
            out_nodes[nid]["signal"] = signal
        if tags.get("railway") == "level_crossing":
            # The engine closes these to road traffic on the train timetable.
            out_nodes[nid]["rail_crossing"] = True

    def geom_of(node_ids):
        return [list(project(nodes[nid]["lat"], nodes[nid]["lon"], lat0, lon0)) for nid in node_ids]

    # Douglas–Peucker on a projected metre-space polyline: drop bend points that lie within
    # SIMPLIFY_TOL_M of the chord they'd otherwise interrupt. Below half a lane width the
    # thinned curve is indistinguishable when driven or drawn, so this shrinks the file (and
    # the engine's per-link geometry work) with no perceptible loss of resolution. Iterative
    # so a long straight way can't blow the recursion limit.
    def simplify(pts, tol=SIMPLIFY_TOL_M):
        if len(pts) < 3:
            return pts
        keep = [False] * len(pts)
        keep[0] = keep[-1] = True
        stack = [(0, len(pts) - 1)]
        while stack:
            lo, hi = stack.pop()
            if hi <= lo + 1:
                continue
            x1, y1 = pts[lo]
            x2, y2 = pts[hi]
            dx, dy = x2 - x1, y2 - y1
            dd = dx * dx + dy * dy
            imax, dmax = lo, -1.0
            for i in range(lo + 1, hi):
                px, py = pts[i]
                if dd == 0.0:
                    d = math.hypot(px - x1, py - y1)
                else:
                    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / dd))
                    d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
                if d > dmax:
                    imax, dmax = i, d
            if dmax > tol:
                keep[imax] = True
                stack.append((lo, imax))
                stack.append((imax, hi))
        return [p for p, k in zip(pts, keep) if k]

    def emit_link(a, b, lanes, speed, geometry, name, ref, layer, road_class, turn_lanes, hov_lanes=None):
        if (a, b) in emitted:
            return
        emitted.add((a, b))
        # Thin the bend points against the full chord (junction endpoints included so the
        # deviation is measured correctly), then keep only the interior — the endpoints are
        # the `from_osm`/`to_osm` nodes and are stored there, not in the geometry.
        ea, eb = out_nodes[a], out_nodes[b]
        geometry = simplify([[ea["x"], ea["y"]], *geometry, [eb["x"], eb["y"]]])[1:-1]
        link = {"from_osm": a, "to_osm": b, "lanes": lanes, "speed_limit": speed, "geometry": geometry}
        if name:
            link["name"] = name  # road name, e.g. "El Camino Real"
        if ref:
            link["ref"] = ref  # route ref, e.g. "CA 82" — used to match real counts
        if layer:
            link["layer"] = layer  # grade separation for render z-order
        # OSM highway class (motorway, motorway_link, primary, residential, …) — lets
        # the engine model freeway↔ramp interchanges as free-flow diverges/merges
        # instead of stop-controlled intersections.
        link["road_class"] = road_class
        if turn_lanes:
            # OSM turn:lanes for this direction, e.g. "left|through|through;right" —
            # the renderer paints the lane-use arrows from it.
            link["turn_lanes"] = turn_lanes
        if hov_lanes:
            # OSM hov:lanes for this direction ("designated|no|no…"), median
            # outward — the engine restricts those lanes to eligible vehicles
            # (the US-101 express/HOV lanes).
            link["hov_lanes"] = hov_lanes
        if landuse:
            # Trip production/attraction weights from the land-use grid at the
            # link's midpoint — the engine tilts demand origins toward homes and
            # destinations toward activity centres.
            res_w, attr_w = landuse.weights((ea["x"] + eb["x"]) / 2, (ea["y"] + eb["y"]) / 2)
            link["res_weight"] = res_w
            link["attr_weight"] = attr_w
        out_links.append(link)

    for way in ways:
        tags = way["tags"]
        highway = tags["highway"]
        oneway = tags.get("oneway") in ("yes", "true", "1") or highway == "motorway"
        lanes = parse_lanes(tags, oneway)
        speed = parse_speed_mps(tags, highway)
        name = tags.get("name")
        ref = tags.get("ref")
        layer = parse_layer(tags)
        # OSM turn:lanes is ordered left→right in each direction of travel. A
        # two-way way splits it into :forward / :backward; a oneway carries it bare.
        tl_forward = tags.get("turn:lanes:forward") or (tags.get("turn:lanes") if oneway else None)
        tl_backward = tags.get("turn:lanes:backward")
        # Per-lane HOV designation, same left→right per-direction convention as
        # turn:lanes; a bare `hov=designated` marks the whole way (rare).
        hov_forward = tags.get("hov:lanes:forward") or (tags.get("hov:lanes") if oneway else None)
        hov_backward = tags.get("hov:lanes:backward")
        if not hov_forward and tags.get("hov") == "designated":
            hov_forward = "|".join(["designated"] * lanes)
            hov_backward = hov_backward or hov_forward

        seq = way["nodes"]
        block_start = 0
        for i in range(1, len(seq)):
            if not (is_junction(seq[i]) or i == len(seq) - 1):
                continue
            a, b = seq[block_start], seq[i]
            if a != b:
                emit_node(a)
                emit_node(b)
                mid = geom_of(seq[block_start + 1 : i])  # intermediate bend points
                emit_link(a, b, lanes, speed, mid, name, ref, layer, highway, tl_forward, hov_forward)
                if not oneway:
                    emit_link(b, a, lanes, speed, list(reversed(mid)), name, ref, layer, highway, tl_backward, hov_backward)
            block_start = i

    return {
        "meta": {"place": place, "bbox": bbox, "origin": [lat0, lon0]},
        "nodes": list(out_nodes.values()),
        "links": out_links,
    }


def resolve_bbox(args):
    if args.bbox:
        return tuple(args.bbox)
    raw = None
    if args.bbox_file:
        with open(args.bbox_file) as f:
            raw = f.read()
    elif os.environ.get("TRAFFIC_BBOX"):
        raw = os.environ["TRAFFIC_BBOX"]
    if not raw:
        raise SystemExit(
            "no bounding box: pass --bbox S W N E, --bbox-file PATH, or set TRAFFIC_BBOX"
        )
    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        raise SystemExit(f"expected 4 bbox values (S W N E), got {len(parts)}")
    return tuple(float(p) for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="millbrae.json")
    ap.add_argument("--place", default="Millbrae, CA")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    ap.add_argument("--bbox-file", dest="bbox_file")
    ap.add_argument(
        "--highways-only", action="store_true",
        help="keep only freeways and their ramps/exits (motorway/trunk + _link)",
    )
    ap.add_argument(
        "--landuse", action="store_true",
        help="also scrape land-use polygons and POIs; weight each link's trip production/attraction",
    )
    args = ap.parse_args()

    bbox = resolve_bbox(args)
    classes = FREEWAY if args.highways_only else None
    landuse = None
    if args.landuse:
        lat0, lon0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        raw = fetch_query(landuse_query(bbox))
        landuse = LandUseGrid(raw, lat0, lon0)
        print(f"land use: {len(landuse.res)} residential cells, {len(landuse.attr)} attraction cells")
    graph = build(fetch(bbox, classes), bbox, args.place, classes or DRIVABLE, landuse)
    if not args.highways_only:
        # Curbside bus stops: the engine dwells buses at these service positions.
        lat0, lon0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        s, w, n, e = bbox
        stops = fetch_query(f'[out:json][timeout:60]; node["highway"="bus_stop"]({s},{w},{n},{e}); out;')
        graph["bus_stops"] = [
            list(project(el["lat"], el["lon"], lat0, lon0)) for el in stops.get("elements", [])
        ]
        print(f"bus stops: {len(graph['bus_stops'])}")
        # Named bus lines: each route relation's member ways stitched into one
        # sampled polyline; the engine resolves it to a link chain and runs
        # scheduled buses along it.
        rels = fetch_query(
            f'[out:json][timeout:90]; rel["route"="bus"]({s},{w},{n},{e}); out body; way(r); out geom;'
        )
        ways = {el["id"]: el for el in rels.get("elements", []) if el["type"] == "way"}
        routes = []
        for el in rels.get("elements", []):
            if el["type"] != "relation":
                continue
            tags = el.get("tags", {})
            name = tags.get("ref") or tags.get("name") or f"route {el['id']}"
            pts = []
            for m in el.get("members", []):
                if m.get("type") != "way" or m.get("role") in ("platform", "stop"):
                    continue
                geom = ways.get(m["ref"], {}).get("geometry")
                if not geom:
                    continue
                # Route relations run far past the box; only the in-box portion
                # can resolve onto the scraped network.
                geom = [g for g in geom if s <= g["lat"] <= n and w <= g["lon"] <= e]
                if len(geom) < 2:
                    continue
                seg = [project(g["lat"], g["lon"], lat0, lon0) for g in geom]
                # Orient each way to continue from the stitched end (relations
                # are ordered but member ways face either way).
                if pts:
                    d_fwd = (seg[0][0] - pts[-1][0]) ** 2 + (seg[0][1] - pts[-1][1]) ** 2
                    d_rev = (seg[-1][0] - pts[-1][0]) ** 2 + (seg[-1][1] - pts[-1][1]) ** 2
                    if d_rev < d_fwd:
                        seg.reverse()
                pts.extend(seg)
            # Thin to ~40 m samples; the engine only needs a link-resolvable trace.
            sampled, acc = [], 1e9
            for i, p in enumerate(pts):
                if i:
                    acc += ((p[0] - pts[i - 1][0]) ** 2 + (p[1] - pts[i - 1][1]) ** 2) ** 0.5
                if acc >= 40.0:
                    sampled.append([round(p[0], 1), round(p[1], 1)])
                    acc = 0.0
            if len(sampled) >= 5:
                routes.append({"name": name, "pts": sampled})
        graph["bus_routes"] = routes
        print(f"bus routes: {len(routes)}")
    with open(args.out, "w") as f:
        json.dump(graph, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(graph['nodes'])} nodes, {len(graph['links'])} links")


if __name__ == "__main__":
    main()
