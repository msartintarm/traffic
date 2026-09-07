#!/usr/bin/env python3
"""Scrape a drivable road graph from OpenStreetMap (via Overpass) and emit the
JSON the engine's `sim::map::OsmMap` consumes.

The output schema is the contract between this tool and the Rust engine:

    {
      "meta":  {"place", "bbox", "origin"},
      "nodes": [{"osm_id", "x", "y", "control", "signal"?}],
      "links": [{"from_osm", "to_osm", "lanes", "speed_limit", "road_class", "turn_lanes"?}],
      "restrictions": [{"from": [a, via], "to": [via, b], "kind"}],
      "rail_lines": [{"kind", "pts", "speeds": [[seg_idx, mps]], "layers": [[seg_idx, layer]], "name"?}],
      "rail_stations": [{"x", "y", "kind", "name"?}],
      "rail_platforms": [[[x, y], ...]]
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


def overpass_query(bbox, classes=None, area=None):
    s, w, n, e = bbox
    # Restricting to specific highway classes server-side keeps a large-area scrape
    # (e.g. the whole peninsula's freeways) a small download instead of every street.
    cls = f'"highway"~"^({"|".join(sorted(classes))})$"' if classes else '"highway"'
    if area:
        # Clip to an admin boundary (e.g. a whole county). `way(area)` keeps ways
        # with any node inside, so boundary-crossing blocks aren't chopped mid-span.
        # A county's worth of streets is a big response — give the server room.
        name, level = area
        return f"""
    [out:json][timeout:600];
    area["name"="{name}"]["admin_level"="{level}"]->.a;
    (
      way(area.a)[{cls}];
    );
    (._;>;);
    out body;
    """
    return f"""
    [out:json][timeout:90];
    (
      way[{cls}]({s},{w},{n},{e});
    );
    (._;>;);
    out body;
    """


def resolve_area(name, level):
    """The bounding box of an admin boundary (for the local projection + the
    supplementary rail/bus/restriction queries, which stay bbox-based). The road
    scrape itself clips to the area, not this box."""
    els = fetch_query(f'[out:json][timeout:120]; rel["name"="{name}"]["admin_level"="{level}"]; out bb;').get("elements", [])
    rels = [e for e in els if e.get("type") == "relation" and "bounds" in e]
    if not rels:
        raise SystemExit(f"no admin boundary named '{name}' at admin_level {level}")
    if len(rels) > 1:
        print(f"warning: {len(rels)} boundaries match '{name}' @ level {level}; using the first")
    b = rels[0]["bounds"]
    return (b["minlat"], b["minlon"], b["maxlat"], b["maxlon"])


def fetch(bbox, classes=None, area=None, attempts=6):
    return fetch_query(overpass_query(bbox, classes, area), attempts)


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
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
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


def build(raw, bbox, place, drivable=DRIVABLE, landuse=None, turn_rels=None):
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

    def is_road_signal(tags):
        # A mid-block pedestrian signal is not a junction controller: promoting
        # it to one drags a phantom signal onto the nearest crossing (the engine
        # relocates stop-line signals junction-ward). With no pedestrians
        # modelled, it is free-flow road.
        return tags.get("highway") == "traffic_signals" and tags.get("traffic_signals") != "pedestrian_crossing"

    def is_junction(nid):
        tags = nodes[nid].get("tags", {})
        # A level crossing must survive as a first-class node (like a signal):
        # the engine gates traffic there on the train timetable.
        return usage[nid] >= 2 or is_road_signal(tags) or tags.get("railway") == "level_crossing"

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
        if is_road_signal(tags):
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

    SIGN_KIND = {"stop": "stop", "give_way": "yield"}
    SIGN_RANK = {None: 0, "yield": 1, "stop": 2}

    def block_signs(seq, lo, hi):
        # Stop/give_way nodes interior to a block: OSM surveys the sign on the
        # way at its stop line, not on the junction node — per travel direction.
        # `direction` names the controlled direction; absent, the sign binds
        # toward the nearer block end (signs stand by the junction they
        # protect). Returns the strongest (forward, backward) sign, each the
        # engine's per-approach `sign` on the corresponding directed link.
        fwd = bwd = None
        for k in range(lo + 1, hi):
            tags = nodes[seq[k]].get("tags", {})
            kind = SIGN_KIND.get(tags.get("highway"))
            if not kind:
                continue
            d = tags.get("direction")
            if d not in ("forward", "backward", "both"):
                sx, sy = project(nodes[seq[k]]["lat"], nodes[seq[k]]["lon"], lat0, lon0)
                ends = [project(nodes[seq[j]]["lat"], nodes[seq[j]]["lon"], lat0, lon0) for j in (lo, hi)]
                to_lo = math.hypot(sx - ends[0][0], sy - ends[0][1])
                to_hi = math.hypot(sx - ends[1][0], sy - ends[1][1])
                d = "forward" if to_hi <= to_lo else "backward"
            if d in ("forward", "both") and SIGN_RANK[kind] > SIGN_RANK[fwd]:
                fwd = kind
            if d in ("backward", "both") and SIGN_RANK[kind] > SIGN_RANK[bwd]:
                bwd = kind
        return fwd, bwd

    def emit_link(a, b, lanes, speed, geometry, name, ref, layer, road_class, turn_lanes, hov_lanes=None, sign=None):
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
        if sign:
            # Per-approach stop/yield ("stop"|"yield"): this directed link
            # serves a line at its downstream junction; the cross street rolls.
            link["sign"] = sign
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
                fwd_sign, bwd_sign = block_signs(seq, block_start, i)
                emit_link(a, b, lanes, speed, mid, name, ref, layer, highway, tl_forward, hov_forward, fwd_sign)
                if not oneway:
                    emit_link(b, a, lanes, speed, list(reversed(mid)), name, ref, layer, highway, tl_backward, hov_backward, bwd_sign)
            block_start = i

    graph = {
        "meta": {"place": place, "bbox": bbox, "origin": [lat0, lon0]},
        "nodes": list(out_nodes.values()),
        "links": out_links,
    }
    if turn_rels:
        graph["restrictions"] = resolve_restrictions(turn_rels, ways, is_junction)
    return graph


# A turn restriction relation names a `from` way, a `via` node (or way), and a
# `to` way; the engine's movement model wants it as a pair of *emitted links*
# — (approach-block, exit-block) node pairs — so resolution walks each way from
# the via point to the adjacent block boundary (the same is_junction split the
# link emitter uses). Best-effort: conditional/exempted relations, multi-way
# vias, and ambiguous unsplit two-way ways are skipped and counted.
EXEMPT_ALL_CARS = {"motorcar", "motor_vehicle"}


def resolve_restrictions(rels, ways, is_junction):
    by_id = {w["id"]: w for w in ways}

    def boundary_from(seq, k, step):
        # The block boundary adjacent to seq[k], walking by `step`: the first
        # split node (or the way's end — the emitter always splits there).
        i = k + step
        while 0 < i < len(seq) - 1 and not is_junction(seq[i]):
            i += step
        return seq[i] if 0 <= i < len(seq) else None

    def walk(way, via, toward_via):
        # The neighbouring block-boundary node of `via` along `way`, on the
        # approach side (`toward_via`, against travel) or the exit side (with
        # travel). None when the geometry can't be resolved: via absent, a loop,
        # a oneway entered/left against its grain, or via interior to an unsplit
        # two-way way (the travel arm is then ambiguous).
        seq = way["nodes"]
        if seq.count(via) != 1:
            return None
        tags = way.get("tags", {})
        oneway = tags.get("oneway") in ("yes", "true", "1") or tags.get("highway") == "motorway"
        k = seq.index(via)
        interior = 0 < k < len(seq) - 1
        if interior and not oneway:
            return None
        if toward_via:
            # Approach travel ends at via. At the way's first node the approach
            # ran against node order (impossible on a oneway); anywhere else it
            # followed node order, so the boundary lies at lower indices.
            if k == 0:
                if oneway:
                    return None
                step = 1
            else:
                step = -1
        else:
            # Exit travel starts at via. At the way's last node it runs against
            # node order (impossible on a oneway); anywhere else it follows it.
            if k == len(seq) - 1:
                if oneway:
                    return None
                step = -1
            else:
                step = 1
        return boundary_from(seq, k, step)

    out, seen, skipped = [], set(), defaultdict(int)
    for rel in rels:
        tags = rel.get("tags", {})
        kind = tags.get("restriction") or tags.get("restriction:motorcar")
        if not kind:
            skipped["conditional/other-class"] += 1
            continue
        if not (kind.startswith("no_") or kind.startswith("only_")):
            skipped["unknown-kind"] += 1
            continue
        if EXEMPT_ALL_CARS & set(tags.get("except", "").split(";")):
            skipped["cars-exempt"] += 1
            continue
        members = rel.get("members", [])
        from_ways = [m["ref"] for m in members if m.get("role") == "from" and m["type"] == "way"]
        to_ways = [m["ref"] for m in members if m.get("role") == "to" and m["type"] == "way"]
        via_nodes = [m["ref"] for m in members if m.get("role") == "via" and m["type"] == "node"]
        via_ways = [m["ref"] for m in members if m.get("role") == "via" and m["type"] == "way"]
        if len(from_ways) != 1 or len(to_ways) != 1:
            skipped["multi-from/to"] += 1
            continue
        fw, tw = by_id.get(from_ways[0]), by_id.get(to_ways[0])
        if fw is None or tw is None:
            skipped["way-outside-graph"] += 1
            continue
        if len(via_nodes) == 1 and not via_ways:
            n1 = n2 = via_nodes[0]
        elif len(via_ways) == 1 and not via_nodes:
            vw = by_id.get(via_ways[0])
            if vw is None:
                skipped["way-outside-graph"] += 1
                continue
            ends = (vw["nodes"][0], vw["nodes"][-1])
            n1 = next((e for e in ends if e in fw["nodes"]), None)
            n2 = next((e for e in ends if e != n1 and e in tw["nodes"]), None)
            if n1 is None or n2 is None:
                skipped["via-way-detached"] += 1
                continue
        else:
            skipped["complex-via"] += 1
            continue
        a = walk(fw, n1, toward_via=True)
        b = walk(tw, n2, toward_via=False)
        if a is None or b is None or a == n1 or b == n2:
            skipped["unresolvable-geometry"] += 1
            continue
        if (a, n1) == (b, n2) or (n1 == n2 and a == b):
            continue  # a u-turn back onto the same block; the engine never wires those
        entry = (a, n1, n2, b, kind)
        if entry in seen:
            continue
        seen.add(entry)
        out.append({"from": [a, n1], "to": [n2, b], "kind": kind})
    if skipped:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items()))
        print(f"turn restrictions: {len(out)} resolved; skipped {detail}")
    else:
        print(f"turn restrictions: {len(out)} resolved")
    return out


# --- rail (`rail_lines` / `rail_stations` / `rail_platforms`) ---------------
# Trains are schedule-driven and live outside the road graph: the engine needs
# continuous track polylines (its chainage source), station anchors to snap
# GTFS stops onto, and platform outlines to draw. Ways are stitched through
# plain track joints and split at switches (>= 3 incident tracks), so one
# physical track becomes one polyline; speed/layer changes along it are carried
# as segment breakpoints rather than splitting the line, keeping chainage
# continuous across bridges. Sidings/yards/spurs are excluded — timetable
# service never runs on them.

RAIL_KINDS = {"rail", "light_rail", "tram", "subway", "narrow_gauge"}
RAIL_SERVICE_EXCLUDE = {"siding", "yard", "spur", "crossover"}
RAIL_DEFAULT_MPH = {"rail": 60, "light_rail": 35, "tram": 25, "subway": 50, "narrow_gauge": 25}
# One physical system retags across tunnel portals (Muni: surface light_rail →
# subway underground; street-running tram sections) — stitch across the family
# so a through service keeps one continuous chainage. Heavy rail stays its own.
URBAN_RAIL_FAMILY = {"light_rail", "tram", "subway"}


def rail_kind_joins(a, b):
    return a == b or (a in URBAN_RAIL_FAMILY and b in URBAN_RAIL_FAMILY)


def rail_query(bbox):
    s, w, n, e = bbox
    kinds = "|".join(sorted(RAIL_KINDS))
    return f"""
    [out:json][timeout:90];
    (
      way["railway"~"^({kinds})$"]({s},{w},{n},{e});
      node["railway"~"^(station|halt|tram_stop)$"]({s},{w},{n},{e});
      way["railway"~"^(station|halt|platform)$"]({s},{w},{n},{e});
      way["public_transport"="platform"]({s},{w},{n},{e});
    );
    (._;>;);
    out body;
    """


def parse_rail_speed_mps(tags, kind):
    raw = tags.get("maxspeed")
    mph = RAIL_DEFAULT_MPH.get(kind, 45)
    if raw:
        token = raw.split()[0]
        try:
            v = float(token)
            mph = v if "mph" in raw else v / 1.60934
        except ValueError:
            pass
    return round(mph * 0.44704, 2)


def scrape_rail(bbox, lat0, lon0):
    els = fetch_query(rail_query(bbox)).get("elements", [])
    node_ll = {e["id"]: (e["lat"], e["lon"]) for e in els if e["type"] == "node"}
    tracks = [
        e for e in els
        if e["type"] == "way"
        and e.get("tags", {}).get("railway") in RAIL_KINDS
        and e.get("tags", {}).get("service") not in RAIL_SERVICE_EXCLUDE
        and len(e.get("nodes", [])) >= 2
    ]

    by_id = {w["id"]: w for w in tracks}
    incident = defaultdict(list)
    for w in tracks:
        incident[w["nodes"][0]].append(w["id"])
        incident[w["nodes"][-1]].append(w["id"])

    def end_dir(way, nid):
        # Unit direction leaving `nid` along `way` (nid is one of its endpoints).
        seq = way["nodes"]
        a, b = (seq[0], seq[1]) if seq[0] == nid else (seq[-1], seq[-2])
        if a not in node_ll or b not in node_ll:
            return None
        pa = project(*node_ll[a], lat0, lon0)
        pb = project(*node_ll[b], lat0, lon0)
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        h = math.hypot(dx, dy)
        return (dx / h, dy / h) if h > 1e-9 else None

    def continuation(nid, d_away, kind, used):
        # The straightest unvisited same-kind track leaving `nid`. A plain joint
        # continues at dot ≈ 1; at a switch the main track (straight) beats the
        # diverging turnout leg (~10° off); below the collinearity floor the
        # line ends — a wye or sharp junction is a genuine break.
        best, best_dot = None, 0.86
        for wid in incident[nid]:
            if wid in used:
                continue
            cand = by_id[wid]
            if not rail_kind_joins(cand["tags"]["railway"], kind):
                continue
            d_out = end_dir(cand, nid)
            if d_out is None:
                continue
            dot = d_away[0] * d_out[0] + d_away[1] * d_out[1]
            if dot > best_dot:
                best, best_dot = wid, dot
        return best

    used, lines = set(), []
    for w0 in tracks:
        if w0["id"] in used:
            continue
        kind = w0["tags"]["railway"]
        used.add(w0["id"])
        chain = [(w0, False)]
        while True:
            way, rev = chain[-1]
            nid = way["nodes"][0 if rev else -1]
            d = end_dir(way, nid)
            nxt = continuation(nid, (-d[0], -d[1]), kind, used) if d else None
            if nxt is None:
                break
            used.add(nxt)
            chain.append((by_id[nxt], by_id[nxt]["nodes"][-1] == nid))
        while True:
            way, rev = chain[0]
            nid = way["nodes"][-1 if rev else 0]
            d = end_dir(way, nid)
            prv = continuation(nid, (-d[0], -d[1]), kind, used) if d else None
            if prv is None:
                break
            used.add(prv)
            chain.insert(0, (by_id[prv], by_id[prv]["nodes"][0] == nid))

        pts, speeds, layers, names = [], [], [], defaultdict(int)
        for way, rev in chain:
            seq = way["nodes"][::-1] if rev else way["nodes"]
            seg = [list(project(node_ll[n][0], node_ll[n][1], lat0, lon0)) for n in seq if n in node_ll]
            if len(seg) < 2:
                continue
            seg = simplify(seg)
            # Segment i spans pts[i]..pts[i+1]; a continuing way's first segment
            # starts at the join point already stored, hence the -1.
            start = len(pts) - 1 if pts else 0
            spd = parse_rail_speed_mps(way["tags"], kind)
            lay = parse_layer(way["tags"])
            if not speeds or speeds[-1][1] != spd:
                speeds.append([start, spd])
            if not layers or layers[-1][1] != lay:
                layers.append([start, lay])
            if way["tags"].get("name"):
                names[way["tags"]["name"]] += 1
            pts.extend(seg[1:] if pts else seg)
        if len(pts) < 2:
            continue
        line = {"kind": kind, "pts": pts, "speeds": speeds, "layers": layers}
        if names:
            line["name"] = max(names, key=names.get)
        lines.append(line)

    stations = []
    for e in els:
        tags = e.get("tags", {})
        r = tags.get("railway")
        if r not in ("station", "halt", "tram_stop"):
            continue
        if e["type"] == "node":
            x, y = project(e["lat"], e["lon"], lat0, lon0)
        elif e["type"] == "way":
            lls = [node_ll[n] for n in e["nodes"] if n in node_ll]
            if not lls:
                continue
            x, y = project(
                sum(p[0] for p in lls) / len(lls), sum(p[1] for p in lls) / len(lls), lat0, lon0
            )
        else:
            continue
        st = {"x": x, "y": y, "kind": r}
        if tags.get("name"):
            st["name"] = tags["name"]
        stations.append(st)

    platforms = []
    for e in els:
        if e["type"] != "way":
            continue
        tags = e.get("tags", {})
        is_rail_platform = tags.get("railway") == "platform" or (
            tags.get("public_transport") == "platform"
            and any(tags.get(m) == "yes" for m in ("train", "tram", "subway", "light_rail"))
        )
        if not is_rail_platform:
            continue
        seg = [list(project(node_ll[n][0], node_ll[n][1], lat0, lon0)) for n in e["nodes"] if n in node_ll]
        if len(seg) >= 2:
            platforms.append(simplify(seg))

    return lines, stations, platforms


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
        "--area",
        help="clip roads to this OSM admin boundary (e.g. 'San Mateo County') instead of a bbox; "
        "projection + rail/bus use the boundary's bounding box",
    )
    ap.add_argument("--admin-level", default="6", help="admin_level of --area (county = 6, city = 8)")
    ap.add_argument(
        "--highways-only", action="store_true",
        help="keep only freeways and their ramps/exits (motorway/trunk + _link)",
    )
    ap.add_argument(
        "--landuse", action="store_true",
        help="also scrape land-use polygons and POIs; weight each link's trip production/attraction",
    )
    args = ap.parse_args()

    area = (args.area, args.admin_level) if args.area else None
    bbox = resolve_area(args.area, args.admin_level) if area else resolve_bbox(args)
    classes = FREEWAY if args.highways_only else None
    landuse = None
    if args.landuse:
        lat0, lon0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        raw = fetch_query(landuse_query(bbox))
        landuse = LandUseGrid(raw, lat0, lon0)
        print(f"land use: {len(landuse.res)} residential cells, {len(landuse.attr)} attraction cells")
    s, w, n, e = bbox
    turn_rels = [
        el for el in fetch_query(
            f'[out:json][timeout:90]; rel["type"="restriction"]({s},{w},{n},{e}); out body;'
        ).get("elements", [])
        if el["type"] == "relation"
    ]
    graph = build(fetch(bbox, classes, area), bbox, args.place, classes or DRIVABLE, landuse, turn_rels)
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
    # Rail: continuous track polylines + station anchors + platform outlines.
    # Always scraped (even highways-only — the corridor renders and the GTFS
    # compiler snaps timetable trips onto these lines).
    lat0, lon0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    rail_lines, rail_stations, rail_platforms = scrape_rail(bbox, lat0, lon0)
    graph["rail_lines"] = rail_lines
    graph["rail_stations"] = rail_stations
    graph["rail_platforms"] = rail_platforms
    print(f"rail: {len(rail_lines)} lines, {len(rail_stations)} stations, {len(rail_platforms)} platforms")
    with open(args.out, "w") as f:
        json.dump(graph, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(graph['nodes'])} nodes, {len(graph['links'])} links")


if __name__ == "__main__":
    main()
