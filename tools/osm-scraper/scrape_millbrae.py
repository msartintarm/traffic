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

DEFAULT_SPEED_MPH = {
    "motorway": 65, "trunk": 55, "primary": 35, "secondary": 35,
    "tertiary": 30, "residential": 25, "unclassified": 25, "living_street": 15,
}


def overpass_query(bbox):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:60];
    (
      way["highway"]({s},{w},{n},{e});
    );
    (._;>;);
    out body;
    """


def fetch(bbox, attempts=6):
    body = urllib.parse.urlencode({"data": overpass_query(bbox)}).encode()
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


def build(raw, bbox, place):
    nodes = {e["id"]: e for e in raw["elements"] if e["type"] == "node"}
    ways = [
        e for e in raw["elements"]
        if e["type"] == "way" and e.get("tags", {}).get("highway") in DRIVABLE
    ]

    usage = defaultdict(int)
    for way in ways:
        for nid in way["nodes"]:
            usage[nid] += 1
    for way in ways:
        for nid in (way["nodes"][0], way["nodes"][-1]):
            usage[nid] += 1

    def is_junction(nid):
        return usage[nid] >= 2 or nodes[nid].get("tags", {}).get("highway") == "traffic_signals"

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

    def geom_of(node_ids):
        return [list(project(nodes[nid]["lat"], nodes[nid]["lon"], lat0, lon0)) for nid in node_ids]

    def emit_link(a, b, lanes, speed, geometry, name, ref, layer, road_class, turn_lanes):
        if (a, b) in emitted:
            return
        emitted.add((a, b))
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
                emit_link(a, b, lanes, speed, mid, name, ref, layer, highway, tl_forward)
                if not oneway:
                    emit_link(b, a, lanes, speed, list(reversed(mid)), name, ref, layer, highway, tl_backward)
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
    args = ap.parse_args()

    bbox = resolve_bbox(args)
    graph = build(fetch(bbox), bbox, args.place)
    with open(args.out, "w") as f:
        json.dump(graph, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(graph['nodes'])} nodes, {len(graph['links'])} links")


if __name__ == "__main__":
    main()
