#!/usr/bin/env python3
"""Fetch real Ohio DOT AADT for a bounding box and emit the `ref,aadt` CSV that
`attach_counts.py` joins to the scraped map by OSM road `ref` (e.g. "I 70",
"US 33", "OH 315") — the Ohio counterpart of `fetch_caltrans.py`, for the
Columbus map.

Source: the public ODOT TIMS ArcGIS service (no key needed):
  StandardMaps/AADT_StandardMaps — road-inventory segments with AADT_TOTAL.

Usage:
  python3 fetch_odot.py --bbox 39.90 -83.10 40.05 -82.90 --out odot.csv   # S W N E
  python3 attach_counts.py --map ../../web/public/map.json --counts odot.csv --write-map

Only Interstate (IR), US (US) and State (SR) routes are joined: municipal,
county and township roads carry no OSM `ref` to match on. That still covers the
freeways and the numbered arterials, which dominate real volumes.
"""

import argparse
import csv
import json
import statistics
import urllib.parse
import urllib.request

SERVICE = "https://tims.dot.state.oh.us/ags/rest/services/StandardMaps/AADT_StandardMaps/MapServer/2/query"

# ODOT ROUTE_TYPE → OSM `ref` prefix. Ohio state routes are signed "OH n" in OSM.
REF_PREFIX = {"IR": "I", "US": "US", "SR": "OH"}


def osm_ref(route_type, route_nbr):
    prefix = REF_PREFIX.get(route_type)
    if not prefix:
        return None
    try:
        return f"{prefix} {int(route_nbr)}"
    except (TypeError, ValueError):
        return None


def fetch(bbox):
    s, w, n, e = bbox
    envelope = json.dumps({"xmin": w, "ymin": s, "xmax": e, "ymax": n})
    features, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "geometry": envelope,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelEnvelopeIntersects",
            "outFields": "ROUTE_TYPE,ROUTE_NBR,AADT_TOTAL",
            "returnGeometry": "false",
            "resultOffset": str(offset),
            "f": "json",
        }
        url = SERVICE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "traffic-sim/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            page = json.load(r)
        if "error" in page:
            raise SystemExit(f"ODOT service error: {page['error']}")
        batch = page.get("features", [])
        features.extend(batch)
        if not page.get("exceededTransferLimit") or not batch:
            return features
        offset += len(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("S", "W", "N", "E"),
                    help="WGS84 box to pull counts for (the map's box)")
    ap.add_argument("--out", default="odot.csv")
    args = ap.parse_args()

    by_route = {}
    for f in fetch(args.bbox):
        at = f["attributes"]
        ref = osm_ref(at.get("ROUTE_TYPE"), at.get("ROUTE_NBR"))
        try:
            aadt = float(at.get("AADT_TOTAL"))
        except (TypeError, ValueError):
            continue
        if ref and aadt > 0:
            by_route.setdefault(ref, []).append(aadt)

    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        for ref, vals in sorted(by_route.items()):
            w.writerow([ref, round(statistics.median(vals))])
    print(f"wrote {args.out}: {len(by_route)} routes within bbox")
    for ref, vals in sorted(by_route.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"  {ref}: AADT {round(statistics.median(vals)):>7} ({len(vals)} segments)")


if __name__ == "__main__":
    main()
