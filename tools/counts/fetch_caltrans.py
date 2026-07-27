#!/usr/bin/env python3
"""Fetch real Caltrans AADT (Annual Average Daily Traffic) for a county and emit
the `ref,aadt` CSV that `attach_counts.py` joins to the scraped map by OSM road
`ref` (e.g. "US 101", "CA 82"). Fully automates the previously-manual download.

Source: the public Caltrans Traffic Census ArcGIS service (no key needed):
  CHhighway/Traffic_AADT — point counts per postmile with AHEAD/BACK AADT.

Usage:
  python3 fetch_caltrans.py --county SM --out caltrans.csv        # San Mateo (Millbrae)
  python3 fetch_caltrans.py --county SM --bbox 37.594 -122.392 37.604 -122.38 --out caltrans.csv

With --bbox (S W N E, WGS84) only counts inside the map's box are used, so each
route's AADT reflects the segment actually in the model.
"""

import argparse
import csv
import json
import statistics
import urllib.parse
import urllib.request

SERVICE = "https://gisdata.dot.ca.gov/arcgis/rest/services/CHhighway/Traffic_AADT/MapServer/0/query"

# Route-number → OSM `ref` prefix. California's Interstates and US routes; every
# other state route becomes "CA n". Matches the scraper's OSM `ref` strings.
INTERSTATES = {5, 8, 10, 15, 40, 80, 105, 110, 205, 210, 215, 280, 380, 405, 505, 580, 605, 680, 710, 780, 805, 880, 980}
US_ROUTES = {6, 50, 95, 97, 101, 199, 395}


def osm_ref(rte):
    n = int(rte)
    if n in INTERSTATES:
        return f"I {n}"
    if n in US_ROUTES:
        return f"US {n}"
    return f"CA {n}"


def fetch(county):
    params = {
        "where": f"CNTY='{county}'",
        "outFields": "RTE,AHEAD_AADT,BACK_AADT",
        "outSR": "4326",
        "returnGeometry": "true",
        "f": "json",
    }
    url = SERVICE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "traffic-sim/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r).get("features", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--county", default="SM", help="Caltrans county code (SM = San Mateo)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"), help="only counts inside this WGS84 box")
    ap.add_argument("--out", default="caltrans.csv")
    args = ap.parse_args()

    inside = None
    if args.bbox:
        s, w, n, e = args.bbox
        inside = lambda x, y: w <= x <= e and s <= y <= n  # noqa: E731

    by_route = {}
    for f in fetch(args.county):
        at = f["attributes"]
        g = f.get("geometry") or {}
        if inside and not (g.get("x") is not None and inside(g["x"], g["y"])):
            continue
        rte = at.get("RTE")
        vals = []
        for v in (at.get("AHEAD_AADT"), at.get("BACK_AADT")):
            try:
                v = float(v)
                if v > 0:
                    vals.append(v)
            except (TypeError, ValueError):
                pass  # blank / non-numeric
        if rte and vals:
            by_route.setdefault(osm_ref(rte), []).extend(vals)

    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        for ref, vals in sorted(by_route.items()):
            w.writerow([ref, round(statistics.median(vals))])
    print(f"wrote {args.out}: {len(by_route)} routes for county {args.county}"
          + (" within bbox" if inside else ""))
    for ref, vals in sorted(by_route.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"  {ref}: AADT {round(statistics.median(vals)):>7} ({len(vals)} points)")


if __name__ == "__main__":
    main()
