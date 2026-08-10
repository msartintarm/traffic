#!/usr/bin/env python3
"""Fetch real commute origin-destination flows for a bounding box from Census
LEHD LODES (LODES8) and aggregate them onto a metre grid — the ground-truth
"who commutes from where to where" that category-sampled demand approximates.

LODES publishes, per state, every home-block → work-block job count in the US
(free, no key). This tool downloads the state's OD file and its geography
crosswalk (which carries block centroid lat/lon), keeps the flows whose home
AND work blocks fall inside the map's bbox, and sums them into coarse grid
cells projected in the same local metre frame as the scraped map — ready to
seed demand with real commute structure (AM flows home→work, PM the reverse).

Usage:
  # Per-map (the deployed form): bbox/origin/state come from the map's own meta,
  # so the OD grid is in the same frame as the map by construction. Writes
  # `<map>.lodes.json` (+ a `.gz` sibling) next to the map.
  python3 fetch_lodes.py --map ../../web/public/map.json
  python3 fetch_lodes.py --map ../../web/public/peninsula.json --state ca  # place has no ", XX" suffix

  # Bare-bbox form (exploration / smoke tests):
  python3 fetch_lodes.py --state ri --bbox 41.80 -71.45 41.85 -71.35 --out test.json

Downloads are cached in ./cache/ (state OD files run tens of MB gzipped).

Output:
  {
    "meta":  {"state", "year", "bbox", "origin", "grid_m", "jobs"},
    "cells": [[x, y], ...],          // grid-cell centres, metres in the map frame
    "flows": [[home, work, jobs], ...]  // indices into cells
  }
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import urllib.request

BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"


def cached_download(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "traffic-sim/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(path + ".part", path)
    return path


def project(lat, lon, lat0, lon0):
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * 6371000.0
    y = math.radians(lat - lat0) * 6371000.0
    return x, y


def blocks_in_bbox(xwalk_path, bbox):
    """Block GEOID → projected (x, y) for blocks inside the bbox, from the state
    crosswalk's block centroid lat/lon columns."""
    s, w, n, e = bbox
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    blocks = {}
    with gzip.open(xwalk_path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["blklatdd"]), float(row["blklondd"])
            except (KeyError, ValueError):
                continue
            if s <= lat <= n and w <= lon <= e:
                blocks[row["tabblk2020"]] = project(lat, lon, lat0, lon0)
    return blocks


def map_meta(path):
    """(bbox, state-or-None) from a scraped map's meta: `bbox` verbatim, state parsed
    from a `place` ending in ", XX" (the scraper's convention)."""
    meta = json.load(open(path))["meta"]
    place = meta.get("place", "")
    state = place.rsplit(",", 1)[-1].strip().lower() if "," in place else None
    return meta["bbox"], state if state and len(state) == 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", dest="map_path", help="scraped map JSON; bbox/state/out derive from it")
    ap.add_argument("--state", help="two-letter state code, e.g. oh (required without --map)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--grid", type=float, default=500.0, help="aggregation cell size, metres")
    ap.add_argument("--job-type", default="JT00", help="LODES job type (JT00 = all jobs)")
    ap.add_argument("--out", help="default: <map>.lodes.json next to --map, else lodes_od.json")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    if args.map_path:
        bbox, state = map_meta(args.map_path)
        args.bbox = args.bbox or bbox
        args.state = args.state or state
        if not args.out:
            base = args.map_path[:-5] if args.map_path.endswith(".json") else args.map_path
            args.out = f"{base}.lodes.json"
    if not args.state or not args.bbox:
        ap.error("--state and --bbox are required unless --map supplies them")
    args.out = args.out or "lodes_od.json"
    st = args.state.lower()

    xwalk = cached_download(f"{BASE}/{st}/{st}_xwalk.csv.gz", args.cache)
    od = cached_download(f"{BASE}/{st}/od/{st}_od_main_{args.job_type}_{args.year}.csv.gz", args.cache)

    blocks = blocks_in_bbox(xwalk, args.bbox)
    print(f"{len(blocks)} census blocks inside the bbox")

    # Sum job counts between grid cells for flows entirely inside the box. (Flows
    # with one end outside are the gateway streams the boundary categories already
    # model; LODES `aux` files would add out-of-state workers.)
    cs = args.grid
    cell_of = lambda p: (int(p[0] // cs), int(p[1] // cs))  # noqa: E731
    flows, jobs_total = {}, 0
    with gzip.open(od, "rt", newline="") as f:
        for row in csv.DictReader(f):
            h, w_ = blocks.get(row["h_geocode"]), blocks.get(row["w_geocode"])
            if h is None or w_ is None:
                continue
            jobs = int(row["S000"])
            jobs_total += jobs
            key = (cell_of(h), cell_of(w_))
            flows[key] = flows.get(key, 0) + jobs

    cells, index = [], {}
    for hc, wc in flows:
        for c in (hc, wc):
            if c not in index:
                index[c] = len(cells)
                cells.append([round((c[0] + 0.5) * cs, 1), round((c[1] + 0.5) * cs, 1)])
    out = {
        "meta": {
            "state": st, "year": args.year, "bbox": list(args.bbox),
            "origin": [(args.bbox[0] + args.bbox[2]) / 2, (args.bbox[1] + args.bbox[3]) / 2],
            "grid_m": cs, "jobs": jobs_total,
        },
        "cells": cells,
        "flows": sorted([index[h], index[w], j] for (h, w), j in flows.items()),
    }
    json.dump(out, open(args.out, "w"), separators=(",", ":"))
    # A pre-compressed sibling, matching the app's gz-first fetch (static hosts
    # don't compress on the wire; the browser inflates it). mtime=0 keeps the
    # bytes deterministic across regenerations.
    with open(args.out, "rb") as f, open(args.out + ".gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as g:
            g.write(f.read())
    print(f"wrote {args.out} (+.gz): {len(cells)} cells, {len(out['flows'])} cell flows, {jobs_total} commuters")


if __name__ == "__main__":
    main()
