#!/usr/bin/env python3
"""Attach real (or synthesized) traffic counts to a scraped network for
calibration. Produces `counts.json` mapping each link to a target peak-hour
volume (vehicles/hour), which the simulation's measured `link_flows()` is
calibrated against.

Real data source for Millbrae, CA: the Caltrans Traffic Census program publishes
AADT (Annual Average Daily Traffic) for state routes — El Camino Real is CA-82.
Download the county's counts and save a CSV of `identifier,aadt` where identifier
matches an OSM road `ref` (e.g. "CA 82") or `name` (e.g. "El Camino Real"):

    https://dot.ca.gov/programs/traffic-operations/census

Usage:
    # attach observed counts (matched to links by ref, then name)
    python3 attach_counts.py --map ../../web/public/map.json --counts caltrans.csv --out counts.json

    # or synthesize plausible counts from road class, to exercise calibration now
    python3 attach_counts.py --map ../../web/public/map.json --synthesize --out counts.json

Peak-hour volume = AADT x K x D (K = fraction of daily traffic in the peak hour,
D = directional split). Defaults K=0.09, D=0.55 are typical urban-arterial values.
"""

import argparse
import csv
import json


def synthesize_aadt(link):
    """A rough AADT from road class (lanes x a speed-based base), for pipeline
    testing before real counts are available."""
    speed = link.get("speed_limit", 13.0)  # m/s
    base = 3000 if speed > 18 else 1500 if speed > 11 else 600  # arterial / collector / local
    return link.get("lanes", 1) * base


def load_observed(path):
    """CSV of `identifier,aadt` → {identifier_lower: aadt}."""
    table = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                table[row[0].strip().lower()] = float(row[1])
            except ValueError:
                continue  # header or blank
    return table


def aadt_for(link, observed):
    for key in (link.get("ref"), link.get("name")):
        if key and key.strip().lower() in observed:
            return observed[key.strip().lower()], "observed"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--counts", help="CSV of identifier,aadt (ref or road name)")
    ap.add_argument("--synthesize", action="store_true", help="derive AADT from road class")
    ap.add_argument("--out", default="counts.json")
    ap.add_argument("--k", type=float, default=0.09, help="peak-hour fraction of AADT")
    ap.add_argument("--d", type=float, default=0.55, help="peak-direction split")
    args = ap.parse_args()

    net = json.load(open(args.map))
    observed = load_observed(args.counts) if args.counts else {}

    targets, matched = [], 0
    for i, link in enumerate(net["links"]):
        aadt, source = aadt_for(link, observed)
        if aadt is None and args.synthesize:
            aadt, source = synthesize_aadt(link), "synthesized"
        if aadt is None:
            continue
        matched += 1
        targets.append({
            "link": i,
            "road": link.get("ref") or link.get("name") or "",
            "aadt": round(aadt),
            "peak_vph": round(aadt * args.k * args.d),
            "source": source,
        })

    out = {
        "meta": {"k_factor": args.k, "d_factor": args.d, "links": len(net["links"]), "matched": matched},
        "targets": targets,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}: {matched}/{len(net['links'])} links have a target volume")


if __name__ == "__main__":
    main()
