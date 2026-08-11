#!/usr/bin/env python3
"""Fit the gravity distance-decay exponent to LODES commute OD data.

The engine's destination choice is P(j|i) ∝ W_j / d_ij^β (demand.rs
`gravity_pick`).  This fits β by maximum likelihood over the *measured*
home→work flows: for each home cell the candidate set is every work cell,
weighted by its total inbound jobs (the opportunity term), so the fitted β
isolates pure distance decay from the geography of where jobs happen to be —
a naive fit of the trip-length histogram would conflate the two.

    python3 fit_gravity.py ../../web/public/*.lodes.json
"""

import gzip
import json
import math
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def fit(doc, floor_m=200.0, max_works=400, max_homes=400):
    cells = doc["cells"]
    flows = [(h, w, v) for h, w, v in doc["flows"] if v > 0]
    if len(flows) < 30:
        return None
    jobs = {}
    for _, w, v in flows:
        jobs[w] = jobs.get(w, 0.0) + v
    # Opportunity set: the biggest job cells carry nearly all the choice mass.
    works = sorted(jobs, key=lambda w: -jobs[w])[:max_works]
    wset = set(works)
    outflow = {}
    for h, w, v in flows:
        if w in wset:
            outflow[h] = outflow.get(h, 0.0) + v
    # Home sample: the highest-outflow homes, so the likelihood keeps most trips.
    homes = sorted(outflow, key=lambda h: -outflow[h])[:max_homes]
    hset = set(homes)
    kept = [(h, w, v) for h, w, v in flows if h in hset and w in wset]

    logw = {w: math.log(jobs[w]) for w in works}
    # Per home: parallel lists of (log jobs, log distance) over the choice set,
    # so each likelihood evaluation is one fused pass.
    cand = {}
    for h in homes:
        hx, hy = cells[h]
        lw, ld = [], []
        for w in works:
            wx, wy = cells[w]
            lw.append(logw[w])
            ld.append(math.log(max(math.hypot(hx - wx, hy - wy), floor_m)))
        cand[h] = (lw, ld)
    obs = {}
    for h, w, v in kept:
        obs.setdefault(h, []).append((logw[w], math.log(max(
            math.hypot(cells[h][0] - cells[w][0], cells[h][1] - cells[w][1]), floor_m)), v))

    def loglik(beta):
        ll = 0.0
        for h in homes:
            lw, ld = cand[h]
            denom = sum(math.exp(a - beta * b) for a, b in zip(lw, ld))
            if denom <= 0:
                continue
            log_denom = math.log(denom)
            for a, b, v in obs[h]:
                ll += v * (a - beta * b - log_denom)
        return ll

    lo, hi = 0.0, 3.0
    for _ in range(28):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if loglik(m1) < loglik(m2):
            lo = m1
        else:
            hi = m2
    beta = (lo + hi) / 2
    total = sum(v for _, _, v in kept)
    mean_km = sum(v * math.exp(b) for h in homes for _, b, v in obs[h]) / total / 1000.0
    return beta, len(kept), total, mean_km


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: fit_gravity.py <lodes.json[.gz]>...")
    weighted, weight = 0.0, 0.0
    for p in paths:
        try:
            doc = load(p)
        except Exception as e:
            print(f"{p}: unreadable ({e})")
            continue
        r = fit(doc)
        if r is None:
            print(f"{p}: too few in-box flows to fit")
            continue
        beta, n, total, mean_km = r
        print(f"{p}: beta={beta:.2f}  ({n} OD pairs, {total:.0f} commuters/day, mean trip {mean_km:.1f} km)")
        weighted += beta * total
        weight += total
    if weight > 0:
        print(f"\ncommuter-weighted beta = {weighted / weight:.2f}")


if __name__ == "__main__":
    main()
