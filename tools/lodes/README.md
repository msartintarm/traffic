# lodes — real commute OD flows

[LEHD LODES](https://lehd.ces.census.gov/data/) publishes every home-block →
work-block job count in the US (free, census-block resolution, updated
annually). `fetch_lodes.py` pulls a state's OD file plus its geography
crosswalk (block centroid lat/lon), keeps flows whose home *and* work blocks
fall inside the map's bbox, and aggregates them onto a coarse metre grid in
the same local projection as the scraped map:

```
# per map — bbox/state/output derive from the map's own meta, so the OD grid is
# in the map's frame by construction; writes web/public/<map>.lodes.json (+ .gz)
python3 fetch_lodes.py --map ../../web/public/map.json
python3 fetch_lodes.py --map ../../web/public/peninsula.json --state ca  # no ", XX" in its place name

# bare bbox (exploration / smoke tests)
python3 fetch_lodes.py --state oh --bbox 39.90 -83.10 40.05 -82.90 --out test.json
```

Output: `cells` (grid centres, metres in the map frame) and `flows`
(`[home_cell, work_cell, jobs]`). Downloads cache in `./cache/`; both are
gitignored (state OD files run tens of MB).

## Engine integration

Every real map ships a committed `<map>.lodes.json` sibling in `web/public/`
(the deploy workflow fails if one is missing), and the app loads it
automatically — or call the bridge's `set_commute_od(json)` directly. When
loaded, the measured flows join the sampled surface streams and **displace
their volume in proportion to the measured share** (floored, and never their
spatial coverage — a small bbox whose commuters mostly work outside it barely
dents the sampled fabric, while a metro box dominates it):

- Each cell anchors to its nearest surface interior links (a few candidates,
  since a trip and its reverse often need opposite carriageways).
- A jobs-weighted sample of flows becomes paired streams: home→work on the
  AM-dominant `Inbound` diurnal shape, work→home on the PM-dominant
  `Outbound` shape.
- Stream rates are sized so the sampled set carries the box's real total
  commuter volume (Σ jobs, once each way per day).
- Commute streams are `anchored`: origin churn never moves them, and their
  rates are absolute demand, not per-origin capacities.

Through-traffic and non-work trips stay on the sampled boundary categories
(LODES covers jobs only).
