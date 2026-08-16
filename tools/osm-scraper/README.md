# osm-scraper

Scrapes a drivable road graph from OpenStreetMap (Overpass API) and emits the
JSON the Rust engine's `sim::map::OsmMap` consumes. It backs every real-map
scenario (Millbrae, San Carlos, San Francisco, the Bay Area peninsula, and the
Columbus, OH freeway network) — the location is set purely by the bounding box.

The bounding box is an **input, never checked in**. Supply it (highest
precedence first) via `--bbox S W N E`, a gitignored `--bbox-file PATH`
containing `S W N E`, or the `TRAFFIC_BBOX` environment variable.

```
python3 scrape_millbrae.py --bbox-file bbox.local --out millbrae.json
TRAFFIC_BBOX="S W N E" python3 scrape_millbrae.py --out millbrae.json
```

### Freeways-only extracts (`--highways-only`)

For a regional freeway scenario, keep only the grade-separated network and its
ramps/exits (`motorway`/`trunk` + their `_link`s). The class filter runs
server-side, so even a whole-region bbox stays a small download. This is how
`web/public/peninsula.json` (US-101, I-280, CA-92, CA-84, CA-380 and their
interchanges) is produced:

```
python3 scrape_millbrae.py --highways-only \
  --bbox 37.42 -122.51 37.71 -122.08 \
  --place "Bay Area Peninsula" --out ../../web/public/peninsula.json
```

The Columbus, OH freeway network (I-70/71/270 outerbelt + the US/SR radials) was
produced the same way — only the bounding box differs:

```
python3 scrape_millbrae.py --highways-only \
  --bbox 39.80 -83.25 40.18 -82.75 \
  --place "Columbus, OH" --out ../../web/public/columbus.json
```

`*.json`, `bbox*`, and `*.local` are gitignored so extracts and coordinates stay
local.

## Output schema (the engine contract)

```jsonc
{
  "meta":  { "place": "Millbrae, CA", "bbox": [s,w,n,e], "origin": [lat0, lon0] },
  "nodes": [
    { "osm_id": 123, "x": 12.3, "y": -45.6, "control": "signal",
      "signal": { "green_secs": 25.0, "yellow_secs": 4.0, "offset": 0.0 } }
  ],
  "links": [
    { "from_osm": 123, "to_osm": 456, "lanes": 2, "speed_limit": 15.6, "sign": "stop" }
  ],
  "restrictions": [
    { "from": [123, 456], "to": [456, 789], "kind": "no_left_turn" }
  ],
  "rail_lines": [
    { "kind": "rail", "name": "Caltrain Peninsula Subdivision",
      "pts": [[x, y]], "speeds": [[seg_idx, mps]], "layers": [[seg_idx, layer]] }
  ],
  "rail_stations": [ { "x": 1.0, "y": 2.0, "kind": "station", "name": "Millbrae" } ],
  "rail_platforms": [ [[x, y]] ]
}
```

- `x`/`y` are metres in a local equirectangular projection about the bbox centre
  (engine geometry is planar).
- `control` ∈ `uncontrolled | signal | stop | yield`; `signal` timing is a
  placeholder plan until real signal data / calibration is available.
  Pedestrian signals (`traffic_signals=pedestrian_crossing`) are not junction
  controllers and stay uncontrolled.
- `sign` (optional, on a link) ∈ `stop | yield`: a stop/give_way surveyed on
  the *way* controls that one approach — the engine models a two-way stop
  (minor street lines up, cross street rolls). The controlled direction comes
  from the node's `direction` tag, else the sign binds toward the nearer block
  end. Signs surveyed on the junction node itself stay node-level `control`
  (all-way semantics).
- `links` are **directed** — a two-way street becomes two links — matching
  `LinkSpec`. Ways are split at every intersection node so one link is one block.

## Land-use weights (`--landuse`)

A second, lightweight Overpass pass fetches land-use polygons
(residential/commercial/retail/industrial) and point-of-interest nodes
(shops, amenities, offices), rasterizes them onto a 150 m grid, and stamps
each link with `res_weight` (trip production — homes) and `attr_weight`
(trip attraction — activity centres), both ~0.3–4 with 1.0 neutral. The
engine's demand generator places trip origins by `res_weight` and multiplies
gravity destination choice by `attr_weight`, so traffic runs homes→shops/jobs
instead of a uniform scatter. Maps scraped without the flag behave as before
(every weight neutral).

## Wiring it into the engine

`sim::map::millbrae_sample()` is a hand-built stand-in with this exact shape. The
remaining step is a small serde loader (behind a cargo feature) turning this JSON
into an `OsmMap`; the field names above line up 1:1 with `NodeSpec`/`LinkSpec`.

## Known limitations (tracked for later)

- Signal phasing is a default plan, not real controller timing; the engine
  rebuilds phases from its conflict graph and coordinates corridors itself.
- Turn restrictions resolve best-effort: conditional (time-of-day) relations,
  multi-way vias, and vias interior to an unsplit two-way way are skipped
  (each scrape prints the counts).
