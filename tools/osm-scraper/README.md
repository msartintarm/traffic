# osm-scraper

Scrapes a drivable road graph from OpenStreetMap (Overpass API) and emits the
JSON the Rust engine's `sim::map::OsmMap` consumes. It backs every real-map
scenario (Millbrae, San Carlos, San Francisco, the Bay Area peninsula).

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
    { "from_osm": 123, "to_osm": 456, "lanes": 2, "speed_limit": 15.6 }
  ]
}
```

- `x`/`y` are metres in a local equirectangular projection about the bbox centre
  (engine geometry is planar).
- `control` ∈ `uncontrolled | signal | stop | yield`; `signal` timing is a
  placeholder plan until real signal data / calibration is available.
- `links` are **directed** — a two-way street becomes two links — matching
  `LinkSpec`. Ways are split at every intersection node so one link is one block.

## Wiring it into the engine

`sim::map::millbrae_sample()` is a hand-built stand-in with this exact shape. The
remaining step is a small serde loader (behind a cargo feature) turning this JSON
into an `OsmMap`; the field names above line up 1:1 with `NodeSpec`/`LinkSpec`.

## Known limitations (tracked for later)

- Link length is straight-line between intersection endpoints; curved-geometry
  arc length is dropped. Add polyline geometry to preserve it.
- Turn restrictions and `turn:lanes` are not yet parsed; the builder currently
  permits all non-U-turn movements.
- Signal phasing is a default two/round-robin plan, not real controller timing.
