# GTFS → transit artifact compiler

Compiles real GTFS feeds into the `*.transit.json.gz` artifact the engine's
`sim::rail::transit_from_json` consumes: rail trips (positioned per timetable,
snapped by the engine onto the scraped `rail_lines`) and bus trips (real
departures + timepoint schedules for the named transit lines). See the module
docstring in `compile_gtfs.py` for the exact schema.

## Usage

```sh
# Discovery mode: every active GTFS feed in the Mobility Database whose
# bounding box intersects the map's (the map artifact carries its own
# bbox/origin in `meta` — nothing is hardcoded).
python3 compile_gtfs.py --map ../../web/public/map.json.gz --out ../../web/public/map.transit.json.gz

# Explicit feeds (URLs or local zips) skip discovery:
python3 compile_gtfs.py --map ../../web/public/map.json.gz --feed caltrain.zip --out map.transit.json.gz
```

With `TRANSIT_511_API_KEY` set, the 511.org Bay Area regional feed (30+
operators in one zip) is added; its data agreement requires visible
attribution, which lands in the artifact's `meta.attribution`.

## Notes

- Stdlib only, like the OSM scraper. Bbox + projection come from the map
  artifact's `meta`; the compiled artifact must pair with a map scraped from
  the same bbox.
- One representative Wednesday and Saturday inside each feed's validity window
  become the weekday/weekend service days (`weekend` flag per trip).
- Trips are clipped to the bbox with boundary pseudo-stops (arr == dep) time-
  interpolated along the shape, so pass-through trains still run across the
  whole box. Per-trip soft-fail: anything unresolvable is dropped and counted,
  never emitted broken.
- Feed licenses vary; `meta.license_urls` records what the catalog declares.
  Check before committing an artifact built from a feed with a restrictive
  license (the Bay Area operators used here permit redistribution with
  attribution).
