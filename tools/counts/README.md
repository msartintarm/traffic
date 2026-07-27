# counts — calibration targets

Attaches real traffic counts to a scraped network so the simulation can be
calibrated: the engine measures its own per-link flow (`Simulation.link_flows()`,
vehicles/hour), and calibration tunes demand so those match observed counts.

## Real data (Millbrae, CA)

The **Caltrans Traffic Census** publishes AADT (Annual Average Daily Traffic) for
state routes. US‑101 runs through the Millbrae box (~219k AADT — its OSM `ref` is
`"US 101"`), and El Camino Real is **CA‑82** (`"CA 82"`, ~26.5k). The scraper
emits `ref` per link, so counts join automatically.

### Automatic fetch (recommended)

`fetch_caltrans.py` pulls AADT straight from the public Caltrans `Traffic_AADT`
ArcGIS service (no key) and writes the `ref,aadt` CSV — no manual download:

```
python3 fetch_caltrans.py --county SM --out caltrans.csv          # San Mateo county
python3 attach_counts.py --map ../../web/public/map.json --counts caltrans.csv --synthesize --out counts.json
```

`--synthesize` fills in local streets (no state-route `ref`) from road class, so
every link gets a target. Pass `--bbox S W N E` to `fetch_caltrans.py` to scope
AADT to just the map's box.

### Manual (fallback)

Download counts from <https://dot.ca.gov/programs/traffic-operations/census> and
save a CSV of `identifier,aadt` (`ref` like `CA 82`, or `name` like `El Camino
Real`), then run `attach_counts.py` as above.

## Without real data yet

`--synthesize` derives plausible AADT from road class (lanes × speed tier), so the
calibration loop can be exercised before real counts are in hand:

```
python3 attach_counts.py --map ../../web/public/map.json --synthesize --out counts.json
```

## Output (`counts.json`)

```jsonc
{
  "meta": { "k_factor": 0.09, "d_factor": 0.55, "links": 440, "matched": 440 },
  "targets": [
    { "link": 12, "road": "CA 82", "aadt": 32000, "peak_vph": 1584, "source": "observed" }
  ]
}
```

`peak_vph = aadt × K × D` (K = peak-hour fraction, D = directional split). Compare
`peak_vph` to the engine's `link_flows()[link]` and adjust OD demand rates until
they align. `counts.json` and downloaded CSVs are gitignored.
