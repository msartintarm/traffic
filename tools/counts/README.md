# counts — calibration targets

Attaches real traffic counts to a scraped network so the simulation can be
calibrated: the engine measures its own per-link flow (`Simulation.link_flows()`,
vehicles/hour), and calibration tunes demand so those match observed counts.

## Real data (Millbrae, CA)

The **Caltrans Traffic Census** publishes AADT (Annual Average Daily Traffic) for
state routes. El Camino Real through Millbrae is **CA-82** — its OSM `ref` is
`"CA 82"`, which the scraper now emits per link.

1. Download San Mateo County counts: <https://dot.ca.gov/programs/traffic-operations/census>
2. Save a CSV of `identifier,aadt`, where `identifier` is an OSM road `ref`
   (e.g. `CA 82`) or `name` (e.g. `El Camino Real`):

   ```csv
   CA 82,32000
   Millbrae Avenue,18000
   ```

3. Attach:

   ```
   python3 attach_counts.py --map ../../web/public/map.json --counts caltrans.csv --out counts.json
   ```

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
