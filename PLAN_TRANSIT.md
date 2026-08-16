# Transit plan: real trains + timetable buses

Research + phased design (2026-08-21) for adding trains — positioned and rendered
accurately per real timetables with realistic acceleration/braking — and giving
buses the same schedule fidelity where it makes sense. References are by
**file + function/symbol**, per TODO.md convention.

## Status (2026-08-21): executed, T0–T5 landed

All phases below are implemented and green (`cargo test --features import`:
386 tests; web: 43 + tsc; both wasm packs + `next build`). What shipped:

- **Engine**: `sim/rail.rs` — `RailNetwork` (chainage polylines with
  speed/layer breakpoints, stations, platforms, riding on `Network` so all
  three render backends see it), `TrainClass` power-limited kinematics
  (closed-form accel curves validated against numeric integration),
  `TrainTrip::build` run-profile fitting (slack → lower cruise, never departs
  early, lateness carry + recovery), `Timetable::active_trains` (pure function
  of day-seconds, past-midnight aware), `crossing_closures` (lead/clear per 49
  CFR 234) with `ClosureSet::union` for parallel tracks. `NetWorld` gained
  `set_timetable`/`trains`/`set_day_clock`/`set_bus_schedule`; `rail_closed`
  prefers real closures, keeping the synthetic cadence as the no-artifact
  fallback. Buses: `TransitLine::set_trips` + real-departure firing in
  `step_transit` (service-day aware, rebuild-safe fast-forward) and timepoint
  holding in `advance_bus_stops` (day-clock hold ÷ day rate, capped).
- **Render**: rail bands above the junction band per grade layer (ballast +
  gauge-true steel rails + platforms; `rail_band_geometry` feeds GPU, raster,
  ASCII, and the colored terminal view); trains as carriage-chain `Instance`s
  appended to the existing instanced draw — poses evaluated directly at the
  prev/current day clock, so trains never touch the per-id pose map and need
  no ids. 2D fallback via `train_poses`; ASCII plots `T` per carriage.
- **Data**: scraper rail pass (`scrape_rail` — through-switch stitching picks
  the straightest continuation, so a corridor with crossovers stays one
  46 km chainage) + `tools/gtfs/compile_gtfs.py` (Mobility-Database discovery
  ranked by map-bbox *coverage*, service-day resolution, stop-time
  interpolation, bbox clipping with boundary pseudo-stops, per-trip
  soft-fail). Millbrae / San Carlos / peninsula extracts re-shipped with rail
  + `*.transit.json.gz` siblings (Caltrain, BART, SamTrans, shuttles;
  peninsula: 1169 snapped trips, 15 concurrent trains at peak, ~35 min
  gate-down/day per corridor track — real numbers).
- **Web**: artifact fetched as a map sibling like LODES; `transit` Control +
  UI toggle (off = synthetic cadence/headways); trains ride the existing GPU
  instance stream untouched.

Follow-up (2026-08-22): **SF shipped** (61 lines / 207 km scraped incl. Muni +
Market St subway; `sf.transit.json.gz` from 10 feeds — Caltrain, BART, 3598
Muni rail trips) and **multi-fragment trip snapping** landed: when no single
scraped line carries a route (Muni branch → portal → subway), the trip
stitches its own path hop-by-hop (`snap_trip_multiline`, `TrainTrip::path`)
and runs in route-distance space; crossing closures now project the crossing
position onto each trip's geometry (`crossing_closures(tt, rail, pos)`), which
also unified parallel-track handling. Snap rates: SF 51% → **100%** (4660
trips, ~150 concurrent trains midday), peninsula 83% → 97%. Buses are now
visually distinct in every backend: the 2D fallback draws per-class
sizes/colours (`vehicle_instances` carries `[len, wid, class]`), the ASCII
view plots buses as `B`, and the GPU path already had the 12 m blue body.

Deferred: bus-only lanes, re-entry gap behavior, articulated buses, gate
visuals, single-track meets, GTFS-RT, and the Columbus transit artifact
(COTA is bus-only; one `compile_gtfs.py` run when wanted).

## Where the codebase already is

Two of the three pillars are partially built; the framing is *upgrade*, not
*greenfield*:

- **Buses are implemented end-to-end** as schedule-driven `NetVehicle`s:
  scraper `route=bus`/`highway=bus_stop` → `Network::{resolve_route_chain,
  attach_bus_stops}` → `demand::TransitLine` + `step_transit` (synthetic 15/30
  min headways) → `VehicleClass::Bus` dwelling at curb stops
  (`bus_stop_line`, flat `BUS_DWELL_SECS` = 25 s). Missing: real GTFS
  departures, per-stop dwell, timepoint holding.
- **Rail level crossings exist without trains**: `railway=level_crossing`
  nodes survive the scraper; `NetWorld::rail_closed` closes them on a
  *synthetic* Caltrain-cadence day-clock timetable, gating traffic three ways
  (virtual stop-line in `gather_context`, `box_entry_blocked` veto, and signal
  preemption via `build_rail_preempts` → forced phases). This plumbing is
  exactly what a real train should drive.
- **Genuinely missing**: rail track geometry (scraper filters `railway` ways
  out — only crossing *nodes* survive), any train entity (every mover is a
  car-on-a-road-lane `NetVehicle { lane: LaneId, .. }`), and any real
  timetable data (no GTFS anywhere).

## Architecture decision: the rail/bus split

Copy the MATSim-SBB / SUMO consensus, which happens to fit our two clock
domains (PLAN.md P0.1) exactly:

- **Trains are schedule-driven, not traffic-driven.** A train's position is a
  pure function of day-clock seconds: a kinematic accelerate–cruise–brake
  profile fitted to the timetable Δt between stations, holding at stations
  until scheduled departure (trains never leave early; timetables carry 3–8%
  slack, so full-performance runs arrive early and hold). Trains live outside
  `Network`/`Fleet` — no IDM, no MOBIL, no conflict points — and touch road
  traffic *only* through the existing crossing gate. MATSim-SBB runs rail
  deterministically for this reason: rail simulated as free-flow link traffic
  arrives unrealistically early; the schedule is ground truth.
- **Buses stay in traffic.** They are already `NetVehicle`s subject to
  congestion; the schedule constrains *departure times only* (spawn time, and
  holding at timepoint stops à la SUMO's `<stop until=.../>`). Arrival
  emerges from traffic — which naturally produces bus bunching under
  congestion, a realism feature. Do **not** make buses schedule-positioned.
- **Clock domain**: timetables and train kinematics are day-clock quantities
  (like `rail_closed` and `step_transit` today). At 60× compression trains
  time-lapse exactly as signal plans and diurnals do; at 1× they are fully
  physical. No new clock machinery needed.

## Data pipeline

### Rail geometry (scraper pass)
Extend `tools/osm-scraper/scrape_millbrae.py` with a rail query:
`way["railway"~"rail|light_rail|tram|subway"]` **excluding**
`service=siding|yard|spur|crossover` (pfaedle and SUMO both drop service
tracks), plus `railway=station|halt`, `public_transport=platform|
stop_position`, `stop_area` relations. Keep `maxspeed` (often "79 mph"
strings), `electrified`, `usage`, and `bridge/tunnel/layer` (rides the
existing grade-band renderer). Emit new top-level arrays `rail_lines`
(stitched polylines + per-segment maxspeed + layer) and `rail_stations`,
mirroring the `bus_routes`/`bus_stops` shape. Associate each existing
`level_crossing` node with (rail line, chainage) so a crossing knows which
track and where along it.

### Timetables (new tool, `tools/gtfs/`)
Fetch in CI, per the existing map-artifact policy (committed `web/public/*.gz`
extracts, bbox never hardcoded):

- **Feed discovery by bbox** (region-agnostic): the Mobility Database catalog
  CSV `https://files.mobilitydatabase.org/feeds_v2.csv` — no auth, one row
  per feed with bounding box + source URL + `license_url`. Intersect with
  `TRAFFIC_BBOX`, download matching feeds. Transitland v2 REST
  (`bbox=` params, free key, 10k req/mo) as a cross-check. For the Bay Area
  the **511.org regional feed** (`api.511.org/transit/datafeeds?operator_id=RG`,
  free key via secret, 60 req/hr) aggregates Caltrain/BART/SamTrans/VTA/etc.
  in one zip; its data agreement allows redistribution with visible
  attribution. Record each feed's `license_url` + attribution string in the
  emitted artifact.
- **Compile GTFS → compact timetable artifact** (`*.transit.json.gz` sibling
  to the map, loaded like `bus_routes_from_json`):
  - Resolve service for one representative weekday + weekend day
    (calendar + calendar_dates; keep the `day % 7` weekday/weekend split the
    demand layer already uses). Handle times past 24:00 (a 25:35 trip belongs
    to the *previous* service day — position lookups check both).
  - Interpolate blank non-timepoint stop_times by `shape_dist_traveled`
    (fallback: cumulative geodesic distance, monotone-in-`stop_sequence`
    stop-onto-shape projection — GTFS shapes are coarse and can self-cross).
  - `route_type` splits the treatment: 0/1/2/12 → train pipeline (map-match
    to `rail_lines`), 3/11 → bus pipeline (match to road links), others
    dropped. Expand `frequencies.txt` trips to explicit departures.
  - **Map-matching, soft-fail per trip** (the A/B Street lesson — their GTFS
    integration wedged intersections and was disabled with "Disable GTFS in
    SF, to unbreak the traffic sim"): shape-penalized shortest path over the
    mode's graph; a trip that won't match is dropped and counted in a
    `transit_missing` report, never emitted broken. Trips crossing the bbox
    edge are clipped with entry/exit times interpolated from the schedule —
    analogous to boundary-aware demand.
  - Per-trip output: stop chainages on a rail line (or `LinkId` chain for
    buses after `resolve_route_chain`), arrival/departure day-seconds,
    `timepoint` flags, vehicle class (from `route_type` + agency: EMU /
    diesel commuter / LRT / metro / bus).

## Train motion model

Per trip, precompute one run profile per inter-station segment, then evaluate
position(day_secs) directly (pure function: cheap, order-independent,
compression-proof, no per-tick integration):

- Kinematics: accel `a(v) = min(a₀, P/(m·v))` (power-limited above ~40–50
  km/h), cruise at `min(track maxspeed, v_needed)`, brake at `b` to the stop.
  If scheduled Δt exceeds the minimum feasible run time, lower cruise speed
  to fit (timetable slack absorbed as slower cruise); if Δt is infeasible
  (dirty data / aggressive schedule), run at full performance and carry
  lateness into the next stop. Hold at stations until scheduled departure.
- Dwell: GTFS `departure − arrival` when they differ; else mode defaults
  (commuter 45 s — Caltrain's typical scheduled dwell; metro 30 s; LRT 20 s).
- Class dynamics (m/s², from TCRP 13 / operator specs):

  | class            | accel a₀ | service brake | notes                          |
  |------------------|----------|---------------|--------------------------------|
  | EMU (Caltrain KISS) | 1.0   | 0.8           | 79 mph corridor cap            |
  | Metro (BART-like)   | 1.34  | 1.0           | tapers above ~50 km/h          |
  | Light rail / tram   | 1.3   | 1.3           |                                |
  | Diesel commuter     | 0.4   | 0.8           | slow power ramp                |
  | Freight             | 0.05  | 0.3           | matters only for crossing time |

- No train–train interaction in v1: timetables presume separation, and the
  deterministic runner can't conflict. (Single-track meets / block signals
  are a later refinement if a mapped bbox needs them.)

## Level crossings from real trains

Replace `rail_closed`'s synthetic cadence with schedule lookahead — since
position is a pure function of day-time, each crossing computes its closure
intervals exactly, no detection model needed. Per 49 CFR 234.225/.223 +
MUTCD practice: lights + bells at ETA − 25 s (regulatory floor 20 s), gates
descending by ETA − 22 s, horizontal ≥ 5 s before arrival, closed while the
train occupies (length/speed — a 160 m EMU at 79 mph ≈ 5 s; freight is
minutes), reopen 3–8 s after the tail clears. Total ≈ 40–60 s per passenger
train — close to today's `RAIL_CLOSURE_SECS` = 45, but now tied to actual
trains, per-crossing, with overlapping trains unioned. The consumer side
(`gather_context` stop-line, `box_entry_blocked`, `build_rail_preempts`
signal preemption) is untouched — only the closure predicate changes.
Promote the timing constants to config. The Peninsula corridor has ~40
at-grade crossings; this is the biggest road-side realism payoff of the
whole feature.

## Rendering

- **Static rail layer**: new entries in `render::geometry::world_bands`
  keyed by (grade `layer`, new rail rank) — ballast/track-bed as fill, rails
  + ties as markings, platforms as a band. Flows automatically to GPU
  (`render_band_ranges`), raster, and ASCII via the shared `draw_world` path.
- **Trains as carriage chains**: a new instance stream (`train_instances` in
  `bridge.rs`, parallel to `render_instances`) rendering each train as a
  chain of ~25 m carriage quads placed along track chainage. This sidesteps
  both baked-in assumptions: `VehicleClass::from_length` caps at Bus, and a
  200 m body has no single clean prev→current Bézier — per-carriage poses
  are locally straight. **Carriage ids must come from the shared
  `DemandGenerator::next_id` space** (the renderer's prev-pose map keys by
  globally-unique id; see the render-id invariant) and must survive
  `rebuild_demand`'s id carry-over. ASCII plots one glyph per carriage.
- Crossing gate state is renderable from the closure predicate (flashing
  marker reusing the signal-head pattern) — optional polish.
- `render2d.ts` draws fixed-size quads regardless of class; extend or accept
  degraded trains on the 2D fallback.

## Bus upgrades (the "same treatment where it makes sense")

1. **Real departures**: `TransitLine` gains per-trip stop_times from the
   GTFS artifact; `step_transit` fires at real day-second departures instead
   of `transit_headway`. Blocked-entrance slip behavior stays.
2. **Timepoint holding**: at `timepoint=1` stops, hold until scheduled
   departure (extend the `bus_dwell`/`bus_stop_line` machinery); when late,
   serve passengers and go. Between timepoints buses just drive.
3. **Per-stop dwell**: replace flat `BUS_DWELL_SECS` with GTFS times where
   present, else TCQSM-style sampled 10–30 s (base ~5 s + per-passenger
   terms), with skip-stop probability at low-demand stops.
4. **Dynamics check**: `VehicleClass::Bus` preset (a = 0.9, 12 m) is already
   inside the observed 1.0–1.7 accel / 1.25–2.1 decel envelope; agencies cap
   near 1.0 for standees. At most nudge accel to 1.0–1.2.
5. Optional, later: bus-only lanes by cloning the HOV mask
   (`lane_is_hov` → `lane_bus_only`, enforced at the same MOBIL check);
   re-entry gap-acceptance with a patience timer (California has no
   yield-to-bus law, so forcing the merge after decay matches reality);
   18 m articulated class.

## Config / knobs

Transit master toggle + train layer visibility + crossing timing + dwell
parameters → `SimConfig` (or a `TransitConfig`), `bridge.rs` setters, and the
three-place protocol ritual (`protocol.ts` `Control` union + `CONTROL_TYPES` +
`engineSession.ts` switch; `protocol.test.ts` guards drift), UI in
`EngineCanvas.tsx`. First constants to promote: the `rail_closed` cadence
(deleted), `RAIL_CLOSURE_SECS` (split into lead/clear times), `BUS_DWELL_SECS`,
`transit_headway` (becomes fallback when no GTFS artifact is present).

## Phases

- **T0 — Rail geometry + static render.** Scraper rail pass; engine-side
  `RailNetwork` (polylines + chainage + maxspeed + layer + station/crossing
  anchors) parsed like `bus_routes_from_json`; rail/platform bands.
  **Accept:** Peninsula map shows the Caltrain corridor correctly layered
  under/over roads in GPU + ASCII; no change to any traffic test.
- **T1 — GTFS tool + timetable artifact.** Discovery, compile, map-match,
  soft-fail report; `*.transit.json.gz` committed like map extracts with
  license/attribution recorded. **Accept:** artifact round-trips through a
  loader test; matched-trip share reported; unmatched trips listed, none
  emitted.
- **T2 — Train entity + runner + render.** Deterministic position(day_secs),
  run profiles, holding, carriage-chain instances on the shared id space.
  **Accept:** at 1× compression a train's simulated station-to-station times
  match the timetable within dwell tolerance; never departs early; renders
  at correct chainage at spot-checked timetable instants.
- **T3 — Crossings driven by real trains.** Delete the synthetic cadence;
  closure intervals from schedule lookahead; constants → config.
  **Accept:** existing rail-preemption tests pass against real-timetable
  closures; queue pulses at a Peninsula crossing line up with the Caltrain
  timetable; no vehicle stopped on tracks (PLAN.md P3.3 criterion, now
  against real data).
- **T4 — Bus schedule fidelity.** Real departures, per-stop dwell, timepoint
  holding. **Accept:** bus double-stop trap test still green; a bus arriving
  early at a timepoint holds until its scheduled second; headway-fallback
  path still works on maps without a transit artifact.
- **T5 — Optional polish.** Bus lanes, re-entry behavior, articulated buses,
  crossing gate visuals, freight paths, single-track meets, GTFS-Realtime.

Sequencing: T0 → T1 → T2 → T3, with T4 parallel after T1. Riskiest: T1's
map-matching against bbox-clipped OSM rail (disconnected fragments —
fall back to raw GTFS shape geometry for rendering-only trips). Highest
realism-per-effort: T3.

## Hard-won constraints (do not violate)

- Trains never join `Fleet`/`world.vehicles()` — crash model, MOBIL, demand
  rebuild, and the GPU context gather all assume road cars (the A/B Street
  failure mode was transit state wedging core sim invariants).
- Every rendered entity id from the one `next_id` space, carried across
  demand rebuilds — else the renderer's prev-map flashes ghosts.
- Timetable quantities in day-clock seconds only; nothing timetable-shaped
  in traffic-dynamics time (P0.1 clock-domain audit extends to trains).
- Feed extracts follow the map-artifact policy: fetched by secret-configured
  CI, license + attribution recorded, bbox never hardcoded.

## Key sources

GTFS reference: gtfs.org/documentation/schedule/reference. Mobility Database
catalog CSV: files.mobilitydatabase.org/feeds_v2.csv. Transitland v2 REST:
transit.land/documentation/rest-api. 511 open data + data agreement:
511.org/open-data/transit. Train dynamics: TCRP Report 13; Stadler KISS
(Caltrain) specs; BART energy report. Crossing timing: 49 CFR 234.223/.225;
FHWA Highway-Rail Crossing Handbook 3rd ed. Dwell: TCQSM Part 4; Caltrain
dwell analysis (caltrain-hsr.blogspot.com, 2016-05). Prior art: SUMO GTFS
tutorial + Public Transport / Railways docs (gtfs2pt, `<stop until>`,
`rail_crossing` node type, 15 s time-gap default); MATSim-SBB extensions
(deterministic PT); pfaedle (ad-freiburg/pfaedle) for GTFS↔OSM map-matching;
A/B Street's disabled-GTFS cautionary tale.
