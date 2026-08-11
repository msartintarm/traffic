# Execution plan: Bay Area realism

Phased plan to close the gaps found in the realism audit (2026-08-11): parameter
deviations from empirical values, missing Bay-Area infrastructure, and absent
output validation. Ordering principle: yardsticks first, engine-artifact fixes
second, recalibration third, infrastructure fourth, system gate last.

## Status (2026-08-11): executed except the items under "Remaining"

Every phase below is implemented and green (`cargo test --features import`:
320 lib + 13 + 3 tests) except the explicitly deferred items. Highlights:

- P0: day-clock compression is a runtime parameter (1×–240×, UI slider);
  measurement APIs + GEH scorecard (`examples/scorecard.rs`); CI `validate`
  job (tests hard-gate the deploy; scorecard is a soft gate + artifact).
- P1: five junction artifact mechanisms found and fixed (physics-floor gap
  acceptance, coalesced-seam admission, same-tick chain races, internal-line
  overruns, lane-union stub gates) — the sober stress matrix runs 12/12
  junction-crash-free; β = 0.7 fitted to LODES (was engine-coupled 1.0).
- P2: IDM recalibrated (capacity band 1,900–2,300 asserted; crash rate fell
  from ~59k/100M VMT to real-world order); HCM gaps; speeding distribution;
  Caltrans truck shares; 8 s min green.
- P3: ALINEA ramp metering (D4 windows), OSM HOV lanes (scraped + enforced,
  Millbrae map regenerated with 11 express-lane links + land-use), rail
  crossings (scraper + timetable closures), pedestrian green floors, corridor
  cycle harmonization. Millbrae map now ships with embedded Caltrans AADT.
- P4: gateway inflow calibrates to embedded AADT (I-280 median GEH 23 → 12);
  README provenance section.
- Bonus (surfaced by the fresh scrape): two more right-of-way defects fixed —
  road-class-labelled "interchange" movements that genuinely cross another
  movement no longer ride the free-flow exemptions, and a turn now yields to a
  conflicting through outright (the parallel-approach angle heuristic had
  waved a ramp-side right across a frontage through). Plus brake-covering on
  priority approaches (ease below the dilemma threshold while a yield may
  still bind) and the same at boxes with live conflicting crossers.

## Remaining

1. **P3.5 named transit lines** (dwell half is DONE): real `highway=bus_stop`
   locations are scraped (`bus_stops` in the map JSON, 33 in Millbrae),
   resolved onto links, and buses dwell ~25 s curbside at them — arterial
   speeds now dip around real stops. Remaining: `route=bus` relations →
   link-chain routes with day-clock headway schedules, replacing the random
   bus draw in `class_of`.
2. **US-101 mainline GEH** (P4.1 loop): gateways now demand AADT-calibrated
   inflow but mid-segment mainline links carry a fraction of target — trace
   where the volume exits (OD through-share vs admission throttling) with the
   scorecard's per-link rows.
3. **Graded box occupancy** (P2.1 residual): binary `box_conflict` double-counts
   crossing majors, starving minor streets at 600 veh/h/dir (fixture expects
   ~200/h per HCM); per-conflict-point timing would lift it and enable the
   TWSC ±15% assertion.
4. **AM/PM directional offset plans** (P3.6 residual) and **rail preemption of
   adjacent signals** (P3.3 residual).
5. **Scorecard hard gate** (P4.2): flip `--assert` on once GEH share stabilizes.

## Phase 0 — Clocks and yardsticks

### P0.1 Parameterize time compression
`RUSH_DAY_COMPRESSION` (demand.rs) becomes a runtime parameter: field +
bridge setter + UI control (1×–120×, default 60×). Clock-domain audit: every
time constant is either *traffic-dynamics time* (headways, modulation epochs,
churn, wreck clearance — real sim seconds) or *day-clock time* (diurnal rates,
metering hours, signal plans, train schedules — compressed clock).
`wreck_clear_secs` expressible in day-clock terms.
**Accept:** rush tests pass at 1× and 60×; diurnal curve identical in
day-clock coordinates at both.

### P0.2 Measurement APIs
VMT accumulator; per-link flow/speed samplers alignable to PeMS stations;
corridor travel-time probes (101-through-bbox, El Camino end-to-end); headless
JSON scorecard (flows, speeds, travel times, crashes/100M-VMT).

### P0.3 Validation harness
Unit tier (synthetic fixtures, `cargo test --features import`): fundamental
diagram capacity band, HCM TWSC capacity fixture, corridor progression fixture.
CI tier (post-scrape, real data; bbox stays secret): GEH < 5 on ≥ 85% of
counted links, travel-time ratios vs PeMS, crash rate in 50–1000 per 100M-VMT
band. Soft gate → hard deploy blocker once stable.

## Phase 1 — Engine fixes that unblock calibration

### P1.1 Burst-demand junction/merge collisions
Stress test at `GRAVITY_BETA = 2.0` + dense platoons (0.3 prob, 4 followers)
until junction-kind crashes appear; fix permissive-box admission and
`merge_yield` projection until junction-artifact crashes ~0 under stress while
realistic rear-ends persist (`crash_counts` splits the kinds).

### P1.2 Re-derive demand parameters from data
Fit `GRAVITY_BETA` to the LODES trip-length distribution inside the box.
Platoon concentration from literature, not the engine-safety cap. Provenance
notes in `tools/`.

## Phase 2 — Behavioral recalibration (scored against P0.3)

### P2.1 HCM-consistent gap acceptance
Per-movement critical gaps at `effective_critical_gap`: major-left 4.1 s,
minor-right 6.2 s, minor-through 6.5 s, minor-left 7.1 s, +1 s trucks;
impatience floor 2.5–3.5 s; follow-up time 2.2–3.5 s at stop signs.
**Accept:** TWSC fixture within ~15% of HCM; unsignalized crash rate falls.

### P2.2 Freeway capacity + reaction time (joint tune)
Freeway headway T→~1.1–1.2 s with reaction →~0.7 s, targeting queue discharge
2,000–2,200 veh/h/ln and crash/VMT in band. Ring-road assertion: freeway
population peak flow in [1,900, 2,300] veh/h/ln. Context-dependent presets
(freeway vs surface) over one global T.

### P2.3 Speeding distribution
Replace hard `capped_to(limit)` with desired = limit × factor (freeway mean
~1.05, sd ~0.08, truncated limit+15 mph; arterial ~1.0; residential under).

### P2.4 Vehicle-class mix from data
Truck-AADT layer in `fetch_caltrans.py`; per-gateway mix where measured
(101 ≈ 5%, 280 ≈ 2%), class defaults otherwise. Random buses removed when
P3.5 lands.

### P2.5 Signal timing minimums
Actuated `MIN_GREEN` → 8 s (8–15 by class); `MAX_GREEN` reviewed by class.

## Phase 3 — Bay Area infrastructure

### P3.1 Ramp metering (ALINEA)
Meters at motorway_link→motorway joins: one-car-per-green, ALINEA rate
(`r += K_R(ô − o)`, setpoint ~0.17–0.22) off per-link occupancy; active in
day-clock peak windows. Node control reusing `stop_line` constraint.
**Accept:** metered merge beats unmetered mainline throughput at peak.

### P3.2 Express/HOV lanes
Scraper: per-lane `hov`/access tags. Engine: per-lane eligibility mask +
per-vehicle eligibility (HOV ~10–15% + time-varying toll-SOV share). MOBIL
hard mask. **Accept:** express lane near free flow at peak while GP degrades.

### P3.3 Caltrain crossings + signal preemption
Scraper: `railway=level_crossing` nodes. Engine: timetable-driven closures
(day-clock schedule, ~4–6 trains/hr/dir peak, 45–60 s each); adjacent signals
get track-clearing preemption. **Accept:** queue pulses match timetable; no
vehicle stopped on tracks.

### P3.4 Pedestrian green floors
At signals in commercial land-use areas: phase green ≥ crossing-distance/1.1
m/s + walk interval, from junction geometry.

### P3.5 Transit as routes
Scrape `route=bus` + `highway=bus_stop` (ECR ~10–15 min headways). Scheduled
spawns on fixed routes, 20–30 s curb dwell at stops. Remove random bus draw.

### P3.6 Corridor cycle harmonization + time-of-day plans
Stretch greens so a corridor shares its max cycle (scale greens only, never
masks); AM/PM/off-peak offset plans switching with the day clock.
**Accept:** platoon progression fixture; direction flips AM vs PM.

## Phase 4 — System calibration and guardrails

- **P4.1** Closed-loop OD calibration to GEH target; freeze params as data.
- **P4.2** CI scorecard flips to deploy blocker.
- **P4.3** README provenance per parameter + residual gaps.

## Sequencing

P0 → P1 → P2 → P4; P3 parallel after P0. Scraper halves of P3.2/P3.3/P3.5
can start any time. Riskiest: P1.1, P2.2. Highest realism-per-effort in P3:
ramp metering.
