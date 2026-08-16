# TODO — Traffic realism improvements

Prioritized backlog for making driver and traffic-light behavior more closely match
real life. References are by **file + function/symbol** (stable across the commented and
comment-free copies), not line numbers.

Priority legend: **[H]** high realism impact · **[M]** medium · **[L]** low / calibration.
Checked items are landed and name the symbol and guarding test — audit them against the
code, not this file, if in doubt.

---

## Landed (audited 2026-08-12)

- [x] **[H] Real transit: timetable trains + GTFS buses.** *(2026-08-21)* See
  `PLAN_TRANSIT.md` (status section) for the full map. Trains are schedule-driven
  outside the road graph (`sim/rail.rs`: run profiles fitted to the timetable,
  power-limited acceleration, hold-at-station, lateness carry; position a pure
  function of the day clock), rendered as carriage chains on the existing
  instanced draw; level crossings close on precomputed real-passage intervals
  (`NetWorld::set_timetable` → `rail_closed`, synthetic cadence as the
  no-artifact fallback, preemption unchanged). Buses fire real GTFS departures
  with per-stop timepoint holding (`TransitLine::set_trips`,
  `advance_bus_stops`). Data: scraper rail pass (through-switch stitching) +
  `tools/gtfs/compile_gtfs.py`; Millbrae/San Carlos/peninsula extracts carry
  rail + `*.transit.json.gz`. Tests:
  `a_slack_schedule_arrives_exactly_on_time_and_never_departs_early`,
  `an_infeasible_schedule_runs_flat_out_and_carries_lateness`,
  `crossing_closures_bracket_the_passage_with_lead_time`,
  `a_real_timetable_replaces_the_synthetic_crossing_cadence`,
  `an_early_bus_holds_at_a_timepoint_until_its_scheduled_departure`,
  `real_trips_fire_at_their_scheduled_departures_with_their_schedules`,
  `rail_draws_above_the_at_grade_roads_it_crosses`,
  `a_multi_fragment_route_stitches_its_own_path`. Diagnostics:
  `diag_transit_artifact`, `diag_rail_views` (both env-driven, ignored).
  *(2026-08-22)* SF shipped (Muni/BART/Caltrain, ~150 concurrent trains
  midday); multi-fragment trips stitch their own path (SF snap 51% → 100%);
  buses render distinctly in all backends (2D fallback per-class dims/colour,
  ASCII `B` glyph).

- [x] **[H] OSM-data accuracy: turn restrictions, two-way stops, pedestrian-signal
  filter.** *(2026-08-14)* The scraper resolves `type=restriction` relations to
  emitted-link node pairs (`resolve_restrictions`; conditional/exempt/complex-via
  skipped and counted) and the engine prunes movement wiring with them
  (`OsmMap::build_with_restrictions`; `no_*` bans an exit, `only_*` bans the rest,
  fail-open so no approach strands; pairs rewritten through both topology transforms).
  Way-mapped stop/give_way become per-approach `LinkSpec::sign` → `Network::
  {approach_stops, all_way_stop}`: the minor street lines up, the major street rolls
  (`two_way_stop_halts_the_minor_street_and_not_the_major`); stand-alone stop-line
  nodes relocate junction-ward (`relocate_sign_nodes` + `join_pass_through` sign
  merge). `traffic_signals=pedestrian_crossing` no longer becomes a junction signal.
  `junction_fan_ring` orients mouth corners by cross product (an arm due west of the
  centre used to reverse across atan2's ±π cut — bowtie ring, wrong curb gap). Tests:
  `turn_restrictions_prune_movements_across_the_collapse`,
  `via_way_restrictions_bind_once_the_junction_merges`, `stranding_restrictions_fail_open`,
  `way_mapped_stop_signs_make_a_two_way_stop`, `stop_line_nodes_relocate_to_their_junction`,
  `fan_ring_keeps_every_mouth_inside_even_across_the_angle_cut`. Maps need a re-scrape
  to carry `restrictions`/`sign`; old extracts run unchanged.

- [x] **[H] Green-wave coordination at runtime.**
  `map::coordinate_green_waves` + `harmonize_corridor_cycles` mark corridor programs
  `coordinated`; the controller (`junction::SignalController::group_state`) runs those
  offset-aware fixed-time (`SignalProgram::state_of`), with AM/PM plan swap at runtime
  (`NetWorld::pm_plan`, `Network::{am,pm}_offsets`). Actuation still owns uncoordinated
  signals. Tests: `signalized_corridor_is_coordinated_into_a_green_wave`,
  `real_map_signals_are_linked_and_not_all_green`, the coordinated-runtime-offset test.
  *Residual refinement:* coordinated-actuated hybrid (force-off/yield point, actuation
  floating only non-coordinated phases) — real corridors are semi-actuated, ours are
  fixed-time while coordinated.

- [x] **[H] Zero-mean, temporally correlated acceleration noise.**
  `constraint::accel_noise`: ±σ-bounded, zero-mean, piecewise-linear across ~2 s noise
  buckets (stateless hash, CPU/GPU identical). Tests:
  `accel_noise_is_zero_mean_two_sided_and_bounded`, `accel_noise_drifts_rather_than_flickers`,
  `acceleration_noise_fluctuates_around_desired_without_downward_bias`.

- [x] **[H] Kinematic (ITE) change and clearance intervals.**
  `map::change_and_clearance_intervals`: yellow = reaction + v/2a clamped [3, 6] s from
  the approach speed limit; all-red = (crossing + car length)/v plus compound-junction
  internal continuation, clamped — per phase, from the same geometry the box model uses.

- [x] **[H] Protected-permissive lefts + right-turn-on-red.**
  `NetWorld::{is_permissive, left_is_permissive, permissive_must_yield, is_rtor}` reuse
  the conflict-point and gap-acceptance machinery; RTOR requires a full stop first.
  Tests: `permissive_left_yields_to_oncoming_then_clears_without_colliding`,
  `right_turn_on_red_clears_before_a_through_that_waits_for_green`,
  `sober_platoon_bursts_through_permissive_lefts_stay_junction_crash_free`.

- [x] **[M] Start-up lost time / reaction lag at signals.**
  Tests: `a_stopped_car_takes_a_startup_reaction_to_launch_on_green`,
  `reaction_delay_causes_start_up_lag`.

- [x] **[M] Urgency-scaled mandatory lane changes.** *(2026-08-12)*
  `best_lane_change`: the positioning window scales with lanes still to cross (one full
  window apiece, capped by `MAX_POSITION_WINDOWS`), and a mandatory change under a
  closing window relaxes MOBIL `safe_braking` with urgency — mirroring
  `effective_critical_gap` impatience. Tests:
  `a_multi_lane_fix_starts_one_window_per_lane_early`,
  `a_surface_car_weaves_across_lanes_to_reach_its_turn_pocket`.

- [x] **[M] Right-turners yield on approach.**
  `should_yield_to`: crossing streams defer by approach priority key for every turn type;
  opposing streams put the left behind. Test: `minor_road_right_turn_yields_to_the_major_through`.

- [x] **[M] Keep-right / passing-lane asymmetry.**
  `KEEP_RIGHT_BIAS` through `mobil::should_change`'s bias term; suppressed near gores so
  it can't nudge a through car into an exit-only lane. Tests:
  `keep_right_drifts_an_unobstructed_car_to_the_curb_lane`, MOBIL bias unit test.

- [x] **[M] FIFO ordering at all-way stops.**
  Tests: `all_way_stop_serves_the_first_to_stop_first`,
  `all_way_stop_does_not_deadlock_when_the_first_car_is_blocked`.

- [x] **[M] Curvature-limited interior speeds.** *(2026-08-12)*
  `Network::interior_min_radius` (analytic Bézier curvature) → per-movement
  `NetWorld::turn_caps` = √(a_lat·r) clamped [2.5, 10] m/s; approach braking and every
  crossing-speed consumer read the same cap; interchange movements stay exempt. Replaces
  the flat Left=6 / Right=5. Test: `interior_speed_caps_follow_curvature`.

- [x] **[M] Zipper merges.**
  `constraint::merge_yield` follows the first-to-the-merge-point conflicting vehicle as a
  leader (cooperative, not gap-gated); `NetWorld::merges` marks multi-feeder lanes. Test:
  `two_lanes_zipper_merge_without_colliding`.

- [x] **[L] Lateral lane-change transition.**
  `LaneChange { from, progress }` over `LANE_CHANGE_DURATION` (2 s) blends the rendered
  pose; also reused by the seam-landing blend. Test:
  `a_lane_change_slides_the_pose_across_gradually`.

- [x] **[L] `max_accel` recalibrated.** `DriverConfig::car()` is 2.0 m/s² (±25% jitter)
  since the 2026-08-13 saturation-flow tune; see the discharge item below.

- [x] **[L] Probabilistic yellow/red running.**
  `yellow_run_prob` (aggression-scaled, committed-speed gated) and `SimConfig::red_run_prob`
  feed the crash model. Test: `a_bounded_minority_of_aggressive_drivers_run_a_stoppable_yellow`.

- [x] **[L] Through streams never land in a closed bay.** *(2026-08-12)*
  `map::retarget_pocket_landings`: a through movement wired into a turn pocket (merged
  shut at the seam) re-lands on the nearest genuine through lane — the confluence becomes
  an explicit modeled merge instead of two uncoupled streams sharing one centreline; bay
  users enter the pocket by lane change where it opens. Tagged bay→bay cluster chains are
  preserved. Test: `through_streams_never_land_in_a_closed_bay`.

- [x] **[M] Mid-box waiting for permissive lefts.** *(2026-08-12)*
  A green permissive left that fits inside the box is admitted as a *waiter*
  (`permissive_waiter_hold`): it advances past the line, stands `PERMISSIVE_HOLD_MARGIN`
  short of its first live conflict point while `permissive_pressure` (the shared
  midpoint-window logic, now impatience-aware) holds, and sweeps on when the gap or the
  change interval arrives. The stationary-waiter rule — a car standing still short of a
  point doesn't claim it — is applied uniformly in the admission gate, the in-box
  avoidance, and the pressure scan, so the opposing flow streams past the waiter's nose.
  One waiter per movement. Tests: `permissive_left_advances_into_the_box_and_waits_at_its_conflict_point`,
  the original yields-then-clears test, and the sober-burst gates.

- [x] **[L] Coordinated-actuated signal control.** *(2026-08-12)*
  Coordinated programs now run through the actuated runtime (`SignalController`) under a
  cycle-clock discipline: the progression phase (`SignalProgram::coordinated_phase`, set
  by `coordinate_green_waves`) is guaranteed its scheduled window (side phases are forced
  to clear before it opens), gaps out only past the window to phases with demand — judged
  per *phase mask*, since protected-permissive programs serve every group in the
  coordinated phase — and rests in green when nothing is waiting. Boot state is
  offset-aligned. Tests: `actuated_controller_honors_coordination_offsets_at_runtime`
  (window guarantee + stagger), `a_coordinated_signal_rests_in_green_without_side_demand`.

- [x] **[L] Online flow calibrators — built, evaluated, default off.** *(2026-08-12)*
  `DemandGenerator::enable_flow_calibration` groups observed links by ref/name, compares
  windowed sim flow to the scorecard's own AADT expectation (compression-aware), and
  nudges the corridor's origin streams by a damped, bounded factor; scorecard
  `--calibrate`. Kept off by default: see the follow-up item below for the honest
  multi-seed verdict. Test: `flow_calibration_boosts_an_underfed_observed_corridor`.

- [x] **[H] Lane-level routing, first step: two-hop lane preference.** *(2026-08-12)*
  `lanes_to_serving` judges a lane by whether its *landing lane on the next link*
  continues toward the hop after that (`second_link_on_path`: route or flow-field), so a
  car pre-positions a block early for a turn off a short block instead of landing and
  weaving; falls back to next-link service when no clean chain exists. Test:
  `lane_choice_prepositions_for_the_turn_after_next`.

- [x] **[M] Scheduler × threads composition.** *(2026-08-13)*
  The active-set scheduler now composes with every backend (the Threads auto-gate is
  gone, with its dead `scheduler_thread_limit` knob removed end-to-end): sleep
  classification rides the fused parallel pass, sleeping queued cars also skip the
  MOBIL scan on a staggered cadence (`SLEEPER_LC_PERIOD`, via the row's `slept` flag),
  and the light passes (lane-change scan, in-lane integrate) got their own measured
  parallel crossover (`LIGHT_PAR_THRESHOLD` — their rayon arms *lost* below ~8k cars,
  the integrate arm by 8×). Bench (`examples/stepbench.rs`, loaded real map): threads
  went from a net loss (7.4 ms vs 6.1 serial at 5k) to the fastest config at 4.3 ms;
  at 7k-car gridlock, 8.6 → 5.0 ms. Backend-independent sleep also removes a latent
  serial↔threads divergence. Gates: `sleep_scheduler_matches_the_all_cars_step`,
  `threads_backend_matches_serial_bit_for_bit`.

- [x] **[M] Per-lane signal detection.** *(2026-08-13)*
  `SignalController` approaches and the world's demand set are keyed by *lane*
  (movements' from-lanes) — real stop-line loops sense metal in a lane, so a through
  queue can no longer call the adjacent bay's protected-left window. Test:
  `a_protected_left_is_called_by_the_bay_not_the_through_queue`.

- [x] **[M] Saturation flow measured + launch recalibrated.** *(2026-08-13)*
  New ground-truth test `queue_discharge_hits_real_saturation_flow` (standing queue
  discharging through a resting green). Findings: a routing artifact first (unrouted
  cars "intending" the left turn measured the left's 2.7 m/s curvature cap — correct
  behavior, wrong fixture); the clean through measure was ~3.3 s/veh, acceleration-
  sensitive. `DriverConfig::car().max_accel` 1.5 → 2.0 (mid comfortable range, the tune
  the old TODO anticipated) brings ~3.05 s (≈1180 veh/h/lane); the residual vs real
  ~1.9 s is the launch time-gap stack — see the open item. `DEPARTING_SPEED` was
  A/B-measured a no-op on clean discharge and left at 3.0.

- [x] **[L] Platoon density raised to the artifact ceiling.** *(2026-08-13)*
  `PLATOON_PROB` 0.25 → 0.30 with all gates green; 0.35 still crashes the zero-crash
  `mixed_class_traffic_does_not_crash_under_sustained_demand` gate, so the coupling the
  old comment warned about persists above 0.30 — documented at the constant.

---

## Open

- [ ] **[H] Lane-level routing graph (Lanelet2) — remainder.**
  On-demand lane preference is now k-hop (see below): `lanes_to_serving` scores each
  lane by its landing-chain depth along the next `LANE_ROUTE_DEPTH` (3) hops, with
  per-depth fallback, and the keep-right/gore guard speaks the same depth measure so
  discretionary drift can't fight pre-positioning. What remains is the *standing*
  lane-graph route search — link costs that know lane-change friction
  (`Network::route_links` is link-level) — a rearchitecture touching the flow-field
  router, demand, and the GPU field solver.

- [ ] **[M] Scale within pure microsimulation (design decision 2026-08-12: no meso —
  every car is modeled distinctly, always).**
  Scheduler × threads composition LANDED (see below): threads+sleep is now the fastest
  configuration (−42% step time at 5k cars, −43% at 7k gridlock vs the old gated
  arrangement on the bench machine; `examples/stepbench.rs` measures the matrix).
  Remaining levers, none exhausted:
  - **GPU accel backend** (native): the binding-fold pass is only part of the step;
    widening what runs on-device (neighbor gather, curve scan) extends it.
  - **Per-pass parallel thresholds**: the light passes (MOBIL scan, in-lane integrate)
    now stay serial below `LIGHT_PAR_THRESHOLD` (8k) — measured crossover on the real
    map; re-measure on a bigger region where they should flip parallel.
  - **Congestion LOD** (`meso.rs` — despite the filename, a cheap *per-car* follower on
    jammed links, consistent with the no-aggregation rule): off by default; measure and
    promote if it keeps fidelity.
  - The profile's tail after composition: neighbor maps, per-group sorts, and the
    lane-change scan (~1.0–1.2 ms of a 4.3 ms step at 5k).

- [ ] **[L] Saturation flow: close the residual time-gap stack.**
  The supply side is now measured and partially calibrated (see below): stop-line
  discharge was ~3.3 s/veh, is ~3.05 s after the `max_accel` 2.0 recalibration —
  against the real ~1.9 s (~1900 veh/h/lane). The residual is structural: IDM's
  time headway plus the perception-reaction delay both apply in full during queue
  discharge, where real drivers anticipate the launch wave. Closing it means a
  launch-anticipation mechanism (not IDM-param surgery — those are guarded), then
  tightening `queue_discharge_hits_real_saturation_flow`'s band toward 2 s. This,
  not inflow scaling, is the GEH lever (the calibrator A/B proved inflow just
  queues at junctions).

- [x] **[M] Grade-separation z-ordering for fills + markings (Option A — landed 2026-08-16).**
  `render::geometry::world_bands` groups fills+markings into painter's-order **render bands**
  keyed by (grade layer, then road-class `at_grade_rank`); `draw_world` (raster/ascii) and the
  GPU (`render_band_ranges` → `gpu.rs` per-band draw) both draw each band's fill then its
  markings bottom-to-top, so an overpass band's opaque fill covers the road AND lane lines it
  crosses (verified: SF interchange `diag_overpass_views` — the marking bleed is gone), and
  same-grade overlaps resolve by road class (the user's "express ordering at equal grade" ask).
  Per-link divider/strip bodies factored (`Network::{link_dividers,link_strips}`) so markings
  bucket by band. Guards: `overpass_band_draws_after_the_surface_it_crosses`,
  `same_grade_bands_order_by_road_class`, `band_ranges_partition_the_concatenated_meshes`
  (pins the GPU range/mesh alignment, since that path is wasm-only). Goldens reblessed (junction
  fill now cleanly covers approach markings at the box). Minor accepted change: the translucent
  occupancy tint now draws on top of markings (was under) — markings stay readable through it.
  Remaining (separate): (2) **vehicles ignore layer** — an under-bridge car still draws over the
  bridge (layer is derivable `v.lane`→link→layer, just not plumbed to the instance buffer);
  (3) density/mass overlay not banded (see the tint note); (4) no elevation cue (overpass same
  colour, occlusion only).

---

## Do NOT destabilize (already solid)

Ballistic integrator with sub-tick stop handling, the min-of-constraints longitudinal
architecture, canonical IDM parameters (T=1.5, s0=2, b=1.5), impatience-based gap acceptance
(`effective_critical_gap`), driver heterogeneity sampling, actuated gap-out/max-out logic, and
the fundamental-diagram validation in `world.rs`. Change these only against their tests.

---

## Network & intersection geometry (2026-08-12 remodel — landed)

Stage 1: junction clusters carry intersection identity; one-way axes recentred
(`center_oneway_axes`); end-direction cache; entry-link signal grouping. Stage 2:
junction-owned arm mouths (`Junction::mouths`); through-rank wiring; `align_through_seams`;
corner-clamped interiors; `untangle_parallel_movements`. Stage 3: Lanelet2-style shared
lane boundaries (`Network::lane_bounds`) as the single geometry authority (vehicles,
dividers, strips, edge lines, mouths); pocket bays as real edge geometry;
`map::stitch_seam_bounds` closes mutual-primary through seams exactly (85% < 5 cm,
98% < 20 cm). Guarded by `el_camino_through_lanes_stay_laterally_continuous`,
`junction_interiors_hug_their_corners_and_never_swap_lanes`,
`through_seams_are_stitched_shut_across_the_map`, and the golden junction screenshots.

Stage 4 (2026-08-15 — junction *render* decomposition): `render::geometry::junction_rings`
no longer paves a sprawling divided crossing as one convex hull (the borked El Camino ×
Millbrae blue box slicing mid-block). A cluster within `SPRAWL_RADIUS` (18 m) stays one box
(street-band for a lone node, `convex_hull` of mouths for a tight split); a wider cluster
decomposes into per-member-node `junction_box`es from each node's `arm_mouth` cross-sections,
tiled at perpendicular bisectors, with the real medians left unpaved. `junction_box` groups
arms into streets by the link's own end direction (not the skew-prone mouth-midpoint bearing)
and spans the node. Marker outlines drawn only for real crossings (≥ 3 arms, ≥ 2 streets, area
≥ 40 m², compactness ≥ 0.55 — a wedge clip-artifact is dropped), non-overlapping. Stop lines /
signal heads snap to each approach's *local* box. Golden screenshots reblessed; guarded by
`a_divided_crossing_decomposes_into_multiple_local_boxes`, `street_groups_*`, `junction_box_*`,
`convex_hull_*`, and the realism metrics (`the_crossing_core_is_solid_pavement`,
`no_stray_nub_islands`). Diagnostics: `diag_junction_structure`, `diag_real_map_views` (ignored).

Stage 4b (2026-08-16 — cross-city hardening; assessed SF / San Carlos / Columbus / peninsula):
divided arterials render excellently everywhere (columbus_0's 4-box diamond, sf_0/sf_4). Two
issues found + fixed. (1) **Complex tangles** (offset/multi-way crossings, e.g. sancarlos_5)
decomposed into all-sub-threshold boxes → no outline + notchy fill. Fix: the sprawl path now
*tries* per-node decomposition and keeps it only when `marked_boxes` finds ≥ 2 clean
non-overlapping crossings; otherwise falls back to one unifying `whole_cluster_box` hull. The
`SPRAWL_RADIUS` is now just a gate to try decomposition, not the final decision. (2)
**Interior-link marking scribble** bled across boxes: `Network::{road_strips,lane_dividers}`
now skip `link_is_junction_interior` links (both ends one cluster), as arrows already did —
benefits the browser feed too (parity). Golden fixtures reblessed (cleaner box interiors).
Guards: `marked_boxes_counts_clean_nonoverlapping_crossings`,
`every_cluster_is_decomposed_or_unified_never_a_sliver_scatter` (invariant over SF/San Carlos/
Millbrae real maps), `interior_link_markings_do_not_scribble_across_the_box`. Diagnostic:
`diag_city_views` (env `MAP=<name>`, ignored — also prints per-junction conflict count + an
interchange-node safety scan).

- [x] **[L] Freeway free-flow merges no longer get a spurious crossing marker.** *(2026-08-16)*
  A merge/diverge (the ugly leaf outline on the peninsula gores) is not an at-grade crossing.
  `render::geometry::outlined_boxes` now drops the blue marker for a cluster whose every member
  node is `Network::is_interchange_node` (all incident links grade-separated); the ribbons +
  junction fill still pave the gore, only the outline is suppressed. Verified safe against the
  at-grade cases — columbus_0 etc. are `primary`/`secondary` Arterial (columbus.json is a FULL
  network despite the README's highways-only recipe), so `is_interchange_node=false` → they
  keep markers. Guards: `a_freeway_merge_carries_no_crossing_outline`,
  `a_divided_crossing_decomposes_into_multiple_local_boxes` (now also asserts the at-grade
  crossing *keeps* its outline). Residual (accepted): the rare genuine trunk×trunk at-grade
  signalized crossing would also lose its outline — but only the outline; fill/markings/signal
  heads stay. A finer gate is noisy (ramp meters read as "signalized", freeway weaves as
  "conflict"), so the blunt all-grade-separated test is the right call.

---

## Suggested order

What remains: the **launch-anticipation saturation residual** (the honest GEH lever),
the **standing lane-graph route search**, and the **GPU/profile-tail scale work** —
each scoped in its item above.
