# TODO — Traffic realism improvements

Prioritized backlog for making driver and traffic-light behavior more closely match
real life. References are by **file + function/symbol** (stable across the commented and
comment-free copies), not line numbers.

Priority legend: **[H]** high realism impact · **[M]** medium · **[L]** low / calibration.
Checked items are landed and name the symbol and guarding test — audit them against the
code, not this file, if in doubt.

---

## Landed (audited 2026-08-12)

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

- [x] **[L] `max_accel` recalibrated.** `DriverConfig::car()` is 1.5 m/s² (±25% jitter).

- [x] **[L] Probabilistic yellow/red running.**
  `yellow_run_prob` (aggression-scaled, committed-speed gated) and `SimConfig::red_run_prob`
  feed the crash model. Test: `a_bounded_minority_of_aggressive_drivers_run_a_stoppable_yellow`.

- [x] **[L] Through streams never land in a closed bay.** *(2026-08-12)*
  `map::retarget_pocket_landings`: a through movement wired into a turn pocket (merged
  shut at the seam) re-lands on the nearest genuine through lane — the confluence becomes
  an explicit modeled merge instead of two uncoupled streams sharing one centreline; bay
  users enter the pocket by lane change where it opens. Tagged bay→bay cluster chains are
  preserved. Test: `through_streams_never_land_in_a_closed_bay`.

---

## Open

- [ ] **[M] Mid-box waiting positions for permissive lefts (SUMO internal junctions).**
  A permissive left currently yields *at the stop line* (`soft_yield` via
  `permissive_must_yield`); real drivers advance into the box, wait at the conflict
  point, and clear on yellow — worth ~1–2 extra lefts per cycle. Requires three
  coordinated changes, none safe alone:
  1. `box_entry_blocked` + the stop-line soft yield must admit a green permissive left
     into the box while oncoming flows (bounded — one waiter per conflict point).
  2. An in-box hold: clamp the left at `conflict_arc − 1` while oncoming is within its
     window; release on gap or oncoming yellow/red (the all-red already covers the exit).
  3. The in-box avoidance's first-to-the-point rule (`they_go_first = their_dist <
     my_dist`) must treat a *stationary* waiter as not claiming the point — today a
     parked left would brake every oncoming through (box gridlock), and without the
     hold in (2) the nearest-first rule would instead let the left barge. Interacts with
     the crash model (a waiter is a realistic amber-runner target) — validate against
     the sober-burst gates and `permissive_left_yields_to_oncoming_then_clears_without_colliding`.

- [ ] **[H] Lane-level routing graph (Lanelet2).**
  Route over lanes with lane-change edges and costs, not links
  (`Network::route_links` is link-level). Cars pre-position for turns blocks early,
  fixing last-second turn-lane misses at the root — the principled superset of the
  (landed) urgency-scaled window. Rearchitecture: touches the flow-field router,
  demand, and the GPU field solver.

- [ ] **[M] Queue-based mesoscopic links (SUMO meso) for the 1M+ goal.**
  Beyond the per-car congestion LOD (`meso.rs`): uncongested links become event-driven
  FIFO queues with capacity servers, no per-car integration. SUMO's ~10–100×; likely the
  only path to a million vehicles. The active-set scheduler is a step in this direction.

- [ ] **[L] Online flow calibrators (SUMO).**
  Devices that nudge gateway inflows toward observed link counts *during* the run — the
  direct lever for the scorecard's GEH<5 share. Note before building: that share moves
  6–14/72 links on seed alone (measured 2026-08-12), so the calibrator and its evaluation
  must be variance-aware (multi-seed) or it will chase noise.

- [ ] **[L] Coordinated-actuated signal control.**
  The residual from the green-wave item: background cycle with force-off/yield points,
  actuation floating only the non-coordinated phases (`SignalController` currently runs
  coordinated programs fixed-time).

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

---

## Suggested order

**Mid-box permissive lefts** is the highest-value bounded item (capacity + realism at
every signalized left); its three-part design above is ready to build against existing
tests. The two rearchitectures (lane-level routing, meso queues) each deserve a dedicated
plan; calibrators only pay off with multi-seed evaluation.
