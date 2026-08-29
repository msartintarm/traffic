# Kinematics plan: axle-based turning everywhere

Design (2026-08-28) for making every car move the way a steered, four-wheeled
vehicle moves — through corners, through degenerate geometry, and through lane
changes — with scenario regression tests as the tuning harness. References are
by **file + function/symbol**, per TODO.md convention.

## Status (2026-08-28): P1–P3 executed

Implemented and green (`cargo test --features import`: 373 lib + 16 load + 3
validation). What shipped, including deviations from the design below:

- **P1** `net_world.rs::track_path` replaces `settle_headings`: rear-axle
  state `NetVehicle::kin` (pose = `kin` + `AXLE_FRONT·len` forward), pure
  pursuit over `reference_point` (lane → interior → landing-lane walk with the
  same geometric-continuity `skip` as `land_or_hold`), κ clamped by
  `AXLE_STEER_TAN/(AXLE_WHEELBASE·len)` and grip. Two design corrections found
  by the motion audit: (1) **rolling is the longitudinal sync** — `ds =
  clamp(long-projection, 0, 1.5·v·dt + 0.3)` and `θ += κ·ds`, so wheels never
  roll backward, arc holds stop the car exactly, backward arc jumps become a
  short pause, and no rotation happens without ground motion; a **maneuver
  floor** (`ds ≥ 0.5·v·dt` when lateral error > 1 m) breaks the frozen-heading
  fixed point of pure projection. (2) The divergence bleed is capped at
  `0.25·ds`, making rear-axle slip structurally ≤ ~14°.
- **P2** came free: the pose ignores the lateral blend (which lives on as the
  divergence/longitudinal reference), and the tracker chasing the target-lane
  reference produces the steered S-curve.
- **P3** `examples/motion_audit.rs` (renamed from spin_audit): crab and
  curvature judged **at the rear axle** — the front bumper legitimately sweeps
  `atan(κ·axle)` sideways in turns, so measuring there flags every real
  corner. Millbrae, 10 sim-min: 0 flips / 0 spins / 0 tight arcs / 0 crab
  events, worst curvature 0.233 < 0.26 limit, divergences 0.38 % of moving
  car-ticks (clustered at the known degenerate mouths). Tests:
  `a_lane_change_is_a_steered_s_curve`, `a_right_turn_sweeps_like_a_car`
  (net_world.rs); `valencia_madera_turns_are_physical` (passage-graded),
  extended `no_impossible_yaw_rates_under_city_demand` (+crab, +curvature,
  +divergence-rate bound) in tests/load.rs. The U-turn scenario was dropped:
  `map.rs` never builds U-turn movements, so the behavior doesn't exist.
- Cost: step 5.9 → 6.5 ms at ~5.3 k cars (the tail lap; parallelizable if it
  ever matters).
- Open: the fresh-build Valencia × Madera spins the sim cannot reproduce —
  suspected render-side (bridge prev-pose interpolation / id reuse per the
  render-id invariant); verify in the browser after the next wasm deploy.

## The three reports, diagnosed

1. **"Cars don't turn like 4-wheeled cars."** Correct, structurally: today the
   pose is a point glued to path geometry (`net_world.rs::path_pose`), rotated
   about its *center*. The 2026-08-28 heading layer (`settle_headings`) bounds
   yaw rate, but position never derives *from* heading — so a car can translate
   in a direction it isn't quite facing (bounded crab), pivots about its middle
   in tight corners, and the nose/rear never sweep the way a steered wheelbase
   sweeps. Fixing this means inverting the relationship: heading steers,
   position follows (a rear-axle bicycle model), and the path becomes the
   *reference being tracked* rather than the pose itself.

2. **"Still spinning at Valencia Drive × Madera Way."** Investigated on the
   current working copy (node 242 at (−756.8, −857.4), 12 movements): every
   interior is a healthy curved corner (6.5–12 m, none straightened), and a
   focused trace of 115 demand-driven passages shows normal 90° turns, zero
   heading-direction reversals, ≤ 5° crab. The working copy cannot reproduce a
   spin there. The 2026-08-28 fixes are **uncommitted and undeployed** —
   `web/out` predates them (Aug 11) — so the observed spinning is almost
   certainly the stale bundle. Phase 0 pins this: deploy, then lock the
   intersection with a named regression test so any regression is caught by
   name. If spins reproduce *post*-deploy, the passage-trace harness (below)
   is the reopening tool.

3. **"Lane changes should turn on the axle too."** Today a lane change
   translates the pose laterally between lane lines with heading pinned to the
   lane tangent (`path_pose`'s `lane_change` branch) — a crab-slide. Under the
   bicycle model the lateral move becomes a genuine S-curve: yaw out,
   straighten, yaw back, with position following the front wheels.

## Design: rear-axle pose tracking (pure pursuit over the arc reference)

The two-layer split stays. **Dynamics remain arc-based and untouched** —
IDM/MOBIL/signals/queues/boundary logic all operate on lane arc positions
exactly as today. What changes is the world pose: from a pure function of arc
position to a per-vehicle kinematic state `(x, y, θ)` at the **rear axle**
that *tracks* the arc-derived reference.

Per vehicle, per tick (replaces `settle_headings`):

- Reference point: the path position at `arc + L_d` lookahead,
  `L_d = clamp(0.8·v, 2 m, 12 m)`, walked lane → interior → landing lane the
  same way the leader horizon walks segments.
- Pure pursuit: `α` = bearing of reference minus `θ`; curvature command
  `κ = 2·sin(α)/L_d`, clamped to `κ_max = tan(δ_max)/wheelbase` (δ_max 35°;
  wheelbase per `DriverConfig` class — car ~2.7 m, bus/truck longer), and to
  the grip bound `YAW_GRIP_LAT/v²`.
- Integrate: `θ += v·κ·dt`; rear axle advances along `θ`. The nose sweeps a
  wider radius than the rear — the visible signature of a real turning car.
- `vehicle_world_pose` returns the kinematic pose; render + collision OBBs
  (`car_rect`) inherit it. The render instance anchor moves from center to
  rear-axle so rotation reads correctly (`bridge.rs` instance marshaling).
- **Divergence guard** (the collision-tuning knob): `ε` = distance from
  kinematic position to the arc position. If `ε > E_max` (start 3 m), bleed
  position back toward the reference at a bounded lateral rate and count it
  (`kinematic_divergence` metric, surfaced by the audit). Tight guard ≈
  today's behavior; loose guard = full kinematics. This is where "may
  increase collisions" is governed, and the scenario tests below are how it
  gets tuned.
- Stationary cars skip integration (already true); state is id-carried across
  demand rebuilds like `heading` today; per-car math only, so determinism and
  the parallel step are unaffected; GPU accel path untouched (accelerations
  only). Perf budget: same O(n) tail pass as `settle_headings`; watch
  `stepbench`.

Lane changes: on commit, the tracker's reference switches to the target
lane's geometry (plus lookahead) — pure pursuit produces the S-curve.
`LaneChange` keeps its decision/arc bookkeeping role; the pose-side lateral
blend in `path_pose` and the landing-blend arithmetic reduce to reference
switches. MOBIL semantics unchanged; an aborted change is just another
reference switch (steer back).

Intersections: reference = interior path, unchanged. Pure pursuit rounds any
residual sampling kinks, and the straightened stubs at the 17 degenerate-mouth
nodes stop looking like pivots: after a stub landing the reference is ahead on
the new road and the car steers a real bounded arc to it — occasionally
swinging wide into the opposing mouth at speed, which is accepted realism
(crash detection already treats that as signal, not bug).

## Phases

**P0 — deploy verification + pinning (small).** Commit the 2026-08-28 fixes,
rebuild the wasm packs, redeploy, and re-observe Valencia × Madera. Extract
the passage-trace harness into a test helper (`tests/load.rs::passage_trace`:
capture every vehicle within radius R of a node; per passage report net
rotation, heading-direction reversals, worst crab, peak per-tick Δθ). Named
test `valencia_madera_turns_are_physical`: net rotation within ±25° of the
movement's required turn, zero reversals above 0.02 rad, crab < 10°.

**P1 — the bicycle integrator (core).** Rear-axle state + pure-pursuit
tracker replacing `settle_headings`; wheelbase on `DriverConfig`; divergence
guard + counter; render anchor shift. Extend `spin_audit` → `motion_audit`
with two new physical invariants: **crab** (angle between velocity direction
and heading ≤ 15° when v > 3 m/s — the "four wheels" invariant the current
code cannot satisfy by construction) and **curvature** (`|Δθ|/Δs ≤ κ_max+ε`).
Scenario tests on synthetic fixtures:
- `right_turn_sweeps_like_a_car`: 90° corner; monotone heading, peak κ ≤
  κ_max, front-corner path radius > rear-axle path radius, no reversal.
- `stub_landing_recovers_within_a_car_length`: degenerate-mouth fixture;
  heading settles to the lane tangent within N metres, no oscillation,
  lateral overshoot under a lane width.
- `u_turn_takes_a_half_circle`: bounded by the turning circle, no pirouette.
Existing pose tests migrate: `no_vehicle_pose_teleports_under_high_load` and
the seam-slide tests assert bounded deviation from the lane line instead of
exact centerline-following.

**P2 — steered lane changes.** Reference switch at commit/abort; delete the
pose-side lateral blend. Scenario tests:
- `lane_change_is_an_s_curve` (15 and 30 m/s): heading deviates 3–8°, exactly
  two yaw phases (out, back), lateral accel ≤ A_LAT, completes in 3–5 s, crab
  < 3° throughout.
- `abandoned_lane_change_steers_back` (MOBIL abort mid-change).
- Highway seam remaps (existing `a_seam_lane_remap_slides_the_pose_across_
  gradually`) revalidated with yaw.
Then re-check the crash-rate band in `scorecard` — side-swipe geometry is now
real, so the rate may move; the scenario thresholds above are the tuning
levers, and the accepted band gets pinned in the validation tier.

**P3 — map-level acceptance.** `motion_audit` thresholds become asserts in
the dynamic regression test (flips/spins/tight-arcs stay zero; crab and
curvature bounds added; divergence counter bounded) over Millbrae and SF.
Divergence clusters, if any, point at remaining geometry defects — that is
the trigger for finally doing the chart-level mouth-ordering fixpoint at the
17 degenerate nodes, which stays deferred until the data demands it.

## Risks & notes

- The lookahead's cross-boundary path walk is the only new shared machinery;
  reuse the leader-walk pattern rather than inventing a second walker.
- Buses/trucks get class wheelbases — long vehicles will genuinely swing wide
  on residential corners; expected, and exactly what the crab/curvature
  invariants permit.
- Congestion-LOD follower cars run the same integrator (their arc positions
  update; pose must too).
- Trains are unaffected (rail poses are schedule-driven, outside the fleet).
