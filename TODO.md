# TODO — Traffic realism improvements

Prioritized backlog for making driver and traffic-light behavior more closely match
real life. References are by **file + function/symbol** (stable across the commented and
comment-free copies), not line numbers.

Priority legend: **[H]** high realism impact · **[M]** medium · **[L]** low / calibration.
Each item notes the fix and, where one exists, the nearby test to validate against.

---

## High impact

- [ ] **[H] Apply green-wave coordination at runtime.**
  `map::coordinate_green_waves` computes a per-program `offset`, but the live actuated
  controller (`junction::SignalController`) never reads `offset` — every signal boots at
  `phase 0, elapsed 0` (`SignalController::build`) and drifts independently. Net effect:
  **no platooning / green waves**, the dominant factor in arterial travel time.
  - Cheap first step: seed each `SignalRuntime` from the coordinated offset at build so
    signals at least *start* phase-aligned.
  - Real fix: coordinated-actuated control — a background cycle length with a fixed
    force-off/yield point for the coordinated phase; actuation floats only the
    non-coordinated phases.
  - Note: `SignalProgram::state_of` (offset-aware, `signal.rs`) is currently used only by
    build-time tests, not the sim path.

- [ ] **[H] Make acceleration noise zero-mean.**
  `constraint::accel_noise` returns `-sigma * uniform01(...)` ∈ (−σ, 0] — it only ever
  *subtracts* acceleration (mean ≈ −σ/2), systematically depressing fleet speed and never
  seeding the small over-accelerations behind realistic stop-and-go waves. Switch to
  zero-mean, ideally temporally correlated (Ornstein–Uhlenbeck) so it reads as throttle
  drift rather than white jitter. Validate mean speed against the ring fundamental-diagram
  tests in `world.rs`.

- [ ] **[H] Derive signal change/clearance intervals kinematically (ITE).**
  Yellow is taken straight from the plan; all-red is a constant `ALL_RED = 2.5`
  (`junction.rs`) regardless of approach speed or intersection width.
  - Yellow ≈ `t_perception + v / (2·(a + g·grade))` (≈3.0 s @ 25 mph, ≈4.3 s @ 45 mph).
  - All-red ≈ `(W + L) / v`.
  - Both derivable from the approach lane `speed_limit` + node geometry. Improves
    dilemma-zone behavior (interacts with `can_stop_before`) and capacity.

- [ ] **[H] Protected-permissive left turns + right-turn-on-red.**
  `map::assign_signal_program` gives conflicting left groups their own protected phase
  (green scaled to `0.45×`). Real signals mostly run protected-**permissive** or
  permissive-only lefts (turn on green, yield to oncoming). Also, `Red` is currently an
  absolute stop for every movement in `gather_context` (no RTOR).
  - Model a permissive left as a green movement that must yield to oncoming through, reusing
    the existing gap-acceptance machinery (`conflicting_priority_traffic` / `merge_conflict`).
  - Allow right-turn-on-red after a full stop when a gap exists.

---

## Medium impact

- [ ] **[M] Apply reaction-time lag to signals/stop-lines, not just the leader.**
  `gather_context` delays the *leader's* perceived gap/speed by `reaction_time/dt` ticks,
  but signal onset, stop lines, and yield lines use instantaneous state — so queues discharge
  with zero start-up latency. Add the same perception-reaction lag (or an explicit per-driver
  green-startup delay) to reproduce real **start-up lost time (~2 s)** and **saturation
  headway (~1.9 s/veh)**. Also note the delay is coarsely quantized (`(0.5/0.2).round()` = 2
  ticks = 0.4 s).

- [ ] **[M] Earlier, urgency-scaled mandatory lane changes.**
  `mandatory_change` only fires when the *adjacent* lane serves the route, and the MOBIL
  threshold in `best_lane_change` is a constant with no distance-to-turn term — so a car
  several lanes from its turn pocket weaves at the last second and can miss the turn. Mirror
  the gap-acceptance impatience already in `effective_critical_gap`: propagate
  `lanes-to-cross × urgency(distance_to_turn)` so the change threshold decays as the junction
  nears.

- [ ] **[M] Right-turners should yield on approach.**
  `should_yield_to` returns `false` for `my_turn == Right`, so a right-turn from a
  minor/stop approach never yields to major through traffic before entering; it's only
  arbitrated mid-crossing by first-to-conflict-point + id tiebreak, which ignores
  right-of-way. Make right-turn-from-minor yield to conflicting major through/left.

- [ ] **[M] Add lane-usage asymmetry (keep-right / passing-lane bias).**
  MOBIL (`mobil.rs` + `best_lane_change`) is symmetric — no lane preference. Add Treiber's
  asymmetric bias term so slower traffic settles right and overtaking uses the left,
  producing realistic lane distributions and less pointless symmetric churn.

- [ ] **[M] FIFO ordering at all-way stops.**
  Service is decided by `priority_key` (speed limit, lane count) + time-to-arrival, not
  first-come-first-served — the actual all-way-stop rule. Stamp arrival time on full stop and
  serve in order.

---

## Low impact / calibration

- [ ] **[L] Lateral lane-change transition.**
  Lane changes teleport to the target lane at the same arc position in one tick. Real changes
  take ~2–4 s and occupy both lanes; add a lateral-transition duration (mostly visual + small
  capacity effect) when touching the renderer.

- [ ] **[L] Revisit `max_accel = 1.0 m/s²` for cars** (`config::DriverConfig::car`).
  On the low side (comfortable ≈ 1.5–2.5). Livelier launches, but entangled with
  saturation-flow calibration — tune against the fundamental-diagram / discharge tests, not
  blind.

- [ ] **[L] Probabilistic yellow/red-running for aggressive drivers.**
  `can_stop_before` is a clean binary. Let a small fraction of high-`desired_speed` samples
  run late yellows to add realistic variance.

---

## Network & intersection architecture

From the 2026-08-12 geometry remodel (stage 1 landed: junction clusters carry intersection
identity, one-way axes recentred, merge-lane topology fixed). Ideas drawn from how
SUMO and Lanelet2 model networks; each names the machinery it builds on.

- [ ] **[H] Stage 2 — junction-owned arm mouths.**
  `Junction` (network.rs) should compute each arm's stop-line cross-section (position,
  direction, lane span) and reconcile through-lane correspondence across the box, with
  links plugging into mouths rather than node points. Partially started: the
  through-alignment pass. Validate against
  `el_camino_through_lanes_stay_laterally_continuous`.

- [ ] **[H] Stage 3 — lane-boundary geometry (Lanelet2's core idea).**
  Store each lane's left/right boundary polylines, shared between neighbours, instead of
  centreline + `(index + 0.5)·LANE_WIDTH` offsets (`Network::lane_lateral_offset`).
  Adjacent lanes then *cannot* misalign — the entire offset-drift bug class becomes
  unrepresentable. Big refactor; do after stage 2 settles.

- [ ] **[H] Lane-level routing graph (Lanelet2).**
  Route over lanes with lane-change edges and costs, not links
  (`Network::route_links` is link-level). Cars pre-position for turns blocks early,
  fixing last-second turn-lane misses at the root — the principled superset of the
  urgency-scaled mandatory lane change item above.

- [ ] **[M] Zipper merges (SUMO's `zipper` junction type).**
  Now that `spread_merge_feeders` makes a ramp share the curb lane at merges, add an
  alternating-priority rule at the shared-lane merge point instead of pure gap
  acceptance — realistic fairness at lane drops and on-ramps.

- [ ] **[M] Mid-box waiting positions for permissive lefts (SUMO internal junctions).**
  A left-turner advances into the box and waits at its conflict point, clearing on
  yellow. Builds on interior commit + slot admission; pairs with the
  protected-permissive left item above.

- [ ] **[M] Curvature-limited interior speeds.**
  Apply `v = √(a_lat·r)` to interior Bézier curvature the way `min_radius_ahead`
  already limits link curves — sharp turns through boxes slow down naturally (SUMO does
  this on internal lanes).

- [ ] **[M] Queue-based mesoscopic links (SUMO meso) for the 1M+ goal.**
  Beyond the per-car congestion LOD: uncongested links become event-driven FIFO queues
  with capacity servers, no per-car integration. SUMO's ~10–100×; likely the only path
  to a million vehicles. The active-set scheduler is a step in this direction.

- [ ] **[L] Online flow calibrators (SUMO).**
  Devices that nudge gateway inflows toward observed link counts *during* the run — the
  direct lever for raising the scorecard's GEH<5 share from ~11% toward the 0.85
  aspirational gate.

---

## Do NOT destabilize (already solid)

Ballistic integrator with sub-tick stop handling, the min-of-constraints longitudinal
architecture, canonical IDM parameters (T=1.5, s0=2, b=1.5), impatience-based gap acceptance
(`effective_critical_gap`), driver heterogeneity sampling, actuated gap-out/max-out logic, and
the fundamental-diagram validation in `world.rs`. Change these only against their tests.

---

## Suggested order

Start with **green-wave offsets** and **zero-mean noise** — both small, high-impact, and each
has nearby test scaffolding to validate against.
