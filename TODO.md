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

## Do NOT destabilize (already solid)

Ballistic integrator with sub-tick stop handling, the min-of-constraints longitudinal
architecture, canonical IDM parameters (T=1.5, s0=2, b=1.5), impatience-based gap acceptance
(`effective_critical_gap`), driver heterogeneity sampling, actuated gap-out/max-out logic, and
the fundamental-diagram validation in `world.rs`. Change these only against their tests.

---

## Suggested order

Start with **green-wave offsets** and **zero-mean noise** — both small, high-impact, and each
has nearby test scaffolding to validate against.
