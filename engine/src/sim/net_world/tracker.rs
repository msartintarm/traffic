//! The kinematic pose layer: a rear-axle bicycle model that *tracks* the
//! sim's arc position (pure pursuit + interior curvature feed-forward),
//! bounded by real steering geometry, a speed-dependent comfort envelope,
//! and a human steering rate. Positions/decisions stay arc-based; this file
//! owns everything about how the drawn, collidable body physically moves.

use super::*;

/// Speed-dependent comfortable lateral acceleration (m/s²), AASHTO
/// side-friction style: drivers accept ~0.35–0.4 g threading an intersection at
/// walking-to-jogging speed and only ~0.12 g at highway speed, falling roughly
/// linearly in between (Green Book side-friction factors ≈ 0.38 at 15 km/h down
/// to ≈ 0.09–0.12 at 110+ km/h). One envelope drives both how fast a car takes
/// a curve of radius r and how hard the tracker may steer at its current speed.
pub(super) fn comfort_lat(v: f64) -> f64 {
    (3.9 - 0.145 * v).clamp(1.2, 3.9)
}
/// The speed a driver chooses for a curve of radius `r` under the envelope:
/// solve `v² = r · a(v)` with the linear branch of [`comfort_lat`], capped by
/// the low-speed plateau.
pub(super) fn comfort_speed(r: f64) -> f64 {
    if !r.is_finite() {
        return f64::INFINITY;
    }
    let quad = (-0.145 * r + (0.021025 * r * r + 15.6 * r).sqrt()) * 0.5;
    quad.min((3.9 * r).sqrt())
}
/// How far ahead a curve is read.
pub(super) const CURVE_LOOKAHEAD: f64 = 45.0;
/// Bicycle-model bounds for [`NetWorld::track_path`]. Steering geometry:
/// maximum front-wheel angle tan 35° (a 10.5–12 m curb-to-curb turning circle —
/// rear-axle path radius ≈ 3.9 m for a 4.5 m car); the wheelbase is [`AXLE_WHEELBASE`] of the
/// vehicle length (car ≈ 2.7 m of 4.5 → ~3.9 m minimum turning radius; a bus
/// proportionally wider), and the rear axle sits [`AXLE_FRONT`] of the length
/// behind the front bumper (the pose anchor the rest of the sim uses). Grip:
/// the tyre lateral limit (m/s², ~0.7 g) that caps curvature at speed —
/// `κ ≤ min(tan δ_max / wheelbase, grip / v²)`.
pub(super) const AXLE_STEER_TAN: f64 = 0.70;
pub(super) const AXLE_WHEELBASE: f64 = 0.6;
pub(super) const AXLE_FRONT: f64 = 0.8;
pub(super) const YAW_GRIP_LAT: f64 = 7.0;
/// Road-wheel steering rate (rad/s): a driver winds lock on and off through the
/// steering ratio, reaching full lock from centre in ~1.3 s — the wheel cannot
/// snap. Divided by the wheelbase this bounds how fast the tracked curvature
/// may change per second.
pub(super) const STEER_RATE: f64 = 0.55;
/// Pure-pursuit lookahead from the rear axle: `0.8·v` clamped to this range
/// (m). The floor sits just above the wheelbase — tight enough to hug a city
/// corner, stable enough not to oscillate; the ceiling keeps highway tracking
/// calm.
pub(super) const LOOKAHEAD_MIN: f64 = 2.8;
pub(super) const LOOKAHEAD_MAX: f64 = 12.0;
/// The divergence guard — the knob that trades path-hugging against full
/// kinematics. Lateral deviation of the kinematic pose from the arc reference
/// beyond this (m) bleeds back at the given rate (m/s) and is counted
/// (`kinematic_divergences`); a cluster of counts marks defective geometry.
/// Longitudinal deviation is corrected *exactly* every tick instead: queue
/// spacing, stop lines, and leader gaps are arc-truth, and the drawn car must
/// sit where the sim says it is along the road.
pub(super) const LAT_DIVERGE_MAX: f64 = 3.0;
pub(super) const LAT_CORRECT_RATE: f64 = 2.0;

/// Signed shortest rotation `a → b`, in (-π, π] — so the heading slew always
/// turns the short way round.
pub(super) fn shortest_angle(a: f64, b: f64) -> f64 {
    let mut d = (b - a) % std::f64::consts::TAU;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    if d <= -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d
}

impl NetWorld {
    /// A vehicle's current world pose `[x, y, heading]` at the front bumper —
    /// the anchor `car_rect` and the render use — derived from the rear-axle
    /// bicycle state [`track_path`](Self::track_path) advances. Heading steers,
    /// position follows the wheels: the pose can only move the way a steered
    /// four-wheeled vehicle moves.
    pub fn vehicle_world_pose(&self, v: &NetVehicle) -> [f64; 3] {
        let f = AXLE_FRONT * v.driver.vehicle_length;
        [v.kin[0] + f * v.kin[2].cos(), v.kin[1] + f * v.kin[2].sin(), v.kin[2]]
    }

    /// A freshly spawned vehicle's kinematic state: rear axle placed so the
    /// front bumper sits exactly on the lane point, heading along the tangent.
    pub(super) fn spawn_kin(net: &Network, lane: LaneId, position: f64, driver: &DriverConfig) -> [f64; 3] {
        let p = net.lane_point(lane, position);
        let f = AXLE_FRONT * driver.vehicle_length;
        [p[0] - f * p[2].cos(), p[1] - f * p[2].sin(), p[2]]
    }

    /// The pose pure path geometry dictates (public for audits/tests): where the
    /// sim's arc position puts the front bumper, against which the kinematic pose
    /// is synced. See [`path_pose`](Self::vehicle_arc_pose).
    pub fn vehicle_arc_pose(&self, v: &NetVehicle) -> [f64; 3] {
        self.path_pose(v)
    }

    /// The pose pure path geometry dictates — the interior crossing path when
    /// inside a node, otherwise the lane position. Its heading is the raw path
    /// tangent (it can step discontinuously at a vertex or a seam); the kinematic
    /// heading steers toward it but never faster than a car can yaw.
    pub(super) fn path_pose(&self, v: &NetVehicle) -> [f64; 3] {
        if let Some(c) = v.crossing {
            let s = self.crossing_arc(v).clamp(0.0, self.network.interior(c.movement).len);
            return self.network.interior_point(c.movement, s);
        }
        let cur = self.network.lane_point(v.lane, v.position);
        let Some(lc) = v.lane_change else { return cur };
        let arc = self.network.lane(v.lane).start_offset + v.position;
        let from_lane = self.network.lane(lc.from);
        let from_pos = arc - from_lane.start_offset;
        if from_pos < 0.0 || from_pos > from_lane.length {
            return cur;
        }
        let from = self.network.lane_point(lc.from, from_pos);
        let t = lc.progress.clamp(0.0, 1.0);
        [from[0] + (cur[0] - from[0]) * t, from[1] + (cur[1] - from[1]) * t, cur[2]]
    }

    pub fn vehicle(&self, id: u32) -> Option<&NetVehicle> {
        self.fleet.rows.iter().find(|v| v.id == id)
    }

    /// The world point the tracker steers toward: `ld` metres further along the
    /// path than the vehicle's arc position — through the interior and onto the
    /// landing lane when the lookahead crosses a node (geometric continuity, the
    /// same `skip` the landing rebase applies), clamped at the stop line before
    /// a boundary is committed (a car aims at the stop bar until its crossing
    /// starts, then the reference sweeps through the turn). Reads the *target*
    /// lane during a lane change — that jump in the reference, chased through
    /// the lookahead, is precisely what makes the change a steered S-curve.
    pub(super) fn reference_point(&self, v: &NetVehicle, ld: f64) -> [f64; 2] {
        if let Some(c) = v.crossing {
            let it = self.network.interior(c.movement);
            let s = self.crossing_arc(v) + ld;
            if s <= it.len {
                let p = self.network.interior_point(c.movement, s.max(0.0));
                return [p[0], p[1]];
            }
            let to = self.network.movement(c.movement).to_lane;
            let start = self.network.lane_point(to, 0.0);
            let skip = ((it.exit[0] - start[0]) * start[2].cos() + (it.exit[1] - start[1]) * start[2].sin()).max(0.0);
            let p = self.network.lane_point(to, (s - it.len + skip).min(self.network.lane(to).length));
            return [p[0], p[1]];
        }
        let lane = self.network.lane(v.lane);
        let p = self.network.lane_point(v.lane, (v.position + ld).min(lane.length));
        [p[0], p[1]]
    }

    /// Advance every moving vehicle's bicycle-model pose one tick: pure pursuit
    /// of the lookahead reference bounds curvature to what steering geometry and
    /// grip allow, then the rear axle rolls forward along the new heading. The
    /// arc reference can step discontinuously (a polyline vertex, a seam, a
    /// landing, a lane-change retarget); this layer guarantees the *pose* never
    /// does — a car physically cannot spin in place, crab sideways, or rotate
    /// while stationary. Longitudinal error against the canonical arc pose is
    /// zeroed each tick (visual queue spacing is arc-truth); lateral error is
    /// resolved by steering alone unless it exceeds the divergence guard.
    pub(super) fn track_path(&mut self, dt: f64) {
        // Per-car and read-only against shared state, so the compute half fans
        // across cores; the tiny write-back stays serial. `None` = untouched
        // (parked) row.
        let n = self.fleet.rows.len();
        let (backend, threshold) = (self.active_backend(), self.par_threshold.max(LIGHT_PAR_THRESHOLD));
        let updates: Vec<Option<([f64; 3], f64, bool)>> = map_collect(backend, threshold, n, |i| self.track_one(i, dt));
        for (i, u) in updates.into_iter().enumerate() {
            if let Some((kin, steer, diverged)) = u {
                self.fleet.rows[i].kin = kin;
                self.fleet.rows[i].steer = steer;
                if diverged {
                    self.divergences += 1;
                }
            }
        }
    }

    /// One vehicle's tracker step: the new kinematic pose, wheel position, and
    /// whether the divergence guard fired — or `None` for a parked car.
    pub(super) fn track_one(&self, i: usize, dt: f64) -> Option<([f64; 3], f64, bool)> {
        {
            let v = &self.fleet.rows[i];
            if v.speed <= 0.05 {
                return None;
            }
            let len = v.driver.vehicle_length;
            let [x, y, th] = v.kin;
            // Lookahead measured from the *rear axle* — the point doing the
            // chasing. The arc reference is stationed at the front bumper, so the
            // path offset subtracts the axle offset; at corner speeds the target
            // sits barely past the bumper and the car tracks the turn tightly
            // instead of lazily cutting toward a point a car-length beyond it.
            let ld = (0.8 * v.speed).clamp(LOOKAHEAD_MIN, LOOKAHEAD_MAX);
            let r = self.reference_point(v, ld - AXLE_FRONT * len);
            let (dx, dy) = (r[0] - x, r[1] - y);
            let dist = dx.hypot(dy);
            let alpha = shortest_angle(th, dy.atan2(dx));
            // Steering authority, in three nested bounds: what the steering
            // geometry can do at all (tan 35° over the wheelbase — the 10.5–12 m
            // curb-to-curb circle of a real car), what a driver is *willing* to
            // pull laterally at this speed (the AASHTO-style comfort envelope:
            // ~0.4 g threading an intersection, ~0.12 g at highway speed), and —
            // only during a recovery crank, where comfort is beside the point —
            // raw tyre grip.
            let kappa_geo = AXLE_STEER_TAN / (AXLE_WHEELBASE * len);
            let v2 = v.speed * v.speed;
            let kappa_max = kappa_geo.min(comfort_lat(v.speed) / v2);
            // Past the shoulder line pure pursuit's sin(α) goes blind (a
            // displaced car would roll straight on, diverging forever; a held
            // car's arc once drove 874 m away). A real driver cranks the wheel:
            // full lock toward the shorter side until the target is ahead again —
            // a bounded, steering-legal loop of a few seconds at worst, confined
            // to the handful of degenerate-mouth handoffs.
            let mut kappa_cmd = if alpha.abs() > std::f64::consts::FRAC_PI_2 {
                kappa_geo.min(YAW_GRIP_LAT / v2) * alpha.signum()
            } else if dist > 0.5 {
                2.0 * alpha.sin() / dist
            } else {
                0.0
            };
            // Feed-forward through the node: steer with the interior's own
            // curvature and let pursuit correct the residual, instead of
            // discovering a tight turn only through accumulating error.
            if let Some(c) = v.crossing {
                kappa_cmd += self.network.interior_curvature(c.movement, self.crossing_arc(v));
            }
            let clamp_hi = if alpha.abs() > std::f64::consts::FRAC_PI_2 { kappa_geo.min(YAW_GRIP_LAT / v2) } else { kappa_max };
            // The wheel winds, it doesn't snap: curvature approaches its command
            // at the steering rate, so lock builds over ~a second — the visible
            // ease-in/ease-out of a real turn.
            let target = kappa_cmd.clamp(-clamp_hi, clamp_hi);
            let dk = (STEER_RATE / (AXLE_WHEELBASE * len)) * dt;
            let kappa = v.steer + (target - v.steer).clamp(-dk, dk);
            // The wheels only roll *forward*, and rolling doubles as the
            // longitudinal sync: ground distance this tick is whatever keeps the
            // front bumper level with the arc pose along the car's axis, floored
            // at zero and capped just above the speed's own step. An arc hold
            // (stop-line clamp) rolls zero and the car simply stops; a backward
            // arc jump (a degenerate-mouth landing the skip rebase can't express)
            // becomes a short pause while the arc catches up — never a rendered
            // backslide; a forward jump is a brief bounded catch-up. Heading
            // advances by κ·ds — rolling geometry: a car that isn't moving cannot
            // rotate. The maneuver floor breaks the one fixed point that rule
            // has: a car pointed well off its line projects ~nothing forward and
            // would otherwise freeze mid-recovery — like a real driver, it keeps
            // rolling (at half pace) so steering can bring it back.
            let p = self.path_pose(v);
            let (c0, s0) = (th.cos(), th.sin());
            let (ex0, ey0) = (p[0] - (x + AXLE_FRONT * len * c0), p[1] - (y + AXLE_FRONT * len * s0));
            let long = ex0 * c0 + ey0 * s0;
            let lat0 = -ex0 * s0 + ey0 * c0;
            let mut ds = long.clamp(0.0, v.speed * dt * 1.5 + 0.3);
            // Ahead-ness measured in the *path's* frame: how far the drawn front
            // sits past the sim's station along the road. The maneuver floor may
            // only roll while the car is not ahead — without this guard a car
            // recovering a lateral error against a stalled arc (a queue, a box
            // admission gate) crept unboundedly into the intersection (32 m ahead
            // observed), parked there until its arc caught up, and its phantom
            // body triggered junction crashes the arc-based conflict logic never
            // scheduled.
            let ahead = -(ex0 * p[2].cos() + ey0 * p[2].sin());
            if lat0.abs() > 1.0 && ahead < 0.5 {
                ds = ds.max(0.5 * v.speed * dt);
            }
            // Hard cap in the path frame: rolling may never take the drawn front
            // more than half a metre past the sim's station (mid-turn, the
            // car-frame projection alone can't see this). A car that got ahead
            // rolls nothing until the arc catches up.
            ds = ds.min((v.speed * dt + 0.5 - ahead).max(0.0));
            let mut th2 = th + kappa * ds;
            if th2 > std::f64::consts::PI {
                th2 -= std::f64::consts::TAU;
            } else if th2 <= -std::f64::consts::PI {
                th2 += std::f64::consts::TAU;
            }
            let (c, s) = (th2.cos(), th2.sin());
            let mut x2 = x + ds * c;
            let mut y2 = y + ds * s;
            // Lateral residual against the arc pose: resolved by steering alone
            // unless it exceeds the divergence guard, which bleeds it at a
            // bounded rate and counts the activation.
            let (ex, ey) = (p[0] - (x2 + AXLE_FRONT * len * c), p[1] - (y2 + AXLE_FRONT * len * s));
            let lat = -ex * s + ey * c;
            let diverged = lat.abs() > LAT_DIVERGE_MAX;
            if diverged {
                // Capped to a fraction of the rolled distance so even the guard
                // cannot make the car slide at more than ~14° to its heading —
                // steering and the maneuver floor do the real recovery work.
                let lat_fix = (lat.abs() - LAT_DIVERGE_MAX).min(LAT_CORRECT_RATE * dt).min(0.25 * ds) * lat.signum();
                x2 -= lat_fix * s;
                y2 += lat_fix * c;
            }
            Some(([x2, y2, th2], kappa, diverged))
        }
    }

    /// How many tick-vehicle lateral divergence-guard activations have occurred
    /// (see [`track_path`](Self::track_path)) — a health metric: clusters mark
    /// geometry the tracker cannot physically follow.
    pub fn kinematic_divergences(&self) -> u64 {
        self.divergences
    }
}
