//! Rail: schedule-driven trains outside the road graph.
//!
//! Trains are not [`super::net_world::NetVehicle`]s: their position is a pure
//! function of day-clock seconds — a kinematic accelerate–cruise–brake profile
//! fitted to the timetable gap between stations, holding at stations until the
//! scheduled departure (a train never leaves early; timetable slack is absorbed
//! as a lower cruise speed). They touch road traffic only through level-crossing
//! closures, which are precomputed intervals from the same profiles. Rail
//! geometry ([`RailNetwork`]) rides on [`super::network::Network`] so every
//! render backend sees it; the schedule ([`Timetable`]) lives in the world.

/// One continuous physical track: a polyline with cumulative chainage, plus
/// per-segment speed-limit and grade-layer breakpoints (a bridge changes layer
/// without breaking chainage). Segment `i` spans `pts[i]..pts[i+1]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RailLine {
    /// OSM `railway` kind: `rail`, `light_rail`, `tram`, `subway`, …
    pub kind: String,
    pub name: String,
    pub pts: Vec<[f64; 2]>,
    /// Cumulative arc length per point; `cum[0] == 0`, `cum.last()` = line length.
    pub cum: Vec<f64>,
    /// `(first_segment, m/s)` speed-limit breakpoints, ascending.
    pub speeds: Vec<(usize, f64)>,
    /// `(first_segment, layer)` grade-layer breakpoints, ascending.
    pub layers: Vec<(usize, i32)>,
    /// `[min_x, min_y, max_x, max_y]` over `pts` — the cheap reject for
    /// snapping queries (a city artifact snaps thousands of trips against
    /// dozens of lines; without this every pair pays a full polyline scan).
    pub bbox: [f64; 4],
}

impl RailLine {
    pub fn new(kind: String, name: String, pts: Vec<[f64; 2]>, speeds: Vec<(usize, f64)>, layers: Vec<(usize, i32)>) -> Self {
        let mut cum = Vec::with_capacity(pts.len());
        let mut acc = 0.0;
        let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                acc += (p[0] - pts[i - 1][0]).hypot(p[1] - pts[i - 1][1]);
            }
            cum.push(acc);
            bbox = [bbox[0].min(p[0]), bbox[1].min(p[1]), bbox[2].max(p[0]), bbox[3].max(p[1])];
        }
        Self { kind, name, pts, cum, speeds, layers, bbox }
    }

    /// Whether `p` lies outside the line's bounding box expanded by `tol` —
    /// i.e. certainly farther than `tol` from every point of the line.
    pub fn beyond(&self, p: [f64; 2], tol: f64) -> bool {
        p[0] < self.bbox[0] - tol || p[0] > self.bbox[2] + tol || p[1] < self.bbox[1] - tol || p[1] > self.bbox[3] + tol
    }

    pub fn length(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    /// Segment index containing chainage `s` (clamped).
    fn seg_at(&self, s: f64) -> usize {
        match self.cum.binary_search_by(|c| c.total_cmp(&s)) {
            Ok(i) => i.min(self.pts.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(self.pts.len().saturating_sub(2)),
        }
    }

    /// World `[x, y, heading]` at chainage `s` (heading faces increasing chainage).
    pub fn pose_at(&self, s: f64) -> [f64; 3] {
        if self.pts.len() < 2 {
            let p = self.pts.first().copied().unwrap_or([0.0, 0.0]);
            return [p[0], p[1], 0.0];
        }
        let s = s.clamp(0.0, self.length());
        let i = self.seg_at(s);
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        let len = (self.cum[i + 1] - self.cum[i]).max(1e-9);
        let t = (s - self.cum[i]) / len;
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, (b[1] - a[1]).atan2(b[0] - a[0])]
    }

    /// Posted speed at chainage `s`.
    pub fn speed_at(&self, s: f64) -> f64 {
        let seg = self.seg_at(s);
        let mut v = self.speeds.first().map_or(25.0, |&(_, v)| v);
        for &(i, sv) in &self.speeds {
            if i <= seg {
                v = sv;
            }
        }
        v
    }

    /// Minimum posted speed over the chainage span `[lo, hi]` — the conservative
    /// cruise ceiling for a run crossing several speed zones.
    pub fn min_speed_over(&self, lo: f64, hi: f64) -> f64 {
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        let (s0, s1) = (self.seg_at(lo), self.seg_at(hi));
        let mut v = f64::INFINITY;
        let mut cur = self.speeds.first().map_or(25.0, |&(_, v)| v);
        let mut k = 0;
        for seg in 0..=s1 {
            while k < self.speeds.len() && self.speeds[k].0 <= seg {
                cur = self.speeds[k].1;
                k += 1;
            }
            if seg >= s0 {
                v = v.min(cur);
            }
        }
        if v.is_finite() { v } else { cur }
    }

    /// Grade layer at chainage `s` (for render banding).
    pub fn layer_at(&self, s: f64) -> i32 {
        let seg = self.seg_at(s);
        let mut l = self.layers.first().map_or(0, |&(_, l)| l);
        for &(i, sl) in &self.layers {
            if i <= seg {
                l = sl;
            }
        }
        l
    }

    /// Contiguous point-index runs of constant grade layer, `(layer, lo..=hi)`
    /// point indices — a run is renderable as one band ribbon.
    pub fn layer_runs(&self) -> Vec<(i32, std::ops::RangeInclusive<usize>)> {
        if self.pts.len() < 2 {
            return Vec::new();
        }
        let mut runs = Vec::new();
        let mut start = 0usize;
        let mut cur = self.layers.first().map_or(0, |&(_, l)| l);
        let mut k = 0usize;
        for seg in 0..self.pts.len() - 1 {
            let mut l = cur;
            while k < self.layers.len() && self.layers[k].0 <= seg {
                l = self.layers[k].1;
                k += 1;
            }
            if l != cur {
                if seg > start {
                    runs.push((cur, start..=seg));
                }
                start = seg;
                cur = l;
            }
        }
        runs.push((cur, start..=self.pts.len() - 1));
        runs
    }

    /// The polyline slice between chainages `a` and `b` (either order),
    /// endpoints interpolated — reversed when `b < a` so the result runs a→b.
    pub fn sub_polyline(&self, a: f64, b: f64) -> Vec<[f64; 2]> {
        let (lo, hi) = (a.min(b).clamp(0.0, self.length()), a.max(b).clamp(0.0, self.length()));
        let pa = self.pose_at(lo);
        let pb = self.pose_at(hi);
        let mut out = vec![[pa[0], pa[1]]];
        for i in 0..self.pts.len() {
            if self.cum[i] > lo && self.cum[i] < hi {
                out.push(self.pts[i]);
            }
        }
        out.push([pb[0], pb[1]]);
        if b < a {
            out.reverse();
        }
        out
    }

    /// Nearest point on the line to `p`: `(chainage, distance)`.
    pub fn nearest(&self, p: [f64; 2]) -> (f64, f64) {
        let mut best = (0.0, f64::INFINITY);
        for i in 0..self.pts.len().saturating_sub(1) {
            let (a, b) = (self.pts[i], self.pts[i + 1]);
            let seg = [b[0] - a[0], b[1] - a[1]];
            let len2 = (seg[0] * seg[0] + seg[1] * seg[1]).max(1e-9);
            let t = (((p[0] - a[0]) * seg[0] + (p[1] - a[1]) * seg[1]) / len2).clamp(0.0, 1.0);
            let q = [a[0] + seg[0] * t, a[1] + seg[1] * t];
            let d = (q[0] - p[0]).hypot(q[1] - p[1]);
            if d < best.1 {
                best = (self.cum[i] + len2.sqrt() * t, d);
            }
        }
        best
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RailStation {
    pub pos: [f64; 2],
    pub name: String,
    /// `station` / `halt` / `tram_stop`.
    pub kind: String,
}

/// Static rail geometry, carried on the road [`super::network::Network`] so the
/// renderers (GPU / raster / ASCII) all see it through the one `&Network` they
/// already take. Empty on hand-built maps.
#[derive(Clone, Debug, Default)]
pub struct RailNetwork {
    pub lines: Vec<RailLine>,
    pub stations: Vec<RailStation>,
    /// Platform outlines (closed ring when first == last point, else an edge).
    pub platforms: Vec<Vec<[f64; 2]>>,
}

impl RailNetwork {
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Nearest line to a point: `(line index, chainage, distance)`.
    pub fn nearest_line(&self, p: [f64; 2]) -> Option<(usize, f64, f64)> {
        self.lines
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let (ch, d) = l.nearest(p);
                (i, ch, d)
            })
            .min_by(|a, b| a.2.total_cmp(&b.2))
    }
}

// --- train dynamics ---------------------------------------------------------

/// Traction class: initial acceleration `a0` until the power limit binds
/// (`a(v) = min(a0, pom / v)`, `pom` = power over mass in W/kg), constant
/// service braking `b`. Figures from TCRP 13 / operator specs (see
/// PLAN_TRANSIT.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrainClass {
    /// Modern EMU (Caltrain KISS): 1.0 m/s² start, power-limited above ~57 km/h.
    Emu,
    /// Metro (BART-like): sharper launch, tapers early.
    Metro,
    /// Light rail / tram.
    LightRail,
    /// Diesel locomotive-hauled commuter.
    DieselCommuter,
    Freight,
}

impl TrainClass {
    /// `(a0 m/s², service brake m/s², power/mass W/kg, class top speed m/s)`.
    pub fn dynamics(self) -> (f64, f64, f64, f64) {
        match self {
            TrainClass::Emu => (1.0, 0.8, 16.0, 49.0),
            TrainClass::Metro => (1.34, 1.0, 18.0, 36.0),
            TrainClass::LightRail => (1.3, 1.3, 25.0, 29.0),
            TrainClass::DieselCommuter => (0.4, 0.8, 6.0, 35.5),
            TrainClass::Freight => (0.05, 0.3, 1.5, 27.0),
        }
    }

    /// Length of one carriage (m) for the chained render.
    pub fn car_length(self) -> f64 {
        match self {
            TrainClass::Emu => 25.0,
            TrainClass::Metro => 21.0,
            TrainClass::LightRail => 27.0,
            TrainClass::DieselCommuter => 26.0,
            TrainClass::Freight => 18.0,
        }
    }

    pub fn width(self) -> f64 {
        match self {
            TrainClass::LightRail => 2.65,
            _ => 3.0,
        }
    }

    /// Minimum station dwell (s) when running late — doors still have to cycle.
    pub fn min_dwell(self) -> f64 {
        match self {
            TrainClass::LightRail => 15.0,
            TrainClass::Freight => 0.0,
            _ => 20.0,
        }
    }

    pub fn from_str(s: &str) -> TrainClass {
        match s {
            "metro" | "subway" => TrainClass::Metro,
            "light_rail" | "lrt" | "tram" => TrainClass::LightRail,
            "diesel" | "diesel_commuter" => TrainClass::DieselCommuter,
            "freight" => TrainClass::Freight,
            _ => TrainClass::Emu,
        }
    }
}

/// Time to accelerate from rest to `v` (constant `a0`, then power-limited).
fn accel_time(a0: f64, pom: f64, v: f64) -> f64 {
    let vp = pom / a0;
    if v <= vp {
        v / a0
    } else {
        vp / a0 + (v * v - vp * vp) / (2.0 * pom)
    }
}

/// Distance covered accelerating from rest to `v`.
fn accel_dist(a0: f64, pom: f64, v: f64) -> f64 {
    let vp = pom / a0;
    if v <= vp {
        v * v / (2.0 * a0)
    } else {
        vp * vp / (2.0 * a0) + (v * v * v - vp * vp * vp) / (3.0 * pom)
    }
}

/// Distance covered `t` seconds into the acceleration phase.
fn accel_dist_at(a0: f64, pom: f64, t: f64) -> f64 {
    let vp = pom / a0;
    let tp = vp / a0;
    if t <= tp {
        0.5 * a0 * t * t
    } else {
        let v = (vp * vp + 2.0 * pom * (t - tp)).sqrt();
        vp * vp / (2.0 * a0) + (v * v * v - vp * vp * vp) / (3.0 * pom)
    }
}

/// One inter-station run in route-distance space, with its fitted profile:
/// accelerate (`t_acc`, `d_acc` — constant `a0` then power-limited), cruise at
/// `v_cruise`, brake (`t_brk`, `d_brk`). Cruise speed was solved so the run
/// time equals the (feasibility-adjusted) schedule gap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RunSegment {
    /// Actual departure / arrival in day-seconds (schedule-feasibility adjusted).
    depart: f64,
    arrive: f64,
    /// Route distance at the segment start; the run covers `dist` metres.
    s0: f64,
    dist: f64,
    v_cruise: f64,
    t_acc: f64,
    d_acc: f64,
    t_brk: f64,
    /// Class accel constants, for the in-phase position curve.
    a0: f64,
    pom: f64,
}

impl RunSegment {
    /// Route distance into the run at `tau` seconds after departure.
    fn pos(&self, tau: f64) -> f64 {
        let total = self.arrive - self.depart;
        let tau = tau.clamp(0.0, total);
        let t_cruise = (total - self.t_acc - self.t_brk).max(0.0);
        let x = if tau <= self.t_acc {
            accel_dist_at(self.a0, self.pom, tau).min(self.d_acc)
        } else if tau <= self.t_acc + t_cruise {
            self.d_acc + self.v_cruise * (tau - self.t_acc)
        } else {
            let tb = tau - self.t_acc - t_cruise;
            let d_cruise = self.v_cruise * t_cruise;
            let b = if self.t_brk > 1e-9 { self.v_cruise / self.t_brk } else { 0.0 };
            self.d_acc + d_cruise + self.v_cruise * tb - 0.5 * b * tb * tb
        };
        x.clamp(0.0, self.dist)
    }
}

/// A scheduled stop on a trip, in route-distance space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripStop {
    /// Route distance from the trip's first stop (monotone increasing).
    pub s: f64,
    /// Scheduled arrival / departure, day-seconds (may exceed 86400 for
    /// past-midnight trips).
    pub arrival: f64,
    pub departure: f64,
}

/// One scheduled train run. Geometry comes either from one [`RailLine`]
/// (`line`/`start_chainage`/`dir` — the common corridor case) or, when a route
/// spans several scraped track fragments (Muni branch → portal → subway), from
/// the trip's own stitched `path` polyline in route-distance space.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainTrip {
    pub line: usize,
    pub class: TrainClass,
    /// Runs on weekend service days (else weekday).
    pub weekend: bool,
    pub carriages: u32,
    /// Chainage on the line at route distance 0, and the travel direction sign
    /// (`+1` = increasing chainage). Unused when `path` is set.
    pub start_chainage: f64,
    pub dir: f64,
    /// Trip-owned geometry for multi-fragment routes: chainage on it *is*
    /// route distance.
    pub path: Option<RailLine>,
    pub stops: Vec<TripStop>,
    /// Fitted run profiles, one per consecutive stop pair.
    segs: Vec<RunSegment>,
}

/// Minimum feasible run time over `dist` with cruise ceiling `vmax`, and the
/// peak speed actually reached (short hops never reach `vmax`).
fn min_run_time(a0: f64, b: f64, pom: f64, vmax: f64, dist: f64) -> (f64, f64) {
    // Peak speed where accel + brake distance alone fill the hop.
    let fits = |v: f64| accel_dist(a0, pom, v) + v * v / (2.0 * b) <= dist;
    let v_pk = if fits(vmax) {
        vmax
    } else {
        let (mut lo, mut hi) = (0.0, vmax);
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let (da, db) = (accel_dist(a0, pom, v_pk), v_pk * v_pk / (2.0 * b));
    let cruise = (dist - da - db).max(0.0) / v_pk.max(1e-9);
    (accel_time(a0, pom, v_pk) + v_pk / b + cruise, v_pk)
}

/// Run time over `dist` cruising at `v` (assumes the profile fits, `v <= v_pk`).
fn run_time_at(a0: f64, b: f64, pom: f64, v: f64, dist: f64) -> f64 {
    let (da, db) = (accel_dist(a0, pom, v), v * v / (2.0 * b));
    accel_time(a0, pom, v) + v / b + (dist - da - db).max(0.0) / v.max(1e-9)
}

impl TrainTrip {
    /// Fit the run profiles: per hop, if the scheduled gap exceeds the minimum
    /// feasible run time the cruise speed is lowered to consume the slack (the
    /// train arrives exactly on time); an infeasible gap runs at full
    /// performance and carries the lateness forward. A train never departs a
    /// stop before its scheduled departure; when late it dwells only the
    /// class's minimum.
    pub fn build(
        line: usize,
        class: TrainClass,
        weekend: bool,
        carriages: u32,
        start_chainage: f64,
        dir: f64,
        stops: Vec<TripStop>,
        vmax_of_hop: impl Fn(usize) -> f64,
    ) -> TrainTrip {
        let (a0, b, pom, class_vmax) = class.dynamics();
        let mut segs = Vec::with_capacity(stops.len().saturating_sub(1));
        let mut ready = f64::NEG_INFINITY; // earliest possible departure from the current stop
        for i in 0..stops.len().saturating_sub(1) {
            let (from, to) = (stops[i], stops[i + 1]);
            let depart = from.departure.max(ready);
            let dist = (to.s - from.s).max(0.1);
            let vmax = vmax_of_hop(i).min(class_vmax).max(2.0);
            let (t_min, v_pk) = min_run_time(a0, b, pom, vmax, dist);
            let gap = to.arrival - depart;
            let (arrive, v_cruise) = if gap > t_min + 1e-9 {
                // Absorb slack: solve run_time(v) == gap, monotone decreasing in v.
                let (mut lo, mut hi) = (0.5, v_pk);
                for _ in 0..48 {
                    let mid = 0.5 * (lo + hi);
                    if run_time_at(a0, b, pom, mid, dist) > gap {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                (to.arrival, hi)
            } else {
                (depart + t_min, v_pk)
            };
            segs.push(RunSegment {
                depart,
                arrive,
                s0: from.s,
                dist,
                v_cruise,
                t_acc: accel_time(a0, pom, v_cruise),
                d_acc: accel_dist(a0, pom, v_cruise),
                t_brk: v_cruise / b,
                a0,
                pom,
            });
            ready = arrive + class.min_dwell();
        }
        TrainTrip { line, class, weekend, carriages, start_chainage, dir, path: None, stops, segs }
    }

    /// Total route distance (the last stop's `s`).
    pub fn total(&self) -> f64 {
        self.stops.last().map_or(0.0, |s| s.s)
    }

    /// World pose at route distance `s`, facing travel.
    pub fn pose(&self, rail: &RailNetwork, s: f64) -> [f64; 3] {
        match &self.path {
            Some(p) => p.pose_at(s),
            None => {
                let Some(line) = rail.lines.get(self.line) else { return [0.0, 0.0, 0.0] };
                let mut pose = line.pose_at(self.chainage(s));
                if self.dir < 0.0 {
                    pose[2] += std::f64::consts::PI;
                }
                pose
            }
        }
    }

    /// Project a world point onto the trip's geometry: `(route distance,
    /// distance off the track)`.
    pub fn project(&self, rail: &RailNetwork, p: [f64; 2]) -> (f64, f64) {
        match &self.path {
            Some(path) => path.nearest(p),
            None => {
                let Some(line) = rail.lines.get(self.line) else { return (0.0, f64::INFINITY) };
                let (ch, d) = line.nearest(p);
                ((ch - self.start_chainage) * self.dir, d)
            }
        }
    }

    /// The trip's active window in day-seconds: dwelling at the first stop from
    /// its scheduled arrival through the last (possibly late) arrival.
    pub fn window(&self) -> (f64, f64) {
        let start = self.stops.first().map_or(0.0, |s| s.arrival);
        let end = self.segs.last().map_or(start, |s| s.arrive);
        (start, end)
    }

    /// Route distance at day-time `t` (monotone; flat during dwells), or `None`
    /// outside the active window.
    pub fn route_pos(&self, t: f64) -> Option<f64> {
        let (start, end) = self.window();
        if t < start || t > end {
            return None;
        }
        // Last segment departing at or before t; before the first departure the
        // train dwells at its first stop.
        let mut pos = 0.0;
        for seg in &self.segs {
            if t < seg.depart {
                break;
            }
            pos = if t <= seg.arrive { seg.s0 + seg.pos(t - seg.depart) } else { seg.s0 + seg.dist };
        }
        Some(pos)
    }

    /// First time the (monotone) route position reaches `s`, or `None` when the
    /// trip never gets there.
    pub fn time_at_pos(&self, s: f64) -> Option<f64> {
        for seg in &self.segs {
            if s <= seg.s0 + 1e-9 {
                return Some(seg.depart);
            }
            if s <= seg.s0 + seg.dist + 1e-9 {
                let (mut lo, mut hi) = (seg.depart, seg.arrive);
                for _ in 0..48 {
                    let mid = 0.5 * (lo + hi);
                    if seg.s0 + seg.pos(mid - seg.depart) < s {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return Some(hi);
            }
        }
        None
    }

    /// Chainage on the line at route distance `s`.
    pub fn chainage(&self, s: f64) -> f64 {
        self.start_chainage + self.dir * s
    }

    pub fn length_m(&self) -> f64 {
        self.carriages as f64 * self.class.car_length()
    }

    /// Actual (feasibility-adjusted) arrival day-seconds at each stop.
    pub fn actual_arrivals(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.stops.len());
        out.push(self.stops.first().map_or(0.0, |s| s.arrival));
        for seg in &self.segs {
            out.push(seg.arrive);
        }
        out
    }
}

/// A live train at one instant: everything the renderer needs to draw the
/// carriage chain — carriage `k`'s centre sits at route distance
/// `route_pos - (k + 0.5) * car_length`, posed via [`TrainTrip::pose`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainState {
    pub trip: usize,
    pub class: TrainClass,
    pub carriages: u32,
    /// Route distance of the train's *front*.
    pub route_pos: f64,
}

/// Every scheduled trip; positions are pure functions of day-seconds, so this
/// is immutable after build.
#[derive(Clone, Debug, Default)]
pub struct Timetable {
    pub trips: Vec<TrainTrip>,
}

impl Timetable {
    pub fn is_empty(&self) -> bool {
        self.trips.is_empty()
    }

    /// All trains active at `day_secs` on the given service day. Checks
    /// `t + 86400` too, so a past-midnight trip (dep 25:35) is found from the
    /// small hours of the following civil day.
    pub fn active_trains(&self, day_secs: f64, weekend: bool) -> Vec<TrainState> {
        let mut out = Vec::new();
        for (i, trip) in self.trips.iter().enumerate() {
            if trip.weekend != weekend {
                continue;
            }
            for t in [day_secs, day_secs + 86400.0] {
                if let Some(s) = trip.route_pos(t) {
                    out.push(TrainState { trip: i, class: trip.class, carriages: trip.carriages, route_pos: s });
                    break;
                }
            }
        }
        out
    }
}

// --- level-crossing closures ------------------------------------------------

/// Lights + bells activate this long before the train reaches the crossing
/// (49 CFR 234.225 floor is 20 s; constant-warning-time detectors target
/// 20–25 s).
pub const CROSSING_LEAD_SECS: f64 = 25.0;
/// Gates lift this long after the train's tail clears.
pub const CROSSING_CLEAR_SECS: f64 = 5.0;

/// Precomputed closed intervals (day-seconds) for one crossing, one service
/// day; sorted and merged.
#[derive(Clone, Debug, Default)]
pub struct ClosureSet {
    pub weekday: Vec<(f64, f64)>,
    pub weekend: Vec<(f64, f64)>,
}

impl ClosureSet {
    /// Fold another crossing's intervals in (parallel tracks at one crossing),
    /// keeping both day sets sorted and merged.
    pub fn union(&mut self, other: ClosureSet) {
        for (mine, theirs) in [(&mut self.weekday, other.weekday), (&mut self.weekend, other.weekend)] {
            mine.extend(theirs);
            mine.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut merged: Vec<(f64, f64)> = Vec::with_capacity(mine.len());
            for &iv in mine.iter() {
                match merged.last_mut() {
                    Some(last) if iv.0 <= last.1 => last.1 = last.1.max(iv.1),
                    _ => merged.push(iv),
                }
            }
            *mine = merged;
        }
    }

    pub fn closed_at(&self, day_secs: f64, weekend: bool) -> bool {
        let set = if weekend { &self.weekend } else { &self.weekday };
        for t in [day_secs, day_secs + 86400.0] {
            let i = set.partition_point(|iv| iv.1 < t);
            if i < set.len() && set[i].0 <= t {
                return true;
            }
        }
        false
    }
}

/// How close a trip's geometry must pass a crossing to close it.
const CROSSING_HIT_TOL: f64 = 30.0;

/// Closure intervals for a crossing at world position `pos` from every trip
/// whose geometry passes over it: `[t_front − lead, t_tail + clear]` per
/// passage, merged. Projection is per trip, so parallel tracks and stitched
/// multi-fragment paths all register.
pub fn crossing_closures(tt: &Timetable, rail: &RailNetwork, pos: [f64; 2]) -> ClosureSet {
    let mut weekday = Vec::new();
    let mut weekend = Vec::new();
    for trip in &tt.trips {
        let (s_c, d) = trip.project(rail, pos);
        let total = trip.total();
        if d > CROSSING_HIT_TOL || s_c < -1e-6 || s_c > total + 1e-6 {
            continue;
        }
        let (Some(t_front), Some(t_tail)) = (
            trip.time_at_pos(s_c.max(0.0)),
            trip.time_at_pos((s_c + trip.length_m()).min(total)),
        ) else {
            continue;
        };
        // A tail that never fully clears in-span (trip ends just past the
        // crossing) holds the gate to the trip's end — conservative and rare.
        let iv = (t_front - CROSSING_LEAD_SECS, t_tail + CROSSING_CLEAR_SECS);
        if trip.weekend { weekend.push(iv) } else { weekday.push(iv) };
    }
    for set in [&mut weekday, &mut weekend] {
        set.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(set.len());
        for &iv in set.iter() {
            match merged.last_mut() {
                Some(last) if iv.0 <= last.1 => last.1 = last.1.max(iv.1),
                _ => merged.push(iv),
            }
        }
        *set = merged;
    }
    ClosureSet { weekday, weekend }
}

// --- transit artifact (GTFS-compiled schedules) -----------------------------

/// One stop of a compiled trip: local-metre position plus scheduled
/// arrival/departure day-seconds. Bbox-edge pseudo-stops carry `arr == dep`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StopSpec {
    pub pos: [f64; 2],
    pub arrival: f64,
    pub departure: f64,
    /// GTFS `timepoint`: an exact schedule time — a bus holds here when early.
    pub timepoint: bool,
}

/// A compiled rail trip, before snapping onto a [`RailNetwork`] line.
#[derive(Clone, Debug, PartialEq)]
pub struct RailTripSpec {
    pub class: TrainClass,
    pub weekend: bool,
    pub carriages: u32,
    pub stops: Vec<StopSpec>,
}

/// A compiled bus trip: the engine matches it to a named transit line's link
/// chain (`demand::TransitLine`) by name and stop positions.
#[derive(Clone, Debug, PartialEq)]
pub struct BusTripSpec {
    pub line: String,
    pub weekend: bool,
    pub stops: Vec<StopSpec>,
}

/// GTFS stops sit on platforms beside the track; accept a snap up to this far
/// from the line's centreline.
const RAIL_SNAP_TOL: f64 = 60.0;

/// Snap compiled rail trips onto the rail geometry and fit their run profiles.
/// Soft-fail per trip (the A/B Street lesson): a trip that doesn't cleanly land
/// on one line is dropped and counted, never emitted broken. Returns the
/// timetable and the dropped-trip count.
pub fn build_timetable(rail: &RailNetwork, trips: &[RailTripSpec]) -> (Timetable, usize) {
    let mut out = Vec::new();
    let mut dropped = 0usize;
    for spec in trips {
        match snap_trip(rail, spec) {
            Some(t) => out.push(t),
            None => dropped += 1,
        }
    }
    (Timetable { trips: out }, dropped)
}

fn snap_trip(rail: &RailNetwork, spec: &RailTripSpec) -> Option<TrainTrip> {
    if spec.stops.len() < 2 {
        return None;
    }
    // Candidate line: minimize the worst stop-to-line distance; every stop must
    // land within tolerance on the same line (single-line trips only — the
    // compiler emits one spec per through-track). Bbox reject + early exit keep
    // this linear in practice: a city artifact carries thousands of trips.
    let mut best: Option<(usize, Vec<f64>, f64)> = None;
    for (li, line) in rail.lines.iter().enumerate() {
        if spec.stops.iter().any(|s| line.beyond(s.pos, RAIL_SNAP_TOL)) {
            continue;
        }
        let mut chain = Vec::with_capacity(spec.stops.len());
        let mut worst = 0.0f64;
        for s in &spec.stops {
            let (ch, d) = line.nearest(s.pos);
            worst = worst.max(d);
            if worst > RAIL_SNAP_TOL {
                break;
            }
            chain.push(ch);
        }
        if chain.len() == spec.stops.len() && worst <= RAIL_SNAP_TOL && best.as_ref().is_none_or(|b| worst < b.2) {
            best = Some((li, chain, worst));
        }
    }
    let Some((li, chain, _)) = best else {
        // No single line carries the whole route (a Muni branch → portal →
        // subway run over fragmented geometry): stitch the trip its own path.
        return snap_trip_multiline(rail, spec);
    };
    let line = &rail.lines[li];
    let dir = if *chain.last().unwrap() >= chain[0] { 1.0 } else { -1.0 };
    let start_chainage = chain[0];
    // Route distances must be strictly increasing; snapping noise near termini
    // can fold a stop backwards — skip those stops rather than the trip.
    let mut stops = Vec::with_capacity(spec.stops.len());
    let mut hops = Vec::new();
    let mut prev_s = f64::NEG_INFINITY;
    let mut prev_ch = start_chainage;
    for (k, st) in spec.stops.iter().enumerate() {
        let s = (chain[k] - start_chainage) * dir;
        if s <= prev_s + 1.0 && k > 0 {
            continue;
        }
        if k > 0 {
            hops.push((prev_ch, chain[k]));
        }
        stops.push(TripStop { s, arrival: st.arrival, departure: st.departure.max(st.arrival) });
        prev_s = s;
        prev_ch = chain[k];
    }
    if stops.len() < 2 {
        return None;
    }
    Some(TrainTrip::build(li, spec.class, spec.weekend, spec.carriages, start_chainage, dir, stops, |i| {
        let (a, b) = hops[i];
        line.min_speed_over(a, b)
    }))
}

/// Stitch a trip its own geometry, hop by hop: each consecutive stop pair
/// snaps to the line that carries *both* ends best (its sub-polyline joins the
/// path), an unmatchable pair bridges straight. The trip then runs in its own
/// route-distance space ([`TrainTrip::pose`] reads the path directly). Most
/// hops must land on real track, or the route just isn't in the geometry.
fn snap_trip_multiline(rail: &RailNetwork, spec: &RailTripSpec) -> Option<TrainTrip> {
    if spec.stops.len() < 2 || rail.lines.is_empty() {
        return None;
    }
    let (_, _, _, class_vmax) = spec.class.dynamics();
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut acc = 0.0f64;
    let push = |pts: &mut Vec<[f64; 2]>, acc: &mut f64, seg: Vec<[f64; 2]>| {
        for p in seg {
            match pts.last() {
                Some(last) => {
                    let d = (p[0] - last[0]).hypot(p[1] - last[1]);
                    if d > 0.01 {
                        *acc += d;
                        pts.push(p);
                    }
                }
                None => pts.push(p),
            }
        }
    };
    let mut stops = Vec::with_capacity(spec.stops.len());
    let first = &spec.stops[0];
    stops.push(TripStop { s: 0.0, arrival: first.arrival, departure: first.departure.max(first.arrival) });
    let (mut matched, mut hop_speeds) = (0usize, Vec::new());
    for pair in spec.stops.windows(2) {
        let (a, b) = (pair[0].pos, pair[1].pos);
        let mut best: Option<(usize, f64, f64, f64)> = None;
        for (li, line) in rail.lines.iter().enumerate() {
            if line.beyond(a, RAIL_SNAP_TOL) || line.beyond(b, RAIL_SNAP_TOL) {
                continue;
            }
            let (ca, da) = line.nearest(a);
            let (cb, db) = line.nearest(b);
            let worst = da.max(db);
            if worst <= RAIL_SNAP_TOL && (ca - cb).abs() > 1.0 && best.is_none_or(|x| worst < x.3) {
                best = Some((li, ca, cb, worst));
            }
        }
        let (seg, v) = match best {
            Some((li, ca, cb, _)) => {
                matched += 1;
                (rail.lines[li].sub_polyline(ca, cb), rail.lines[li].min_speed_over(ca, cb))
            }
            None => (vec![a, b], class_vmax),
        };
        hop_speeds.push(v);
        push(&mut pts, &mut acc, seg);
        let prev = stops.last().map_or(0.0, |s: &TripStop| s.s);
        stops.push(TripStop {
            s: acc.max(prev + 0.5),
            arrival: pair[1].arrival,
            departure: pair[1].departure.max(pair[1].arrival),
        });
    }
    if pts.len() < 2 || matched * 2 < hop_speeds.len() {
        return None;
    }
    let path = RailLine::new("path".into(), String::new(), pts, vec![(0, class_vmax)], vec![(0, 0)]);
    let mut trip = TrainTrip::build(0, spec.class, spec.weekend, spec.carriages, 0.0, 1.0, stops, |i| hop_speeds[i]);
    trip.path = Some(path);
    Some(trip)
}

// --- JSON import ------------------------------------------------------------

#[cfg(feature = "import")]
mod io {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct JsonRailLine {
        #[serde(default)]
        kind: String,
        #[serde(default)]
        name: String,
        pts: Vec<[f64; 2]>,
        #[serde(default)]
        speeds: Vec<(usize, f64)>,
        #[serde(default)]
        layers: Vec<(usize, i32)>,
    }

    #[derive(Deserialize)]
    struct JsonRailStation {
        x: f64,
        y: f64,
        #[serde(default)]
        kind: String,
        #[serde(default)]
        name: String,
    }

    #[derive(Deserialize)]
    struct JsonRailDoc {
        #[serde(default)]
        rail_lines: Vec<JsonRailLine>,
        #[serde(default)]
        rail_stations: Vec<JsonRailStation>,
        #[serde(default)]
        rail_platforms: Vec<Vec<[f64; 2]>>,
    }

    impl RailNetwork {
        /// Rail geometry from the map JSON's top-level `rail_lines` /
        /// `rail_stations` / `rail_platforms` (empty when absent — old extracts
        /// load unchanged).
        pub fn from_map_json(s: &str) -> RailNetwork {
            let doc: JsonRailDoc = match serde_json::from_str(s) {
                Ok(d) => d,
                Err(_) => return RailNetwork::default(),
            };
            RailNetwork {
                lines: doc
                    .rail_lines
                    .into_iter()
                    .filter(|l| l.pts.len() >= 2)
                    .map(|l| RailLine::new(l.kind, l.name, l.pts, l.speeds, l.layers))
                    .collect(),
                stations: doc
                    .rail_stations
                    .into_iter()
                    .map(|s| RailStation { pos: [s.x, s.y], name: s.name, kind: s.kind })
                    .collect(),
                platforms: doc.rail_platforms,
            }
        }
    }

    #[derive(Deserialize)]
    struct JsonStop {
        x: f64,
        y: f64,
        arr: f64,
        dep: f64,
        #[serde(default)]
        timepoint: bool,
    }

    #[derive(Deserialize)]
    struct JsonRailTrip {
        #[serde(default)]
        class: String,
        #[serde(default)]
        weekend: bool,
        #[serde(default = "default_carriages")]
        carriages: u32,
        stops: Vec<JsonStop>,
    }

    fn default_carriages() -> u32 {
        6
    }

    #[derive(Deserialize)]
    struct JsonBusTrip {
        line: String,
        #[serde(default)]
        weekend: bool,
        stops: Vec<JsonStop>,
    }

    #[derive(Deserialize)]
    struct JsonTransitDoc {
        #[serde(default)]
        rail_trips: Vec<JsonRailTrip>,
        #[serde(default)]
        bus_trips: Vec<JsonBusTrip>,
    }

    /// Parse a compiled transit artifact (`tools/gtfs`): rail trip specs to
    /// snap onto the rail network, and bus trip specs for the transit lines.
    pub fn transit_from_json(s: &str) -> Result<(Vec<RailTripSpec>, Vec<BusTripSpec>), String> {
        let doc: JsonTransitDoc = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let stops = |v: Vec<JsonStop>| {
            v.into_iter()
                .map(|s| StopSpec { pos: [s.x, s.y], arrival: s.arr, departure: s.dep, timepoint: s.timepoint })
                .collect::<Vec<_>>()
        };
        Ok((
            doc.rail_trips
                .into_iter()
                .map(|t| RailTripSpec {
                    class: TrainClass::from_str(&t.class),
                    weekend: t.weekend,
                    carriages: t.carriages.clamp(1, 32),
                    stops: stops(t.stops),
                })
                .collect(),
            doc.bus_trips
                .into_iter()
                .map(|t| BusTripSpec { line: t.line, weekend: t.weekend, stops: stops(t.stops) })
                .collect(),
        ))
    }
}

#[cfg(feature = "import")]
pub use io::transit_from_json;

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_line(len: f64, speed: f64) -> RailLine {
        RailLine::new(
            "rail".into(),
            "test".into(),
            vec![[0.0, 0.0], [len, 0.0]],
            vec![(0, speed)],
            vec![(0, 0)],
        )
    }

    #[test]
    fn accel_closed_forms_match_numeric_integration() {
        let (a0, pom): (f64, f64) = (1.0, 16.0);
        // Integrate dv/dt = min(a0, pom/v) numerically.
        let (mut v, mut x, mut t) = (0.0f64, 0.0f64, 0.0f64);
        let dt = 1e-4;
        let target = 30.0;
        while v < target {
            v += a0.min(pom / v.max(1e-9)) * dt;
            x += v * dt;
            t += dt;
        }
        assert!((accel_time(a0, pom, target) - t).abs() < 0.05, "t {t} vs {}", accel_time(a0, pom, target));
        assert!((accel_dist(a0, pom, target) - x).abs() < 2.0, "x {x} vs {}", accel_dist(a0, pom, target));
        // The time-parameterized curve agrees with the speed-parameterized one.
        let tt = accel_time(a0, pom, target);
        assert!((accel_dist_at(a0, pom, tt) - accel_dist(a0, pom, target)).abs() < 1e-6);
    }

    #[test]
    fn a_slack_schedule_arrives_exactly_on_time_and_never_departs_early() {
        // 2 km hop, 79 mph track: full performance needs ~80 s; schedule 180 s.
        let line = straight_line(2000.0, 35.3);
        let stops = vec![
            TripStop { s: 0.0, arrival: 1000.0, departure: 1030.0 },
            TripStop { s: 2000.0, arrival: 1210.0, departure: 1240.0 },
        ];
        let trip = TrainTrip::build(0, TrainClass::Emu, false, 7, 0.0, 1.0, stops, |_| line.min_speed_over(0.0, 2000.0));
        let arr = trip.actual_arrivals();
        assert!((arr[1] - 1210.0).abs() < 0.5, "on-time arrival, got {}", arr[1]);
        // Dwelling before departure: position stays 0 through the scheduled dwell.
        assert_eq!(trip.route_pos(1010.0), Some(0.0));
        assert_eq!(trip.route_pos(1029.9), Some(0.0));
        // Slack absorbed as a lower cruise speed, not an early arrival.
        let mid = trip.route_pos(1120.0).unwrap();
        assert!(mid > 100.0 && mid < 1900.0, "mid-run position {mid}");
        // Monotone.
        let mut last = 0.0;
        for k in 0..200 {
            let p = trip.route_pos(1030.0 + k as f64).unwrap_or(2000.0);
            assert!(p >= last - 1e-9);
            last = p;
        }
    }

    #[test]
    fn an_infeasible_schedule_runs_flat_out_and_carries_lateness() {
        // 3 km in a scheduled 30 s is impossible; the train runs at full
        // performance and the next hop starts late (after the minimum dwell).
        let stops = vec![
            TripStop { s: 0.0, arrival: 0.0, departure: 0.0 },
            TripStop { s: 3000.0, arrival: 30.0, departure: 40.0 },
            TripStop { s: 6000.0, arrival: 400.0, departure: 420.0 },
        ];
        let trip = TrainTrip::build(0, TrainClass::Emu, false, 7, 0.0, 1.0, stops, |_| 35.3);
        let arr = trip.actual_arrivals();
        let (a0, b, pom, vmax) = TrainClass::Emu.dynamics();
        let (t_min, _) = min_run_time(a0, b, pom, vmax.min(35.3), 3000.0);
        assert!((arr[1] - t_min).abs() < 1.0, "late arrival = min run time {t_min}, got {}", arr[1]);
        assert!(arr[1] > 60.0);
        // Second hop departs after the late arrival + minimum dwell, not at 40 s;
        // generous slack lets it recover to the scheduled arrival.
        assert!((arr[2] - 400.0).abs() < 0.5, "recovered by stop 3, got {}", arr[2]);
    }

    #[test]
    fn time_at_pos_inverts_route_pos() {
        let stops = vec![
            TripStop { s: 0.0, arrival: 0.0, departure: 10.0 },
            TripStop { s: 1500.0, arrival: 130.0, departure: 150.0 },
            TripStop { s: 2500.0, arrival: 260.0, departure: 280.0 },
        ];
        let trip = TrainTrip::build(0, TrainClass::Emu, false, 6, 0.0, 1.0, stops, |_| 30.0);
        for s in [1.0, 200.0, 749.0, 1500.0, 2000.0, 2499.0] {
            let t = trip.time_at_pos(s).unwrap();
            let p = trip.route_pos(t).unwrap();
            assert!((p - s).abs() < 1.0, "s={s} t={t} p={p}");
        }
    }

    #[test]
    fn crossing_closures_bracket_the_passage_with_lead_time() {
        let stops = vec![
            TripStop { s: 0.0, arrival: 0.0, departure: 0.0 },
            TripStop { s: 4000.0, arrival: 200.0, departure: 210.0 },
        ];
        let trip = TrainTrip::build(0, TrainClass::Emu, false, 7, 0.0, 1.0, stops, |_| 35.3);
        let t_pass = trip.time_at_pos(2000.0).unwrap();
        let tt = Timetable { trips: vec![trip] };
        let rail = RailNetwork { lines: vec![straight_line(4000.0, 35.3)], ..Default::default() };
        let set = crossing_closures(&tt, &rail, [2000.0, 0.0]);
        assert_eq!(set.weekday.len(), 1);
        let (lo, hi) = set.weekday[0];
        assert!((t_pass - lo - CROSSING_LEAD_SECS).abs() < 1.0, "lead: {} vs {}", t_pass - lo, CROSSING_LEAD_SECS);
        assert!(hi > t_pass, "closed until the tail clears");
        assert!(set.closed_at(t_pass, false));
        assert!(!set.closed_at(lo - 5.0, false));
        assert!(!set.closed_at(hi + 5.0, false));
        assert!(!set.closed_at(t_pass, true), "weekday trip closes nothing on weekends");
    }

    #[test]
    fn past_midnight_trips_close_crossings_in_the_small_hours() {
        // Dep 25:35 = 1:35 the next civil day.
        let stops = vec![
            TripStop { s: 0.0, arrival: 92_100.0, departure: 92_100.0 },
            TripStop { s: 4000.0, arrival: 92_300.0, departure: 92_310.0 },
        ];
        let trip = TrainTrip::build(0, TrainClass::Emu, false, 7, 0.0, 1.0, stops, |_| 35.3);
        let t_pass = trip.time_at_pos(2000.0).unwrap();
        let tt = Timetable { trips: vec![trip] };
        let rail = RailNetwork { lines: vec![straight_line(4000.0, 35.3)], ..Default::default() };
        let set = crossing_closures(&tt, &rail, [2000.0, 0.0]);
        // Queried with the wrapped civil clock (t mod 86400).
        assert!(set.closed_at(t_pass - 86_400.0, false));
        let trains = tt.active_trains(t_pass - 86_400.0, false);
        assert_eq!(trains.len(), 1);
    }

    #[test]
    fn line_geometry_pose_speed_and_layer_runs() {
        let line = RailLine::new(
            "rail".into(),
            "l".into(),
            vec![[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [100.0, 150.0]],
            vec![(0, 20.0), (1, 10.0)],
            vec![(0, 0), (1, 1), (2, 0)],
        );
        assert!((line.length() - 250.0).abs() < 1e-9);
        let p = line.pose_at(50.0);
        assert!((p[0] - 50.0).abs() < 1e-9 && p[1].abs() < 1e-9 && p[2].abs() < 1e-9);
        let p = line.pose_at(150.0);
        assert!((p[0] - 100.0).abs() < 1e-9 && (p[1] - 50.0).abs() < 1e-9);
        assert!((p[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert_eq!(line.speed_at(50.0), 20.0);
        assert_eq!(line.speed_at(150.0), 10.0);
        assert_eq!(line.min_speed_over(0.0, 250.0), 10.0);
        assert_eq!(line.min_speed_over(0.0, 90.0), 20.0);
        let runs = line.layer_runs();
        assert_eq!(runs, vec![(0, 0..=1), (1, 1..=2), (0, 2..=3)]);
        let (ch, d) = line.nearest([60.0, 5.0]);
        assert!((ch - 60.0).abs() < 1e-6 && (d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn snapping_matches_a_trip_to_the_right_line_and_direction() {
        let rail = RailNetwork {
            lines: vec![
                straight_line(5000.0, 35.0),
                RailLine::new(
                    "rail".into(),
                    "offset".into(),
                    vec![[0.0, 500.0], [5000.0, 500.0]],
                    vec![(0, 35.0)],
                    vec![(0, 0)],
                ),
            ],
            ..Default::default()
        };
        // Reverse-direction trip along the second (offset) line, stops ~20 m off.
        let spec = RailTripSpec {
            class: TrainClass::Emu,
            weekend: false,
            carriages: 7,
            stops: vec![
                StopSpec { pos: [4500.0, 520.0], arrival: 0.0, departure: 10.0, timepoint: true },
                StopSpec { pos: [2000.0, 520.0], arrival: 120.0, departure: 140.0, timepoint: true },
                StopSpec { pos: [500.0, 520.0], arrival: 260.0, departure: 260.0, timepoint: true },
            ],
        };
        let (tt, dropped) = build_timetable(&rail, &[spec]);
        assert_eq!(dropped, 0);
        let trip = &tt.trips[0];
        assert_eq!(trip.line, 1);
        assert_eq!(trip.dir, -1.0);
        // Mid-run the front sits inside the first hop; its world pose tracks
        // the line's decreasing-chainage direction.
        let trains = tt.active_trains(60.0, false);
        assert_eq!(trains.len(), 1);
        assert!(trains[0].route_pos > 0.0 && trains[0].route_pos < 2500.0);
        let pose = tt.trips[0].pose(&rail, trains[0].route_pos);
        assert!(pose[0] < 4500.0 && pose[0] > 2000.0, "world x between the stops: {}", pose[0]);
        assert!((pose[1] - 500.0).abs() < 1.0, "on the offset line");
        // A trip nowhere near any line is dropped, not emitted broken.
        let far = RailTripSpec {
            class: TrainClass::Emu,
            weekend: false,
            carriages: 7,
            stops: vec![
                StopSpec { pos: [0.0, 9000.0], arrival: 0.0, departure: 0.0, timepoint: true },
                StopSpec { pos: [1000.0, 9000.0], arrival: 60.0, departure: 60.0, timepoint: true },
            ],
        };
        let (tt2, dropped2) = build_timetable(&rail, &[far]);
        assert_eq!((tt2.trips.len(), dropped2), (0, 1));
    }

    #[test]
    fn a_multi_fragment_route_stitches_its_own_path() {
        // Two disjoint collinear track fragments with a 100 m gap (a portal the
        // scrape didn't bridge): no single line carries the trip, so it stitches
        // a path — motion, poses, and crossing closures all run in the trip's
        // own route space.
        let rail = RailNetwork {
            lines: vec![
                straight_line(2000.0, 30.0),
                RailLine::new(
                    "rail".into(),
                    "b".into(),
                    vec![[2100.0, 0.0], [5000.0, 0.0]],
                    vec![(0, 30.0)],
                    vec![(0, 0)],
                ),
            ],
            ..Default::default()
        };
        let spec = RailTripSpec {
            class: TrainClass::Emu,
            weekend: false,
            carriages: 4,
            stops: vec![
                StopSpec { pos: [100.0, 5.0], arrival: 0.0, departure: 10.0, timepoint: true },
                StopSpec { pos: [1900.0, 5.0], arrival: 90.0, departure: 100.0, timepoint: true },
                StopSpec { pos: [2900.0, 5.0], arrival: 200.0, departure: 210.0, timepoint: true },
                StopSpec { pos: [4800.0, 5.0], arrival: 320.0, departure: 320.0, timepoint: true },
            ],
        };
        let (tt, dropped) = build_timetable(&rail, &[spec]);
        assert_eq!(dropped, 0, "the multi-fragment trip snaps via its own path");
        let trip = &tt.trips[0];
        assert!(trip.path.is_some());
        // Route length ≈ world span (100 → 4800).
        assert!((trip.total() - 4700.0).abs() < 30.0, "total {}", trip.total());
        // Mid-second-hop the pose sits in the inter-fragment gap region and on the axis.
        let t = tt.active_trains(150.0, false);
        assert_eq!(t.len(), 1);
        let pose = trip.pose(&rail, t[0].route_pos);
        assert!(pose[0] > 1900.0 && pose[0] < 2900.0, "mid gap-hop x {}", pose[0]);
        assert!(pose[1].abs() < 6.0);
        // A crossing on fragment A closes around the passage, via projection.
        let t_pass = trip.time_at_pos(900.0).unwrap();
        let set = crossing_closures(&tt, &rail, [1000.0, 0.0]);
        assert!(set.closed_at(t_pass, false));
        assert!(!set.closed_at(t_pass + 600.0, false));
    }

    #[test]
    fn active_trains_filters_by_service_day_and_window() {
        let mk = |weekend: bool| {
            TrainTrip::build(
                0,
                TrainClass::Emu,
                weekend,
                7,
                0.0,
                1.0,
                vec![
                    TripStop { s: 0.0, arrival: 100.0, departure: 120.0 },
                    TripStop { s: 2000.0, arrival: 260.0, departure: 270.0 },
                ],
                |_| 35.0,
            )
        };
        let tt = Timetable { trips: vec![mk(false), mk(true)] };
        assert_eq!(tt.active_trains(150.0, false).len(), 1);
        assert_eq!(tt.active_trains(150.0, true).len(), 1);
        assert_eq!(tt.active_trains(50.0, false).len(), 0, "before the window");
        assert_eq!(tt.active_trains(500.0, false).len(), 0, "after the window");
    }

    /// Manual diagnostic against real artifacts:
    /// `MAP_JSON=... TRANSIT_JSON=... cargo test --features import -- --ignored diag_transit_artifact --nocapture`
    /// (plain JSON paths; gunzip the committed `.gz` extracts first).
    #[cfg(feature = "import")]
    #[test]
    #[ignore]
    fn diag_transit_artifact() {
        let map = std::fs::read_to_string(std::env::var("MAP_JSON").expect("MAP_JSON")).unwrap();
        let transit = std::fs::read_to_string(std::env::var("TRANSIT_JSON").expect("TRANSIT_JSON")).unwrap();
        let rail = RailNetwork::from_map_json(&map);
        println!("rail geometry: {} lines, {} stations, {} platforms", rail.lines.len(), rail.stations.len(), rail.platforms.len());
        for l in &rail.lines {
            println!("  {:<10} {:>7.2} km  {}", l.kind, l.length() / 1000.0, l.name);
        }
        let (rt, bt) = transit_from_json(&transit).unwrap();
        let (tt, dropped) = build_timetable(&rail, &rt);
        println!("rail trips: kept {} / dropped {} (of {})", tt.trips.len(), dropped, rt.len());
        let mut by_class: std::collections::BTreeMap<String, usize> = Default::default();
        for t in &tt.trips {
            *by_class.entry(format!("{:?}", t.class)).or_default() += 1;
        }
        println!("  kept by class: {by_class:?}");
        // Per-hour presence: trains are only in-box for the few minutes a trip
        // spans it, so sample every 20 s and report the hour's peak + total
        // train-minutes.
        for wknd in [false, true] {
            let mut line_out = String::new();
            for h in 5..23 {
                let (mut peak, mut mins) = (0usize, 0.0);
                for k in 0..180 {
                    let n = tt.active_trains(h as f64 * 3600.0 + k as f64 * 20.0, wknd).len();
                    peak = peak.max(n);
                    mins += n as f64 * 20.0 / 60.0;
                }
                line_out.push_str(&format!(" {h}h:{peak}/{mins:.0}m"));
            }
            println!("  {} peak/train-minutes:{}", if wknd { "sat" } else { "wed" }, line_out);
        }
        // Closure volume at a probe point mid-way along each line.
        for (li, l) in rail.lines.iter().enumerate() {
            let p = l.pose_at(l.length() * 0.5);
            let set = crossing_closures(&tt, &rail, [p[0], p[1]]);
            let total: f64 = set.weekday.iter().map(|(a, b)| b - a).sum();
            if !set.weekday.is_empty() {
                println!(
                    "  line {li} mid-point: {} weekday closures, {:.0} min gate-down/day",
                    set.weekday.len(),
                    total / 60.0
                );
            }
        }
        println!("bus trips parsed: {}", bt.len());
    }

    #[cfg(feature = "import")]
    #[test]
    fn map_and_transit_json_round_trip() {
        let map = r#"{"rail_lines":[{"kind":"rail","name":"Test Sub","pts":[[0,0],[1000,0]],"speeds":[[0,35.3]],"layers":[[0,0]]}],
                      "rail_stations":[{"x":10,"y":5,"kind":"station","name":"A"}],
                      "rail_platforms":[[[0,4],[50,4]]]}"#;
        let rail = RailNetwork::from_map_json(map);
        assert_eq!(rail.lines.len(), 1);
        assert_eq!(rail.stations[0].name, "A");
        assert_eq!(rail.platforms.len(), 1);
        assert!(RailNetwork::from_map_json("{}").is_empty(), "old extracts load unchanged");

        let transit = r#"{"rail_trips":[{"class":"emu","carriages":7,
                          "stops":[{"x":0,"y":0,"arr":100,"dep":120},{"x":900,"y":0,"arr":220,"dep":230}]}],
                          "bus_trips":[{"line":"ECR","stops":[{"x":0,"y":0,"arr":50,"dep":50,"timepoint":true}]}]}"#;
        let (rt, bt) = transit_from_json(transit).unwrap();
        assert_eq!((rt.len(), bt.len()), (1, 1));
        assert_eq!(rt[0].class, TrainClass::Emu);
        assert!(bt[0].stops[0].timepoint);
        let (tt, dropped) = build_timetable(&rail, &rt);
        assert_eq!((tt.trips.len(), dropped), (1, 0));
    }
}
