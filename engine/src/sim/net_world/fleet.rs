//! The vehicle fleet: the per-vehicle row (public API plus step-private
//! state), its bounded position history, and the storage that owns them —
//! including the optional memory-locality reorder that keeps neighbor reads
//! walking adjacent rows.

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct NetVehicle {
    pub id: u32,
    pub lane: LaneId,
    pub position: f64,
    pub speed: f64,
    /// Kinematic pose `[x, y, heading]` at the **rear axle**: a bicycle-model
    /// state advanced by [`NetWorld::track_path`] — heading steers (bounded by
    /// steering geometry and grip), position follows the wheels. The arc position
    /// is the *reference being tracked*, not the pose itself, so no geometry
    /// defect — a kinked polyline, a degenerate interior, a mouth overlap — can
    /// render as motion a four-wheeled car cannot produce. This is what the world
    /// pose, the render, and the collision footprint all see.
    pub(super) kin: [f64; 3],
    /// Current steered path curvature (1/m, left-positive) — the wheel position.
    /// [`NetWorld::track_path`] slews it toward its command at [`STEER_RATE`],
    /// so lock builds and unwinds at a human pace instead of snapping.
    pub(super) steer: f64,
    pub driver: DriverConfig,
    pub route: Vec<LinkId>,
    pub route_idx: usize,
    /// Destination link for flow-field routing; when set (with a world router)
    /// it supersedes `route` and reroutes live around congestion.
    pub dest: Option<LinkId>,
    /// The stop-controlled node this vehicle has already halted at, so a stop
    /// sign is enforced once rather than forever.
    pub(super) stopped_at: Option<NodeId>,
    /// Consecutive ticks spent essentially stopped — drives yield impatience.
    pub(super) wait_ticks: u32,
    /// When set, the vehicle has crossed its current lane's stop line and is
    /// traversing this movement's node interior. `position` keeps counting past
    /// `lane.length`, so the interior arc is `position - lane.length` — the road
    /// is one continuous corridor across the seam. Cleared when it lands on the
    /// destination lane (`position` rebased to the new lane's frame).
    pub(super) crossing: Option<Crossing>,
    pub(super) lane_change: Option<LaneChange>,
    /// Whether the active-set scheduler classified this car as sleeping on the
    /// last step — read by the next step's lane-change pass to stagger the
    /// (slow-timescale) queue-jump evaluation of parked cars.
    pub(super) slept: bool,
    /// Local routing only: the next link this driver intends to take, decided
    /// on link entry by a bounded neighbourhood search and read O(1) each tick
    /// (`None` for field-routed cars, or a local car that has arrived).
    pub(super) next_link: Option<LinkId>,
    /// Ticks until this wreck is cleared from the road. `None` = not crashed. A
    /// wreck holds its pose at speed 0 and blocks traffic like any stopped car
    /// (leader chains, box occupancy) until the timer removes it.
    pub(super) wreck: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Crossing {
    pub(super) movement: MovementId,
    /// Lateral metres (right-positive) the car sat off its lane line when it hit
    /// the boundary — a lane-change blend still in flight. Carried through the
    /// crossing so the seam-landing blend starts from where the car actually is.
    pub(super) lat_shift: f64,
    /// Ticks elapsed since this crossing began — lets a permissive-left waiter
    /// stuck mid-box past the sneaker window creep out (see `PERMISSIVE_SNEAK_SECS`).
    pub(super) held: u16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LaneChange {
    pub(super) from: LaneId,
    pub(super) progress: f64,
}

/// Outcome of advancing a vehicle one tick.
#[derive(Clone, Copy)]
pub(super) enum Fate {
    Alive,
    /// Crossed onto a new link (its id, for entry counting).
    Entered(LinkId),
    /// Left the network legitimately — reached its destination, finished its
    /// route, or ran off a genuine dead end.
    Exited,
    /// Removed at an interior node despite still having a routable next hop — a
    /// vehicle that *disappeared* at an intersection. Should never happen; counted
    /// so a regression in movement resolution is caught rather than silent.
    Leaked,
}

/// How many `(position, speed)` samples to retain — enough for the largest
/// plausible reaction delay at the fixed timestep.
pub(super) const HISTORY_LEN: usize = 8;

pub(super) type History = [(f64, f64); HISTORY_LEN];

/// Vehicle storage as columns: the rows plus the per-vehicle reaction-delay
/// history kept out of the row (it is only read for a leader, not while iterating
/// every row). Columns stay index-aligned; mutations go through here.
#[derive(Clone, Debug, Default)]
pub(super) struct Fleet {
    pub(super) rows: Vec<NetVehicle>,
    pub(super) hist: Vec<History>,
    pub(super) hist_len: Vec<u8>,
}

impl Fleet {
    pub(super) fn push(&mut self, v: NetVehicle) {
        let mut h = [(0.0, 0.0); HISTORY_LEN];
        h[0] = (v.position, v.speed);
        self.hist.push(h);
        self.hist_len.push(1);
        self.rows.push(v);
    }

    pub(super) fn clear(&mut self) {
        self.rows.clear();
        self.hist.clear();
        self.hist_len.clear();
    }

    /// Row `i`'s `(position, speed)` `ticks` steps ago (clamped to the oldest kept).
    pub(super) fn delayed(&self, i: usize, ticks: usize) -> (f64, f64) {
        let n = self.hist_len[i] as usize;
        self.hist[i][n - 1 - ticks.min(n - 1)]
    }

    /// Whether row `i` has more than `ticks` samples on its *current* lane, so a
    /// `ticks`-delayed lookup is a real in-frame position rather than one clamped back
    /// to (or across) a recent segment crossing. History is reset on crossing, so this
    /// gates the reaction-delay model back on only once the car has settled.
    pub(super) fn settled(&self, i: usize, ticks: usize) -> bool {
        self.hist_len[i] as usize > ticks
    }

    /// Drop row `i`'s retained history down to just `(position, speed)`, so its
    /// delayed leader-gap lookup falls back to the true current gap until it has
    /// re-accumulated a full window — used when a lane change moves the car to a new
    /// lane (and thus a new leader) whose old-frame positions would phantom-brake it.
    pub(super) fn reset_history(&mut self, i: usize, position: f64, speed: f64) {
        self.hist[i][0] = (position, speed);
        self.hist_len[i] = 1;
    }
}

/// Append the current `(position, speed)` to a history column entry, dropping the
/// oldest sample once full.
pub(super) fn record_history(hist: &mut History, len: &mut u8, position: f64, speed: f64) {
    let n = *len as usize;
    if n < HISTORY_LEN {
        hist[n] = (position, speed);
        *len += 1;
    } else {
        hist.copy_within(1.., 0);
        hist[HISTORY_LEN - 1] = (position, speed);
    }
}

impl NetVehicle {
    /// The rear-axle point of the kinematic pose — the bicycle model's ground
    /// truth. By construction it moves (almost exactly) along the heading, so
    /// this is where slip/crab invariants should be measured; the front bumper
    /// legitimately sweeps sideways through turns.
    pub fn rear_axle(&self) -> [f64; 2] {
        [self.kin[0], self.kin[1]]
    }

    /// Whether the vehicle is currently inside a node traversing a movement's
    /// interior path (`position` counts on past its `lane`'s length; the overrun
    /// is the interior arc).
    pub fn is_crossing(&self) -> bool {
        self.crossing.is_some()
    }

    /// Whether this vehicle is a crashed wreck awaiting clearance.
    pub fn is_wrecked(&self) -> bool {
        self.wreck.is_some()
    }

    /// Consecutive ticks spent essentially stopped — the wait the yield
    /// impatience and the HUD's junction readout run on.
    pub fn wait_ticks(&self) -> u32 {
        self.wait_ticks
    }
}

impl NetWorld {
    /// Permute the fleet (rows + histories, together) into (lane, arc-position)
    /// order. Any order is a valid simulation; this one makes every per-car
    /// pass's neighbor reads mostly-sequential in memory.
    pub(super) fn locality_reorder(&mut self) {
        let n = self.fleet.rows.len();
        let shard_of = |lane: LaneId| self.sharding.as_ref().map_or(0, |sh| sh.home_of(lane));
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| {
            let (va, vb) = (&self.fleet.rows[a as usize], &self.fleet.rows[b as usize]);
            (shard_of(va.lane), va.lane.0, va.position)
                .partial_cmp(&(shard_of(vb.lane), vb.lane.0, vb.position))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rows = std::mem::take(&mut self.fleet.rows);
        let hist = std::mem::take(&mut self.fleet.hist);
        let hist_len = std::mem::take(&mut self.fleet.hist_len);
        let mut rows: Vec<Option<NetVehicle>> = rows.into_iter().map(Some).collect();
        let mut hist: Vec<Option<History>> = hist.into_iter().map(Some).collect();
        for &i in &order {
            self.fleet.rows.push(rows[i as usize].take().unwrap());
            self.fleet.hist.push(hist[i as usize].take().unwrap());
            self.fleet.hist_len.push(hist_len[i as usize]);
        }
    }
}
