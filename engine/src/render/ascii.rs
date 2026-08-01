//! ASCII rasteriser for world geometry — a browser-free way to *see* and assert
//! layout (junction fills, road ribbons, interior paths, vehicle placement) in
//! `cargo test`. It rasterises the exact [`StaticMesh`] triangles the GPU draws
//! (a world vertex is `center + offset`), so what it prints is what the browser
//! renders. This is the fast regression loop: no DOM, no device, no dev server.

use super::{RenderTarget, StaticMesh};

pub struct Ascii {
    cols: usize,
    rows: usize,
    min: [f64; 2],
    max: [f64; 2],
    buf: Vec<char>,
}

impl Ascii {
    pub fn new(min: [f64; 2], max: [f64; 2], cols: usize, rows: usize) -> Self {
        Self { cols, rows, min, max, buf: vec![' '; cols * rows] }
    }

    /// A square world view of half-extent `r` metres centred on `c`. `cols` is
    /// doubled relative to `rows` so the ~2:1 character cell reads roughly square.
    pub fn centered(c: [f64; 2], r: f64, rows: usize) -> Self {
        Self::new([c[0] - r, c[1] - r], [c[0] + r, c[1] + r], rows * 2, rows)
    }

    fn cell_center(&self, col: usize, row: usize) -> [f64; 2] {
        [
            self.min[0] + (col as f64 + 0.5) / self.cols as f64 * (self.max[0] - self.min[0]),
            self.max[1] - (row as f64 + 0.5) / self.rows as f64 * (self.max[1] - self.min[1]), // north up
        ]
    }

    fn cell_of(&self, p: [f64; 2]) -> Option<(usize, usize)> {
        if p[0] < self.min[0] || p[0] > self.max[0] || p[1] < self.min[1] || p[1] > self.max[1] {
            return None;
        }
        let col = ((p[0] - self.min[0]) / (self.max[0] - self.min[0]) * self.cols as f64) as usize;
        let row = ((self.max[1] - p[1]) / (self.max[1] - self.min[1]) * self.rows as f64) as usize;
        Some((col.min(self.cols - 1), row.min(self.rows - 1)))
    }

    /// Whether the cell at `(col, row)` is set to `ch` — for assertions.
    pub fn at(&self, col: usize, row: usize) -> char {
        self.buf[row * self.cols + col]
    }

    pub fn cell_at_world(&self, p: [f64; 2]) -> Option<char> {
        self.cell_of(p).map(|(c, r)| self.at(c, r))
    }

    pub fn plot(&mut self, p: [f64; 2], ch: char) {
        if let Some((c, r)) = self.cell_of(p) {
            self.buf[r * self.cols + c] = ch;
        }
    }

    /// Plot a polyline path (e.g. an interior Bézier sampled to points).
    pub fn plot_path(&mut self, pts: &[[f64; 2]], ch: char) {
        for &p in pts {
            self.plot(p, ch);
        }
    }

    /// Fill every cell whose centre lies inside a triangle of `mesh`.
    pub fn fill_mesh(&mut self, mesh: &StaticMesh, ch: char) {
        for tri in mesh.indices.chunks_exact(3) {
            let v = |i: u32| {
                let sv = mesh.vertices[i as usize];
                [(sv.center[0] + sv.offset[0]) as f64, (sv.center[1] + sv.offset[1]) as f64]
            };
            self.fill_tri([v(tri[0]), v(tri[1]), v(tri[2])], ch);
        }
    }

    fn fill_tri(&mut self, t: [[f64; 2]; 3], ch: char) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                if point_in_tri(t, self.cell_center(col, row)) {
                    self.buf[row * self.cols + col] = ch;
                }
            }
        }
    }

    pub fn render(&self) -> String {
        let mut s = String::with_capacity((self.cols + 1) * self.rows);
        for row in 0..self.rows {
            s.extend(&self.buf[row * self.cols..(row + 1) * self.cols]);
            s.push('\n');
        }
        s
    }

    /// Count of cells set to `ch` — a cheap assertable measure of coverage.
    pub fn count(&self, ch: char) -> usize {
        self.buf.iter().filter(|&&c| c == ch).count()
    }
}

/// The ASCII rasteriser as a drop-in [`RenderTarget`]: `super::draw_world` feeds
/// it the same geometry the GPU renderer gets, so the two stay in parity.
impl RenderTarget for Ascii {
    fn world(&mut self, mesh: &StaticMesh) {
        self.fill_mesh(mesh, '#');
    }
    fn vehicle(&mut self, pose: [f64; 3]) {
        self.plot([pose[0], pose[1]], '@');
    }
}

fn point_in_tri(t: [[f64; 2]; 3], p: [f64; 2]) -> bool {
    let edge = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let (s0, s1, s2) = (edge(t[0], t[1]), edge(t[1], t[2]), edge(t[2], t[0]));
    (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
}

/// Scenario tests driven through the ASCII renderer: build a small map, run the
/// sim, and both assert behaviour and (with `--nocapture`) *watch* it — the
/// browser-free way to unit-test dynamic behaviour.
#[cfg(test)]
mod scenarios {
    use super::*;
    use crate::render::draw_world;
    use crate::sim::config::{DriverConfig, SimConfig};
    use crate::sim::map::{LinkSpec, NodeSpec, OsmMap};
    use crate::sim::net_world::NetWorld;
    use crate::sim::network::LinkId;

    /// Render the world (roads + current vehicle poses) to ASCII.
    fn snapshot(w: &NetWorld, center: [f64; 2], r: f64) -> Ascii {
        let mut a = Ascii::centered(center, r, 22);
        let poses: Vec<[f64; 3]> = w.vehicles().iter().map(|v| w.vehicle_world_pose(v)).collect();
        draw_world(&w.network, &poses, &mut a);
        a
    }

    #[test]
    fn a_car_changes_into_its_turn_lane_before_the_junction() {
        // 2-lane west approach into a 4-way; channelisation puts the left turn (to
        // N) on the left lane (index 0, where cars spawn) and the right turn (to S)
        // on the right lane. A car spawned in the left lane but bound for S must
        // lane-change to reach its turn — which dest-routed cars only do now that
        // next_link_on_path drives the mandatory change.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),    // junction
                NodeSpec::uncontrolled(1, -220.0, 0.0), // W (2-lane approach)
                NodeSpec::uncontrolled(2, 220.0, 0.0),  // E (through)
                NodeSpec::uncontrolled(3, 0.0, 220.0),  // N (left)
                NodeSpec::uncontrolled(4, 0.0, -220.0), // S (right)
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 2, 15.0), // 0: W→junction, two lanes
                LinkSpec::oneway(0, 2, 1, 15.0), // 1: →E through
                LinkSpec::oneway(0, 3, 1, 15.0), // 2: →N left
                LinkSpec::oneway(0, 4, 1, 15.0), // 3: →S right
            ],
        }
        .build();
        let mut w = NetWorld::new(net, SimConfig::default_config());
        w.install_router(&[LinkId(3)]); // destination: the right-turn exit (→S)
        assert!(w.spawn_to(1, LinkId(0), LinkId(3), 10.0, DriverConfig::car()), "spawns in the left lane");

        for t in 0..300 {
            w.step();
            if t == 40 {
                println!("\nturn-lane change (car should be shifting right):\n{}", snapshot(&w, [-40.0, -40.0], 90.0).render());
            }
        }
        // The car reached the right-turn exit link — it found its turn lane.
        assert!(w.link_flows()[3] > 0.0, "the car reached the right-turn exit (changed into its turn lane)");
        assert_eq!(w.crashed(), 0);
    }

    #[test]
    fn a_car_signals_toward_the_turn_lane_it_needs() {
        // Same 2-lane west approach: a car spawned in the left lane (index 0) but
        // bound for the right-turn exit (→S) must signal right (+1) — toward the
        // higher-index lane that serves its route — until it has merged over.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, -220.0, 0.0),
                NodeSpec::uncontrolled(2, 220.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, 220.0),
                NodeSpec::uncontrolled(4, 0.0, -220.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 2, 15.0),
                LinkSpec::oneway(0, 2, 1, 15.0),
                LinkSpec::oneway(0, 3, 1, 15.0),
                LinkSpec::oneway(0, 4, 1, 15.0),
            ],
        }
        .build();
        let mut w = NetWorld::new(net, SimConfig::default_config());
        w.install_router(&[LinkId(3)]); // destination: the right-turn exit (→S)
        // Fill the right lane's entrance so the test car is forced into the left
        // lane (index 0), which serves only the left turn — the wrong lane for →S.
        assert!(w.spawn_to(1, LinkId(0), LinkId(3), 10.0, DriverConfig::car()));
        assert!(w.spawn_to(2, LinkId(0), LinkId(3), 10.0, DriverConfig::car()));
        let wrong = w.vehicles().iter().find(|v| w.network.lane(v.lane).index_in_link == 0).expect("a car in the left lane");
        assert_eq!(w.vehicle_blinker(wrong), 1, "signals right toward the turn lane");
    }

    #[test]
    fn a_car_stuck_in_the_wrong_lane_crosses_instead_of_vanishing() {
        // Regression for cars disappearing when they enter an intersection: a
        // dest-routed car that can't reach a lane serving its route must still
        // traverse the junction (taking an available movement and rerouting), not
        // be silently deleted at the stop line.
        //
        // 2-lane west approach into a 4-way. Channelisation gives lane 0 the left
        // turn (→N) only, and lane 1 the through (→E) and right (→S). A car bound
        // for E is spawned in lane 0; a second car pinned alongside in lane 1 blocks
        // the merge, so the first arrives at the stop line in a lane that can't make
        // its turn. It must still come out the far side — on the N exit its lane
        // does serve — rather than vanishing inside the box.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, -140.0, 0.0), // W (2-lane approach, +x)
                NodeSpec::uncontrolled(2, 140.0, 0.0),  // E (through)
                NodeSpec::uncontrolled(3, 0.0, 140.0),  // N (left)
                NodeSpec::uncontrolled(4, 0.0, -140.0), // S (right)
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 2, 15.0), // 0: W→junction, two lanes
                LinkSpec::oneway(0, 2, 1, 15.0), // 1: →E through
                LinkSpec::oneway(0, 3, 1, 15.0), // 2: →N left
                LinkSpec::oneway(0, 4, 1, 15.0), // 3: →S right
            ],
        }
        .build();
        let mut w = NetWorld::new(net, SimConfig::default_config());
        w.install_router(&[LinkId(1)]); // destination: the through exit (→E), which lane 0 can't serve
        let lanes: Vec<_> = w.network.lanes_of(LinkId(0)).collect();
        let stop = w.network.lane(lanes[0]).length;
        // Place the routed car at the stop line in lane 0 (bound for E, which only
        // lane 1 serves), with a car pinned in lane 1 at the same line so it cannot
        // merge before it enters the box.
        w.spawn_to_in_lane(1, lanes[0], stop, LinkId(1), 6.0, DriverConfig::car());
        w.spawn(2, lanes[1], stop, 6.0, DriverConfig::car());

        let mut saw_on_approach = w.vehicle(1).is_some();
        for t in 0..300 {
            w.step();
            if t == 0 {
                saw_on_approach &= w.vehicle(1).is_some();
                println!("\nstuck in the wrong lane at the line:\n{}", snapshot(&w, [-20.0, 0.0], 70.0).render());
            }
            if t == 60 {
                println!("\nemerged past the junction (not vanished):\n{}", snapshot(&w, [0.0, 45.0], 70.0).render());
            }
        }
        assert!(saw_on_approach, "the routed car exists while approaching");
        // It came out on the N exit (the movement its lane actually serves) — proof
        // it crossed the box instead of being deleted at the stop line. Before the
        // fix `link_flows()[2]` was 0 because the car vanished on entry.
        assert!(w.link_flows()[2] > 0.0, "the stuck car crossed onto the N exit instead of vanishing");
        assert_eq!(w.crashed(), 0, "no collision");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::draw_world;
    use crate::sim::config::{DriverConfig, SimConfig};
    use crate::sim::map::{arterial_intersection, LinkSpec, NodeSpec, OsmMap};
    use crate::sim::net_world::NetWorld;
    use crate::sim::network::{LinkId, MovementId, Network, NodeId, TurnType};

    /// Render a network (no vehicles) through the shared `draw_world` path — the
    /// same geometry the browser GPU renderer receives.
    fn draw(net: &Network, center: [f64; 2], r: f64) -> Ascii {
        let mut a = Ascii::centered(center, r, 30);
        draw_world(net, &[], &mut a);
        a
    }

    #[test]
    fn ascii_shows_a_freeway_diverge_as_a_ramp_not_an_intersection() {
        // A freeway that continues straight (+x) and sheds an off-ramp. Because both
        // sides are grade-separated, the diverge has no stop box: the pavement runs
        // continuously through it and the ramp peels off the side, rather than the
        // arms pulling back into an intersection gap.
        let hw = |a, b, lanes, sp| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let ramp = |a, b, lanes, sp| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -160.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),     // diverge
                NodeSpec::uncontrolled(3, 160.0, 0.0),
                NodeSpec::uncontrolled(4, 120.0, -120.0), // off-ramp
            ],
            links: vec![hw(1, 2, 3, 29.0), hw(2, 3, 3, 29.0), ramp(2, 4, 1, 25.0)],
        }
        .build();
        assert!(net.is_interchange_node(NodeId(1)), "the diverge is a pure interchange node");
        let a = draw(&net, [0.0, -20.0], 60.0);
        println!("\nfreeway diverge (ramp peels off, no intersection box):\n{}", a.render());
        // Pavement is continuous along the mainline through the diverge point…
        assert_eq!(a.cell_at_world([0.0, 0.0]), Some('#'), "the diverge point is paved");
        assert_eq!(a.cell_at_world([-40.0, 0.0]), Some('#'), "mainline in");
        assert_eq!(a.cell_at_world([40.0, 0.0]), Some('#'), "mainline continues");
        // …and the ramp peels off the curb side: the 3-lane mainline spans y∈[-10.5,0],
        // so any pavement well below that in the exit quadrant is the ramp.
        let mut ramp_pavement = 0;
        for xi in 0..12 {
            for yi in 0..12 {
                let p = [xi as f64 * 5.0, -15.0 - yi as f64 * 5.0];
                if a.cell_at_world(p) == Some('#') {
                    ramp_pavement += 1;
                }
            }
        }
        assert!(ramp_pavement >= 4, "the off-ramp peels off the curb side ({ramp_pavement} cells)");
    }

    #[test]
    fn ascii_shows_a_paved_intersection_covering_the_node() {
        let net = arterial_intersection(); // signalized 4-way centred at (0,0)
        let a = draw(&net, [0.0, 0.0], 40.0);
        println!("\narterial intersection:\n{}", a.render());
        assert_eq!(a.cell_at_world([0.0, 0.0]), Some('#'), "the junction centre is paved");
        assert_eq!(a.cell_at_world([-38.0, 0.0]), Some('#'), "west arm present");
        assert_eq!(a.cell_at_world([38.0, 0.0]), Some('#'), "east arm present");
        assert_eq!(a.cell_at_world([0.0, 38.0]), Some('#'), "north arm present");
        assert_eq!(a.cell_at_world([0.0, -38.0]), Some('#'), "south arm present");
    }

    /// A divided crossing split across two nodes 20 m apart (staggered cross
    /// street), merged by target B — the case that produced skewed geometry.
    fn merged_crossing() -> Network {
        let mut links = LinkSpec::twoway(1, 2, 2, 20.0).to_vec();
        links.extend(LinkSpec::twoway(10, 1, 2, 20.0));
        links.extend(LinkSpec::twoway(1, 11, 1, 13.0));
        links.extend(LinkSpec::twoway(2, 12, 2, 20.0));
        links.extend(LinkSpec::twoway(2, 13, 1, 13.0));
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 20.0, 0.0),
                NodeSpec::uncontrolled(10, -120.0, 0.0),
                NodeSpec::uncontrolled(11, 0.0, -120.0),
                NodeSpec::uncontrolled(12, 140.0, 0.0),
                NodeSpec::uncontrolled(13, 20.0, 120.0),
            ],
            links,
        }
        .merge_split_intersections()
        .build()
    }

    #[test]
    fn ascii_reveals_a_merged_split_crossing() {
        let net = merged_crossing();
        let a = draw(&net, [10.0, 0.0], 45.0);
        println!("\nmerged split crossing (centroid ~[10,0]):\n{}", a.render());
        assert!(a.count('#') > 0, "the merged junction paves something");
    }

    fn tangent_at(net: &Network, m: MovementId, s: f64) -> [f64; 2] {
        let p = net.interior_point(m, s);
        [p[2].cos(), p[2].sin()]
    }

    #[test]
    fn interiors_enter_and_exit_aligned_with_their_roads() {
        // Every movement should leave its arrival lane heading along that road and
        // meet its departure lane heading along that one — so a driver never jerks
        // sideways onto a diagonal. Holds even for the staggered merged crossing.
        for net in [arterial_intersection(), merged_crossing()] {
            for m in (0..net.movements.len() as u32).map(MovementId) {
                let mv = net.movement(m);
                let arr = net.arrival_dir(net.lane(mv.from_lane).link);
                let dep = net.departure_dir(net.lane(mv.to_lane).link);
                let te = tangent_at(&net, m, 0.0);
                let tx = tangent_at(&net, m, net.interior(m).len);
                let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
                assert!(dot(te, arr) > 0.9, "entry tangent misaligned with the arrival road (movement {})", m.0);
                assert!(dot(tx, dep) > 0.9, "exit tangent misaligned with the departure road (movement {})", m.0);
            }
        }
    }

    #[test]
    fn ascii_places_vehicles_on_the_carriageway() {
        // A spawned, moving vehicle should rasterise onto paved road — the BE (sim)
        // and FE (geometry) agreeing, checked without a browser.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 160.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0)],
        }
        .build();
        let mut w = NetWorld::new(net, SimConfig::default_config());
        let lane = w.network.lanes_of(LinkId(0)).next().unwrap();
        w.spawn(1, lane, 60.0, 8.0, DriverConfig::car());
        w.run_ticks(2);
        let pose = w.vehicle_world_pose(w.vehicle(1).unwrap());

        let mut a = Ascii::new([0.0, -15.0], [160.0, 15.0], 80, 15);
        assert_eq!(a.cell_at_world([pose[0], pose[1]]), None.or(Some(' ')), "blank before drawing");
        draw_world(&w.network, &[pose], &mut a);
        println!("\nvehicle on a one-way link:\n{}", a.render());
        assert_eq!(a.count('@'), 1, "the vehicle is rasterised");
        // Its cell neighbourhood is paved (it sits on the carriageway).
        let paved = a.cell_at_world([pose[0], pose[1] + 2.0]) == Some('#')
            || a.cell_at_world([pose[0], pose[1] - 2.0]) == Some('#');
        assert!(paved, "the vehicle is on the carriageway");
    }

    fn seg_cross(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> bool {
        let d = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
        let (d1, d2, d3, d4) = (d(b0, b1, a0), d(b0, b1, a1), d(a0, a1, b0), d(a0, a1, b1));
        (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
    }

    fn interior_pts(net: &Network, m: MovementId) -> Vec<[f64; 2]> {
        let len = net.interior(m).len;
        (0..=16).map(|i| { let p = net.interior_point(m, len * i as f64 / 16.0); [p[0], p[1]] }).collect()
    }

    fn paths_cross(net: &Network, a: MovementId, b: MovementId) -> bool {
        let (pa, pb) = (interior_pts(net, a), interior_pts(net, b));
        pa.windows(2).any(|sa| pb.windows(2).any(|sb| seg_cross(sa[0], sa[1], sb[0], sb[1])))
    }

    #[test]
    fn crossing_interiors_are_flagged_as_conflicts() {
        // If two movements' paths physically cross (different approach, different
        // exit — a genuine crossing, not a merge or diverge) they MUST be recorded
        // as conflicting, or vehicles would drive through each other unchecked.
        for net in [arterial_intersection(), merged_crossing()] {
            let n = net.movements.len() as u32;
            for a in 0..n {
                for b in a + 1..n {
                    let (ma, mb) = (net.movement(MovementId(a)), net.movement(MovementId(b)));
                    if ma.node != mb.node {
                        continue;
                    }
                    let same_approach = net.lane(ma.from_lane).link == net.lane(mb.from_lane).link;
                    let same_exit = net.lane(ma.to_lane).link == net.lane(mb.to_lane).link;
                    if same_approach || same_exit {
                        continue; // shared approach/exit = follow or merge, not a crossing
                    }
                    if paths_cross(&net, MovementId(a), MovementId(b)) {
                        assert!(
                            net.movements_conflict(MovementId(a), MovementId(b)),
                            "movements {a} and {b} cross but are not flagged conflicting",
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "import")]
    #[test]
    fn real_millbrae_junctions_render_and_stay_sound() {
        use crate::sim::map::millbrae_junction;
        for n in 0..3 {
            let net = millbrae_junction(n);
            let a = draw(&net, [0.0, 0.0], 60.0);
            println!("\n===== real Millbrae junction {n} ({} nodes / {} links) =====\n{}", net.nodes.len(), net.links.len(), a.render());

            // Same invariants as the synthetic cases, now on real complexity.
            for m in (0..net.movements.len() as u32).map(MovementId) {
                let mv = net.movement(m);
                let arr = net.arrival_dir(net.lane(mv.from_lane).link);
                let dep = net.departure_dir(net.lane(mv.to_lane).link);
                let te = tangent_at(&net, m, 0.0);
                let tx = tangent_at(&net, m, net.interior(m).len);
                let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
                assert!(dot(te, arr) > 0.85, "junction {n} movement {} enters misaligned (dot {:.2})", m.0, dot(te, arr));
                assert!(dot(tx, dep) > 0.85, "junction {n} movement {} exits misaligned (dot {:.2})", m.0, dot(tx, dep));
            }
            let nm = net.movements.len() as u32;
            for a in 0..nm {
                for b in a + 1..nm {
                    let (ma, mb) = (net.movement(MovementId(a)), net.movement(MovementId(b)));
                    if ma.node != mb.node
                        || net.lane(ma.from_lane).link == net.lane(mb.from_lane).link
                        || net.lane(ma.to_lane).link == net.lane(mb.to_lane).link
                    {
                        continue;
                    }
                    if paths_cross(&net, MovementId(a), MovementId(b)) {
                        assert!(net.movements_conflict(MovementId(a), MovementId(b)),
                            "junction {n}: movements {a},{b} cross but aren't flagged conflicting");
                    }
                }
            }
        }
    }

    /// Max perpendicular deviation (m) of a movement's interior Bézier from the
    /// straight line between its entry and exit — 0 means a truly linear path.
    fn interior_bow(net: &Network, m: MovementId) -> f64 {
        let it = net.interior(m);
        let (a, b) = (it.entry, it.exit);
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = d[0].hypot(d[1]).max(1e-9);
        (0..=20)
            .map(|i| {
                let p = net.interior_point(m, it.len * i as f64 / 20.0);
                ((p[0] - a[0]) * d[1] - (p[1] - a[1]) * d[0]).abs() / len
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn through_movements_are_linear_on_an_aligned_crossing() {
        // On a well-formed (grid-aligned) crossing, every straight-through movement
        // must drive a straight line — the property the intersection should keep.
        let net = arterial_intersection();
        let throughs: Vec<MovementId> = (0..net.movements.len() as u32)
            .map(MovementId)
            .filter(|&m| net.movement_turn(m) == TurnType::Through)
            .collect();
        assert!(!throughs.is_empty(), "the arterial has through movements");
        for m in throughs {
            assert!(interior_bow(&net, m) < 0.5, "through interior should be straight, bow={}", interior_bow(&net, m));
        }
    }
}
