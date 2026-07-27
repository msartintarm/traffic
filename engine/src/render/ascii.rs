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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{draw_world, geometry::bezier};
    use crate::sim::config::{DriverConfig, SimConfig};
    use crate::sim::map::{arterial_intersection, LinkSpec, NodeSpec, OsmMap};
    use crate::sim::net_world::NetWorld;
    use crate::sim::network::{LinkId, MovementId, Network, TurnType};

    /// Render a network (no vehicles) through the shared `draw_world` path — the
    /// same geometry the browser GPU renderer receives.
    fn draw(net: &Network, center: [f64; 2], r: f64) -> Ascii {
        let mut a = Ascii::centered(center, r, 30);
        draw_world(net, &[], &mut a);
        a
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

    #[test]
    fn ascii_reveals_a_merged_split_crossing() {
        // A divided crossing split across two nodes 20 m apart, merged by target B.
        let mut links = LinkSpec::twoway(1, 2, 2, 20.0).to_vec();
        links.extend(LinkSpec::twoway(10, 1, 2, 20.0));
        links.extend(LinkSpec::twoway(1, 11, 1, 13.0));
        links.extend(LinkSpec::twoway(2, 12, 2, 20.0));
        links.extend(LinkSpec::twoway(2, 13, 1, 13.0));
        let net = OsmMap {
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
        .build();
        let a = draw(&net, [10.0, 0.0], 45.0);
        println!("\nmerged split crossing (centroid ~[10,0]):\n{}", a.render());
        assert!(a.count('#') > 0, "the merged junction paves something");
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

    /// Max perpendicular deviation (m) of a movement's interior Bézier from the
    /// straight line between its entry and exit — 0 means a truly linear path.
    fn interior_bow(net: &Network, m: MovementId) -> f64 {
        let it = net.interior(m);
        let (a, b) = (it.entry, it.exit);
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = d[0].hypot(d[1]).max(1e-9);
        (0..=20)
            .map(|i| {
                let p = bezier(a, it.ctrl, b, i as f64 / 20.0);
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
