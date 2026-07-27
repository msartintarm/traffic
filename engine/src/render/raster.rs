//! Software rasteriser: renders the world [`StaticMesh`] to an RGB pixel buffer
//! with no GPU and no browser — the basis for golden-image regression tests of
//! intersection rendering. It draws the *same* geometry the browser uploads
//! (via the [`RenderTarget`] seam and [`super::draw_world`]), filling each
//! triangle in its vertex colour, so a pixel diff catches appearance changes.
//!
//! It draws true world widths (the GPU shader's min-on-screen-width expansion is
//! zoom cosmetics), which for a zoomed-in junction view matches the GPU closely.

use super::{RenderTarget, StaticMesh};

pub const BG: [u8; 3] = [11, 14, 19];

pub struct Raster {
    w: usize,
    h: usize,
    min: [f64; 2],
    max: [f64; 2],
    bg: [u8; 3],
    px: Vec<[u8; 3]>,
}

impl Raster {
    pub fn new(min: [f64; 2], max: [f64; 2], w: usize, h: usize, bg: [u8; 3]) -> Self {
        Self { w, h, min, max, bg, px: vec![bg; w * h] }
    }

    /// A square `size`×`size` view of half-extent `r` metres centred on `c`.
    pub fn centered(c: [f64; 2], r: f64, size: usize, bg: [u8; 3]) -> Self {
        Self::new([c[0] - r, c[1] - r], [c[0] + r, c[1] + r], size, size, bg)
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Flat row-major RGB bytes (`w*h*3`).
    pub fn rgb(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h * 3);
        for p in &self.px {
            out.extend_from_slice(p);
        }
        out
    }

    fn pixel_center(&self, col: usize, row: usize) -> [f64; 2] {
        [
            self.min[0] + (col as f64 + 0.5) / self.w as f64 * (self.max[0] - self.min[0]),
            self.max[1] - (row as f64 + 0.5) / self.h as f64 * (self.max[1] - self.min[1]),
        ]
    }

    /// Fill every mesh triangle in the colour carried by its vertices.
    pub fn fill_mesh(&mut self, mesh: &StaticMesh) {
        for tri in mesh.indices.chunks_exact(3) {
            let v = |i: u32| {
                let sv = mesh.vertices[i as usize];
                [(sv.center[0] + sv.offset[0]) as f64, (sv.center[1] + sv.offset[1]) as f64]
            };
            let c = mesh.vertices[tri[0] as usize].color;
            let color = [(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8];
            self.fill_tri([v(tri[0]), v(tri[1]), v(tri[2])], color);
        }
    }

    fn fill_tri(&mut self, t: [[f64; 2]; 3], color: [u8; 3]) {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in t {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        // World bbox → pixel row/col range (y flips), clamped to the canvas.
        let to_col = |x: f64| ((x - self.min[0]) / (self.max[0] - self.min[0]) * self.w as f64).floor() as isize;
        let to_row = |y: f64| ((self.max[1] - y) / (self.max[1] - self.min[1]) * self.h as f64).floor() as isize;
        let c0 = to_col(lo[0]).clamp(0, self.w as isize - 1);
        let c1 = to_col(hi[0]).clamp(0, self.w as isize - 1);
        let r0 = to_row(hi[1]).clamp(0, self.h as isize - 1);
        let r1 = to_row(lo[1]).clamp(0, self.h as isize - 1);
        for row in r0..=r1 {
            for col in c0..=c1 {
                if point_in_tri(t, self.pixel_center(col as usize, row as usize)) {
                    self.px[row as usize * self.w + col as usize] = color;
                }
            }
        }
    }
}

// --- realism metrics: how closely the render matches a real intersection ------

impl Raster {
    fn paved(&self, i: usize) -> bool {
        self.px[i] != self.bg
    }

    fn world_to_px(&self, p: [f64; 2]) -> (isize, isize) {
        let col = ((p[0] - self.min[0]) / (self.max[0] - self.min[0]) * self.w as f64) as isize;
        let row = ((self.max[1] - p[1]) / (self.max[1] - self.min[1]) * self.h as f64) as isize;
        (col, row)
    }

    /// Sizes (px) of paved regions (8-connectivity) that do **not** touch the
    /// image border — largest first. A road truncated at the crop edge touches the
    /// border (a legitimate neighbour, ignored); an interior island of this size
    /// is a stray nub floating off the roadway.
    pub fn interior_paved_fragments(&self) -> Vec<usize> {
        let (w, h) = (self.w, self.h);
        let mut seen = vec![false; w * h];
        let mut sizes = Vec::new();
        for start in 0..w * h {
            if seen[start] || !self.paved(start) {
                continue;
            }
            let mut stack = vec![start];
            seen[start] = true;
            let (mut size, mut border) = (0usize, false);
            while let Some(i) = stack.pop() {
                size += 1;
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                if x == 0 || y == 0 || x == w as isize - 1 || y == h as isize - 1 {
                    border = true;
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if !seen[j] && self.paved(j) {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
            if !border {
                sizes.push(size);
            }
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// Count of background pixels fully enclosed by pavement (no 4-connected path
    /// to the image border) — dark holes inside the intersection. A real
    /// intersection's pavement is a solid surface, so this should be ~0.
    pub fn enclosed_holes(&self) -> usize {
        let (w, h) = (self.w, self.h);
        let mut reached = vec![false; w * h];
        let mut stack = Vec::new();
        for i in 0..w * h {
            let (x, y) = (i % w, i / w);
            let border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            if border && !self.paved(i) && !reached[i] {
                reached[i] = true;
                stack.push(i);
            }
        }
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if !reached[j] && !self.paved(j) {
                    reached[j] = true;
                    stack.push(j);
                }
            }
        }
        (0..w * h).filter(|&i| !self.paved(i) && !reached[i]).count()
    }

    /// Fraction of pixels within `radius_m` of world point `c` that are paved —
    /// the crossing core should be ~fully paved at a real intersection.
    pub fn fill_ratio(&self, c: [f64; 2], radius_m: f64) -> f64 {
        let (cx, cy) = self.world_to_px(c);
        let rpx = radius_m / (self.max[0] - self.min[0]) * self.w as f64;
        let (mut paved, mut total) = (0usize, 0usize);
        let r = rpx.ceil() as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                if (dx * dx + dy * dy) as f64 > rpx * rpx {
                    continue;
                }
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= self.w as isize || y >= self.h as isize {
                    continue;
                }
                total += 1;
                if self.paved(y as usize * self.w + x as usize) {
                    paved += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            paved as f64 / total as f64
        }
    }
}

impl RenderTarget for Raster {
    fn world(&mut self, mesh: &StaticMesh) {
        self.fill_mesh(mesh);
    }
    fn markings(&mut self, mesh: &StaticMesh) {
        self.fill_mesh(mesh);
    }
    fn vehicle(&mut self, _pose: [f64; 3]) {} // regression screenshots are car-free
}

fn point_in_tri(t: [[f64; 2]; 3], p: [f64; 2]) -> bool {
    let edge = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let (s0, s1, s2) = (edge(t[0], t[1]), edge(t[1], t[2]), edge(t[2], t[0]));
    (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
}

/// How closely the rendered intersections match real ones, measured on subsets
/// of the Millbrae map. These describe realism, not exact pixels, so they survive
/// intended tweaks while still failing when the intersection looks wrong.
#[cfg(all(test, feature = "import"))]
mod realism {
    use super::*;
    use crate::render::draw_world;
    use crate::sim::map::millbrae_junction;
    use crate::sim::network::{Network, NodeId};

    fn render(n: usize) -> (Network, Raster) {
        let net = millbrae_junction(n);
        let mut r = Raster::centered([0.0, 0.0], 70.0, 512, BG);
        draw_world(&net, &[], &mut r);
        (net, r)
    }

    /// The busiest node — the crossing itself — where a real intersection's box is.
    fn crossing_center(net: &Network) -> [f64; 2] {
        let degree = |n: NodeId| net.links.iter().filter(|l| l.from == n || l.to == n).count();
        let node = (0..net.nodes.len() as u32).map(NodeId).max_by_key(|&n| degree(n)).unwrap();
        net.node(node).position
    }

    #[test]
    fn the_crossing_core_is_solid_pavement() {
        // Right at the crossing, the pavement is unbroken (no median splits the box
        // there) — the strongest "this reads as an intersection" signal.
        for n in 0..3 {
            let (net, r) = render(n);
            let f = r.fill_ratio(crossing_center(&net), 8.0);
            // >83%: a solid crossing box (skew crossings lose a little to acute
            // corners). Well clear of the ~72% a gappy star-fill scored.
            assert!(f > 0.83, "junction {n}: crossing core only {:.0}% paved — a real intersection box is solid pavement", f * 100.0);
        }
    }

    #[test]
    fn no_stray_nub_islands() {
        // Every paved region is either the connected road network (large) or
        // negligible speckle — never a mid-sized fragment floating off the roads,
        // which is what a leftover stub renders as.
        for n in 0..3 {
            let nubs: Vec<usize> =
                render(n).1.interior_paved_fragments().into_iter().filter(|&s| (60..8000).contains(&s)).collect();
            assert!(nubs.is_empty(), "junction {n}: stray interior paved fragment(s) of size {nubs:?} px — a real junction has no islands off the roadway");
        }
    }
}

#[cfg(all(test, feature = "import"))]
mod golden {
    use super::*;
    use crate::render::draw_world;
    use crate::sim::map::millbrae_junction;

    fn encode(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(rgb).unwrap();
        out
    }

    fn decode(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
        let dec = png::Decoder::new(bytes);
        let mut reader = dec.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        (info.width as usize, info.height as usize, buf)
    }

    #[test]
    #[ignore] // diagnostic: link inventory for a junction fixture
    fn dump_junction_links() {
        use crate::sim::network::LinkId;
        let net = millbrae_junction(0);
        println!("junction 0: {} nodes, {} links", net.nodes.len(), net.links.len());
        let mut lens: Vec<(usize, f64, u32)> = (0..net.links.len())
            .map(|i| {
                let poly = &net.polylines[i];
                let len: f64 = poly.windows(2).map(|w| (w[0][0] - w[1][0]).hypot(w[0][1] - w[1][1])).sum();
                (i, len, net.link(LinkId(i as u32)).lane_count)
            })
            .collect();
        lens.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (i, len, lanes) in &lens {
            let l = net.link(LinkId(*i as u32));
            let (f, t) = (net.node(l.from).position, net.node(l.to).position);
            println!("link {i}: len={len:.1}m lanes={lanes}  from=({:.0},{:.0}) to=({:.0},{:.0})", f[0], f[1], t[0], t[1]);
        }
    }

    /// Render each real Millbrae junction (no cars) to a PNG and compare against a
    /// committed golden. First run (no golden) writes it for review; a mismatch
    /// writes `<name>.actual.png` beside the golden and fails, so intersection
    /// rendering can never change unnoticed.
    #[test]
    fn millbrae_junctions_match_golden_screenshots() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/render/fixtures");
        std::fs::create_dir_all(dir).unwrap();
        const SIZE: usize = 512;
        for n in 0..3 {
            let net = millbrae_junction(n);
            let mut r = Raster::centered([0.0, 0.0], 70.0, SIZE, BG);
            draw_world(&net, &[], &mut r); // car-free static geometry
            let rgb = r.rgb();
            let png_bytes = encode(SIZE, SIZE, &rgb);

            let golden_path = format!("{dir}/junction_{n}.png");
            match std::fs::read(&golden_path) {
                Ok(golden) => {
                    let (_, _, gpx) = decode(&golden);
                    let diff = rgb.iter().zip(&gpx).filter(|(a, b)| a != b).count();
                    if diff > 0 {
                        std::fs::write(format!("{dir}/junction_{n}.actual.png"), &png_bytes).unwrap();
                        panic!("junction {n} render changed ({diff} byte diffs); wrote junction_{n}.actual.png — review, then update the golden if intended");
                    }
                }
                Err(_) => {
                    std::fs::write(&golden_path, &png_bytes).unwrap();
                    println!("wrote initial golden junction_{n}.png ({SIZE}x{SIZE}) — review it and commit");
                }
            }
        }
    }
}
