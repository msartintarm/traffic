use crate::sim::network::{LaneId, Network};

pub fn control_point(
    net: &Network,
    cur_lane: u32,
    prev_lane: Option<u32>,
    prev_crossing: bool,
    prev: [f32; 2],
    cur: [f32; 2],
) -> [f32; 2] {
    let mid = [(prev[0] + cur[0]) * 0.5, (prev[1] + cur[1]) * 0.5];
    if prev_crossing {
        return mid;
    }
    let Some(pl) = prev_lane else { return mid };
    if pl == cur_lane {
        return mid;
    }
    let prev_link = net.lane(LaneId(pl)).link;
    let cur_link = net.lane(LaneId(cur_lane)).link;
    if prev_link != cur_link && net.link(prev_link).to == net.link(cur_link).from {
        // Pull toward the corner node, but never farther off the chord than the
        // chord itself is long: for tick-adjacent poses (a couple of metres) the
        // path is locally straight and a full node pull bulged the car metres
        // off its interior; a big catch-up hop still sweeps the corner.
        let np = net.node(net.link(cur_link).from).position;
        let (dx, dy) = (np[0] as f32 - mid[0], np[1] as f32 - mid[1]);
        let d = (dx * dx + dy * dy).sqrt();
        let chord = (cur[0] - prev[0]).hypot(cur[1] - prev[1]);
        let max_off = 0.5 * chord;
        if d <= max_off || d < 1e-6 {
            [np[0] as f32, np[1] as f32]
        } else {
            [mid[0] + dx / d * max_off, mid[1] + dy / d * max_off]
        }
    } else {
        mid
    }
}

pub fn interp_pos(prev: [f32; 2], control: [f32; 2], cur: [f32; 2], alpha: f32) -> [f32; 2] {
    let u = 1.0 - alpha;
    let (w0, w1, w2) = (u * u, 2.0 * u * alpha, alpha * alpha);
    [
        w0 * prev[0] + w1 * control[0] + w2 * cur[0],
        w0 * prev[1] + w1 * control[1] + w2 * cur[1],
    ]
}
