//! Fixed-timestep clock with play/pause/single-step/speed. The render loop
//! feeds real elapsed seconds into [`SimClock::advance`], which returns how
//! many fixed ticks to run; leftover time accumulates so speed never alters the
//! physics.

use super::config::SimConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayState {
    Paused,
    Playing,
}

#[derive(Clone, Debug)]
pub struct SimClock {
    dt: f64,
    state: PlayState,
    speed: f64,
    accumulator: f64,
    tick: u64,
}

impl SimClock {
    pub fn new(cfg: &SimConfig) -> Self {
        Self {
            dt: cfg.dt,
            state: PlayState::Paused,
            speed: 1.0,
            accumulator: 0.0,
            tick: 0,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Fractional progress into the next tick, in `[0, 1)` — the blend factor
    /// for interpolating the render between the last two committed sim states.
    pub fn alpha(&self) -> f64 {
        (self.accumulator / self.dt).clamp(0.0, 1.0)
    }

    pub fn state(&self) -> PlayState {
        self.state
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn play(&mut self) {
        self.state = PlayState::Playing;
    }

    pub fn pause(&mut self) {
        self.state = PlayState::Paused;
    }

    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.clamp(0.0, 64.0);
    }

    pub fn single_step(&mut self) -> u64 {
        let completed = self.tick;
        self.tick += 1;
        self.accumulator = 0.0;
        completed
    }

    /// Discard accumulated-but-unrun sim time. Called when a frame's catch-up hits
    /// its wall-clock budget, so an overloaded frame degrades to slow-motion rather
    /// than building an unpayable backlog that freezes the main thread.
    pub fn drop_backlog(&mut self) {
        self.accumulator = 0.0;
    }

    pub fn advance(&mut self, real_elapsed: f64, max_ticks: u32) -> u32 {
        if self.state == PlayState::Paused || self.speed == 0.0 {
            return 0;
        }
        self.accumulator += real_elapsed * self.speed;
        let mut ran = 0;
        let threshold = self.dt - 1e-9;
        while self.accumulator >= threshold && ran < max_ticks {
            self.accumulator -= self.dt;
            self.tick += 1;
            ran += 1;
        }
        if ran == max_ticks {
            self.accumulator = 0.0;
        }
        ran
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> SimClock {
        SimClock::new(&SimConfig { dt: 0.2, seed: 0 })
    }

    #[test]
    fn paused_clock_runs_no_ticks() {
        let mut c = clock();
        assert_eq!(c.advance(10.0, 1000), 0);
        assert_eq!(c.tick(), 0);
    }

    #[test]
    fn playing_converts_elapsed_time_into_ticks() {
        let mut c = clock();
        c.play();
        assert_eq!(c.advance(1.0, 1000), 5); // 1.0s / 0.2s
        assert_eq!(c.tick(), 5);
    }

    #[test]
    fn leftover_time_accumulates_across_calls() {
        let mut c = clock();
        c.play();
        assert_eq!(c.advance(0.15, 1000), 0);
        assert_eq!(c.advance(0.15, 1000), 1); // 0.30s total ⇒ one 0.2s tick
    }

    #[test]
    fn speed_scales_tick_rate() {
        let mut c = clock();
        c.play();
        c.set_speed(4.0);
        assert_eq!(c.advance(1.0, 1000), 20); // 4× as many ticks
    }

    #[test]
    fn catch_up_is_bounded() {
        let mut c = clock();
        c.play();
        assert_eq!(c.advance(1000.0, 10), 10);
    }

    #[test]
    fn drop_backlog_discards_pending_time() {
        let mut c = clock();
        c.play();
        assert_eq!(c.advance(0.5, 1000), 2); // 0.5s / 0.2s = 2 ticks, 0.1s left over
        assert!(c.alpha() > 0.0, "leftover time should register as sub-tick progress");
        c.drop_backlog();
        assert_eq!(c.alpha(), 0.0);
        assert_eq!(c.advance(0.0, 1000), 0, "nothing carried into the next frame");
    }

    #[test]
    fn alpha_tracks_sub_tick_progress() {
        let mut c = clock();
        c.play();
        assert_eq!(c.advance(0.1, 1000), 0); // half a 0.2s tick, no tick yet
        assert!((c.alpha() - 0.5).abs() < 1e-9);
        assert_eq!(c.advance(0.1, 1000), 1); // completes the tick
        assert!(c.alpha() < 1e-9);
    }

    #[test]
    fn single_step_advances_while_paused() {
        let mut c = clock();
        assert_eq!(c.single_step(), 0);
        assert_eq!(c.tick(), 1);
    }
}
