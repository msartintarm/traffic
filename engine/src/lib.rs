//! Traffic engine core.
//!
//! Architecture mirrors the sibling `../plant` project: one Rust crate,
//! compiled to WebAssembly for the browser, split so the *simulation* is a
//! pure, dependency-free core (`sim/`) that runs and is exhaustively tested
//! under plain `cargo test` — no GPU, no DOM, no browser. The GPU render
//! pipeline and the GPU-compute mass-simulation path are added later behind a
//! `cfg(target_arch = "wasm32")` gate so this core stays native-testable.
//!
//! The design targets 1M+ vehicles through a two-layer model (see
//! `../README.md`):
//!   * a global mass layer (GPU compute, added later), and
//!   * a camera-local micro layer of continuous car-following behaviour —
//!     the [`sim`] module implemented here, which is also the CPU *reference*
//!     the GPU path is validated against.
//!
//! Per the current requirements, exact tick-for-tick replay is **not** a goal;
//! behavioural realism is. Reproducibility is nonetheless kept as a cheap
//! debugging affordance via a counter-based RNG (`sim::rng`) plus a fixed
//! timestep (`sim::clock`), and correctness is asserted **statistically** —
//! the flagship test reproduces the empirical flow/density *fundamental
//! diagram* rather than any golden byte stream.

pub mod render;
pub mod sim;

#[cfg(target_arch = "wasm32")]
mod bridge;

#[cfg(target_arch = "wasm32")]
pub use bridge::Simulation;

#[cfg(target_arch = "wasm32")]
pub use render::gpu::Renderer;

// Re-export so wasm-bindgen emits the `initThreadPool` JS entry point the browser
// awaits to spin up the rayon worker pool (the `wasm-threads` build only).
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
pub use wasm_bindgen_rayon::init_thread_pool;
