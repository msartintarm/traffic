//! Pure simulation core: no GPU, no DOM, no browser. Native `cargo test`
//! exercises every model here, and this CPU path is the reference the future
//! GPU-compute mass layer is validated against.

pub mod clock;
pub mod config;
pub mod constraint;
pub mod demand;
pub mod idm;
pub mod map;
pub mod meso;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod meso_gpu;
pub mod net_world;
pub mod network;
pub mod rng;
pub mod signal;
pub mod vehicle;
pub mod world;

pub use clock::{PlayState, SimClock};
pub use config::{DriverConfig, SimConfig};
pub use demand::{DemandGenerator, OdPair};
pub use map::OsmMap;
pub use net_world::{NetVehicle, NetWorld};
pub use network::Network;
pub use signal::{SignalProgram, SignalState};
pub use vehicle::Vehicle;
pub use world::World;
