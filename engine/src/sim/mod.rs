//! Pure simulation core: no GPU, no DOM, no browser. Native `cargo test`
//! exercises every model here.

pub mod boundary;
pub mod clock;
pub mod config;
pub mod constraint;
pub mod demand;
pub mod flowfield;
// Compiled whenever a wgpu device is available: natively under the `gpu` feature
// (Vulkan/GL), and always in the browser (WebGPU) so routing can run on the
// renderer's device. Device-creating helpers inside are further gated to native.
#[cfg(any(feature = "gpu", target_arch = "wasm32"))]
pub mod flowfield_gpu;
pub mod idm;
pub mod junction;
pub mod map;
pub mod meso;
pub mod mobil;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod meso_gpu;
// GPU execution of the per-vehicle binding acceleration (`accel.wgsl`), validated
// against the CPU reference. Native + `gpu` for now; the browser path is a follow-up.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod accel_gpu;
pub mod net_world;
pub mod network;
pub mod rng;
pub mod router;
pub mod signal;
pub mod vehicle;
pub mod world;

pub use clock::{PlayState, SimClock};
pub use config::{DriverConfig, SimConfig, VehicleClass};
pub use demand::{DemandGenerator, OdPair};
pub use map::OsmMap;
pub use net_world::{NetVehicle, NetWorld};
pub use network::Network;
pub use signal::{SignalProgram, SignalState};
pub use vehicle::Vehicle;
pub use world::World;
