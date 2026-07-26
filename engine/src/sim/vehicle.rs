use super::config::DriverConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vehicle {
    pub id: u32,
    pub position: f64,
    pub speed: f64,
    pub driver: DriverConfig,
}

impl Vehicle {
    pub fn new(id: u32, position: f64, speed: f64, driver: DriverConfig) -> Self {
        Self { id, position, speed, driver }
    }
}
