//! Optional GPU execution of the mass layer: the exact `meso.wgsl` kernel run on
//! a real `wgpu` device with ping-pong storage buffers. Enabled by the `gpu`
//! feature. Acquisition is fallible ([`MesoGpu::new`] returns `None` when no
//! adapter is available), so GPU is genuinely *optional* — callers fall back to
//! the CPU [`super::meso::MesoCorridor`] reference, which this is validated against.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::meso::CtmParams;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    cell_max: f32,
    capacity: f32,
    backward_ratio: f32,
    inflow_demand: f32,
    cell_count: u32,
    blocked_exit: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct MesoGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    buffers: [wgpu::Buffer; 2],
    staging: wgpu::Buffer,
    count: u32,
    byte_len: u64,
    parity: usize,
}

impl MesoGpu {
    pub fn new(params: CtmParams, inflow_demand: f64, blocked_exit: bool, cells: &[f32]) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(request_adapter(&instance))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("meso"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let count = cells.len() as u32;
        let byte_len = (cells.len() * std::mem::size_of::<f32>()) as u64;

        let gp = GpuParams {
            cell_max: params.cell_max as f32,
            capacity: params.capacity as f32,
            backward_ratio: params.backward_ratio as f32,
            inflow_demand: inflow_demand as f32,
            cell_count: count,
            blocked_exit: blocked_exit as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&gp),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cells_a"),
            contents: bytemuck::cast_slice(cells),
            usage: storage_usage,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells_b"),
            size: byte_len,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meso.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("meso.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meso-bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meso-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meso-pipe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let make_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("meso-bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                ],
            })
        };
        let bind_groups = [make_bg(&buf_a, &buf_b), make_bg(&buf_b, &buf_a)];

        Some(Self {
            device,
            queue,
            pipeline,
            bind_groups,
            buffers: [buf_a, buf_b],
            staging,
            count,
            byte_len,
            parity: 0,
        })
    }

    pub fn run(&mut self, ticks: u32) {
        let groups = self.count.div_ceil(64).max(1);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..ticks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.parity], &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.parity ^= 1;
        }
        self.queue.submit([encoder.finish()]);
    }

    /// The latest committed cell occupancies, copied back from the GPU.
    pub fn read(&self) -> Vec<f32> {
        let latest = &self.buffers[self.parity];
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(latest, 0, &self.staging, 0, self.byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();
        out
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

async fn request_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback in [true, false] {
        let opts = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: force_fallback,
            compatible_surface: None,
        };
        if let Ok(adapter) = instance.request_adapter(&opts).await {
            return Some(adapter);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::meso::{Boundary, MesoCorridor};
    use super::*;

    #[test]
    fn gpu_matches_the_cpu_reference() {
        let params = CtmParams::car_arterial(1.0);
        let demand = 0.42;
        let start: Vec<f32> = vec![0.3, 1.8, 0.0, 2.05, 0.7, 1.1, 0.2, 1.6, 0.4, 0.9];

        let Some(mut gpu) = MesoGpu::new(params, demand, false, &start) else {
            eprintln!("no GPU adapter available; skipping GPU/CPU equivalence test");
            return;
        };
        let ticks = 150;
        gpu.run(ticks);
        let gpu_cells = gpu.read();

        let mut cpu = MesoCorridor::new(
            params,
            Boundary::Open { inflow_demand: demand, blocked_exit: false },
            start.iter().map(|&x| x as f64).collect(),
        );
        cpu.run(ticks);

        for (i, (&g, &c)) in gpu_cells.iter().zip(cpu.cells()).enumerate() {
            assert!((g as f64 - c).abs() < 1e-3, "cell {i}: gpu {g} vs cpu {c}");
        }
    }
}
