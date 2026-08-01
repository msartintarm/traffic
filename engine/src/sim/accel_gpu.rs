//! Optional GPU execution of the per-vehicle binding acceleration: the `accel.wgsl`
//! kernel run on a real `wgpu` device. Enabled by the `gpu` feature and validated
//! against the CPU reference (`net_world::VehicleContext::binding`). Software adapters
//! (Mesa lavapipe) work, so it runs headless in CI. The kernel computes only the
//! deterministic constraint fold in f32; `accel_noise` is added on the CPU afterwards
//! (its SplitMix64 RNG is u64, absent from WGSL), so GPU and CPU agree only within
//! f32 tolerance, not bit-for-bit.

use bytemuck::cast_slice;

use super::net_world::VehicleContextGpu;

/// Binding acceleration (no noise) for each context, computed on the GPU. `None` if
/// no adapter is available. One-shot: acquires a device, dispatches, blocks on
/// readback — the native validation path (the browser will drive a persistent,
/// async version when the accel kernel is wired into the step there).
pub fn binding_accels_gpu(contexts: &[VehicleContextGpu]) -> Option<Vec<f32>> {
    let n = contexts.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(request_adapter(&instance))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("accel"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    use wgpu::util::DeviceExt;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let ctx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contexts"),
        contents: cast_slice(contexts),
        usage: storage,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_accel"),
        size: (n * 4) as u64,
        usage: storage,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("accel.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("accel.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ctx_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64).max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (n * 4) as u64);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait);
    let out: Vec<f32> = cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    Some(out)
}

/// Persistent GPU solver for the per-vehicle binding fold, driven every step. Holds
/// the device, pipeline and a set of working buffers reused via `write_buffer`
/// (reallocated only when the vehicle count outgrows them), so the hot path is one
/// dispatch + one blocking readback with no per-step allocation. Native only: the
/// browser cannot block on the GPU, and the accel result is needed within the same
/// step (unlike routing, which may lag a frame), so there is no async variant.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub struct GpuAccel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Current buffer capacity in contexts; grown (never shrunk) as the fleet does.
    cap: usize,
    work: Option<AccelWork>,
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
struct AccelWork {
    ctx_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
impl GpuAccel {
    /// Build a solver on a freshly-acquired native device (Vulkan/GL, software
    /// adapters included). `None` if no adapter is available.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(request_adapter(&instance))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("accel-batch"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accel.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("accel.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Self { device, queue, pipeline, layout, cap: 0, work: None })
    }

    /// (Re)allocate the working buffers when the vehicle count outgrows them.
    fn ensure_work(&mut self, n: usize) {
        if self.work.is_some() && n <= self.cap {
            return;
        }
        let cap = n.next_power_of_two().max(64);
        let ctx_stride = std::mem::size_of::<VehicleContextGpu>() as u64;
        let ctx_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contexts"),
            size: cap as u64 * ctx_stride,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_accel"),
            size: (cap * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (cap * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ctx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        });
        self.cap = cap;
        self.work = Some(AccelWork { ctx_buf, out_buf, staging, bind });
    }

    /// The binding acceleration (no noise) for each context. **Blocking** readback —
    /// native only; the caller adds the CPU-side noise term afterwards.
    pub fn binding_accels(&mut self, contexts: &[VehicleContextGpu]) -> Vec<f32> {
        let n = contexts.len();
        if n == 0 {
            return Vec::new();
        }
        self.ensure_work(n);
        let w = self.work.as_ref().unwrap();
        self.queue.write_buffer(&w.ctx_buf, 0, cast_slice(contexts));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &w.bind, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64).max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&w.out_buf, 0, &w.staging, 0, (n * 4) as u64);
        self.queue.submit([encoder.finish()]);

        let slice = w.staging.slice(..(n * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let out: Vec<f32> = cast_slice(&slice.get_mapped_range()).to_vec();
        w.staging.unmap();
        out
    }
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

async fn request_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force in [true, false] {
        let opts = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: force,
            compatible_surface: None,
        };
        if let Ok(a) = instance.request_adapter(&opts).await {
            return Some(a);
        }
    }
    None
}
