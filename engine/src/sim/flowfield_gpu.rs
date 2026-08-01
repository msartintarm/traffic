//! Optional GPU execution of flow-field routing: the `flowfield.wgsl` Bellman-Ford
//! relaxation run on a real `wgpu` device with ping-pong distance buffers.
//! Enabled by the `gpu` feature; validated against the CPU
//! [`super::flowfield::distances_to`] reference. Software adapters (Mesa
//! lavapipe) work, so it runs headless in CI.

use bytemuck::cast_slice;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    link_count: u32,
    dest: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Reverse shortest-path distances (ms, `u32::MAX` = unreachable) from every
/// link to `dest`, computed on the GPU. `None` if no adapter is available.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn distances_to_gpu(offsets: &[u32], targets: &[u32], cost: &[u32], dest: u32) -> Option<Vec<u32>> {
    let n = cost.len();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(request_adapter(&instance))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flowfield"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(), // needs 5 storage buffers/stage
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    use wgpu::util::DeviceExt;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let init: Vec<u32> = (0..n).map(|i| if i as u32 == dest { 0 } else { u32::MAX }).collect();

    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&Params { link_count: n as u32, dest, _pad0: 0, _pad1: 0 }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(offsets), usage: storage });
    let targets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(if targets.is_empty() { &[0u32] } else { targets }), usage: storage });
    let cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(cost), usage: storage });
    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(&init), usage: storage });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(&init), usage: storage });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flowfield.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("flowfield.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
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

    let bind = |din: &wgpu::Buffer, dout: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: targets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cost_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: din.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dout.as_entire_binding() },
            ],
        })
    };
    let bind_groups = [bind(&buf_a, &buf_b), bind(&buf_b, &buf_a)];

    let groups = (n as u32).div_ceil(64).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut parity = 0usize;
    for _ in 0..n {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_groups[parity], &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        parity ^= 1;
    }
    let latest = if parity == 0 { &buf_a } else { &buf_b };
    encoder.copy_buffer_to_buffer(latest, 0, &staging, 0, (n * 4) as u64);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait);
    let data = slice.get_mapped_range();
    let out: Vec<u32> = cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Some(out)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchParams {
    link_count: u32,
    slot_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Persistent GPU solver for many flow fields at once. Holds the device, pipelines
/// and static CSR-adjacency buffers, plus per-recompute working buffers reused via
/// `write_buffer` (reallocated only when the destination count changes). One
/// dispatch stream relaxes every destination field (`slot_count × link_count`
/// threads) in place, gating itself to a stop once the fields stop changing.
pub struct GpuFlowField {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The `relax` entry point (`flowfield_batch.wgsl`) — one edge-relaxation pass.
    relax: wgpu::ComputePipeline,
    /// The `gate` entry point — latches convergence and resets the changed flag.
    gate: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    offsets_buf: wgpu::Buffer,
    targets_buf: wgpu::Buffer,
    link_count: usize,
    /// Working buffers, (re)allocated only when the destination count changes.
    work: Option<Work>,
}

/// The per-recompute buffers, sized for `slot_count` destinations. Persisted so a
/// recompute writes into them (`queue.write_buffer`) rather than allocating anew.
/// `dist` is relaxed in place (`atomicMin`), so there is no second ping-pong buffer;
/// `flags` is the two-word `[changed, done]` convergence state.
struct Work {
    slot_count: usize,
    params: wgpu::Buffer,
    cost_buf: wgpu::Buffer,
    dests_buf: wgpu::Buffer,
    dist: wgpu::Buffer,
    flags: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl GpuFlowField {
    /// Build a solver on a freshly-acquired native device (Vulkan/GL, software
    /// adapters included). `None` if no adapter is available. The browser instead
    /// injects its render device with [`from_device`](Self::from_device).
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn new(offsets: &[u32], targets: &[u32]) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(request_adapter(&instance))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flowfield-batch"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some(Self::from_device(device, queue, offsets, targets))
    }

    /// Build a solver on an existing device/queue — the browser shares the
    /// renderer's WebGPU device this way, so compute and rendering run on one
    /// context (no second adapter, no headers).
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, offsets: &[u32], targets: &[u32]) -> Self {
        use wgpu::util::DeviceExt;
        let link_count = offsets.len().saturating_sub(1);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(if offsets.is_empty() { &[0u32] } else { offsets }), usage: storage });
        let targets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_slice(if targets.is_empty() { &[0u32] } else { targets }), usage: storage });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flowfield_batch.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("flowfield_batch.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(5, wgpu::BufferBindingType::Storage { read_only: false }), // dist (in-place)
                entry(6, wgpu::BufferBindingType::Storage { read_only: false }), // flags
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let compute = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let (relax, gate) = (compute("relax"), compute("gate"));
        Self { device, queue, relax, gate, layout, offsets_buf, targets_buf, link_count, work: None }
    }

    /// (Re)allocate the working buffers when the destination count changes.
    fn ensure_work(&mut self, slot_count: usize) {
        if self.work.as_ref().is_some_and(|w| w.slot_count == slot_count) {
            return;
        }
        let l = self.link_count;
        let total = (l * slot_count).max(1);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buf = |size: usize, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (size * 4).max(4) as u64, usage, mapped_at_creation: false });
        let params = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("batch-params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let cost_buf = buf(l.max(1), storage);
        let dests_buf = buf(slot_count.max(1), storage);
        let dist = buf(total, storage);
        let flags = buf(2, storage);
        let staging = buf(total, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.targets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cost_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dests_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dist.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: flags.as_entire_binding() },
            ],
        });
        self.work = Some(Work { slot_count, params, cost_buf, dests_buf, dist, flags, staging, bind });
    }

    /// Encode and submit the relaxation for `cost`/`dests` into the working
    /// buffers. `dist` is relaxed in place, so the result is always in `w.dist`
    /// (no parity to track). Shared by the blocking and async readback paths.
    fn dispatch(&mut self, cost: &[u32], dests: &[u32]) {
        let (l, k) = (self.link_count, dests.len());
        self.ensure_work(k);
        let w = self.work.as_ref().unwrap();
        let total = l * k;
        let init: Vec<u32> = (0..total).map(|g| if (g % l) as u32 == dests[g / l] { 0 } else { u32::MAX }).collect();
        self.queue.write_buffer(&w.params, 0, bytemuck::bytes_of(&BatchParams { link_count: l as u32, slot_count: k as u32, _pad0: 0, _pad1: 0 }));
        self.queue.write_buffer(&w.cost_buf, 0, cast_slice(cost));
        self.queue.write_buffer(&w.dests_buf, 0, cast_slice(dests));
        self.queue.write_buffer(&w.dist, 0, cast_slice(&init));
        self.queue.write_buffer(&w.flags, 0, cast_slice(&[0u32, 0u32])); // [changed, done]

        let groups = (total as u32).div_ceil(64).max(1);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // `link_count` is Bellman-Ford's worst-case bound; the `gate` pass latches
        // convergence, so `relax` passes past the fixed point cost nothing (they read
        // the `done` flag and return without touching `dist`).
        for _ in 0..l.max(1) {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&self.relax);
                pass.set_bind_group(0, &w.bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&self.gate);
                pass.set_bind_group(0, &w.bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&w.dist, 0, &w.staging, 0, (total * 4) as u64);
        self.queue.submit([encoder.finish()]);
    }

    /// Reverse distances for every destination in `dests`, one `Vec` per
    /// destination (`u32::MAX` = unreachable). **Blocking** readback — native only
    /// (the browser cannot block; it drives the async path instead).
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn distances(&mut self, cost: &[u32], dests: &[u32]) -> Vec<Vec<u32>> {
        let (l, k) = (self.link_count, dests.len());
        if l == 0 || k == 0 {
            return vec![Vec::new(); k];
        }
        self.dispatch(cost, dests);
        let staging = &self.work.as_ref().unwrap().staging;
        let slice = staging.slice(..(l * k * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let all: Vec<u32> = cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        all.chunks(l).map(<[u32]>::to_vec).collect()
    }
}

/// An in-flight async readback: the GPU work is submitted and the staging buffer
/// is being mapped; `ready` flips when the browser reports the map complete.
#[cfg(target_arch = "wasm32")]
pub struct PendingReadback {
    ready: std::rc::Rc<std::cell::Cell<bool>>,
    slot_count: usize,
}

/// Browser (async) readback: the GPU cannot be blocked on, so a recompute is
/// *dispatched* on one frame and *collected* on a later one.
#[cfg(target_arch = "wasm32")]
impl GpuFlowField {
    /// Dispatch a recompute and begin mapping the result. Poll [`try_take`](Self::try_take)
    /// on later frames. Non-blocking.
    pub fn begin_readback(&mut self, cost: &[u32], dests: &[u32]) -> PendingReadback {
        let k = dests.len();
        self.dispatch(cost, dests);
        let ready = std::rc::Rc::new(std::cell::Cell::new(false));
        let flag = ready.clone();
        let bytes = (self.link_count * k * 4) as u64;
        let staging = &self.work.as_ref().unwrap().staging;
        staging.slice(..bytes.max(4)).map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                flag.set(true);
            }
        });
        PendingReadback { ready, slot_count: k }
    }

    /// If the pending map has completed, read the distance fields and unmap the
    /// staging buffer (freeing it for the next dispatch); else `None`.
    pub fn try_take(&self, pending: &PendingReadback) -> Option<Vec<Vec<u32>>> {
        let _ = self.device.poll(wgpu::PollType::Poll); // pump; a no-op driver on the web
        if !pending.ready.get() {
            return None;
        }
        let l = self.link_count;
        let bytes = (l * pending.slot_count * 4) as u64;
        let staging = &self.work.as_ref().unwrap().staging;
        let all: Vec<u32> = cast_slice(&staging.slice(..bytes.max(4)).get_mapped_range()).to_vec();
        staging.unmap();
        Some(all.chunks(l.max(1)).map(<[u32]>::to_vec).collect())
    }
}

/// Recompute a [`FieldRouter`]'s next-hop fields on the GPU in one batched
/// dispatch and feed the reverse distances back, **reusing** a persistent
/// `solver` (built once over the network's adjacency). Blocking — native path.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn recompute_router_gpu(solver: &mut GpuFlowField, router: &mut super::router::FieldRouter, cost: &[u64]) -> bool {
    let cost32: Vec<u32> = cost.iter().map(|&c| c.min(u32::MAX as u64) as u32).collect();
    let dests: Vec<u32> = router.dests_in_slot_order().iter().map(|d| d.0).collect();
    let dist_per_slot: Vec<Vec<u64>> = solver
        .distances(&cost32, &dests)
        .into_iter()
        .map(|f| f.into_iter().map(|d| if d == u32::MAX { super::flowfield::UNREACHABLE } else { d as u64 }).collect())
        .collect();
    router.recompute_from_distances(cost, &dist_per_slot);
    true
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
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

#[cfg(test)]
mod tests {
    use super::super::flowfield::{adjacency, csr, distances_to, UNREACHABLE};
    use super::super::map::{LinkSpec, NodeSpec, OsmMap};
    use super::super::network::LinkId;
    use super::*;

    #[test]
    fn gpu_distances_match_the_cpu_reference() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, -10.0),
                NodeSpec::uncontrolled(3, 200.0, 300.0),
                NodeSpec::uncontrolled(4, 300.0, 0.0),
                NodeSpec::uncontrolled(5, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(0, 1, 1, 20.0),
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(1, 3, 1, 20.0),
                LinkSpec::oneway(2, 4, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
                LinkSpec::oneway(4, 5, 1, 20.0),
            ],
        }
        .build();
        let adj = adjacency(&net);
        let (offsets, targets) = csr(&adj);
        let cost64: Vec<u64> = (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect();
        let cost32: Vec<u32> = cost64.iter().map(|&c| c as u32).collect();
        let dest = 5u32;

        let cpu = distances_to(&adj, LinkId(dest), &cost64);
        let Some(gpu) = distances_to_gpu(&offsets, &targets, &cost32, dest) else {
            eprintln!("no GPU adapter; skipping flow-field GPU/CPU equivalence test");
            return;
        };
        for (i, (&g, &c)) in gpu.iter().zip(&cpu).enumerate() {
            let c32 = if c == UNREACHABLE { u32::MAX } else { c as u32 };
            assert_eq!(g, c32, "link {i}: gpu {g} != cpu {c32}");
        }
    }

    #[test]
    fn batched_gpu_distances_match_cpu_for_every_destination() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, -10.0),
                NodeSpec::uncontrolled(3, 200.0, 300.0),
                NodeSpec::uncontrolled(4, 300.0, 0.0),
                NodeSpec::uncontrolled(5, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(0, 1, 1, 20.0),
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(1, 3, 1, 20.0),
                LinkSpec::oneway(2, 4, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
                LinkSpec::oneway(4, 5, 1, 20.0),
            ],
        }
        .build();
        let adj = adjacency(&net);
        let (offsets, targets) = csr(&adj);
        let cost64: Vec<u64> = (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect();
        let cost32: Vec<u32> = cost64.iter().map(|&c| c as u32).collect();
        let dests = [4u32, 5u32];

        let Some(mut solver) = GpuFlowField::new(&offsets, &targets) else {
            eprintln!("no GPU adapter; skipping batched flow-field test");
            return;
        };
        let batched = solver.distances(&cost32, &dests);
        for (slot, &dest) in dests.iter().enumerate() {
            let cpu = distances_to(&adj, LinkId(dest), &cost64);
            for (i, &g) in batched[slot].iter().enumerate() {
                let c32 = if cpu[i] == UNREACHABLE { u32::MAX } else { cpu[i] as u32 };
                assert_eq!(g, c32, "dest {dest} link {i}: batched gpu {g} != cpu {c32}");
            }
        }
    }

    #[test]
    fn gpu_backed_router_matches_the_cpu_router() {
        use super::super::router::FieldRouter;
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, -10.0),
                NodeSpec::uncontrolled(3, 200.0, 300.0),
                NodeSpec::uncontrolled(4, 300.0, 0.0),
                NodeSpec::uncontrolled(5, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(0, 1, 1, 20.0),
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(1, 3, 1, 20.0),
                LinkSpec::oneway(2, 4, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
                LinkSpec::oneway(4, 5, 1, 20.0),
            ],
        }
        .build();
        let cost: Vec<u64> = (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect();
        let dests = [LinkId(5), LinkId(4)];
        let cpu = FieldRouter::new(&net, &dests, &cost);

        let adj = adjacency(&net);
        let (offsets, targets) = csr(&adj);
        let Some(mut solver) = GpuFlowField::new(&offsets, &targets) else {
            eprintln!("no GPU adapter; skipping GPU-backed router equivalence test");
            return;
        };
        let mut gpu = FieldRouter::new(&net, &dests, &cost);
        // Recompute twice to exercise the persistent working-buffer reuse path.
        recompute_router_gpu(&mut solver, &mut gpu, &cost);
        recompute_router_gpu(&mut solver, &mut gpu, &cost);
        for &dest in &dests {
            for from in 0..net.links.len() as u32 {
                assert_eq!(
                    gpu.next_hop(dest, LinkId(from)),
                    cpu.next_hop(dest, LinkId(from)),
                    "GPU router next hop to {dest:?} from {from} must match the CPU router"
                );
            }
        }
    }
}
