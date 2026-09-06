//! WebGPU renderer (WebGL2 fallback), wasm32 only. Owns a wgpu device on the
//! page canvas, uploads the static world mesh once, and instance-draws vehicles
//! through `scene.wgsl` each frame with GPU-side prev→current interpolation.
//!
//! Driven from JS (the render loop lives in the host, not here): `Renderer.create`
//! is async (adapter/device acquisition), `set_world_mesh` uploads the baked
//! geometry once, and `render` is called per animation frame. Correctness of the
//! geometry/scene math it draws is covered by the pure `render::*` tests; this
//! layer is device plumbing that needs the browser to verify visually.

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::geometry::{parse_world_directory, DirBand, MeshTile};
use super::{scene, Instance, StaticVertex, Vertex};

const CLEAR: wgpu::Color = wgpu::Color { r: 0.043, g: 0.055, b: 0.075, a: 1.0 };

/// Lane markings vanish into sub-pixel noise when zoomed out; only draw them once
/// the view is closer than this (world metres per pixel).
const MARKING_MAX_MPP: f32 = 0.7;

const VERTEX_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3, 2 => Float32];
const STATIC_ATTRS: [wgpu::VertexAttribute; 6] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x3, 3 => Float32, 4 => Float32, 5 => Float32];
const INSTANCE_ATTRS: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
    3 => Float32x2, 4 => Float32x2, 5 => Float32x2, 6 => Float32x2, 7 => Float32x3, 8 => Float32, 9 => Float32, 10 => Float32, 11 => Float32
];

#[wasm_bindgen]
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    static_pipeline: wgpu::RenderPipeline,
    instanced_pipeline: wgpu::RenderPipeline,
    cam_buf: wgpu::Buffer,
    cam_bind_group: wgpu::BindGroup,
    car_vbuf: wgpu::Buffer,
    car_ibuf: wgpu::Buffer,
    car_index_count: u32,
    inst_buf: wgpu::Buffer,
    inst_capacity: u64,
    signal_vbuf: wgpu::Buffer,
    signal_ibuf: wgpu::Buffer,
    signal_index_count: u32,
    signal_inst_buf: wgpu::Buffer,
    signal_inst_capacity: u64,
    crash_inst_buf: wgpu::Buffer,
    crash_inst_capacity: u64,
    density_vbuf: wgpu::Buffer,
    density_ibuf: wgpu::Buffer,
    density_vcap: u64,
    density_icap: u64,
    world: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    markings: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    /// Painter's-order render bands (grade layer, then road class) with per-band
    /// zoom cutoffs and spatial tile ranges — each band's fill then its markings,
    /// so overpasses layer over the roads they cross. Empty until `set_world_mesh`.
    bands: Vec<DirBand>,
    /// The static world cached as a texture: rendered over an expanded viewport
    /// only when the camera leaves the cached margin (or zooms, or the mesh /
    /// surface changes), then composited per frame as one fullscreen triangle —
    /// the whole road network costs three vertices while the camera rests.
    cache: Option<CacheLayer>,
    /// The view-projection the cache was rendered with; `None` = cache stale.
    cached_vp: Option<[f32; 16]>,
    /// Last frame's view-projection: the cache is only (re)built on a frame
    /// where the camera has stopped moving, so interaction frames pay the
    /// plain direct draw — never the margin-sized cache render.
    last_vp: Vec<f32>,
    cam_cache_buf: wgpu::Buffer,
    cam_cache_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_params_buf: wgpu::Buffer,
    blit_sampler: wgpu::Sampler,
}

/// Size-dependent half of the static-layer cache (rebuilt on resize).
struct CacheLayer {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    w: u32,
    h: u32,
}

/// How much wider than the viewport the cache is rendered, so small pans stay
/// inside it. Kept modest: the texture costs `MARGIN_K²`× the canvas in pixels.
const MARGIN_K: f32 = 1.3;

#[wasm_bindgen]
impl Renderer {
    pub async fn create(canvas: HtmlCanvasElement) -> Result<Renderer, JsValue> {
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(err)?;
        from_surface(instance, surface, width, height).await
    }

    /// The same renderer on an `OffscreenCanvas` — usable from a Web Worker (the page canvas is
    /// `transferControlToOffscreen`ed to it), so the map build and the render loop run off the
    /// main thread and never freeze the page.
    pub async fn create_offscreen(canvas: web_sys::OffscreenCanvas) -> Result<Renderer, JsValue> {
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance.create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas)).map_err(err)?;
        from_surface(instance, surface, width, height).await
    }
}

/// Device/pipeline setup shared by [`Renderer::create`] and [`Renderer::create_offscreen`] —
/// everything after the surface, which is the only part that differs between a page canvas and
/// an offscreen one.
async fn from_surface(
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    width: u32,
    height: u32,
) -> Result<Renderer, JsValue> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(err)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("traffic"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(err)?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
        });

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cam-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cam_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam-bg"),
            layout: &cam_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&cam_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRS,
        };
        let static_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<StaticVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &STATIC_ATTRS,
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRS,
        };

        let make_pipeline = |label: &str, vs: &str, buffers: &[wgpu::VertexBufferLayout]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let static_pipeline = make_pipeline("static", "vs_static", &[static_layout]);
        let instanced_pipeline =
            make_pipeline("instanced", "vs_instanced", &[vertex_layout, instance_layout]);

        let cam_cache_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-cache"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_cache_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam-cache-bg"),
            layout: &cam_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_cache_buf.as_entire_binding() }],
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit-pl"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_blit"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_blit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let blit_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blit-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let car = scene::unit_car_mesh();
        let car_vbuf = buffer_init(&device, "car-v", bytemuck::cast_slice(&car.vertices), wgpu::BufferUsages::VERTEX);
        let car_ibuf = buffer_init(&device, "car-i", bytemuck::cast_slice(&car.indices), wgpu::BufferUsages::INDEX);

        let inst_stride = std::mem::size_of::<Instance>() as u64;
        let inst_capacity = 4096 * inst_stride;
        let inst_buf = instance_buffer(&device, inst_capacity);

        let signal = scene::signal_head_mesh();
        let signal_vbuf = buffer_init(&device, "sig-v", bytemuck::cast_slice(&signal.vertices), wgpu::BufferUsages::VERTEX);
        let signal_ibuf = buffer_init(&device, "sig-i", bytemuck::cast_slice(&signal.indices), wgpu::BufferUsages::INDEX);
        let signal_inst_capacity = 256 * inst_stride;
        let signal_inst_buf = instance_buffer(&device, signal_inst_capacity);
        let crash_inst_capacity = 256 * inst_stride;
        let crash_inst_buf = instance_buffer(&device, crash_inst_capacity);

        let density_vcap = 8192 * std::mem::size_of::<StaticVertex>() as u64;
        let density_vbuf = instance_buffer(&device, density_vcap);
        let density_icap = 8192 * 4;
        let density_ibuf = index_buffer(&device, density_icap);

        Ok(Renderer {
            surface,
            device,
            queue,
            config,
            static_pipeline,
            instanced_pipeline,
            cam_buf,
            cam_bind_group,
            car_vbuf,
            car_ibuf,
            car_index_count: car.indices.len() as u32,
            inst_buf,
            inst_capacity,
            signal_vbuf,
            signal_ibuf,
            signal_index_count: signal.indices.len() as u32,
            signal_inst_buf,
            signal_inst_capacity,
            crash_inst_buf,
            crash_inst_capacity,
            density_vbuf,
            density_ibuf,
            density_vcap,
            density_icap,
            world: None,
            markings: None,
            bands: Vec::new(),
            cache: None,
            cached_vp: None,
            last_vp: Vec::new(),
            cam_cache_buf,
            cam_cache_bind_group,
            blit_pipeline,
            blit_bgl,
            blit_params_buf,
            blit_sampler,
        })
}

#[wasm_bindgen]
impl Renderer {
    /// Upload the baked static geometry once. Roads+junctions (`world_*`) draw at
    /// every zoom; markings (`mark_*`) only when zoomed in. Both are flat
    /// `StaticVertex` arrays (center.xy, offset.xy, color.rgb, light).
    pub fn set_world_mesh(&mut self, world_v: Vec<f32>, world_i: Vec<u32>, mark_v: Vec<f32>, mark_i: Vec<u32>, bands: Vec<u32>) {
        self.world = Some(self.upload_mesh("world", &world_v, &world_i));
        self.markings = Some(self.upload_mesh("markings", &mark_v, &mark_i));
        // Tiled band directory (or the legacy flat ranges — both parse). Fall
        // back to one whole-buffer band if the caller supplied none.
        self.bands = parse_world_directory(&bands);
        if self.bands.is_empty() {
            const ALL: [f32; 4] = [f32::MIN, f32::MIN, f32::MAX, f32::MAX];
            self.bands = vec![DirBand {
                max_mpp: 0.0,
                fill_tiles: vec![MeshTile { start: 0, count: world_i.len() as u32, bbox: ALL }],
                mark_tiles: vec![MeshTile { start: 0, count: mark_i.len() as u32, bbox: ALL }],
            }];
        }
        self.cached_vp = None;
    }

    fn upload_mesh(&self, label: &str, vertices: &[f32], indices: &[u32]) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let vbuf = buffer_init(&self.device, label, bytemuck::cast_slice(vertices), wgpu::BufferUsages::VERTEX);
        let ibuf = buffer_init(&self.device, label, bytemuck::cast_slice(indices), wgpu::BufferUsages::INDEX);
        (vbuf, ibuf, indices.len() as u32)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.cached_vp = None;
        }
    }

    /// Draw a frame. `view_proj` is a 16-element column-major matrix, `alpha` the
    /// sub-tick blend, `instances` the raw `Instance` bytes, `count` the vehicle
    /// count.
    pub fn render(
        &mut self,
        view_proj: Vec<f32>,
        alpha: f32,
        mpp: f32,
        instances: Vec<u8>,
        count: u32,
        signals: Vec<u8>,
        signal_count: u32,
        crashes: Vec<u8>,
        crash_count: u32,
        density_v: Vec<f32>,
        density_i: Vec<u32>,
    ) {
        let mut cam = [0f32; 20];
        cam[..16].copy_from_slice(&view_proj);
        cam[16] = alpha;
        cam[17] = mpp;
        self.queue.write_buffer(&self.cam_buf, 0, bytemuck::cast_slice(&cam));

        if !instances.is_empty() {
            if instances.len() as u64 > self.inst_capacity {
                self.inst_capacity = (instances.len() as u64).next_power_of_two();
                self.inst_buf = instance_buffer(&self.device, self.inst_capacity);
            }
            self.queue.write_buffer(&self.inst_buf, 0, &instances);
        }
        if !signals.is_empty() {
            if signals.len() as u64 > self.signal_inst_capacity {
                self.signal_inst_capacity = (signals.len() as u64).next_power_of_two();
                self.signal_inst_buf = instance_buffer(&self.device, self.signal_inst_capacity);
            }
            self.queue.write_buffer(&self.signal_inst_buf, 0, &signals);
        }
        if !crashes.is_empty() {
            if crashes.len() as u64 > self.crash_inst_capacity {
                self.crash_inst_capacity = (crashes.len() as u64).next_power_of_two();
                self.crash_inst_buf = instance_buffer(&self.device, self.crash_inst_capacity);
            }
            self.queue.write_buffer(&self.crash_inst_buf, 0, &crashes);
        }
        let density_count = density_i.len() as u32;
        if density_count > 0 {
            let vbytes: &[u8] = bytemuck::cast_slice(&density_v);
            if vbytes.len() as u64 > self.density_vcap {
                self.density_vcap = (vbytes.len() as u64).next_power_of_two();
                self.density_vbuf = instance_buffer(&self.device, self.density_vcap);
            }
            self.queue.write_buffer(&self.density_vbuf, 0, vbytes);
            let ibytes: &[u8] = bytemuck::cast_slice(&density_i);
            if ibytes.len() as u64 > self.density_icap {
                self.density_icap = (ibytes.len() as u64).next_power_of_two();
                self.density_ibuf = index_buffer(&self.device, self.density_icap);
            }
            self.queue.write_buffer(&self.density_ibuf, 0, ibytes);
        }

        // ---- static-layer cache maintenance ----
        // Recreate the cache texture on resize; even margin so the fresh cache
        // samples land exactly on texel centres (the border is whole pixels).
        let limit = self.device.limits().max_texture_dimension_2d;
        let margin = |px: u32| ((px as f32 * (MARGIN_K - 1.0)).ceil() as u32 + 1) & !1u32;
        let cw = (self.config.width + margin(self.config.width)).min(limit);
        let ch = (self.config.height + margin(self.config.height)).min(limit);
        if self.cache.as_ref().is_none_or(|c| c.w != cw || c.h != ch) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("world-cache"),
                size: wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit-bg"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.blit_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
                ],
            });
            self.cache = Some(CacheLayer { view, bind_group, w: cw, h: ch });
            self.cached_vp = None;
        }
        // The cache is fresh while the live view still maps inside it at the
        // same zoom: `a·ndc + b` sends frame NDC into cache NDC (the camera is a
        // pure scale + translate), so the corners staying in [-1, 1] is exactly
        // "the viewport is inside the cached margin".
        let (kx, ky) = (cw as f32 / self.config.width as f32, ch as f32 / self.config.height as f32);
        let map_to_cache = |c: &[f32; 16]| {
            let (ax, ay) = (c[0] / view_proj[0], c[5] / view_proj[5]);
            (ax, ay, c[12] - ax * view_proj[12], c[13] - ay * view_proj[13])
        };
        let fresh = self.cached_vp.as_ref().map(map_to_cache).is_some_and(|(ax, ay, bx, by)| {
            (ax * kx - 1.0).abs() < 1e-3
                && (ay * ky - 1.0).abs() < 1e-3
                && ax.abs() + bx.abs() <= 1.0
                && ay.abs() + by.abs() <= 1.0
        });
        // Only build the cache once the camera has settled: while it moves,
        // frames draw the world directly (the pre-cache path, culled to the
        // viewport), so interaction never pays the margin-sized cache render.
        let build_cache = !fresh && self.last_vp == view_proj;
        if build_cache {
            let mut cvp = [0f32; 16];
            cvp.copy_from_slice(&view_proj);
            for i in [0usize, 4, 8, 12] {
                cvp[i] /= kx;
            }
            for i in [1usize, 5, 9, 13] {
                cvp[i] /= ky;
            }
            let mut ccam = [0f32; 20];
            ccam[..16].copy_from_slice(&cvp);
            ccam[16] = alpha;
            ccam[17] = mpp;
            self.queue.write_buffer(&self.cam_cache_buf, 0, bytemuck::cast_slice(&ccam));
            self.cached_vp = Some(cvp);
        }
        let use_cache = fresh || build_cache;
        if use_cache {
            let cvp = self.cached_vp.expect("fresh or just built");
            let (ax, ay, bx, by) = map_to_cache(&cvp);
            // Snap the pan offset to whole cache texels: the static layer stays
            // crisp while dragging (at most half a pixel of skew against the
            // live layers) instead of going soft under linear filtering.
            let bx = (bx * cw as f32 * 0.5).round() / (cw as f32 * 0.5);
            let by = (by * ch as f32 * 0.5).round() / (ch as f32 * 0.5);
            self.queue.write_buffer(&self.blit_params_buf, 0, bytemuck::cast_slice(&[ax, ay, bx, by]));
        }
        self.last_vp = view_proj.clone();

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // World-space rect a view-projection covers, for tile culling.
        let rect_of = |m: &[f32]| {
            let span = |sc: f32, t: f32| {
                let (a, b) = ((-1.0 - t) / sc, (1.0 - t) / sc);
                (a.min(b), a.max(b))
            };
            let (x0, x1) = span(m[0], m[12]);
            let (y0, y1) = span(m[5], m[13]);
            [x0, y0, x1, y1]
        };
        // Painter's-order bands culled to `rect`: each band's fill then its
        // markings (zoom-gated), so overpasses layer over what they cross.
        let zoomed_in = mpp < MARKING_MAX_MPP;
        let draw_world = |pass: &mut wgpu::RenderPass<'_>, bands: &[DirBand], rect: [f32; 4]| {
            if let (Some((wv, wi, _)), Some((mv, mi, _))) = (&self.world, &self.markings) {
                pass.set_pipeline(&self.static_pipeline);
                for band in bands {
                    if !band.fill_tiles.is_empty() {
                        pass.set_vertex_buffer(0, wv.slice(..));
                        pass.set_index_buffer(wi.slice(..), wgpu::IndexFormat::Uint32);
                        draw_visible_tiles(pass, &band.fill_tiles, rect);
                    }
                    if zoomed_in && !band.mark_tiles.is_empty() {
                        pass.set_vertex_buffer(0, mv.slice(..));
                        pass.set_index_buffer(mi.slice(..), wgpu::IndexFormat::Uint32);
                        draw_visible_tiles(pass, &band.mark_tiles, rect);
                    }
                }
            }
        };
        if build_cache && self.world.is_some() {
            let cvp = self.cached_vp.expect("just built");
            let cache = self.cache.as_ref().expect("created above");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world-cache"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cache.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(CLEAR), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.cam_cache_bind_group, &[]);
            draw_world(&mut pass, &self.bands, rect_of(&cvp));
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(CLEAR), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // At rest the whole static world is the cached texture, composited
            // as one fullscreen triangle; while the camera moves it draws
            // directly (viewport-culled), same as before the cache existed.
            if use_cache && self.world.is_some() {
                let cache = self.cache.as_ref().expect("ensured above");
                pass.set_pipeline(&self.blit_pipeline);
                pass.set_bind_group(0, &cache.bind_group, &[]);
                pass.draw(0..3, 0..1);
                pass.set_bind_group(0, &self.cam_bind_group, &[]);
            } else {
                pass.set_bind_group(0, &self.cam_bind_group, &[]);
                draw_world(&mut pass, &self.bands, rect_of(&view_proj));
            }
            pass.set_pipeline(&self.static_pipeline);

            // The occupancy tint is a translucent overlay on top of the layered
            // world (it and the opaque markings blend, both readable).
            if density_count > 0 {
                pass.set_vertex_buffer(0, self.density_vbuf.slice(..));
                pass.set_index_buffer(self.density_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..density_count, 0, 0..1);
            }

            if count > 0 {
                pass.set_pipeline(&self.instanced_pipeline);
                pass.set_vertex_buffer(0, self.car_vbuf.slice(..));
                pass.set_vertex_buffer(1, self.inst_buf.slice(..));
                pass.set_index_buffer(self.car_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.car_index_count, 0, 0..count);
            }

            if signal_count > 0 {
                pass.set_pipeline(&self.instanced_pipeline);
                pass.set_vertex_buffer(0, self.signal_vbuf.slice(..));
                pass.set_vertex_buffer(1, self.signal_inst_buf.slice(..));
                pass.set_index_buffer(self.signal_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.signal_index_count, 0, 0..signal_count);
            }

            // Crash markers reuse the signal emissive-square mesh (a diamond via per-instance
            // rotation) but are drawn at every zoom — collision hot-spots stay visible fitted out.
            if crash_count > 0 {
                pass.set_pipeline(&self.instanced_pipeline);
                pass.set_vertex_buffer(0, self.signal_vbuf.slice(..));
                pass.set_vertex_buffer(1, self.crash_inst_buf.slice(..));
                pass.set_index_buffer(self.signal_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.signal_index_count, 0, 0..crash_count);
            }
        }
        self.queue.submit([encoder.finish()]);
        frame.present();
    }
}

// Not `#[wasm_bindgen]`: a Rust-only accessor so the sim can build a GPU compute
// context on the *same* WebGPU device the renderer already owns (device/queue are
// cheap Arc handles), sharing one context between rendering and flow-field compute.
impl Renderer {
    pub(crate) fn device_queue(&self) -> (wgpu::Device, wgpu::Queue) {
        (self.device.clone(), self.queue.clone())
    }
}

fn buffer_init(device: &wgpu::Device, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
}

fn instance_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indices"),
        size,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Draw the tiles of one band that intersect `rect` (`[min_x, min_y, max_x,
/// max_y]`, world m), coalescing contiguous visible runs into single draws — a
/// fully visible band collapses back to one `draw_indexed`.
fn draw_visible_tiles(pass: &mut wgpu::RenderPass<'_>, tiles: &[MeshTile], rect: [f32; 4]) {
    let mut run: Option<(u32, u32)> = None;
    for t in tiles {
        let visible = t.count > 0
            && t.bbox[0] <= rect[2]
            && t.bbox[2] >= rect[0]
            && t.bbox[1] <= rect[3]
            && t.bbox[3] >= rect[1];
        if !visible {
            continue;
        }
        run = Some(match run {
            Some((s, e)) if e == t.start => (s, t.start + t.count),
            Some((s, e)) => {
                pass.draw_indexed(s..e, 0, 0..1);
                (t.start, t.start + t.count)
            }
            None => (t.start, t.start + t.count),
        });
    }
    if let Some((s, e)) = run {
        pass.draw_indexed(s..e, 0, 0..1);
    }
}

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}
