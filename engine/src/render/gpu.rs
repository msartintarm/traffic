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

use super::{scene, Instance, Vertex};

const CLEAR: wgpu::Color = wgpu::Color { r: 0.043, g: 0.055, b: 0.075, a: 1.0 };

const VERTEX_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3, 2 => Float32];
const INSTANCE_ATTRS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
    3 => Float32x2, 4 => Float32x2, 5 => Float32x2, 6 => Float32x3, 7 => Float32, 8 => Float32, 9 => Float32
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
    density_vbuf: wgpu::Buffer,
    density_capacity: u64,
    world: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
}

#[wasm_bindgen]
impl Renderer {
    pub async fn create(canvas: HtmlCanvasElement) -> Result<Renderer, JsValue> {
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(err)?;
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
        let static_pipeline = make_pipeline("static", "vs_static", &[vertex_layout.clone()]);
        let instanced_pipeline =
            make_pipeline("instanced", "vs_instanced", &[vertex_layout, instance_layout]);

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

        let density_capacity = 8192 * std::mem::size_of::<Vertex>() as u64;
        let density_vbuf = instance_buffer(&device, density_capacity);

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
            density_vbuf,
            density_capacity,
            world: None,
        })
    }

    /// Upload the baked road/junction/marking mesh once. `vertices` is the flat
    /// `Vertex` array (pos.xy, color.rgb, light) and `indices` the triangle list.
    pub fn set_world_mesh(&mut self, vertices: Vec<f32>, indices: Vec<u32>) {
        let vbuf = buffer_init(&self.device, "world-v", bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX);
        let ibuf = buffer_init(&self.device, "world-i", bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX);
        self.world = Some((vbuf, ibuf, indices.len() as u32));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Draw a frame. `view_proj` is a 16-element column-major matrix, `alpha` the
    /// sub-tick blend, `instances` the raw `Instance` bytes, `count` the vehicle
    /// count.
    pub fn render(
        &mut self,
        view_proj: Vec<f32>,
        alpha: f32,
        instances: Vec<u8>,
        count: u32,
        signals: Vec<u8>,
        signal_count: u32,
        density: Vec<f32>,
        density_count: u32,
    ) {
        let mut cam = [0f32; 20];
        cam[..16].copy_from_slice(&view_proj);
        cam[16] = alpha;
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
        if !density.is_empty() {
            let bytes: &[u8] = bytemuck::cast_slice(&density);
            if bytes.len() as u64 > self.density_capacity {
                self.density_capacity = (bytes.len() as u64).next_power_of_two();
                self.density_vbuf = instance_buffer(&self.device, self.density_capacity);
            }
            self.queue.write_buffer(&self.density_vbuf, 0, bytes);
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
            pass.set_bind_group(0, &self.cam_bind_group, &[]);

            if let Some((vbuf, ibuf, icount)) = &self.world {
                pass.set_pipeline(&self.static_pipeline);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*icount, 0, 0..1);
            }

            if density_count > 0 {
                pass.set_pipeline(&self.static_pipeline);
                pass.set_vertex_buffer(0, self.density_vbuf.slice(..));
                pass.draw(0..density_count, 0..1);
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
        }
        self.queue.submit([encoder.finish()]);
        frame.present();
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

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}
