// The wasm-bindgen surface the app consumes, plus the 2D-fallback scene shape. Extracted
// from the component so the engine session (worker or inline) and the 2D renderer share one
// definition of the `Simulation` / `Renderer` API.

export type Sim = {
  advance(dtSecs: number): number;
  vehicle_instances(): Float32Array;
  signal_heads(): Float32Array;
  junctions(): Float32Array;
  road_strips(): Float32Array;
  lane_dividers(): Float32Array;
  world_bounds(): Float32Array;
  link_names(): string[];
  link_polylines(): Float32Array;
  world_mesh_vertices(): Float32Array;
  world_mesh_indices(): Uint32Array;
  marking_mesh_vertices(): Float32Array;
  marking_mesh_indices(): Uint32Array;
  view_proj(): Float32Array;
  alpha(): number;
  render_instances(): Uint8Array;
  render_instance_count(): number;
  signal_instances(): Uint8Array;
  signal_instance_count(): number;
  crash_instances(): Uint8Array;
  crash_instance_count(): number;
  set_show_crashes(on: boolean): void;
  show_crashes(): boolean;
  clear_crashes(): void;
  density_vertices(): Float32Array;
  density_indices(): Uint32Array;
  set_viewport(w: number, h: number): void;
  fit(): void;
  pan_pixels(dx: number, dy: number): void;
  zoom_at(factor: number, sx: number, sy: number): void;
  set_meters_per_pixel(mpp: number): void;
  meters_per_pixel(): number;
  camera_params(): Float32Array;
  vehicle_count(): number;
  crashed(): number;
  set_selected_link(i: number): void;
  link_stats(i: number): Float32Array;
  play(): void;
  pause(): void;
  set_speed(s: number): void;
  effective_speed(): number;
  selected_speed(): number;
  is_throttled(): boolean;
  set_frame_budget(enabled: boolean): void;
  frame_budget(): boolean;
  set_accel_backend(name: string): void;
  accel_backend(): string;
  set_threads_ready(ready: boolean): void;
  set_par_threshold(n: number): void;
  par_threshold(): number;
  set_scheduler_thread_limit(n: number): void;
  scheduler_thread_limit(): number;
  set_parallel_routing(on: boolean): void;
  parallel_routing(): boolean;
  set_cache_sort(on: boolean): void;
  cache_sort(): boolean;
  set_demand_sources(highway: boolean, surface: boolean): void;
  demand_highway(): boolean;
  demand_surface(): boolean;
  set_rush_hour(enabled: boolean): void;
  demand_rush_hour(): boolean;
  rush_hour_time(): number;
  rush_hour_flows(): Float32Array;
  demand_queued(): number;
  set_demand_rate(scale: number): void;
  set_entry_speed_cap(mps: number): void;
  set_congestion_enabled(enabled: boolean): void;
  set_congestion_engage(occ: number): void;
  congestion_active_links(): number;
  set_sleep_scheduler(on: boolean): void;
  asleep_count(): number;
  enable_gpu_routing(renderer: Renderer): void;
};

export type Renderer = {
  set_world_mesh(wv: Float32Array, wi: Uint32Array, mv: Float32Array, mi: Uint32Array): void;
  render(
    vp: Float32Array,
    alpha: number,
    mpp: number,
    inst: Uint8Array,
    count: number,
    signals: Uint8Array,
    signalCount: number,
    crashes: Uint8Array,
    crashCount: number,
    densityV: Float32Array,
    densityI: Uint32Array,
  ): void;
  resize(width: number, height: number): void;
};

// The wasm-bindgen module namespace from the generated `engine.js`. The threaded build
// additionally exports `initThreadPool` (the rayon worker pool). Both `create` (on-screen
// canvas) and `create_offscreen` (worker OffscreenCanvas) build the same `Renderer`.
export type EngineModule = {
  default: (input?: unknown) => Promise<unknown>;
  Simulation: {
    new (seed: number): Sim;
    scenario(name: string, seed: number): Sim;
    from_map_json?(json: string, seed: number): Sim;
  };
  Renderer: {
    create(canvas: HTMLCanvasElement): Promise<Renderer>;
    create_offscreen(canvas: OffscreenCanvas): Promise<Renderer>;
  };
};

export type ThreadedEngineModule = EngineModule & { initThreadPool(numThreads: number): Promise<void> };

// The 2D-canvas fallback scene: static geometry snapshotted once from the sim.
export type Scene = { roads: Float32Array; dividers: Float32Array; junctions: Float32Array };

// A canvas either renderer can draw to: the on-screen element or a worker's OffscreenCanvas.
export type AnyCanvas = HTMLCanvasElement | OffscreenCanvas;
