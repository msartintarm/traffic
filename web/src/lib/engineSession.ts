// The engine session: load wasm, build the sim + renderer, run the draw loop, apply
// controls, and emit a HUD snapshot per frame. Transport-agnostic — it talks only in the
// tested `Control` / `StatsSnapshot` shapes via callbacks — so the same code runs inside the
// OffscreenCanvas worker and, as a fallback, inline on the main thread. Everything here is
// genuinely browser-only (wasm, WebGPU, the rayon pool); the risk it carries is confined to
// this file and it degrades gracefully at each boundary rather than throwing to the user.

import {
  type AnyCanvas,
  type EngineModule,
  type Renderer,
  type Scene,
  type Sim,
  type ThreadedEngineModule,
} from "./engineTypes.ts";
import { fetchMapText } from "./loadMap.ts";
import { nearestLink } from "./hitTest.ts";
import { REAL_MAPS } from "./maps.ts";
import { rescaleMpp } from "./camera.ts";
import { buildScene, render2d } from "./render2d.ts";
import {
  type Control,
  type InitConfig,
  type SelectedInfo,
  type StatsSnapshot,
} from "./protocol.ts";

export type SessionCallbacks = {
  onReady: (r: {
    backend: string;
    mapLabel: string;
    gpuRouting: boolean;
    fitMpp: number;
    congestionEnabled: boolean;
  }) => void;
  onFrame: (f: { snapshot: StatsSnapshot; selected: SelectedInfo | null; fitMpp: number; ascii?: string | null }) => void;
  onHover: (name: string | null, x: number, y: number) => void;
  // Boot progress for the loading bar: `fraction` 0→1, `stage` a short human label.
  onProgress: (fraction: number, stage: string) => void;
  onFatal: (message: string) => void;
};

export type SessionHandle = { applyControl(c: Control): void; dispose(): void };

// Worker global scopes historically lack requestAnimationFrame, so drive the loop with it
// where present (vsync-paced, pauses on a hidden tab) and a ~60 Hz timer otherwise.
const hasRaf = typeof requestAnimationFrame === "function";
const schedule = (cb: (t: number) => void): number =>
  hasRaf ? requestAnimationFrame(cb) : (setTimeout(() => cb(performance.now()), 16) as unknown as number);
const unschedule = (id: number): void => (hasRaf ? cancelAnimationFrame(id) : clearTimeout(id));

// Vertical resolution of the ASCII "terminal" view. The engine derives the column count
// from the camera aspect (monospace cells are ~2:1), and the main thread scales the font
// to fit — so this is purely the detail/cost knob, not tied to the display size.
const ASCII_ROWS = 54;

function assertNever(x: never): never {
  throw new Error(`unhandled control: ${JSON.stringify(x)}`);
}

// The engine's own links as polylines (world coords), index-aligned with link ids, so a
// click maps to the exact link the engine selects. `link_polylines` is a self-describing
// flat buffer: per link `[n, x0, y0, …]`.
function linksFromSim(sim: Sim): { name: string; pts: number[][] }[] {
  const names = sim.link_names();
  const flat = sim.link_polylines();
  const out: { name: string; pts: number[][] }[] = [];
  let i = 0;
  for (let link = 0; link < names.length; link++) {
    const n = flat[i++];
    const pts: number[][] = [];
    for (let k = 0; k < n; k++) pts.push([flat[i++], flat[i++]]);
    out.push({ name: names[link], pts });
  }
  return out;
}

// Fingerprint the build (`version.txt`, never cached) so a new build always beats the
// browser cache while an unchanged one keeps it — no stale wasm crashing the app.
async function fingerprint(basePath: string, dir: string): Promise<string> {
  try {
    const r = await fetch(`${basePath}/${dir}/version.txt`, { cache: "no-store" });
    if (r.ok) return `?v=${(await r.text()).trim()}`;
  } catch {}
  return "";
}

async function loadEngine(config: InitConfig): Promise<{ mod: EngineModule; threadsReady: boolean }> {
  const wantThreads = config.compute === "threads";
  const isolated = !!globalThis.crossOriginIsolated;
  if (wantThreads && isolated) {
    try {
      const q = await fingerprint(config.basePath, "wasm-pkg-threads");
      const t = (await import(/* webpackIgnore: true */ `${config.basePath}/wasm-pkg-threads/engine.js${q}`)) as ThreadedEngineModule;
      await t.default({ module_or_path: `${config.basePath}/wasm-pkg-threads/engine_bg.wasm${q}` });
      // Bound the pool init so a build whose workers can't boot degrades instead of hanging.
      await Promise.race([
        t.initThreadPool(navigator.hardwareConcurrency || 4),
        new Promise((_, reject) => setTimeout(() => reject(new Error("initThreadPool timed out")), 10000)),
      ]);
      return { mod: t, threadsReady: true };
    } catch (e) {
      console.warn("CPU-threads build unavailable; using single-threaded:", e);
    }
  }
  const q = await fingerprint(config.basePath, "wasm-pkg");
  const mod = (await import(/* webpackIgnore: true */ `${config.basePath}/wasm-pkg/engine.js${q}`)) as EngineModule;
  await mod.default({ module_or_path: `${config.basePath}/wasm-pkg/engine_bg.wasm${q}` });
  return { mod, threadsReady: false };
}

export async function startEngineSession(
  canvas: AnyCanvas,
  isOffscreen: boolean,
  config: InitConfig,
  cb: SessionCallbacks,
): Promise<SessionHandle> {
  let disposed = false;
  let rafId = 0;
  let last = performance.now();
  let renderer: Renderer | null = null;
  let scene: Scene | null = null;
  let gpuRouting = false;
  let selectedIndex = -1;
  let selectedJunction = -1;
  let width = config.width;
  let height = config.height;
  let fitMpp = 1;
  let asciiMode = false; // when on, ship an ASCII grid each frame and skip the GPU/2D draw
  let roads: { name: string; pts: number[][] }[] = [];

  cb.onProgress(0.04, "Loading engine…");
  const { mod, threadsReady } = await loadEngine(config);
  if (disposed) throw new Error("disposed during boot");

  // Prefer the pre-compressed `.gz` (inflated in-browser) — the static export has no server
  // to gzip a 30 MB map, so this cuts the download to ~1/7 and keeps the parse off the UI.
  // The download drives the bar 0.1→0.6; the parse (a single blocking wasm call the main
  // thread can't watch) then sits at "Building road network…" until it returns.
  const realMap = REAL_MAPS[config.scenario];
  let sim: Sim;
  let mapLabel: string;
  if (realMap && mod.Simulation.from_map_json) {
    try {
      cb.onProgress(0.1, "Downloading map…");
      const text = await fetchMapText(`${config.basePath}/${realMap.file}`, (f) =>
        cb.onProgress(0.1 + 0.5 * f, "Downloading map…"),
      );
      cb.onProgress(0.6, "Building road network…");
      sim = mod.Simulation.from_map_json(text, 0xc0ffee, config.splitJunctions);
      mapLabel = realMap.name;
      // Real commute OD (tools/lodes): every real map ships a `<map>.lodes.json`
      // sibling (generated by tools/lodes/fetch_lodes.py --map, deploy-gated); the
      // measured home→work flows join the sampled surface demand, displacing its
      // volume by their share. A missing or rejected file degrades to sampled
      // demand, but loudly — presence is the design, not the exception.
      if (sim.set_commute_od) {
        const lodesFile = realMap.file.replace(/\.json$/, ".lodes.json");
        try {
          const ok = sim.set_commute_od(await fetchMapText(`${config.basePath}/${lodesFile}`));
          if (!ok) console.warn(`commute OD rejected by engine: ${lodesFile}`);
        } catch (e) {
          console.warn(`commute OD missing for ${realMap.name} (${lodesFile}): ${e}`);
        }
      }
      // Compiled transit artifact (tools/gtfs): real train timetable + bus
      // trips. Optional sibling like the LODES file — a map without one keeps
      // the synthetic crossings/headways.
      if (sim.set_transit_json) {
        cb.onProgress(0.72, "Loading transit schedules…");
        const transitFile = realMap.file.replace(/\.json$/, ".transit.json");
        try {
          const counts = sim.set_transit_json(await fetchMapText(`${config.basePath}/${transitFile}`));
          console.info(
            `transit: ${counts[0]} rail trips (+${counts[1]} dropped), ${counts[2]} bus trips (+${counts[3]} dropped)`,
          );
        } catch (e) {
          console.info(`no transit artifact for ${realMap.name} (${transitFile}): ${e}`);
        }
      }
    } catch {
      sim = new mod.Simulation(0xc0ffee);
      mapLabel = "sample map";
    }
  } else if (realMap) {
    sim = new mod.Simulation(0xc0ffee);
    mapLabel = "sample map";
  } else {
    cb.onProgress(0.4, "Building scenario…");
    sim = mod.Simulation.scenario(config.scenario, 0xc0ffee);
    mapLabel = `${config.scenario} scenario`;
  }
  if (disposed) throw new Error("disposed during boot");
  cb.onProgress(0.82, "Preparing renderer…");

  if (threadsReady) sim.set_threads_ready(true);
  sim.set_accel_backend(config.compute);
  sim.set_sleep_scheduler(true);
  const congestionEnabled = config.scenario === "gridlock";
  sim.set_congestion_engage(config.congestionEngage);
  sim.set_congestion_enabled(congestionEnabled);
  roads = linksFromSim(sim);

  canvas.width = width;
  canvas.height = height;
  sim.set_viewport(width, height);
  sim.fit();
  fitMpp = sim.meters_per_pixel();

  let backend: string;
  try {
    renderer = isOffscreen
      ? await mod.Renderer.create_offscreen(canvas as OffscreenCanvas)
      : await mod.Renderer.create(canvas as HTMLCanvasElement);
    renderer.set_world_mesh(
      sim.world_mesh_vertices(),
      sim.world_mesh_indices(),
      sim.marking_mesh_vertices(),
      sim.marking_mesh_indices(),
      sim.render_band_ranges(),
    );
    if (config.gpu) {
      try {
        sim.enable_gpu_routing(renderer);
        gpuRouting = true;
        backend = "WebGPU / WebGL2 · GPU routing";
      } catch {
        backend = "WebGPU / WebGL2";
      }
    } else {
      backend = "WebGPU / WebGL2";
    }
  } catch {
    renderer = null;
    scene = buildScene(sim);
    backend = "2D canvas (fallback)";
  }
  if (disposed) throw new Error("disposed during boot");

  cb.onProgress(1, "Ready");
  cb.onReady({ backend, mapLabel, gpuRouting, fitMpp, congestionEnabled });

  const snapshot = (): StatsSnapshot => {
    const cam = sim.camera_params();
    const flows = sim.rush_hour_flows();
    return {
      vehicles: sim.vehicle_count(),
      crashed: sim.crashed(),
      selectedSpeed: sim.selected_speed(),
      effectiveSpeed: sim.effective_speed(),
      throttled: sim.is_throttled(),
      backend: sim.accel_backend(),
      parThreshold: sim.par_threshold(),
      idleSkipped: sim.asleep_count(),
      linksQueued: sim.congestion_active_links(),
      waiting: sim.demand_queued(),
      gpuRouting,
      dayTime: sim.day_time_hours ? sim.day_time_hours() : Number.NaN,
      rushHour: sim.demand_rush_hour(),
      rushHourTime: sim.rush_hour_time(),
      rushHourFlows: [flows[0], flows[1], flows[2], flows[3]],
      camera: [cam[0], cam[1], cam[2], cam[3], cam[4]],
      metersPerPixel: sim.meters_per_pixel(),
    };
  };

  const selectedInfo = (): SelectedInfo | null => {
    if (selectedJunction >= 0 && sim.junction_stats) {
      const st = sim.junction_stats(selectedJunction);
      return {
        kind: "junction",
        name: sim.junction_label?.(selectedJunction) || `junction ${selectedJunction}`,
        control: sim.junction_control?.(selectedJunction) || "",
        stats: [st[0], st[1], st[2], st[3]],
      };
    }
    if (selectedIndex < 0) return null;
    const st = sim.link_stats(selectedIndex);
    return {
      kind: "link",
      name: roads[selectedIndex]?.name || `link ${selectedIndex}`,
      stats: [st[0], st[1], st[2], st[3]],
    };
  };

  const draw = (now: number) => {
    if (disposed) return;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    sim.advance(dt);
    // ASCII view: the engine rasterises the same geometry to text; the pixel render is
    // skipped (the overlay covers the canvas) so it costs nothing while the toggle is on.
    // A stale wasm build lacking `ascii_view` just falls through to the normal render.
    const ascii = asciiMode && sim.ascii_view ? sim.ascii_view(ASCII_ROWS) : null;
    if (ascii === null) {
      if (renderer) {
        renderer.render(
          sim.view_proj(),
          sim.alpha(),
          sim.meters_per_pixel(),
          sim.render_instances(),
          sim.render_instance_count(),
          sim.signal_instances(),
          sim.signal_instance_count(),
          sim.crash_instances(),
          sim.crash_instance_count(),
          sim.density_vertices(),
          sim.density_indices(),
        );
      } else {
        render2d(canvas, sim, scene);
      }
    }
    cb.onFrame({ snapshot: snapshot(), selected: selectedInfo(), fitMpp, ascii });
    rafId = schedule(draw);
  };
  rafId = schedule(draw);

  const resize = (w: number, h: number) => {
    if (w === width && h === height) return;
    const oldW = width;
    width = w;
    height = h;
    canvas.width = w;
    canvas.height = h;
    // Preserve the world span across the resize, keeping the slider's fit reference in step.
    sim.set_meters_per_pixel(rescaleMpp(sim.meters_per_pixel(), oldW, w));
    fitMpp = rescaleMpp(fitMpp, oldW, w);
    sim.set_viewport(w, h);
    renderer?.resize(w, h);
  };

  const applyControl = (c: Control) => {
    if (disposed) return;
    switch (c.type) {
      case "speed": sim.set_speed(c.value); break;
      case "demandRate": sim.set_demand_rate(c.value); break;
      case "rushHour": sim.set_rush_hour(c.value); break;
      case "dayCompression": sim.set_day_compression?.(c.value); break;
      case "rampMetering": sim.set_ramp_metering?.(c.value); break;
      case "sleepScheduler": sim.set_sleep_scheduler(c.value); break;
      case "parThreshold": sim.set_par_threshold(c.value); break;
      case "parallelRouting": sim.set_parallel_routing(c.value); break;
      case "cacheSort": sim.set_cache_sort(c.value); break;
      case "showCrashes": sim.set_show_crashes(c.value); break;
      case "clearCrashes": sim.clear_crashes(); break;
      case "entrySpeedCap": sim.set_entry_speed_cap(c.value); break;
      case "congestionEngage": sim.set_congestion_engage(c.value); break;
      case "congestionEnabled": sim.set_congestion_enabled(c.value); break;
      case "demandSources": sim.set_demand_sources(c.north, c.south); break;
      case "fit": sim.fit(); break;
      case "metersPerPixel": sim.set_meters_per_pixel(c.value); break;
      case "zoomAt": sim.zoom_at(c.factor, c.bx, c.by); break;
      case "panBy": sim.pan_pixels(c.dx, c.dy); break;
      case "resize": resize(c.w, c.h); break;
      case "play": sim.play(); break;
      case "pause": sim.pause(); break;
      case "frameBudget": sim.set_frame_budget(c.value); break;
      case "ascii": asciiMode = c.value; break;
      case "transit": sim.set_transit_enabled?.(c.value); break;
      case "select": {
        // A click inside a junction footprint selects the intersection; anywhere
        // else selects the nearest road segment.
        selectedJunction = sim.junction_hit ? sim.junction_hit(c.wx, c.wy) : -1;
        selectedIndex = selectedJunction >= 0 ? -1 : nearestLink(roads, c.wx, c.wy, c.radius);
        sim.set_selected_link(selectedIndex);
        sim.set_selected_junction?.(selectedJunction);
        break;
      }
      case "hover": {
        const j = sim.junction_hit && sim.junction_label ? sim.junction_hit(c.wx, c.wy) : -1;
        if (j >= 0) {
          cb.onHover(sim.junction_label!(j) || null, c.x, c.y);
          break;
        }
        const i = nearestLink(roads, c.wx, c.wy, 12); // ~a lane-and-a-half in world metres
        cb.onHover(i >= 0 ? roads[i]?.name || null : null, c.x, c.y);
        break;
      }
      default:
        assertNever(c);
    }
  };

  return {
    applyControl,
    dispose: () => {
      disposed = true;
      unschedule(rafId);
    },
  };
}
