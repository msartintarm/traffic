"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import { basePath } from "../lib/basePath";
import { isDrag, panDelta, pinch, type Scale } from "../lib/gestures";
import styles from "./EngineCanvas.module.css";

/** A show/hide panel: an emoji (optionally labelled) header that toggles a vertically
 * stacked body. Shared by the settings menu and the two corner info overlays so they
 * expand/collapse identically. Uncontrolled — defined at module scope so its open state
 * survives the parent's per-frame re-renders. */
function Collapsible({
  icon,
  label,
  title,
  defaultOpen = true,
  className,
  children,
}: {
  icon: string;
  label?: string;
  title?: string;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={className}>
      <button
        className={styles.overlayToggle}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={title ?? (open ? "Hide" : "Show")}
      >
        {icon}
        {label ? ` ${label}` : ""} {open ? "▾" : "▸"}
      </button>
      {open && <div className={styles.overlayBody}>{children}</div>}
    </div>
  );
}

type Sim = {
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
  set_accel_backend(name: string): void;
  accel_backend(): string;
  set_threads_ready(ready: boolean): void;
  set_par_threshold(n: number): void;
  par_threshold(): number;
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

type Renderer = {
  set_world_mesh(wv: Float32Array, wi: Uint32Array, mv: Float32Array, mi: Uint32Array): void;
  render(
    vp: Float32Array,
    alpha: number,
    mpp: number,
    inst: Uint8Array,
    count: number,
    signals: Uint8Array,
    signalCount: number,
    densityV: Float32Array,
    densityI: Uint32Array,
  ): void;
  resize(width: number, height: number): void;
};

// The wasm-bindgen module namespace we consume from the generated `engine.js`. The
// threaded build additionally exports `initThreadPool` (the rayon worker pool).
type EngineModule = {
  default: (input?: unknown) => Promise<unknown>;
  Simulation: {
    new (seed: number): Sim;
    scenario(name: string, seed: number): Sim;
    from_map_json?(json: string, seed: number): Sim;
  };
  Renderer: { create(canvas: HTMLCanvasElement): Promise<Renderer> };
};
type ThreadedEngineModule = EngineModule & { initThreadPool(numThreads: number): Promise<void> };

const ZOOM_RANGE = 60; // fit-out … max-in ratio driving the slider

// Speed unit display: m/s → the shown unit, and its label.
const MPS_TO = { mi: 2.23694, km: 3.6 } as const;
const UNIT_LABEL = { mi: "mph", km: "km/h" } as const;

export default function EngineCanvas() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Sim | null>(null);
  const sceneRef = useRef<Scene | null>(null);
  const sliderRef = useRef<HTMLInputElement>(null);
  const statsRef = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const roadsRef = useRef<{ name: string; pts: number[][] }[]>([]);
  const selectedRef = useRef<number>(-1);
  const fitMppRef = useRef(1);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [backend, setBackend] = useState("");
  const [mapLabel, setMapLabel] = useState("");
  const [scenario, setScenario] = useState("millbrae");
  const [highwayTraffic, setHighwayTraffic] = useState(true);
  const [surfaceTraffic, setSurfaceTraffic] = useState(true);
  const [rushHour, setRushHour] = useState(false);
  const rushClockRef = useRef<HTMLSpanElement>(null);
  const [accelBackend, setAccelBackend] = useState("serial");
  const [parThreshold, setParThreshold] = useState(2000);
  const [demandRate, setDemandRate] = useState(1);
  const [congestionEnabled, setCongestionEnabled] = useState(false);
  const [congestionEngage, setCongestionEngage] = useState(0.85);
  const [sleepScheduler, setSleepScheduler] = useState(true); // on by default (see bridge assemble)
  const [startSpeedMps, setStartSpeedMps] = useState(36); // ≥ every road limit ⇒ "enter at limit"
  const [units, setUnits] = useState<"mi" | "km">("mi");
  const unitsRef = useRef<"mi" | "km">("mi"); // read inside the rAF draw loop (avoids stale closure)
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    setScenario(new URLSearchParams(window.location.search).get("scenario") ?? "millbrae");
  }, []);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let disposed = false;
    let removeResize = () => {};
    // Any-button drag pans; a press that never moves past the threshold is a click
    // (road selection). Touch mirrors this: one-finger drag pans, a tap selects.
    const DRAG_THRESHOLD = 5; // CSS px before a press becomes a drag rather than a click
    const pointer = { pressed: false, dragged: false, down: { x: 0, y: 0 } };
    const touch = {
      mode: "none",
      prev: { x: 0, y: 0 }, // last single-finger point
      tap: { x: 0, y: 0 }, // where a one-finger gesture started (tap-vs-drag origin)
      dragged: false,
      a: { x: 0, y: 0 }, // last two-finger points, for pinch
      b: { x: 0, y: 0 },
    };
    const canvas = canvasRef.current!;

    const canvasScale = () => {
      const rect = canvas.getBoundingClientRect();
      return { sx: canvas.width / rect.width, sy: canvas.height / rect.height, rect };
    };
    // Client→backing-store mapping the pure gesture helpers consume.
    const scale = (): Scale => {
      const rect = canvas.getBoundingClientRect();
      return { sx: canvas.width / rect.width, sy: canvas.height / rect.height, left: rect.left, top: rect.top };
    };
    // Map a client point to world coords and select the nearest road under it.
    const selectAt = (clientX: number, clientY: number, radius: number) => {
      const sim = simRef.current;
      if (!sim) return;
      const { sx, sy, rect } = canvasScale();
      const [cx, cy, mpp, vw, vh] = sim.camera_params();
      const wx = cx + ((clientX - rect.left) * sx - vw / 2) * mpp;
      const wy = cy - ((clientY - rect.top) * sy - vh / 2) * mpp;
      const i = nearestLink(roadsRef.current, wx, wy, radius);
      selectedRef.current = i;
      sim.set_selected_link(i);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const sim = simRef.current;
      if (!sim) return;
      const { sx, sy, rect } = canvasScale();
      const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
      sim.zoom_at(factor, (e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy);
    };
    const onContext = (e: MouseEvent) => e.preventDefault();
    const onDown = (e: MouseEvent) => {
      pointer.pressed = true;
      pointer.dragged = false;
      pointer.down = { x: e.clientX, y: e.clientY };
    };
    const onMove = (e: MouseEvent) => {
      const sim = simRef.current;
      if (!sim) return;
      const { sx, sy, rect } = canvasScale();
      if (pointer.pressed) {
        if (!pointer.dragged && isDrag(pointer.down, { x: e.clientX, y: e.clientY }, DRAG_THRESHOLD)) {
          pointer.dragged = true;
          if (tipRef.current) tipRef.current.style.display = "none";
        }
        if (pointer.dragged) {
          sim.pan_pixels(e.movementX * sx, e.movementY * sy);
          return;
        }
      }
      // Hover: map the cursor to world space and name the nearest road.
      const tip = tipRef.current;
      if (!tip || roadsRef.current.length === 0) return;
      const [cx, cy, mpp, vw, vh] = sim.camera_params();
      const wx = cx + ((e.clientX - rect.left) * sx - vw / 2) * mpp;
      const wy = cy - ((e.clientY - rect.top) * sy - vh / 2) * mpp;
      let best = "";
      let bestD = 12; // world metres; ~a lane-and-a-half
      for (const road of roadsRef.current) {
        if (!road.name || road.pts.length < 2) continue;
        const d = distToPolyline(wx, wy, road.pts);
        if (d < bestD) {
          bestD = d;
          best = road.name;
        }
      }
      if (best) {
        tip.textContent = best;
        tip.style.left = `${e.clientX + 12}px`;
        tip.style.top = `${e.clientY + 12}px`;
        tip.style.display = "block";
      } else {
        tip.style.display = "none";
      }
    };
    const onLeave = () => {
      if (tipRef.current) tipRef.current.style.display = "none";
    };
    const onClick = (e: MouseEvent) => {
      // Suppress selection when the press was a drag (a pan), not a click.
      if (e.button !== 0 || pointer.dragged) return;
      selectAt(e.clientX, e.clientY, 12);
    };
    const onUp = () => {
      pointer.pressed = false;
    };

    // Touch: one finger pans (a tap selects); two fingers pinch-zoom toward the
    // midpoint and pan by the midpoint's movement.
    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        const t = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        touch.mode = "pan";
        touch.dragged = false;
        touch.prev = t;
        touch.tap = t;
      } else if (e.touches.length >= 2) {
        touch.mode = "pinch";
        touch.a = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        touch.b = { x: e.touches[1].clientX, y: e.touches[1].clientY };
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const sim = simRef.current;
      if (!sim) return;
      const s = scale();
      if (touch.mode === "pan" && e.touches.length === 1) {
        const cur = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        const d = panDelta(touch.prev, cur, s);
        sim.pan_pixels(d.x, d.y);
        touch.prev = cur;
        if (isDrag(touch.tap, cur, DRAG_THRESHOLD)) touch.dragged = true;
      } else if (e.touches.length >= 2) {
        touch.mode = "pinch";
        const curA = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        const curB = { x: e.touches[1].clientX, y: e.touches[1].clientY };
        const g = pinch(touch.a, touch.b, curA, curB, s);
        sim.zoom_at(g.factor, g.focusX, g.focusY); // zoom toward the pinch midpoint
        sim.pan_pixels(g.panX, g.panY); // and follow the midpoint's drag
        touch.a = curA;
        touch.b = curB;
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (touch.mode === "pan" && !touch.dragged) {
        selectAt(touch.tap.x, touch.tap.y, 20); // a more forgiving tap radius on touch
      }
      if (e.touches.length === 0) {
        touch.mode = "none";
      } else if (e.touches.length === 1) {
        // Lifting one finger of a pinch drops back to single-finger panning.
        touch.mode = "pan";
        touch.dragged = true; // continuing a gesture, not a tap
        touch.prev = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    };
    const onSlider = () => {
      const sim = simRef.current;
      const slider = sliderRef.current;
      if (!sim || !slider) return;
      const t = Number(slider.value) / 1000;
      sim.set_meters_per_pixel(fitMppRef.current * Math.pow(1 / ZOOM_RANGE, t));
    };

    (async () => {
      try {
        // The compute backend is decided at page load because it dictates *which* wasm
        // module loads — the CPU-threads build (rayon + SharedArrayBuffer) is a separate
        // artifact from the single-threaded one, so it can't be switched at runtime. The
        // dropdown reloads with `?compute=<serial|threads|gpu>`; we honour it here.
        const compute = new URLSearchParams(window.location.search).get("compute") ?? "serial";
        setAccelBackend(compute);
        const wantThreads = compute === "threads";
        // Threads need cross-origin isolation (COOP/COEP via the shim SW, which reloads
        // once) — only bother when threads are actually requested.
        if (wantThreads) await ensureCrossOriginIsolation();
        if (disposed) return;
        const isolated = typeof window !== "undefined" && window.crossOriginIsolated;
        // Fingerprint the build (`version.txt`, never cached) and stamp it onto the glue
        // and wasm URLs, so a new build always beats the browser cache while an unchanged
        // one keeps it — no stale wasm crashing the app after a rebuild.
        const fingerprint = async (dir: string) => {
          try {
            const r = await fetch(`${basePath()}/${dir}/version.txt`, { cache: "no-store" });
            if (r.ok) return `?v=${(await r.text()).trim()}`;
          } catch {}
          return "";
        };
        let mod: EngineModule | null = null;
        let threadsReady = false;
        if (wantThreads && isolated) {
          try {
            const q = await fingerprint("wasm-pkg-threads");
            const t = (await import(/* webpackIgnore: true */ `${basePath()}/wasm-pkg-threads/engine.js${q}`)) as ThreadedEngineModule;
            await t.default({ module_or_path: `${basePath()}/wasm-pkg-threads/engine_bg.wasm${q}` });
            // Bound the pool init so a build whose workers can't boot degrades instead of hanging.
            await Promise.race([
              t.initThreadPool(navigator.hardwareConcurrency || 4),
              new Promise((_, reject) => setTimeout(() => reject(new Error("initThreadPool timed out")), 10000)),
            ]);
            mod = t;
            threadsReady = true;
          } catch (e) {
            console.warn("CPU-threads build unavailable; using single-threaded:", e);
          }
        }
        if (!mod) {
          const q = await fingerprint("wasm-pkg");
          mod = (await import(/* webpackIgnore: true */ `${basePath()}/wasm-pkg/engine.js${q}`)) as EngineModule;
          await mod.default({ module_or_path: `${basePath()}/wasm-pkg/engine_bg.wasm${q}` });
        }
        if (disposed) return;

        const scenario = new URLSearchParams(window.location.search).get("scenario") ?? "millbrae";
        let loaded: Sim | null = null;
        let label = "sample map";
        if (scenario === "arterial" || scenario === "corridor" || scenario === "gridlock") {
          loaded = mod.Simulation.scenario(scenario, 0xc0ffee);
          label = `${scenario} scenario`;
        } else {
          try {
            const res = await fetch(`${basePath()}/map.json`);
            if (res.ok && mod.Simulation.from_map_json) {
              const text = await res.text();
              loaded = mod.Simulation.from_map_json(text, 0xc0ffee);
              label = "OSM map";
            }
          } catch {
            loaded = null;
          }
        }
        const sim: Sim = loaded ?? new mod.Simulation(0xc0ffee);
        simRef.current = sim;
        if (threadsReady) sim.set_threads_ready(true); // CPU-threads pool usable
        sim.set_accel_backend(compute); // honour ?compute= (falls back to serial if unavailable)
        // The gridlock scenario exists to show the congestion LOD, so engage it by
        // default there; elsewhere it stays off (full per-car detail is the default).
        const congestionOn = scenario === "gridlock";
        setCongestionEnabled(congestionOn);
        sim.set_congestion_engage(congestionEngage);
        sim.set_congestion_enabled(congestionOn);
        setMapLabel(label);
        // Hover/click hit-test against the engine's own links (post collapse/merge),
        // index-aligned with link ids, so selection can't deviate from the engine.
        roadsRef.current = linksFromSim(sim);

        sim.set_viewport(canvas.width, canvas.height);
        sim.fit();
        fitMppRef.current = sim.meters_per_pixel();

        let renderer: Renderer | null = null;
        try {
          renderer = await mod.Renderer.create(canvas);
          renderer!.set_world_mesh(
            sim.world_mesh_vertices(),
            sim.world_mesh_indices(),
            sim.marking_mesh_vertices(),
            sim.marking_mesh_indices(),
          );
          // Route recomputes run on the renderer's WebGPU device by default (it
          // removes the flow-field recompute from the CPU step). Pass `?gpu=0` to
          // fall back to the CPU amortized path.
          if (new URLSearchParams(window.location.search).get("gpu") !== "0") {
            try {
              sim.enable_gpu_routing(renderer!);
              setBackend("WebGPU / WebGL2 · GPU routing");
            } catch {
              setBackend("WebGPU / WebGL2");
            }
          } else {
            setBackend("WebGPU / WebGL2");
          }
        } catch {
          renderer = null;
          sceneRef.current = buildScene(sim);
          setBackend("2D canvas (fallback)");
        }
        setReady(true);

        // Size the backing store to the displayed CSS size × devicePixelRatio, so
        // the canvas stays crisp on HiDPI and under browser zoom (which raises the
        // pixel ratio). Preserve the world span so the view/zoom is unaffected.
        const resizeCanvas = () => {
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
          const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
          if (canvas.width === w && canvas.height === h) return;
          const scale = canvas.width / w;
          canvas.width = w;
          canvas.height = h;
          sim.set_meters_per_pixel(sim.meters_per_pixel() * scale);
          fitMppRef.current *= scale;
          sim.set_viewport(w, h);
          renderer?.resize(w, h);
        };
        resizeCanvas();
        window.addEventListener("resize", resizeCanvas);
        // Entering/leaving fullscreen resizes the canvas to (or from) the whole screen.
        const onFsChange = () => {
          setIsFullscreen(!!document.fullscreenElement);
          resizeCanvas();
        };
        document.addEventListener("fullscreenchange", onFsChange);
        removeResize = () => {
          window.removeEventListener("resize", resizeCanvas);
          document.removeEventListener("fullscreenchange", onFsChange);
        };

        canvas.addEventListener("wheel", onWheel, { passive: false });
        canvas.addEventListener("contextmenu", onContext);
        canvas.addEventListener("mousedown", onDown);
        canvas.addEventListener("mouseleave", onLeave);
        canvas.addEventListener("click", onClick);
        canvas.addEventListener("touchstart", onTouchStart, { passive: false });
        canvas.addEventListener("touchmove", onTouchMove, { passive: false });
        canvas.addEventListener("touchend", onTouchEnd);
        canvas.addEventListener("touchcancel", onTouchEnd);
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);

        const draw = (now: number) => {
          const dt = Math.min((now - last) / 1000, 0.1);
          last = now;
          sim.advance(dt);
          if (renderer) {
            renderer.render(
              sim.view_proj(),
              sim.alpha(),
              sim.meters_per_pixel(),
              sim.render_instances(),
              sim.render_instance_count(),
              sim.signal_instances(),
              sim.signal_instance_count(),
              sim.density_vertices(),
              sim.density_indices(),
            );
          } else {
            render2d(canvas, sim, sceneRef.current);
          }
          if (sliderRef.current) {
            const t = Math.log(sim.meters_per_pixel() / fitMppRef.current) / Math.log(1 / ZOOM_RANGE);
            sliderRef.current.value = String(Math.round(Math.min(1, Math.max(0, t)) * 1000));
          }
          if (statsRef.current) {
            const sel = sim.selected_speed();
            // Show the selected multiplier while the sim keeps up; only when the frame
            // budget is actually dropping ticks show the achieved speed alongside it.
            const speedStr = sim.is_throttled()
              ? `${sim.effective_speed().toFixed(1)}×/${sel}× (throttled)`
              : `${sel}×`;
            const count = sim.vehicle_count();
            // Effective per-frame executor. `accel_backend()` is the active backend after
            // availability fallback, but the Threads backend only parallelizes at/above
            // the crossover — below it the step still runs serial — so surface which is
            // actually in use right now, and the count where threads kick in.
            const backendName = sim.accel_backend();
            let execStr = backendName;
            if (backendName === "threads") {
              const thr = sim.par_threshold();
              execStr = count >= thr ? `threads ▸ parallel (≥${thr})` : `threads ▸ serial (<${thr})`;
            }
            // Active-set scheduler diagnostic: how many cars skipped the full step this
            // frame (parked in queues / cruising open road). 0 when the toggle is off.
            const idle = sim.asleep_count();
            const lines = [`${count} vehicles`, `${sim.crashed()} crashed`, speedStr, execStr];
            if (idle > 0) lines.push(`${idle} idle-skipped`);
            const queued = sim.congestion_active_links();
            if (queued > 0) lines.push(`${queued} links queued`);
            const waiting = sim.demand_queued();
            if (waiting > 0) lines.push(`${waiting} waiting to enter`);
            // One metric per line as bullets (the container renders `\n` as line breaks).
            statsRef.current.textContent = lines.map((l) => `• ${l}`).join("\n");
          }
          if (rushClockRef.current && sim.demand_rush_hour()) {
            const h = sim.rush_hour_time();
            const hh = Math.floor(h);
            const mm = Math.floor((h - hh) * 60);
            const [n101, s101, n280, s280] = sim.rush_hour_flows();
            const dir = (n: number, s: number) => `N${Math.round(n)}/S${Math.round(s)}`;
            rushClockRef.current.textContent =
              ` ${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")} · US-101 ${dir(n101, s101)} · I-280 ${dir(n280, s280)} veh/h/ln`;
          }
          if (panelRef.current) {
            const i = selectedRef.current;
            if (i >= 0) {
              const st = sim.link_stats(i);
              const name = roadsRef.current[i]?.name || `link ${i}`;
              panelRef.current.textContent =
                `${name} — ${st[0] | 0} veh · ${Math.round(st[1] * MPS_TO[unitsRef.current])} ${UNIT_LABEL[unitsRef.current]} · ${Math.round(st[2])} veh/h · ${Math.round(st[3] * 100)}% full`;
              panelRef.current.style.display = "block";
            } else {
              panelRef.current.style.display = "none";
            }
          }
          raf = requestAnimationFrame(draw);
        };
        raf = requestAnimationFrame(draw);
      } catch (e) {
        setError(String(e));
      }
    })();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("contextmenu", onContext);
      canvas.removeEventListener("mousedown", onDown);
      canvas.removeEventListener("mouseleave", onLeave);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("touchcancel", onTouchEnd);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      removeResize();
    };
  }, []);

  const toggle = () => {
    const sim = simRef.current;
    if (!sim) return;
    if (playing) sim.pause();
    else sim.play();
    setPlaying(!playing);
  };

  const toggleFullscreen = async () => {
    const el = wrapperRef.current;
    if (!el) return;
    try {
      if (!document.fullscreenElement) {
        await el.requestFullscreen();
        // Prefer landscape on devices that can rotate (phones/tablets); harmless
        // no-op / rejected promise on desktop, so swallow it.
        try {
          await (screen.orientation as unknown as { lock?: (o: string) => Promise<void> })?.lock?.("landscape");
        } catch {}
      } else {
        try {
          (screen.orientation as unknown as { unlock?: () => void })?.unlock?.();
        } catch {}
        await document.exitFullscreen();
      }
    } catch {}
  };

  return (
    <div ref={wrapperRef} className={styles.wrapper}>
      <canvas ref={canvasRef} width={900} height={600} className={styles.canvas} />

      <div className={styles.topLeft}>
        <div className={styles.controls}>
          <button className={styles.button} onClick={toggle} disabled={!ready}>
            {playing ? "Pause" : "Play"}
          </button>
          {[1, 2, 8, 16, 32].map((s) => (
            <button key={s} className={styles.button} disabled={!ready} onClick={() => simRef.current?.set_speed(s)}>
              {s}×
            </button>
          ))}
          <button
            className={styles.button}
            onClick={toggleFullscreen}
            title="Toggle fullscreen (landscape on mobile)"
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? "⛶ Exit" : "⛶"}
          </button>
        </div>
        <Collapsible
          className={styles.settings}
          icon="⚙️"
          label="Settings"
          defaultOpen={false}
          title="Show or hide settings"
        >
            <label className={styles.zoomLabel}>
              Zoom
              <input
                ref={sliderRef}
                className={styles.slider}
                type="range"
                min={0}
                max={1000}
                defaultValue={0}
                disabled={!ready}
                onInput={() => {
                  const sim = simRef.current;
                  const slider = sliderRef.current;
                  if (!sim || !slider) return;
                  sim.set_meters_per_pixel(fitMppRef.current * Math.pow(1 / ZOOM_RANGE, Number(slider.value) / 1000));
                }}
              />
            </label>
            <button className={styles.button} disabled={!ready} onClick={() => simRef.current?.fit()}>
              Fit
            </button>
            <select
              className={styles.button}
              value={scenario}
              onChange={(e) => {
                const url = new URL(window.location.href);
                url.searchParams.set("scenario", e.target.value);
                window.location.href = url.toString();
              }}
            >
              <option value="millbrae">Millbrae (real map)</option>
              <option value="arterial">Test: arterial junction</option>
              <option value="corridor">Test: signal corridor</option>
              <option value="gridlock">Test: gridlock (fast jam)</option>
            </select>
            <label className={styles.zoomLabel} title="Freeway through-traffic: enters at a highway gateway, bound for the far end of the same highway, another highway exit, or a surface street.">
              <input
                type="checkbox"
                checked={highwayTraffic}
                disabled={!ready}
                onChange={(e) => {
                  simRef.current?.set_demand_sources(e.target.checked, surfaceTraffic);
                  setHighwayTraffic(e.target.checked);
                }}
              />
              Highway traffic
            </label>
            <label className={styles.zoomLabel} title="Local/arterial traffic on the surface streets (through the city, arriving, leaving, and internal trips).">
              <input
                type="checkbox"
                checked={surfaceTraffic}
                disabled={!ready}
                onChange={(e) => {
                  simRef.current?.set_demand_sources(highwayTraffic, e.target.checked);
                  setSurfaceTraffic(e.target.checked);
                }}
              />
              Surface traffic
            </label>
            <label
              className={styles.zoomLabel}
              title="Drive the whole map by the simulated time of day: US-101 and I-280 at their real per-lane volumes (Caltrans PeMS typical-weekday, split by direction), and the surface streets by the urban-arterial diurnal shape. Both build and fade with the commute as the day advances."
            >
              <input
                type="checkbox"
                checked={rushHour}
                disabled={!ready || (!highwayTraffic && !surfaceTraffic)}
                onChange={(e) => {
                  simRef.current?.set_rush_hour(e.target.checked);
                  setRushHour(e.target.checked);
                }}
              />
              Rush hour
              {rushHour && highwayTraffic && <span ref={rushClockRef} className={styles.rushClock} />}
            </label>
            <label className={styles.zoomLabel} title="Spawn-rate multiplier applied to every enabled traffic stream.">
              Rate {demandRate.toFixed(2)}×
              <input
                type="range"
                min={0}
                max={4}
                step={0.25}
                value={demandRate}
                disabled={!ready}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  simRef.current?.set_demand_rate(v);
                  setDemandRate(v);
                }}
              />
            </label>
            <label className={styles.zoomLabel} title="Cap on the speed vehicles enter the map at — still never above the origin road's own speed limit.">
              Start ≤ {Math.round(startSpeedMps * MPS_TO[units])} {UNIT_LABEL[units]}
              <input
                type="range"
                min={0}
                max={36}
                step={1}
                value={startSpeedMps}
                disabled={!ready}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  simRef.current?.set_entry_speed_cap(v);
                  setStartSpeedMps(v);
                }}
              />
            </label>
            <select
              className={styles.button}
              value={units}
              disabled={!ready}
              title="Speed units for the road readout and the start-speed control."
              onChange={(e) => {
                const v = e.target.value as "mi" | "km";
                unitsRef.current = v;
                setUnits(v);
              }}
            >
              <option value="mi">Units: mph</option>
              <option value="km">Units: km/h</option>
            </select>
            <select
              className={styles.button}
              value={accelBackend}
              disabled={!ready}
              title="Executor for the per-vehicle step. Switching reloads the page (the CPU-threads build is a separate wasm module). Unavailable choices fall back to serial — see the active backend in the status line."
              onChange={(e) => {
                // The compute backend dictates which wasm module loads, so it can't be
                // switched in place — reload with the choice as a query param.
                const url = new URL(window.location.href);
                url.searchParams.set("compute", e.target.value);
                window.location.href = url.toString();
              }}
            >
              <option value="serial">Compute: serial (1 core)</option>
              <option value="threads">Compute: CPU threads</option>
              <option value="gpu">Compute: GPU</option>
            </select>
            {accelBackend === "threads" && (
              <label
                className={styles.zoomLabel}
                title="Vehicle count at/above which CPU threads parallelize; below it the step runs serial (rayon overhead isn't worth it). Lower it to see threads engage at fewer cars."
              >
                Threads ≥
                <input
                  className={styles.button}
                  type="number"
                  min={0}
                  step={500}
                  value={parThreshold}
                  disabled={!ready}
                  style={{ width: "5.5em" }}
                  onChange={(e) => {
                    const v = Math.max(0, Math.floor(Number(e.target.value) || 0));
                    simRef.current?.set_par_threshold(v);
                    setParThreshold(v);
                  }}
                />
                cars
              </label>
            )}
            <label
              className={styles.zoomLabel}
              title="Congestion level-of-detail: while a link stays saturated, its queued cars use cheap leader-only car-following instead of the full model, and skip lane-change checks. Cars stay individual and keep their positions — this only cuts per-car cost in jams. Off by default (full detail everywhere)."
            >
              <input
                type="checkbox"
                checked={congestionEnabled}
                disabled={!ready}
                onChange={(e) => {
                  simRef.current?.set_congestion_enabled(e.target.checked);
                  setCongestionEnabled(e.target.checked);
                }}
              />
              Fast jam model
            </label>
            <label
              className={styles.zoomLabel}
              title="Active-set scheduler: cars that are provably idle this tick — parked in a standing queue, or cruising open straight road far from any junction — skip the full per-car decision and use a cheap synthesized one, so per-frame cost tracks the cars actually making decisions rather than the whole fleet. Behaviour-preserving in normal traffic; a big win in gridlock. On by default (single-threaded builds); it automatically stands down when CPU threads are actively parallelizing."
            >
              <input
                type="checkbox"
                checked={sleepScheduler}
                disabled={!ready}
                onChange={(e) => {
                  simRef.current?.set_sleep_scheduler(e.target.checked);
                  setSleepScheduler(e.target.checked);
                }}
              />
              Idle-car skipping
            </label>
            {congestionEnabled && (
              <label
                className={styles.zoomLabel}
                title="Occupancy (share of jam density) a link must hold before its queued cars switch to the cheap follower model. Lower engages sooner on lighter congestion."
              >
                Engage {Math.round(congestionEngage * 100)}%
                <input
                  type="range"
                  min={0.4}
                  max={0.98}
                  step={0.01}
                  value={congestionEngage}
                  disabled={!ready}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    simRef.current?.set_congestion_engage(v);
                    setCongestionEngage(v);
                  }}
                />
              </label>
            )}
        </Collapsible>
      </div>

      <Collapsible className={styles.status} icon="📊" title="Show or hide diagnostics">
        {!ready && !error && <span>loading engine…</span>}
        {ready && <span>• {backend}</span>}
        {ready && <span>• {mapLabel}</span>}
        {ready && <span ref={statsRef} className={styles.statusStats} />}
        {error && <span style={{ color: "#ff7b72" }}>• engine failed: {error}</span>}
      </Collapsible>

      <div ref={panelRef} className={styles.panel} />

      <Collapsible className={styles.hint} icon="❓" title="Show or hide the controls guide">
        <ul className={styles.hintList}>
          <li>Wheel to zoom</li>
          <li>Right-drag to pan</li>
          <li>Hover a road for its name</li>
          <li>Click to select</li>
        </ul>
      </Collapsible>

      <div ref={tipRef} className={styles.tooltip} />
    </div>
  );
}

/** Register the COOP/COEP shim service worker so the page becomes cross-origin
 * isolated (enabling SharedArrayBuffer, and thus the wasm CPU-thread pool). On the
 * first visit the worker isn't controlling yet, so reload once (guarded against a
 * loop); on a browser without service workers, or if it fails, we silently stay
 * single-threaded. */
async function ensureCrossOriginIsolation(): Promise<void> {
  if (typeof window === "undefined" || window.crossOriginIsolated) return;
  if (!("serviceWorker" in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register(`${basePath()}/coi-serviceworker.js`);
    if (reg.active && !navigator.serviceWorker.controller && !sessionStorage.getItem("coiReloaded")) {
      sessionStorage.setItem("coiReloaded", "1");
      window.location.reload();
    }
  } catch (e) {
    console.warn("COOP/COEP shim registration failed:", e);
  }
}

// ---- road-name hover --------------------------------------------------------

/** The engine's own links as polylines (world coords), index-aligned with link
 * ids, so a click maps to the exact link the engine selects. Built from the
 * engine's post-transform geometry — never the raw import — so they can't drift.
 * `link_polylines` is a self-describing flat buffer: per link `[n, x0,y0,…]`. */
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

/** Nearest link index to a world point within `maxD` metres, or -1. */
function nearestLink(links: { pts: number[][] }[], wx: number, wy: number, maxD: number): number {
  let best = -1;
  let bestD = maxD;
  for (let i = 0; i < links.length; i++) {
    if (links[i].pts.length < 2) continue;
    const d = distToPolyline(wx, wy, links[i].pts);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function distToPolyline(px: number, py: number, pts: number[][]): number {
  let best = Infinity;
  for (let i = 1; i < pts.length; i++) {
    best = Math.min(best, distToSegment(px, py, pts[i - 1], pts[i]));
  }
  return best;
}

function distToSegment(px: number, py: number, a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / len2)) : 0;
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

// ---- 2D canvas fallback -----------------------------------------------------

type Scene = { roads: Float32Array; dividers: Float32Array; junctions: Float32Array };

function buildScene(sim: Sim): Scene {
  return { roads: sim.road_strips(), dividers: sim.lane_dividers(), junctions: sim.junctions() };
}

function render2d(canvas: HTMLCanvasElement, sim: Sim, scene: Scene | null) {
  if (!scene) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const [cx, cy, mpp, vw, vh] = sim.camera_params();
  const sx = (x: number) => vw / 2 + (x - cx) / mpp;
  const sy = (y: number) => vh / 2 - (y - cy) / mpp;
  const scale = 1 / mpp;

  ctx.fillStyle = "#0b0e13";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#2b313a";
  ctx.lineCap = "butt";
  for (let i = 0; i < scene.roads.length; i += 5) {
    ctx.lineWidth = Math.max(1, scene.roads[i + 4] * scale);
    ctx.beginPath();
    ctx.moveTo(sx(scene.roads[i]), sy(scene.roads[i + 1]));
    ctx.lineTo(sx(scene.roads[i + 2]), sy(scene.roads[i + 3]));
    ctx.stroke();
  }

  ctx.fillStyle = "#31373f";
  for (let i = 0; i < scene.junctions.length; i += 3) {
    ctx.beginPath();
    ctx.arc(sx(scene.junctions[i]), sy(scene.junctions[i + 1]), scene.junctions[i + 2] * scale, 0, Math.PI * 2);
    ctx.fill();
  }

  const inst = sim.vehicle_instances();
  const len = 4.6 * scale;
  const wid = 2.0 * scale;
  for (let i = 0; i < inst.length; i += 5) {
    const brake = inst[i + 3];
    const blink = inst[i + 4]; // -1 left, +1 right, 0 none (already blink-gated)
    ctx.save();
    ctx.translate(sx(inst[i]), sy(inst[i + 1]));
    ctx.rotate(-inst[i + 2]);
    ctx.fillStyle = "#cdd3da";
    ctx.fillRect(-len / 2, -wid / 2, len, wid);
    if (brake > 0.01) {
      ctx.fillStyle = `rgba(255,40,30,${(0.3 + 0.7 * brake).toFixed(3)})`;
      ctx.fillRect(-len / 2, -wid / 2, len * 0.18, wid);
    }
    if (blink !== 0) {
      // Front-corner amber lamp; world-left is local −y after the sy() flip.
      const bw = len * 0.16;
      const bh = wid * 0.42;
      const by = blink < 0 ? -wid / 2 : wid / 2 - bh;
      ctx.fillStyle = "#ff9e00";
      ctx.fillRect(len / 2 - bw, by, bw, bh);
    }
    ctx.restore();
  }

  const heads = sim.signal_heads();
  for (let i = 0; i < heads.length; i += 5) {
    ctx.fillStyle = `rgb(${(heads[i + 2] * 255) | 0},${(heads[i + 3] * 255) | 0},${(heads[i + 4] * 255) | 0})`;
    ctx.beginPath();
    ctx.arc(sx(heads[i]), sy(heads[i + 1]), Math.max(2.5, 1.6 * scale), 0, Math.PI * 2);
    ctx.fill();
  }
}
