"use client";

import { useEffect, useRef, useState } from "react";
import { basePath } from "../lib/basePath";

type Sim = {
  advance(dtSecs: number): number;
  vehicle_instances(): Float32Array;
  signal_heads(): Float32Array;
  junctions(): Float32Array;
  road_strips(): Float32Array;
  lane_dividers(): Float32Array;
  world_bounds(): Float32Array;
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
};

const ZOOM_RANGE = 60; // fit-out … max-in ratio driving the slider

export default function EngineCanvas() {
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

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let disposed = false;
    const panning = { active: false };
    const canvas = canvasRef.current!;

    const canvasScale = () => {
      const rect = canvas.getBoundingClientRect();
      return { sx: canvas.width / rect.width, sy: canvas.height / rect.height, rect };
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
      if (e.button === 2) panning.active = true;
    };
    const onMove = (e: MouseEvent) => {
      const sim = simRef.current;
      if (!sim) return;
      const { sx, sy, rect } = canvasScale();
      if (panning.active) {
        sim.pan_pixels(e.movementX * sx, e.movementY * sy);
        return;
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
      const sim = simRef.current;
      if (!sim || e.button !== 0) return;
      const { sx, sy, rect } = canvasScale();
      const [cx, cy, mpp, vw, vh] = sim.camera_params();
      const wx = cx + ((e.clientX - rect.left) * sx - vw / 2) * mpp;
      const wy = cy - ((e.clientY - rect.top) * sy - vh / 2) * mpp;
      const i = nearestLink(roadsRef.current, wx, wy, 12);
      selectedRef.current = i;
      sim.set_selected_link(i);
    };
    const onUp = () => {
      panning.active = false;
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
        const mod = await import(/* webpackIgnore: true */ `${basePath()}/wasm-pkg/engine.js`);
        await mod.default();
        if (disposed) return;

        let loaded: Sim | null = null;
        let label = "sample map";
        try {
          const res = await fetch(`${basePath()}/map.json`);
          if (res.ok && mod.Simulation.from_map_json) {
            const text = await res.text();
            loaded = mod.Simulation.from_map_json(text, 0xc0ffee);
            label = "OSM map";
            roadsRef.current = buildLinks(text); // all links (index-aligned) for hover + click
          }
        } catch {
          loaded = null;
        }
        const sim: Sim = loaded ?? new mod.Simulation(0xc0ffee);
        simRef.current = sim;
        setMapLabel(label);

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
          setBackend("WebGPU / WebGL2");
        } catch {
          renderer = null;
          sceneRef.current = buildScene(sim);
          setBackend("2D canvas (fallback)");
        }
        setReady(true);

        canvas.addEventListener("wheel", onWheel, { passive: false });
        canvas.addEventListener("contextmenu", onContext);
        canvas.addEventListener("mousedown", onDown);
        canvas.addEventListener("mouseleave", onLeave);
        canvas.addEventListener("click", onClick);
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
            statsRef.current.textContent = `${sim.vehicle_count()} vehicles · ${sim.crashed()} crashed`;
          }
          if (panelRef.current) {
            const i = selectedRef.current;
            if (i >= 0) {
              const st = sim.link_stats(i);
              const name = roadsRef.current[i]?.name || `link ${i}`;
              panelRef.current.textContent =
                `${name} — ${st[0] | 0} veh · ${Math.round(st[1] * 3.6)} km/h · ${Math.round(st[2])} veh/h · ${Math.round(st[3] * 100)}% full`;
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
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const toggle = () => {
    const sim = simRef.current;
    if (!sim) return;
    if (playing) sim.pause();
    else sim.play();
    setPlaying(!playing);
  };

  return (
    <div>
      <div style={{ margin: "8px 0", display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={toggle} disabled={!ready}>
          {playing ? "Pause" : "Play"}
        </button>
        {[1, 2, 8].map((s) => (
          <button key={s} disabled={!ready} onClick={() => simRef.current?.set_speed(s)}>
            {s}×
          </button>
        ))}
        <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, opacity: 0.85 }}>
          Zoom
          <input
            ref={sliderRef}
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
        <button disabled={!ready} onClick={() => simRef.current?.fit()}>
          Fit
        </button>
        {!ready && !error && <span style={{ opacity: 0.6 }}>loading engine…</span>}
        {ready && <span style={{ opacity: 0.5, fontSize: 12 }}>{backend} · {mapLabel}</span>}
        {ready && <span ref={statsRef} style={{ opacity: 0.7, fontSize: 12 }} />}
        {error && <span style={{ color: "#ff7b72" }}>engine failed: {error}</span>}
      </div>
      <p style={{ margin: "0 0 8px", opacity: 0.5, fontSize: 12 }}>
        Wheel to zoom · right-drag to pan · hover a road for its name · click to select · Fit to reset
      </p>
      <div
        ref={panelRef}
        style={{
          display: "none",
          margin: "0 0 8px",
          padding: "4px 10px",
          background: "rgba(37,133,180,0.15)",
          border: "1px solid #2585b4",
          borderRadius: 6,
          fontSize: 13,
          fontVariantNumeric: "tabular-nums",
        }}
      />
      <canvas
        ref={canvasRef}
        width={900}
        height={600}
        style={{ width: "100%", maxWidth: 900, border: "1px solid #222", borderRadius: 8, display: "block" }}
      />
      <div
        ref={tipRef}
        style={{
          position: "fixed",
          display: "none",
          pointerEvents: "none",
          padding: "2px 6px",
          background: "rgba(11,14,19,0.9)",
          color: "#e6edf3",
          border: "1px solid #333",
          borderRadius: 4,
          fontSize: 12,
          zIndex: 10,
        }}
      />
    </div>
  );
}

// ---- road-name hover --------------------------------------------------------

/** All links as polylines (world coords), index-aligned with the engine's links,
 * so a click maps to a link index. `name` may be empty. */
function buildLinks(mapJson: string): { name: string; pts: number[][] }[] {
  const m = JSON.parse(mapJson);
  const node = new Map<number, number[]>(m.nodes.map((n: { osm_id: number; x: number; y: number }) => [n.osm_id, [n.x, n.y]]));
  return m.links.map((l: { from_osm: number; to_osm: number; name?: string; geometry?: number[][] }) => {
    const from = node.get(l.from_osm);
    const to = node.get(l.to_osm);
    return { name: l.name ?? "", pts: from && to ? [from, ...(l.geometry ?? []), to] : [] };
  });
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
  for (let i = 0; i < inst.length; i += 4) {
    const brake = inst[i + 3];
    ctx.save();
    ctx.translate(sx(inst[i]), sy(inst[i + 1]));
    ctx.rotate(-inst[i + 2]);
    ctx.fillStyle = "#cdd3da";
    ctx.fillRect(-len / 2, -wid / 2, len, wid);
    if (brake > 0.01) {
      ctx.fillStyle = `rgba(255,40,30,${(0.3 + 0.7 * brake).toFixed(3)})`;
      ctx.fillRect(-len / 2, -wid / 2, len * 0.18, wid);
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
