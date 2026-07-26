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
  play(): void;
  pause(): void;
  set_speed(s: number): void;
};

type Scene = {
  roads: Float32Array;
  dividers: Float32Array;
  junctions: Float32Array;
  scale: number;
  minX: number;
  minY: number;
  pad: number;
  height: number;
};

export default function EngineCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Sim | null>(null);
  const sceneRef = useRef<Scene | null>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let disposed = false;

    (async () => {
      try {
        const mod = await import(
          /* webpackIgnore: true */ `${basePath()}/wasm-pkg/engine.js`
        );
        await mod.default();
        if (disposed) return;
        const sim: Sim = new mod.Simulation(0xc0ffee);
        simRef.current = sim;
        sceneRef.current = buildScene(sim, canvasRef.current);
        setReady(true);

        const draw = (now: number) => {
          const dt = Math.min((now - last) / 1000, 0.1);
          last = now;
          sim.advance(dt);
          render(canvasRef.current, sim, sceneRef.current);
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
      <div style={{ margin: "8px 0", display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={toggle} disabled={!ready}>
          {playing ? "Pause" : "Play"}
        </button>
        {[1, 2, 8].map((s) => (
          <button key={s} disabled={!ready} onClick={() => simRef.current?.set_speed(s)}>
            {s}×
          </button>
        ))}
        {!ready && !error && <span style={{ opacity: 0.6 }}>loading engine…</span>}
        {error && <span style={{ color: "#ff7b72" }}>engine failed: {error}</span>}
      </div>
      <canvas
        ref={canvasRef}
        width={900}
        height={600}
        style={{ width: "100%", maxWidth: 900, border: "1px solid #222", borderRadius: 8 }}
      />
    </div>
  );
}

function buildScene(sim: Sim, canvas: HTMLCanvasElement | null): Scene {
  const pad = 32;
  const width = canvas?.width ?? 900;
  const height = canvas?.height ?? 600;
  const [minX, minY, maxX, maxY] = sim.world_bounds();
  const scale = Math.min(
    (width - pad * 2) / Math.max(1, maxX - minX),
    (height - pad * 2) / Math.max(1, maxY - minY),
  );
  return {
    roads: sim.road_strips(),
    dividers: sim.lane_dividers(),
    junctions: sim.junctions(),
    scale,
    minX,
    minY,
    pad,
    height,
  };
}

function render(canvas: HTMLCanvasElement | null, sim: Sim, scene: Scene | null) {
  if (!canvas || !scene) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { scale, minX, minY, pad, height } = scene;
  const sx = (x: number) => pad + (x - minX) * scale;
  const sy = (y: number) => height - (pad + (y - minY) * scale);

  ctx.fillStyle = "#0b0e13";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // carriageways
  ctx.strokeStyle = "#2b313a";
  ctx.lineCap = "butt";
  for (let i = 0; i < scene.roads.length; i += 5) {
    const w = scene.roads[i + 4];
    ctx.lineWidth = Math.max(2, w * scale);
    ctx.beginPath();
    ctx.moveTo(sx(scene.roads[i]), sy(scene.roads[i + 1]));
    ctx.lineTo(sx(scene.roads[i + 2]), sy(scene.roads[i + 3]));
    ctx.stroke();
  }

  // paved junctions
  ctx.fillStyle = "#31373f";
  for (let i = 0; i < scene.junctions.length; i += 3) {
    ctx.beginPath();
    ctx.arc(sx(scene.junctions[i]), sy(scene.junctions[i + 1]), scene.junctions[i + 2] * scale, 0, Math.PI * 2);
    ctx.fill();
  }

  // lane dividers
  ctx.strokeStyle = "#5a6472";
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 8]);
  for (let i = 0; i < scene.dividers.length; i += 4) {
    ctx.beginPath();
    ctx.moveTo(sx(scene.dividers[i]), sy(scene.dividers[i + 1]));
    ctx.lineTo(sx(scene.dividers[i + 2]), sy(scene.dividers[i + 3]));
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // vehicles: body + brake-lit rear
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

  // signal heads
  const heads = sim.signal_heads();
  for (let i = 0; i < heads.length; i += 5) {
    ctx.fillStyle = `rgb(${(heads[i + 2] * 255) | 0},${(heads[i + 3] * 255) | 0},${(heads[i + 4] * 255) | 0})`;
    ctx.beginPath();
    ctx.arc(sx(heads[i]), sy(heads[i + 1]), Math.max(2.5, 1.6 * scale), 0, Math.PI * 2);
    ctx.fill();
  }
}
