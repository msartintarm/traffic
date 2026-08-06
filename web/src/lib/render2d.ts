// The Canvas-2D fallback renderer, used when WebGPU is unavailable. Extracted from the
// component and generalised to draw to either the on-screen canvas or a worker's
// OffscreenCanvas — the 2D drawing API is identical on both — so the fallback runs in the
// engine worker exactly as the WebGPU path does.

import { type AnyCanvas, type Scene, type Sim } from "./engineTypes.ts";

export function buildScene(sim: Sim): Scene {
  return { roads: sim.road_strips(), dividers: sim.lane_dividers(), junctions: sim.junctions() };
}

export function render2d(canvas: AnyCanvas, sim: Sim, scene: Scene | null) {
  if (!scene) return;
  const ctx = (canvas as HTMLCanvasElement).getContext("2d") as CanvasRenderingContext2D | null;
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
  const hh = Math.max(2.5, 1.6 * scale); // half-size in px
  for (let i = 0; i < heads.length; i += 7) {
    ctx.fillStyle = `rgb(${(heads[i + 2] * 255) | 0},${(heads[i + 3] * 255) | 0},${(heads[i + 4] * 255) | 0})`;
    const heading = heads[i + 5];
    const isLeft = heads[i + 6] > 0.5;
    ctx.save();
    ctx.translate(sx(heads[i]), sy(heads[i + 1]));
    ctx.rotate(-heading); // world heading → screen frame (sy flips y); local −y is world-left
    if (isLeft) {
      // A left-turn arrow pointing to the driver's left (local −y): shaft + barbs.
      const a = hh * 1.6;
      const bw = hh * 1.15; // barb half-width
      const sw = hh * 0.42; // shaft half-width
      const bh = hh * 0.95; // barb depth from the tip
      ctx.beginPath();
      ctx.moveTo(0, -a); // tip
      ctx.lineTo(bw, -a + bh);
      ctx.lineTo(sw, -a + bh);
      ctx.lineTo(sw, a);
      ctx.lineTo(-sw, a);
      ctx.lineTo(-sw, -a + bh);
      ctx.lineTo(-bw, -a + bh);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.fillRect(-hh, -hh, hh * 2, hh * 2);
    }
    ctx.restore();
  }
}
