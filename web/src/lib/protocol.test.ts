import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CONTROL_TYPES,
  type Control,
  type StatsSnapshot,
  isControl,
  overlayFromSnapshot,
} from "./protocol.ts";

test("isControl accepts every declared control and rejects the rest", () => {
  const ok: Control[] = [
    { type: "speed", value: 2 },
    { type: "rushHour", value: true },
    { type: "demandSources", north: true, south: false },
    { type: "fit" },
    { type: "zoomAt", factor: 1.1, bx: 10, by: 20 },
    { type: "select", wx: 1, wy: 2, radius: 12 },
    { type: "hover", wx: 1, wy: 2, x: 3, y: 4 },
    { type: "play" },
    { type: "pause" },
    { type: "frameBudget", value: false },
  ];
  for (const m of ok) assert.equal(isControl(m), true);

  assert.equal(isControl({ type: "bogus" }), false);
  assert.equal(isControl({ value: 3 }), false);
  assert.equal(isControl(null), false);
  assert.equal(isControl("speed"), false);
  assert.equal(isControl(undefined), false);
});

test("CONTROL_TYPES is the complete control set", () => {
  assert.equal(CONTROL_TYPES.size, 21);
  for (const t of ["speed", "fit", "resize", "select", "demandSources", "hover", "play", "pause", "frameBudget", "parallelRouting"]) {
    assert.equal(CONTROL_TYPES.has(t as Control["type"]), true);
  }
});

const snap: StatsSnapshot = {
  vehicles: 500,
  crashed: 0,
  selectedSpeed: 3,
  effectiveSpeed: 2.4,
  throttled: true,
  backend: "threads",
  parThreshold: 400,
  idleSkipped: 5,
  linksQueued: 0,
  waiting: 0,
  gpuRouting: true,
  rushHour: true,
  rushHourTime: 7.5,
  rushHourFlows: [1200, 800, 300, 200],
  camera: [100, 200, 10, 800, 600],
  metersPerPixel: 10,
};

test("overlayFromSnapshot projects the snapshot into overlay strings", () => {
  const o = overlayFromSnapshot(snap, { fitMpp: 10, zoomRange: 60 });
  assert.equal(
    o.stats,
    "• 500 vehicles\n• 0 crashed\n• 2.4×/3× (throttled)\n• threads ▸ parallel (≥400)\n• 5 idle-skipped",
  );
  assert.equal(o.perf, "▶ threads ▸ parallel (≥400) · 5 idle-skipped · routing GPU");
  assert.equal(o.rushClock, " 07:30 · US-101 N1200/S800 · I-280 N300/S200 veh/h/ln");
  assert.equal(o.sliderValue, 0); // mpp == fitMpp → slider at the fit end
  assert.deepEqual(o.camera, { cx: 100, cy: 200, mpp: 10, vw: 800, vh: 600 });
});

test("overlayFromSnapshot hides the rush clock when rush hour is off", () => {
  const o = overlayFromSnapshot({ ...snap, rushHour: false }, { fitMpp: 10, zoomRange: 60 });
  assert.equal(o.rushClock, null);
});

test("overlayFromSnapshot maps a zoomed-in mpp onto the slider", () => {
  const o = overlayFromSnapshot({ ...snap, metersPerPixel: 10 / 60 }, { fitMpp: 10, zoomRange: 60 });
  assert.equal(o.sliderValue, 1000); // fully zoomed in
});
