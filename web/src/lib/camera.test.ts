import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type Camera,
  backingSize,
  cameraFromParams,
  clientToBacking,
  clientToWorld,
  mppToSlider,
  rescaleMpp,
  sliderToMpp,
  wheelZoomFactor,
} from "./camera.ts";

const cam: Camera = { cx: 100, cy: 200, mpp: 2, vw: 800, vh: 600 };
const noScale = { sx: 1, sy: 1 };
const atOrigin = { left: 0, top: 0 };

test("cameraFromParams reads the sim.camera_params() tuple", () => {
  assert.deepEqual(cameraFromParams([1, 2, 3, 4, 5]), { cx: 1, cy: 2, mpp: 3, vw: 4, vh: 5 });
});

test("clientToBacking applies rect offset then backing scale", () => {
  assert.deepEqual(
    clientToBacking(50, 30, { left: 10, top: 5 }, { sx: 2, sy: 2 }),
    { bx: 80, by: 50 },
  );
});

test("the viewport centre maps to the camera centre", () => {
  assert.deepEqual(clientToWorld(400, 300, atOrigin, noScale, cam), { wx: 100, wy: 200 });
});

test("clientToWorld moves +x east and +y north (screen-y inverted)", () => {
  assert.deepEqual(clientToWorld(500, 300, atOrigin, noScale, cam), { wx: 300, wy: 200 });
  assert.deepEqual(clientToWorld(400, 200, atOrigin, noScale, cam), { wx: 100, wy: 400 });
});

test("a HiDPI backing scale is folded into the world transform", () => {
  // 2× backing: CSS (200,150) → backing (400,300) → the centre.
  assert.deepEqual(clientToWorld(200, 150, atOrigin, { sx: 2, sy: 2 }, cam), { wx: 100, wy: 200 });
});

test("backingSize multiplies by DPR and caps at 2×", () => {
  assert.deepEqual(backingSize(800, 600, 1), { w: 800, h: 600 });
  assert.deepEqual(backingSize(800, 600, 3), { w: 1600, h: 1200 });
  assert.deepEqual(backingSize(800, 600, 0), { w: 800, h: 600 });
  assert.deepEqual(backingSize(0.2, 0.2, 1), { w: 1, h: 1 });
});

test("rescaleMpp preserves the world span across a resize", () => {
  assert.equal(rescaleMpp(4, 800, 1600), 2);
  assert.equal(800 * 4, 1600 * rescaleMpp(4, 800, 1600));
  assert.equal(rescaleMpp(4, 0, 1600), 4);
  assert.equal(rescaleMpp(4, 800, 0), 4);
});

test("wheelZoomFactor zooms in on scroll-up and out on scroll-down", () => {
  assert.equal(wheelZoomFactor(-1), 1 / 1.1);
  assert.equal(wheelZoomFactor(1), 1.1);
});

test("sliderToMpp spans fit (t=0) to range× closer in (t=1)", () => {
  assert.equal(sliderToMpp(0, 10, 60), 10);
  assert.ok(Math.abs(sliderToMpp(1, 10, 60) - 10 / 60) < 1e-9);
});

test("mppToSlider is the clamped inverse of sliderToMpp", () => {
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    assert.ok(Math.abs(mppToSlider(sliderToMpp(t, 10, 60), 10, 60) - t) < 1e-9);
  }
  assert.equal(mppToSlider(20, 10, 60), 0); // zoomed further out than fit → clamp low
  assert.equal(mppToSlider(10 / 3600, 10, 60), 1); // beyond max-in → clamp high
});
