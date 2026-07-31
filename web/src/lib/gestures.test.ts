import assert from "node:assert/strict";
import { test } from "node:test";

import { dist, isDrag, mid, panDelta, pinch, toCanvas, type Scale } from "./gestures.ts";

// A 900×600 backing store shown at 450×300 CSS px (retina 2×), the canvas offset
// 10px/20px into the page — a realistic mobile mapping.
const RETINA: Scale = { sx: 2, sy: 2, left: 10, top: 20 };
// A 1:1 mapping at the origin, for arithmetic that shouldn't be scaled.
const UNIT: Scale = { sx: 1, sy: 1, left: 0, top: 0 };

test("dist and mid are plain Euclidean helpers", () => {
  assert.equal(dist({ x: 0, y: 0 }, { x: 3, y: 4 }), 5);
  assert.deepEqual(mid({ x: 0, y: 0 }, { x: 4, y: 10 }), { x: 2, y: 5 });
});

test("toCanvas maps a client point into backing-store pixels", () => {
  // 60px right / 45px down of the canvas origin, at 2× → 120,90 backing px.
  assert.deepEqual(toCanvas({ x: 70, y: 65 }, RETINA), { x: 120, y: 90 });
});

test("a tap (small move) is not a drag; a real move is", () => {
  const down = { x: 100, y: 100 };
  assert.equal(isDrag(down, { x: 103, y: 101 }, 5), false, "within threshold ⇒ tap/select");
  assert.equal(isDrag(down, { x: 108, y: 106 }, 5), true, "beyond threshold ⇒ drag/pan");
});

test("one-finger pan scales the client delta by the device pixel ratio", () => {
  // Finger drags 20 CSS px right, 15 down; on a 2× canvas that's 40×30 backing px.
  assert.deepEqual(panDelta({ x: 200, y: 200 }, { x: 220, y: 185 }, RETINA), { x: 40, y: -30 });
});

test("pinch OUT (fingers apart) zooms in toward the midpoint", () => {
  // Fingers 100px apart → 200px apart, midpoint unchanged.
  const g = pinch({ x: 100, y: 300 }, { x: 200, y: 300 }, { x: 50, y: 300 }, { x: 250, y: 300 }, UNIT);
  assert.ok(g.factor < 1, `spreading fingers zoom in (factor < 1), got ${g.factor}`);
  assert.equal(g.factor, 0.5, "distance doubled ⇒ factor 100/200 = 0.5");
  assert.deepEqual({ x: g.focusX, y: g.focusY }, { x: 150, y: 300 }, "zoom focus is the pinch midpoint");
  assert.deepEqual({ x: g.panX, y: g.panY }, { x: 0, y: 0 }, "a centred pinch does not pan");
});

test("pinch IN (fingers together) zooms out", () => {
  const g = pinch({ x: 0, y: 0 }, { x: 200, y: 0 }, { x: 50, y: 0 }, { x: 150, y: 0 }, UNIT);
  assert.ok(g.factor > 1, `pinching together zooms out (factor > 1), got ${g.factor}`);
  assert.equal(g.factor, 2, "distance halved ⇒ factor 200/100 = 2");
});

test("a two-finger drag pans by the midpoint movement (scaled)", () => {
  // Both fingers slide 30 CSS px right with a constant gap: pure pan, no zoom.
  const g = pinch({ x: 100, y: 100 }, { x: 200, y: 100 }, { x: 130, y: 100 }, { x: 230, y: 100 }, RETINA);
  assert.equal(g.factor, 1, "constant finger spacing ⇒ no zoom");
  assert.deepEqual({ x: g.panX, y: g.panY }, { x: 60, y: 0 }, "midpoint moved 30 CSS px ⇒ 60 backing px");
});

test("a degenerate pinch (coincident fingers) is a no-op zoom", () => {
  const g = pinch({ x: 100, y: 100 }, { x: 100, y: 100 }, { x: 100, y: 100 }, { x: 120, y: 100 }, UNIT);
  assert.equal(g.factor, 1, "zero previous distance ⇒ factor 1 (no divide-by-zero blowup)");
});
