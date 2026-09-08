import test from "node:test";
import assert from "node:assert/strict";
import {
  countyOverlayOpacity,
  countyRect,
  outlineToWorld,
  projectLatLon,
  rectToCss,
  worldToCssMatrix,
} from "./countyOverlay.ts";
import { type MapMeta } from "./loadMap.ts";

test("the origin projects to (0,0) and north/east are positive", () => {
  const origin: [number, number] = [37.2, -121.7];
  const o = projectLatLon(37.2, -121.7, origin);
  assert.equal(o.x, 0);
  assert.equal(o.y, 0);
  const ne = projectLatLon(37.3, -121.6, origin);
  assert.ok(ne.x > 0 && ne.y > 0);
  // One degree of latitude ≈ 111 km under the scraper's spherical model.
  assert.ok(Math.abs(projectLatLon(38.2, -121.7, origin).y - 111195) < 10);
});

test("a neighbour bbox lands west of an origin east of it", () => {
  const origin: [number, number] = [37.2, -121.7];
  const meta: MapMeta = { place: "West County", bbox: [37.0, -122.6, 37.7, -122.1], origin: [37.35, -122.35] };
  const r = countyRect("west", "West County", meta, origin);
  assert.ok(r.x1 < 0, "entirely west of the current origin");
  assert.ok(r.y0 < 0 && r.y1 > 0, "straddles the origin latitude");
  assert.ok(r.x0 < r.x1 && r.y0 < r.y1);
});

test("rectToCss maps the camera centre to the canvas centre and culls off-screen", () => {
  const cam = { cx: 0, cy: 0, mpp: 10, vw: 1000, vh: 800 };
  // A 2 km square centred on the camera → a 200-px square centred in the box.
  const r = { key: "k", name: "n", x0: -1000, y0: -1000, x1: 1000, y1: 1000 };
  const css = rectToCss(r, cam, 500, 400)!; // CSS box at half the backing size
  assert.ok(Math.abs(css.left - 200) < 1e-9 && Math.abs(css.top - 150) < 1e-9);
  assert.ok(Math.abs(css.width - 100) < 1e-9 && Math.abs(css.height - 100) < 1e-9);
  assert.ok(Math.abs(css.chipX - 250) < 1e-9 && Math.abs(css.chipY - 200) < 1e-9);
  // Far off to the east → culled.
  const off = rectToCss({ ...r, x0: 1e7, x1: 1.1e7 }, cam, 500, 400);
  assert.equal(off, null);
});

test("an outline re-framed into its own origin round-trips to itself", () => {
  const origin: [number, number] = [37.2, -121.7];
  const points: [number, number][] = [[-30000, 22000], [0, 0], [15000, -8000]];
  for (const [i, [x, y]] of outlineToWorld({ origin, points }, origin).entries()) {
    assert.ok(Math.abs(x - points[i][0]) < 0.01 && Math.abs(y - points[i][1]) < 0.01);
  }
});

test("outlineToWorld shifts by the same offset the bbox projection uses", () => {
  const cur: [number, number] = [37.2, -121.7];
  const other: [number, number] = [37.35, -122.35];
  const world = outlineToWorld({ origin: other, points: [[0, 0]] }, cur)[0];
  const direct = projectLatLon(other[0], other[1], cur);
  assert.ok(Math.abs(world[0] - direct.x) < 0.01 && Math.abs(world[1] - direct.y) < 0.01);
});

test("worldToCssMatrix agrees with rectToCss on the same corner", () => {
  const cam = { cx: 120, cy: -40, mpp: 7, vw: 1000, vh: 800 };
  const r = { key: "k", name: "n", x0: -900, y0: -700, x1: 1100, y1: 500 };
  const css = rectToCss(r, cam, 500, 400)!;
  const m = worldToCssMatrix(cam, 500, 400);
  assert.ok(Math.abs(m.a * r.x0 + m.e - css.left) < 1e-9);
  assert.ok(Math.abs(m.d * r.y1 + m.f - css.top) < 1e-9, "north edge maps to the top");
});

test("the overlay hides at street zoom and is full at county zoom", () => {
  assert.equal(countyOverlayOpacity(1), 0, "street level: invisible");
  assert.equal(countyOverlayOpacity(8), 0, "fade threshold: still invisible");
  assert.ok(countyOverlayOpacity(16) > 0.4 && countyOverlayOpacity(16) < 0.6, "mid-fade");
  assert.equal(countyOverlayOpacity(24), 1, "county scale: full");
  assert.equal(countyOverlayOpacity(120), 1, "fit zoom: full");
});

test("the chip stays on-screen when the rect hangs off an edge", () => {
  const cam = { cx: 0, cy: 0, mpp: 10, vw: 1000, vh: 800 };
  // Rect extending far west beyond the left edge; visible sliver on the left.
  const r = { key: "k", name: "n", x0: -100000, y0: -1000, x1: -3000, y1: 1000 };
  const css = rectToCss(r, cam, 500, 400)!;
  assert.ok(css.left < 0, "west edge is off-screen");
  assert.ok(css.chipX >= 0 && css.chipX <= 500, "chip clamped into the viewport");
});
