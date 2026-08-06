// Pure camera / viewport math shared by the main thread and the OffscreenCanvas worker.
// No DOM and no wasm here, so the client→world transform, backing-store sizing, and the
// zoom mapping — the parts that silently break when the canvas moves to a worker — are
// pinned by `camera.test.ts` instead of eyeballed in a browser.
//
// `Camera` mirrors the tuple `sim.camera_params()` returns: [cx, cy, mpp, vw, vh], where
// (cx, cy) is the world point at the viewport centre, `mpp` is metres per backing pixel,
// and (vw, vh) is the backing-store size in pixels.

export type Camera = { cx: number; cy: number; mpp: number; vw: number; vh: number };

export function cameraFromParams(p: ArrayLike<number>): Camera {
  return { cx: p[0], cy: p[1], mpp: p[2], vw: p[3], vh: p[4] };
}

export type ClientRect = { left: number; top: number };

// Backing-store pixels per CSS pixel (x, y): `canvas.width / rect.width`, `.height / .height`.
export type BackingScale = { sx: number; sy: number };

// A client (mouse/touch) point → backing-store pixel, the space the renderer and
// `sim.zoom_at` work in.
export function clientToBacking(
  clientX: number, clientY: number, rect: ClientRect, s: BackingScale,
): { bx: number; by: number } {
  return { bx: (clientX - rect.left) * s.sx, by: (clientY - rect.top) * s.sy };
}

// A client point → world coords, inverting how vehicles are drawn. Drives hover/tap
// hit-testing; must agree with the renderer's forward transform to the pixel.
export function clientToWorld(
  clientX: number, clientY: number, rect: ClientRect, s: BackingScale, cam: Camera,
): { wx: number; wy: number } {
  const { bx, by } = clientToBacking(clientX, clientY, rect, s);
  return { wx: cam.cx + (bx - cam.vw / 2) * cam.mpp, wy: cam.cy - (by - cam.vh / 2) * cam.mpp };
}

export const MAX_DPR = 2;

// The backing-store size for a CSS box at a device-pixel ratio, DPR capped so a 3–4×
// phone (or heavy browser zoom) doesn't quadruple the fill cost. Always at least 1×1.
export function backingSize(
  cssW: number, cssH: number, dpr: number, maxDpr = MAX_DPR,
): { w: number; h: number } {
  const d = Math.min(dpr || 1, maxDpr);
  return { w: Math.max(1, Math.round(cssW * d)), h: Math.max(1, Math.round(cssH * d)) };
}

// Rescale metres-per-pixel so the visible world span is unchanged when the backing store
// resizes from `oldPx` to `newPx` wide. A zero/absent old size (first layout) is a no-op.
export function rescaleMpp(mpp: number, oldPx: number, newPx: number): number {
  if (!oldPx || !newPx) return mpp;
  return (mpp * oldPx) / newPx;
}

// One wheel notch → zoom factor for `sim.zoom_at`; <1 zooms in (scroll up), >1 out.
export function wheelZoomFactor(deltaY: number, step = 1.1): number {
  return deltaY > 0 ? step : 1 / step;
}

// The zoom slider is logarithmic between the fit-to-view zoom (t=0) and `range`× closer in
// (t=1). `sliderToMpp` / `mppToSlider` are exact inverses; the latter clamps t to [0, 1].
export function sliderToMpp(t: number, fitMpp: number, range: number): number {
  return fitMpp * Math.pow(1 / range, t);
}

export function mppToSlider(mpp: number, fitMpp: number, range: number): number {
  const t = Math.log(mpp / fitMpp) / Math.log(1 / range);
  return Math.min(1, Math.max(0, t));
}
