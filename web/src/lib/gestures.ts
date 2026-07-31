// Pure touch/pointer gesture math for the map viewport — no DOM, so it is unit
// testable. The EngineCanvas handlers convert raw touch/mouse events into these
// calls; keeping the arithmetic here means the mobile pan/pinch/tap behaviour is
// verified without a browser.

export type Pt = { x: number; y: number };

/** Client→canvas mapping: `sx`/`sy` are backing-store px per CSS px, `left`/`top`
 *  the canvas's client-rect origin. */
export type Scale = { sx: number; sy: number; left: number; top: number };

export function dist(a: Pt, b: Pt): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function mid(a: Pt, b: Pt): Pt {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/** A press/touch has moved far enough (CSS px) to be a drag/pan rather than a
 *  click/tap that selects. */
export function isDrag(down: Pt, cur: Pt, threshold: number): boolean {
  return dist(down, cur) > threshold;
}

/** A client point in canvas backing-store pixels (what the camera consumes). */
export function toCanvas(p: Pt, s: Scale): Pt {
  return { x: (p.x - s.left) * s.sx, y: (p.y - s.top) * s.sy };
}

/** Pan distance (backing-store px) for a pointer that moved `prev`→`cur`. */
export function panDelta(prev: Pt, cur: Pt, s: Scale): Pt {
  return { x: (cur.x - prev.x) * s.sx, y: (cur.y - prev.y) * s.sy };
}

export type Pinch = {
  /** Multiply the camera's metres-per-pixel by this: <1 zooms in, >1 zooms out —
   *  matching `zoom_at`. Fingers spreading apart yields <1. */
  factor: number;
  /** Zoom focus (backing-store px): the pinch midpoint stays put on screen. */
  focusX: number;
  focusY: number;
  /** Pan (backing-store px) so the map also follows the midpoint's movement. */
  panX: number;
  panY: number;
};

/** Two-finger pinch from previous finger positions `prevA`/`prevB` to current
 *  `curA`/`curB`: a zoom toward the current midpoint plus the midpoint's pan. */
export function pinch(prevA: Pt, prevB: Pt, curA: Pt, curB: Pt, s: Scale): Pinch {
  const prevD = dist(prevA, prevB);
  const curD = dist(curA, curB);
  const factor = prevD > 0 && curD > 0 ? prevD / curD : 1;
  const focus = toCanvas(mid(curA, curB), s);
  const pan = panDelta(mid(prevA, prevB), mid(curA, curB), s);
  return { factor, focusX: focus.x, focusY: focus.y, panX: pan.x, panY: pan.y };
}
