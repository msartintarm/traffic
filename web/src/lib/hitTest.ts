// Pure geometry hit-testing: the nearest polyline to a world point. Extracted from the
// canvas component so hover/tap road selection is verified in `hitTest.test.ts` — a wrong
// distance here silently selects the wrong road (or nothing) with no visible crash.

export type Polyline = { pts: number[][] };

// Perpendicular distance from (px, py) to segment a–b, clamped to the endpoints.
export function distToSegment(px: number, py: number, a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / len2)) : 0;
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

export function distToPolyline(px: number, py: number, pts: number[][]): number {
  let best = Infinity;
  for (let i = 1; i < pts.length; i++) best = Math.min(best, distToSegment(px, py, pts[i - 1], pts[i]));
  return best;
}

// Index of the nearest link within `maxD` metres of (wx, wy), or -1. Degenerate (<2-point)
// links are skipped; ties resolve to the earliest, matching the strict `<` in the scan.
export function nearestLink(links: Polyline[], wx: number, wy: number, maxD: number): number {
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
