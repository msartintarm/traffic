// Pure math for the county-jump overlay: place another map's GPS extent inside
// the current map's local frame (mirroring the scraper's equirectangular
// projection) and turn a world rect into CSS-pixel placement under the live
// camera. No DOM here, so `countyOverlay.test.ts` pins the transforms.

import { type Camera } from "./camera.ts";
import { type CountyOutline, type MapMeta } from "./loadMap.ts";

const EARTH_R = 6371000.0;

// The scraper's projection (tools/osm-scraper `project`): metres east/north of
// the origin, longitude scaled by the origin latitude's cosine.
export function projectLatLon(lat: number, lon: number, origin: [number, number]): { x: number; y: number } {
  const rad = Math.PI / 180;
  return {
    x: (lon - origin[1]) * rad * Math.cos(origin[0] * rad) * EARTH_R,
    y: (lat - origin[0]) * rad * EARTH_R,
  };
}

export type CountyRect = {
  key: string;
  name: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /// The county's road-network silhouette in the current map's frame, when its
  /// outline sidecar exists — drawn instead of the bounding box.
  outline?: [number, number][];
};

// A neighbour's outline (metres in ITS local frame) re-framed into the current
// map's: invert the neighbour's projection back to GPS, then project with the
// current origin. Both frames are the scraper's equirectangular model, so the
// round trip is exact up to float noise.
export function outlineToWorld(outline: CountyOutline, origin: [number, number]): [number, number][] {
  const rad = Math.PI / 180;
  const [lat0, lon0] = outline.origin;
  return outline.points.map(([x, y]) => {
    const lat = y / (EARTH_R * rad) + lat0;
    const lon = x / (EARTH_R * rad * Math.cos(lat0 * rad)) + lon0;
    const p = projectLatLon(lat, lon, origin);
    return [p.x, p.y];
  });
}

// An SVG path (world coordinates; the y-flip lives in the render transform).
export function outlinePath(points: [number, number][]): string {
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x} ${y}`).join(" ") + " Z";
}

// Zoom gate: the overlay is a region-scale affordance, so it stays out of the
// way at street zoom and fades in as the view pulls back. Fully hidden below
// `COUNTY_FADE_MIN_MPP` (viewport ≲ 7 km on a 900-px canvas), fully visible
// past `COUNTY_FADE_FULL_MPP` (≳ 22 km — county context).
export const COUNTY_FADE_MIN_MPP = 8;
export const COUNTY_FADE_FULL_MPP = 24;

export function countyOverlayOpacity(mpp: number): number {
  return Math.min(1, Math.max(0, (mpp - COUNTY_FADE_MIN_MPP) / (COUNTY_FADE_FULL_MPP - COUNTY_FADE_MIN_MPP)));
}

// The affine world→CSS-pixel transform under the camera, as SVG matrix terms
// (negative `d` flips world-north to screen-up).
export function worldToCssMatrix(
  cam: Camera,
  cssW: number,
  cssH: number,
): { a: number; d: number; e: number; f: number } {
  const a = cssW / cam.vw / cam.mpp;
  const d = -(cssH / cam.vh) / cam.mpp;
  return { a, d, e: cssW / 2 - cam.cx * a, f: cssH / 2 - cam.cy * d };
}

// Another map's scrape bbox, projected into the frame of the map currently
// loaded (whose scraper origin is `origin`).
export function countyRect(key: string, name: string, meta: MapMeta, origin: [number, number]): CountyRect {
  const [latMin, lonMin, latMax, lonMax] = meta.bbox;
  const a = projectLatLon(latMin, lonMin, origin);
  const b = projectLatLon(latMax, lonMax, origin);
  return {
    key,
    name,
    x0: Math.min(a.x, b.x),
    y0: Math.min(a.y, b.y),
    x1: Math.max(a.x, b.x),
    y1: Math.max(a.y, b.y),
  };
}

export type CssRect = { left: number; top: number; width: number; height: number; chipX: number; chipY: number };

// A world rect under the camera, in CSS pixels of a canvas box `cssW`×`cssH` —
// or null when fully off-screen. The chip point is the centre of the rect's
// *visible* portion, so the invitation stays on screen while any of the county
// does.
export function rectToCss(r: CountyRect, cam: Camera, cssW: number, cssH: number): CssRect | null {
  const sx = cssW / cam.vw;
  const sy = cssH / cam.vh;
  const left = ((r.x0 - cam.cx) / cam.mpp + cam.vw / 2) * sx;
  const right = ((r.x1 - cam.cx) / cam.mpp + cam.vw / 2) * sx;
  const top = ((cam.cy - r.y1) / cam.mpp + cam.vh / 2) * sy; // north edge → top
  const bottom = ((cam.cy - r.y0) / cam.mpp + cam.vh / 2) * sy;
  if (right < 0 || bottom < 0 || left > cssW || top > cssH) {
    return null;
  }
  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
    chipX: (Math.max(left, 0) + Math.min(right, cssW)) / 2,
    chipY: (Math.max(top, 0) + Math.min(bottom, cssH)) / 2,
  };
}
