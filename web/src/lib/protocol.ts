// The typed main-thread ↔ worker contract. In the OffscreenCanvas split the worker owns
// the canvas, wasm, and the sim; the main thread forwards user input / control changes and
// renders the HUD from a per-frame snapshot posted back. Centralising the message shapes
// (and the snapshot→overlay projection) here means a renamed field or a dropped control
// fails `protocol.test.ts` rather than silently no-op'ing a setting across the boundary.

import {
  type Camera,
  cameraFromParams,
  mppToSlider,
} from "./camera.ts";
import {
  type Stats,
  type Units,
  execString,
  perfStatus,
  rushClockText,
  speedString,
  statsLines,
  statsText,
} from "./hud.ts";

// main → worker. Every value is plain data (no wasm handles), so a control cannot be
// silently lost at the `postMessage` boundary. View gestures are pre-transformed on the
// main thread against the last snapshot's camera, so the worker only sees world/backing
// coordinates it can apply directly.
export type Control =
  | { type: "speed"; value: number }
  | { type: "demandRate"; value: number }
  | { type: "rushHour"; value: boolean }
  | { type: "sleepScheduler"; value: boolean }
  | { type: "schedulerThreadLimit"; value: number }
  | { type: "parThreshold"; value: number }
  | { type: "parallelRouting"; value: boolean }
  | { type: "cacheSort"; value: boolean }
  | { type: "showCrashes"; value: boolean }
  | { type: "clearCrashes" }
  | { type: "entrySpeedCap"; value: number }
  | { type: "congestionEngage"; value: number }
  | { type: "congestionEnabled"; value: boolean }
  | { type: "demandSources"; north: boolean; south: boolean }
  | { type: "fit" }
  | { type: "metersPerPixel"; value: number }
  | { type: "zoomAt"; factor: number; bx: number; by: number }
  | { type: "panBy"; dx: number; dy: number }
  | { type: "resize"; w: number; h: number }
  | { type: "select"; wx: number; wy: number; radius: number }
  | { type: "hover"; wx: number; wy: number; x: number; y: number }
  | { type: "play" }
  | { type: "pause" }
  | { type: "frameBudget"; value: boolean };

export type ControlType = Control["type"];

export const CONTROL_TYPES: ReadonlySet<ControlType> = new Set([
  "speed", "demandRate", "rushHour", "sleepScheduler", "schedulerThreadLimit",
  "parThreshold", "parallelRouting", "cacheSort", "showCrashes", "clearCrashes", "entrySpeedCap", "congestionEngage", "congestionEnabled",
  "demandSources", "fit", "metersPerPixel", "zoomAt", "panBy", "resize", "select",
  "hover", "play", "pause", "frameBudget",
] satisfies ControlType[]);

export function isControl(msg: unknown): msg is Control {
  return (
    typeof msg === "object" &&
    msg !== null &&
    "type" in msg &&
    typeof (msg as { type: unknown }).type === "string" &&
    CONTROL_TYPES.has((msg as { type: ControlType }).type)
  );
}

// worker → main, once per rendered frame. Mirrors the sim reads the HUD used to make
// directly; `overlayFromSnapshot` turns it back into the strings the overlay shows.
export type StatsSnapshot = {
  vehicles: number;
  crashed: number;
  selectedSpeed: number;
  effectiveSpeed: number;
  throttled: boolean;
  backend: string;
  parThreshold: number;
  idleSkipped: number;
  linksQueued: number;
  waiting: number;
  gpuRouting: boolean;
  rushHour: boolean;
  rushHourTime: number;
  rushHourFlows: [number, number, number, number];
  camera: [number, number, number, number, number];
  metersPerPixel: number;
};

export type Overlay = {
  stats: string;
  perf: string;
  rushClock: string | null;
  sliderValue: number;
  camera: Camera;
};

export type OverlayOpts = { fitMpp: number; zoomRange: number };

// Project a frame snapshot to the overlay's strings + the cached camera the main thread
// uses to pre-transform the next gesture. Pure composition over camera.ts + hud.ts.
export function overlayFromSnapshot(s: StatsSnapshot, opts: OverlayOpts): Overlay {
  const speed = speedString(s.selectedSpeed, s.effectiveSpeed, s.throttled);
  const exec = execString(s.backend, s.vehicles, s.parThreshold);
  const stats: Stats = {
    vehicles: s.vehicles,
    crashed: s.crashed,
    speed,
    exec,
    idleSkipped: s.idleSkipped,
    linksQueued: s.linksQueued,
    waiting: s.waiting,
  };
  const t = mppToSlider(s.metersPerPixel, opts.fitMpp, opts.zoomRange);
  return {
    stats: statsText(statsLines(stats)),
    perf: perfStatus(exec, s.idleSkipped, s.gpuRouting),
    rushClock: s.rushHour ? rushClockText(s.rushHourTime, s.rushHourFlows) : null,
    sliderValue: Math.round(t * 1000),
    camera: cameraFromParams(s.camera),
  };
}

// The selected link's live readout, posted each frame so the main thread formats the panel
// in the user's chosen units (units are a display concern the worker never needs to know).
export type SelectedInfo = { name: string; stats: [number, number, number, number] };

// The one-shot boot config the worker needs to build the scene. Everything is plain data;
// the OffscreenCanvas travels separately in the transfer list of the init message.
export type InitConfig = {
  scenario: string;
  compute: string; // "serial" | "threads" | "gpu"
  gpu: boolean; // GPU flow-field routing (the `?gpu=0` opt-out)
  basePath: string;
  width: number;
  height: number;
  congestionEngage: number;
  zoomRange: number;
};

// main → worker init (carries the transferred canvas). Distinct from `Control` because it
// is one-shot and bears a transferable; the worker treats everything else as a `Control`.
export type InitMsg = { type: "init"; canvas: OffscreenCanvas; config: InitConfig };

// worker → main. `ready` reports the built backend once; `frame` carries a HUD snapshot per
// rendered frame (plus the selected readout and the live fit zoom for the slider); `hover`
// answers a hover hit-test, echoing the client point so the tooltip lands there; `fatal`
// surfaces a boot/parse failure as the on-screen error.
export type FromWorker =
  | { type: "ready"; backend: string; mapLabel: string; gpuRouting: boolean; fitMpp: number; congestionEnabled: boolean }
  | { type: "frame"; snapshot: StatsSnapshot; selected: SelectedInfo | null; fitMpp: number }
  | { type: "hover"; name: string | null; x: number; y: number }
  | { type: "progress"; fraction: number; stage: string }
  | { type: "fatal"; message: string };

// The units carried alongside the overlay for the selected-link panel and slider label.
export type { Units };
