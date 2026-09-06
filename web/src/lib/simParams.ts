// The full set of boot-time simulation levers, one source of truth (SCHEMA) that
// drives the splash "Params" menu, the URL codec, and the boot wiring. Adding a
// lever = one SCHEMA row.
//
// URL encoding: a compact, versioned binary blob (a protobuf-like positional
// scheme — bools bit-packed, numbers quantized/fixed-width) base64url-encoded into
// `?c=`. We hand-roll it rather than pull in protobuf.js so nothing external ships
// and the codec stays a hundred lines with no schema-compiler step. Defaults are
// NOT encoded: a stock launch carries no `?c=` at all (just `?scenario=`), so the
// blob only exists when the config differs — short in practice, exhaustive when
// needed. `scenario` stays a readable param (it selects splash-vs-boot).

export type Compute = "threads" | "serial" | "gpu";

export type SimParams = {
  // Execution (boot)
  compute: Compute;
  gpuRouting: boolean;
  sharded: boolean;
  localRouting: boolean;
  splitJunctions: boolean;
  prePopulate: boolean;
  sleepScheduler: boolean;
  parThreshold: number;
  parallelRouting: boolean;
  asyncRouting: boolean;
  cacheSort: boolean;
  localitySort: boolean;
  followerLod: boolean;
  // Routing
  targetedRouting: boolean;
  arterialRouting: boolean;
  stopCostRouting: boolean;
  laneEvalStagger: boolean;
  // Behavior / demand volume
  rushHour: boolean;
  demandRate: number;
  dayCompression: number;
  rampMetering: boolean;
  congestionEnabled: boolean;
  congestionEngage: number;
  transit: boolean;
  // Demand generation (topology levers → sim.set_demand_tuning)
  roadFunctionWeighting: boolean;
  gravityBeta: number;
  internalBeta: number;
  onRampShare: number;
  corridorThroughShare: number;
  corridorAccessShare: number;
};

export const DEFAULTS: SimParams = {
  compute: "threads",
  gpuRouting: true,
  sharded: true,
  localRouting: true,
  splitJunctions: true,
  prePopulate: false,
  sleepScheduler: true,
  parThreshold: 500,
  parallelRouting: true,
  asyncRouting: true,
  cacheSort: true,
  localitySort: false,
  followerLod: false,
  targetedRouting: true,
  arterialRouting: false,
  stopCostRouting: true,
  laneEvalStagger: true,
  rushHour: false,
  demandRate: 1,
  dayCompression: 60,
  rampMetering: true,
  congestionEnabled: false,
  congestionEngage: 0.85,
  transit: true,
  roadFunctionWeighting: true,
  gravityBeta: 0.7,
  internalBeta: 1.8,
  onRampShare: 0.25,
  corridorThroughShare: 0.6,
  corridorAccessShare: 0.15,
};

// How a lever reaches the engine at boot:
//  - "init": consumed while building InitConfig / picking the wasm (compute, gpu, split).
//  - "control": applied via applyControl({type: controlType, value}) after boot.
//  - "tuning": collected into sim.set_demand_tuning(...).
//  - "boot": applied at boot through a bespoke path (localRouting/sharded/prePopulate,
//     which already had dedicated boot handling).
export type Apply = "init" | "control" | "tuning" | "boot";

type Kind =
  | { t: "bool" }
  | { t: "enum"; values: string[] }
  | { t: "u16" } // raw integer 0..65535
  | { t: "q"; min: number; max: number }; // quantized float over [min,max] → u16

export type Field = {
  key: keyof SimParams;
  kind: Kind;
  group: "Execution" | "Routing" | "Behavior" | "Demand generation";
  label: string;
  help?: string;
  apply: Apply;
  control?: string; // Control `type` when apply === "control"
  step?: number; // UI slider step for numbers
  uiMin?: number; // UI slider bounds for `u16` fields (q fields use kind.min/max)
  uiMax?: number;
};

/** UI slider bounds for a numeric field. */
export function fieldRange(f: Field): { min: number; max: number } {
  if (f.kind.t === "q") return { min: f.kind.min, max: f.kind.max };
  return { min: f.uiMin ?? 0, max: f.uiMax ?? 65535 };
}

// ORDER IS THE WIRE FORMAT. Only ever append; bump VERSION on any change to an
// existing row's kind/order.
export const SCHEMA: Field[] = [
  { key: "compute", kind: { t: "enum", values: ["threads", "serial", "gpu"] }, group: "Execution", label: "Compute backend", apply: "init", help: "CPU threads (default), single-thread serial, or GPU. Changing it reloads with a different wasm module." },
  { key: "gpuRouting", kind: { t: "bool" }, group: "Execution", label: "GPU flow-field routing", apply: "init", help: "Solve the routing field on the WebGPU device. Only used with the field router (off when local routing is on)." },
  { key: "sharded", kind: { t: "bool" }, group: "Execution", label: "Sharded parallelism", apply: "boot", control: "sharding", help: "Divide the map into regions carried by separate cores (SPMD). On by default." },
  { key: "localRouting", kind: { t: "bool" }, group: "Execution", label: "Local routing", apply: "boot", control: "localRouting", help: "Per-driver bounded-search routing (map-size-independent, measured faster than the flow-field). On by default." },
  { key: "splitJunctions", kind: { t: "bool" }, group: "Execution", label: "Align large junctions", apply: "init", help: "Split large divided-road junctions into aligned nodes so carriageways run straight through." },
  { key: "prePopulate", kind: { t: "bool" }, group: "Execution", label: "Pre-populate traffic", apply: "boot", help: "Fast-forward up to an hour headlessly before the map appears, so roads start busy." },
  { key: "sleepScheduler", kind: { t: "bool" }, group: "Execution", label: "Sleep scheduler", apply: "control", control: "sleepScheduler", help: "Skip queued/isolated cars' full per-tick work. On by default (stands down under parallel threads)." },
  { key: "parThreshold", kind: { t: "u16" }, group: "Execution", label: "Parallel threshold (cars)", apply: "control", control: "parThreshold", step: 100, uiMin: 0, uiMax: 20000, help: "Fleet size above which per-vehicle passes go parallel." },
  { key: "parallelRouting", kind: { t: "bool" }, group: "Execution", label: "Parallel routing", apply: "control", control: "parallelRouting" },
  { key: "asyncRouting", kind: { t: "bool" }, group: "Execution", label: "Async routing", apply: "control", control: "asyncRouting", help: "Solve reroutes off-thread (threads backend)." },
  { key: "cacheSort", kind: { t: "bool" }, group: "Execution", label: "Cache-friendly sort", apply: "control", control: "cacheSort" },
  { key: "localitySort", kind: { t: "bool" }, group: "Execution", label: "Locality reorder", apply: "control", control: "localitySort", help: "Periodically reorder the fleet by road position for cache locality. Off by default." },
  { key: "followerLod", kind: { t: "bool" }, group: "Execution", label: "Front-of-lane LOD", apply: "control", control: "followerLod", help: "Skip node scans for cars behind an un-crossed leader. Off by default (measured ~0% on SF)." },
  { key: "targetedRouting", kind: { t: "bool" }, group: "Routing", label: "Targeted route refresh", apply: "control", control: "targetedRouting" },
  { key: "arterialRouting", kind: { t: "bool" }, group: "Routing", label: "Arterial-first fields", apply: "control", control: "arterialRouting", help: "Bias routing onto arterials. Off by default." },
  { key: "stopCostRouting", kind: { t: "bool" }, group: "Routing", label: "Control-aware routing", apply: "control", control: "stopCostRouting", help: "Cost stops/signals into route choice. On by default." },
  { key: "laneEvalStagger", kind: { t: "bool" }, group: "Routing", label: "Human-cadence lane changes", apply: "control", control: "laneEvalStagger" },
  { key: "rushHour", kind: { t: "bool" }, group: "Behavior", label: "Rush hour", apply: "control", control: "rushHour", help: "Drive the map by the simulated time of day (real PeMS freeway curves + arterial diurnal)." },
  { key: "demandRate", kind: { t: "q", min: 0, max: 10 }, group: "Behavior", label: "Demand rate ×", apply: "control", control: "demandRate", step: 0.1, help: "Multiplier on all spawn rates." },
  { key: "dayCompression", kind: { t: "u16" }, group: "Behavior", label: "Day compression", apply: "control", control: "dayCompression", step: 1, uiMin: 1, uiMax: 240, help: "Simulated seconds per real second for the diurnal clock (1–240)." },
  { key: "rampMetering", kind: { t: "bool" }, group: "Behavior", label: "Ramp metering", apply: "control", control: "rampMetering", help: "Meter freeway on-ramps at the D4 peak windows. On by default." },
  { key: "congestionEnabled", kind: { t: "bool" }, group: "Behavior", label: "Congestion LOD", apply: "control", control: "congestionEnabled", help: "Cheap per-car follower on jammed links. Off by default." },
  { key: "congestionEngage", kind: { t: "q", min: 0, max: 1 }, group: "Behavior", label: "Congestion engage", apply: "control", control: "congestionEngage", step: 0.01, help: "Link fullness at which the congestion LOD engages." },
  { key: "transit", kind: { t: "bool" }, group: "Behavior", label: "Transit", apply: "control", control: "transit", help: "Run scheduled trains/buses when a timetable artifact is present. On by default." },
  { key: "roadFunctionWeighting", kind: { t: "bool" }, group: "Demand generation", label: "Road-function weighting", apply: "tuning", help: "Weight origins/destinations by road class (Local produces, Arterial is through). On by default." },
  { key: "gravityBeta", kind: { t: "q", min: 0, max: 3 }, group: "Demand generation", label: "Gravity β (commute)", apply: "tuning", step: 0.05, help: "Distance decay for commute/through trips. Lower = longer trips." },
  { key: "internalBeta", kind: { t: "q", min: 0, max: 4 }, group: "Demand generation", label: "Gravity β (local)", apply: "tuning", step: 0.05, help: "Distance decay for local errands. Higher = they stay local." },
  { key: "onRampShare", kind: { t: "q", min: 0, max: 1 }, group: "Demand generation", label: "On-ramp share", apply: "tuning", step: 0.05, help: "Share of freeway demand that loads at interior on-ramps vs edge gateways." },
  { key: "corridorThroughShare", kind: { t: "q", min: 0, max: 1 }, group: "Demand generation", label: "Corridor through share", apply: "tuning", step: 0.05, help: "Share of a counted corridor's flow that rides it end-to-end vs mid-corridor access." },
  { key: "corridorAccessShare", kind: { t: "q", min: 0, max: 0.5 }, group: "Demand generation", label: "Corridor access share", apply: "tuning", step: 0.05, help: "Per-cross-street mid-corridor access share." },
];

const VERSION = 1;

function bytesToBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlToBytes(s: string): Uint8Array {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function q16(v: number, min: number, max: number): number {
  const c = Math.max(min, Math.min(max, v));
  return Math.round(((c - min) / (max - min)) * 65535);
}
function dq16(u: number, min: number, max: number): number {
  return min + (u / 65535) * (max - min);
}

/** Encode a full SimParams to a compact base64url blob (positional, versioned). */
export function encodeParams(p: SimParams): string {
  const bools = SCHEMA.filter((f) => f.kind.t === "bool");
  const others = SCHEMA.filter((f) => f.kind.t !== "bool");
  const bitBytes = new Uint8Array(Math.ceil(bools.length / 8));
  bools.forEach((f, i) => {
    if (p[f.key] as boolean) bitBytes[i >> 3] |= 1 << (i & 7);
  });
  const nums: number[] = []; // u16 words
  for (const f of others) {
    const v = p[f.key];
    if (f.kind.t === "enum") nums.push(Math.max(0, f.kind.values.indexOf(v as string)));
    else if (f.kind.t === "u16") nums.push(Math.max(0, Math.min(65535, Math.round(v as number))));
    else if (f.kind.t === "q") nums.push(q16(v as number, f.kind.min, f.kind.max));
  }
  const buf = new Uint8Array(1 + bitBytes.length + nums.length * 2);
  buf[0] = VERSION;
  buf.set(bitBytes, 1);
  let o = 1 + bitBytes.length;
  for (const n of nums) {
    buf[o++] = n & 0xff;
    buf[o++] = (n >> 8) & 0xff;
  }
  return bytesToBase64Url(buf);
}

/** Decode a blob to SimParams, merged onto DEFAULTS. Any error → DEFAULTS. */
export function decodeParams(s: string): SimParams {
  const p: SimParams = { ...DEFAULTS };
  try {
    const buf = base64UrlToBytes(s);
    if (buf.length < 1 || buf[0] !== VERSION) return p;
    const bools = SCHEMA.filter((f) => f.kind.t === "bool");
    const others = SCHEMA.filter((f) => f.kind.t !== "bool");
    const bitLen = Math.ceil(bools.length / 8);
    bools.forEach((f, i) => {
      (p[f.key] as boolean) = (buf[1 + (i >> 3)] & (1 << (i & 7))) !== 0;
    });
    let o = 1 + bitLen;
    for (const f of others) {
      if (o + 1 >= buf.length) break;
      const n = buf[o] | (buf[o + 1] << 8);
      o += 2;
      if (f.kind.t === "enum") p[f.key] = (f.kind.values[n] ?? f.kind.values[0]) as never;
      else if (f.kind.t === "u16") (p[f.key] as number) = n;
      else if (f.kind.t === "q") (p[f.key] as number) = dq16(n, f.kind.min, f.kind.max);
    }
  } catch {
    return { ...DEFAULTS };
  }
  return p;
}

/** Read the boot config from the URL: the compact `?c=` blob if present, else the
 * legacy readable params (`?compute`/`?gpu`/`?split`/`?localrouting`/`?shard`/
 * `?warmup`) merged onto defaults, so old links keep working. `?scenario` is read
 * separately by the caller. */
export function paramsFromUrl(search: string): SimParams {
  const u = new URLSearchParams(search);
  const c = u.get("c");
  if (c) return decodeParams(c);
  const p: SimParams = { ...DEFAULTS };
  const compute = u.get("compute");
  if (compute === "serial" || compute === "gpu" || compute === "threads") p.compute = compute;
  if (u.get("gpu") === "0") p.gpuRouting = false;
  if (u.get("split") === "0") p.splitJunctions = false;
  if (u.get("localrouting") === "0") p.localRouting = false;
  if (u.get("shard") === "0") p.sharded = false;
  if (u.get("warmup") === "1") p.prePopulate = true;
  return p;
}

/** Import the `Control` type lazily via a structural alias to avoid a hard import
 * cycle; the caller passes its real `applyControl`. */
type ControlMsg = { type: string; [k: string]: unknown };

/** Apply every runtime lever in `p` at boot: the `control`-kind levers directly,
 * the `boot`-kind ones (sharding / local routing / pre-populate warmup), and the
 * demand-generation levers as one `demandTuning` message. The `init`-kind levers
 * (compute / gpu / split) are consumed while building `InitConfig`, not here.
 * Applying at default is a no-op in the engine, so this is safe to call always. */
export function applyRuntimeParams(p: SimParams, apply: (c: ControlMsg) => void): void {
  for (const f of SCHEMA) {
    if (f.apply === "control" && f.control) {
      apply({ type: f.control, value: p[f.key] });
    } else if (f.apply === "boot") {
      if (f.key === "sharded") apply({ type: "sharding", value: p.sharded });
      else if (f.key === "localRouting") apply({ type: "localRouting", value: p.localRouting });
      else if (f.key === "prePopulate" && p.prePopulate) apply({ type: "warmup", seconds: 3600 });
    }
  }
  apply({
    type: "demandTuning",
    roadFunctionWeighting: p.roadFunctionWeighting,
    gravityBeta: p.gravityBeta,
    internalBeta: p.internalBeta,
    onRampShare: p.onRampShare,
    corridorThroughShare: p.corridorThroughShare,
    corridorAccessShare: p.corridorAccessShare,
  });
}

export function isDefault(p: SimParams): boolean {
  return SCHEMA.every((f) => {
    if (f.kind.t === "q") return Math.abs((p[f.key] as number) - (DEFAULTS[f.key] as number)) < 1e-3;
    return p[f.key] === DEFAULTS[f.key];
  });
}
