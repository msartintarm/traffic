// Pure HUD string builders — the numbers a running sim reports, formatted for the overlay.
// In the worker split these are computed from a plain stats snapshot posted back to the
// main thread, so keeping them DOM-free lets `hud.test.ts` pin the exact text (units,
// rounding, pluralisation, the throttle / parallel annotations) without a browser.

export type Units = "mi" | "km";
const MPS_TO: Record<Units, number> = { mi: 2.23694, km: 3.6 };
const UNIT_LABEL: Record<Units, string> = { mi: "mph", km: "km/h" };

// Speed multiplier: the selected value while the sim keeps up; the achieved value
// alongside it only once the frame budget is actually dropping ticks.
export function speedString(selected: number, effective: number, throttled: boolean): string {
  return throttled ? `${effective.toFixed(1)}×/${selected}× (throttled)` : `${selected}×`;
}

// The execution-backend line: the bare backend name, except the CPU-thread pool annotates
// whether the current vehicle count crosses the parallel-dispatch threshold.
export function execString(backend: string, vehicles: number, parThreshold: number): string {
  if (backend !== "threads") return backend;
  return vehicles >= parThreshold
    ? `threads ▸ parallel (≥${parThreshold})`
    : `threads ▸ serial (<${parThreshold})`;
}

export type Stats = {
  // Simulated time of day in fractional hours; NaN hides the clock line
  // (an older wasm build without the accessor).
  dayTime: number;
  vehicles: number;
  crashed: number;
  speed: string;
  exec: string;
  idleSkipped: number;
  linksQueued: number;
  waiting: number;
  /** Per-shard `[cars, deferred, accelUs, resolveUs]` rows, flattened; only while the toggle is on. */
  shards?: number[] | null;
};

// The stats overlay, one metric per line; the clock leads when the sim reports a
// time of day, and the trailing three appear only when non-zero.
export function statsLines(s: Stats): string[] {
  const lines = Number.isFinite(s.dayTime) ? [clockText(s.dayTime)] : [];
  lines.push(`${s.vehicles} vehicles`, `${s.crashed} crashed`, s.speed, s.exec);
  if (s.idleSkipped > 0) lines.push(`${s.idleSkipped} idle-skipped`);
  if (s.linksQueued > 0) lines.push(`${s.linksQueued} links queued`);
  if (s.waiting > 0) lines.push(`${s.waiting} waiting to enter`);
  if (s.shards && s.shards.length >= 4) lines.push(...shardLines(s.shards));
  return lines;
}

// One compact line per shard thread: its car count, boundary-deferred count, and
// (native only — the browser's worker threads have no clock) the fused-pass and
// resolve-span times. The imbalance between rows is the point of the display.
export function shardLines(flat: number[]): string[] {
  const out: string[] = [];
  for (let s = 0; s + 3 < flat.length; s += 4) {
    const [cars, deferred, accelUs, resolveUs] = [flat[s], flat[s + 1], flat[s + 2], flat[s + 3]];
    const t = accelUs + resolveUs > 0 ? ` · ${((accelUs + resolveUs) / 1000).toFixed(1)}ms` : "";
    out.push(`shard ${s / 4}: ${cars} cars · ${deferred} at nodes${t}`);
  }
  return out;
}

// Fractional hour → "HH:MM" wall-clock text.
export function clockText(hour: number): string {
  const h = ((hour % 24) + 24) % 24;
  const hh = Math.floor(h);
  return `${pad2(hh)}:${pad2(Math.floor((h - hh) * 60))}`;
}

// Exponential smoothing plus an integer deadband for the HUD's live counts: raw
// per-frame values (vehicle totals, queue depths) jitter by a few units every
// frame, which reads as flicker. Each keyed series is low-pass filtered, and the
// *displayed* integer only moves once the filtered value clearly leaves it — so
// the numbers glide instead of vibrating. A large step (map reset, demand clear)
// snaps immediately rather than gliding through seconds of stale values.
export class StatsSmoother {
  private ema = new Map<string, number>();
  private shown = new Map<string, number>();

  // Smoothed integer for display. `alpha` is the per-frame EMA weight
  // (~0.1 ≈ a fifth of a second at 60 fps).
  count(key: string, raw: number, alpha = 0.1): number {
    const filtered = this.filter(key, raw, alpha);
    const shown = this.shown.get(key);
    if (shown === undefined || Math.abs(filtered - shown) > 0.6) {
      this.shown.set(key, Math.round(filtered));
    }
    return this.shown.get(key)!;
  }

  // Smoothed continuous value (no deadband) for readouts formatted with decimals.
  filter(key: string, raw: number, alpha = 0.1): number {
    const prev = this.ema.get(key);
    const snap = prev === undefined || Math.abs(raw - prev) > Math.max(20, 0.25 * Math.max(Math.abs(raw), Math.abs(prev)));
    const next = snap ? raw : prev + alpha * (raw - prev);
    this.ema.set(key, next);
    if (snap) this.shown.set(key, Math.round(next));
    return next;
  }
}

// The stats lines rendered as the overlay's bulleted text block.
export function statsText(lines: string[]): string {
  return lines.map((l) => `• ${l}`).join("\n");
}

// The compact Performance-panel readout: what's actually running this frame.
export function perfStatus(exec: string, idleSkipped: number, gpuRouting: boolean): string {
  const idle = idleSkipped > 0 ? ` · ${idleSkipped} idle-skipped` : "";
  return `▶ ${exec}${idle} · routing ${gpuRouting ? "GPU" : "CPU"}`;
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

// Directional flow in veh/h/ln, north over south.
function dir(n: number, s: number): string {
  return `N${Math.round(n)}/S${Math.round(s)}`;
}

// The rush-hour clock: fractional hour → "HH:MM" plus the two corridors' directional flows.
// `flows` mirrors `sim.rush_hour_flows()`: [n101, s101, n280, s280].
export function rushClockText(hour: number, flows: ArrayLike<number>): string {
  return (
    ` ${clockText(hour)} · US-101 ${dir(flows[0], flows[1])}` +
    ` · I-280 ${dir(flows[2], flows[3])} veh/h/ln`
  );
}

// The selected-link panel: name, occupancy, mean speed in the chosen units, flow, and how
// full the link is. `stats` mirrors `sim.link_stats(i)`: [vehicles, speedMps, flow, fullFrac].
export function panelText(name: string, stats: ArrayLike<number>, units: Units): string {
  const speed = Math.round(stats[1] * MPS_TO[units]);
  return (
    `${name} — ${stats[0] | 0} veh · ${speed} ${UNIT_LABEL[units]}` +
    ` · ${Math.round(stats[2])} veh/h · ${Math.round(stats[3] * 100)}% full`
  );
}

// The selected-junction panel: crossing name, control regime, live queue/occupancy,
// the longest current wait at its lines, and served throughput. `stats` mirrors
// `sim.junction_stats(i)`: [queued, crossing, maxWaitSecs, throughputVph].
export function junctionPanelText(name: string, control: string, stats: ArrayLike<number>): string {
  return (
    `${name} — ${control} · ${stats[0] | 0} queued · ${stats[1] | 0} crossing` +
    ` · ${Math.round(stats[2])}s max wait · ${Math.round(stats[3])} veh/h`
  );
}

// The followed-vehicle panel: which vehicle the camera is tracking and its live
// speed. `stats` mirrors `sim.selected_vehicle_stats()`: [speedMps, classId].
export function vehiclePanelText(name: string, stats: ArrayLike<number>, units: Units): string {
  const klass = ["car", "truck", "bus"][stats[1] | 0] ?? "vehicle";
  const speed = Math.round(stats[0] * MPS_TO[units]);
  return `Following ${name} (${klass}) — ${speed} ${UNIT_LABEL[units]}`;
}

// The driver-introspection panel: what the followed car perceives and the reason
// its throttle is bound right now. `json` is `sim.selected_vehicle_report()` — a
// small object built engine-side (SI units + labels); we format with the user's
// units. Returns a multi-line string (the panel renders `white-space: pre-line`).
type DriverReport = {
  state: string; speed: number; desired: number; limit: number; accel: number;
  leaderGap: number | null; leaderSpeed: number | null; stopLine: number | null;
  stopSign: number | null; yield: number | null; curve: number | null; mergeGap: number | null;
  lane: number; lanes: number; turn: string; next: string; changing: boolean; waitSecs: number;
};

export function vehicleReportText(name: string, stats: ArrayLike<number>, json: string, units: Units): string {
  const head = vehiclePanelText(name, stats, units);
  if (!json) return head;
  let r: DriverReport;
  try {
    r = JSON.parse(json) as DriverReport;
  } catch {
    return head;
  }
  const u = UNIT_LABEL[units];
  const sp = (mps: number | null) => (mps == null ? "—" : `${Math.round(mps * MPS_TO[units])} ${u}`);
  const m = (v: number | null) => (v == null ? "—" : `${Math.round(v)} m`);
  const lines = [
    head,
    `doing: ${r.state}${r.changing ? " · changing lanes" : ""}`,
    `speed ${sp(r.speed)} → want ${sp(r.desired)} (limit ${sp(r.limit)}) · accel ${r.accel >= 0 ? "+" : ""}${r.accel.toFixed(1)} m/s²`,
    `lane ${r.lane + 1}/${r.lanes} · next ${r.turn} → ${r.next || "—"}`,
  ];
  // Only surface the inputs that are actually active this instant.
  if (r.leaderGap != null) lines.push(`car ahead: ${m(r.leaderGap)} gap @ ${sp(r.leaderSpeed)}`);
  if (r.stopLine != null) lines.push(`stopping at line in ${m(r.stopLine)}`);
  if (r.stopSign != null) lines.push(`stop sign in ${m(r.stopSign)}`);
  if (r.yield != null) lines.push(`yielding — line in ${m(r.yield)}`);
  if (r.curve != null) lines.push(`curve ahead: ${sp(r.curve)}`);
  if (r.mergeGap != null) lines.push(`merge conflict: ${m(r.mergeGap)} gap`);
  if (r.waitSecs >= 1) lines.push(`waited ${Math.round(r.waitSecs)} s`);
  return lines.join("\n");
}

// The demand-slider label ("Start ≤ N mph") for a start-speed cap in m/s.
export function startSpeedLabel(startSpeedMps: number, units: Units): string {
  return `Start ≤ ${Math.round(startSpeedMps * MPS_TO[units])} ${UNIT_LABEL[units]}`;
}
