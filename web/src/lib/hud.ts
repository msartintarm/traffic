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
  vehicles: number;
  crashed: number;
  speed: string;
  exec: string;
  idleSkipped: number;
  linksQueued: number;
  waiting: number;
};

// The stats overlay, one metric per line; the trailing three appear only when non-zero.
export function statsLines(s: Stats): string[] {
  const lines = [`${s.vehicles} vehicles`, `${s.crashed} crashed`, s.speed, s.exec];
  if (s.idleSkipped > 0) lines.push(`${s.idleSkipped} idle-skipped`);
  if (s.linksQueued > 0) lines.push(`${s.linksQueued} links queued`);
  if (s.waiting > 0) lines.push(`${s.waiting} waiting to enter`);
  return lines;
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
  const hh = Math.floor(hour);
  const mm = Math.floor((hour - hh) * 60);
  return (
    ` ${pad2(hh)}:${pad2(mm)} · US-101 ${dir(flows[0], flows[1])}` +
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

// The demand-slider label ("Start ≤ N mph") for a start-speed cap in m/s.
export function startSpeedLabel(startSpeedMps: number, units: Units): string {
  return `Start ≤ ${Math.round(startSpeedMps * MPS_TO[units])} ${UNIT_LABEL[units]}`;
}
