// Sharding assessment: attach to the real browser, load SF (threads) warmup-full,
// then time bare per-tick cost across shard counts + follower-LOD, using the
// clean debug bench (not the catch-up advance loop). Reports per-shard car
// balance too. Usage: node scripts/shard-sweep.mjs [cdp]
import { chromium } from "playwright";
const cdp = process.argv[2] ?? "http://localhost:9222";
const url = "http://localhost:3001/?scenario=sf&compute=threads&debug=1&warmup=1";

const browser = await chromium.connectOverCDP(cdp);
const ctx = browser.contexts()[0];
const page = ctx.pages().find((p) => p.url().includes("scenario")) ?? (await ctx.newPage());
page.on("console", (m) => { const t = m.text(); if (t.startsWith("transit:") || t.startsWith("[")) return; });
console.log("[sweep] goto", url);
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(6000);
await page.waitForFunction(() => !!window.__stats, null, { timeout: 45000 });

// engine worker (non-blob module worker owning __sim)
let ew = null;
for (let i = 0; i < 30 && !ew; i++) {
  ew = page.workers().find((w) => /engineWorker/.test(w.url()) && !w.url().startsWith("blob:"));
  if (!ew) await page.waitForTimeout(1000);
}
console.log("[sweep] engineWorker", ew ? "found" : "NOT FOUND");
if (!ew) process.exit(1);

console.log("[sweep] waiting for warmup…");
await page.waitForFunction(() => window.__stats && !window.__stats.warmup, null, { timeout: 180000, polling: 1000 }).catch(() => {});
await page.evaluate(() => window.__ctl && window.__ctl({ type: "pause" }));
await page.waitForTimeout(500);

const bench = (label, cfg) =>
  ew.evaluate((c) => {
    const sim = globalThis.__sim;
    sim.set_follower_lod(!!c.lod);
    sim.debug_set_shard_count(c.shards);
    if (sim.debug_set_shard_accel) sim.debug_set_shard_accel(!!c.saccel);
    for (let i = 0; i < 4; i++) sim.debug_bench_steps(20); // warm
    let total = 0;
    for (let i = 0; i < 25; i++) total += sim.debug_bench_steps(20);
    const sh = sim.shard_stats ? Array.from(sim.shard_stats()) : [];
    const cars = [];
    for (let i = 0; i + 3 < sh.length; i += 4) cars.push(sh[i]);
    return { mspt: total / (25 * 20), cars, fleet: sim.vehicle_count(), par: sim.debug_par_threshold() };
  }, cfg);

const cfgs = [
  { label: "classic (0)              ", shards: 0, lod: false, saccel: false },
  { label: "shard 8 per-shard-accel  ", shards: 8, lod: false, saccel: true },
  { label: "shard 8 worksteal-accel  ", shards: 8, lod: false, saccel: false },
  { label: "shard 12 worksteal-accel ", shards: 12, lod: false, saccel: false },
  { label: "shard 8 worksteal+follow ", shards: 8, lod: true, saccel: false },
  { label: "classic + follower       ", shards: 0, lod: true, saccel: false },
];
console.log("[sweep] fleet + par threshold measured per row");
for (const c of cfgs) {
  const r = await bench(c.label, c);
  const bal = r.cars.length ? ` bal=${Math.min(...r.cars)}..${Math.max(...r.cars)}` : "";
  console.log(`  ${c.label} ${r.mspt.toFixed(3)} ms/tick  fleet=${r.fleet} par=${r.par}${bal}`);
}
process.exit(0);
