// Tight interleaved A/B to beat the fleet-drain confound: alternate two configs
// in short batches, per-car normalize, report medians. node scripts/shard-ab.mjs
import { chromium } from "playwright";
const browser = await chromium.connectOverCDP(process.argv[2] ?? "http://localhost:9222");
const ctx = browser.contexts()[0];
const page = ctx.pages().find((p) => p.url().includes("scenario")) ?? (await ctx.newPage());
const url = "http://localhost:3001/?scenario=sf&compute=threads&debug=1&warmup=1";
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(6000);
await page.waitForFunction(() => !!window.__stats, null, { timeout: 45000 });
let ew = null;
for (let i = 0; i < 30 && !ew; i++) { ew = page.workers().find((w) => /engineWorker/.test(w.url()) && !w.url().startsWith("blob:")); if (!ew) await page.waitForTimeout(1000); }
await page.waitForFunction(() => window.__stats && !window.__stats.warmup, null, { timeout: 180000, polling: 1000 }).catch(() => {});
await page.evaluate(() => window.__ctl && window.__ctl({ type: "pause" }));

const A = { shards: 0, lod: false, saccel: false };            // classic threads
const B = { shards: 8, lod: false, saccel: false };            // shard-8, worksteal accel
const one = (c) => ew.evaluate((c) => {
  const s = globalThis.__sim;
  s.set_follower_lod(!!c.lod); s.debug_set_shard_count(c.shards);
  if (s.debug_set_shard_accel) s.debug_set_shard_accel(!!c.saccel);
  s.debug_bench_steps(20); // warm the config switch
  const n = s.vehicle_count();
  const ms = s.debug_bench_steps(60);
  return ms / 60 / Math.max(1, n) * 1000; // ns/car... actually us/car *1000 = ns; keep us/car
}, c);
const median = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
const as = [], bs = [];
for (let i = 0; i < 14; i++) { as.push(await one(A)); bs.push(await one(B)); }
const ma = median(as), mb = median(bs);
console.log(`[AB] classic         median ${ma.toFixed(4)} us/car  (${as.map((x)=>x.toFixed(2)).join(",")})`);
console.log(`[AB] shard8-worksteal median ${mb.toFixed(4)} us/car  (${bs.map((x)=>x.toFixed(2)).join(",")})`);
console.log(`[AB] shard vs classic = ${(((ma - mb) / ma) * 100).toFixed(1)}% (neg = shard slower)`);
process.exit(0);
