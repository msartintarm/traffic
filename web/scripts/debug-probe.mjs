// Probe against the running dev server: loads a scenario with `?debug=1`,
// attaches to the engine worker, and samples the sim once a second — frame
// advance/render wall times, fleet size, throttle state, per-shard work rows —
// printing a summary at the end.
//   node scripts/debug-probe.mjs [url] [seconds]            headless chromium
//   node scripts/debug-probe.mjs [url] [seconds] --cdp [ep] attach to a real
//     browser over CDP (true GPU / thread-pool numbers); default endpoint
//     http://localhost:9222.
import { chromium } from "playwright";

const url = process.argv[2] ?? "http://localhost:3000/?scenario=sf&compute=threads&debug=1";
const secs = Number(process.argv[3] ?? 60);
const cdpIdx = process.argv.indexOf("--cdp");
const cdp = cdpIdx > 0 ? (process.argv[cdpIdx + 1] ?? "http://localhost:9222") : null;

const browser = cdp
  ? await chromium.connectOverCDP(cdp)
  : await chromium.launch({
      headless: true,
      args: ["--enable-features=SharedArrayBuffer", "--enable-unsafe-webgpu"],
    });
const ctx = cdp ? browser.contexts()[0] : browser;
const page = cdp
  ? (ctx.pages().find((p) => p.url().includes("scenario")) ?? (await ctx.newPage()))
  : await browser.newPage({ viewport: { width: 1280, height: 800 } });

page.on("console", (m) => {
  const t = m.text();
  if (!t.startsWith("[HMR]") && !t.includes("Download the React DevTools")) console.log("[console]", t);
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

// Always (re)navigate so the attached tab picks up the current build.
console.log(`[probe] goto ${url}`);
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(6000);
const isolated = await page.evaluate(() => window.crossOriginIsolated).catch(() => false);
console.log(`[probe] crossOriginIsolated=${isolated} workers=${page.workers().length}`);

// The engine worker's rayon pool blocks on Atomics.wait, so evaluating in it
// hangs. Debug mode mirrors the per-frame snapshot to `window.__stats` and the
// frame timings to the worker's `__frame` — but we read both from the MAIN page
// (`?debug=1` also stashes advance/render there via the session). Poll the main
// page only.
await page
  .waitForFunction(() => !!(window.__stats), null, { timeout: 45000 })
  .catch(() => console.log("[probe] window.__stats never appeared — is ?debug=1 set and the build current?"));

// The engine worker (owns __sim) is the module worker whose URL names it; the
// rayon pool workers are blob: URLs that block on Atomics.wait and must not be
// evaluated. Find the engine worker by url, once.
let engineWorker = null;
for (let i = 0; i < 30 && !engineWorker; i++) {
  engineWorker = page.workers().find((w) => /engineWorker|engine\.worker|\.worker\./.test(w.url()) && !w.url().startsWith("blob:"));
  if (!engineWorker) {
    // fall back: probe non-blob workers with a short race
    for (const w of page.workers().filter((w) => !w.url().startsWith("blob:"))) {
      const ok = await Promise.race([
        w.evaluate(() => !!globalThis.__sim).catch(() => false),
        new Promise((r) => setTimeout(() => r(false), 400)),
      ]);
      if (ok) { engineWorker = w; break; }
    }
  }
  if (!engineWorker) await page.waitForTimeout(1000);
}
console.log(`[probe] engineWorker=${engineWorker ? engineWorker.url() : "NOT FOUND"}`);

// If a warmup was requested, wait for it to finish, then drive the sim at a
// high speed so `advance()` actually steps the full fleet (otherwise a paused
// or 1× clock makes the measurement meaningless).
if (url.includes("warmup=1")) {
  console.log("[probe] waiting for warmup to finish…");
  await page
    .waitForFunction(() => window.__stats && !window.__stats.warmup, null, { timeout: 180000, polling: 1000 })
    .catch(() => console.log("[probe] warmup wait timed out"));
}
await page.evaluate(() => {
  const ctl = window.__ctl;
  if (ctl) {
    ctl({ type: "play" });
    ctl({ type: "speed", value: 8 });
  }
});
await page.waitForTimeout(2000);

// Clean per-tick A/B: pause the render/catch-up loop, then time bare steps via
// the debug bench so the measurement is pure per-tick cost at a fixed fleet.
await page.evaluate(() => window.__ctl && window.__ctl({ type: "pause" }));
const benchOnMain = async (label, lod) => {
  if (!engineWorker) { console.log(`[AB] ${label}: no engine worker`); return -1; }
  const ms = await engineWorker.evaluate((on) => {
    const sim = globalThis.__sim;
    sim.set_follower_lod(on);
    for (let i = 0; i < 3; i++) sim.debug_bench_steps(20); // warm
    let total = 0;
    for (let i = 0; i < 25; i++) total += sim.debug_bench_steps(20);
    return total / (25 * 20);
  }, lod);
  console.log(`[AB] ${label}: ${ms.toFixed(3)} ms/tick`);
  return ms;
};
const off = await benchOnMain("followerLod OFF", false);
const on = await benchOnMain("followerLod ON ", true);
if (off > 0 && on > 0) console.log(`[AB] per-tick delta = ${(((off - on) / off) * 100).toFixed(1)}% (${off.toFixed(3)} → ${on.toFixed(3)} ms/tick)`);
process.exit(0);

const samples = [];
const t0 = Date.now();
while (Date.now() - t0 < secs * 1000) {
  const s = await page
    .evaluate(() => {
      const st = window.__stats;
      if (!st) return { err: "no __stats yet" };
      return {
        advance: st.frameAdvanceMs ?? null,
        render: st.frameRenderMs ?? null,
        vehicles: st.vehicles,
        effective: st.effectiveSpeed,
        throttled: st.throttled,
        backend: st.backend,
        shards: st.shards ?? [],
      };
    })
    .catch((e) => ({ err: String(e) }));
  if (s.err) console.log("[probe] sample error:", s.err);
  else {
    samples.push(s);
    const sh = s.shards.length
      ? ` shards[cars]=${Array.from({ length: s.shards.length / 4 }, (_, i) => s.shards[i * 4]).join(",")}`
      : "";
    console.log(
      `[t+${((Date.now() - t0) / 1000).toFixed(0)}s] ${s.vehicles} cars adv=${s.advance?.toFixed(1)}ms rend=${s.render?.toFixed(1)}ms eff=${s.effective?.toFixed(1)}x thr=${s.throttled} ${s.backend}${sh}`,
    );
  }
  await page.waitForTimeout(1000);
}

const good = samples.filter((s) => s.advance !== null);
const last = good.slice(-Math.min(15, good.length));
const mean = (a, k) => a.reduce((x, s) => x + s[k], 0) / Math.max(1, a.length);
console.log("\n[summary]");
console.log(`  samples=${good.length} final fleet=${good.at(-1)?.vehicles} backend=${good.at(-1)?.backend}`);
console.log(`  last-15s mean advance=${mean(last, "advance").toFixed(2)}ms render=${mean(last, "render").toFixed(2)}ms`);
if (good.at(-1)?.shards?.length) {
  const sh = good.at(-1).shards;
  for (let i = 0; i + 3 < sh.length; i += 4)
    console.log(`  shard ${i / 4}: cars=${sh[i]} deferred=${sh[i + 1]} accelUs=${sh[i + 2]} resolveUs=${sh[i + 3]}`);
}
process.exit(0);
