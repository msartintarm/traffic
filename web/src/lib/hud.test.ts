import { test } from "node:test";
import assert from "node:assert/strict";
import {
  execString,
  panelText,
  perfStatus,
  rushClockText,
  speedString,
  startSpeedLabel,
  statsLines,
  statsText,
} from "./hud.ts";

test("speedString shows the achieved rate only when throttled", () => {
  assert.equal(speedString(3, 3, false), "3×");
  assert.equal(speedString(3, 2.4, true), "2.4×/3× (throttled)");
});

test("execString annotates the thread pool at the parallel crossover", () => {
  assert.equal(execString("gpu", 999, 400), "gpu");
  assert.equal(execString("threads", 500, 400), "threads ▸ parallel (≥400)");
  assert.equal(execString("threads", 399, 400), "threads ▸ serial (<400)");
  assert.equal(execString("threads", 400, 400), "threads ▸ parallel (≥400)");
});

const baseStats = {
  vehicles: 12, crashed: 0, speed: "3×", exec: "gpu",
  idleSkipped: 0, linksQueued: 0, waiting: 0,
};

test("statsLines lists the always-on metrics", () => {
  assert.deepEqual(statsLines(baseStats), ["12 vehicles", "0 crashed", "3×", "gpu"]);
});

test("statsLines appends the situational metrics only when non-zero", () => {
  assert.deepEqual(
    statsLines({ ...baseStats, idleSkipped: 5, linksQueued: 2, waiting: 7 }),
    ["12 vehicles", "0 crashed", "3×", "gpu", "5 idle-skipped", "2 links queued", "7 waiting to enter"],
  );
});

test("statsText bullets each line", () => {
  assert.equal(statsText(["a", "b"]), "• a\n• b");
});

test("perfStatus folds idle-skipped and the routing backend", () => {
  assert.equal(perfStatus("gpu", 0, false), "▶ gpu · routing CPU");
  assert.equal(perfStatus("threads ▸ serial (<400)", 5, true),
    "▶ threads ▸ serial (<400) · 5 idle-skipped · routing GPU");
});

test("rushClockText renders HH:MM and both corridors, rounding flows", () => {
  assert.equal(
    rushClockText(7.5, [1200.4, 800.6, 300, 200]),
    " 07:30 · US-101 N1200/S801 · I-280 N300/S200 veh/h/ln",
  );
  assert.equal(
    rushClockText(17.0, [0, 0, 0, 0]),
    " 17:00 · US-101 N0/S0 · I-280 N0/S0 veh/h/ln",
  );
});

test("panelText converts m/s to the chosen units", () => {
  assert.equal(panelText("I-280 N", [12.7, 20, 1500, 0.5], "mi"),
    "I-280 N — 12 veh · 45 mph · 1500 veh/h · 50% full");
  assert.equal(panelText("I-280 N", [12.7, 20, 1500, 0.5], "km"),
    "I-280 N — 12 veh · 72 km/h · 1500 veh/h · 50% full");
});

test("startSpeedLabel formats a start-speed cap", () => {
  assert.equal(startSpeedLabel(13.4, "mi"), "Start ≤ 30 mph");
  assert.equal(startSpeedLabel(13.4, "km"), "Start ≤ 48 km/h");
});
