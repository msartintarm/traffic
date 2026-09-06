import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, SCHEMA, type SimParams, decodeParams, encodeParams, isDefault } from "./simParams.ts";

test("defaults round-trip through the codec", () => {
  const back = decodeParams(encodeParams(DEFAULTS));
  for (const f of SCHEMA) {
    if (f.kind.t === "q") {
      assert.ok(Math.abs((back[f.key] as number) - (DEFAULTS[f.key] as number)) < 1e-3, `${f.key}`);
    } else {
      assert.equal(back[f.key], DEFAULTS[f.key], `${f.key}`);
    }
  }
  assert.equal(isDefault(back), true);
});

test("a tweaked config round-trips (bools, enum, u16, quantized)", () => {
  const p: SimParams = {
    ...DEFAULTS,
    compute: "serial",
    gpuRouting: false,
    rushHour: true,
    followerLod: true,
    parThreshold: 12000,
    dayCompression: 24,
    demandRate: 3.5,
    gravityBeta: 1.2,
    onRampShare: 0.4,
    roadFunctionWeighting: false,
  };
  const back = decodeParams(encodeParams(p));
  assert.equal(back.compute, "serial");
  assert.equal(back.gpuRouting, false);
  assert.equal(back.rushHour, true);
  assert.equal(back.followerLod, true);
  assert.equal(back.parThreshold, 12000);
  assert.equal(back.dayCompression, 24);
  assert.ok(Math.abs(back.demandRate - 3.5) < 0.01);
  assert.ok(Math.abs(back.gravityBeta - 1.2) < 0.01);
  assert.ok(Math.abs(back.onRampShare - 0.4) < 0.01);
  assert.equal(back.roadFunctionWeighting, false);
  assert.equal(isDefault(back), false);
});

test("garbage or wrong-version blob decodes to defaults", () => {
  assert.equal(isDefault(decodeParams("not-a-real-blob!!!")), true);
  assert.equal(isDefault(decodeParams("")), true);
});
