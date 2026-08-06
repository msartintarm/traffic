import { test } from "node:test";
import assert from "node:assert/strict";
import { distToPolyline, distToSegment, nearestLink } from "./hitTest.ts";

test("distToSegment is perpendicular within the span", () => {
  assert.equal(distToSegment(5, 3, [0, 0], [10, 0]), 3);
});

test("distToSegment clamps to the endpoints beyond the span", () => {
  assert.equal(distToSegment(-4, 0, [0, 0], [10, 0]), 4);
  assert.equal(distToSegment(13, 0, [0, 0], [10, 0]), 3);
});

test("distToSegment handles a degenerate zero-length segment", () => {
  assert.equal(distToSegment(3, 4, [0, 0], [0, 0]), 5);
});

test("distToPolyline takes the nearest of its segments", () => {
  const bend = [[0, 0], [10, 0], [10, 10]];
  assert.equal(distToPolyline(10, 5, bend), 0);
  assert.equal(distToPolyline(5, -2, bend), 2);
});

const links = [
  { pts: [[0, 0], [10, 0]] },
  { pts: [[0, 20], [10, 20]] },
];

test("nearestLink returns the closest link within the radius", () => {
  assert.equal(nearestLink(links, 5, 1, 3), 0);
  assert.equal(nearestLink(links, 5, 19, 3), 1);
});

test("nearestLink returns -1 when nothing is within the radius", () => {
  assert.equal(nearestLink(links, 5, 10, 3), -1);
});

test("nearestLink skips degenerate links and breaks ties to the earliest", () => {
  const withDegenerate = [{ pts: [[5, 0]] }, ...links];
  assert.equal(nearestLink(withDegenerate, 5, 0, 3), 1);
  const coincident = [{ pts: [[0, 0], [10, 0]] }, { pts: [[0, 0], [10, 0]] }];
  assert.equal(nearestLink(coincident, 5, 0, 3), 0);
});
