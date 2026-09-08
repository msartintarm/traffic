"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import { basePath } from "../lib/basePath";
import { isDrag, panDelta, pinch, type Scale } from "../lib/gestures";
import {
  backingSize,
  type Camera,
  cameraFromParams,
  clientToBacking,
  clientToWorld,
  sliderToMpp,
  wheelZoomFactor,
} from "../lib/camera";
import { StatsSmoother, junctionPanelText, panelText, startSpeedLabel, vehicleReportText } from "../lib/hud";
import { type Control, type InitConfig, overlayFromSnapshot } from "../lib/protocol";
import { createSession, type Session } from "../lib/session";
import { SCENARIOS, scenarioName } from "../lib/maps";
import {
  type Compute,
  type Field as ParamField,
  type SimParams,
  DEFAULTS as PARAM_DEFAULTS,
  SCHEMA as PARAM_SCHEMA,
  applyRuntimeParams,
  encodeParams,
  fieldRange,
  isDefault as paramsAreDefault,
  paramsFromUrl,
} from "../lib/simParams";
import styles from "./EngineCanvas.module.css";

/** A show/hide panel: an emoji (optionally labelled) header that toggles a vertically
 * stacked body. Shared by the settings menu and the two corner info overlays so they
 * expand/collapse identically. Uncontrolled — defined at module scope so its open state
 * survives the parent's per-frame re-renders. */
function Collapsible({
  icon,
  label,
  title,
  defaultOpen = true,
  className,
  children,
}: {
  icon: string;
  label?: string;
  title?: string;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={className}>
      <button
        className={styles.overlayToggle}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={title ?? (open ? "Hide" : "Show")}
      >
        {icon}
        {label ? ` ${label}` : ""} {open ? "▾" : "▸"}
      </button>
      {open && <div className={styles.overlayBody}>{children}</div>}
    </div>
  );
}

const ZOOM_RANGE = 60; // fit-out … max-in ratio driving the slider

// Day-clock speeds the slider snaps to (day-seconds per sim second); 1 = real time.
const DAY_COMPRESSION_STEPS = [1, 5, 15, 30, 60, 120, 240];

/** Scale the ASCII overlay's font so its `cols × rows` grid fills the canvas box without
 * distortion. The engine already sized the grid to the view aspect (columns ≈ 2·rows·vw/vh,
 * since monospace cells are ~2:1), so a single font size fits both axes; we take the smaller
 * of the width- and height-limited sizes. `html` is the colourised markup: `cols` is the
 * first row's visible-character count (tags stripped), `rows` the line count. The `data-fit`
 * key skips re-styling on frames where nothing that affects the fit changed. */
function fitAscii(pre: HTMLPreElement, html: string) {
  const box = pre.parentElement;
  if (!box) return;
  const nl = html.indexOf("\n");
  const firstLine = nl < 0 ? html : html.slice(0, nl);
  const cols = firstLine.replace(/<[^>]*>/g, "").length; // visible chars, minus <span> tags
  if (cols === 0) return;
  let rows = 0;
  for (let i = 0; i < html.length; i++) if (html.charCodeAt(i) === 10) rows++;
  if (rows === 0) return;
  const w = box.clientWidth;
  const h = box.clientHeight;
  const key = `${cols}:${rows}:${w}:${h}`;
  if (pre.dataset.fit === key) return;
  pre.dataset.fit = key;
  const CHAR_ADVANCE = 0.62; // monospace glyph advance as a fraction of font size (slight over-estimate avoids edge clipping)
  const fontPx = Math.max(2, Math.min(w / (cols * CHAR_ADVANCE), h / rows));
  pre.style.fontSize = `${fontPx}px`;
  pre.style.lineHeight = `${fontPx}px`;
}

export default function EngineCanvas() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<Session | null>(null);
  const bootedRef = useRef(false); // boot the engine once even under React StrictMode's double-mount
  const cameraRef = useRef<Camera | null>(null); // last frame's camera, for client→world input transforms
  const sliderRef = useRef<HTMLInputElement>(null);
  const statsRef = useRef<HTMLSpanElement>(null);
  // Glides the HUD's jittery per-frame counts (EMA + deadband) across frames.
  const smootherRef = useRef(new StatsSmoother());
  // Loading-screen live counts, glided through their own smoother by a rAF loop
  // (worker progress messages land in bursts; raw numbers would teleport).
  const loadingCountsRef = useRef<{ done: number; total: number; unit: string } | null>(null);
  const loadingDetailRef = useRef<HTMLSpanElement>(null);
  const loadingSmootherRef = useRef(new StatsSmoother());
  const tipRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const asciiRef = useRef<HTMLPreElement>(null); // the ASCII "terminal" overlay (shown when asciiMode is on)
  const fitMppRef = useRef(1);
  const [ready, setReady] = useState(false);
  const [loadingFraction, setLoadingFraction] = useState(0); // boot progress 0→1 for the loading bar
  const [loadingStage, setLoadingStage] = useState("Starting…");
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [backend, setBackend] = useState("");
  const [mapLabel, setMapLabel] = useState("");
  const [scenario, setScenario] = useState("millbrae");
  // "splash" (no `?scenario=`: show the picker) · "sim" (boot the chosen scene) · null
  // (undetermined on the server / first client paint, before the URL is read).
  const [route, setRoute] = useState<"splash" | "sim" | null>(null);
  const [highwayTraffic, setHighwayTraffic] = useState(true);
  const [surfaceTraffic, setSurfaceTraffic] = useState(true);
  const [rushHour, setRushHour] = useState(false);
  const [dayCompression, setDayCompression] = useState(60); // engine DEFAULT_DAY_COMPRESSION
  const [rampMetering, setRampMetering] = useState(true); // engine default: on (D4 peak windows)
  const [transit, setTransit] = useState(true); // engine default: on (real timetables when the artifact loads)
  const rushClockRef = useRef<HTMLSpanElement>(null);
  const [accelBackend, setAccelBackend] = useState("serial");
  const [parThreshold, setParThreshold] = useState(500); // matches engine DEFAULT_PAR_THRESHOLD
  const [parallelRouting, setParallelRouting] = useState(true); // on by default (see net_world default)
  const [demandRate, setDemandRate] = useState(1);
  const warmupRef = useRef<HTMLSpanElement>(null); // live pre-population progress under the demand panel
  const [congestionEnabled, setCongestionEnabled] = useState(false);
  const [congestionEngage, setCongestionEngage] = useState(0.85);
  const [sleepScheduler, setSleepScheduler] = useState(true); // on by default (see bridge assemble)
  const [smoothPlayback, setSmoothPlayback] = useState(true); // frame budget on: smooth view, sim slows under load
  const [showCrashes, setShowCrashes] = useState(false); // crash-location overlay, off by default
  const [cacheSort, setCacheSort] = useState(true); // cache-friendly sort, on by default (see net_world default)
  const [stopCostRouting, setStopCostRouting] = useState(true); // control-aware routing, on by default (see net_world default)
  const [laneEvalStagger, setLaneEvalStagger] = useState(true); // human-cadence lane decisions, on by default (see net_world default)
  const [arterialRouting, setArterialRouting] = useState(false); // arterial-first fields, off by default (see net_world default)
  const [targetedRouting, setTargetedRouting] = useState(true); // targeted route refresh, on by default (see net_world default)
  const [localitySort, setLocalitySort] = useState(false); // fleet memory-locality reorder, off by default (see net_world default)
  const [sharding, setSharding] = useState(true); // sharded (SPMD) execution, on by default; `?shard=0` (splash toggle) opts out
  const [asyncRouting, setAsyncRouting] = useState(true); // background reroute solves, on by default (threads backend only)
  const [followerLod, setFollowerLod] = useState(false); // front-of-lane LOD toggle; measured ~0% runtime benefit on SF, so off by default (deadlock now fixed, so safe if wanted)
  const [localRouting, setLocalRouting] = useState(true); // per-driver bounded-search routing, default on (map-size-independent, measured faster than the field)
  const [shardStats, setShardStats] = useState(false); // per-thread work rows in the diagnostics overlay
  // Mirror the boot config (from `?c=`) into the in-play menu so its toggles match
  // what actually launched — otherwise the menu would show its hardcoded defaults.
  const syncMenuFromParams = (p: SimParams) => {
    setRushHour(p.rushHour);
    setDayCompression(p.dayCompression);
    setRampMetering(p.rampMetering);
    setTransit(p.transit);
    setParThreshold(p.parThreshold);
    setParallelRouting(p.parallelRouting);
    setDemandRate(p.demandRate);
    setCongestionEngage(p.congestionEngage);
    setSleepScheduler(p.sleepScheduler);
    setCacheSort(p.cacheSort);
    setStopCostRouting(p.stopCostRouting);
    setLaneEvalStagger(p.laneEvalStagger);
    setArterialRouting(p.arterialRouting);
    setTargetedRouting(p.targetedRouting);
    setLocalitySort(p.localitySort);
    setSharding(p.sharded);
    setAsyncRouting(p.asyncRouting);
    setFollowerLod(p.followerLod);
    setLocalRouting(p.localRouting);
  };
  const [startSpeedMps, setStartSpeedMps] = useState(36); // ≥ every road limit ⇒ "enter at limit"
  const [units, setUnits] = useState<"mi" | "km">("mi");
  const unitsRef = useRef<"mi" | "km">("mi"); // read inside the per-frame HUD update (avoids stale closure)
  const [asciiMode, setAsciiMode] = useState(false); // render the sim as an ASCII grid instead of pixels
  const [isFullscreen, setIsFullscreen] = useState(false);
  // iPhone Safari has no element Fullscreen API at all, so we fall back to a CSS overlay
  // (`position: fixed` filling the viewport) tracked by this flag.
  const [pseudoFs, setPseudoFs] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setScenario(params.get("scenario") ?? "millbrae");
    setRoute(params.has("scenario") ? "sim" : "splash");
  }, []);

  // Pseudo-fullscreen (iPhone Safari): the wrapper's CSS class changes the layout, so nudge
  // the canvas to resize to the new viewport and stop the page behind it from scrolling.
  useEffect(() => {
    window.dispatchEvent(new Event("resize"));
    if (!pseudoFs) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [pseudoFs]);

  // While loading, glide the stage's item counter through the same EMA+deadband
  // the live HUD uses, so bursty worker progress reads as steady motion.
  useEffect(() => {
    if (ready || route !== "sim") return;
    let raf = 0;
    const tick = () => {
      const c = loadingCountsRef.current;
      const el = loadingDetailRef.current;
      if (el && c) {
        const shown = loadingSmootherRef.current.count(`load:${c.unit}`, c.done, 0.15);
        el.textContent = `${shown.toLocaleString()} of ${c.total.toLocaleString()} ${c.unit}`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [ready, route]);

  useEffect(() => {
    if (route !== "sim") return; // splash / undetermined: no scene to boot
    const canvas = canvasRef.current!;
    // Any-button drag pans; a press that never moves past the threshold is a click
    // (road selection). Touch mirrors this: one-finger drag pans, a tap selects.
    const DRAG_THRESHOLD = 5; // CSS px before a press becomes a drag rather than a click
    const pointer = { pressed: false, dragged: false, down: { x: 0, y: 0 } };
    const touch = {
      mode: "none",
      prev: { x: 0, y: 0 }, // last single-finger point
      tap: { x: 0, y: 0 }, // where a one-finger gesture started (tap-vs-drag origin)
      dragged: false,
      a: { x: 0, y: 0 }, // last two-finger points, for pinch
      b: { x: 0, y: 0 },
    };

    const apply = (c: Control) => sessionRef.current?.applyControl(c);

    // The current client→backing/world transform, from the on-screen rect and the last
    // frame's camera. Null until the first frame arrives (then input becomes live).
    const frame = (): { rect: DOMRect; cam: Camera; scale: Scale } | null => {
      const cam = cameraRef.current;
      if (!cam) return null;
      const rect = canvas.getBoundingClientRect();
      return { rect, cam, scale: { sx: cam.vw / rect.width, sy: cam.vh / rect.height, left: rect.left, top: rect.top } };
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const f = frame();
      if (!f) return;
      const { bx, by } = clientToBacking(e.clientX, e.clientY, f.rect, f.scale);
      apply({ type: "zoomAt", factor: wheelZoomFactor(e.deltaY), bx, by });
    };
    const onContext = (e: MouseEvent) => e.preventDefault();
    const onDown = (e: MouseEvent) => {
      pointer.pressed = true;
      pointer.dragged = false;
      pointer.down = { x: e.clientX, y: e.clientY };
    };
    const onMove = (e: MouseEvent) => {
      const f = frame();
      if (!f) return;
      if (pointer.pressed) {
        if (!pointer.dragged && isDrag(pointer.down, { x: e.clientX, y: e.clientY }, DRAG_THRESHOLD)) {
          pointer.dragged = true;
          if (tipRef.current) tipRef.current.style.display = "none";
        }
        if (pointer.dragged) {
          apply({ type: "panBy", dx: e.movementX * f.scale.sx, dy: e.movementY * f.scale.sy });
          return;
        }
      }
      // Hover: map the cursor to world space; the worker names the nearest road.
      const { wx, wy } = clientToWorld(e.clientX, e.clientY, f.rect, f.scale, f.cam);
      apply({ type: "hover", wx, wy, x: e.clientX, y: e.clientY });
    };
    const onLeave = () => {
      if (tipRef.current) tipRef.current.style.display = "none";
    };
    const onClick = (e: MouseEvent) => {
      // Suppress selection when the press was a drag (a pan), not a click.
      if (e.button !== 0 || pointer.dragged) return;
      const f = frame();
      if (!f) return;
      const { wx, wy } = clientToWorld(e.clientX, e.clientY, f.rect, f.scale, f.cam);
      apply({ type: "select", wx, wy, radius: 12 });
    };
    const onUp = () => {
      pointer.pressed = false;
    };
    // Escape releases a follow-camera; a manual pan (below) does too, so the user
    // can always break free of a tracked car.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") apply({ type: "releaseFollow" });
    };

    // Touch: one finger pans (a tap selects); two fingers pinch-zoom toward the
    // midpoint and pan by the midpoint's movement.
    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        const t = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        touch.mode = "pan";
        touch.dragged = false;
        touch.prev = t;
        touch.tap = t;
      } else if (e.touches.length >= 2) {
        touch.mode = "pinch";
        touch.a = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        touch.b = { x: e.touches[1].clientX, y: e.touches[1].clientY };
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const f = frame();
      if (!f) return;
      if (touch.mode === "pan" && e.touches.length === 1) {
        const cur = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        const d = panDelta(touch.prev, cur, f.scale);
        apply({ type: "panBy", dx: d.x, dy: d.y });
        touch.prev = cur;
        if (isDrag(touch.tap, cur, DRAG_THRESHOLD)) touch.dragged = true;
      } else if (e.touches.length >= 2) {
        touch.mode = "pinch";
        const curA = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        const curB = { x: e.touches[1].clientX, y: e.touches[1].clientY };
        const g = pinch(touch.a, touch.b, curA, curB, f.scale);
        apply({ type: "zoomAt", factor: g.factor, bx: g.focusX, by: g.focusY }); // zoom toward the pinch midpoint
        apply({ type: "panBy", dx: g.panX, dy: g.panY }); // and follow the midpoint's drag
        touch.a = curA;
        touch.b = curB;
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (touch.mode === "pan" && !touch.dragged) {
        const f = frame();
        if (f) {
          const { wx, wy } = clientToWorld(touch.tap.x, touch.tap.y, f.rect, f.scale, f.cam);
          apply({ type: "select", wx, wy, radius: 20 }); // a more forgiving tap radius on touch
        }
      }
      if (e.touches.length === 0) {
        touch.mode = "none";
      } else if (e.touches.length === 1) {
        // Lifting one finger of a pinch drops back to single-finger panning.
        touch.mode = "pan";
        touch.dragged = true; // continuing a gesture, not a tap
        touch.prev = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    };

    // Size the backing store to the displayed CSS size × devicePixelRatio (capped), so the
    // canvas stays crisp on HiDPI and under browser zoom. The worker owns the backing store
    // (it may be a transferred OffscreenCanvas), so we post the target size, not set it here.
    const resizeCanvas = () => {
      const { w, h } = backingSize(canvas.clientWidth || 900, canvas.clientHeight || 600, window.devicePixelRatio || 1);
      apply({ type: "resize", w, h });
    };
    const onFsChange = () => {
      const doc = document as Document & { webkitFullscreenElement?: Element | null };
      setIsFullscreen(!!(document.fullscreenElement ?? doc.webkitFullscreenElement));
      resizeCanvas();
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("contextmenu", onContext);
    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mouseleave", onLeave);
    canvas.addEventListener("click", onClick);
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd);
    canvas.addEventListener("touchcancel", onTouchEnd);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", resizeCanvas);
    document.addEventListener("fullscreenchange", onFsChange);
    document.addEventListener("webkitfullscreenchange", onFsChange);

    // Boot the engine session exactly once. StrictMode mounts → cleans up → mounts again on
    // the same instance; the OffscreenCanvas transfer is one-shot, so guard it. Every
    // scenario/compute change is a full page navigation, so cleanup only detaches listeners.
    if (!bootedRef.current) {
      bootedRef.current = true;
      const params = new URLSearchParams(window.location.search);
      const scenarioKey = params.get("scenario") ?? "millbrae";
      // All boot levers come from the compact `?c=` config blob (or the legacy
      // readable params, for old links) — one source of truth shared with the
      // splash "Params" menu. See `lib/simParams`.
      const p = paramsFromUrl(window.location.search);
      // The compute backend dictates which wasm module the worker loads (CPU-threads is a
      // separate atomics-enabled artifact).
      const compute = p.compute;
      setAccelBackend(compute);
      const gpu = p.gpuRouting;
      const splitJunctions = p.splitJunctions;
      // Mirror the boot config into the in-play menu's state so its toggles match.
      syncMenuFromParams(p);
      (async () => {
        // Threads need cross-origin isolation (COOP/COEP via the shim SW, which reloads once).
        // Establish it here, before the worker boots and checks `crossOriginIsolated`.
        if (compute === "threads") await ensureCrossOriginIsolation();
        const { w, h } = backingSize(canvas.clientWidth || 900, canvas.clientHeight || 600, window.devicePixelRatio || 1);
        const config: InitConfig = {
          scenario: scenarioKey,
          compute,
          gpu,
          splitJunctions,
          basePath: basePath(),
          width: w,
          height: h,
          congestionEngage: p.congestionEngage,
          zoomRange: ZOOM_RANGE,
          debug: params.get("debug") === "1",
        };
        sessionRef.current = createSession(canvas, config, {
          onReady: (r) => {
            setBackend(r.backend);
            setMapLabel(r.mapLabel);
            fitMppRef.current = r.fitMpp;
            setCongestionEnabled(r.congestionEnabled);
            setReady(true);
            // Apply every boot lever from the config: runtime controls, the boot
            // toggles (sharding / local routing / pre-populate), and the demand-
            // generation tuning. Applying at default is a no-op in the engine, so
            // this drives the whole config in one pass. (`local routing` is also the
            // engine boot default, set before the router install so no flow-field is
            // built; re-asserting it here is harmless.)
            applyRuntimeParams(p, (c) => sessionRef.current?.applyControl(c as Control));
          },
          onFrame: (f) => {
            fitMppRef.current = f.fitMpp;
            cameraRef.current = cameraFromParams(f.snapshot.camera);
            if (params.get("debug") === "1") {
              (window as { __stats?: unknown }).__stats = f.snapshot;
              (window as { __ctl?: unknown }).__ctl = (c: Control) => sessionRef.current?.applyControl(c);
            }
            const pre = asciiRef.current;
            const box = pre?.parentElement;
            if (pre && box) {
              if (f.ascii != null) {
                pre.innerHTML = f.ascii; // engine-generated <span> runs; colours + safe chars only
                box.style.display = "flex";
                fitAscii(pre, f.ascii);
              } else if (box.style.display !== "none") {
                box.style.display = "none";
              }
            }
            const o = overlayFromSnapshot(f.snapshot, { fitMpp: f.fitMpp, zoomRange: ZOOM_RANGE }, smootherRef.current);
            if (statsRef.current) statsRef.current.textContent = o.stats;
            if (warmupRef.current) {
              const w = f.snapshot.warmup;
              warmupRef.current.textContent = w
                ? `pre-populating… ${Math.round((100 * w[0]) / w[1])}% · ${Math.round(w[0] / 60)} sim-min · ${w[2]} vehicles`
                : "";
            }
            if (rushClockRef.current) rushClockRef.current.textContent = o.rushClock ?? "";
            if (sliderRef.current) sliderRef.current.value = String(o.sliderValue);
            if (panelRef.current) {
              if (f.selected) {
                panelRef.current.textContent =
                  f.selected.kind === "junction"
                    ? junctionPanelText(f.selected.name, f.selected.control, f.selected.stats)
                    : f.selected.kind === "vehicle"
                      ? vehicleReportText(f.selected.name, f.selected.stats, f.selected.report, unitsRef.current)
                      : panelText(f.selected.name, f.selected.stats, unitsRef.current);
                panelRef.current.style.display = "block";
              } else {
                panelRef.current.style.display = "none";
              }
            }
          },
          onProgress: (fraction, stage, counts) => {
            setLoadingFraction(fraction);
            setLoadingStage(stage);
            loadingCountsRef.current = counts ?? null;
            if (!counts && loadingDetailRef.current) loadingDetailRef.current.textContent = "";
          },
          onHover: (name, x, y) => {
            const tip = tipRef.current;
            if (!tip) return;
            if (name) {
              tip.textContent = name;
              tip.style.left = `${x + 12}px`;
              tip.style.top = `${y + 12}px`;
              tip.style.display = "block";
            } else {
              tip.style.display = "none";
            }
          },
          onFatal: (message) => setError(message),
        });
      })();
    }

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("contextmenu", onContext);
      canvas.removeEventListener("mousedown", onDown);
      canvas.removeEventListener("mouseleave", onLeave);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("touchcancel", onTouchEnd);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", resizeCanvas);
      document.removeEventListener("fullscreenchange", onFsChange);
      document.removeEventListener("webkitfullscreenchange", onFsChange);
    };
  }, [route]);

  const toggle = () => {
    const s = sessionRef.current;
    if (!s) return;
    s.applyControl({ type: playing ? "pause" : "play" });
    setPlaying(!playing);
  };

  const toggleFullscreen = async () => {
    const el = wrapperRef.current as
      | (HTMLDivElement & { webkitRequestFullscreen?: () => Promise<void> | void })
      | null;
    if (!el) return;
    const doc = document as Document & {
      webkitFullscreenElement?: Element | null;
      webkitExitFullscreen?: () => Promise<void> | void;
    };
    const request = el.requestFullscreen?.bind(el) ?? el.webkitRequestFullscreen?.bind(el);
    // No native Fullscreen API (iPhone Safari): toggle the CSS-overlay fallback instead.
    if (!request) {
      setPseudoFs((v) => !v);
      return;
    }
    const active = document.fullscreenElement ?? doc.webkitFullscreenElement ?? null;
    const exit = doc.exitFullscreen?.bind(doc) ?? doc.webkitExitFullscreen?.bind(doc);
    try {
      if (!active) {
        await request();
        // Prefer landscape on devices that can rotate (phones/tablets); harmless
        // no-op / rejected promise on desktop, so swallow it.
        try {
          await (screen.orientation as unknown as { lock?: (o: string) => Promise<void> })?.lock?.("landscape");
        } catch {}
      } else {
        try {
          (screen.orientation as unknown as { unlock?: () => void })?.unlock?.();
        } catch {}
        await exit?.();
      }
    } catch {}
  };

  // No `?scenario=` yet: show the picker (which also warms up cross-origin isolation)
  // instead of booting a scene. `null` is the pre-URL paint — render the empty frame so
  // there's no splash flash before we know which way to go.
  if (route === "splash") return <SplashScreen />;
  if (route === null) return <div className={styles.wrapper} aria-busy="true" />;

  return (
    <div ref={wrapperRef} className={`${styles.wrapper}${pseudoFs ? ` ${styles.pseudoFullscreen}` : ""}`}>
      <canvas ref={canvasRef} width={900} height={600} className={styles.canvas} />
      {/* ASCII "terminal" overlay: the container centres the grid and is shown by onFrame
          when the mode is on; the inner <pre> holds the per-frame colourised HTML.
          pointer-events:none so pan/zoom/hover still reach the canvas underneath. */}
      <div className={styles.ascii} aria-hidden="true">
        <pre ref={asciiRef} className={styles.asciiGrid} />
      </div>

      {!ready && !error && (
        <div className={styles.loading} role="status" aria-live="polite">
          <div className={styles.loadingInner}>
            <div className={styles.loadingName}>{scenarioName(scenario)}</div>
            <div className={styles.loadingTrack}>
              <div className={styles.loadingFill} style={{ width: `${Math.round(loadingFraction * 100)}%` }} />
            </div>
            <div className={styles.loadingStage}>{loadingStage}</div>
            <div className={styles.loadingDetail}>
              <span ref={loadingDetailRef} />
            </div>
          </div>
        </div>
      )}

      <div className={styles.topLeft}>
        <div className={styles.controls}>
          <button className={styles.button} onClick={toggle} disabled={!ready}>
            {playing ? "Pause" : "Play"}
          </button>
          {[1, 2, 8, 16, 32].map((s) => (
            <button
              key={s}
              className={styles.button}
              disabled={!ready}
              onClick={() => sessionRef.current?.applyControl({ type: "speed", value: s })}
            >
              {s}×
            </button>
          ))}
          <button
            className={styles.button}
            onClick={toggleFullscreen}
            title="Toggle fullscreen (landscape on mobile)"
            aria-label={isFullscreen || pseudoFs ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen || pseudoFs ? "⛶ Exit" : "⛶"}
          </button>
        </div>
        <Collapsible
          className={styles.settings}
          icon="⚙️"
          label="Settings"
          defaultOpen={false}
          title="Show or hide settings"
        >
            <label className={styles.zoomLabel}>
              Zoom
              <input
                ref={sliderRef}
                className={styles.slider}
                type="range"
                min={0}
                max={1000}
                defaultValue={0}
                disabled={!ready}
                onInput={() => {
                  const slider = sliderRef.current;
                  if (!slider) return;
                  sessionRef.current?.applyControl({
                    type: "metersPerPixel",
                    value: sliderToMpp(Number(slider.value) / 1000, fitMppRef.current, ZOOM_RANGE),
                  });
                }}
              />
            </label>
            <button
              className={styles.button}
              disabled={!ready}
              onClick={() => sessionRef.current?.applyControl({ type: "fit" })}
            >
              Fit
            </button>
            <select
              className={`${styles.button} ${styles.mapSelect}`}
              value={scenario}
              onChange={(e) => {
                const url = new URL(window.location.href);
                url.searchParams.set("scenario", e.target.value);
                window.location.href = url.toString();
              }}
            >
              <option value="millbrae">Millbrae (real map)</option>
              <option value="sancarlos">San Carlos (real map)</option>
              <option value="sf">San Francisco (real map)</option>
              <option value="peninsula">Bay Area Peninsula (real map)</option>
              <option value="columbus">Columbus, OH (real map)</option>
              <option value="arterial">Test: arterial junction</option>
              <option value="corridor">Test: signal corridor</option>
              <option value="gridlock">Test: gridlock (fast jam)</option>
            </select>
            <label className={styles.zoomLabel} title="Freeway through-traffic: enters at a highway gateway, bound for the far end of the same highway, another highway exit, or a surface street.">
              <input
                type="checkbox"
                checked={highwayTraffic}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "demandSources", north: e.target.checked, south: surfaceTraffic });
                  setHighwayTraffic(e.target.checked);
                }}
              />
              Highway traffic
            </label>
            <label className={styles.zoomLabel} title="Local/arterial traffic on the surface streets (through the city, arriving, leaving, and internal trips).">
              <input
                type="checkbox"
                checked={surfaceTraffic}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "demandSources", north: highwayTraffic, south: e.target.checked });
                  setSurfaceTraffic(e.target.checked);
                }}
              />
              Surface traffic
            </label>
            <label
              className={styles.zoomLabel}
              title="Drive the whole map by the simulated time of day: US-101 and I-280 at their real per-lane volumes (Caltrans PeMS typical-weekday, split by direction), and the surface streets by the urban-arterial diurnal shape. Both build and fade with the commute as the day advances."
            >
              <input
                type="checkbox"
                checked={rushHour}
                disabled={!ready || (!highwayTraffic && !surfaceTraffic)}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "rushHour", value: e.target.checked });
                  setRushHour(e.target.checked);
                }}
              />
              Rush hour
              {rushHour && highwayTraffic && <span ref={rushClockRef} className={styles.rushClock} />}
            </label>
            <label className={styles.zoomLabel} title="ALINEA ramp meters on the freeway on-ramps: one car per green, paced by mainline occupancy. Under rush hour they run the real Caltrans D4 peak windows (6–10 h, 15–19 h).">
              <input
                type="checkbox"
                checked={rampMetering}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "rampMetering", value: e.target.checked });
                  setRampMetering(e.target.checked);
                }}
              />
              Ramp meters
            </label>
            <label className={styles.zoomLabel} title="Real transit from GTFS timetables: trains run their scheduled trips on the rail corridor (level crossings close for actual passages), and buses follow their published departures with timepoint holding. Off reverts to synthetic crossing cadences and headway buses.">
              <input
                type="checkbox"
                checked={transit}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "transit", value: e.target.checked });
                  setTransit(e.target.checked);
                }}
              />
              Transit
            </label>
            {rushHour && (
              <label className={styles.zoomLabel} title="How fast the simulated day plays: 1× is real time (accuracy mode), 60× plays the 24 h in ~24 min. Only the day clock scales — traffic dynamics always run in real time.">
                Day {dayCompression}×
                <input
                  type="range"
                  min={0}
                  max={DAY_COMPRESSION_STEPS.length - 1}
                  step={1}
                  value={DAY_COMPRESSION_STEPS.indexOf(dayCompression)}
                  disabled={!ready}
                  onChange={(e) => {
                    const v = DAY_COMPRESSION_STEPS[Number(e.target.value)];
                    sessionRef.current?.applyControl({ type: "dayCompression", value: v });
                    setDayCompression(v);
                  }}
                />
              </label>
            )}
            <label className={styles.zoomLabel} title="Spawn-rate multiplier applied to every enabled traffic stream.">
              Rate {demandRate.toFixed(2)}×
              <input
                type="range"
                min={0}
                max={4}
                step={0.25}
                value={demandRate}
                disabled={!ready}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  sessionRef.current?.applyControl({ type: "demandRate", value: v });
                  setDemandRate(v);
                }}
              />
            </label>
            <button
              className={styles.button}
              disabled={!ready}
              title="Fast-forward up to one simulated hour of travel headlessly, so the roads start realistically full instead of empty: the day clock holds at the current hour (traffic fills in at this hour's demand) and the run stops early once the network reaches steady state. The page stays responsive; watch progress below. Click again to stop early and keep whatever traffic exists."
              onClick={() => {
                const active = !!warmupRef.current?.textContent;
                sessionRef.current?.applyControl({ type: "warmup", seconds: active ? 0 : 3600 });
              }}
            >
              Pre-populate traffic (≤1 sim-hour)
            </button>
            <span ref={warmupRef} className={styles.zoomLabel} />
            <label className={styles.zoomLabel} title="Cap on the speed vehicles enter the map at — still never above the origin road's own speed limit.">
              {startSpeedLabel(startSpeedMps, units)}
              <input
                type="range"
                min={0}
                max={36}
                step={1}
                value={startSpeedMps}
                disabled={!ready}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  sessionRef.current?.applyControl({ type: "entrySpeedCap", value: v });
                  setStartSpeedMps(v);
                }}
              />
            </label>
            <select
              className={styles.button}
              value={units}
              disabled={!ready}
              title="Speed units for the road readout and the start-speed control."
              onChange={(e) => {
                const v = e.target.value as "mi" | "km";
                unitsRef.current = v;
                setUnits(v);
              }}
            >
              <option value="mi">Units: mph</option>
              <option value="km">Units: km/h</option>
            </select>
            <label
              className={styles.zoomLabel}
              title="Render the current view as an ASCII grid — the exact road geometry (#) and vehicle placement (@) the engine draws, as text. Pan, zoom, and hover still work through it."
            >
              <input
                type="checkbox"
                checked={asciiMode}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "ascii", value: e.target.checked });
                  setAsciiMode(e.target.checked);
                }}
              />
              ASCII render
            </label>
        </Collapsible>
        <Collapsible
          className={styles.settings}
          icon="⚡"
          label="Performance"
          defaultOpen={false}
          title="Show or hide performance controls"
        >
            <select
              className={styles.button}
              value={accelBackend}
              disabled={!ready}
              title="Executor for the per-vehicle step. Switching reloads the page (the CPU-threads build is a separate wasm module). Unavailable choices fall back to serial — see the active backend in the status line."
              onChange={(e) => {
                // The compute backend dictates which wasm module loads, so it can't be
                // switched in place — reload with the choice as a query param.
                const url = new URL(window.location.href);
                url.searchParams.set("compute", e.target.value);
                window.location.href = url.toString();
              }}
            >
              <option value="serial">Compute: serial (1 core)</option>
              <option value="threads">Compute: CPU threads</option>
              <option value="gpu">Compute: GPU</option>
            </select>
            {accelBackend === "threads" && (
              <>
                <label
                  className={styles.zoomLabel}
                  title="Serial ↔ threads crossover: below this car count the step runs serial (rayon overhead isn't worth it); at/above it, CPU threads parallelize. Set it above the live car count to force serial, below to force threads — a lever to A/B the two at a fixed load."
                >
                  Parallelize ≥
                  <input
                    className={styles.button}
                    type="number"
                    min={0}
                    step={500}
                    value={parThreshold}
                    disabled={!ready}
                    style={{ width: "5.5em" }}
                    onChange={(e) => {
                      const v = Math.max(0, Math.floor(Number(e.target.value) || 0));
                      sessionRef.current?.applyControl({ type: "parThreshold", value: v });
                      setParThreshold(v);
                    }}
                  />
                  cars → threads
                </label>
                <label
                  className={styles.zoomLabel}
                  title="Solve several routing (flow-field) recompute destinations at once across the worker pool, instead of one at a time. Spreads a whole-map reroute over cores, cutting its per-frame cost by ~the core count. On by default; uncheck to A/B against serial routing. Only affects the CPU-threads build's routing path."
                >
                  <input
                    type="checkbox"
                    checked={parallelRouting}
                    disabled={!ready}
                    onChange={(e) => {
                      sessionRef.current?.applyControl({ type: "parallelRouting", value: e.target.checked });
                      setParallelRouting(e.target.checked);
                    }}
                  />
                  Parallel routing
                </label>
              </>
            )}
            <label
              className={styles.zoomLabel}
              title="Under heavy load, cap the simulation to a per-frame time budget so the view stays smooth — the sim then runs slower than the selected speed (it shows as 'throttled'). Uncheck to run the simulation at the full selected speed instead, letting the frame rate drop when a step can't finish in time."
            >
              <input
                type="checkbox"
                checked={smoothPlayback}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "frameBudget", value: e.target.checked });
                  setSmoothPlayback(e.target.checked);
                }}
              />
              Smooth playback
            </label>
            <label
              className={styles.zoomLabel}
              title="Drivers plan around control delay: a link ending at a stop sign costs ~9 s extra (a yield ~3 s) in the route planner, keeping through-traffic on the arterials instead of rat-running the stop-sign grid — the way real drivers path. Applies from the next reroute cycle. On by default."
            >
              <input
                type="checkbox"
                checked={stopCostRouting}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "stopCostRouting", value: e.target.checked });
                  setStopCostRouting(e.target.checked);
                }}
              />
              Stop-sign-aware routing
            </label>
            <label
              className={styles.zoomLabel}
              title="Away from a junction (or standing in queue) a driver re-weighs a discretionary lane change about once a second, not five times — spread those scans on a ~1 s staggered cadence. Mandatory turn positioning near intersections still evaluates every tick. Cuts the lane-change phase substantially at city scale. On by default."
            >
              <input
                type="checkbox"
                checked={laneEvalStagger}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "laneEvalStagger", value: e.target.checked });
                  setLaneEvalStagger(e.target.checked);
                }}
              />
              Relaxed lane decisions
            </label>
            <label
              className={styles.zoomLabel}
              title="Plan trips the way drivers do: over the arterial network, with local streets only near each trip's ends (a car deep in local fabric first heads for a main road). Roughly halves routing solve and startup cost and shifts through-traffic onto arterials. Toggling rebuilds the router — expect a brief pause on a city map. Off by default."
            >
              <input
                type="checkbox"
                checked={arterialRouting}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "arterialRouting", value: e.target.checked });
                  setArterialRouting(e.target.checked);
                }}
              />
              Arterial-first routing
            </label>
            <label
              className={styles.zoomLabel}
              title="Refresh routing the way it's actually read: skip destinations whose routes still price correctly, and solve the rest only far enough to cover the cars (and entrances) that will query them — untouched parts of a field keep their previous, still-valid answers, and every field is fully refreshed on a bounded cadence. Cuts routing recompute work severalfold and makes rerouting react faster. On by default — uncheck to run every cycle exhaustively."
            >
              <input
                type="checkbox"
                checked={targetedRouting}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "targetedRouting", value: e.target.checked });
                  setTargetedRouting(e.target.checked);
                }}
              />
              Targeted route refresh
            </label>
            <label
              className={styles.zoomLabel}
              title="Periodically reorder the vehicle array by road position so each car's neighbor lookups read adjacent memory — measurably faster on big fleets (~6% on the serial engine). Changes nothing visible; simulation tie-breaks may differ marginally from the unsorted order. Off by default."
            >
              <input
                type="checkbox"
                checked={localitySort}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "localitySort", value: e.target.checked });
                  setLocalitySort(e.target.checked);
                }}
              />
              Memory locality sort
            </label>
            <label
              className={styles.zoomLabel}
              title="Split intersection/boundary resolution into 8 spatial shards (whole junction clusters each) resolved on separate cores. Deterministic, but tie-breaks can differ from the unsharded engine. Measured slightly slower on the development machine — exposed as an experiment for hardware where the parallel win beats the dispatch overhead. Off by default."
            >
              <input
                type="checkbox"
                checked={sharding}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "sharding", value: e.target.checked });
                  setSharding(e.target.checked);
                }}
              />
              Sharded intersections
            </label>
            <label
              className={styles.zoomLabel}
              title="Solve rerouting flow-fields on a spare CPU core in the background instead of spreading the work across simulation ticks — routes update when a solve lands. Only active on the threads compute backend. On by default."
            >
              <input
                type="checkbox"
                checked={asyncRouting}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "asyncRouting", value: e.target.checked });
                  setAsyncRouting(e.target.checked);
                }}
              />
              Background rerouting
            </label>
            <label
              className={styles.zoomLabel}
              title="Front-of-lane LOD: a car queued behind another skips the right-of-way and box scans its leader already runs, so a driver's decision cost stops scaling with queue depth and junction size. Off by default — it slightly changes how intersections discharge (a prototype under evaluation)."
            >
              <input
                type="checkbox"
                checked={followerLod}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "followerLod", value: e.target.checked });
                  setFollowerLod(e.target.checked);
                }}
              />
              Front-of-lane LOD
            </label>
            <label
              className={styles.zoomLabel}
              title="Route each driver by a bounded local search toward the destination (ALT landmarks) instead of a global flow-field — a driver only searches its own neighbourhood, as a real driver would, so per-driver cost is independent of map size. Delivers cars identically to the field, is congestion-aware (steers around jams), and is faster on large maps. On by default; uncheck for the classic global router. Rebuilds routing when toggled."
            >
              <input
                type="checkbox"
                checked={localRouting}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "localRouting", value: e.target.checked });
                  setLocalRouting(e.target.checked);
                }}
              />
              Local routing
            </label>
            <label
              className={styles.zoomLabel}
              title="Cache-friendly sort: order the per-lane vehicle groups (for following, lane changes, and crash checks) against a compact position array instead of reading a full vehicle record per comparison. The simulation result is identical; this is purely a speed option. On by default — uncheck to compare."
            >
              <input
                type="checkbox"
                checked={cacheSort}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "cacheSort", value: e.target.checked });
                  setCacheSort(e.target.checked);
                }}
              />
              Cache-friendly sort
            </label>
            <label
              className={styles.zoomLabel}
              title="Mark where collisions happen with a persistent amber diamond at each crash site (visible at every zoom), so you can spot problem intersections. Off by default. Use Clear to reset the markers."
            >
              <input
                type="checkbox"
                checked={showCrashes}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "showCrashes", value: e.target.checked });
                  setShowCrashes(e.target.checked);
                }}
              />
              Show crashes
              {showCrashes && (
                <button
                  type="button"
                  className={styles.button}
                  disabled={!ready}
                  style={{ marginLeft: "0.5em" }}
                  onClick={() => sessionRef.current?.applyControl({ type: "clearCrashes" })}
                >
                  Clear
                </button>
              )}
            </label>
            <label
              className={styles.zoomLabel}
              title="Congestion level-of-detail: while a link stays saturated, its queued cars use cheap leader-only car-following instead of the full model, and skip lane-change checks. Cars stay individual and keep their positions — this only cuts per-car cost in jams. Off by default (full detail everywhere)."
            >
              <input
                type="checkbox"
                checked={congestionEnabled}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "congestionEnabled", value: e.target.checked });
                  setCongestionEnabled(e.target.checked);
                }}
              />
              Fast jam model
            </label>
            <label
              className={styles.zoomLabel}
              title="Active-set scheduler: cars that are provably idle this tick — parked in a standing queue, or cruising open straight road far from any junction — skip the full per-car decision and use a cheap synthesized one, so per-frame cost tracks the cars actually making decisions rather than the whole fleet. Behaviour-preserving in normal traffic; a big win in gridlock. On by default (single-threaded builds); it automatically stands down when CPU threads are actively parallelizing."
            >
              <input
                type="checkbox"
                checked={sleepScheduler}
                disabled={!ready}
                onChange={(e) => {
                  sessionRef.current?.applyControl({ type: "sleepScheduler", value: e.target.checked });
                  setSleepScheduler(e.target.checked);
                }}
              />
              Idle-car skipping
            </label>
            {congestionEnabled && (
              <label
                className={styles.zoomLabel}
                title="Occupancy (share of jam density) a link must hold before its queued cars switch to the cheap follower model. Lower engages sooner on lighter congestion."
              >
                Engage {Math.round(congestionEngage * 100)}%
                <input
                  type="range"
                  min={0.4}
                  max={0.98}
                  step={0.01}
                  value={congestionEngage}
                  disabled={!ready}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    sessionRef.current?.applyControl({ type: "congestionEngage", value: v });
                    setCongestionEngage(v);
                  }}
                />
              </label>
            )}
        </Collapsible>
      </div>

      <Collapsible className={styles.status} icon="📊" title="Show or hide diagnostics">
        {!ready && !error && <span>loading engine…</span>}
        {ready && <span>• {backend}</span>}
        {ready && <span>• {mapLabel}</span>}
        {ready && <span ref={statsRef} className={styles.statusStats} />}
        {ready && (
          <label
            className={styles.zoomLabel}
            title="Show one row per shard thread — its car count, cars at node boundaries, and (native builds) its pass times — to see how evenly the regions load the cores. Costs one small array copy per frame while on."
          >
            <input
              type="checkbox"
              checked={shardStats}
              onChange={(e) => {
                sessionRef.current?.applyControl({ type: "shardStats", value: e.target.checked });
                setShardStats(e.target.checked);
              }}
            />
            Per-thread stats
          </label>
        )}
        {error && <span style={{ color: "#ff7b72" }}>• engine failed: {error}</span>}
      </Collapsible>

      <div ref={panelRef} className={styles.panel} />

      <Collapsible className={styles.hint} icon="❓" title="Show or hide the controls guide">
        <ul className={styles.hintList}>
          <li>Wheel to zoom</li>
          <li>Right-drag to pan</li>
          <li>Hover a road for its name</li>
          <li>Click to select</li>
        </ul>
      </Collapsible>
      <a
        className={styles.osmCredit}
        href="https://www.openstreetmap.org/copyright"
        target="_blank"
        rel="noreferrer"
      >
        © OpenStreetMap contributors
      </a>

      <div ref={tipRef} className={styles.tooltip} />
    </div>
  );
}

/** The scenario picker shown when the URL carries no `?scenario=`. It warms up
 * cross-origin isolation on mount so the one-time COOP/COEP service-worker reload
 * happens *here*, before any map loads — then a card's navigation boots the scene
 * already-isolated, and the threaded wasm loads with no further reload. */
/** One editor row for a lever, rendered by its schema kind. */
function ParamRow({ field, params, set }: { field: ParamField; params: SimParams; set: (k: keyof SimParams, v: SimParams[keyof SimParams]) => void }) {
  const v = params[field.key];
  const label = (
    <span style={{ flex: 1 }} title={field.help}>
      {field.label}
    </span>
  );
  if (field.kind.t === "bool") {
    return (
      <label className={styles.paramRow} title={field.help}>
        {label}
        <input type="checkbox" checked={v as boolean} onChange={(e) => set(field.key, e.target.checked)} />
      </label>
    );
  }
  if (field.kind.t === "enum") {
    return (
      <label className={styles.paramRow} title={field.help}>
        {label}
        <select value={v as string} onChange={(e) => set(field.key, e.target.value as Compute)}>
          {field.kind.values.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
    );
  }
  const { min, max } = fieldRange(field);
  const step = field.step ?? 1;
  return (
    <label className={styles.paramRow} title={field.help}>
      {label}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={v as number}
        onChange={(e) => set(field.key, Number(e.target.value))}
        style={{ flex: 1 }}
      />
      <span style={{ width: "3.5em", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
        {step < 1 ? (v as number).toFixed(2) : Math.round(v as number)}
      </span>
    </label>
  );
}

const PARAM_GROUPS = ["Execution", "Routing", "Behavior", "Demand generation"] as const;

/** A collapsible group of lever rows; the title keeps the `paramGroupTitle` font. */
function ParamGroupSection({ title, fields, params, set }: { title: string; fields: ParamField[]; params: SimParams; set: (k: keyof SimParams, v: SimParams[keyof SimParams]) => void }) {
  const [open, setOpen] = useState(true);
  return (
    <div className={styles.paramGroup}>
      <button className={styles.paramGroupTitle} onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span>{title}</span>
        <span>{open ? "▾" : "▸"}</span>
      </button>
      {open && fields.map((f) => <ParamRow key={f.key} field={f} params={params} set={set} />)}
    </div>
  );
}

function SplashScreen() {
  const [params, setParams] = useState<SimParams>({ ...PARAM_DEFAULTS });
  useEffect(() => {
    void ensureCrossOriginIsolation();
  }, []);
  const set = (k: keyof SimParams, v: SimParams[keyof SimParams]) => setParams((p) => ({ ...p, [k]: v }));
  const pick = (key: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("scenario", key);
    // Only carry a config blob when it differs from defaults, so a stock launch is
    // just `?scenario=…`. Legacy readable params are cleared to avoid ambiguity.
    for (const k of ["split", "warmup", "shard", "gpu", "compute", "localrouting"]) url.searchParams.delete(k);
    if (paramsAreDefault(params)) url.searchParams.delete("c");
    else url.searchParams.set("c", encodeParams(params));
    window.location.href = url.toString();
  };
  return (
    <div className={styles.splash}>
      <div className={styles.splashInner}>
        <h1 className={styles.splashTitle}>Traffic</h1>
        <p className={styles.splashSubtitle}>Pick a map to simulate</p>
        <Collapsible icon="⚙️" label="Params" title="Simulation configuration" defaultOpen={false} className={styles.splashParams}>
          <div className={styles.paramsHint}>
            <button className={styles.paramReset} onClick={() => setParams({ ...PARAM_DEFAULTS })} disabled={paramsAreDefault(params)}>
              Reset
            </button>
          </div>
          {PARAM_GROUPS.map((g) => (
            <ParamGroupSection key={g} title={g} fields={PARAM_SCHEMA.filter((f) => f.group === g)} params={params} set={set} />
          ))}
        </Collapsible>
        <div className={styles.splashGrid}>
          {SCENARIOS.map((s) => (
            <button key={s.key} className={styles.splashCard} onClick={() => pick(s.key)}>
              <span className={styles.splashCardName}>{s.name}</span>
              <span className={`${styles.splashBadge} ${s.kind === "Test" ? styles.splashBadgeTest : ""}`}>{s.kind}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Register the COOP/COEP shim service worker so the page becomes cross-origin
 * isolated (enabling SharedArrayBuffer, and thus the wasm CPU-thread pool). On the
 * first visit the worker isn't controlling yet, so reload once (guarded against a
 * loop); on a browser without service workers, or if it fails, we silently stay
 * single-threaded. */
async function ensureCrossOriginIsolation(): Promise<void> {
  if (typeof window === "undefined" || window.crossOriginIsolated) return;
  if (!("serviceWorker" in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register(`${basePath()}/coi-serviceworker.js`);
    if (reg.active && !navigator.serviceWorker.controller && !sessionStorage.getItem("coiReloaded")) {
      sessionStorage.setItem("coiReloaded", "1");
      window.location.reload();
    }
  } catch (e) {
    console.warn("COOP/COEP shim registration failed:", e);
  }
}
