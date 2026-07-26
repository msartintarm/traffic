# traffic

A large-scale traffic simulator that runs in the browser on the GPU, plays on
discrete ticks, and imports real road maps from OpenStreetMap. The first map is
**Millbrae, CA**.

The architecture follows the sibling `../plant` project: one Rust crate compiled
to WebAssembly, with the *simulation* kept as a pure, dependency-free core that
runs and is exhaustively tested under plain `cargo test` — no GPU, no DOM, no
browser — and the GPU/browser concerns layered on top behind a `wasm32` gate.

## Design in one screen

Two simulation layers, chosen to satisfy *1M+ vehicles*, *behavioural realism*,
and *browser delivery* at once:

| Layer | Model | Where it runs | Status |
|---|---|---|---|
| Camera-local **micro** | Continuous IDM car-following, heterogeneous drivers, signalized multi-lane intersections | CPU now (`sim/`), also the GPU reference | **built** |
| Global **mass** | Cell Transmission Model flow over the whole network | GPU compute (WGSL) | **CPU reference + WGSL kernel run on a real GPU, tested equal** |

Replay determinism is deliberately *not* a requirement (behavioural realism wins
where they conflict), but reproducibility is kept as a cheap debugging affordance
via a stateless counter-based RNG plus a fixed timestep. Correctness is asserted
**statistically** — the flagship test reproduces the empirical flow/density
*fundamental diagram* rather than any golden byte stream.

## Layout

- `engine/` — Rust → WebAssembly
  - `src/sim/` — the pure core (native `cargo test`, no GPU/DOM):
    - `idm.rs` — Intelligent Driver Model car-following, as pure functions
    - `world.rs` — closed-ring micro world for fundamental-diagram validation
    - `network.rs` — index-addressed road graph (nodes/links/lanes/movements),
      SoA-shaped for a future GPU storage buffer
    - `signal.rs` — fixed-time signal programs; O(1)-per-vehicle, GPU-friendly
    - `constraint.rs` — **composable longitudinal constraints**: a vehicle's
      acceleration is the binding (minimum) of an extensible list of pure
      constraint functions — desired speed, car-following, signal stop line,
      anticipatory downstream-speed braking, stop sign, priority give-way, and
      cooperative (zipper) merge. Adding a behaviour is appending one function;
      accuracy improves monotonically. Pure/stateless, so it inlines into a WGSL
      kernel later.
    - `net_world.rs` — vehicles on the graph: constraint-driven acceleration +
      lane hand-off + **route-following**, with **driver-behaviour regression tests**
    - `meso.rs` / `meso.wgsl` / `meso_gpu.rs` — the **mass layer**: Cell
      Transmission Model as a CPU reference (conservation / capacity / backward-
      wave tests), its WGSL compute mirror (`naga`-validated), and an optional
      `wgpu` runner (`--features gpu`) that executes that kernel on a real device
      and is tested equal to the CPU reference
    - `demand.rs` — origin–destination travel demand injecting routed vehicles
    - `map.rs` — OSM import schema (`OsmMap`), the builder into `Network`, and a
      serde JSON loader (`from_json`, `--features import`) for scraped extracts;
      `millbrae_sample()` is a hand-built stand-in matching the scraper output
    - `config.rs` — every tunable as plain data; `rng.rs`, `clock.rs`, `vehicle.rs`
    - `network.rs` also does link-graph **shortest-path routing** and emits road/
      lane render geometry (`road_strips`, `lane_dividers`, `bounds`)
  - `src/render/` — rendering, plant-style split: pure, native-tested math
    (`camera` pan/zoom/view-proj, `geometry` road/junction/marking meshes +
    turn-arc splines, `scene` instances/LOD/brake-lights/signal-heads, `mass`
    congestion colouring) and a `naga`-validated `scene.wgsl` (static + instanced
    with GPU-side interpolation). The wgpu device loop is the remaining wasm piece.
  - `src/bridge.rs` — `wasm-bindgen` entry (`Simulation`), wasm32 only
- `web/` — Next.js app (static export) that loads the wasm and renders in-browser
- `tools/osm-scraper/` — Overpass scraper emitting the `OsmMap` JSON for Millbrae

## Run the tests (the fast inner loop)

```
cd engine
cargo test                                  # native, ~1s, 58 tests, zero deps
cargo test --features import                # + serde OSM-JSON loader
cargo test --features gpu                   # + runs meso.wgsl on a real GPU,
                                            #   asserted equal to the CPU reference
cargo build --target wasm32-unknown-unknown # confirms the browser build
```

The `gpu` feature uses `wgpu` over Vulkan/GL and accepts **software adapters**
(Mesa `lavapipe`), so the GPU compute path runs headless in CI without a physical
GPU. It is optional at runtime too: `MesoGpu::new` returns `None` when no adapter
is present, and callers fall back to the CPU `MesoCorridor`.

Highlights: `fundamental_diagram_has_free_flow_capacity_and_jam_branches`,
`steady_state_matches_theoretical_equilibrium_speed`, and the behaviour
regressions in `net_world.rs` (`vehicle_stops_before_a_red_light`,
`a_standing_queue_discharges_on_green`, `lanes_are_independent`, …).

## Run it in the browser

```
cd web
npm install
npm run dev        # predev builds the wasm via wasm-pack
```

Requires `wasm-pack` (`cargo install wasm-pack`). The page loads
`Simulation`, ticks it each animation frame, and draws vehicle instances.

## Import Millbrae

```
cd tools/osm-scraper
python3 scrape_millbrae.py --bbox-file bbox.local --out millbrae.json
```

The bounding box is an input, never checked in (flag, gitignored file, or
`TRAFFIC_BBOX`). See that tool's README for the JSON schema — it lines up 1:1 with `NodeSpec` /
`LinkSpec`. A small serde loader (behind a cargo feature) is the remaining step
to swap `millbrae_sample()` for the scraped extract.

## Build order / roadmap

1. ✅ Fixed-tick CPU micro core; fundamental-diagram + behaviour regressions.
2. ✅ Graph network + scalable signals + multi-lane, wired to the OSM schema.
3. ✅ Browser bridge + web scaffold (`npm run dev` on port 3000) rendering roads,
   lane dividers, and oriented vehicles at real widths on a 2D canvas, with
   sub-tick render interpolation (fixed timestep + `alpha` blend) for smooth
   motion between ticks.
4. ✅ Shortest-path routing + origin–destination demand.
5. ✅ Serde loader for scraped `millbrae.json` (`--features import`).
6. ✅ Mass layer: CTM CPU reference + `naga`-validated WGSL compute kernel.
7. ✅ Congestion-reactive routing (`route_links_with_costs` over live travel times).
8. ✅ WGSL kernel dispatched on a real `wgpu::Device` (ping-pong buffers), tested
   equal to the CPU reference on a software Vulkan adapter (`--features gpu`).
9. ⏳ Scale the GPU runner to the whole network (per-link cell arrays) and profile
   toward 1M+; expose it through the browser bridge via WebGPU.
10. ⏳ WebGPU **instanced renderer** for the micro layer (port `../plant`'s pipeline).
11. ⏳ Couple the layers: promote mass-layer cells to micro vehicles near the camera.
