import EngineCanvas from "../components/EngineCanvas";

export default function Home() {
  return (
    <main style={{ padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 18, fontWeight: 600 }}>Traffic — Millbrae, CA</h1>
      <p style={{ opacity: 0.7, fontSize: 13 }}>
        Tick-based micro simulation running in your browser via Rust/WASM.
      </p>
      <EngineCanvas />
    </main>
  );
}
