import EngineCanvas from "../components/EngineCanvas";

export default function Home() {
  return (
    <main style={{ padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 18, fontWeight: 600 }}>Traffic — SF Peninsula</h1>
      <p style={{ opacity: 0.7, fontSize: 13 }}>Watch Bay Area traffic</p>
      <EngineCanvas />
    </main>
  );
}
