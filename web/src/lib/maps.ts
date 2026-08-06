// Selectable real (OSM-scraped) maps: scenario key → the public map file and its display
// name. Add a city by scraping it to `web/public/<file>` and listing it here. Shared by the
// component (splash + settings menu) and the worker (which fetches the file to parse).
export const REAL_MAPS: Record<string, { file: string; name: string }> = {
  millbrae: { file: "map.json", name: "Millbrae, CA" },
  sancarlos: { file: "sancarlos.json", name: "San Carlos, CA" },
  sf: { file: "sf.json", name: "San Francisco, CA" },
  peninsula: { file: "peninsula.json", name: "Bay Area Peninsula" },
  columbus: { file: "columbus.json", name: "Columbus, OH" },
};

// The display name for a scenario key (real map or test scene), falling back to the key.
export function scenarioName(key: string): string {
  return SCENARIOS.find((s) => s.key === key)?.name ?? REAL_MAPS[key]?.name ?? key;
}

// Splash-screen scenario menu: the real maps followed by the synthetic test scenes, in the
// order they're offered. Picking one navigates to `?scenario=<key>`, which boots that scene.
export const SCENARIOS: { key: string; name: string; kind: "Real map" | "Test" }[] = [
  { key: "millbrae", name: "Millbrae, CA", kind: "Real map" },
  { key: "sancarlos", name: "San Carlos, CA", kind: "Real map" },
  { key: "sf", name: "San Francisco, CA", kind: "Real map" },
  { key: "peninsula", name: "Bay Area Peninsula", kind: "Real map" },
  { key: "columbus", name: "Columbus, OH", kind: "Real map" },
  { key: "arterial", name: "Arterial junction", kind: "Test" },
  { key: "corridor", name: "Signal corridor", kind: "Test" },
  { key: "gridlock", name: "Gridlock", kind: "Test" },
];
