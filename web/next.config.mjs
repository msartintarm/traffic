/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? "",
  trailingSlash: true,
  images: { unoptimized: true },
  // Dev-server only (static export ignores this — production keeps the
  // coi-serviceworker shim): serve COOP/COEP so the page is cross-origin
  // isolated on first load, enabling SharedArrayBuffer threads without the
  // shim's one-time reload — for the real browser and headless probes alike.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
