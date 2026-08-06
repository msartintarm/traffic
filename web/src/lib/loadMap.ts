// Fetch a scraped map's JSON, preferring a pre-compressed `.gz` sibling and decompressing it
// in the browser. The app is a static export (no server), so a plain host may ship a 30 MB
// map uncompressed; a committed `<file>.gz` is ~1/7 the size and `DecompressionStream` inflates
// it here — cutting the download regardless of host config. Falls back to the plain JSON when
// there is no `.gz` or the browser lacks `DecompressionStream`. Pure `fetch` + streams, so it
// runs unchanged on the main thread or inside a Web Worker.
//
// `onProgress` receives the download fraction 0→1 (against the response's `content-length`),
// so a big map (SF, Columbus) can drive a real progress bar. It counts the bytes actually
// pulled over the wire — the compressed bytes for the `.gz` path — which is what the user
// waits on. Omitted or an absent length simply means no ticks (the caller shows it indeterminate).
export async function fetchMapText(url: string, onProgress?: (fraction: number) => void): Promise<string> {
  if (typeof DecompressionStream !== "undefined") {
    try {
      const gz = await fetch(`${url}.gz`);
      if (gz.ok && gz.body) {
        const counted = withProgress(gz.body, Number(gz.headers.get("content-length")) || 0, onProgress);
        const inflated = counted.pipeThrough(new DecompressionStream("gzip"));
        return await new Response(inflated).text();
      }
    } catch {
      // fall through to the uncompressed map
    }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`map fetch failed (${res.status}) for ${url}`);
  if (res.body) {
    const counted = withProgress(res.body, Number(res.headers.get("content-length")) || 0, onProgress);
    return await new Response(counted).text();
  }
  return await res.text();
}

// Tee a byte stream through a pass-through that reports cumulative bytes / total as a 0→1
// fraction. No total (unknown length) → no ticks, and the stream is returned untouched.
function withProgress<T extends Uint8Array>(
  body: ReadableStream<T>,
  total: number,
  onProgress?: (fraction: number) => void,
): ReadableStream<T> {
  if (!onProgress || total <= 0) return body;
  let received = 0;
  return body.pipeThrough(
    new TransformStream<T, T>({
      transform(chunk, controller) {
        received += chunk.byteLength;
        onProgress(Math.min(1, received / total));
        controller.enqueue(chunk);
      },
    }),
  );
}
