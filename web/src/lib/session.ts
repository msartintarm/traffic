// The session façade the component talks to: it runs the engine in a Web Worker driving a
// transferred OffscreenCanvas when the browser supports it (the whole sim + render loop off
// the main thread), and falls back to running the identical `engineSession` inline on an old
// browser. Either way the component sees the same `applyControl` + callback surface, so none
// of its input/HUD code knows which path is live.

import { startEngineSession, type SessionCallbacks, type SessionHandle } from "./engineSession.ts";
import { type Control, type FromWorker, type InitConfig, type InitMsg } from "./protocol.ts";

export type Session = { applyControl(c: Control): void; dispose(): void };

// A worker driving an OffscreenCanvas needs all three: the Worker constructor, the transfer
// method on the canvas, and the OffscreenCanvas type.
function canUseWorker(canvas: HTMLCanvasElement): boolean {
  return (
    typeof Worker !== "undefined" &&
    typeof OffscreenCanvas !== "undefined" &&
    typeof (canvas as HTMLCanvasElement & { transferControlToOffscreen?: unknown }).transferControlToOffscreen ===
      "function"
  );
}

export function createSession(canvas: HTMLCanvasElement, config: InitConfig, cb: SessionCallbacks): Session {
  if (canUseWorker(canvas)) {
    try {
      return workerSession(canvas, config, cb);
    } catch (e) {
      // A transfer that throws (e.g. the canvas was already handed off) leaves it unusable
      // for the inline path too, so surface it rather than silently falling through.
      cb.onFatal(String(e));
      return { applyControl: () => {}, dispose: () => {} };
    }
  }
  return inlineSession(canvas, config, cb);
}

function workerSession(canvas: HTMLCanvasElement, config: InitConfig, cb: SessionCallbacks): Session {
  const worker = new Worker(new URL("../worker/engineWorker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data;
    switch (m.type) {
      case "ready": cb.onReady(m); break;
      case "frame": cb.onFrame(m); break;
      case "hover": cb.onHover(m.name, m.x, m.y); break;
      case "progress": cb.onProgress(m.fraction, m.stage); break;
      case "fatal": cb.onFatal(m.message); break;
    }
  };
  worker.onerror = (e) => cb.onFatal(e.message || "engine worker error");
  const offscreen = canvas.transferControlToOffscreen();
  worker.postMessage({ type: "init", canvas: offscreen, config } satisfies InitMsg, [offscreen]);
  return {
    applyControl: (c) => worker.postMessage(c),
    dispose: () => worker.terminate(),
  };
}

// Fallback: the same engine session on the main thread. `startEngineSession` is async, so
// buffer controls until it resolves; the returned façade stays usable from the first call.
function inlineSession(canvas: HTMLCanvasElement, config: InitConfig, cb: SessionCallbacks): Session {
  let handle: SessionHandle | null = null;
  let disposed = false;
  const pending: Control[] = [];
  startEngineSession(canvas, false, config, cb)
    .then((h) => {
      if (disposed) {
        h.dispose();
        return;
      }
      handle = h;
      for (const c of pending.splice(0)) h.applyControl(c);
    })
    .catch((err) => cb.onFatal(String(err)));
  return {
    applyControl: (c) => (handle ? handle.applyControl(c) : pending.push(c)),
    dispose: () => {
      disposed = true;
      handle?.dispose();
    },
  };
}
