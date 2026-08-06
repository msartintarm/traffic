// The engine worker: it owns the transferred OffscreenCanvas, wasm, the sim, and the render
// loop, so all of that runs off the main thread. It is deliberately thin — every non-trivial
// decision lives in the tested `engineSession` / `protocol` / `camera` / `hud` modules — and
// simply relays the session's callbacks out as `FromWorker` messages and inbound `Control`s
// into the session, buffering any that arrive before boot completes.

import { startEngineSession, type SessionHandle } from "../lib/engineSession.ts";
import { type Control, type FromWorker, type InitMsg, isControl } from "../lib/protocol.ts";

type WorkerScope = {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage(m: unknown): void;
};
const ctx = self as unknown as WorkerScope;
const post = (m: FromWorker) => ctx.postMessage(m);

let session: SessionHandle | null = null;
const pending: Control[] = [];

ctx.onmessage = async (e: MessageEvent) => {
  const msg = e.data as InitMsg | Control | undefined;
  if (msg && msg.type === "init") {
    try {
      session = await startEngineSession(msg.canvas, true, msg.config, {
        onReady: (r) => post({ type: "ready", ...r }),
        onFrame: (f) => post({ type: "frame", ...f }),
        onHover: (name, x, y) => post({ type: "hover", name, x, y }),
        onProgress: (fraction, stage) => post({ type: "progress", fraction, stage }),
        onFatal: (message) => post({ type: "fatal", message }),
      });
      for (const c of pending.splice(0)) session.applyControl(c);
    } catch (err) {
      post({ type: "fatal", message: String(err) });
    }
    return;
  }
  if (!isControl(msg)) return;
  if (session) session.applyControl(msg);
  else pending.push(msg);
};
