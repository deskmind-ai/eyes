"""Local grounding service for the desktop agent: our 4B grounder behind one HTTP endpoint on 127.0.0.1.

DeskMind Hands (a different venv) asks it where a described element is. Screenshots never leave the machine: the request
names a PNG on local disk, the answer is points.

    python -m deskmind_eyes.ground_server --model ~/models/eyes-4b-mlx --port 8010

    POST /ground {"image": "/abs/path.png", "size": [w, h] (optional), "queries": {"发送": "send message button ...", ...}}
      -> {"points": {"发送": [0.97, 0.95], ...}, "seconds": 12.3}      # relative (0-1) to the image, null if unparsed
    GET /health -> {"ok": true}

With DESKMIND_EYES_TOKEN set, every request must carry `Authorization: Bearer <token>` (else 401): any process on
the Mac can reach 127.0.0.1, and the DeskMind app reuses a server already on its port only when it answers to the
app's token. Unset, no header is needed.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from deskmind_eyes.common import PROMPT, parse_point, resize_to_max_pixels


def authorized(header: str | None, token: str) -> bool:
    """No token configured: anything goes. Otherwise the request's Bearer token must match it."""
    if not token:
        return True
    return hmac.compare_digest((header or "").encode(), f"Bearer {token}".encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/eyes-4b-mlx"))
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    ap.add_argument("--cache-limit-gb", type=float, default=2.0)
    a = ap.parse_args()
    import mlx.core as mx                       # here, not at import: the token check is testable without MLX
    from mlx_vlm import apply_chat_template, generate, load
    mx.set_cache_limit(int(a.cache_limit_gb * 1024**3))
    model, proc = load(a.model)
    lock = threading.Lock()                     # one generation at a time: MLX state is not shared safely

    def ground(image: Image.Image, desc: str):
        prompt = apply_chat_template(proc, model.config, PROMPT.format(instruction=desc), num_images=1,
                                     enable_thinking=False)
        res = generate(model, proc, prompt, image=[image], max_tokens=64, temperature=0.0)
        return parse_point(res.text)

    token = os.environ.get("DESKMIND_EYES_TOKEN", "")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code: int, out: dict) -> None:
            data = json.dumps(out, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            return authorized(self.headers.get("Authorization"), token)

        def do_GET(self):
            if not self._authorized():
                return self._reply(401, {"error": "unauthorized"})
            self._reply(200, {"ok": True} if self.path.rstrip("/") == "/health" else {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return self._reply(401, {"error": "unauthorized"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                img = Image.open(body["image"]).convert("RGB")
                if body.get("size"):            # e.g. a Retina capture brought back to the window's point size
                    img = img.resize(tuple(int(v) for v in body["size"]), Image.Resampling.LANCZOS)
                img = resize_to_max_pixels(img, a.max_pixels)
                t = time.time()
                with lock:
                    pts = {k: ground(img, d) for k, d in body["queries"].items()}
                out, code = {"points": pts, "seconds": round(time.time() - t, 2)}, 200
            except Exception as exc:            # noqa: BLE001 -- reported to the caller, the server stays up
                out, code = {"error": f"{type(exc).__name__}: {exc}"}, 400
            self._reply(code, out)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"grounder {a.model} on http://127.0.0.1:{a.port}/ground", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
