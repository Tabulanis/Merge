"""Keep-warm image-gen server. Loads SD-Turbo once and holds it in RAM so
each request skips the ~30s model reload — only the ~15-25s draw remains.
CPU only (GPU stays Merge's). Start it with: ~/forge/start-model.sh imagegen

  GET  /health           -> {"status":"ok"} once the model is warm
  POST /generate {...}   -> {"path": "..."} | {"error": "..."}
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # hide the GPU before torch loads

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from forge.imagegen import _pipe, generate

PORT = int(os.environ.get("IMGGEN_PORT", "8771"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet — no per-request stderr spam
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            path = generate(req["prompt"], req["out_path"],
                            steps=req.get("steps", 3), seed=req.get("seed"),
                            size=req.get("size", 512))
            self._json({"path": path})
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, code=500)


if __name__ == "__main__":
    print("[imagegen] warming up SD-Turbo (loading model into RAM)...", flush=True)
    _pipe()   # load once, now
    print(f"[imagegen] ready, serving on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
