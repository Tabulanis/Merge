"""
The dashboard's backend.

Small on purpose: it reads and writes the same config file the CLI reads, so
"switch the model in the browser" needs no messaging between processes. The
CLI notices on its next turn.

It also knows how to ask a local runner (Ollama, llama.cpp, LM Studio) what
models it has, so adding a local model is picking from a list rather than
typing a name exactly right.
"""

from __future__ import annotations

import time
import asyncio
import json
import queue
import secrets
import threading
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (CONFIG_PATH, load_config, load_pipelines, save_config,
                     save_pipelines)
from . import identity, power, recall
from .doctor import run_checks
from .help_content import ORDER, TOPICS, VERSION
from .media import capabilities, list_voices, load_media_config, synth_wav
from .pipeline import Pipeline
from .session import SessionStore

app = FastAPI(title="Forge")
WEB_DIR = Path(__file__).parent / "web"
STORE = SessionStore()


def require_not_kid() -> None:
    """Settings-touching routes refuse in kid mode. The switch lives only
    in ~/.forge/config.yaml on the computer itself — nothing reachable
    from a tablet can flip it."""
    if load_config().get("kid_mode"):
        raise HTTPException(403, "Kid mode is on — settings are locked. "
                                 "Edit ~/.forge/config.yaml at the computer "
                                 "to change this.")


def require_token(request: Request,
                  x_forge_token: str | None = Header(default=None)) -> None:
    """
    Gate every route when a token is configured.

    No token set means machine-only use (host stays 127.0.0.1), so nothing to
    protect against. The moment a token exists — which is what you set before
    binding to the network — every request must carry it. The token may ride
    in a header or a ?token= query param, because a phone opening a link can
    only do the latter.
    """
    want = (load_config().get("server", {}) or {}).get("token", "")
    if not want:
        return
    got = x_forge_token or request.query_params.get("token", "")
    if not secrets.compare_digest(str(got), str(want)):
        raise HTTPException(401, "Bad or missing token")


class ModelSpec(BaseModel):
    name: str
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096


class ActiveSpec(BaseModel):
    name: str


class AgentSpec(BaseModel):
    max_steps: int | None = None
    permission_mode: str | None = None
    # "" clears it; a name assigns that model the memory-summary side-job
    summarizer_model: str | None = None
    superego: bool | None = None


@app.get("/api/config", dependencies=[Depends(require_token)])
def get_config():
    cfg = load_config()
    # Never ship secrets to the browser — just whether one is set.
    safe = {**cfg, "models": {
        name: {**m, "api_key": "***" if m.get("api_key") else ""}
        for name, m in cfg["models"].items()
    }}
    safe["config_path"] = str(CONFIG_PATH)
    return safe


@app.post("/api/active", dependencies=[Depends(require_token), Depends(require_not_kid)])
def set_active(spec: ActiveSpec):
    cfg = load_config()
    if spec.name not in cfg["models"]:
        raise HTTPException(404, f"No model named {spec.name!r}")
    cfg["active_model"] = spec.name
    save_config(cfg)
    return {"ok": True, "active_model": spec.name}


@app.post("/api/models", dependencies=[Depends(require_token), Depends(require_not_kid)])
def upsert_model(spec: ModelSpec):
    cfg = load_config()
    entry = {
        "provider": spec.provider,
        "model": spec.model,
        "max_tokens": spec.max_tokens,
    }
    if spec.base_url:
        entry["base_url"] = spec.base_url
    # An empty/masked key means "leave whatever's there alone" — so saving
    # from the browser can't wipe a key the browser was never shown.
    existing = cfg["models"].get(spec.name, {})
    entry["api_key"] = (existing.get("api_key", "")
                        if spec.api_key in (None, "", "***") else spec.api_key)
    cfg["models"][spec.name] = entry
    save_config(cfg)
    return {"ok": True, "models": list(cfg["models"])}


@app.delete("/api/models/{name}", dependencies=[Depends(require_token), Depends(require_not_kid)])
def delete_model(name: str):
    cfg = load_config()
    if name not in cfg["models"]:
        raise HTTPException(404, f"No model named {name!r}")
    if cfg["active_model"] == name:
        raise HTTPException(400, "That's the active model — switch to another first.")
    del cfg["models"][name]
    save_config(cfg)
    return {"ok": True}


@app.post("/api/agent", dependencies=[Depends(require_token), Depends(require_not_kid)])
def set_agent(spec: AgentSpec):
    cfg = load_config()
    if spec.max_steps is not None:
        cfg["agent"]["max_steps"] = spec.max_steps
    if spec.permission_mode is not None:
        if spec.permission_mode not in ("ask", "auto", "deny"):
            raise HTTPException(400, "permission_mode must be ask, auto, or deny")
        cfg["agent"]["permission_mode"] = spec.permission_mode
    if spec.summarizer_model is not None:
        name = spec.summarizer_model.strip()
        if name and name not in cfg["models"]:
            raise HTTPException(404, f"No model named {name!r}")
        cfg["agent"]["summarizer_model"] = name
    if spec.superego is not None:
        cfg["agent"]["superego"] = bool(spec.superego)
    save_config(cfg)
    return {"ok": True, "agent": cfg["agent"]}


@app.get("/api/discover")
def discover(base_url: str = "http://localhost:11434"):
    """
    Ask a local runner what models it has.

    Tries Ollama's native endpoint first, then the OpenAI-compatible one that
    llama.cpp/LM Studio/vLLM serve, so one button covers every common setup.
    """
    base = base_url.rstrip("/")
    found: list[str] = []
    errors: list[str] = []

    for url, extract in (
        (f"{base}/api/tags", lambda d: [m["name"] for m in d.get("models", [])]),
        (f"{base}/v1/models", lambda d: [m["id"] for m in d.get("data", [])]),
        (f"{base}/models", lambda d: [m["id"] for m in d.get("data", [])]),
    ):
        try:
            r = httpx.get(url, timeout=4.0)
            if r.status_code == 200:
                names = extract(r.json())
                if names:
                    found = names
                    break
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}")

    return {"models": found, "checked": base,
            "hint": "" if found else "Nothing answered there. Is the model server running?"}


@app.get("/api/health")
def health():
    """Is the active model actually reachable right now?"""
    cfg = load_config()
    name = cfg["active_model"]
    m = cfg["models"].get(name, {})
    if m.get("provider") == "anthropic":
        import os
        has_key = bool(m.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
        return {"model": name, "reachable": has_key,
                "detail": "API key present" if has_key else "No API key set"}
    base = (m.get("base_url") or "").rstrip("/")
    if not base:
        return {"model": name, "reachable": False, "detail": "No base_url configured"}
    try:
        r = httpx.get(f"{base}/models", timeout=4.0)
        ok = r.status_code == 200
        return {"model": name, "reachable": ok,
                "detail": "Server responded" if ok else f"HTTP {r.status_code}"}
    except Exception as e:
        return {"model": name, "reachable": False, "detail": f"{type(e).__name__}"}


# ------------------------------------------------------------------ chat
# The part that makes Forge usable from a phone: a real conversation with the
# agent, streamed as it happens.


class ChatSpec(BaseModel):
    message: str
    session: str | None = None
    workspace: str | None = None
    env: str = ""            # where the user is: phone / PC / AR / VR / …
    mode: str = ""           # operating style: flash / muse / balanced / precise / deep
    privacy: str = ""        # normal / offrec (off the record) / sandbox (knowledge only)


class PermissionSpec(BaseModel):
    session: str
    id: str
    allow: bool


@app.post("/api/chat", dependencies=[Depends(require_token)])
def chat(spec: ChatSpec):
    """Start a turn. Returns immediately; watch /api/stream for what happens."""
    STORE.rescan()   # a chat the terminal saved must be continuable here
    _priv = (spec.privacy or "normal").strip().lower()
    sess = STORE.get_or_create(spec.session, spec.workspace, _priv)
    sess.agent.client_env = (spec.env or "").strip()[:60]   # let her know the device
    # Identity checks fire only when switched on AND a phrase is set (armed).
    sess.agent.identity_owner = (
        (load_config().get("identity") or {}).get("owner_name", "")
        if identity.armed() else "")
    if spec.mode:
        sess.agent.mode = spec.mode.strip().lower()         # operating style
    sess.agent.privacy = _priv                               # privacy axis (tools + prompt)
    sess.ephemeral = sess.agent.ephemeral                    # gate all persistence
    sess.emit("user", {"text": spec.message})
    # She runs one request at a time. If another chat is mid-turn, this one
    # will wait its turn on the model — say so, so a queue doesn't read as a hang.
    if not sess.busy and STORE.busy_count(exclude=sess.id):
        sess.emit("note", {"text": (
            "⏳ She's busy with another request right now — yours is queued "
            "behind it and starts the moment she's free.")})
    threading.Thread(target=sess.run_message, args=(spec.message,),
                     daemon=True).start()
    return {"session": sess.id, "workspace": str(sess.workspace),
            "model": sess.model_name}


@app.get("/api/stream/{session_id}", dependencies=[Depends(require_token)])
async def stream(session_id: str, since: int = 0):
    """
    Server-sent events: the agent's work as it happens.

    SSE rather than websockets — it reconnects on its own when a phone drops
    to sleep or switches networks, which is exactly the failure mode here, and
    it needs nothing special from the browser.
    """
    sess = STORE.get(session_id)
    if not sess:
        STORE.rescan()   # it may be a chat the terminal saved
        sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "No such session")

    async def gen():
        # `since` lets a reconnecting phone say what it already has, so it
        # gets exactly what it missed — no gap, no duplicates.
        cursor = since
        while True:
            items, cursor = await asyncio.to_thread(sess.since, cursor, 20.0)
            if not items:
                # a comment line keeps proxies and phone radios from
                # deciding the connection is dead
                yield ": keepalive\n\n"
                continue
            for item in items:
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class StopSpec(BaseModel):
    session: str


@app.post("/api/stop", dependencies=[Depends(require_token)])
def stop(spec: StopSpec):
    """The panic button: end the current turn cleanly, keep the session."""
    sess = STORE.get(spec.session)
    if not sess:
        raise HTTPException(404, "No such session")
    sess.stop()
    return {"ok": True}


@app.post("/api/permission", dependencies=[Depends(require_token)])
def permission(spec: PermissionSpec):
    """The allow/deny tap coming back from the browser."""
    sess = STORE.get(spec.session)
    if not sess:
        raise HTTPException(404, "No such session")
    if not sess.answer_permission(spec.id, spec.allow):
        raise HTTPException(409, "That request is no longer waiting")
    return {"ok": True}


@app.get("/api/sessions", dependencies=[Depends(require_token)])
def sessions():
    STORE.rescan()   # terminal chats belong in the drawer too
    return {"sessions": STORE.listing()}


@app.delete("/api/sessions/{session_id}", dependencies=[Depends(require_token), Depends(require_not_kid)])
def drop_session(session_id: str):
    return {"ok": STORE.drop(session_id)}


@app.get("/api/browse", dependencies=[Depends(require_token)])
def browse(path: str = ""):
    """Folder picker for choosing a workspace from a phone."""
    base = Path(path).expanduser() if path else Path.home()
    if not base.is_dir():
        base = Path.home()
    try:
        dirs = sorted([d.name for d in base.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])[:200]
    except PermissionError:
        dirs = []
    return {"path": str(base), "parent": str(base.parent), "dirs": dirs}


@app.get("/api/media", dependencies=[Depends(require_token)])
def media_status():
    return capabilities(load_media_config(load_config()))


# Her real voice, server-side. The browser's speechSynthesis is at the mercy of
# whatever robotic voices the phone happens to ship; piper gives her one good
# voice that sounds identical on every device. /api/voices lists the menu;
# /api/speak renders a line to WAV the browser just plays.
# Owner identity + challenge phrase. GET returns the name and whether a phrase
# is set — never the phrase. POST sets them; the phrase is hashed in identity.py
# before it touches disk, and is never sent back out.
@app.get("/api/identity", dependencies=[Depends(require_token)])
def get_identity():
    return identity.status()


class IdentitySpec(BaseModel):
    enabled: bool | None = None    # the on/off switch (None = leave as-is)
    owner_name: str | None = None
    phrase: str | None = None      # None = leave as-is, "" = clear, else = set


@app.post("/api/identity", dependencies=[Depends(require_token), Depends(require_not_kid)])
def set_identity(spec: IdentitySpec):
    return identity.set_identity(enabled=spec.enabled, owner_name=spec.owner_name,
                                 phrase=spec.phrase)


@app.get("/api/voices", dependencies=[Depends(require_token)])
def voices():
    return {"voices": list_voices()}


class SpeakSpec(BaseModel):
    text: str
    voice: str = ""


@app.post("/api/speak", dependencies=[Depends(require_token)])
def speak_text(spec: SpeakSpec):
    text = (spec.text or "").strip()
    if not text:
        raise HTTPException(400, "Nothing to say")
    wav = synth_wav(text, spec.voice)
    if not wav:
        raise HTTPException(503, "TTS unavailable (piper or voice missing)")
    return Response(content=wav, media_type="audio/wav",
                    headers={"Cache-Control": "no-store"})


# Serve an image file to the browser so her drawings and the things she looks
# at show up inline in the chat. Token-gated and fenced to the home dir (this
# is a personal LAN tool; everything under home is already the user's), and
# images only — never an arbitrary file read.
_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@app.get("/api/file", dependencies=[Depends(require_token)])
def serve_file(path: str):
    p = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    if p != home and home not in p.parents:
        raise HTTPException(403, "Outside the home directory")
    if not p.is_file():
        raise HTTPException(404, "No such file")
    if p.suffix.lower() not in _IMG_EXT:
        raise HTTPException(415, "Not an image")
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@app.post("/api/upload", dependencies=[Depends(require_token)])
async def upload_image(request: Request, session: str = ""):
    """Save a photo or sound the user attached (raw bytes in the body, like
    /api/listen) into the session's workspace/uploads, and return its path so
    the chat can send it to her (look_at_image for photos, see_sound for audio)."""
    sess = STORE.get(session) if session else None
    if not sess and session:
        STORE.rescan()
        sess = STORE.get(session)
    if not sess:
        sess = STORE.get_or_create(None, "")   # a photo can start a fresh chat
    body = await request.body()
    if not body or len(body) < 100:
        raise HTTPException(400, "No file received")
    if len(body) > 20_000_000:
        raise HTTPException(413, "File too large")
    ctype = (request.headers.get("content-type") or "").lower()
    is_audio = ctype.startswith("audio/")
    AUDIO_EXT = {
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav", "audio/x-wav": ".wav",
        "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
    }
    if is_audio:
        ext = AUDIO_EXT.get(ctype, ".bin")
    else:
        ext = ".png" if "png" in ctype else ".webp" if "webp" in ctype \
            else ".gif" if "gif" in ctype else ".jpg"
    updir = Path(sess.workspace) / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    prefix = "sound-" if is_audio else "photo-"
    path = updir / f"{prefix}{int(time.time())}{ext}"
    path.write_bytes(body)
    return {"path": str(path), "session": sess.id}


# ------------------------------------------------------------------ power
# Stop the local models to hand the GPU back; start them again later. The
# dashboard itself keeps running — it holds no VRAM and is the wake button.


class PowerSpec(BaseModel):
    action: str            # "off" or "on"
    model: str = "big"     # which one to wake, for "on"


@app.get("/api/power", dependencies=[Depends(require_token)])
def power_status():
    running = [power.short(u) for u in power.running()]
    # "running" means systemd started it; "ready" means the weights are
    # loaded and she can answer. The gap between the two is the loading bar.
    return {"running": running,
            "ready": [m for m in running if power.is_ready(m)],
            "vram": power.vram()}


@app.post("/api/power", dependencies=[Depends(require_token), Depends(require_not_kid)])
def power_switch(spec: PowerSpec):
    if spec.action == "off":
        return power.off()
    if spec.action == "on":
        res = power.on(spec.model)
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return res
    raise HTTPException(400, "action must be off or on")


@app.post("/api/listen", dependencies=[Depends(require_token)])
async def listen_endpoint(request: Request):
    """
    Voice input: the browser records audio (webm/ogg), posts the raw bytes
    here, whisper turns them into text, the text goes back to the page.
    Raw body instead of a multipart form on purpose — no extra dependency,
    and the browser side is a two-line fetch.
    """
    import tempfile as _tf

    from .media import listen as media_listen

    body = await request.body()
    if not body or len(body) < 200:
        raise HTTPException(400, "No audio received")
    if len(body) > 25_000_000:
        raise HTTPException(413, "Recording too long")
    suffix = ".webm"
    ctype = (request.headers.get("content-type") or "").lower()
    if "ogg" in ctype:
        suffix = ".ogg"
    elif "wav" in ctype:
        suffix = ".wav"
    elif "mp4" in ctype or "m4a" in ctype or "aac" in ctype:
        suffix = ".m4a"
    tmp = Path(_tf.mkdtemp()) / f"speech{suffix}"
    tmp.write_bytes(body)
    text = await asyncio.to_thread(
        media_listen, str(tmp), load_media_config(load_config()))
    if text.startswith("Error"):
        raise HTTPException(500, text)
    return {"text": text}


# -------------------------------------------------------------- pipelines
# The brick editor reads and writes these. They are the same recipes the CLI
# runs, so anything built in the browser works from the terminal and back.


class PipelineSpec(BaseModel):
    name: str
    description: str = ""
    steps: list


class RunSpec(BaseModel):
    name: str
    task: str
    workspace: str | None = None


@app.get("/api/pipelines", dependencies=[Depends(require_token)])
def get_pipelines():
    return {"pipelines": load_pipelines()}


@app.post("/api/pipelines", dependencies=[Depends(require_token), Depends(require_not_kid)])
def save_pipeline(spec: PipelineSpec):
    if not spec.name.strip():
        raise HTTPException(400, "A recipe needs a name")
    flows = load_pipelines()
    flows[spec.name] = {"description": spec.description, "steps": spec.steps}
    save_pipelines(flows)
    return {"ok": True, "pipelines": list(flows)}


@app.delete("/api/pipelines/{name}", dependencies=[Depends(require_token), Depends(require_not_kid)])
def delete_pipeline(name: str):
    flows = load_pipelines()
    if name not in flows:
        raise HTTPException(404, f"No recipe named {name!r}")
    del flows[name]
    save_pipelines(flows)
    return {"ok": True}


@app.post("/api/pipelines/run", dependencies=[Depends(require_token)])
def run_pipeline(spec: RunSpec):
    """
    Run a recipe, reporting progress through a normal chat session so the
    browser watches it the same way it watches everything else — one stream,
    one set of permission cards, no second mechanism to build or learn.
    """
    flows = load_pipelines()
    if spec.name not in flows:
        raise HTTPException(404, f"No recipe named {spec.name!r}")
    sess = STORE.get_or_create(None, spec.workspace)
    cfg = load_config()

    def go():
        sess.emit("user", {"text": f"▶ {spec.name}: {spec.task}"})
        try:
            pipe = Pipeline({**flows[spec.name], "name": spec.name},
                            cfg["models"], sess.workspace,
                            permission_mode=cfg["agent"].get("permission_mode", "ask"))
            for ev in pipe.run(spec.task, ask=sess.ask_permission):
                if ev.kind == "step_start":
                    sess.emit("tool", {"tool": ev.step, "summary": f"({ev.text})"})
                elif ev.kind == "loop_round":
                    sess.emit("note", {"text": f"{ev.step} — round {ev.round}"})
                elif ev.kind == "step_done":
                    sess.emit("result", {"tool": ev.step,
                                         "text": ("ok: " if ev.ok else "failed: ") + (ev.text or "")[:600]})
                elif ev.kind == "error":
                    sess.emit("error", {"text": ev.text})
                elif ev.kind == "done":
                    sess.emit("text", {"text": ev.text or "(finished)"})
                    sess.emit("done", {"usage": {}})
        except Exception as e:
            sess.emit("error", {"text": f"{type(e).__name__}: {e}"})
            sess.emit("done", {"usage": {}})

    threading.Thread(target=go, daemon=True).start()
    return {"session": sess.id, "workspace": str(sess.workspace)}


# ------------------------------------------------------------------- help
# Same words as the terminal — help_content.py is the only copy.


@app.get("/api/help", dependencies=[Depends(require_token)])
def get_help():
    return {"version": VERSION,
            "order": ORDER,
            "topics": {k: TOPICS[k] for k in ORDER}}


@app.get("/api/doctor", dependencies=[Depends(require_token)])
def get_doctor():
    """Live diagnosis, so the help page can say what's wrong right now."""
    checks = run_checks()
    return {"checks": [{"name": c.name, "status": c.status,
                        "detail": c.detail, "fix": c.fix} for c in checks]}



def _page(name: str) -> FileResponse:
    """Serve an HTML page with caching off. These pages are a few KB and
    change when Forge updates — a browser showing last week's cached copy
    of the chat page is how 'I fixed it' and 'it's still broken' happen
    at the same time."""
    return FileResponse(WEB_DIR / name,
                        headers={"Cache-Control": "no-store"})

@app.get("/help")
def help_page():
    return _page("help.html")


@app.get("/")
def index():
    return _page("index.html")


@app.get("/bricks")
def bricks_page():
    return _page("bricks.html")


@app.get("/chat")
def chat_page():
    return _page("chat.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def serve() -> None:
    """
    Entry point for `forge-dash`.

    Flags exist because the network decision deserves a deliberate act, not a
    config file people edit once and forget. Turning it on tells you what the
    tradeoff is, right there in the terminal.
    """
    import argparse
    import socket
    import uvicorn

    ap = argparse.ArgumentParser(prog="forge-dash",
                                 description="Forge's web interface.")
    ap.add_argument("--network", action="store_true",
                    help="Reachable from phones/tablets on your network (implies a token)")
    ap.add_argument("--local", action="store_true",
                    help="This machine only (the default)")
    ap.add_argument("--new-token", action="store_true",
                    help="Mint a fresh token, invalidating the old one")
    ap.add_argument("--port", type=int, help="Port to listen on")
    args = ap.parse_args()

    # The librarian works out of this process because it's the one that's
    # always running — cards get filed whichever door the chat came through.
    threading.Thread(target=recall.process_queue_forever, daemon=True).start()

    cfg = load_config()
    srv = cfg.setdefault("server", {})
    changed = False

    if args.port:
        srv["port"] = args.port
        changed = True
    if args.local:
        srv["host"] = "127.0.0.1"
        changed = True
    if args.network:
        srv["host"] = "0.0.0.0"
        changed = True
    # Exposing this without a token would put a shell on the network for
    # anyone who can reach the machine, so one is minted rather than offered.
    if args.new_token or (srv.get("host") == "0.0.0.0" and not srv.get("token")):
        srv["token"] = secrets.token_urlsafe(18)
        changed = True
        print(f"\n  new token: {srv['token']}")
    if changed:
        save_config(cfg)
        cfg = load_config()
    srv = cfg.get("server", {})
    host = srv.get("host", "127.0.0.1")
    port = int(srv.get("port", 8770))
    token = srv.get("token", "")

    # HTTPS when a certificate exists (make one with openssl into
    # ~/.forge/tls/). Browsers only allow microphone capture and WebXR (the
    # AR headset path) on secure pages, so this is the key that unlocks
    # voice and AR. Self-signed: each device accepts the warning once.
    tls_dir = Path.home() / ".forge" / "tls"
    certfile, keyfile = tls_dir / "cert.pem", tls_dir / "key.pem"
    use_tls = certfile.exists() and keyfile.exists()
    scheme = "https" if use_tls else "http"

    print()
    if host in ("0.0.0.0", "::"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan = s.getsockname()[0]
            s.close()
        except Exception:
            lan = "your-ip"
        suffix = f"?token={token}" if token else ""
        print(f"  Merge on this machine → {scheme}://127.0.0.1:{port}/chat{suffix}")
        print(f"  Merge from your phone → {scheme}://{lan}:{port}/chat{suffix}")
        if not token:
            print()
            print("  WARNING: bound to the network with NO TOKEN set.")
            print("  Anyone who can reach this machine can run commands on it.")
            print("  Set one:  forge-dash --new-token")
    else:
        print(f"  Merge → {scheme}://{host}:{port}/chat")
        print("  (machine-only. To reach it from a phone: forge-dash --network)")
    # Her human voice (piper) is found next to THIS interpreter. If we were
    # launched under a bare python instead of the venv, piper is missing and she
    # silently drops to the robot TTS — warn loudly rather than fail quietly.
    from .media import _piper_bin
    if not _piper_bin():
        print("  ⚠ Human voice OFF — piper not found next to this interpreter.")
        print("    Launch under the venv:  ~/forge/.venv/bin/python -m forge.server")
        print("    or simply:  bash ~/forge/start-dashboard.sh")
        print()

    if use_tls:
        uvicorn.run(app, host=host, port=port, log_level="warning",
                    ssl_certfile=str(certfile), ssl_keyfile=str(keyfile))
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
