"""
`forge doctor` — checks everything, and tells you the exact command to fix
whatever's broken.

Documentation explains how things should work. This says what's actually
wrong right now, on this machine, and what to type about it. For someone
new to a terminal that's the difference between being stuck and not.

Every check returns a fix string when it fails. A check that can only say
"something's wrong" isn't worth running — it leaves you exactly where you
started, but with more anxiety.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import CONFIG_PATH, load_config
from .media import capabilities, load_media_config

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str          # ok | warn | fail
    detail: str
    fix: str = ""        # exact command or step, empty when nothing to do


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "your-computer's-address"


def _reachable_local(models: dict) -> str | None:
    """
    Name a local model that is actually answering right now.

    Suggesting any configured model would be worse than useless — pointing
    someone at a server that was never running turns one problem into two.
    """
    for name, m in models.items():
        if m.get("provider") == "anthropic":
            continue
        base = (m.get("base_url") or "").rstrip("/")
        if not base:
            continue
        try:
            if httpx.get(f"{base}/models", timeout=1.5).status_code == 200:
                return name
        except Exception:
            continue
    return None


def run_checks() -> list[Check]:
    out: list[Check] = []
    cfg = load_config()

    # -- settings file -------------------------------------------------
    out.append(Check("Settings file", OK, str(CONFIG_PATH)))

    # -- the model it's pointed at -------------------------------------
    active = cfg.get("active_model", "")
    models = cfg.get("models", {})
    if active not in models:
        alt = _reachable_local(models) or next(iter(models), "qwen30b")
        out.append(Check("Active model", FAIL,
                         f"Set to {active!r}, which isn't configured.",
                         f"/model {alt}"))
        return out + _extras(cfg)

    m = models[active]
    kind = m.get("provider", "?")
    out.append(Check("Active model", OK, f"{active} — {m.get('model')} ({kind})"))

    # -- can it actually be reached? -----------------------------------
    if kind == "anthropic":
        has_key = bool(m.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
        if has_key:
            out.append(Check("Claude API key", OK, "Found."))
        else:
            alt = _reachable_local(models)
            out.append(Check("Claude API key", FAIL,
                             "You're pointed at Claude but there's no key set.",
                             (f"Switch to the local model that's already running:  /model {alt}"
                              if alt else
                              "export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY"
                              "     (or start a local one: ~/forge/start-model.sh big)")))
    else:
        base = (m.get("base_url") or "").rstrip("/")
        try:
            r = httpx.get(f"{base}/models", timeout=3.0)
            if r.status_code == 200:
                out.append(Check("Model server", OK, f"Answering at {base}"))
            elif r.status_code == 503:
                # llama-server binds its port immediately and serves 503 until
                # the weights finish loading — a minute or more for a 30B.
                # Telling someone to start it again here is the worst advice
                # available: a second copy fights the first for the card.
                out.append(Check("Model server", WARN,
                                 f"{base} is up but still loading the model. "
                                 f"Big models take a minute or two.",
                                 "Wait, then run: forge doctor     "
                                 "(don't start it again — two copies fight "
                                 "over the graphics card)"))
            else:
                out.append(Check("Model server", FAIL,
                                 f"{base} replied {r.status_code}",
                                 "~/forge/start-model.sh big"))
        except Exception:
            out.append(Check("Model server", FAIL,
                             f"Nothing is answering at {base}. "
                             f"This is the usual reason Forge seems to hang.",
                             "~/forge/start-model.sh big     (then wait ~40 seconds)"))

    return out + _extras(cfg)


def _extras(cfg: dict) -> list[Check]:
    out: list[Check] = []

    # -- other local models that happen to be up -----------------------
    running = []
    for port, label in ((8080, "mote 3B"), (8081, "tiny"),
                        (8082, "coder14"), (8083, "little 3B"),
                        (8084, "big / coding"), (8090, "vision")):
        if _port_open(port):
            running.append(f"{label} (:{port})")
    out.append(Check("Local models running", OK if running else WARN,
                     ", ".join(running) if running else "None are running.",
                     "" if running else "~/forge/start-model.sh big"))

    # -- eyes and ears -------------------------------------------------
    caps = capabilities(load_media_config(cfg))
    for key, label, fix in (
        ("vision", "Seeing (images)", "~/forge/start-model.sh vision"),
        ("speech_in", "Hearing (voice → text)",
         "build whisper.cpp in ~/whisper.cpp — see: forge help media"),
        ("screenshot", "Screenshots", ""),
        ("speech_out", "Speaking aloud", "sudo apt install speech-dispatcher"),
    ):
        c = caps.get(key, {})
        out.append(Check(label, OK if c.get("ok") else WARN,
                         c.get("detail", ""), "" if c.get("ok") else fix))

    # -- the web interface --------------------------------------------
    srv = cfg.get("server", {})
    port = int(srv.get("port", 8770))
    host = srv.get("host", "127.0.0.1")
    token = srv.get("token", "")
    if _port_open(port):
        where = (f"http://{_lan_ip()}:{port}/chat" if host == "0.0.0.0"
                 else f"http://127.0.0.1:{port}/chat")
        link = where + (f"?token={token}" if token else "")
        out.append(Check("Web interface", OK, f"Running — {link}"))
    else:
        out.append(Check("Web interface", WARN, "Not running.",
                         "forge-dash --network     (to reach it from a phone)"))

    # -- exposed with no lock on the door ------------------------------
    if host == "0.0.0.0" and not token:
        out.append(Check("Web security", FAIL,
                         "Reachable from your whole network with NO token. "
                         "Anyone on your wifi could run commands on this computer.",
                         "forge-dash --new-token"))
    elif host == "0.0.0.0":
        out.append(Check("Web security", OK, "Network access is token-protected."))

    # -- room to work --------------------------------------------------
    try:
        free = shutil.disk_usage(Path.home()).free // (1024 ** 3)
        out.append(Check("Disk space", OK if free > 5 else WARN, f"{free} GB free",
                         "" if free > 5 else "Models are large; free some space."))
    except Exception:
        pass

    # -- graphics card -------------------------------------------------
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            line = (r.stdout or "").strip().splitlines()[0]
            out.append(Check("Graphics card", OK, line))
        except Exception:
            pass

    # -- exact math (compute tool) --------------------------------------
    try:
        import sympy  # noqa: F401
        out.append(Check("Exact math (compute)", OK, "sympy is available"))
    except ImportError:
        out.append(Check("Exact math (compute)", WARN,
                         "sympy missing — the compute tool will fail",
                         "Install it:  ~/forge/.venv/bin/pip install sympy"))

    # -- saved conversations --------------------------------------------
    sess_dir = Path.home() / ".forge" / "sessions"
    n = len(list(sess_dir.glob("*.json"))) if sess_dir.is_dir() else 0
    out.append(Check("Saved conversations", OK,
                     f"{n} stored — they survive restarts and reboots"))

    # -- self-starting services -----------------------------------------
    if shutil.which("systemctl"):
        try:
            r = subprocess.run(
                ["systemctl", "--user", "is-enabled",
                 "forge-model-big", "forge-model-vision", "forge-dash"],
                capture_output=True, text=True, timeout=5)
            states = (r.stdout or "").split()
            if states.count("enabled") == 3:
                out.append(Check("Start at boot", OK,
                                 "Model, vision and dashboard all start themselves"))
            else:
                out.append(Check("Start at boot", WARN,
                                 f"Some services not enabled: {' '.join(states)}",
                                 "systemctl --user enable forge-model-big "
                                 "forge-model-vision forge-dash"))
        except Exception:
            pass

    # -- kid mode --------------------------------------------------------
    if cfg.get("kid_mode"):
        out.append(Check("Kid mode", OK,
                         "ON — chat only, commands fenced to the workspace, "
                         "settings locked"))

    # -- hazards that each broke a real session (2026-08-22) -----------
    # 1) A directory literally named "~": a path written without expanduser
    #    lands there; writes "succeed", reads find nothing, agents spin.
    import glob as _glob
    tilde_dirs = [d for d in _glob.glob(os.path.expanduser("~/~")) +
                  _glob.glob(os.path.expanduser("~/*/~")) +
                  _glob.glob(os.path.expanduser("~/*/*/~")) if os.path.isdir(d)]
    if tilde_dirs:
        out.append(Check("Literal '~' directories", FAIL,
                         "Misfiled writes: " + ", ".join(tilde_dirs),
                         "merge contents into the real paths, then delete"))
    else:
        out.append(Check("Literal '~' directories", OK, "none found"))

    # 2) llama-server running without --jinja: tool calling silently
    #    disabled — the tools array is ignored with no error at all.
    nojinja = []
    for pid in _glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = open(pid, "rb").read().decode(errors="ignore").split("\0")
        except OSError:
            continue
        if (argv and os.path.basename(argv[0]) == "llama-server"
                and "--jinja" not in argv and "--embedding" not in argv):
            port = ""
            if "--port" in argv:
                port = ":" + argv[argv.index("--port") + 1]
            nojinja.append(port or "?")
    if nojinja:
        out.append(Check("llama-server --jinja", WARN,
                         "tool calling DISABLED on " + ", ".join(nojinja),
                         "restart those servers with --jinja"))
    else:
        out.append(Check("llama-server --jinja", OK, "all servers have it"))

    # 3) config base_url vs what actually answers there: a stale port
    #    entry silently serves the WRONG model (bit us when mote took 8080).
    import json as _json, urllib.request as _rq
    for name, m in (cfg.get("models") or {}).items():
        url = (m.get("base_url") or "").rstrip("/")
        want = m.get("model", "")
        if not url or not want:
            continue
        try:
            with _rq.urlopen(url + "/models", timeout=1) as r:
                got = _json.load(r)["models"][0]["name"]
        except Exception:
            continue        # not running — the reachability check covers that
        if not (got.startswith(want) or want.startswith(got)):
            out.append(Check(f"Config truth: {name}", WARN,
                             f"config says {want!r} but {url} serves {got!r}",
                             "fix base_url or restart the right server"))

    return out


def report(checks: list[Check]) -> tuple[int, int]:
    """(failures, warnings) — for deciding what to say at the end."""
    return (sum(1 for c in checks if c.status == FAIL),
            sum(1 for c in checks if c.status == WARN))
