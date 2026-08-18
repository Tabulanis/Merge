"""
The power switch.

The local models are the only thing here that holds the GPU — the 30B alone
owns ~23 of the card's 24GB. Everything runs under systemd user units, so
"shut her down" is just stopping those units; this module is the one place
that knows their names. The dashboard is left running on purpose: it uses no
VRAM, and its Power card is how you wake her back up without a terminal.
"""

from __future__ import annotations

import subprocess
import time

# Order matters only for display. big is the workhorse; the others exist
# but are usually stopped anyway.
MODEL_UNITS = ("forge-model-big", "forge-model-merge", "forge-model-vision",
               "forge-model-little", "forge-model-tiny")

PORTS = {"big": 8080, "merge": 8085, "vision": 8090, "little": 8083, "tiny": 8081}

# How much VRAM each model takes once loaded — measured. Only used to draw
# the loading bar; if a model ever changes, the bar just runs fast or slow.
EXPECTED_LOAD_MB = {"big": 19300, "merge": 21000}

# Models that can't share the card at once (measured: big alone holds ~19GB,
# merge needs ~21GB of the 24GB total — together they don't fit). Starting
# one now auto-stops the other first, instead of leaving that as a comment
# a human has to remember — same lesson as everything else found this
# session: a rule that isn't enforced gets crossed eventually.
EXCLUSIVE = {"big": "merge", "merge": "big"}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=60)


def vram() -> str:
    """'22.8 / 24.0 GB used' — or '' on a machine with no NVIDIA GPU."""
    try:
        r = _run("nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits")
        used, total = r.stdout.strip().splitlines()[0].split(",")
        return f"{int(used) / 1024:.1f} / {int(total) / 1024:.1f} GB used"
    except Exception:
        return ""


def vram_mb() -> int | None:
    try:
        r = _run("nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits")
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def is_ready(which: str) -> bool:
    """llama-server answers /health with 200 only once the weights are
    actually loaded — before that the port refuses or says 503. This is
    the difference between 'systemd started it' and 'she can talk'."""
    import urllib.request
    port = PORTS.get(which)
    if not port:
        return False
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def running() -> list[str]:
    out = []
    for unit in MODEL_UNITS:
        r = _run("systemctl", "--user", "is-active", unit + ".service")
        if r.stdout.strip() == "active":
            out.append(unit)
    return out


def short(unit: str) -> str:
    return unit.removeprefix("forge-model-")


def off() -> dict:
    """Stop every running model service and report the VRAM that came back."""
    stopped = running()
    for unit in stopped:
        _run("systemctl", "--user", "stop", unit + ".service")
    # systemctl stop waits for the process, but the driver takes a beat to
    # actually release the memory — without this the report shows the old
    # number and looks like nothing happened.
    if stopped:
        time.sleep(1.5)
    return {"stopped": [short(u) for u in stopped], "vram": vram()}


def on(which: str = "big") -> dict:
    """Start one model service. Loading a model takes a minute — this
    returns as soon as systemd accepts the job, it does not wait.
    Auto-stops whatever this model can't share the card with (see
    EXCLUSIVE) — VRAM math is not something to leave to a reminder."""
    unit = f"forge-model-{which}"
    if unit not in MODEL_UNITS:
        return {"started": [], "vram": vram(),
                "error": f"no model service named {which!r} — "
                         f"try: {', '.join(short(u) for u in MODEL_UNITS)}"}
    stopped = []
    rival = EXCLUSIVE.get(which)
    if rival and rival in running():
        _run("systemctl", "--user", "stop", f"forge-model-{rival}.service")
        stopped = [rival]
        time.sleep(1.5)   # let the driver actually release the VRAM
    r = _run("systemctl", "--user", "start", unit + ".service")
    err = r.stderr.strip()
    return {"started": [] if err else [which], "stopped_for_room": stopped,
            "vram": vram(), "error": err}
