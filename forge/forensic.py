"""Forensic logging — the flight recorder, for when nobody was watching.

The session log is written for replaying a conversation: summaries, truncated
results, no timings. Good for reading, useless for a post-mortem. When
something goes wrong at 2am the questions are always the same — what exactly
did she pass, what exactly came back, how long did it take, what did the
reviewer say, and where did the turn actually go off? — and the normal log
answers none of them.

This records the whole turn: full arguments, full results, wall-clock timings,
superego verdicts, hiccups, and the loop/budget events. One JSON object per
line in ~/.forge/forensic/<session>.jsonl, so `babysit` can reconstruct a
failure afterwards without anyone having been present.

It is OFF unless asked for, because it writes everything: turn on Bug Hunt
mode (or FORGE_FORENSIC=1) when you're chasing something. Two guards keep it
from becoming its own liability:

  * secrets are never recorded — anything from the vault is a handle already,
    and obvious credential-shaped fields are redacted on the way in;
  * each file is capped, and rotates rather than growing without bound.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

DIR = Path.home() / ".forge" / "forensic"
MAX_BYTES = 8_000_000          # per session file, then it rotates
MAX_FIELD = 4000               # per recorded value

_SECRETISH = re.compile(
    r"(pass(word|wd)?|secret|token|api[_-]?key|auth|credential|cookie|"
    r"bearer|private[_-]?key)", re.I)


# FORGE_FORENSIC=1 in the environment arms the recorder for a WHOLE run and is
# the user's explicit choice; a per-turn mode switch must never clear it (an
# early version did, silently discarding the recording someone asked for).
_MODE_ON = False


def enabled() -> bool:
    """On when the env var was set for the run, or Bug Hunt mode set it."""
    return bool(os.environ.get("FORGE_FORENSIC")) or _MODE_ON


def set_enabled(on: bool) -> None:
    """Per-turn switch, driven by the mode. Leaves the env override alone."""
    global _MODE_ON
    _MODE_ON = bool(on)


def _redact(obj, key: str = ""):
    """Never write a secret into the flight recorder. Values under a
    credential-shaped key are replaced, not shortened."""
    if _SECRETISH.search(key or ""):
        return "[redacted]"
    if isinstance(obj, dict):
        return {k: _redact(v, str(k)) for k, v in list(obj.items())[:40]}
    if isinstance(obj, (list, tuple)):
        return [_redact(v, key) for v in list(obj)[:40]]
    s = str(obj)
    if len(s) > MAX_FIELD:
        return s[:MAX_FIELD] + f" …[+{len(s)-MAX_FIELD} chars]"
    return s


def _path(session: str) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(session or "nosession"))[:40]
    p = DIR / f"{safe}.jsonl"
    try:
        if p.exists() and p.stat().st_size > MAX_BYTES:
            p.rename(p.with_suffix(f".jsonl.{int(time.time())}"))
    except Exception:
        pass
    return p


def record(session: str, kind: str, **fields) -> None:
    """Append one event. Never raises — a broken recorder must not break a turn."""
    if not enabled():
        return
    try:
        rec = {"t": round(time.time(), 3), "kind": kind}
        rec.update({k: _redact(v, k) for k, v in fields.items()})
        with _path(session).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def read(session: str, limit: int = 0) -> list[dict]:
    """Read a session's flight recording back (used by babysit)."""
    p = DIR / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', str(session))[:40]}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out[-limit:] if limit else out


def sessions() -> list[tuple[str, float, int]]:
    """(session id, last write, event count) for every recording on disk."""
    if not DIR.is_dir():
        return []
    out = []
    for p in DIR.glob("*.jsonl"):
        try:
            out.append((p.stem, p.stat().st_mtime,
                        sum(1 for _ in p.open(encoding="utf-8"))))
        except Exception:
            continue
    return sorted(out, key=lambda x: -x[1])
