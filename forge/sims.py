"""She builds her own sims.

Deterministic physics/math she'd otherwise have to REASON (and hit her ceiling
on) becomes code she writes once, validates against a known answer, and reuses
forever. Two things make this cheap in exactly the ways that matter:

  - Sims live on DISK (abundant) — the shelf can grow to thousands, kept forever,
    including the odd experimental ones that show how she thinks.
  - Only the RESULT of a run enters her CONTEXT (scarce) — so a huge library
    never bloats her working memory. She pays context for the answer, not the shelf.

A sim only earns the ✓ "validated" mark by reproducing a case whose answer is
already known (its SELFTEST). Un-validated ones are kept but flagged EXPERIMENTAL
— the idea might be the interesting part, but the numbers aren't trusted yet.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

SIMS_DIR = Path.home() / "forge" / "sims"
_CATALOG = SIMS_DIR / "_catalog.json"
_RUNNER = Path(__file__).parent / "_sim_runner.py"

# The scaffold she fills in — the "shortcut": boilerplate done, just the physics
# and a known case to prove it.
TEMPLATE = '''META = {
    "name": "your_sim_name",
    "description": "one line — what it models",
    "why": "the connection/reason you built it",
}

def run(params):
    """params: a dict of inputs. Return a dict of results (numbers + short labels)."""
    x = float(params.get("x", 0))
    return {"answer": x * 2}

# A known case so the sim can prove itself. It's marked VALIDATED only if run()
# reproduces `expect` within `tol`. Leave SELFTEST out and it stays EXPERIMENTAL.
# Make the ✓ mean something: `expect` should be an answer you KNOW is right from an
# INDEPENDENT source (a textbook, a worked example) — NOT one you did in your own
# head, or the sim just inherits your slip and the test proves nothing. Keep `tol`
# tight (0.005 = 0.5% or less for an exact formula); a loose tolerance rubber-stamps
# a subtly-wrong sim.
SELFTEST = {
    "params": {"x": 21},
    "expect": {"answer": 42},
    "tol": 0.005,
}'''


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")[:60]


def _load_catalog() -> dict:
    try:
        return json.loads(_CATALOG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_catalog(cat: dict) -> None:
    SIMS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CATALOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    tmp.replace(_CATALOG)


def _subprocess(sim_path: Path, mode: str, arg: str | None = None, timeout: int = 30):
    cmd = [sys.executable, str(_RUNNER), str(sim_path), mode]
    if arg is not None:
        cmd.append(arg)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return False, (r.stderr or "").strip()[-600:]
        return True, (r.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def save_sim(name: str, code: str) -> str:
    """Write a sim she authored, then validate it against its own SELFTEST."""
    name = _safe_name(name)
    if not name:
        return "Error: give the sim a name (letters/numbers/underscores)."
    try:
        compile(code, f"{name}.py", "exec")
    except SyntaxError as e:
        return f"Won't save — syntax error line {e.lineno}: {e.msg}. Fix and resubmit."
    SIMS_DIR.mkdir(parents=True, exist_ok=True)
    path = SIMS_DIR / f"{name}.py"
    path.write_text(code, encoding="utf-8")

    ok, out = _subprocess(path, "check")
    if not ok:
        return (f"Saved '{name}', but it FAILED to load or run:\n{out}\n"
                f"Fix it — an unrunnable sim is no use.")
    try:
        info = json.loads(out)
    except Exception:
        return f"Saved '{name}', but the check output was unreadable: {out[:200]}"
    if not (info.get("has_run") and info.get("has_meta")):
        return ("Saved, but a sim needs BOTH a META dict and a run(params) function. "
                "One is missing — see the template.")

    meta = info.get("meta") or {}
    st = info.get("selftest")
    validated = bool(st and st.get("pass"))
    cat = _load_catalog()
    cat[name] = {
        "description": meta.get("description", ""),
        "why": meta.get("why", ""),
        "validated": validated,
        "created": (cat.get(name) or {}).get("created") or time.time(),
        "updated": time.time(),
    }
    _save_catalog(cat)

    if st is None:
        return (f"Saved '{name}' as ⚠ EXPERIMENTAL — no SELFTEST, so its numbers "
                f"aren't trusted yet. Add a known case (SELFTEST) to earn the ✓.")
    if validated:
        return (f"Saved '{name}' — ✓ VALIDATED against its known case. (That ✓ is only as "
                f"strong as the case: an independently-known `expect` and a tight `tol` "
                f"≤0.5% are what make it airtight — a number from your own head with a loose "
                f"tolerance can pass a subtly-wrong sim.)")
    return (f"Saved '{name}' as ⚠ EXPERIMENTAL — its SELFTEST did NOT match:\n"
            f"{json.dumps(st.get('detail'))[:400]}\n"
            f"The equations are off; don't trust its numbers until this passes.")


def run_sim(name: str, params: dict | None = None) -> str:
    name = _safe_name(name)
    path = SIMS_DIR / f"{name}.py"
    if not path.exists():
        return f"No sim called '{name}'. Use list_sims to see the shelf, or build_sim to make it."
    ok, out = _subprocess(path, "run", json.dumps(params or {}))
    if not ok:
        return f"Sim '{name}' errored: {out}"
    validated = (_load_catalog().get(name) or {}).get("validated")
    tag = "✓ validated" if validated else "⚠ EXPERIMENTAL — numbers not yet trusted"
    return f"[{name} · {tag}]\n{out}"


def list_sims() -> str:
    cat = _load_catalog()
    if not cat:
        return "The shelf is empty — no sims built yet. Use build_sim to make one."
    lines = [f"{len(cat)} sim(s) on the shelf:"]
    for n, m in sorted(cat.items(), key=lambda kv: -(kv[1].get("updated") or 0)):
        mark = "✓" if m.get("validated") else "⚠"
        line = f"  {mark} {n} — {m.get('description', '')}"
        if m.get("why"):
            line += f"   [why: {m['why']}]"
        lines.append(line)
    return "\n".join(lines)


SHELF_MAX = 14   # most she carries in her always-on prompt; the rest via list_sims


def shelf_line() -> str:
    """Compact one-liner of what's on the sim shelf, for her context so she knows
    her instruments without a tool call. Capped so a big shelf can't bloat her
    prompt — newest first, with a pointer to list_sims for the rest. Empty if bare."""
    cat = _load_catalog()
    if not cat:
        return ""
    items = sorted(cat.items(), key=lambda kv: -(kv[1].get("updated") or 0))
    parts = []
    for n, m in items[:SHELF_MAX]:
        mark = "✓" if m.get("validated") else "⚠"
        desc = m.get("description", "")
        parts.append(f"{n} {mark}" + (f" — {desc}" if desc else ""))
    line = "; ".join(parts)
    extra = len(items) - min(len(items), SHELF_MAX)
    if extra > 0:
        line += f"; …+{extra} more (use list_sims for the full shelf)"
    return line


def verify_shelf() -> str:
    """The immune system: re-run every cataloged sim's SELFTEST right now.
    A sim that no longer reproduces its known answer gets DEMOTED to ⚠ in the
    catalog — a stale ✓ is how confident garbage poisons future work. Also
    flags catalog/disk mismatches (orphans both ways)."""
    cat = _load_catalog()
    lines, changed = [], False
    for name, meta in sorted(cat.items()):
        path = SIMS_DIR / f"{name}.py"
        if not path.exists():
            lines.append(f"  ✗ {name}: FILE MISSING — catalog orphan (rebuild or remove)")
            continue
        ok, out = _subprocess(path, "check")
        if not ok:
            now_valid, status = False, f"BROKEN — won't run ({out[:70]})"
        else:
            try:
                info = json.loads(out)
            except Exception:
                info = {}
            st = info.get("selftest")
            now_valid = bool(st and st.get("pass"))
            status = ("✓ revalidated" if now_valid
                      else ("⚠ SELFTEST NOW FAILING" if st else "⚠ no selftest"))
        was = bool(meta.get("validated"))
        if was != now_valid:
            meta["validated"] = now_valid
            changed = True
            status += "  << DEMOTED from ✓" if was else "  << promoted"
        lines.append(f"  {name}: {status}")
    for f in SIMS_DIR.glob("*.py"):
        if f.stem not in cat:
            lines.append(f"  ? {f.stem}: on disk but not in catalog")
    if changed:
        _save_catalog(cat)
    return "Sim shelf verification:\n" + ("\n".join(lines) or "  (empty shelf)")


def read_sim(name: str) -> str:
    name = _safe_name(name)
    path = SIMS_DIR / f"{name}.py"
    if not path.exists():
        return f"No sim called '{name}'."
    return path.read_text(encoding="utf-8")[:8000]
