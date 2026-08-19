"""Runs one of Merge's sims in an isolated subprocess (see sims.py).

Usage:
  python _sim_runner.py <sim.py> run   '<params-json>'   -> prints result JSON
  python _sim_runner.py <sim.py> check                    -> prints structure +
                                                            selftest verdict JSON
"""
import importlib.util
import json
import re
import sys
from pathlib import Path


def _dataset(name):
    """Injected into every sim so its run() can pull cited real-world data —
    e.g. mats = dataset("rocket_materials"). Returns {} if the dataset is missing."""
    n = re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")
    p = Path.home() / "forge" / "datasets" / f"{n}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("data", {})
    except Exception:
        return {}


def _load(path):
    spec = importlib.util.spec_from_file_location("sim", path)
    m = importlib.util.module_from_spec(spec)
    m.dataset = _dataset          # grounded real values available inside the sim
    spec.loader.exec_module(m)
    return m


def main():
    path, mode = sys.argv[1], sys.argv[2]
    m = _load(path)

    if mode == "run":
        print(json.dumps(m.run(json.loads(sys.argv[3])), default=str))
        return

    # mode == "check": confirm shape, pull metadata, and run the selftest.
    meta = getattr(m, "META", None)
    out = {
        "has_run": callable(getattr(m, "run", None)),
        "has_meta": isinstance(meta, dict),
        "meta": {"description": (meta or {}).get("description", ""),
                 "why": (meta or {}).get("why", "")} if isinstance(meta, dict) else {},
    }
    st = getattr(m, "SELFTEST", None)
    if isinstance(st, dict) and st.get("expect"):
        got = m.run(st.get("params", {}))
        tol = min(float(st.get("tol", 0.005)), 0.1)   # tight default; hard cap so a
        #                                              huge tol can't rubber-stamp a wrong sim
        ok, detail = True, {}
        for k, v in st["expect"].items():
            g = (got or {}).get(k)
            detail[k] = {"expect": v, "got": g}
            try:
                import math
                fg, fv = float(g), float(v)
                if not (math.isfinite(fg) and math.isfinite(fv)):
                    ok = False          # NaN/inf never counts as a match
                elif abs(fg - fv) > abs(fv) * tol + 1e-9:
                    ok = False
            except Exception:
                ok = ok and (g == v)
        out["selftest"] = {"pass": ok, "detail": detail}
    else:
        out["selftest"] = None
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
