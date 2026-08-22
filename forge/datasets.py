"""Her dataset shelf — cited real-world reference data on disk.

The counterpart to the sim shelf, and its fuel. Sims are validated deterministic
COMPUTATION; datasets are cited stable real-world FACTS she looks up. A sim is
only as good as what you feed it — clean equations on guessed inputs is confident
garbage — so the datasets are the grounded values: material properties, physical
constants, empirical figures, each carrying WHERE it came from and WHEN.

Trust here is PROVENANCE, not validation. Real-world data changes and can't be
reproduced against a fixed answer the way a sim's SELFTEST can, so every dataset
is required to cite a source, and it's stamped with the date it was saved. That's
what makes it re-checkable instead of just asserted.

Sims pull from a dataset directly: inside a sim's run(), call dataset("name") to
get its data, so the computation runs on grounded real values, not made-up ones.
Kept on disk forever (cheap); only a query's result enters her context (scarce).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

DATASETS_DIR = Path.home() / "forge" / "datasets"
_CATALOG = DATASETS_DIR / "_catalog.json"


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")[:60]


def _load_catalog() -> dict:
    try:
        return json.loads(_CATALOG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_catalog(cat: dict) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CATALOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    tmp.replace(_CATALOG)


def _norm_sources(sources) -> list[str]:
    if isinstance(sources, str):
        sources = [sources]
    out: list[str] = []
    for s in (sources or []):
        s = str(s).strip()
        if s and s.lower() not in [x.lower() for x in out]:
            out.append(s)
    return out


def _root_domain(s: str) -> str:
    """The registrable-ish root of a source, so wikipedia.org/a and
    en.wikipedia.org/b count as ONE source, not two."""
    import re as _re
    m = _re.search(r"([a-z0-9-]+\.[a-z]{2,})(?:[/?#]|$)", s.lower())
    if m:
        return m.group(1)
    return s.strip().lower()


def save_dataset(name: str, data, sources, description: str = "") -> str:
    name = _safe_name(name)
    if not name:
        return "Error: give the dataset a name (letters/numbers/underscores)."
    srcs = _norm_sources(sources)
    if not srcs:
        return ("Error: cite at least one source — where did this data come from? "
                "Real-world data can't be validated like a sim; citation is its trust.")
    try:
        json.dumps(data)
    except Exception as e:
        return f"Error: data isn't JSON-serializable ({e}). Pass a plain object/list of values."
    # The gate used to check only the SOURCES, never the DATA — so an empty {}
    # and the bare string "nope" both saved wearing a ✓ CORROBORATED stamp
    # (the string was counted as "4 entries": it measured its letters). A
    # checkmark on nothing is worse than no checkmark, because a future sim
    # pulls it and reasons from it. The payload has to be real too.
    if isinstance(data, str) or isinstance(data, (int, float, bool)) or data is None:
        return ("Error: data must be a table of values, not a bare "
                f"{type(data).__name__} — e.g. {{\"tungsten\": {{\"melting_k\": 3695}}}}. "
                "A loose value has no keys to look up and can't be verified.")
    if not data:
        return ("Error: nothing to save — the data is empty. An empty dataset "
                "with a corroborated stamp is worse than no dataset, because "
                "anything that pulls it later trusts it.")
    # One touchstone isn't fact. Corroborated needs two sources from DIFFERENT
    # roots — two links off the same site, or two look-alike strings, are one
    # voice, not two (found live: 'wikipedia.org/a' + 'en.wikipedia.org/b' and
    # a pair of invented .blog names both passed as corroborated).
    roots = {_root_domain(s) for s in srcs}
    corroborated = len(roots) >= 2
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"data": data, "sources": srcs, "corroborated": corroborated,
           "description": str(description or "").strip(), "saved": time.time()}
    tmp = DATASETS_DIR / f"{name}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(DATASETS_DIR / f"{name}.json")
    n = len(data) if hasattr(data, "__len__") else None
    cat = _load_catalog()
    cat[name] = {"description": rec["description"], "sources": srcs,
                 "corroborated": corroborated, "saved": rec["saved"], "n": n,
                 "keys": (list(data.keys())[:40] if isinstance(data, dict) else None)}
    _save_catalog(cat)
    if corroborated:
        return (f"Saved '{name}' ({n if n is not None else '?'} entries) — ✓ CORROBORATED "
                f"by {len(srcs)} sources. Fact-grade ONLY IF those sources are independent — "
                f"two sites both copying the same origin isn't real corroboration; check that "
                f"they actually agree from different roots. Sims pull it with dataset('{name}').")
    return (f"Saved '{name}' ({n if n is not None else '?'} entries) as ⚠ SINGLE-SOURCE "
            f"(only: {srcs[0]}). NOT confirmed as fact — find a second INDEPENDENT source that "
            f"agrees and rebuild to promote it. Usable, but the numbers are provisional; say so "
            f"when you rely on them.")


def _cite(rec: dict) -> str:
    srcs = rec.get("sources") or ([rec["source"]] if rec.get("source") else [])
    tier = "corroborated ✓" if (rec.get("corroborated") or len(srcs) >= 2) else "⚠ SINGLE-SOURCE (provisional)"
    return f"[{tier} — sources: {'; '.join(srcs) or 'none'}]"


def query_dataset(name: str, key: str | None = None) -> str:
    name = _safe_name(name)
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        return f"No dataset '{name}'. Use list_datasets, or build_dataset to make it."
    rec = json.loads(path.read_text(encoding="utf-8"))
    data = rec.get("data")
    cite = _cite(rec)
    if key is not None and isinstance(data, dict):
        if key in data:
            return f"{name}[{key!r}] = {json.dumps(data[key], default=str)}  {cite}"
        hits = {k: v for k, v in data.items() if str(key).lower() in k.lower()}
        if hits:
            return f"{json.dumps(hits, default=str)[:1500]}  {cite}"
        return f"'{key}' not found in {name}. Keys: {list(data.keys())[:40]}"
    return f"{json.dumps(data, default=str)[:2500]}  {cite}"


def list_datasets() -> str:
    cat = _load_catalog()
    if not cat:
        return "No datasets yet. Use build_dataset to save cited real-world data."
    lines = [f"{len(cat)} dataset(s):"]
    for n, m in sorted(cat.items()):
        srcs = m.get("sources") or ([m["source"]] if m.get("source") else [])
        mark = "✓" if (m.get("corroborated") or len(srcs) >= 2) else "⚠"
        lines.append(f"  {mark} {n} — {m.get('description', '')} "
                     f"({m.get('n', '?')} entries; {len(srcs)} source"
                     f"{'s' if len(srcs) != 1 else ''}: {'; '.join(srcs)})")
    return "\n".join(lines)


SHELF_MAX = 14   # most she carries in her always-on prompt; the rest via list_datasets


def shelf_line() -> str:
    """Compact one-liner of the dataset shelf, for her context. Capped so a big
    shelf can't bloat her prompt — newest first, pointer to list_datasets for the
    rest. Empty if none."""
    cat = _load_catalog()
    if not cat:
        return ""
    items = sorted(cat.items(), key=lambda kv: -(kv[1].get("saved") or 0))
    parts = []
    for n, m in items[:SHELF_MAX]:
        srcs = m.get("sources") or ([m["source"]] if m.get("source") else [])
        mark = "✓" if (m.get("corroborated") or len(srcs) >= 2) else "⚠"
        desc = m.get("description", "")
        parts.append(f"{n} {mark}" + (f" — {desc}" if desc else ""))
    line = "; ".join(parts)
    extra = len(items) - min(len(items), SHELF_MAX)
    if extra > 0:
        line += f"; …+{extra} more (use list_datasets for the full shelf)"
    return line


def verify_shelf() -> str:
    """Integrity pass over the dataset shelf: every cataloged dataset must
    exist, parse, and still carry its data + sources. Flags orphans both ways.
    (Can't re-verify facts against the world — that's what corroboration was
    for at save time — but a corrupt or half-written file gets caught here.)"""
    cat = _load_catalog()
    lines = []
    for name in sorted(cat):
        path = DATASETS_DIR / f"{name}.json"
        if not path.exists():
            lines.append(f"  ✗ {name}: FILE MISSING — catalog orphan")
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            srcs = rec.get("sources") or []
            if rec.get("data") is None or not srcs:
                lines.append(f"  ⚠ {name}: file loads but data/sources incomplete")
            else:
                mark = "✓" if len(srcs) >= 2 else "⚠ single-source"
                lines.append(f"  {name}: {mark} intact ({len(srcs)} source(s))")
        except Exception as e:
            lines.append(f"  ✗ {name}: CORRUPT — {type(e).__name__}")
    for f in DATASETS_DIR.glob("*.json"):
        if f.stem not in cat and f.name != "_catalog.json":
            lines.append(f"  ? {f.stem}: on disk but not in catalog")
    return "Dataset shelf verification:\n" + ("\n".join(lines) or "  (empty shelf)")


def load_data(name: str):
    """For sims (via the injected dataset() helper): just the data, {} if missing."""
    path = DATASETS_DIR / f"{_safe_name(name)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("data", {})
    except Exception:
        return {}
