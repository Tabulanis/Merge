"""The X-Files — a casebook of market anomalies, and the hunt for the third party.

An "odd couple" is two assets that move together when they've no business to
(Ethereum tracking the S&P). The interesting question isn't THAT they correlate
— it's WHO'S PULLING BOTH STRINGS. This module:

  * pulls aligned daily data across asset classes (crypto via Kraken, stocks &
    macro via FRED — keyless),
  * measures whether an odd couple's correlation is even real (out-of-sample),
  * and hunts the THIRD PARTY: for each candidate driver Z, computes the partial
    correlation of A,B controlling for Z — if the A–B link collapses once you
    account for Z, Z is the puppet master. Ranked, and confirmed out-of-sample.

Honest to the bone (the market-rig rule): a correlation that dies out-of-sample
is reported as noise, and a "third party" only counts if removing it actually
kills the link on data it never saw. It finds the SUSPECT, not a conviction.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (research)"}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
KRAKEN = "https://api.kraken.com/0/public/OHLC"
CASES = Path.home() / "forge" / "datasets" / "xfiles"

# Friendly name -> (source, code). Crypto is Kraken; the rest are FRED series.
TICKERS = {
    "ETH": ("kraken", "ETHUSD"), "BTC": ("kraken", "XXBTZUSD"),
    "SOL": ("kraken", "SOLUSD"), "ADA": ("kraken", "ADAUSD"),
    "XRP": ("kraken", "XRPUSD"), "LINK": ("kraken", "LINKUSD"),
    "SPX": ("fred", "SP500"), "SP500": ("fred", "SP500"), "S&P": ("fred", "SP500"),
    "VIX": ("fred", "VIXCLS"), "DXY": ("fred", "DTWEXBGS"), "DOLLAR": ("fred", "DTWEXBGS"),
    "US10Y": ("fred", "DGS10"), "RATES": ("fred", "DGS10"),
    "OIL": ("fred", "DCOILWTICO"), "NASDAQ": ("fred", "NASDAQCOM"),
    "HYSPREAD": ("fred", "BAMLH0A0HYM2"),   # high-yield credit spread = risk appetite
    "M2": ("fred", "WM2NS"),                # money supply = liquidity
}
# default third-party suspects when none named
DEFAULT_SUSPECTS = ["SPX", "VIX", "DXY", "US10Y", "OIL", "BTC", "HYSPREAD"]


def _norm(name: str) -> str:
    return name.strip().upper().replace("USD", "").replace("-", "").replace(" ", "") or name


def _fetch_fred(code: str) -> dict:
    r = httpx.get(FRED, params={"id": code}, headers=UA, timeout=25, follow_redirects=True)
    r.raise_for_status()
    out = {}
    for ln in r.text.splitlines()[1:]:
        parts = ln.split(",")
        if len(parts) >= 2 and parts[-1] not in ("", ".", "null"):
            try:
                out[parts[0]] = float(parts[-1])
            except ValueError:
                continue
    return out


def _fetch_kraken(pair: str) -> dict:
    r = httpx.get(KRAKEN, params={"pair": pair, "interval": 1440}, headers=UA, timeout=25)
    r.raise_for_status()
    res = r.json().get("result", {})
    rows = next((v for k, v in res.items() if k != "last"), [])
    out = {}
    for row in rows:
        d = time.strftime("%Y-%m-%d", time.gmtime(int(row[0])))
        out[d] = float(row[4])          # close
    return out


def _series(name: str) -> dict:
    key = _norm(name)
    src, code = TICKERS.get(key, TICKERS.get(name.upper(), (None, None)))
    if src is None:
        # bare crypto pair like "DOTUSD"
        return _fetch_kraken(name.upper() if name.upper().endswith("USD") else name.upper() + "USD")
    return _fetch_fred(code) if src == "fred" else _fetch_kraken(code)


def _aligned_returns(names, n=400, required=2):
    """Daily log-returns aligned on common trading days. The first `required`
    names (the odd couple) define the window and MUST fetch; the rest (suspects)
    are optional — one that fails to fetch or doesn't cover the window is
    dropped, not fatal. Returns (dates, rets, dropped)."""
    series, dropped = {}, []
    for i, nm in enumerate(names):
        try:
            s = _series(nm)
            if not s:
                raise ValueError("no data")
            series[nm] = s
        except Exception:
            if i < required:
                raise ValueError(f"couldn't fetch '{nm}' — can't run without it")
            dropped.append(nm)
    base = [series[names[i]] for i in range(required)]
    common = set.intersection(*(set(s) for s in base))
    dates = sorted(common)[-n:]
    if len(dates) < 30:
        raise ValueError(f"only {len(dates)} common days for {names[:required]} — not enough overlap")
    rets = {}
    for nm, s in series.items():
        is_pair = nm in names[:required]
        have = [d for d in dates if d in s]
        # a suspect used to be dropped WHOLESALE for one missing holiday (VIX
        # covered 399/400 and got tossed). Keep any that covers ~90%+ of the
        # window and forward-fill the handful of gaps; only truly sparse series
        # (e.g. weekly M2 at ~17%) are too thin to trust and get dropped.
        if not is_pair and len(have) < max(int(0.9 * len(dates)), 100):
            dropped.append(nm)
            continue
        lv, last = [], None
        for d in dates:
            if d in s:
                last = s[d]
            lv.append(last if last is not None else s[have[0]])
        rets[nm] = np.diff(np.log(np.clip(np.array(lv, dtype=float), 1e-9, None)))
    return dates, rets, dropped


def _corr(x, y):
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _partial(a, b, z):
    """First-order partial correlation of a,b controlling for z."""
    rab, raz, rbz = _corr(a, b), _corr(a, z), _corr(b, z)
    denom = np.sqrt(max(1e-9, (1 - raz ** 2) * (1 - rbz ** 2)))
    return float((rab - raz * rbz) / denom)


def find_third_party(a: str, b: str, suspects=None, n: int = 400) -> str:
    """The hunt: who drives BOTH `a` and `b`? Measures their correlation, checks
    it's real out-of-sample, then for each suspect Z reports how much the a–b
    link collapses once Z is accounted for. A suspect that kills the link (on
    held-out data too) is the third party."""
    if _norm(a) == _norm(b):
        return (f"'{a}' and '{b}' are the same thing — an odd couple needs two "
                "different assets. Pick a real pair.")
    # a model often hands `suspects` as a string ("VIX" or "VIX,DXY") — iterating
    # that char-by-char turns it into bogus one-letter suspects. Split it first.
    if isinstance(suspects, str):
        suspects = [s for s in re.split(r"[,\s]+", suspects) if s]
    suspects = [s.strip() for s in (suspects or DEFAULT_SUSPECTS)
                if s and str(s).strip() and _norm(str(s).strip()) not in (_norm(a), _norm(b))]
    names = [a, b] + suspects
    try:
        dates, rets, dropped = _aligned_returns(names, n)
    except Exception as e:
        return f"Couldn't assemble the data: {e}"
    suspects = [z for z in suspects if z in rets]      # keep only those that aligned
    A, B = rets[a], rets[b]
    cut = int(len(A) * 0.6)

    base_all = _corr(A, B)
    if abs(base_all) > 0.999:
        return (f"'{a}' and '{b}' are effectively the SAME asset (correlation "
                f"{base_all:+.2f}) — two names for one thing. Pick two genuinely "
                "different markets to make an odd couple.")
    base_tr, base_te = _corr(A[:cut], B[:cut]), _corr(A[cut:], B[cut:])
    crit = 2.6 / np.sqrt(max(len(A) - cut, 2))     # ~99% critical |r| for the test window
    real = (abs(base_te) > max(0.12, crit) and abs(base_tr) > crit
            and np.sign(base_tr) == np.sign(base_te))

    lines = [f"THE ODD COUPLE: {a} vs {b}  ({len(A)} aligned days)",
             f"  correlation: {base_all:+.2f} overall  "
             f"(train {base_tr:+.2f} / out-of-sample {base_te:+.2f})"]
    if not real:
        lines.append("  ⚠ the link itself doesn't hold out-of-sample — it may be "
                     "noise or a passing phase. Hunting a cause for a correlation "
                     "that isn't stable would be chasing a ghost. Verdict: no "
                     "solid case here yet.")
        return "\n".join(lines)

    lines.append("  the link holds out-of-sample — worth hunting the third party.")
    lines.append("\nSUSPECTS (how much each KILLS the link when accounted for):")
    scored = []
    for z in suspects:
        Z = rets[z]
        p_all = _partial(A, B, Z)
        p_te = _partial(A[cut:], B[cut:], Z[cut:])
        drop = abs(base_all) - abs(p_all)                     # correlation removed
        oos_drop = abs(base_te) - abs(p_te)
        scored.append((drop, oos_drop, p_all, z))
    scored.sort(reverse=True)
    for drop, oos_drop, p_all, z in scored:
        pct = 100 * drop / (abs(base_all) or 1)
        tag = ""
        if abs(p_all) < 0.12 and oos_drop > 0.05:
            tag = "  ← THIRD PARTY: removing it collapses the link (holds out-of-sample)"
        elif drop > 0.10 and oos_drop > 0:
            tag = "  ← strong influence"
        lines.append(f"  control for {z:6}: link {base_all:+.2f} → {p_all:+.2f} "
                     f"({pct:+.0f}% of it gone){tag}")
    if not scored:
        lines.append("\n→ None of the named suspects could be lined up with enough "
                     "overlapping data to test (bad names, or no shared history). The "
                     "link is real, but the lineup came up empty — try other suspects "
                     "(e.g. SPX, VIX, DXY, US10Y, OIL, M2) and re-run.")
        return "\n".join(lines)
    top = scored[0]
    if abs(top[2]) < 0.12 and top[1] > 0.05:
        lines.append(f"\n→ Prime suspect: {top[3]}. Once you account for it, {a} and "
                     f"{b} barely relate — it's plausibly driving both.")
    else:
        lines.append("\n→ No single suspect fully explains the link — it may be a mix, "
                     "or a driver not in the lineup. Add suspects and re-run.")
    lines.append("Honest limit: this finds a SUSPECT (a statistical common driver), "
                 "not proof of cause. Not investment advice.")
    return "\n".join(lines)


# ---- the casebook: flag / list / annotate -------------------------------
def _slug(s):
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50] or "case"


def flag_xfile(title: str, a: str, b: str, note: str = "") -> str:
    """Open an X-File: flag an odd couple as a case to investigate."""
    CASES.mkdir(parents=True, exist_ok=True)
    cid = _slug(title)
    p = CASES / f"{cid}.json"
    case = {"id": cid, "title": title, "a": a, "b": b, "status": "open",
            "opened": time.time(), "note": note, "findings": []}
    if p.exists():
        case = json.loads(p.read_text()); case["note"] = note or case.get("note", "")
    p.write_text(json.dumps(case, indent=1))
    return f"X-File '{title}' flagged [{a} × {b}]. Run find_third_party on it to hunt the driver."


def list_xfiles() -> str:
    if not CASES.is_dir():
        return "No X-Files yet. flag_xfile to open one."
    rows = []
    for f in sorted(CASES.glob("*.json")):
        try:
            c = json.loads(f.read_text())
            rows.append(f"  [{c.get('status','?')}] {c['title']} — {c['a']} × {c['b']}"
                        f"  ({len(c.get('findings',[]))} findings)")
        except Exception:
            continue
    return "THE X-FILES:\n" + ("\n".join(rows) if rows else "  (empty)")
