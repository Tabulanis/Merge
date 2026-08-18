"""Multi-signal scanner with the overfitting demon caged.

Throw everything at it — price features, calendar, lunar, and their CROSS
PRODUCTS (pairwise interactions) — and it can only hand back what clears a gauntlet:

  - a MIN-SAMPLE floor per bucket (sparse interactions auto-rejected — a pattern
    in 5 days is a coincidence),
  - an OUT-OF-SAMPLE split (found on train, must still hold on unseen test),
  - and the kill-shot: it re-runs the ENTIRE search on SHUFFLED (fake) data to
    measure how many 'survivors' pure luck produces. If your real survivors don't
    beat that chance baseline, the whole haul is noise — no matter how good it looks.

Markets aren't textbook, so we DO want to look wide. This just makes sure a wide
look can't fool us: more combinations tested = a higher bar, measured directly.
"""
from __future__ import annotations

import datetime as dt
import math
import random

import httpx
import numpy as np


def _fetch(pair, interval=1440):
    r = httpx.get("https://api.kraken.com/0/public/OHLC",
                  params={"pair": pair, "interval": interval}, timeout=25).json()
    key = next(k for k in r["result"] if k != "last")
    rows = r["result"][key]
    return (np.array([int(x[0]) for x in rows]),
            np.array([float(x[4]) for x in rows]),
            np.array([float(x[6]) for x in rows]))   # volume — the manipulation footprint


def _lunar(ts):
    ph = ((ts - 947182440) % (29.530588853 * 86400)) / (29.530588853 * 86400)
    return (1 - math.cos(2 * math.pi * ph)) / 2


def _zspike(x, win=30, thr=2.0):
    """Per-day z-score of x vs its trailing window — the 'huh?' / anomaly detector."""
    n = len(x)
    z = np.zeros(n)
    for i in range(n):
        w = x[max(0, i - win):i]
        if len(w) > 5 and w.std() > 0:
            z[i] = (x[i] - w.mean()) / w.std()
    return z


def _features(t, c, v):
    """Bucketed feature labels per day (categorical, human-readable)."""
    n = len(c)
    rets = np.diff(c) / c[:-1]
    absret = np.abs(np.concatenate([[0], rets]))
    f = {}
    # manipulation footprints: abnormal volume / abnormal move sticking out of the noise
    f["vol_shock"] = np.where(_zspike(v) > 2.0, "vol_SPIKE", "vol_normal")
    f["move_shock"] = np.where(_zspike(absret) > 2.0, "move_SHOCK", "move_calm")
    dow = np.array([dt.datetime.fromtimestamp(x, dt.UTC).weekday() for x in t])
    dom = np.array([dt.datetime.fromtimestamp(x, dt.UTC).day for x in t])
    f["weekday"] = np.where(dow >= 5, "weekend", "weekday")
    f["turn_of_month"] = np.where(np.isin(dom, [28, 29, 30, 31, 1, 2, 3]), "ToM", "mid")
    illum = np.array([_lunar(x) for x in t])
    f["moon"] = np.where(illum > 0.7, "full", np.where(illum < 0.3, "new", "mid"))
    trail = np.array([c[i] / c[i - 5] - 1 if i >= 5 else 0 for i in range(n)])
    f["trail_5d"] = np.where(trail > 0, "up", "down")
    vol = np.array([rets[max(0, i - 20):i].std() if i > 5 else 0 for i in range(n)])
    f["vol"] = np.where(vol > np.median(vol[60:]), "high_vol", "low_vol")
    dd = np.array([c[i] / c[max(0, i - 60):i + 1].max() - 1 for i in range(n)])
    f["drawdown"] = np.where(dd < -0.10, "deep_dd", "near_high")
    ma10 = np.array([c[max(0, i - 9):i + 1].mean() for i in range(n)])
    ma30 = np.array([c[max(0, i - 29):i + 1].mean() for i in range(n)])
    f["momentum"] = np.where(ma10 > ma30, "bull", "bear")
    return f


def scan(pair: str = "XBTUSD", horizon: int = 5, min_n: int = 25,
         thresh: float = 0.015, perms: int = 25) -> dict:
    t, c, vol_arr = _fetch(pair)
    n = len(c)
    fwd = c[horizon:] / c[:-horizon] - 1
    feats = {k: arr[:-horizon] for k, arr in _features(t, c, vol_arr).items()}
    warm = 60
    idx = np.arange(warm, len(fwd))
    train = idx[idx < warm + len(idx) // 2]
    test = idx[idx >= warm + len(idx) // 2]

    # candidate masks: singles + pairwise cross products
    cands = []
    names = list(feats)
    for k in names:                                   # singles
        for val in np.unique(feats[k]):
            cands.append((f"{k}={val}", feats[k] == val))
    for a in range(len(names)):                        # pairwise crosses
        for b in range(a + 1, len(names)):
            ka, kb = names[a], names[b]
            for va in np.unique(feats[ka]):
                for vb in np.unique(feats[kb]):
                    cands.append((f"{ka}={va} & {kb}={vb}", (feats[ka] == va) & (feats[kb] == vb)))

    def survivors(target):
        b_tr, b_te = target[train].mean(), target[test].mean()
        keep, tested = [], 0
        for label, mask in cands:
            mtr, mte = np.intersect1d(train, idx[mask[idx]]), np.intersect1d(test, idx[mask[idx]])
            if len(mtr) < min_n or len(mte) < min_n:
                continue
            tested += 1
            etr = target[mtr].mean() - b_tr
            ete = target[mte].mean() - b_te
            if np.sign(etr) == np.sign(ete) and abs(ete) >= thresh:
                keep.append((label, ete))
        return keep, tested

    real, tested = survivors(fwd)
    rng = random.Random(7)

    def _block_shuffle(x):
        # shuffle in blocks of ~horizon so the null PRESERVES the autocorrelation
        # of overlapping forward returns — otherwise long horizons throw false
        # positives (a plain shuffle destroys the overlap and under-counts chance).
        bs = max(horizon, 1)
        blocks = [x[i:i + bs] for i in range(0, len(x), bs)]
        rng.shuffle(blocks)
        return np.concatenate(blocks)[:len(x)]

    null = []
    for _ in range(perms):
        null.append(len(survivors(_block_shuffle(fwd))[0]))
    null_avg = np.mean(null)

    return {"pair": pair, "horizon": horizon, "tested": tested, "perms": perms,
            "thresh": thresh, "real": real, "chance": float(null_avg)}


def run(pair: str = "XBTUSD", horizon: int = 5, min_n: int = 25,
        thresh: float = 0.015, perms: int = 25) -> str:
    r = scan(pair, horizon, min_n, thresh, perms)
    real, null_avg, tested = r["real"], r["chance"], r["tested"]
    n_single = sum(1 for lbl, _ in real if "&" not in lbl)
    lines = [f"[scanner · {r['pair']} · {tested} combos with enough data (singles + cross products), "
             f"{horizon}-day forward return, ≥{thresh*100:.1f}% out-of-sample]",
             f"  REAL survivors: {len(real)}  ({n_single} single, {len(real)-n_single} cross-product)",
             f"  survivors on PURE LUCK (shuffled data, avg of {perms} runs): {null_avg:.1f}",
             f"  VERDICT: " + (
                 f"real ({len(real)}) ≤ chance ({null_avg:.1f}) → it's ALL NOISE. The wide search "
                 f"found nothing luck wouldn't. This is the honest, near-universal result."
                 if len(real) <= null_avg * 1.3 else
                 f"real ({len(real)}) exceeds chance ({null_avg:.1f}) — a whiff worth a HARDER look "
                 f"(more windows, formal correction). Not proof, just not dismissed yet.")]
    if real and len(real) > null_avg * 1.3:
        lines.append("  strongest surviving combos (still suspects, not conclusions):")
        for lbl, e in sorted(real, key=lambda x: -abs(x[1]))[:5]:
            lines.append(f"    {lbl}   ({e*100:+.1f}% OOS)")
    lines.append("  note: this is ONE asset, ONE split. Even a chance-beating hit needs many rolling "
                 "windows + real multiple-testing correction before it's more than a suspect.")
    return "\n".join(lines)
