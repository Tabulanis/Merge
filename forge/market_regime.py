"""Long-term regime analysis — how trends and strategies behave across bull/bear.

Two honest ideas, on real long-history data (daily, ~2 years from public Kraken):

  1. THE STATE VECTOR (the "4-D" view): each day is a point in a small feature
     space — [trend, momentum, volatility, drawdown]. Regimes (bull / bear /
     neutral) are regions of that space, labelled by trend-vs-long-average.

  2. REGIME-CONDITIONAL PERFORMANCE: a strategy that's brilliant in bulls and
     awful in bears has NO real edge — just a bet on the regime. So we split a
     strategy's return BY regime and show it next to buy-and-hold in the same
     regime. That's the honest test of "amazing in bulls, shit in the other."

Descriptive, not predictive: regimes are only clean in HINDSIGHT (you learn a
bear started after it's confirmed), so this explains structure — it does not
forecast the next regime, and switching strategy by a *guessed* regime adds its
own error. Read it to understand, not to time.
"""
from __future__ import annotations

import numpy as np

from .paper_market import DEFAULT_FEE, DEFAULT_SLIP, Portfolio, STRATEGIES, fetch_closes

TREND_WIN = 30      # medium trend lookback (days)
VOL_WIN = 30        # volatility lookback
DD_WIN = 90         # drawdown-from-peak lookback
MA_LONG = 100       # long average for the bull/bear line
WARMUP = MA_LONG    # need this much history before labelling
TREND_BAND = 0.03   # dead-band so we don't flip regime on noise


def _state_vectors(closes: np.ndarray):
    """Per-day [trend, momentum, volatility, drawdown] and a regime label."""
    n = len(closes)
    rets = np.diff(closes) / closes[:-1]
    vecs, regimes, idx = [], [], []
    for i in range(WARMUP, n):
        trend = closes[i] / closes[i - TREND_WIN] - 1
        momentum = closes[i - 9:i + 1].mean() / closes[i - 49:i + 1].mean() - 1
        vol = rets[i - VOL_WIN:i].std() * np.sqrt(365)
        drawdown = closes[i] / closes[i - DD_WIN:i + 1].max() - 1
        ma_long = closes[i - MA_LONG + 1:i + 1].mean()
        if closes[i] > ma_long and trend > TREND_BAND:
            reg = "bull"
        elif closes[i] < ma_long and trend < -TREND_BAND:
            reg = "bear"
        else:
            reg = "neutral"
        vecs.append([trend, momentum, vol, drawdown])
        regimes.append(reg)
        idx.append(i)
    return np.array(vecs), regimes, idx


def _runs(regimes):
    """Average consecutive-run length per regime (how long they last)."""
    out = {}
    if not regimes:
        return out
    cur, length = regimes[0], 1
    runs = {}
    for r in regimes[1:]:
        if r == cur:
            length += 1
        else:
            runs.setdefault(cur, []).append(length)
            cur, length = r, 1
    runs.setdefault(cur, []).append(length)
    for r, ls in runs.items():
        out[r] = sum(ls) / len(ls)
    return out


def run(pair: str = "XBTUSD", strategy: str = "sma_cross",
        strat_params: dict | None = None, start_cash: float = 1000.0) -> str:
    fn = STRATEGIES.get((strategy or "").strip().lower())
    if not fn:
        return f"Unknown strategy '{strategy}'. Available: {', '.join(STRATEGIES)}."
    try:
        closes = np.array(fetch_closes(pair, 1440))   # daily
    except Exception as e:
        return f"Couldn't fetch daily data ({e})."
    if len(closes) < WARMUP + 40:
        return "Not enough daily history to analyze regimes."

    vecs, regimes, idx = _state_vectors(closes)
    params = strat_params or {}

    # --- run the strategy vs buy-and-hold over the labelled span, honest costs ---
    p = Portfolio(start_cash, DEFAULT_FEE, DEFAULT_SLIP)
    bh_units = (start_cash * (1 - DEFAULT_FEE - DEFAULT_SLIP)) / closes[idx[0]]
    prev_strat = start_cash
    prev_close = closes[idx[0]]
    by = {r: {"strat": 1.0, "bh": 1.0, "days": 0} for r in ("bull", "bear", "neutral")}
    for k, i in enumerate(idx):
        price = closes[i]
        p.go_long(price) if fn(list(closes[:i + 1]), params) == 1 else p.go_flat(price)
        strat_eq = p.equity(price)
        if k > 0:
            r = regimes[k]
            by[r]["strat"] *= strat_eq / prev_strat
            by[r]["bh"] *= price / prev_close
            by[r]["days"] += 1
        prev_strat, prev_close = strat_eq, price
    strat_total = p.equity(closes[idx[-1]]) / start_cash - 1
    bh_total = bh_units * closes[idx[-1]] / start_cash - 1

    # --- regime characterization (the mean state vector per regime) ---
    regs = np.array(regimes)
    runs = _runs(regimes)
    days = len(regimes)
    lines = [f"[market_regime · {pair} daily, ~{days} labelled days]",
             f"  overall: strategy {strat_total*100:+.1f}%   buy-and-hold {bh_total*100:+.1f}%",
             "  --- regimes (share of time · avg run · mean state vector [trend/mom/vol/draw]) ---"]
    for r in ("bull", "bear", "neutral"):
        mask = regs == r
        share = mask.mean() * 100 if days else 0
        if not mask.any():
            lines.append(f"  {r:8} {share:4.0f}%  (none)")
            continue
        mv = vecs[mask].mean(axis=0)
        lines.append(f"  {r:8} {share:4.0f}%  avg run {runs.get(r,0):4.0f}d  "
                     f"[{mv[0]*100:+5.0f}% / {mv[1]*100:+4.0f}% / {mv[2]*100:3.0f}%vol / {mv[3]*100:4.0f}%dd]")
    lines.append("  --- the point: strategy vs just-holding, BY regime ---")
    for r in ("bull", "bear", "neutral"):
        if by[r]["days"]:
            s = (by[r]["strat"] - 1) * 100
            b = (by[r]["bh"] - 1) * 100
            edge = s - b
            lines.append(f"  in {r:8}: strategy {s:+6.1f}%  vs hold {b:+6.1f}%   "
                         f"({'beats hold' if edge > 0 else 'loses to hold'} by {abs(edge):.1f}%)")
    lines.append("  note: a strategy that only wins in bull is basically 'be long' in disguise — "
                 "in a bull, holding wins too. And regimes are only clean in HINDSIGHT; you can't "
                 "know today's for sure, so switching by a guessed regime adds its own errors. "
                 "Understand the structure here; don't mistake it for a crystal ball.")
    return "\n".join(lines)
