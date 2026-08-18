"""Out-of-sample / walk-forward testing — the honesty rig for strategies.

The one law that separates a real edge from a curve-fit: does it survive on data
it never saw? Split history into TRAIN (where you pick the 'best' params) and
TEST (held out). Grade the in-sample winner OUT of sample. The gap between
in-sample glory and out-of-sample reality IS the overfitting, measured. And the
kill-shot metric: the rank correlation between in-sample and out-of-sample
returns across the whole param grid — if it's ~0 or negative, picking the
'best' backtest tells you NOTHING about the future. That's most 'strategies'.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .paper_market import DEFAULT_FEE, DEFAULT_SLIP, backtest, fetch_closes, strat_sma_cross


def _hold(c, p):
    return 1


def _grid():
    return [(s, l) for s in (5, 10, 15, 20) for l in (30, 50, 80, 120) if s < l]


def run(pair: str = "XBTUSD", interval: int = 1440, train_frac: float = 0.6,
        start_cash: float = 1000.0) -> str:
    closes = fetch_closes(pair, interval)
    n = len(closes)
    if n < 200:
        return "Not enough history for a clean train/test split."
    cut = int(n * train_frac)
    train, test = closes[:cut], closes[cut:]
    grid = _grid()

    ins = [((s, l), backtest(strat_sma_cross, train, {"short": s, "long": l})["return"]) for s, l in grid]
    oos = {(s, l): backtest(strat_sma_cross, test, {"short": s, "long": l})["return"] for s, l in grid}
    ins.sort(key=lambda x: -x[1])
    best_params, best_ins = ins[0]
    best_oos = oos[best_params]
    bh_test = backtest(_hold, test, {})["return"]
    oos_avg = float(np.mean(list(oos.values())))

    # does in-sample ranking predict out-of-sample? (the kill-shot)
    in_r = [r for _, r in ins]
    out_r = [oos[p] for p, _ in ins]
    rho, _ = spearmanr(in_r, out_r)

    beat_hold = best_oos > bh_test
    return (
        f"[walk-forward · {pair} daily · train {cut}d / test {n-cut}d · {len(grid)} param sets]\n"
        f"  IN-SAMPLE best: SMA{best_params}  ->  {best_ins*100:+.1f}%  (the 'winner' you'd have picked)\n"
        f"  OUT-OF-SAMPLE, that same winner: {best_oos*100:+.1f}%   (buy-and-hold on test: {bh_test*100:+.1f}%)\n"
        f"  -> in-sample winner {'BEATS' if beat_hold else 'LOSES to'} just holding, out of sample\n"
        f"  out-of-sample avg across ALL {len(grid)} params: {oos_avg*100:+.1f}%  "
        f"(is the 'best' special, or was it lucky in-sample?)\n"
        f"  KILL-SHOT — in-sample vs out-of-sample rank correlation: rho = {rho:+.2f}\n"
        f"     ({'~0/negative: the best backtest predicts NOTHING about the future — textbook overfitting.' if rho < 0.3 else 'some persistence — worth a longer, honest look, still not proof.'})\n"
        f"  note: this is ONE split on one asset — real proof needs many splits, many assets, walk-forward "
        f"windows, and it STILL only tells you what didn't fail yet. But a strategy that can't clear this "
        f"bar is done here, for free.")
