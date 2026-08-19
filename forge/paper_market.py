"""Honest paper-trading simulator — real market data, fake money, real costs.

The safe way to test trading logic: pull REAL prices (public, read-only, no
keys, no real trades), run a strategy against them, and — the part that makes it
honest instead of a fantasy — charge the REAL costs (fee + slippage) on every
fill, then benchmark against just HOLDING and against RANDOM. A strategy that
makes money but loses to buy-and-hold, or sits inside the noise of random, has
no edge. This is the skeptic framework made executable: it proves whether an
edge is real, on real numbers, for free — before a cent is ever exposed.

Long/flat only (no shorting, no leverage) — deliberately simple and honest.
"""
from __future__ import annotations

import random as _random

import httpx

# Kraken taker fee ~0.26%; a touch of slippage for a market fill. Real-ish.
DEFAULT_FEE = 0.0026
DEFAULT_SLIP = 0.0005


def fetch_closes(pair: str = "XBTUSD", interval: int = 60) -> list[float]:
    """Real historical closes from a public, no-auth endpoint. Read-only."""
    r = httpx.get("https://api.kraken.com/0/public/OHLC",
                  params={"pair": pair, "interval": interval}, timeout=25)
    r.raise_for_status()
    res = r.json().get("result", {})
    key = next((k for k in res if k != "last"), None)
    if not key:
        raise ValueError(f"no data for pair '{pair}'")
    return [float(c[4]) for c in res[key]]


class Portfolio:
    """Cash + one position. Every fill pays fee + slippage — that's the honesty."""

    def __init__(self, cash: float, fee: float, slip: float):
        self.start = cash
        self.cash = cash
        self.coin = 0.0
        self.fee = fee
        self.slip = slip
        self.trades = 0
        self.fees_paid = 0.0

    @property
    def long(self) -> bool:
        return self.coin > 0

    def _cost(self, notional: float) -> float:
        c = notional * (self.fee + self.slip)
        self.fees_paid += c
        return c

    def go_long(self, price: float):
        if self.long or self.cash <= 0:
            return
        notional = self.cash
        self.coin = (notional - self._cost(notional)) / price
        self.cash = 0.0
        self.trades += 1

    def go_flat(self, price: float):
        if not self.long:
            return
        notional = self.coin * price
        self.cash = notional - self._cost(notional)
        self.coin = 0.0
        self.trades += 1

    def equity(self, price: float) -> float:
        return self.cash + self.coin * price


def backtest(strategy, closes: list[float], params: dict | None = None,
             fee: float = DEFAULT_FEE, slip: float = DEFAULT_SLIP,
             start_cash: float = 1000.0) -> dict:
    """Step a long/flat strategy over real closes with honest costs.
    strategy(history_closes, params) -> 1 (be long) or 0 (be flat)."""
    p = Portfolio(start_cash, fee, slip)
    params = params or {}
    curve, peak, max_dd = [], start_cash, 0.0
    for i in range(1, len(closes)):
        price = closes[i]
        target = strategy(closes[:i + 1], params)
        if target == 1:
            p.go_long(price)
        elif target == 0:
            p.go_flat(price)
        # any other value (e.g. -1) = HOLD current position — lets band/
        # hysteresis strategies wait between their buy and sell lines.
        eq = p.equity(price)
        curve.append(eq)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak else 0)
    final = p.equity(closes[-1])
    return {"final": final, "return": final / start_cash - 1, "trades": p.trades,
            "fees_paid": p.fees_paid, "max_drawdown": max_dd, "curve": curve}


# ---- built-in strategies (history_closes, params) -> 0/1 ----
def strat_buy_and_hold(closes, params):
    return 1


def strat_sma_cross(closes, params):
    s = int(params.get("short", 10))
    lng = int(params.get("long", 30))
    if len(closes) < lng:
        return 0
    short_ma = sum(closes[-s:]) / s
    long_ma = sum(closes[-lng:]) / lng
    return 1 if short_ma > long_ma else 0


def strat_band(closes, params):
    """The classic hand-trader's routine, formalized: buy when price is
    stretched LOW vs its recent average, sell when stretched HIGH, and WAIT
    (hold whatever you're in) between the lines. 'Harvest the wiggle.'"""
    win = int(params.get("win", 20))
    band = float(params.get("band", 0.05))
    if len(closes) < win:
        return 0
    ma = sum(closes[-win:]) / win
    price = closes[-1]
    if price < ma * (1 - band):
        return 1        # low vs its own range -> buy
    if price > ma * (1 + band):
        return 0        # high vs its own range -> sell
    return -1           # between the lines -> wait


STRATEGIES = {"buy_and_hold": strat_buy_and_hold, "sma_cross": strat_sma_cross,
              "band": strat_band}


def _random_benchmark(closes, params, fee, slip, start_cash, runs=25):
    """Average return of coin-flip strategies — should be net-NEGATIVE after
    fees (random churns costs), the baseline any real edge must clear."""
    rng = _random.Random(1234)
    total = 0.0
    for _ in range(runs):
        seq = [rng.random() for _ in closes]

        def rnd(hist, _p, _seq=seq):
            return 1 if _seq[len(hist) - 1] > 0.5 else 0
        total += backtest(rnd, closes, {}, fee, slip, start_cash)["return"]
    return total / runs


def run(pair: str = "XBTUSD", interval: int = 60, strategy: str = "sma_cross",
        params: dict | None = None, start_cash: float = 1000.0) -> str:
    """Fetch real data, backtest the strategy honestly, and score it against
    buy-and-hold AND random. Returns a paste-ready honest scorecard."""
    fn = STRATEGIES.get((strategy or "").strip().lower())
    if not fn:
        return f"Unknown strategy '{strategy}'. Available: {', '.join(STRATEGIES)}."
    try:
        closes = fetch_closes(pair, interval)
    except Exception as e:
        return f"Couldn't fetch market data ({e}). Read-only public data — check the pair/network."
    if len(closes) < 40:
        return "Not enough data returned to backtest."
    r = backtest(fn, closes, params, DEFAULT_FEE, DEFAULT_SLIP, start_cash)
    bh = backtest(strat_buy_and_hold, closes, {}, DEFAULT_FEE, DEFAULT_SLIP, start_cash)
    rnd = _random_benchmark(closes, params, DEFAULT_FEE, DEFAULT_SLIP, start_cash)
    span_hrs = len(closes) * interval / 60
    beat_hold = r["return"] > bh["return"]
    above_random = r["return"] > rnd
    verdict = ("NO EDGE — " + ("it even lost to just holding. " if not beat_hold else "")
               + ("and it's not above random. " if not above_random else "")
               ).strip() or ("cleared both benchmarks on THIS one window — but a strategy "
                              "clears both on a single window roughly HALF the time by pure luck, "
                              "so this means 'not yet dismissed', NOT 'promising'. Only many "
                              "windows and out-of-sample survival would make it interesting.")
    return (
        f"[paper_market · {strategy} on {pair}, real data]\n"
        f"  window: ~{span_hrs:.0f} hours ({len(closes)} candles), ${start_cash:,.0f} start, fees/slippage ON\n"
        f"  strategy return:    {r['return']*100:+.2f}%   ({r['trades']} trades, ${r['fees_paid']:.2f} paid in fees)\n"
        f"  buy-and-hold:       {bh['return']*100:+.2f}%   (1 trade — the benchmark to beat)\n"
        f"  random (avg of 25): {rnd*100:+.2f}%   (the noise floor)\n"
        f"  max drawdown:       {r['max_drawdown']*100:.1f}%\n"
        f"  VERDICT: {verdict}\n"
        f"  note: one window on real fees is a start, not proof — a real edge survives many windows, "
        f"different periods, and out-of-sample data. Most 'strategies' die right here, to the fees.")
