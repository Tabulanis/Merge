"""Opportunity / edge math — the quantitative side of 'a zillion odd ways to profit'.

The honest half of pattern-finding: math that tells you whether an edge is real
AND what it does to you if you're wrong. Prediction-market EV & implied odds,
Kelly sizing, two-venue arbitrage (after costs), forex carry, real-estate cap
rate / cash-on-cash / DSCR, commodity contango roll, and risk of ruin.

Every calc is selftest-verified. None of it is a recommendation to trade —
it's how you evaluate an opportunity YOU bring, and most of these notes exist
to remind you that +EV on paper is not the same as safe in the world.
"""
from __future__ import annotations

from .business import _fmt, _need


def ev(p: dict):
    """Expected value. Give `outcomes` (list of {prob, payoff}), or the simple
    triple win_prob / win_payoff (net gain) / loss_amount (net loss, positive)."""
    if p.get("outcomes"):
        outs = p["outcomes"]
        e = sum(float(o["prob"]) * float(o["payoff"]) for o in outs)
        tp = sum(float(o["prob"]) for o in outs)
        rows = [("expected value", e, "money")]
        if abs(tp - 1) > 0.01:
            rows.append(("_note", f"probabilities sum to {tp:.2f}, not 1 — check them.", "text"))
        else:
            rows.append(("_note", "positive EV = pays on average; says nothing about one outcome or your risk of ruin.", "text"))
        return rows
    win, W, L = _need(p, "win_prob", "win_payoff", "loss_amount")
    e = win * W - (1 - win) * L
    return [
        ("expected value", e, "money"),
        ("edge per $1 at risk", e / L if L else 0, "ratio"),
        ("_note", "+EV on average is not the same as safe — size it (Kelly) and check risk of ruin.", "text"),
    ]


def implied_prob(p: dict):
    """Turn odds into the probability the market is charging you. Give
    decimal_odds (2.5) or american_odds (+150 / -200). Add your_prob for the edge."""
    if p.get("decimal_odds") is not None:
        d = float(p["decimal_odds"])
    elif p.get("american_odds") is not None:
        a = float(p["american_odds"])
        d = (a / 100 + 1) if a > 0 else (100 / (-a) + 1)
    else:
        raise KeyError("decimal_odds or american_odds")
    ip = 1 / d
    rows = [("implied probability", ip, "pct"), ("break-even win rate", ip, "pct")]
    if p.get("your_prob") is not None:
        yp = float(p["your_prob"])
        rows += [
            ("your probability", yp, "pct"),
            ("edge vs market", yp - ip, "pct"),
            ("EV per $1 staked", yp * d - 1, "money"),
            ("_note", ("+EV — but ONLY if your probability is better-calibrated than the market's. That 'if' is the whole game."
                       if yp > ip else "no edge: the market prices it at least as high as you do."), "text"),
        ]
    return rows


def kelly(p: dict):
    """Growth-optimal bet size. win_prob + net_odds (b, payoff-to-1) or decimal_odds."""
    win, = _need(p, "win_prob")
    if p.get("decimal_odds") is not None:
        b = float(p["decimal_odds"]) - 1
    elif p.get("net_odds") is not None:
        b = float(p["net_odds"])
    else:
        raise KeyError("decimal_odds or net_odds (b = payoff-to-1)")
    f = (win * b - (1 - win)) / b if b else 0
    rows = [("full Kelly fraction", f, "pct"), ("half Kelly (safer)", f / 2, "pct")]
    if f <= 0:
        rows.append(("_note", "Kelly ≤ 0 → no edge, bet nothing. A positive f assumes your win_prob is honest; most aren't.", "text"))
    else:
        rows.append(("_note", "full Kelly is the growth-optimal MAX; bet a fraction (half or less) — a slightly-wrong win_prob makes full Kelly ruinous.", "text"))
    return rows


def arbitrage(p: dict):
    """Two-venue arb: decimal odds_a and odds_b on the two mutually-exclusive
    outcomes. Arb exists only if 1/a + 1/b < 1. cost_pct trims the margin."""
    a, b = _need(p, "odds_a", "odds_b")
    inv = 1 / a + 1 / b
    costs = float(p.get("cost_pct", 0) or 0)
    if inv >= 1:
        return [("combined implied prob", inv, "pct"),
                ("_note", "≥100% — no arbitrage; the books' margin eats it. This is the normal case, not the exception.", "text")]
    gross = 1 / inv - 1
    net = gross - costs
    return [
        ("combined implied prob", inv, "pct"),
        ("gross arb margin", gross, "pct"),
        ("net margin after costs", net, "pct"),
        ("stake on A (of $1)", (1 / a) / inv, "ratio"),
        ("stake on B (of $1)", (1 / b) / inv, "ratio"),
        ("_note", ("real but tiny and fragile: odds move, books cap your size, and one leg can fail (execution risk)."
                   if net > 0 else "costs wipe out the gross edge — not worth touching."), "text"),
    ]


def carry(p: dict):
    """Forex/rates carry. notional, rate_diff (annual, e.g. 0.03), holding_months,
    leverage (default 1)."""
    notional, rate_diff, months = _need(p, "notional", "rate_diff", "holding_months")
    lev = float(p.get("leverage", 1) or 1)
    return [
        ("carry income", notional * rate_diff * (months / 12), "money"),
        ("return on margin", rate_diff * lev * (months / 12), "pct"),
        ("_note", "carry is the easy part; the FX move can erase a year of it in a day — and leverage multiplies that loss too. Not free money.", "text"),
    ]


def cap_rate(p: dict):
    """Real estate: annual net operating income / price."""
    noi, price = _need(p, "noi", "price")
    return [("cap rate", noi / price, "pct"),
            ("_note", "cap rate = unlevered yield; higher can mean cheaper OR riskier. It ignores financing and appreciation.", "text")]


def cash_on_cash(p: dict):
    """Real estate: annual pre-tax cash flow / actual cash invested."""
    cf, cash = _need(p, "annual_cash_flow", "cash_invested")
    return [("cash-on-cash return", cf / cash, "pct"),
            ("_note", "this is the levered cash yield; it flatters you when debt is cheap and bites when it isn't.", "text")]


def dscr(p: dict):
    """Debt service coverage: NOI / annual debt payments. <1 = can't cover the loan."""
    noi, ds = _need(p, "noi", "annual_debt_service")
    r = noi / ds
    return [("DSCR", r, "ratio"),
            ("_note", ("comfortable — lenders usually want ≥1.25." if r >= 1.25
                       else "thin — under ~1.25 lenders balk and a bad month misses the mortgage."), "text")]


def contango(p: dict):
    """Commodity roll: spot, futures_price, months apart → annualized roll yield.
    Positive = contango (a cost to a long holder rolling futures)."""
    spot, fut, months = _need(p, "spot", "futures", "months")
    ann = (fut / spot - 1) * (12 / months)
    return [("annualized roll", ann, "pct"),
            ("_note", ("contango: futures above spot, so a long roller BLEEDS this per year — why commodity ETFs lag spot."
                       if ann > 0 else "backwardation: roll works in a long holder's favor."), "text")]


def risk_of_ruin(p: dict):
    """Gambler's-ruin probability for even-money bets: win_prob and bankroll in
    units (bankroll / bet size)."""
    win, units = _need(p, "win_prob", "bankroll_units")
    if win <= 0.5:
        return [("risk of ruin", 1.0, "pct"),
                ("_note", "no edge (≤50% at even money) → ruin is eventually CERTAIN. Don't.", "text")]
    ror = ((1 - win) / win) ** units
    return [("risk of ruin", ror, "pct"),
            ("_note", "even a real edge blows up if the bet is big vs bankroll — sizing beats being right. Deeper bankroll, smaller bets → ruin falls fast.", "text")]


def position(p: dict):
    """Map a signal to an ACTION tier — hold / small buy / big buy / short —
    sized by fractional Kelly. The gate: unless validated=true (the signal
    survived out-of-sample), the honest action is HOLD. shorts need
    allow_short=true and are the riskiest tier."""
    validated = bool(p.get("validated", False))
    if not validated:
        return [("action", "HOLD (0%)", "raw"),
                ("_note", "signal NOT validated out-of-sample → HOLD. Sizing an unproven signal "
                          "just dresses noise up as graded conviction — worse than useless. Prove "
                          "it (walk_forward) first.", "text")]
    win, = _need(p, "win_prob")
    if p.get("decimal_odds") is not None:
        b = float(p["decimal_odds"]) - 1
    else:
        b = float(p.get("net_odds", 1) or 1)
    full = (win * b - (1 - win)) / b if b else 0
    half = full / 2
    allow_short = bool(p.get("allow_short", False))
    if full <= 0:
        if allow_short and full < -0.02:
            sz = min(abs(half), 0.15)
            tier = "BIG SHORT" if sz >= 0.05 else "SMALL SHORT"
            return [("action", f"{tier} ({sz*100:.1f}% of bankroll)", "raw"),
                    ("_note", "validated NEGATIVE edge → short. But shorts add borrow cost, "
                              "liquidation risk, and unbounded downside — the riskiest tier. "
                              "Half-Kelly, hard stop, only on a strong edge that held out-of-sample.", "text")]
        return [("action", "HOLD (0%)", "raw"),
                ("_note", "no positive edge (Kelly ≤ 0) → HOLD. A validated signal doesn't always say 'buy'.", "text")]
    sz = min(half, 0.15)
    tier = "BIG BUY" if sz >= 0.05 else ("SMALL BUY" if sz >= 0.01 else "HOLD")
    return [("action", f"{tier} ({sz*100:.1f}% of bankroll)", "raw"),
            ("_note", "size = HALF Kelly, capped at 15%. 'big' vs 'small' is just how much edge you've "
                      "PROVEN — never full Kelly; a slightly-wrong win_prob makes it ruinous.", "text")]


_DISPATCH = {
    "ev": ev, "implied_prob": implied_prob, "kelly": kelly, "arbitrage": arbitrage,
    "carry": carry, "cap_rate": cap_rate, "cash_on_cash": cash_on_cash,
    "dscr": dscr, "contango": contango, "risk_of_ruin": risk_of_ruin, "position": position,
}
KINDS = ", ".join(_DISPATCH)


def calc(kind: str, params: dict | None = None) -> str:
    fn = _DISPATCH.get((kind or "").strip().lower())
    if not fn:
        return f"Unknown calc '{kind}'. Available: {KINDS}."
    try:
        rows = fn(params or {})
    except KeyError as e:
        return f"Error: missing required param(s) for {kind}: {e}."
    except (ValueError, ZeroDivisionError, TypeError) as e:
        return f"Error in {kind}: {e}"
    lines = [f"[markets_calc · {kind}]"]
    for label, value, vkind in rows:
        lines.append(f"  note: {value}" if label == "_note" else f"  {label}: {_fmt(value, vkind)}")
    return "\n".join(lines)


_SELFTESTS = [
    ("ev", {"win_prob": 0.4, "win_payoff": 100, "loss_amount": 50}, {"expected value": 10.0}),
    ("implied_prob", {"decimal_odds": 2.5}, {"implied probability": 0.40}),
    ("kelly", {"win_prob": 0.6, "net_odds": 1}, {"full Kelly fraction": 0.20}),
    ("arbitrage", {"odds_a": 2.1, "odds_b": 2.1}, {"gross arb margin": 0.05}),
    ("carry", {"notional": 10000, "rate_diff": 0.03, "holding_months": 12}, {"carry income": 300.0}),
    ("cap_rate", {"noi": 24000, "price": 400000}, {"cap rate": 0.06}),
    ("cash_on_cash", {"annual_cash_flow": 8000, "cash_invested": 100000}, {"cash-on-cash return": 0.08}),
    ("dscr", {"noi": 120000, "annual_debt_service": 100000}, {"DSCR": 1.20}),
    ("contango", {"spot": 100, "futures": 105, "months": 3}, {"annualized roll": 0.20}),
    ("risk_of_ruin", {"win_prob": 0.55, "bankroll_units": 10}, {"risk of ruin": 0.1344}),
]


def selftest(tol: float = 0.01) -> str:
    out, ok_all = [], True
    for kind, params, expect in _SELFTESTS:
        rows = _DISPATCH[kind](params)
        got = {label: value for label, value, _ in rows if label != "_note"}
        bad = [f"{k}: got {got.get(k)}, want {want}"
               for k, want in expect.items()
               if got.get(k) is None or abs(got[k] - want) > max(abs(want) * tol, 1e-4)]
        ok_all = ok_all and not bad
        out.append(f"  {'PASS' if not bad else 'FAIL'}  {kind}" + (f"  [{'; '.join(bad)}]" if bad else ""))
    return ("ALL PASS" if ok_all else "SOME FAILED") + "\n" + "\n".join(out)
