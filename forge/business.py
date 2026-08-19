"""Engineer's-eye business math — the money mechanics, computed exactly.

Business for someone who builds: first-principles and quantitative, no b-school
fluff. Each function is a lever you'd otherwise do in your head and occasionally
slip on — unit economics, runway, break-even, discounted cash flow, loan
amortization, pricing (margin vs markup, the classic trap), fundraise dilution,
growth. One dispatcher tool (business_calc) exposes them all.

Trust here is the same as the sim shelf: every calc carries a known-answer
selftest (see selftest()), so the numbers are verified, not asserted. What the
tool can't do is judgment — that stays honest framework + her own read; and it
never does jurisdiction-specific legal/tax/investment procedure (that's
'check a current source / a real accountant', not a formula).
"""
from __future__ import annotations


# ---- formatting: label, value, type ----
def _fmt(value, kind: str) -> str:
    if kind == "money":
        return f"${value:,.2f}"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ratio":
        return f"{value:.2f}x"
    if kind == "months":
        return f"{value:.1f} months"
    if kind == "int":
        return f"{int(round(value)):,}"
    if kind == "raw":
        return str(value)
    return f"{value:,.2f}"


def _need(p: dict, *keys):
    missing = [k for k in keys if p.get(k) is None]
    if missing:
        raise KeyError(", ".join(missing))
    return [float(p[k]) for k in keys]


# ---- the calculators: each returns [(label, value, type), ...] ----

def unit_economics(p: dict):
    """SaaS/subscription unit economics. arpu = avg revenue per customer per
    period; gross_margin as a fraction (or give cogs); churn as a fraction per
    period (or give lifetime_periods)."""
    arpu, cac = _need(p, "arpu", "cac")
    if p.get("gross_margin") is not None:
        gm = float(p["gross_margin"])
    elif p.get("cogs") is not None:
        gm = (arpu - float(p["cogs"])) / arpu
    else:
        raise KeyError("gross_margin (or cogs)")
    if p.get("monthly_churn") is not None:
        churn = float(p["monthly_churn"])
        lifetime = 1.0 / churn if churn else float("inf")
    elif p.get("lifetime_periods") is not None:
        lifetime = float(p["lifetime_periods"])
    else:
        raise KeyError("monthly_churn (or lifetime_periods)")
    gross_per_period = arpu * gm
    ltv = gross_per_period * lifetime
    payback = cac / gross_per_period if gross_per_period else float("inf")
    out = [
        ("gross margin", gm, "pct"),
        ("gross profit / period", gross_per_period, "money"),
        ("customer lifetime", lifetime, "months"),
        ("LTV", ltv, "money"),
        ("LTV : CAC", ltv / cac if cac else float("inf"), "ratio"),
        ("CAC payback", payback, "months"),
    ]
    ratio = ltv / cac if cac else 0
    verdict = ("healthy (>3x is the usual bar)" if ratio >= 3
               else "thin — LTV:CAC under 3x means acquisition barely pays back")
    out.append(("_note", verdict, "text"))
    return out


def runway(p: dict):
    """Months until cash hits zero. If revenue grows, burn shrinks over time."""
    cash, monthly_costs = _need(p, "cash", "monthly_costs")
    rev = float(p.get("monthly_revenue", 0) or 0)
    growth = float(p.get("revenue_growth", 0) or 0)
    net0 = monthly_costs - rev
    if net0 <= 0:
        return [("net burn", net0, "money"),
                ("_note", "already cash-flow positive at current revenue — runway is effectively unlimited.", "text")]
    months = 0.0
    c = cash
    if growth <= 0:
        months = cash / net0
    else:
        r = rev
        while c > 0 and months < 600:
            net = monthly_costs - r
            if net <= 0:
                return [("months to profitability", months, "months"),
                        ("_note", "revenue growth outpaces costs — becomes profitable before running out.", "text")]
            c -= net
            r *= (1 + growth)
            months += 1
    return [
        ("net burn (month 1)", net0, "money"),
        ("runway", months, "months"),
        ("_note", "raise or reach profit before this runs out; under ~6 months is the danger zone.", "text"),
    ]


def break_even(p: dict):
    """Units and revenue needed to cover fixed costs."""
    fixed, price, var = _need(p, "fixed_costs", "price", "variable_cost")
    contribution = price - var
    if contribution <= 0:
        raise ValueError("price must exceed variable cost, or you lose money on every unit.")
    units = fixed / contribution
    return [
        ("contribution / unit", contribution, "money"),
        ("contribution margin", contribution / price, "pct"),
        ("break-even units", units, "int"),
        ("break-even revenue", units * price, "money"),
    ]


def npv(p: dict):
    """Net present value. cashflows is a list; index 0 is today (t=0)."""
    rate = float(p.get("rate", 0) or 0)
    cfs = p.get("cashflows")
    if not cfs:
        raise KeyError("cashflows (a list, e.g. [-1000, 500, 500, 500])")
    val = sum(float(cf) / (1 + rate) ** t for t, cf in enumerate(cfs))
    return [("discount rate", rate, "pct"), ("NPV", val, "money"),
            ("_note", "positive NPV = the cash flows beat your discount rate; negative = they don't.", "text")]


def irr(p: dict):
    """Internal rate of return — the discount rate that makes NPV zero."""
    cfs = [float(x) for x in (p.get("cashflows") or [])]
    if not cfs:
        raise KeyError("cashflows (list, first usually negative — the investment)")

    def _npv(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cfs))

    lo, hi = -0.9999, 10.0
    if _npv(lo) * _npv(hi) > 0:
        raise ValueError("no sign change — IRR isn't defined for these cash flows.")
    for _ in range(200):
        mid = (lo + hi) / 2
        if _npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return [("IRR", (lo + hi) / 2, "pct"),
            ("_note", "compare IRR to your cost of capital; above it, the deal creates value.", "text")]


def loan(p: dict):
    """Amortizing loan: monthly payment, total interest."""
    principal, annual_rate, months = _need(p, "principal", "annual_rate", "months")
    r = annual_rate / 12
    n = int(months)
    if r == 0:
        pay = principal / n
    else:
        pay = principal * r / (1 - (1 + r) ** -n)
    total = pay * n
    return [
        ("monthly payment", pay, "money"),
        ("total paid", total, "money"),
        ("total interest", total - principal, "money"),
    ]


def pricing(p: dict):
    """Price from cost + a target. Give margin (fraction of PRICE) or markup
    (fraction of COST). Margin and markup are NOT the same — the classic trap."""
    cost, = _need(p, "cost")
    if p.get("margin") is not None:
        margin = float(p["margin"])
        if margin >= 1:
            raise ValueError("margin is a fraction of price, must be < 1 (e.g. 0.6 for 60%).")
        price = cost / (1 - margin)
    elif p.get("markup") is not None:
        markup = float(p["markup"])
        price = cost * (1 + markup)
        margin = (price - cost) / price
    else:
        raise KeyError("margin (fraction of price) or markup (fraction of cost)")
    return [
        ("price", price, "money"),
        ("gross margin", margin, "pct"),
        ("markup", (price - cost) / cost, "pct"),
        ("_note", "margin = profit/price; markup = profit/cost. A 60% margin is a 150% markup — don't confuse them.", "text"),
    ]


def dilution(p: dict):
    """Fundraise dilution. pre_money valuation + investment. Optional
    option_pool as a fraction of the post-round cap table."""
    pre, invest = _need(p, "pre_money", "investment")
    pool = float(p.get("option_pool", 0) or 0)
    post = pre + invest
    investor = invest / post
    founders = (1 - pool) - investor
    if founders < 0:
        raise ValueError("investment + option pool exceed 100% — check the numbers.")
    return [
        ("post-money valuation", post, "money"),
        ("investor ownership", investor, "pct"),
        ("option pool", pool, "pct"),
        ("existing holders keep", founders, "pct"),
        ("_note", "simplified single-round view; multiple rounds and pool timing (pre vs post) shift these — model each round.", "text"),
    ]


def growth(p: dict):
    """Project a value forward at a per-period growth rate."""
    start, rate, periods = _need(p, "start", "rate", "periods")
    if (1 + rate) < 0 and float(periods) != int(periods):
        raise ValueError("a growth rate below -100% over a fractional number of "
                         "periods isn't a real value.")
    end = start * (1 + rate) ** periods
    return [
        ("start", start, "num"),
        ("rate / period", rate, "pct"),
        ("periods", periods, "num"),
        ("projected value", end, "num"),
        ("total growth", (end / start - 1) if start else 0, "pct"),
    ]


def cagr(p: dict):
    """Compound annual (or per-period) growth rate from start, end, periods."""
    start, end, periods = _need(p, "start", "end", "periods")
    if start <= 0 or end <= 0 or periods <= 0:
        raise ValueError("start, end, and periods must all be positive — a "
                         "growth rate through zero or negative isn't a real number.")
    rate = (end / start) ** (1 / periods) - 1
    return [("CAGR / period", rate, "pct"),
            ("_note", "the smooth per-period rate that turns start into end over the periods given.", "text")]


_DISPATCH = {
    "unit_economics": unit_economics, "runway": runway, "break_even": break_even,
    "npv": npv, "irr": irr, "loan": loan, "pricing": pricing,
    "dilution": dilution, "growth": growth, "cagr": cagr,
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
    lines = [f"[business_calc · {kind}]"]
    for label, value, vkind in rows:
        if label == "_note":
            lines.append(f"  note: {value}")
        else:
            lines.append(f"  {label}: {_fmt(value, vkind)}")
    return "\n".join(lines)


# ---- known-answer selftests: every calc verified, not asserted ----
_SELFTESTS = [
    ("unit_economics", {"arpu": 100, "gross_margin": 0.6, "cac": 300, "monthly_churn": 0.05},
     {"LTV": 1200.0, "LTV : CAC": 4.0, "CAC payback": 5.0}),
    ("runway", {"cash": 500000, "monthly_costs": 50000}, {"runway": 10.0}),
    ("break_even", {"fixed_costs": 10000, "price": 50, "variable_cost": 30},
     {"break-even units": 500.0, "break-even revenue": 25000.0}),
    ("npv", {"rate": 0.1, "cashflows": [-1000, 500, 500, 500]}, {"NPV": 243.43}),
    ("irr", {"cashflows": [-1000, 500, 500, 500]}, {"IRR": 0.2337}),
    ("loan", {"principal": 100000, "annual_rate": 0.06, "months": 360},
     {"monthly payment": 599.55, "total interest": 115838.19}),
    ("pricing", {"cost": 40, "margin": 0.6}, {"price": 100.0, "markup": 1.5}),
    ("dilution", {"pre_money": 4000000, "investment": 1000000},
     {"post-money valuation": 5000000.0, "investor ownership": 0.2}),
    ("growth", {"start": 100, "rate": 0.1, "periods": 12}, {"projected value": 313.84}),
    ("cagr", {"start": 100, "end": 313.84, "periods": 12}, {"CAGR / period": 0.10}),
]


def selftest(tol: float = 0.01) -> str:
    """Run every calc against a hand-verified case. Returns a pass/fail report."""
    out = []
    ok_all = True
    for kind, params, expect in _SELFTESTS:
        rows = _DISPATCH[kind](params)
        got = {label: value for label, value, _ in rows if label != "_note"}
        bad = []
        for k, want in expect.items():
            have = got.get(k)
            if have is None or abs(have - want) > max(abs(want) * tol, 1e-6):
                bad.append(f"{k}: got {have}, want {want}")
        ok_all = ok_all and not bad
        out.append(f"  {'PASS' if not bad else 'FAIL'}  {kind}" + (f"  [{'; '.join(bad)}]" if bad else ""))
    return ("ALL PASS" if ok_all else "SOME FAILED") + "\n" + "\n".join(out)
