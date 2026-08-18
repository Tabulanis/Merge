"""Structured thinking scaffolds — the judgment half of the business toolkit.

Frameworks are lenses she FILLS IN, not answers. They exist so analysis is a
rigorous pass instead of vibes: the opportunity canvas structures any money
idea, and the skeptic checklist tries to KILL it before it's trusted (that's
where the honesty lives). Returned as a template + guiding questions; she works
through it out loud and reports what she found — including 'this fails at step N'.
"""
from __future__ import annotations

_FRAMEWORKS = {
    "opportunity": """OPPORTUNITY CANVAS — structure any money-making idea before believing it.
Work through every line; a blank or hand-wave is itself a finding.

  1. THE EDGE — in one sentence, what do you know/have/can-do that others don't?
     (If you can't name it crisply, there's probably no edge — just activity.)
  2. WHY IT PERSISTS — why hasn't this been arbitraged away already? Who's on the
     other side of the trade, and why are they willing to lose to you?
     (Efficient-market prior: assume it's gone UNLESS you can answer this.)
  3. EXPECTED VALUE — the honest number (use markets_calc/business_calc). After
     costs, fees, taxes, and your time.
  4. VARIANCE & RISK OF RUIN — what's the downside, how often, and can it wipe
     you out? A +EV bet that can bankrupt you is still a trap.
  5. CAPITAL & TIME — what does it cost to enter, and what's your $/hour really?
  6. SCALABILITY — does it grow, or is it a one-off / capacity-capped?
  7. WHAT KILLS IT — the top 3 things that make this stop working. Be specific.
  VERDICT — pursue / watch / kill, and the single fact the whole thing hinges on.""",

    "skeptic": """SKEPTIC / RED-TEAM — try to destroy the idea. If it survives honestly, THEN it's interesting.
Default stance: this pattern is noise until proven otherwise.

  □ SAMPLE SIZE — how many independent observations? A 'pattern' from a handful
    of instances is a story, not evidence.
  □ DATA-MINING / OVERFITTING — did you go looking until something 'worked'? Test
    enough rules and some will fit the past by luck and fail the future.
  □ SURVIVORSHIP — are you only seeing the winners? (The losers went quiet.)
  □ ALREADY PRICED IN — if it's public and obvious, faster/bigger players already ate it.
  □ TRANSACTION COSTS — spreads, fees, slippage, taxes. Many 'edges' are real and
    smaller than the cost of capturing them.
  □ REGIME — did this only hold in one environment (low rates, a bull run) that's over?
  □ SELECTION OF THE MESSENGER — who's telling you this, and what do they gain?
  □ BASE RATE — how often do things in this category actually work? Start there, not at the anecdote.
  RESULT — which check does it FAIL first? If none, say what evidence would change your mind.""",

    "business_model_canvas": """BUSINESS MODEL CANVAS — the whole business on one page.
  • Value proposition — the specific pain you kill / gain you create.
  • Customer segments — who exactly, and who is NOT the customer.
  • Channels — how they find, buy, and get the product.
  • Customer relationships — self-serve, high-touch, community?
  • Revenue streams — how money comes in (and the unit economics: use business_calc).
  • Cost structure — the big line items; fixed vs variable.
  • Key resources / activities — what you must have and do.
  • Key partners — who you rely on (and the dependency risk).
  TEST: could a stranger fund it from this page? Where's the weakest cell?""",

    "swot": """SWOT — honest inventory, not a feel-good list.
  • Strengths — real, defensible, and RELEVANT to winning here (not generic).
  • Weaknesses — the ones a competitor would attack first. Name them plainly.
  • Opportunities — external and time-bound; why now?
  • Threats — what could kill or shrink this from outside.
  Then the move most people skip: match them — use a Strength on an Opportunity,
  and pick the ONE Weakness×Threat pair that's most dangerous.""",

    "positioning": """POSITIONING — own a clear spot in the customer's head.
  • For [target] who [need/context],
  • [product] is a [category]
  • that [single most important benefit],
  • unlike [main alternative],
  • because [reason to believe — the proof].
  TEST: is the benefit specific and true? Could a competitor say the exact same
  sentence? If yes, you haven't positioned — you've described.""",

    "lean_validation": """LEAN VALIDATION — cheapest test that could prove you WRONG, fast.
  1. Riskiest assumption — the belief that, if false, sinks the whole thing.
  2. Falsifiable hypothesis — 'if X, then [specific measurable Y] within [time].'
  3. Smallest test — the cheapest experiment that gives a real signal (a landing
     page, 10 customer calls, a pre-order, a manual concierge version).
  4. Kill/continue line — the number that decides, set BEFORE you run it.
  Rule: you're hunting for disconfirmation, not applause. Friends saying 'cool'
  is not validation; a stranger paying (or refusing) is.""",
}

NAMES = ", ".join(_FRAMEWORKS)


def get(name: str) -> str:
    key = (name or "").strip().lower()
    fw = _FRAMEWORKS.get(key)
    if not fw:
        return f"Unknown framework '{name}'. Available: {NAMES}."
    return fw + "\n\n[Fill this in with the real specifics and report what you find — including where it fails.]"
