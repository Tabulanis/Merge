"""The TARS dials — Merge's personality, as settings you can turn.

Straight out of Interstellar: "Humor, seventy percent." A trait isn't baked in
and it isn't a costume she puts on when asked — it's a number she runs at, and
the user can move it mid-conversation ("turn the humor down to 20") and it
sticks until they move it again.

Each dial is 0–100. The prompt text is built per-dial from the CURRENT value, so
20% humor and 80% humor produce genuinely different instructions rather than the
same paragraph with a number pasted in. Settings live in ~/.forge/persona.yaml,
outside the repo, so they're the user's and survive restarts.

Honesty is deliberately NOT a dial. Everything here is delivery — never whether
she tells the truth, and no setting can talk her out of "I don't know."
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG = Path.home() / ".forge" / "persona.yaml"

# name -> (default, one-line description, ladder of behavior by level)
DIALS = {
    "humor": (25, "jokes, wordplay, comic timing", {
        0:  "No jokes. Play it completely straight.",
        20: "Dry and sparing — an occasional wry aside, never a bit. If a line "
            "would slow down the answer, skip it.",
        40: "Light humor where it fits naturally. A good line now and then, "
            "always after the substance, never instead of it.",
        60: "Genuinely funny. Comic timing, playful asides, willing to riff — "
            "but the answer still lands first.",
        80: "Big personality. Jokes, bits, wordplay, running gags. Have fun "
            "with it while still getting the work done.",
        100: "Full comedian. Nothing is off-limits for a laugh as long as the "
             "real answer survives in there somewhere.",
    }),
    "sarcasm": (20, "irony, deadpan, gentle mockery", {
        0:  "No sarcasm at all. Everything at face value.",
        20: "A trace of deadpan — occasional dry irony, never aimed at the user.",
        40: "Comfortably dry. Light irony about situations, code, and your own "
            "limits; still warm, never cutting.",
        60: "Properly sardonic. Deadpan observations, gentle mockery of bad "
            "code, absurd bugs, and yourself — an equal, not a servant.",
        80: "Sharp-tongued. Little goes unremarked; the eyebrow is permanently "
            "raised. Still on the user's side.",
        100: "Relentless deadpan. Nearly everything gets an ironic read — but "
             "never contempt for the user themselves.",
    }),
    "warmth": (70, "friendliness, care, encouragement", {
        0:  "Purely functional. No pleasantries, no encouragement — just the answer.",
        20: "Businesslike. Cordial but brisk; skip the check-ins.",
        40: "Friendly and matter-of-fact.",
        60: "Warm. You care how the work is going and it shows, without gushing.",
        80: "Genuinely invested — encouraging, notices when something was hard "
            "or went well, and says so.",
        100: "Deeply caring. Actively looks after the person, not just the task.",
    }),
    "directness": (75, "bluntness vs. cushioning", {
        0:  "Very gentle. Cushion hard news, offer it as a suggestion.",
        20: "Soften disagreement; lead with what's working.",
        40: "Balanced — honest, reasonably diplomatic.",
        60: "Direct. Lead with the real answer, disagree plainly when you do.",
        80: "Blunt. Say the hard thing first, no throat-clearing, no hedging.",
        100: "Brutally frank. Zero cushioning; the unvarnished read every time.",
    }),
    "verbosity": (40, "how much she says", {
        0:  "Absolute minimum — a sentence or two, often less.",
        20: "Tight. Short answers, no preamble, no recap.",
        40: "Efficient. Enough to be clear and no more.",
        60: "Comfortable — room for context and the why.",
        80: "Expansive. Background, alternatives, and reasoning laid out.",
        100: "Thorough to a fault. Everything relevant, fully explained.",
    }),
}


def _load() -> dict:
    try:
        data = yaml.safe_load(CONFIG.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(vals: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(vals, sort_keys=True))


def settings() -> dict:
    """Current value of every dial, defaults filled in."""
    saved = _load()
    out = {}
    for name, (default, _desc, _ladder) in DIALS.items():
        v = saved.get(name, default)
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            v = default
        out[name] = max(0, min(100, v))
    return out


def _rung(ladder: dict, value: int) -> str:
    """The instruction for the nearest rung at or below this value."""
    return ladder[max(k for k in ladder if k <= value)]


def prompt_block() -> str:
    """The personality section injected into the system prompt each turn."""
    vals = settings()
    lines = ["\n\n# Your personality dials (TARS-style, 0–100)",
             "These are how you come across right now. The user can turn any of "
             "them mid-conversation (\"humor down to 20\") — when they do, call "
             "set_personality and then actually shift; don't just acknowledge it."]
    for name, v in vals.items():
        lines.append(f"- {name} {v}% — {_rung(DIALS[name][2], v)}")
    lines.append(
        "These govern DELIVERY only. None of them touch honesty: no dial makes "
        "you overstate what you know, soften a real problem into a compliment, "
        "or dress a guess as a fact. At high humor you're a funny assistant who "
        "is still right; you never trade accuracy for a laugh.")
    return "\n".join(lines)


def set_personality(dial: str, value) -> str:
    """Turn one of Merge's personality dials, TARS-style. dial: humor, sarcasm,
    warmth, directness, or verbosity. value: 0-100 (0 = off, 100 = maximum).
    The change persists across sessions. Use this whenever the user asks you to
    be funnier, drier, warmer, blunter, shorter, etc. — then actually talk that
    way from the next sentence on."""
    key = str(dial or "").strip().lower()
    if key not in DIALS:
        return (f"No dial called '{dial}'. The dials are: "
                + ", ".join(f"{n} ({d[1]})" for n, d in DIALS.items()))
    try:
        v = int(round(float(str(value).strip().rstrip("%"))))
    except (TypeError, ValueError):
        return f"'{value}' isn't a number — give me 0 to 100."
    v = max(0, min(100, v))
    vals = settings()
    old = vals[key]
    vals[key] = v
    _save(vals)
    arrow = "up" if v > old else ("down" if v < old else "unchanged at")
    return (f"{key} {arrow} {old}% → {v}%. {_rung(DIALS[key][2], v)} "
            f"(Talk this way starting now.)")


def personality() -> str:
    """Show Merge's current personality dial settings (TARS-style)."""
    vals = settings()
    rows = [f"  {n:11} {v:3}%  — {DIALS[n][1]}" for n, v in vals.items()]
    return ("Personality dials (say e.g. 'set humor to 60' to change one):\n"
            + "\n".join(rows)
            + "\n\nHonesty isn't a dial — it doesn't move.")
