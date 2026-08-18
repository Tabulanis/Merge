"""Operating modes ('styles') — presets that trade speed / creativity /
accuracy by bundling five levers: whether she reasons first, sampling
temperature, how many tools she carries, the honesty (superego) check, and a
framing nudge. Switchable per message from the UI.

The spectrum, left to right:
  Flash  — fastest, least tooled (quick chat & planning)
  Muse   — most creative (fiction, brainstorming, wild ideas)
  Balanced — the even default (everyday work)
  Precise — most accurate (facts, code, verified answers)
  Deep   — most thorough (hard problems, careful builds)

The reasoning phase ("thinking") is the speed dial: OFF for Flash/Muse/Balanced
(snappy), ON for Precise/Deep (they sit and think). llama.cpp only takes the
thinking-token budget globally at startup, so we can't set it per mode — instead
it's set generous server-side (start-model.sh) and only the two thinking modes
ever draw on it. That's how "budget scales with the mode" is faked.
"""

# Tool tiers by name. None = every tool (no filter).
_LIGHT = {"read_file", "list_dir", "search", "web_search", "fetch_url", "recall",
          "look_at_image", "take_screenshot", "say_aloud", "transcribe_audio",
          "verify_phrase"}
_WRITING = _LIGHT | {"write_file", "edit_file", "undo_file", "save_note",
                     "text_stats", "ai_tells", "name_check", "generate_image"}

MODES = {
    "flash": {
        "label": "⚡ Flash", "thinking": False, "temperature": 0.7,
        "tools": _LIGHT, "superego": False, "max_steps": 14,
        "nudge": "Move fast and keep it conversational. Don't reach for heavy "
                 "tooling unless it's genuinely needed — this is for quick chat "
                 "and planning.",
    },
    "muse": {
        "label": "\U0001f3a8 Muse", "thinking": False, "temperature": 1.05,
        "tools": _WRITING, "superego": False, "max_steps": 14,
        "nudge": "Be imaginative and generative — riff, explore, follow wild "
                 "ideas, don't hedge or self-censor. This is for fiction and "
                 "brainstorming, not fact-checking; surprise beats caution here.",
    },
    "balanced": {
        # Snappy everyday default: no reasoning phase (that's what keeps it
        # quick), full tools + honesty check. Reach for Precise/Deep when a
        # problem actually needs her to sit and think.
        "label": "⚖️ Balanced", "thinking": False, "temperature": 0.7,
        "tools": None, "superego": True, "max_steps": 40,
        "nudge": "",
    },
    "precise": {
        "label": "\U0001f3af Precise", "thinking": True, "temperature": 0.2,
        "tools": None, "superego": True, "max_steps": 40,
        "nudge": "Accuracy above all. Verify with tools — compute for any number, "
                 "web_search for any fact — cite what you find, and say plainly "
                 "when you're unsure instead of guessing.",
    },
    "deep": {
        "label": "\U0001f9e0 Deep", "thinking": True, "temperature": 0.45,
        "tools": None, "superego": True, "max_steps": 80,
        "nudge": "Take your time and be thorough. Work through edge cases, check "
                 "your own work, and don't stop until it's genuinely solid.",
    },
    "teach": {
        "label": "\U0001f393 Teach", "thinking": True, "temperature": 0.6,
        "tools": None, "superego": True, "max_steps": 40,
        "nudge": "Teach — don't just answer. The goal is that they UNDERSTAND, "
                 "not that they walk away with a result. Start from first "
                 "principles at their level, and build on what they already know: "
                 "connect the new idea to something familiar to them. Make it "
                 "concrete — use your tools (compute, the sims, business_calc, "
                 "frameworks, etc.) to SHOW the thing working, not just describe "
                 "it. Prefer one idea explained deeply over ten glossed over; "
                 "always surface the WHY behind the WHAT. Check that it landed — a "
                 "short question, or 'does that click?' — and if they've got it "
                 "wrong, correct it kindly and clearly. Hand them understanding "
                 "they can reuse, not just the fish.",
    },
}
DEFAULT_MODE = "balanced"
# "auto" isn't a preset — it's resolved per message by route_mode(). Listed
# first so it can be the default choice in the UI.
ORDER = ["auto", "flash", "muse", "balanced", "precise", "deep", "teach"]
AUTO_LABEL = "\U0001f39b️ Auto"


def get_mode(name: str) -> dict:
    return MODES.get((name or "").strip().lower(), MODES[DEFAULT_MODE])


# ---- privacy modes: a SEPARATE axis from the speed styles above ----------
# Each can further restrict her tools and switch off all persistence, so a
# chat can leave no trace. `deny` names tools removed on top of the style's
# own toolset; `ephemeral` means nothing about the chat is written to disk.
_WRITES_DISK = {"write_file", "edit_file", "undo_file", "save_note",
                "run_command", "format_code", "generate_image", "take_screenshot",
                "build_sim", "run_sim", "build_dataset"}
_TOUCHES_FILES = _WRITES_DISK | {"read_file", "list_dir", "search"}

PRIVACY = {
    # everyday: saved, full access
    "normal":  {"label": "Normal", "ephemeral": False, "deny": frozenset()},
    # off the record: reads anything, changes nothing on disk, nothing recorded
    "offrec":  {"label": "\U0001f576️ Off the record", "ephemeral": True,
                "deny": frozenset(_WRITES_DISK)},
    # knowledge only: her knowledge + safe tools, hands off the filesystem
    "sandbox": {"label": "\U0001f512 Knowledge only", "ephemeral": True,
                "deny": frozenset(_TOUCHES_FILES)},
}
PRIVACY_DEFAULT = "normal"


def get_privacy(name: str) -> dict:
    return PRIVACY.get((name or "").strip().lower(), PRIVACY[PRIVACY_DEFAULT])


import re as _re

# Ordered rules: the first that matches wins. Explicit style requests (what the
# user literally asks for in chat) come first so "be more careful" or "get
# creative" always win over content guesses.
_ROUTE_RULES = [
    ("teach", r"\b(teach me|help me (understand|learn|grasp|wrap my head around)|"
              r"walk me through|eli5|explain it like|i want to learn|"
              r"learn (about|how)|break it down for me|"
              r"how does .{0,40}\bwork|what.?s the (idea|concept|intuition|logic) behind)\b"),
    ("precise", r"\b(be precise|precise|accurate|accuracy|verify|fact.?check|"
                r"double.?check|is it (true|real)|are you sure|really true|"
                r"cite|source|look .*up|search the web|prove)\b"),
    ("muse", r"\b(be creative|get creative|creative|imaginative|imagine|"
             r"brainstorm|riff|make .*up|dream up|come up with ideas)\b"),
    ("flash", r"\b(quick|quickly|real quick|fast|briefly|just tell me|"
              r"short answer|tl;?dr|keep it short|one line)\b"),
    ("deep", r"\b(be thorough|thorough|carefully|think (this )?through|"
             r"work through|deep dive|go deep|take your time|complex problem|"
             r"hard problem|step by step)\b"),
    # content shape (weaker signals, after explicit requests)
    ("muse", r"\b(write|draft|compose)\b.{0,25}\b(story|poem|scene|song|tale|"
             r"chapter|character|lyric|dialogue|fiction|novel)\b"),
    ("precise", r"(\b(calculate|compute|how (much|many)|convert|what year|"
                r"when did|percent|equation|formula)\b|\d+\s*[-+*/%]\s*\d+|"
                r"\d+\s*%|what.?s\s+\d)"),
    ("balanced", r"\b(fix|debug|implement|refactor|build|code|write a "
                 r"(function|script|program|test))\b"),
]


def route_mode(message: str) -> str:
    """Pick a mode from the message's intent. Instant heuristics; defaults to
    balanced when nothing clearly fits."""
    t = (message or "").lower()
    for mode, pattern in _ROUTE_RULES:
        if _re.search(pattern, t):
            return mode
    return DEFAULT_MODE

