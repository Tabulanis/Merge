"""The Deduction Pad — translation by ELIMINATION, not by reading minds.

You can't observe what an animal call MEANS. But you can play Clue: rule out
what it can't mean, one card at a time, and whatever's left standing is the
suspect. This never claims 'call X means danger'. It says 'for call X we ruled
out food, greeting, mating (they fire in contexts these don't fit); the alarm
family is still on the board; 70% of the deck is face-up.'

The pieces:
  * CALLS come from study_calls (the discrete signals — the cards).
  * OBSERVATIONS are the crime scenes: for each time a call fired, what was
    true in the world — an ethologist's log of context CUES (threat present?
    food present? did the animal flee after? juvenile calling?).
  * MEANINGS is a polymath library of candidate glosses, each tied to the cues
    it PREDICTS. The wider the net of guesses, the finer the elimination.

The engine cross-references: a meaning is RULED OUT for a call when the call
fires largely WITHOUT the context that meaning requires, and SURVIVES when the
call's context is both consistent with it AND more specific to it than chance.
Every elimination is kept on the pad — the negative space is the progress.

Honest to the bone: it corners meaning, it never decodes it. A surviving
hypothesis is a lead to field-test, not a translation.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RENDERS = Path.home() / "forge" / "datasets" / "doolittle"
BG = (13, 16, 23); INK = (230, 236, 245); SOFT = (150, 162, 178)
CYAN = (59, 214, 228); GOLD = (245, 174, 61); GREEN = (111, 208, 140); RED = (200, 90, 74)

# The polymath meaning library — a BLOB, not a word-list. Animals don't talk
# in single words; they talk in whole MEANINGS. So each candidate is a PHRASE,
# and we throw the whole rough mass on the pad first — every angle a polymath
# would try (predator-type, direction, urgency, who it's aimed at, mood,
# resource, coordination, identity) — including WILD guesses and SECONDARY /
# compound ones. Elimination is the chisel: it knocks away the phrases the call
# can't carry until a tight shape is left, and only then does the blob collapse
# toward a word. A wild guess getting crossed off is information; a wild one
# surviving is a lead. `lens` = which angle it came from (so survivors cluster
# into a shape). `kind`: primary / secondary (compound/derived) / wild (a long
# shot we keep BECAUSE ruling it out is progress). `cues` = context that should
# tend to hold; `absent` = context that should tend to be MISSING.
MEANINGS = [
    # --- THREAT, sculpted by what & where (prairie dogs really do resolve this)
    {"id": "air_pred",  "lens": "threat", "kind": "primary",
     "label": "\"raptor overhead — dive for cover NOW\"",
     "cues": ["threat", "predator_above", "then_flee_or_freeze"], "absent": ["calm"]},
    {"id": "ground_pred", "lens": "threat", "kind": "primary",
     "label": "\"ground predator close — get up / bolt\"",
     "cues": ["threat", "predator_ground", "then_flee_or_freeze"], "absent": ["calm"]},
    {"id": "human",     "lens": "threat", "kind": "primary",
     "label": "\"a human is coming\"",
     "cues": ["threat", "human_present"], "absent": ["calm"]},
    {"id": "urgent",    "lens": "threat", "kind": "primary",
     "label": "\"danger RIGHT NOW — everyone scatter\"",
     "cues": ["threat", "high_intensity", "then_flee_or_freeze"], "absent": ["calm"]},
    {"id": "unease",    "lens": "threat", "kind": "secondary",
     "label": "\"something's off — stay sharp\" (alert, not flight)",
     "cues": ["novel_object"], "absent": ["then_flee_or_freeze", "calm"]},
    {"id": "warn_young", "lens": "threat", "kind": "secondary",
     "label": "\"kids, hide — danger\"",
     "cues": ["juvenile", "threat", "then_flee_or_freeze"], "absent": ["calm"]},
    # --- FOOD & RESOURCE
    {"id": "food_here", "lens": "resource", "kind": "primary",
     "label": "\"food here — come eat\"",
     "cues": ["food", "then_approach"], "absent": ["threat"]},
    {"id": "food_rich", "lens": "resource", "kind": "primary",
     "label": "\"a LOT of food — everybody come\"",
     "cues": ["food", "resource_rich", "then_group_move"], "absent": ["threat"]},
    {"id": "good_spot", "lens": "resource", "kind": "wild",
     "label": "\"good place here — settle\"",
     "cues": ["resource_rich", "calm"], "absent": ["threat"]},
    # --- CONTACT & COORDINATION
    {"id": "where_you", "lens": "coord", "kind": "primary",
     "label": "\"where are you?\"",
     "cues": ["conspecific_far", "then_regroup"], "absent": []},
    {"id": "here_me",   "lens": "coord", "kind": "primary",
     "label": "\"I'm over here\"",
     "cues": ["conspecific_far", "movement"], "absent": ["threat"]},
    {"id": "regroup",   "lens": "coord", "kind": "primary",
     "label": "\"regroup on me\"",
     "cues": ["conspecific_far", "then_regroup", "movement"], "absent": []},
    {"id": "move_out",  "lens": "coord", "kind": "primary",
     "label": "\"time to move — follow\"",
     "cues": ["then_group_move", "movement"], "absent": ["threat"]},
    # --- SOCIAL
    {"id": "greet",     "lens": "social", "kind": "primary",
     "label": "\"hello, friend\"",
     "cues": ["conspecific_near", "calm"], "absent": ["threat", "conflict"]},
    {"id": "court",     "lens": "social", "kind": "primary",
     "label": "\"courting — come closer\"",
     "cues": ["breeding", "opposite_sex_near"], "absent": ["threat"]},
    {"id": "back_off",  "lens": "social", "kind": "primary",
     "label": "\"back off — I'm boss here\"",
     "cues": ["conflict", "competitor"], "absent": ["calm"]},
    {"id": "my_turf",   "lens": "social", "kind": "primary",
     "label": "\"this is my turf — keep out\"",
     "cues": ["intruder", "boundary"], "absent": ["calm"]},
    {"id": "submit",    "lens": "social", "kind": "secondary",
     "label": "\"you win — don't hurt me\"",
     "cues": ["conflict", "isolation_or_injury"], "absent": ["competitor"]},
    # --- PARENTAL
    {"id": "feed_me",   "lens": "parental", "kind": "primary",
     "label": "\"feed me\" (begging)",
     "cues": ["juvenile", "parent_near", "food"], "absent": ["threat"]},
    {"id": "come_back",  "lens": "parental", "kind": "secondary",
     "label": "\"pup, come back to me\"",
     "cues": ["juvenile", "then_regroup"], "absent": ["threat"]},
    # --- AFFECT (mood, not referent — wilder)
    {"id": "hurt",      "lens": "affect", "kind": "primary",
     "label": "\"I'm hurt / trapped\"",
     "cues": ["isolation_or_injury"], "absent": ["calm"]},
    {"id": "excited",   "lens": "affect", "kind": "wild",
     "label": "\"so excited!\" (arousal, no referent)",
     "cues": ["high_intensity", "calm"], "absent": ["threat"]},
    {"id": "content",   "lens": "affect", "kind": "wild",
     "label": "\"all's well\" (contentment hum)",
     "cues": ["calm", "conspecific_near"], "absent": ["threat", "conflict"]},
    {"id": "play",      "lens": "affect", "kind": "primary",
     "label": "\"let's play\"",
     "cues": ["juvenile", "calm"], "absent": ["threat", "conflict"]},
    # --- IDENTITY & the deliberately-probably-wrong (kept BECAUSE crossing them
    #     off is progress; if one survives unexpectedly, that's the real find)
    {"id": "self_name", "lens": "identity", "kind": "wild",
     "label": "\"it's me\" (signature / name call)",
     "cues": ["repeated_bout"], "absent": ["threat", "food"]},
    {"id": "status",    "lens": "identity", "kind": "wild",
     "label": "\"I'm here and I'm strong\" (status broadcast)",
     "cues": ["repeated_bout", "competitor"], "absent": ["calm"]},
    {"id": "babble",    "lens": "identity", "kind": "wild",
     "label": "\"just noise / practice\" (no referent at all)",
     "cues": ["juvenile", "repeated_bout"], "absent": ["threat", "food", "conspecific_far"]},
]
RULE_OUT = 0.30      # support below this -> the call fires mostly without the context -> ruled out
STRONG = 0.60        # support above this AND specific -> a live lead


def _support(obs_for_call, meaning):
    """Mean fraction of the meaning's required cues present when this call
    fired, minus a penalty for its should-be-absent cues showing up."""
    if not obs_for_call:
        return 0.0
    req = meaning["cues"]
    presence = []
    for o in obs_for_call:
        cues = o.get("cues", {})
        pos = np.mean([1.0 if cues.get(c) else 0.0 for c in req]) if req else 0.0
        pen = np.mean([1.0 if cues.get(c) else 0.0 for c in meaning["absent"]]) if meaning["absent"] else 0.0
        presence.append(max(0.0, pos - 0.5 * pen))
    return float(np.mean(presence))


def _base_rate(all_obs, meaning):
    """How often the meaning's cues hold across ALL calls — the chance floor,
    so a call is only credited when it's MORE than baseline-specific."""
    req = meaning["cues"]
    if not all_obs or not req:
        return 0.0
    return float(np.mean([np.mean([1.0 if o.get("cues", {}).get(c) else 0.0 for c in req])
                          for o in all_obs]))


# ---- the prior question: is this call even a SIGNAL? --------------------
# Before you ask what a call MEANS you have to earn the assumption that it's
# communication at all, not a meaningless byproduct (a sneeze). The honest test
# from a context log: does knowing THIS call fired reduce your uncertainty about
# the world? A real signal reliably marks a specific context (low uncertainty);
# noise leaves the context a coin-flip. That's mutual information — and we hold
# it to a shuffle null, so a lucky-looking fit can't pass for a signal. The
# score becomes the CEILING on how hard the pad may chisel: no signal, no
# meaning. (A real study_calls result can override this from the audio itself.)
def _consistency(obs_for_call, cues, thresh=0.15):
    """How REPEATABLE is the context this call fires in, over the cues it
    actually engages (present in >=thresh of its sightings)? 1.0 = fires in a
    fixed context every time (deterministic); 0 = every cue a coin-flip. Only
    counts cues the call uses, so it's never credited for 'reliably lacking'
    something it simply never logged (missing != confirmed-absent)."""
    if not obs_for_call:
        return 0.0
    rates = {c: float(np.mean([1.0 if o.get("cues", {}).get(c) else 0.0 for o in obs_for_call]))
             for c in cues}
    active = [c for c in cues if rates[c] >= thresh]
    if not active:
        return 0.0
    return float(np.mean([abs(2 * rates[c] - 1) for c in active]))


def _signal_score(call, by_call, all_obs, cues, rng, n_null=200):
    """0..1: fraction of shuffle-nulls this call's context-consistency beats. A
    real signal fires in a repeatable context (beats a random draw of the same
    size from the pooled observations); noise doesn't. None = uncomputable (only
    one call — no background to tell signal from noise off a single log)."""
    if len(by_call) < 2 or len(all_obs) < 6:
        return None
    observed = _consistency(by_call[call], cues)
    n, L = len(by_call[call]), len(all_obs)
    beat = 0
    for _ in range(n_null):
        idx = rng.choice(L, size=n, replace=False)
        if _consistency([all_obs[i] for i in idx], cues) < observed:
            beat += 1
    return round(beat / float(n_null), 2)


SIGNAL_BAR = 0.90    # must beat 90% of shuffles to count as a confirmed signal


def _clean_obs(observations):
    """Merge hands this model-written JSON, so treat every input as hostile:
    keep only dict entries, coerce an unhashable call to a string, and force a
    non-dict cues into an empty dict. Anything unparseable is dropped, not fatal."""
    if not isinstance(observations, list):
        return []
    clean = []
    for o in observations:
        if not isinstance(o, dict):
            continue
        call = o.get("call")
        if isinstance(call, (dict, list, set)):
            call = str(call)
        cues = o.get("cues")
        if not isinstance(cues, dict):
            cues = {}
        clean.append({"call": call, "cues": cues})
    return clean


def deduce(observations, labels=None, signal_scores=None):
    """Run the pad. observations = list of {"call": <id/label>, "cues": {cue: bool}}.
    Returns per-call: ruled_out, standing (with support + lift), % of deck
    eliminated, and a signal_score (is it even communication?). signal_scores =
    optional {call: 0..1} to override the log-based test with a real study_calls
    result. Empty structure if there's nothing to work with."""
    observations = _clean_obs(observations)
    if not observations:
        return {"calls": [], "note": "no observations — log some crime scenes first."}
    by_call = {}
    for o in observations:
        by_call.setdefault(o.get("call"), []).append(o)
    # lift (how call-SPECIFIC a cue-set is) only means something when there are
    # other calls to contrast against. With a single call in the log there's no
    # contrast population, so specificity is uncomputable — fall back to support
    # + the strict conclusiveness gate instead of wiping the whole board.
    multi = len(by_call) >= 2
    cues_seen = sorted({k for o in observations for k in (o.get("cues") or {})})
    rng = np.random.RandomState(0)   # fixed seed -> the pad is reproducible
    ext_signal = signal_scores if isinstance(signal_scores, dict) else {}
    pad = {"calls": []}
    for call, obs in sorted(by_call.items(), key=lambda x: str(x[0])):
        ruled_out, standing = [], []
        for m in MEANINGS:
            sup = _support(obs, m)
            base = _base_rate(observations, m)
            lift = sup / base if base > 0.02 else (sup / 0.02)
            # a meaning resting on ONE common cue is too easy — weight by how
            # many cues it stakes its claim on (a 2-3 cue hypothesis that holds
            # is far more telling than a 1-cue coincidence)
            depth = min(1.0, 0.55 + 0.25 * len(m["cues"]))
            sup *= depth
            too_common = multi and lift < 1.35   # only judge specificity with a contrast pop.
            if sup < RULE_OUT or too_common:
                ruled_out.append({"id": m["id"], "label": m["label"],
                                  "support": round(sup, 2),
                                  "why": ("fires mostly without its context" if sup < RULE_OUT
                                          else "no more specific than chance")})
            else:
                standing.append({"id": m["id"], "label": m["label"],
                                 "lens": m.get("lens", "?"), "kind": m.get("kind", "primary"),
                                 "support": round(sup, 2), "lift": round(lift, 1)})
        standing.sort(key=lambda x: -x["support"])
        elim_pct = round(100 * len(ruled_out) / len(MEANINGS))
        top_sup = standing[0]["support"] if standing else 0.0
        runner = standing[1]["support"] if len(standing) > 1 else 0.0
        # the prior question first: is this call even a signal? A real
        # study_calls result (ext_signal) wins; else test the log itself.
        sig = ext_signal.get(call)
        if sig is not None:
            try:
                sig = min(1.0, max(0.0, float(sig)))   # clamp a hand-fed override
            except (TypeError, ValueError):
                sig = None
        if sig is None:
            sig = _signal_score(call, by_call, observations, cues_seen, rng)
        # cornered only if the lead is strong AND clearly ahead — else the cues
        # just don't discriminate (honest 'inconclusive', not a lucky pin)
        conclusive = bool(standing) and top_sup >= STRONG and (top_sup - runner) >= 0.12
        # ...and NO meaning may collapse unless the call clears the signal bar.
        # If signalhood is uncomputable (single call, no background) the lead is
        # allowed but flagged provisional; if it's computed and weak, the pad
        # stays a blob no matter how tidy the cues looked — no signal, no meaning.
        signal_ok = (sig is None) or (sig >= SIGNAL_BAR)
        provisional = conclusive and sig is None
        conclusive = conclusive and signal_ok
        pad["calls"].append({"call": call, "n_obs": len(obs),
                             "eliminated_pct": elim_pct, "conclusive": conclusive,
                             "provisional": provisional, "signal_score": sig,
                             "ruled_out": ruled_out, "standing": standing})
    return pad


# ---- render the pad: calls × meanings grid (Clue sheet) -----------------
def _render(pad, title):
    calls = pad["calls"]
    if not calls:
        return None
    cols = MEANINGS
    cw, rh = 128, 40
    W = 210 + cw * 1  # legend width handled below; use a compact matrix
    W = 230 + 44 * len(cols)
    H = 90 + rh * len(calls) + 40
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((14, 12), f"\U0001f5d2 the deduction pad — {title}", fill=CYAN)
    d.text((14, 30), "green = still standing (a lead) · red = ruled out · brighter = stronger",
           fill=SOFT)
    x0, y0 = 200, 60
    # column headers (meaning ids, rotated-ish: just short)
    for j, m in enumerate(cols):
        d.text((x0 + j * 44 + 4, y0 - 14), m["id"][:5], fill=SOFT)
    for i, c in enumerate(calls):
        y = y0 + i * rh
        d.text((14, y + 10), f"call {str(c['call'])[:14]}", fill=INK)
        d.text((150, y + 10), f"{c['eliminated_pct']}%", fill=GOLD)
        standing_ids = {s["id"]: s["support"] for s in c["standing"]}
        for j, m in enumerate(cols):
            x = x0 + j * 44
            if m["id"] in standing_ids:
                sup = standing_ids[m["id"]]
                g = int(80 + 175 * min(1, sup))
                d.rectangle([x, y + 4, x + 40, y + rh - 4], fill=(30, g, 60))
                d.text((x + 14, y + 12), "✓", fill=(230, 255, 235))
            else:
                d.rectangle([x, y + 4, x + 40, y + rh - 4], fill=(40, 22, 20), outline=(70, 40, 36))
                d.text((x + 15, y + 12), "✗", fill=(150, 90, 84))
    RENDERS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:40]
    out = RENDERS / f"pad_{safe}.png"
    img.save(out)
    return out


def deduce_meaning(observations, title: str = "case", signal_scores=None) -> str:
    """Play Clue with animal calls: given a log of OBSERVATIONS (each = a call
    plus the context cues true when it fired), rule out the meanings it CAN'T
    carry and report what's left standing. `observations` is a list of
    {"call": <name>, "cues": {"threat": true, "then_flee_or_freeze": true, ...}}.
    Known cues (log the ones you saw; more/finer cues chisel a sharper shape):
    threat, predator_above, predator_ground, human_present, high_intensity,
    novel_object, then_flee_or_freeze, food, resource_rich, then_approach,
    then_group_move, conspecific_near, conspecific_far, then_regroup, movement,
    calm, breeding, opposite_sex_near, intruder, boundary, isolation_or_injury,
    conflict, competitor, juvenile, parent_near, repeated_bout. It first asks
    the PRIOR question — is this call even a signal, or noise? (mutual info vs a
    shuffle null); a call that doesn't clear the bar stays a blob no matter how
    tidy its cues. Optional signal_scores = {call: 0..1} to feed a real
    study_calls result in as that answer. Renders the pad. Starts as a BLOB of
    candidate PHRASES and chisels down by elimination. NEVER claims a call's
    meaning — a surviving lead is for field-testing, not a translation."""
    if isinstance(observations, str):
        import json
        try:
            observations = json.loads(observations)
        except Exception as e:
            return f"observations wasn't valid JSON: {e}"
    observations = _clean_obs(observations)   # count the header off the CLEAN log
    title = str(title)[:80]                    # keep the header/filename sane
    pad = deduce(observations, signal_scores=signal_scores)
    if not pad["calls"]:
        return pad.get("note", "nothing to deduce.")
    out = _render(pad, title)
    lines = [f"THE DEDUCTION PAD — {title} ({len(observations)} observations):"]
    for c in pad["calls"]:
        sig = c.get("signal_score")
        if sig is None:
            sigline = "signal check: uncomputable (only one call — no background to contrast; " \
                      "lead below is PROVISIONAL until you log other calls or a study_calls score)"
        elif sig >= SIGNAL_BAR:
            sigline = f"signal check: {sig} — reliably carries information; reads as a real signal ✓"
        else:
            sigline = f"signal check: {sig} — barely departs from noise; can't confirm it's even " \
                      "communication, so meaning stays a blob by rule"
        lines.append(f"\n● call '{c['call']}' ({c['n_obs']} sightings) — "
                     f"{c['eliminated_pct']}% of the meaning-deck ruled out")
        lines.append(f"    {sigline}")
        if c["standing"]:
            lead = c["standing"][0]
            # the BLOB shape first: which angles survived, grouped by lens, so
            # you see the rough form of the meaning before it collapses to words
            shape = {}
            for s in c["standing"]:
                shape.setdefault(s["lens"], []).append(s)
            lines.append("    blob shape — what's left, by angle:")
            for lens, group in sorted(shape.items(), key=lambda kv: -max(s["support"] for s in kv[1])):
                group.sort(key=lambda s: -s["support"])
                tag = "".join(" ⚡" if g["kind"] == "wild" else "" for g in group[:1])
                lines.append(f"      • {lens}{tag}: " + "; ".join(
                    f"{g['label']} ({g['support']})" for g in group[:3]))
            # would this have collapsed on strength+separation, only for the
            # signal gate to hold it back? (strong lead, clearly ahead, weak signal)
            _sep = lead["support"] - (c["standing"][1]["support"] if len(c["standing"]) > 1 else 0.0)
            gated = (c.get("signal_score") is not None
                     and c["signal_score"] < SIGNAL_BAR
                     and lead["support"] >= STRONG and _sep >= 0.12)
            if c.get("conclusive"):
                lines.append(f"    → CHISELED DOWN to: {lead['label']} — strong, clearly ahead, "
                             "AND the call clears the signal bar. A lead worth field-testing.")
            elif c.get("provisional"):
                lines.append(f"    → PROVISIONAL lead: {lead['label']} — strong and clearly ahead, "
                             "but signalhood is unconfirmed (one call, no background). Treat as a "
                             "hunch, not a finding, until you can contrast it against other calls.")
            elif gated:
                lines.append(f"    → HELD as a blob: {lead['label']} leads the cues, but the call "
                             "didn't clear the signal bar — no signal, no meaning. Confirm it's "
                             "communication first (more calls to contrast, or a study_calls score).")
            else:
                lines.append("    → STILL A BLOB — nothing is strong-and-separated enough to "
                             "collapse to one meaning yet. The survivors above are the shape so "
                             "far; log more sightings (or finer cues) to keep chiseling.")
        else:
            lines.append("    nothing survived — the cues logged don't fit any candidate, "
                         "or the observations are too thin/noisy. Log more, or add candidates.")
        if c["ruled_out"]:
            lines.append(f"    ruled out: " + ", ".join(r["label"] for r in c["ruled_out"][:5])
                         + (" …" if len(c["ruled_out"]) > 5 else ""))
    if out:
        lines.append(f"\nRendered the pad to {out} — look_at_image it (the Clue sheet: "
                     "calls × meanings, green standing / red ruled out).")
    lines.append("HARD LIMIT: this ELIMINATES, it does not translate. A standing meaning is a "
                 "lead to test in the field, never a claim about what the animal said.")
    return "\n".join(lines)
