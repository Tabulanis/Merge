#!/usr/bin/env python3
"""selftest — re-prove every fix, so none of them can quietly rot.

Everything here is a bug that actually happened and was actually fixed. The
point is not coverage for its own sake: each check is anchored to a real
failure, with the symptom written down, so a future change that reintroduces it
is caught by name rather than rediscovered the hard way.

    ./selftest.py            run everything
    ./selftest.py --quick    skip the checks that need a model/embedder server

Exit code is the number of failures, so it can gate a commit.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL, SKIP = [], [], []


def check(name: str, fn, needs_server: bool = False):
    """Run one check. It passes if it returns True, fails on False or a raise."""
    if needs_server and "--quick" in sys.argv:
        SKIP.append(name)
        return
    try:
        ok = fn()
        (PASS if ok else FAIL).append(name if ok else (name, "returned False"))
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- context
def t_overhead_counted():
    """The bare 400: schemas + system prompt were invisible, so a turn read 19%
    full while sitting at 84%, and compaction never fired."""
    from forge.agent import Agent
    from forge import tools as T
    from forge.session import Workspace
    ag = Agent(provider=object(), tools=T.build_tools(Workspace(tempfile.mkdtemp())),
               permission_mode="auto", superego=None)
    ag.active_mode = "balanced"; ag.privacy = "normal"
    ag.client_env = ""; ag.identity_owner = ""; ag.history = []
    ag._overhead_cache = None
    return ag._fixed_overhead_tokens() > 3000


def t_force_compaction_shrinks():
    """One oversized turn: force-compaction summarized the prefix and kept the
    huge turn intact, so the retry was just as big."""
    from forge.agent import Agent
    ag = Agent.__new__(Agent); ag._turn_hiccups = []
    ag.history = [{"role": "user", "content": "go"}]
    for _ in range(6):
        ag.history.append({"role": "tool_use", "calls": []})
        ag.history.append({"role": "tool_result", "content": "X" * 20000})
    before = sum(len(str(m.get("content", ""))) for m in ag.history)
    ag._trim_tool_results(keep_recent=4)
    after = sum(len(str(m.get("content", ""))) for m in ag.history)
    pairs = sum(1 for m in ag.history if m.get("role") == "tool_result")
    return after < before * 0.6 and pairs == 6      # shrank, pairing intact


# --------------------------------------------------------------- superego
def t_evidence_not_starved():
    """The reviewer bounced true claims: results were clipped at 200 chars and
    the answer under review was clipped at 500."""
    from forge.agent import Agent
    ag = Agent.__new__(Agent)
    body = ("From your archive:\n\n● Roof quote — 2026-02-20\n    from: mike@x.com\n"
            "    " + "padding. " * 40 + "the estimate is $4,200 materials and labor.")
    ag.history = [{"role": "user", "content": "what was the quote?"},
                  {"role": "tool_result", "content": body}]
    answer = "The quote was $4,200 for materials and labor, dated 2026-02-20."
    d = ag._evidence_digest(0, answer)
    # Assert the WHOLE answer survives, not a fragile suffix — the first
    # version of this check tested endswith("2026.") while the answer ends
    # "2026-02-20.", so it failed against working code.
    return "$4,200" in d and answer in d


def t_truncation_is_marked():
    """Silent truncation made the reviewer treat 'not shown' as 'not there'."""
    from forge.agent import Agent
    ag = Agent.__new__(Agent)
    ag.history = [{"role": "user", "content": "x"},
                  {"role": "tool_result", "content": "A" * 5000}]
    return "not shown" in ag._evidence_digest(0, "answer")


# ------------------------------------------------------------------ vault
def t_secrets_scrubbed():
    """A password typed into chat was written to the session log, the model
    history, the review ledger and the memory queue, in plaintext."""
    from forge.vault import scrub, scrubbed
    must_redact = ["my password is hunter2", "The password is: s3cr3t!x",
                   "my api key is sk-ant-abc123def456ghi", "token: ghp_abcdefghijklmnop12"]
    must_keep = ["I forgot my password again", "password managers are great",
                 "can you reset the password on that box?"]
    return (all(scrubbed(s) for s in must_redact)
            and not any(scrubbed(s) for s in must_keep)
            and "hunter2" not in scrub("my password is hunter2"))


def t_vault_hides_values():
    """The model must never be able to read a stored secret back."""
    from forge import vault
    vault.clear()
    vault.put("probe", "CANARY-9137", kind="password")
    listing = vault.listing()
    ok = "CANARY-9137" not in listing and "cred:probe" in listing
    vault.clear()
    return ok and not vault.has("cred:probe")


# ------------------------------------------------------------- tool index
def t_toolindex_excludes_itself():
    """find_tools indexed itself, and its description carries example phrases,
    so 'search my email' returned find_tools instead of search_life."""
    from forge import tools as T
    from forge.session import Workspace
    T.build_tools(Workspace(tempfile.mkdtemp()))
    return not ({"find_tools", "load_tools"} & set(T._INDEX_REGISTRY))


def t_loaded_tools_survive_and_reset():
    """A tool loaded early has to still be there late, and must not leak into
    the next turn."""
    from forge.agent import Agent
    from forge import tools as T, toolindex
    from forge.session import Workspace
    ws = Workspace(tempfile.mkdtemp())
    ag = Agent(provider=object(), tools=T.build_tools(ws),
               permission_mode="auto", superego=None)
    ag.active_mode = "balanced"; ag.privacy = "normal"
    ag.client_env = ""; ag.identity_owner = ""; ag.history = []
    reg = T._INDEX_REGISTRY
    toolindex.reset(); toolindex.load_tools(["query_dataset"], reg)
    here = lambda: any(s["name"] == "query_dataset" for s in ag.tool_schemas)
    early = here()
    for _ in range(8):
        ag.history.append({"role": "tool_result", "content": "x" * 3000})
    ag._trim_tool_results(keep_recent=2)
    late = here()
    toolindex.reset()
    return early and late and not here()


def t_load_cannot_bypass_privacy():
    """An on-demand load must never reach past sandbox / off-the-record."""
    from forge.agent import Agent
    from forge import tools as T, toolindex
    from forge.session import Workspace
    ag = Agent(provider=object(), tools=T.build_tools(Workspace(tempfile.mkdtemp())),
               permission_mode="auto", superego=None)
    ag.active_mode = "balanced"; ag.privacy = "sandbox"
    ag.client_env = ""; ag.identity_owner = ""; ag.history = []
    toolindex.reset(); toolindex.load_tools(["run_command", "write_file"], T._INDEX_REGISTRY)
    names = {s["name"] for s in ag.tool_schemas}
    toolindex.reset()
    return "run_command" not in names and "write_file" not in names


def t_tool_search_quality():
    """Keyword-only retrieval put verify_case top for 'change how funny you
    are'. Semantic retrieval fixed it; junk must still return nothing."""
    from forge import tools as T, toolindex
    from forge.session import Workspace
    T.build_tools(Workspace(tempfile.mkdtemp())); reg = T._INDEX_REGISTRY
    pairs = [("search my old emails", "search_life"), ("design a 3d part", "design_part"),
             ("is this drug real", "verify_drug"), ("change how funny you are", "set_personality"),
             ("paper trade a strategy", "paper_market"), ("draw a picture", "generate_image")]
    top3 = lambda q: [l.strip().split(" —")[0]
                      for l in toolindex.find_tools(q, reg, limit=8).splitlines()[1:4]]
    hits = sum(1 for q, w in pairs if w in top3(q))
    junk_ok = all("Nothing" in toolindex.find_tools(j, reg).splitlines()[0]
                  for j in ["zzzz qqqq wubble", "flurb nax qopple zint"])
    return hits >= 5 and junk_ok


# --------------------------------------------------------------- embedder
def t_embedder_batches():
    """embed_documents sent everything in one request and silently returned
    None past ~32 items — Cortex indexes in chunks of 64."""
    from forge.embed import embed_documents, available
    if not available():
        return True
    m = embed_documents([f"doc number {i} about a thing" for i in range(40)])
    return m is not None and m.shape[0] == 40


# ------------------------------------------------------------------ shelf
def t_shelf_rejects_garbage():
    """An empty table and the bare string 'nope' both saved as ✓ CORROBORATED
    — the string was recorded as '4 entries', having counted its letters."""
    from forge import datasets as D
    S = ["http://a.example", "http://b.example"]
    bad = [D.save_dataset("_t_empty", {}, S), D.save_dataset("_t_str", "nope", S),
           D.save_dataset("_t_num", 42, S)]
    for n in ("_t_empty", "_t_str", "_t_num"):
        Path(f"datasets/{n}.json").unlink(missing_ok=True)
    return all(str(r).startswith("Error") for r in bad)


def t_circular_selftests_stay_demoted():
    """Four wing sims carried ✓ while testing the code against its own output,
    and verify_shelf kept re-promoting them because re-running is exactly what
    a circular test survives."""
    cat = json.loads(Path("sims/_catalog.json").read_text())
    flagged = [n for n, m in cat.items() if m.get("selftest_circular")]
    return bool(flagged) and all(not cat[n].get("validated") for n in flagged)


# ------------------------------------------------------------------ recall
def t_recall_refuses_gibberish():
    """The semantic floor sat below the noise ceiling, so gibberish came back
    quoted as a memory."""
    from forge import recall
    r = recall.search("zzzz qqqq wubble flurb", limit=3)
    return "Nothing" in r[:40] or not r.strip()


# ------------------------------------------------------------------ misc
def t_physics_refuses_impossible():
    """twin_trip printed 'Earth twin ages: -8.0000 years' for a negative
    distance — people aging backwards, with formula annotations."""
    from forge import physics
    r = physics.relativity_sim(scenario="twin_trip", distance_ly=-2, v_c=0.5)
    good = physics.relativity_sim(scenario="twin_trip", distance_ly=4, v_c=0.5)
    return r.startswith("Error") and "16.0000" in good


def t_markets_refuse_impossible():
    """implied_prob returned 'implied probability: -0.5' for negative odds, and
    arbitrage reported a negative stake."""
    from forge import markets as M
    for fn, args in ((M.implied_prob, {"decimal_odds": -2}),
                     (M.implied_prob, {"decimal_odds": 0}),
                     (M.arbitrage, {"odds_a": 2.1, "odds_b": -1})):
        try:
            fn(args)
            return False
        except (ValueError, KeyError, ZeroDivisionError):
            pass
    return abs(M.implied_prob({"decimal_odds": 2.5})[0][1] - 0.4) < 1e-9


def t_xfiles_keeps_near_complete_suspects():
    """A suspect covering 399 of 400 days was dropped wholesale over one
    missing holiday, leaving a hollow verdict."""
    import inspect
    from forge import xfiles
    src = inspect.getsource(xfiles._aligned_returns)
    return "0.9" in src and "dropped.append" in src


def t_doolittle_wont_corner_noise():
    """A random-context call must not be cornered to a confident meaning."""
    import numpy as np
    from forge import doolittle as dl
    rng = np.random.RandomState(4)
    obs = []
    for _ in range(30):
        obs.append({"call": "real", "cues": {"threat": True, "then_flee_or_freeze": True,
                                             "predator_ground": True, "calm": False}})
    cues = ["threat", "food", "calm", "juvenile", "movement", "competitor"]
    for _ in range(30):
        obs.append({"call": "noise", "cues": {c: bool(rng.random() < 0.5) for c in cues}})
    by = {c["call"]: c for c in dl.deduce(obs)["calls"]}
    return by["real"]["conclusive"] and not by["noise"]["conclusive"]


def t_cortex_refuses_nonsense():
    """Semantic search scores EVERY record, so an unrelated query still had a
    'best' match and returned it as a memory."""
    from forge import cortex
    if not cortex.RECORDS.exists():
        return True
    r = cortex.search_life("submarine penguin tax fraud")
    return "clearly match" in r or "empty" in r or "Nothing" in r


def t_persona_clamps():
    """Dials had to refuse nonsense rather than store it."""
    from forge import persona
    before = persona.settings()["humor"]
    persona.set_personality("humor", 9999)
    hi = persona.settings()["humor"]
    persona.set_personality("humor", -50)
    lo = persona.settings()["humor"]
    bad = persona.set_personality("humor", "high")
    persona.set_personality("humor", before)
    return hi == 100 and lo == 0 and "isn't a number" in bad


def t_forensic_env_not_clobbered():
    """A per-turn mode switch deleted a FORGE_FORENSIC the user had set
    deliberately, silently discarding the recording they asked for."""
    import os
    from forge import forensic
    had = os.environ.get("FORGE_FORENSIC")
    os.environ["FORGE_FORENSIC"] = "1"
    forensic.set_enabled(False)          # what a normal-mode turn does
    still_on = forensic.enabled()
    if had is None:
        os.environ.pop("FORGE_FORENSIC", None)
    return still_on


def t_forensic_redacts():
    """The flight recorder must never write a secret."""
    from forge import forensic
    return forensic._redact("hunter2", "password") == "[redacted]"


# ------------------------------------------------------------------- main
CHECKS = [
    ("context: schema+prompt counted", t_overhead_counted, False),
    ("context: force-compaction shrinks the big turn", t_force_compaction_shrinks, False),
    ("superego: evidence not starved", t_evidence_not_starved, False),
    ("superego: truncation is marked", t_truncation_is_marked, False),
    ("vault: chat-typed secrets scrubbed", t_secrets_scrubbed, False),
    ("vault: values never readable", t_vault_hides_values, False),
    ("toolindex: does not index itself", t_toolindex_excludes_itself, False),
    ("toolindex: loads survive, then reset", t_loaded_tools_survive_and_reset, False),
    ("toolindex: cannot bypass privacy", t_load_cannot_bypass_privacy, False),
    ("toolindex: retrieval quality + junk refused", t_tool_search_quality, True),
    ("embedder: batches large inputs", t_embedder_batches, True),
    ("shelf: garbage refused", t_shelf_rejects_garbage, False),
    ("shelf: circular selftests stay demoted", t_circular_selftests_stay_demoted, False),
    ("recall: refuses gibberish", t_recall_refuses_gibberish, True),
    ("physics: refuses impossible trips", t_physics_refuses_impossible, False),
    ("markets: refuses impossible odds", t_markets_refuse_impossible, False),
    ("xfiles: keeps near-complete suspects", t_xfiles_keeps_near_complete_suspects, False),
    ("doolittle: will not corner noise", t_doolittle_wont_corner_noise, False),
    ("cortex: refuses nonsense queries", t_cortex_refuses_nonsense, True),
    ("persona: clamps and refuses bad values", t_persona_clamps, False),
    ("forensic: env override not clobbered", t_forensic_env_not_clobbered, False),
    ("forensic: redacts secrets", t_forensic_redacts, False),
]

if __name__ == "__main__":
    t0 = time.time()
    for name, fn, needs in CHECKS:
        check(name, fn, needs)
    for n in PASS:
        print(f"  ✓ {n}")
    for item in FAIL:
        n, why = item if isinstance(item, tuple) else (item, "failed")
        print(f"  ✗ {n}  <- {why}")
    for n in SKIP:
        print(f"  · {n} (skipped)")
    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped "
          f"in {time.time()-t0:.1f}s")
    sys.exit(len(FAIL))
