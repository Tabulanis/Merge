#!/usr/bin/env python3
"""babysit — a supervisor's view of Merge, and a running tally of her flaws.

Watching her work is what actually finds bugs (unit tests never surfaced the
loop, the string-suspect crash, or the truncated-evidence false bounce). This
reads the same on-disk record she leaves behind — every session log plus the
superego ledger — so any session can be audited afterwards, including ones
started from the dashboard or another terminal.

  babysit tally              health across every session, and the open findings
  babysit sessions [n]       recent sessions, newest first, with health flags
  babysit check <id|last>    audit one session turn by turn
  babysit flag <id> <text>   record a finding (the running tally)
  babysit fixed <n>          mark finding #n fixed
  babysit findings           the running tally on its own
  babysit watch [secs]       live: watch her work and flag trouble as it happens
  babysit replay <id>        the flight recording (Bug Hunt mode) for a session
  babysit ask "<task>"       hand a job to Merge FIRST; flag it only if she trips
  babysit ask --want=tool,tool "<task>"   also assert WHICH tools she used

Read-only against her data; the only thing it writes is the findings file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

FORGE = Path.home() / ".forge"
SESSIONS = FORGE / "sessions"
LEDGER = FORGE / "ledger.jsonl"
FINDINGS = FORGE / "babysit-findings.jsonl"

# Signals that something went wrong in a turn, and what each one means.
LOOP_AT = 6          # same tool this many times in one session = suspicious


def _sessions() -> list[dict]:
    out = []
    for f in SESSIONS.glob("*.json"):
        try:
            d = json.load(f.open())
            d["_file"] = f
            out.append(d)
        except Exception:
            continue
    return sorted(out, key=lambda d: d.get("last_used", 0), reverse=True)


def _ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _health(sess: dict) -> dict:
    """What went wrong in this session, mechanically."""
    log = sess.get("log", [])
    tools, errors, notes = {}, [], []
    for e in log:
        k = e.get("kind")
        if k == "tool":
            tools[e.get("tool", "?")] = tools.get(e.get("tool", "?"), 0) + 1
        elif k == "error":
            errors.append((e.get("text") or "")[:120])
        elif k == "result" and "budget spent" in str(e.get("text", "")):
            notes.append("hit the per-tool budget (loop stopped)")
        elif k == "note":
            t = str(e.get("text", ""))
            if "compact" in t.lower() or "condensed" in t.lower():
                notes.append("memory compaction fired")
            elif "Stopped" in t:
                notes.append("stopped early")
    loops = {k: v for k, v in tools.items() if v >= LOOP_AT}
    return {"tools": tools, "loops": loops, "errors": errors,
            "notes": sorted(set(notes)),
            "turns": sum(1 for e in log if e.get("kind") == "user"),
            "answered": sum(1 for e in log if e.get("kind") == "done")}


def _findings() -> list[dict]:
    if not FINDINGS.exists():
        return []
    out = []
    for line in FINDINGS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _ago(ts: float) -> str:
    if not ts:
        return "?"
    d = time.time() - ts
    for n, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if d >= n:
            return f"{int(d // n)}{unit} ago"
    return "just now"


# ---- commands -----------------------------------------------------------
def cmd_tally() -> None:
    sess, led = _sessions(), _ledger()
    tools_total, loop_sessions, err_sessions, compacted = 0, 0, 0, 0
    for s in sess:
        h = _health(s)
        tools_total += sum(h["tools"].values())
        loop_sessions += 1 if h["loops"] else 0
        err_sessions += 1 if h["errors"] else 0
        compacted += 1 if any("compaction" in n for n in h["notes"]) else 0
    bounces = [l for l in led if l.get("verdict") == "bounce"]

    print("BABYSITTING TALLY")
    print(f"  sessions on record : {len(sess)}")
    print(f"  tool calls         : {tools_total}")
    print(f"  superego reviews   : {len(led)}  ({len(bounces)} bounced"
          f"{f', {100*len(bounces)//len(led)}%' if led else ''})")
    print(f"  sessions w/ a loop : {loop_sessions}"
          f"   (same tool >= {LOOP_AT}x)")
    print(f"  sessions w/ errors : {err_sessions}")
    print(f"  memory compactions : {compacted}")
    if bounces:
        print("\n  most recent bounce reasons:")
        for b in bounces[-3:]:
            print(f"    · {(b.get('reason') or '?')[:96]}")
    open_f = [f for f in _findings() if not f.get("fixed")]
    print(f"\n  OPEN FINDINGS: {len(open_f)}")
    for f in open_f:
        print(f"    #{f['n']} [{f.get('session','?')}] {f['text'][:88]}")
    if not open_f:
        print("    (none — nothing outstanding)")


def cmd_sessions(n: int = 12) -> None:
    print(f"{'id':14} {'when':10} {'turns':>5} {'tools':>6}  flags")
    for s in _sessions()[:n]:
        h = _health(s)
        flags = []
        if h["loops"]:
            flags.append("LOOP:" + ",".join(f"{k}x{v}" for k, v in h["loops"].items())[:34])
        if h["errors"]:
            flags.append(f"ERR x{len(h['errors'])}")
        if h["turns"] and h["answered"] < h["turns"]:
            flags.append(f"UNANSWERED {h['turns']-h['answered']}")
        flags += [n for n in h["notes"] if "compaction" in n or "stopped" in n]
        print(f"{s.get('id','?'):14} {_ago(s.get('last_used',0)):10} "
              f"{h['turns']:5} {sum(h['tools'].values()):6}  "
              f"{'; '.join(flags) if flags else 'clean'}")


def cmd_check(sid: str) -> None:
    sess = _sessions()
    if sid == "last":
        target = sess[0] if sess else None
    else:
        target = next((s for s in sess if str(s.get("id", "")).startswith(sid)), None)
    if not target:
        print(f"No session matching '{sid}'. Try: babysit sessions")
        return
    h = _health(target)
    print(f"SESSION {target.get('id')}  ({_ago(target.get('last_used',0))})")
    print(f"  workspace : {target.get('workspace','?')}")
    print(f"  model     : {target.get('model','?')}")
    print(f"  turns {h['turns']} · answered {h['answered']} · "
          f"tool calls {sum(h['tools'].values())}")
    if h["tools"]:
        print("  tools used: " + ", ".join(
            f"{k}x{v}" for k, v in sorted(h["tools"].items(), key=lambda x: -x[1])))
    for w in h["loops"]:
        print(f"  ⚠ LOOP: {w} called {h['loops'][w]}x")
    for e in h["errors"]:
        print(f"  ⚠ ERROR: {e}")
    for n in h["notes"]:
        print(f"  · {n}")

    print("\n  --- transcript ---")
    for e in target.get("log", []):
        k = e.get("kind")
        if k == "user":
            print(f"\n  USER: {(e.get('text') or '')[:300]}")
        elif k == "tool":
            print(f"    → {e.get('tool')}({str(e.get('summary',''))[:60]})")
        elif k == "result":
            t = str(e.get("text", "")).replace("\n", " ")[:140]
            print(f"      {t}")
        elif k == "text":
            print(f"  MERGE: {(e.get('text') or '')[:400]}")
        elif k == "error":
            print(f"  !! ERROR: {(e.get('text') or '')[:200]}")

    # superego verdicts that overlap this session's window
    t0 = min([e.get("t", 0) for e in target.get("log", []) if e.get("t")] or [0])
    t1 = target.get("last_used", 0) + 5
    rel = [l for l in _ledger() if t0 <= l.get("t", 0) <= t1]
    if rel:
        print("\n  --- superego verdicts this session ---")
        for l in rel:
            print(f"    [{l.get('verdict')}] {(l.get('reason') or '')[:120]}")


def cmd_flag(sid: str, text: str) -> None:
    n = len(_findings()) + 1
    rec = {"n": n, "session": sid, "text": text, "t": time.time(), "fixed": False}
    with FINDINGS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"Logged finding #{n}: {text}")


def cmd_fixed(n: int) -> None:
    recs = _findings()
    hit = False
    for r in recs:
        if r["n"] == n:
            r["fixed"] = True
            r["fixed_t"] = time.time()
            hit = True
    if not hit:
        print(f"No finding #{n}.")
        return
    FINDINGS.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"Finding #{n} marked fixed.")


def cmd_findings() -> None:
    recs = _findings()
    if not recs:
        print("No findings logged yet.")
        return
    for r in recs:
        mark = "✔ fixed" if r.get("fixed") else "○ open "
        print(f"  {mark} #{r['n']} [{r.get('session','?')}] {r['text']}")
    print(f"\n  {sum(1 for r in recs if not r.get('fixed'))} open, "
          f"{sum(1 for r in recs if r.get('fixed'))} fixed.")


def cmd_watch(interval: float = 5.0) -> None:
    """Sit over her shoulder. Polls the session logs and the ledger, and prints
    only what's new and worth knowing — a loop forming, an error, a bounce, a
    turn that ended without an answer. Ctrl-C to stop."""
    print(f"watching Merge (every {interval:g}s) — Ctrl-C to stop\n")
    seen_evt: dict[str, int] = {}
    seen_led = len(_ledger())
    warned: set = set()
    try:
        while True:
            for s in _sessions()[:6]:
                sid = s.get("id", "?")
                log = s.get("log", [])
                start = seen_evt.get(sid)
                if start is None:                 # first sight: don't replay history
                    seen_evt[sid] = len(log)
                    continue
                for e in log[start:]:
                    k = e.get("kind")
                    if k == "user":
                        print(f"[{sid[:8]}] USER: {(e.get('text') or '')[:110]}")
                    elif k == "tool":
                        print(f"[{sid[:8]}]   → {e.get('tool')}")
                    elif k == "error":
                        print(f"[{sid[:8]}]   !! ERROR: {(e.get('text') or '')[:150]}")
                    elif k == "text":
                        print(f"[{sid[:8]}] MERGE: {(e.get('text') or '')[:150]}")
                seen_evt[sid] = len(log)

                h = _health(s)
                for tool, n in h["loops"].items():
                    key = f"{sid}:{tool}:{n // 5}"
                    if key not in warned:
                        warned.add(key)
                        print(f"[{sid[:8]}] ⚠ LOOP forming — {tool} called {n}x")
                if h["turns"] and h["answered"] < h["turns"]:
                    key = f"{sid}:unanswered:{h['turns']}"
                    if key not in warned:
                        warned.add(key)
                        print(f"[{sid[:8]}] ⚠ a turn ended without an answer")

            led = _ledger()
            for l in led[seen_led:]:
                if l.get("verdict") == "bounce":
                    print(f"           ⚠ BOUNCED: {(l.get('reason') or '?')[:110]}")
            seen_led = len(led)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped watching.")


def cmd_replay(sid: str) -> None:
    """The flight recording for a session — full args, full results, timings."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from forge import forensic
    except Exception as e:
        print(f"can't load the recorder: {e}")
        return
    if sid == "last":
        got = forensic.sessions()
        if not got:
            print("No flight recordings yet. Use Bug Hunt mode to make one.")
            return
        sid = got[0][0]
    recs = forensic.read(sid)
    if not recs:
        print(f"No recording for '{sid}'. Recordings exist for: "
              + ", ".join(s[0] for s in forensic.sessions()[:8]))
        return
    print(f"FLIGHT RECORDING {sid} — {len(recs)} events")
    t0 = recs[0]["t"]
    for r in recs:
        dt = f"+{r['t']-t0:6.1f}s"
        if r["kind"] == "turn_start":
            print(f"\n{dt} TURN [{r.get('mode')}] {str(r.get('request'))[:120]}")
        elif r["kind"] == "tool":
            print(f"{dt}   → {r.get('tool')}({str(r.get('args'))[:90]}) "
                  f"[{r.get('seconds')}s, attempt {r.get('attempt')}]")
            print(f"{'':9}     {str(r.get('result'))[:200]}")
        elif r["kind"] == "superego_evidence":
            print(f"{dt}   ⚖ reviewer saw {len(str(r.get('digest')))} chars of evidence")
        else:
            print(f"{dt}   {r['kind']}: {str(r)[:120]}")


# Things that mean the turn went wrong, and what each one is.
_HICCUPS = [
    ("error 400",        "model server rejected the request (usually oversized)"),
    ("Model call failed", "the model call failed outright"),
    ("budget spent",     "hit the per-tool ceiling — she was looping"),
    ("repeat blocked",   "retried an identical failing call"),
    ("Error running",    "a tool raised"),
    ("emergency",        "emergency memory compaction"),
    ("Superego review",  "her answer was bounced by the reviewer"),
    ("Stopped after",    "ran out of steps without finishing"),
]


def _tools_used_since(t0: float) -> list:
    """Which tools actually ran after t0 — the path, not just the answer."""
    used = []
    for s in _sessions()[:4]:
        for e in s.get("log", []):
            if e.get("kind") == "tool" and e.get("t", 0) >= t0:
                used.append(e.get("tool"))
    return used


def cmd_ask(task: str, model: str = "merge38", workspace: str = "",
            want: list | None = None) -> None:
    """Hand a job to Merge first. Print her answer, then say plainly whether
    she got there cleanly or tripped — and log a finding when she trips, so
    the escalation to a bigger model is evidence-driven rather than a hunch."""
    import subprocess
    # Default to wherever you are, not a path from one machine.
    ws = workspace or os.environ.get("BABYSIT_WORKSPACE") or os.getcwd()
    forge_bin = str(Path.home() / "forge" / ".venv" / "bin" / "forge")
    print(f"→ handing to Merge ({model})…\n")
    _t0 = time.time()
    try:
        r = subprocess.run([forge_bin, "-m", model, "--auto", "-w", ws, task],
                           capture_output=True, text=True, timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        out = "[TIMED OUT after 600s]"
    except Exception as e:
        out = f"[could not run: {e}]"

    print(out.strip()[-3000:])

    # Match only on HARNESS lines, not on her prose. The first version fired
    # because she wrote "either error out or give trivially scaled" — a
    # detector that cries wolf on the word "error" is worse than none, since
    # every clean run then looks like a failure.
    harness = "\n".join(
        ln for ln in out.splitlines()
        if ln.lstrip().startswith(("·", "!!", "Error running", "Model call failed",
                                   "Stopped after", "Superego review", "⚠"))
        or "error 400" in ln.lower())
    found = [why for pat, why in _HICCUPS if pat.lower() in harness.lower()]

    # Merge's own point while designing these tests: a right-shaped answer
    # reached by the wrong path is indistinguishable from a real pass unless
    # you record WHICH tool ran. Her example was live — asked to read a saved
    # dataset she shelled out with `cat` instead of touching the loader, and
    # the answer was correct, so the test "passed" while proving nothing.
    used = _tools_used_since(_t0)
    if used:
        print(f"\n  path taken: {', '.join(dict.fromkeys(used))}")
    if want:
        missing = [w for w in want if w not in used]
        if missing:
            found.append(f"did NOT use the expected tool(s): {', '.join(missing)} "
                         f"— took {', '.join(dict.fromkeys(used)) or 'no tools'} instead")
    answered = bool(out.strip()) and "[TIMED OUT" not in out
    print("\n" + "-" * 60)
    if not answered:
        found.append("produced no answer at all")
    if found:
        print("⚠ SHE TRIPPED — worth a look:")
        for f in sorted(set(found)):
            print(f"    · {f}")
        sid = "unknown"
        s = _sessions()
        if s:
            sid = s[0].get("id", "unknown")
        cmd_flag(sid, f"ask: {'; '.join(sorted(set(found)))} — task: {task[:70]}")
        print("\n  (logged; `babysit replay last` for the full recording)")
    else:
        print("✓ clean run — no hiccups detected. No need to escalate.")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "tally"
    if cmd == "tally":
        cmd_tally()
    elif cmd == "sessions":
        cmd_sessions(int(a[1]) if len(a) > 1 else 12)
    elif cmd == "check" and len(a) > 1:
        cmd_check(a[1])
    elif cmd == "flag" and len(a) > 2:
        cmd_flag(a[1], " ".join(a[2:]))
    elif cmd == "fixed" and len(a) > 1:
        cmd_fixed(int(a[1]))
    elif cmd == "findings":
        cmd_findings()
    elif cmd == "watch":
        cmd_watch(float(a[1]) if len(a) > 1 else 5.0)
    elif cmd == "replay" and len(a) > 1:
        cmd_replay(a[1])
    elif cmd == "ask" and len(a) > 1:
        rest, want = a[1:], None
        if rest and rest[0].startswith("--want="):
            want = [w for w in rest[0].split("=", 1)[1].split(",") if w]
            rest = rest[1:]
        cmd_ask(" ".join(rest), want=want)
    else:
        print(__doc__)
