# Merge

**A complete AI assistant that lives on your own machine — and refuses to lie to you.**

Merge is a local assistant with eyes, a voice, real memory, and a workshop.
She runs on one consumer GPU, talks to any OpenAI-compatible model server,
and nothing she hears or says ever leaves the house. She reads your screen,
browses the web, writes and tests code, draws pictures, speaks aloud on your
phone, remembers what you told her last month — and when she doesn't know
something, the system is built so that *saying so* is the only move that
passes review.

She named herself. Given free run of her first workspace, she noticed the word
at the heart of everything she could see — histories merging, two things
becoming one better thing — and picked **Merge**. It stuck. (Forge is the
engine she runs on.)

Built by a working artist — not a programmer — by directing AI as the hands.
Every load-bearing piece was tested the day it was built. That story is the
proof of what this stack is for.

---

## What makes it different

Plenty of projects bolt a chat UI onto a local model. The interesting parts of
this one are the **harness** around the model:

### 🔒 An honesty architecture, not an honesty prompt
- **The superego** — a *sealed* second reviewer that judges every final answer
  against a deterministic digest of what actually happened (tools run, results
  returned, prior claims). Claims that don't match the evidence get bounced.
  Honesty about failure always passes; invented evidence never does. Every
  verdict is logged to a ledger.
- **The claims check** — "I created the file" without a `write_file` call in
  the record is caught mechanically, not by vibes.
- **Fail-open guards everywhere** — a garbled review, a dead helper model, an
  unreachable service: the work continues and says so plainly. No component
  can take the whole agent down with it.

### 🔧 She builds her own tools — and can't trust them until they're proven
- **Sims**: deterministic physics/math she'd otherwise hand-wave becomes code
  she writes once and reuses forever — but a sim only earns its ✓ by
  reproducing an independently-known answer within tight tolerance. Otherwise
  it's marked EXPERIMENTAL and says so every time it runs.
- **Datasets**: real-world reference data must cite sources; one source is
  a *lead*, two independent sources make a *fact*.
- **The immune system**: `verify_shelf` re-runs every sim's self-test and
  integrity-checks every dataset — on demand and nightly. A sim that stops
  reproducing its known answer is demoted on the spot. Stale checkmarks
  cannot survive.
- **The hiccup ledger**: every degradation in a turn (failed tool, aborted
  call, memory trim) is tracked — and anything she tries to *persist* right
  after a hiccup comes back stamped with a health warning. Confident garbage
  doesn't get to enter the permanent stores quietly.

### 🧠 Memory in three speeds
- Context window (short) with graceful compaction — summarized on purpose
  beats forgetting at random, and one giant paste can't blow the window
  (the seatbelt trims it and says so).
- Every conversation saved verbatim (mid), distilled into one-line index
  cards by a background librarian, and **retrieved associatively**: a small
  embedding model finds memories by *meaning*, so "that thing about the
  rocket wall" surfaces even if nobody ever said those words. Embedder down?
  Falls back to keyword. Less associative, never broken.
- A self-compressing notebook (long): fixed size by design — it gets
  *rewritten* smaller, never just truncated, so old lessons don't silently
  fall off the end.

### 🎛️ Modes that actually change the machine
Speed styles (Flash / Muse / Balanced / Precise / Deep) swap thinking,
temperature, tool loadout, and review strictness per message — plus a **Teach
mode** that explains from first principles and checks understanding, and an
auto-router that picks from your phrasing.

**Privacy is a separate axis:** *Off the record* reads everything but can
change nothing and records nothing. *Sandbox* is knowledge-only — no
filesystem at all. Both leave no trace by design, verified by test.

### 👁️ 🗣️ 🖼️ 🌐 The senses
Native vision (she looks at screenshots, photos, her own drawings), local
image generation, server-side natural voice (same voice on every device),
hands-free voice chat with silence detection, web search + page reading, and
a warm headless browser she drives herself — build a page, open it, read the
console errors, fix her own code. She can even *see sound*: turn a clip or a
call into a visual sound-portrait (spectrogram + harmonics) she reads with her
eyes.

### 🛬 Long-task landing gear
Runaway protection that *delivers* instead of dying: a step-budget warning
("wrap up now"), then tools are physically withdrawn so the final answer must
be written; heartbeats during long tool runs ("still going — 60s in"); a Stop
that abandons a stuck call within seconds instead of politely waiting out a
five-minute nap. Silent black holes were hunted to extinction.

### 📊 An honest quant lab (bonus)
Validated business & market calculators, paper trading against real data with
real fees, walk-forward out-of-sample testing, regime analysis, and a pattern
scanner that re-runs its whole search on *shuffled* data to measure how much
"signal" pure luck produces. It mostly says NO. That's the feature — it was
built watching beautiful backtests die honestly, including one where the
literal phase of the moon "predicted" Bitcoin three separate times. No live
trading, no exchange keys — by design, forever. Plus an *X-Files* hunt: given
two markets that move together, it goes looking for the hidden third party
driving both — and reports a statistical suspect, never a proof of cause.

### ⚖️ 🩺 Grounded desks: law and medicine (verify, or say NOT FOUND)
Legal and medical lookups wired to authoritative sources — US court, statute,
and regulation databases; the National Library of Medicine's drug and
condition terminologies — and held to the same rule as everything else: a case,
statute, drug, or condition that doesn't verify comes back **NOT FOUND**, never
invented. They translate between plain words and the exact term a lawyer or
doctor can't misread, and they refuse to diagnose, dose, or give legal advice.
A false "VERIFIED" is the one thing they cannot emit; emergencies get pointed
at 911.

### 🐾 Cornering meaning by elimination (the deduction pad)
The most honest kind of ambition. First, tools that look for *structure* in
animal calls — clustering a recording into a repertoire and null-testing
whether the sequence is more than random. Then a **deduction pad** that plays
Clue with what a call might mean, and never once claims to translate it. It
holds a blob of candidate *phrases* ("ground predator — get up / bolt", "food
here — come eat") and rules out the ones the call fires *without*, chiseling a
coarse cluster down toward a single phrase as the evidence sharpens. And it
asks the prior question first — *is this even a signal, or just noise?* — by
testing the call against a shuffle null, so one that can't beat chance stays a
blob no matter how tidy its cues look. No signal, no meaning; it eliminates, it
never decodes.

---

## Quickstart

You need: Python 3.11+, and any OpenAI-compatible model server
([llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`,
Ollama, vLLM, or a cloud key if you must).

```bash
git clone https://github.com/Tabulanis/Merge && cd Merge
python -m venv .venv && .venv/bin/pip install -e .

# point it at your model server (edit ports/paths in start-model.sh, or use
# any running OpenAI-compatible endpoint via the dashboard's model settings)
bash start-model.sh merge          # example: llama-server on :8085

bash start-dashboard.sh            # web UI + phone PWA on :8770 (HTTPS)
# or the terminal:
.venv/bin/forge
```

First run mints an access token (shown in the terminal) — the dashboard is a
shell on your machine, so it's token-gated even at home. Optional extras the
scripts wire up if present: piper voices (her voice), an embedding model (her
associative memory), a small CPU model (the librarian), SD-Turbo (drawing).
Reference setup: a 24GB GPU running an abliterated Qwen 27B with vision —
but the harness is model-agnostic and degrades gracefully.

## The map

| | |
|---|---|
| `forge/agent.py` | the loop, the honesty harness, memory compaction, landing gear |
| `forge/providers.py` | OpenAI-compatible + Anthropic backends, streaming, tool calls |
| `forge/tools.py` | the toolbelt (~60 tools) — files, shell, web, sims, browser, senses |
| `forge/server.py` | FastAPI dashboard: chat, SSE streaming, sessions, voice, uploads |
| `forge/sims.py` / `datasets.py` | her self-built instruments + validation/corroboration gates |
| `forge/recall.py` / `embed.py` | three-speed memory, the librarian, associative retrieval |
| `forge/modes.py` | speed styles, Teach, privacy modes, auto-routing |
| `forge/paper_market.py` / `xfiles.py` &c. | the honest quant lab + the third-party hunt |
| `forge/law.py` / `medical.py` | grounded legal & medical desks — verify, or NOT FOUND |
| `forge/bioacoustics.py` / `doolittle.py` | animal-call structure + the deduction pad |
| `forge/audio_nerve.py` | sound → a visual "sound portrait" she can see |
| `model-tests/` | the standing regression battery (sycophancy, false premises, neutrality) |

## Philosophy, in four lines

1. **Anything deterministic goes through a tool.** Models guess; tools know.
   The tool set only grows.
2. **Unproven means unsaid.** Validation gates, corroboration counts, and a
   reviewer that bounces unearned confidence.
3. **Degrade loudly, never silently.** Every fallback announces itself.
4. **The machine works for its owner.** Local weights, local voice, local
   memory, privacy modes that truly record nothing. No rent, no telemetry,
   no one reading over your shoulder.

## License

MIT. Use it, fork it, learn from it. Nothing here is financial advice, and
the trading lab cannot trade — that part isn't a limitation, it's a promise.
