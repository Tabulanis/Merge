"""
Multi-model orchestration — chaining models into loops that do real work.

The thing that makes a 14B-30B local model genuinely useful isn't a bigger
model, it's a better loop. A weak model that writes code, *runs the tests*,
reads the failure, and tries again will beat a strong model that answers once
and hopes. So the loops here are built around checks that are real — a shell
command's exit code — not a model's opinion of its own work.

Six kinds of step:

  agent    — a full agent loop with tools. Reads, writes, runs things.
  ask      — one model call, no tools. Analysis, critique, routing decisions.
  command  — run a shell command. No model involved. The honest referee.
  loop     — repeat inner steps until a check passes, or give up after N.
  plan     — one call that breaks a big, open-ended ask into a numbered
             list of small, checkable sub-steps. Exists because an open
             task ("find the gap and fill it") is where a local model
             fabricates — it can't verify a claim that big in one pass.
             Narrow sub-steps it CAN verify.
  foreach  — runs its inner steps once per item a `plan` step produced,
             {item} bound to that one sub-step's text. Pairs with plan to
             turn "one big guess" into "several small, checked answers."
  gate     — an `ask` call whose answer can stop the WHOLE pipeline early.
             Exists because "draft only from confirmed facts, say so if
             something's missing" is a soft instruction a confident model
             talks past — verified live: research came back thin AND
             wrong (a citation from an unrelated part of the book), and
             the very next step drafted a full fabricated scene anyway,
             instructions notwithstanding. A gate makes "is there enough
             REAL evidence to proceed" its own narrow, constrained
             judgment call — the kind of check this model is actually
             decent at, per the divergence test that first surfaced this
             pattern — instead of trusting the same pass that's about to
             write prose to also police itself.

Each step's output is stored under its name and can be dropped into any later
prompt with {braces}, so steps feed each other:

  steps:
    - name: build
      kind: agent
      model: qwen30b
      prompt: "{input}"
    - name: verify
      kind: command
      command: "python3 -m pytest -q"

A pipeline is plain data (see pipelines.yaml), so the visual brick editor can
read and write these without touching Python.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .agent import Agent
from .providers import build_provider
from .tools import Workspace, build_tools


@dataclass
class StepResult:
    name: str
    kind: str
    output: str
    ok: bool = True
    detail: str = ""


@dataclass
class RunEvent:
    """Progress from a pipeline run, so a UI can show it live."""
    kind: str          # "step_start" | "step_done" | "loop_round" | "done" | "error"
    step: str = ""
    text: str = ""
    round: int = 0
    ok: bool = True


class PipelineError(Exception):
    pass


def _fill(template: str, values: dict[str, str]) -> str:
    """
    Substitute {name} from earlier step outputs.

    Deliberately not str.format: prompts are full of braces that aren't
    placeholders (JSON examples, f-strings, code samples), and format() would
    choke on every one of them. Only names we actually have are replaced;
    anything else is left exactly as written.
    """
    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        return values.get(key, m.group(0))
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", sub, template)


class Pipeline:
    """One named orchestration recipe, ready to run."""

    def __init__(self, spec: dict, models: dict, workspace: Path,
                 permission_mode: str = "auto"):
        self.name = spec.get("name", "pipeline")
        self.description = spec.get("description", "")
        self.steps = spec.get("steps", [])
        self.models = models
        self.workspace = Path(workspace)
        self.permission_mode = permission_mode
        if not self.steps:
            raise PipelineError(f"Pipeline {self.name!r} has no steps.")

    # -- model access ------------------------------------------------

    def _provider(self, model_name: str):
        if model_name not in self.models:
            raise PipelineError(
                f"Step wants model {model_name!r}, which isn't configured. "
                f"Available: {', '.join(self.models) or '(none)'}"
            )
        return build_provider(self.models[model_name])

    # -- the four step kinds -----------------------------------------

    def _run_agent(self, step: dict, values: dict, ask: Callable | None) -> StepResult:
        provider = self._provider(step["model"])
        ws = Workspace(self.workspace)
        agent = Agent(
            provider, build_tools(ws),
            max_steps=int(step.get("max_steps", 30)),
            permission_mode=step.get("permission_mode", self.permission_mode),
            system_prompt=step.get("system") or _AGENT_SYSTEM,
        )
        prompt = _fill(step.get("prompt", "{input}"), values)
        final_text: list[str] = []
        for ev in agent.run(prompt, ask=ask or (lambda *a: True)):
            if ev.kind == "text" and ev.text.strip():
                final_text.append(ev.text)
            elif ev.kind == "error":
                return StepResult(step["name"], "agent", "\n".join(final_text),
                                  ok=False, detail=ev.text)
        return StepResult(step["name"], "agent", "\n".join(final_text))

    def _run_ask(self, step: dict, values: dict) -> StepResult:
        provider = self._provider(step["model"])
        prompt = _fill(step.get("prompt", "{input}"), values)
        system = _fill(step.get("system", "You are a careful, concise reviewer."), values)
        # A step can pin the answer's shape. `choices: [A, B]` is the common
        # case (routing, yes/no verdicts) and builds the grammar for you.
        grammar = step.get("grammar")
        if not grammar and step.get("choices"):
            opts = " | ".join(f'"{c}"' for c in step["choices"])
            grammar = f"root ::= {opts}"
        reply = provider.complete(system, [{"role": "user", "content": prompt}], [],
                                  grammar=grammar)
        return StepResult(step["name"], "ask", reply.text.strip())

    def _run_command(self, step: dict, values: dict) -> StepResult:
        cmd = _fill(step["command"], values)
        try:
            r = subprocess.run(cmd, shell=True, cwd=str(self.workspace),
                               capture_output=True, text=True,
                               timeout=int(step.get("timeout", 300)))
        except subprocess.TimeoutExpired:
            return StepResult(step["name"], "command", "", ok=False, detail="timed out")
        out = ((r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")).strip()
        return StepResult(step["name"], "command", out[:8000],
                          ok=(r.returncode == 0), detail=f"exit {r.returncode}")

    # -- planning: one big task -> several small, checkable ones -----

    _PLAN_SYSTEM = (
        "Break the task into a numbered list of small, concrete, "
        "independently checkable sub-steps — each narrow enough to "
        "research or verify on its own before anything gets written. "
        "Every sub-step should be something you could find and quote "
        "evidence for, or clearly say you couldn't find. Never include a "
        "drafting, writing, or inventing step — sub-steps establish facts, "
        "they don't create content. 3 to 6 steps. Reply with ONLY the "
        "numbered list, one sub-step per line ('1. ...'), nothing else — "
        "no preamble, no summary."
    )

    def _run_plan(self, step: dict, values: dict) -> StepResult:
        provider = self._provider(step["model"])
        prompt = _fill(step.get("prompt", "{input}"), values)
        system = step.get("system") or self._PLAN_SYSTEM
        reply = provider.complete(system, [{"role": "user", "content": prompt}], [])
        items = [m.group(1).strip() for line in reply.text.splitlines()
                 if (m := re.match(r"\s*\d+[\.\)]\s*(.+)", line))]
        if not items:
            # Never produce zero sub-steps — degrade to the whole task as
            # one step rather than silently doing nothing.
            items = [prompt.strip()]
        values[f"__plan__{step['name']}"] = "\n".join(items)
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
        return StepResult(step["name"], "plan", listing, detail=f"{len(items)} sub-step(s)")

    def _run_foreach(self, step: dict, values: dict,
                     ask: Callable | None) -> Iterator[Any]:
        plan_name = step.get("over")
        if not plan_name:
            raise PipelineError(f"foreach step {step['name']!r} needs 'over: <plan step name>'")
        raw = values.get(f"__plan__{plan_name}")
        if raw is None:
            raise PipelineError(
                f"foreach step {step['name']!r} refers to plan {plan_name!r}, "
                f"which hasn't run yet (or isn't a plan step) — plan must "
                f"come before foreach in the step list.")
        items = [ln for ln in raw.splitlines() if ln.strip()]
        inner = step.get("steps", [])
        pieces = []
        for i, item in enumerate(items, 1):
            yield RunEvent(kind="loop_round", step=step["name"], round=i)
            # A copy per item: each sub-step gets the shared context (input,
            # earlier steps) plus its own {item}, but sub-steps don't leak
            # into each other's values — they're independent, on purpose.
            item_values = dict(values)
            item_values["item"] = item
            item_values["item_number"] = str(i)
            last: StepResult | None = None
            for sub in inner:
                for out in self._dispatch(sub, item_values, ask):
                    if isinstance(out, RunEvent):
                        yield out
                    else:
                        last = out
                        item_values[out.name] = out.output
            if last:
                pieces.append(f"### {i}. {item}\n\n{last.output}")
        combined = "\n\n".join(pieces)
        values[step["name"]] = combined
        yield StepResult(step["name"], "foreach", combined,
                         detail=f"{len(items)} sub-task(s)")

    # -- gate: a narrow judgment call that can stop the pipeline -----

    def _run_gate(self, step: dict, values: dict) -> StepResult:
        res = self._run_ask(step, values)
        pass_word = step.get("pass_word", "yes").strip().lower()
        passed = res.output.strip().lower().startswith(pass_word)
        return StepResult(step["name"], "gate", res.output, ok=passed)

    # -- the loop ----------------------------------------------------

    def _run_loop(self, step: dict, values: dict,
                  ask: Callable | None) -> Iterator[Any]:
        """
        Repeat inner steps until the exit check passes.

        Exit check, in order of preference:
          until_ok: <step name>    the named step succeeded (a command exited 0)
          until_text: <marker>     the last output contains this text
          otherwise                run max_rounds times

        Yields RunEvents and finally a StepResult.
        """
        inner = step.get("steps", [])
        max_rounds = int(step.get("max_rounds", 3))
        until_ok = step.get("until_ok")
        until_text = step.get("until_text")
        last: StepResult | None = None

        def satisfied() -> bool:
            if until_ok:
                return values.get(f"__ok__{until_ok}") == "1"
            if until_text and last:
                return until_text.lower() in last.output.lower()
            return False

        for rnd in range(1, max_rounds + 1):
            yield RunEvent(kind="loop_round", step=step["name"], round=rnd)
            values["round"] = str(rnd)
            for sub in inner:
                for item in self._dispatch(sub, values, ask):
                    if isinstance(item, RunEvent):
                        yield item
                    else:
                        last = item
                        values[item.name] = item.output
                # Check after EVERY step, not just at the end of the round.
                # Otherwise a passing check still falls through to the "fix"
                # step below it, and a model gets invited to "repair" code
                # that already works — wasted time and a real chance of
                # breaking something that was fine.
                if satisfied():
                    yield RunEvent(kind="step_done", step=step["name"],
                                   text=f"check passed on round {rnd}", ok=True)
                    yield StepResult(step["name"], "loop",
                                     (last.output if last else ""), ok=True,
                                     detail=f"passed on round {rnd}")
                    return

        detail = f"gave up after {max_rounds} rounds"
        yield RunEvent(kind="step_done", step=step["name"], text=detail, ok=False)
        yield StepResult(step["name"], "loop", (last.output if last else ""),
                         ok=(not (until_ok or until_text)), detail=detail)

    def _dispatch(self, step: dict, values: dict,
                  ask: Callable | None) -> Iterator[Any]:
        kind = step.get("kind", "agent")
        name = step.get("name") or kind
        step.setdefault("name", name)

        if kind == "loop":
            yield from self._run_loop(step, values, ask)
            return
        if kind == "foreach":
            yield RunEvent(kind="step_start", step=name, text=kind)
            res = None
            for out in self._run_foreach(step, values, ask):
                if isinstance(out, RunEvent):
                    yield out
                else:
                    res = out
            values[f"__ok__{name}"] = "1" if (res and res.ok) else "0"
            yield RunEvent(kind="step_done", step=name, ok=bool(res and res.ok),
                           text=(res.detail if res else ""))
            if res:
                yield res
            return

        yield RunEvent(kind="step_start", step=name, text=kind)
        if kind == "agent":
            res = self._run_agent(step, values, ask)
        elif kind == "ask":
            res = self._run_ask(step, values)
        elif kind == "command":
            res = self._run_command(step, values)
        elif kind == "plan":
            res = self._run_plan(step, values)
        elif kind == "gate":
            res = self._run_gate(step, values)
        else:
            raise PipelineError(f"Unknown step kind {kind!r} in step {name!r}")

        # Record success separately so a loop's until_ok can see it even
        # though prompts only ever interpolate the text output.
        values[f"__ok__{name}"] = "1" if res.ok else "0"
        yield RunEvent(kind="step_done", step=name, ok=res.ok,
                       text=(res.detail or res.output[:200]))
        yield res

    # -- entry point -------------------------------------------------

    def run(self, user_input: str, ask: Callable | None = None) -> Iterator[RunEvent]:
        values: dict[str, str] = {"input": user_input}
        results: list[StepResult] = []
        try:
            for step in self.steps:
                for item in self._dispatch(step, values, ask):
                    if isinstance(item, RunEvent):
                        yield item
                    else:
                        results.append(item)
                        values[item.name] = item.output
                        # A failed gate stops everything after it — the
                        # whole point is that "not enough real evidence"
                        # must never reach a step that writes prose.
                        if item.kind == "gate" and not item.ok:
                            yield RunEvent(kind="done", text=item.output, ok=False)
                            return
        except PipelineError as e:
            yield RunEvent(kind="error", text=str(e), ok=False)
            return
        except Exception as e:
            yield RunEvent(kind="error", text=f"{type(e).__name__}: {e}", ok=False)
            return

        final = results[-1] if results else None
        yield RunEvent(kind="done", text=(final.output if final else ""),
                       ok=all(r.ok for r in results))


_AGENT_SYSTEM = """You are a coding agent working in a project directory.

Use your tools to do real work — read files before changing them, make the
change, then verify it by running something. Don't describe what you would
do; do it, then report what happened.

Be brief. When you're done, say what you did in one or two sentences."""
