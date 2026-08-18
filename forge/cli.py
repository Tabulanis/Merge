"""
The terminal interface — what you actually talk to.

Deliberately close to Claude Code's feel: you're dropped into your project
folder, you type in plain language, tools run with your say-so, and slash
commands handle the meta stuff.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from .config import (CONFIG_PATH, active_model_config, load_config,
                     load_pipelines, save_config)
from .doctor import FAIL, OK, WARN, report, run_checks
from .help_content import ORDER, TOPICS, WELCOME, search, topic
from .media import capabilities, load_media_config
from . import recall
from .pipeline import Pipeline
from .session import SESS_DIR, Session

console = Console()

BANNER = """[bold cyan]FORGE[/bold cyan] — your own coding agent
workspace: [dim]{ws}[/dim]
model: [bold]{model}[/bold] [dim]({provider})[/dim]   permissions: [bold]{perm}[/bold]

[dim]/help  anything you're unsure about · /doctor  if something's broken
/model  switch brains · /sessions  earlier chats · Ctrl-D  quit[/dim]"""

def show_help(arg: str = "") -> None:
    """`/help`, `/help models`, or `/help why is it slow` — all one door."""
    if not arg:
        console.print("\n[bold cyan]Forge help[/bold cyan]  "
                      "[dim]— /help <topic>, or just ask: /help why is it slow[/dim]\n")
        for key in ORDER:
            console.print(f"  [cyan]{key:<11}[/cyan] {TOPICS[key]['blurb']}")
        console.print("\n[dim]Stuck right now? [cyan]/doctor[/cyan] checks your setup "
                      "and names the fix.[/dim]")
        return

    found = topic(arg)
    if found:
        _print_topic(found[1])
        return

    # not a topic name — treat it as a question
    hits = search(arg)
    if not hits:
        console.print(f"[dim]Nothing about {arg!r}. Topics:[/dim] "
                      + ", ".join(f"[cyan]{k}[/cyan]" for k in ORDER))
    elif len(hits) == 1:
        _print_topic(hits[0][1])
    else:
        console.print(f"\n[dim]{len(hits)} topics mention that:[/dim]")
        for key, t in hits:
            console.print(f"  [cyan]{key:<11}[/cyan] {t['blurb']}")
        console.print("\n[dim]Read one with /help <name>[/dim]")


def _print_topic(t: dict) -> None:
    console.print(f"\n[bold cyan]{t['title']}[/bold cyan]\n")
    console.print(escape(t["body"]))


def show_doctor() -> None:
    """Check everything, and name the fix for whatever is broken."""
    console.print("\n[bold cyan]Checking your setup...[/bold cyan]\n")
    checks = run_checks()
    mark = {OK: "[green] OK [/green]", WARN: "[yellow]MEH [/yellow]",
            FAIL: "[red]BAD [/red]"}
    for c in checks:
        console.print(f"  {mark[c.status]} [bold]{c.name}[/bold] — {escape(c.detail)}")
        if c.fix:
            console.print(f"        [dim]fix:[/dim] [cyan]{escape(c.fix)}[/cyan]")
    bad, meh = report(checks)
    console.print()
    if bad:
        console.print(f"[red]{bad} thing(s) need fixing[/red] — run the "
                      f"[cyan]fix:[/cyan] line shown under each.")
    elif meh:
        console.print("[green]Nothing is broken.[/green] "
                      f"[dim]{meh} optional extra(s) not set up — that's fine.[/dim]")
    else:
        console.print("[green]Everything's working.[/green]")


def show_power_off() -> None:
    from . import power
    res = power.off()
    if res["stopped"]:
        console.print(f"[green]stopped:[/green] {', '.join(res['stopped'])}")
    else:
        console.print("[dim]no local models were running[/dim]")
    if res["vram"]:
        console.print(f"[dim]graphics memory now: {res['vram']}[/dim]")
    console.print("[dim]wake her later with[/dim] [cyan]forge on[/cyan] "
                  "[dim](or the Power card in the dashboard)[/dim]")


def show_power_on(which: str = "big") -> None:
    import time
    from . import power
    if power.is_ready(which):
        console.print(f"[green]{which} is already up.[/green] "
                      "[dim]cd into your project and type[/dim] [cyan]merge[/cyan]")
        return
    res = power.on(which)
    if res.get("error"):
        console.print(f"[red]{escape(res['error'])}[/red]")
        return

    from rich.progress import (BarColumn, Progress, TextColumn,
                               TimeElapsedColumn)
    # For the big model the bar is real: the weights fill the graphics card
    # as they load, so VRAM growth IS the progress. CPU models give no such
    # signal — their bar paces itself on a typical load time instead.
    base = power.vram_mb()
    expect = power.EXPECTED_LOAD_MB.get(which)
    started = time.monotonic()
    try:
        with Progress(TextColumn(f"[cyan]waking {which}[/cyan]"),
                      BarColumn(bar_width=30),
                      TextColumn("{task.percentage:>3.0f}%"),
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("load", total=100)
            while not power.is_ready(which):
                if expect and base is not None:
                    used = power.vram_mb()
                    pct = ((used - base) / expect * 100) if used else 0
                else:
                    pct = (time.monotonic() - started) / 60 * 100
                # hold just short of full until she actually answers
                prog.update(task, completed=max(1.0, min(pct, 97.0)))
                if time.monotonic() - started > 300:
                    prog.stop()
                    console.print("[yellow]five minutes and still loading — "
                                  "something's wrong. run[/yellow] "
                                  "[cyan]forge doctor[/cyan]")
                    return
                time.sleep(1)
            prog.update(task, completed=100)
    except KeyboardInterrupt:
        console.print("\n[dim]she keeps loading in the background — "
                      "run[/dim] [cyan]forge doctor[/cyan] [dim]to check on her[/dim]")
        return
    console.print("[green]she's ready.[/green] [dim]cd into your project "
                  "and type[/dim] [cyan]merge[/cyan]")


def _session_files() -> list[Path]:
    """Saved conversations, newest first — the one list both doors share."""
    try:
        return sorted(SESS_DIR.glob("*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def open_session(cfg: dict, workspace: Path, resume: bool) -> Session:
    """
    A conversation IS a Session — the same object the web chat uses, saved
    to the same folder. That's the whole one-brain-two-doors trick: this
    terminal and the browser drawer are just two views of ~/.forge/sessions/.
    """
    if resume:
        for f in _session_files():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if Path(d.get("workspace", "")) != workspace:
                continue
            s = Session.load(f, cfg)
            if s:
                console.print(f"[dim]continuing your last chat here "
                              f"({len(s.agent.history)} earlier entries) — "
                              f"also visible in the web drawer[/dim]")
                return s
        console.print("[dim]no earlier chat in this folder — starting fresh[/dim]")
    return Session(uuid.uuid4().hex[:12], workspace, cfg)


def ask_permission(tool_name: str, args: dict, summary: str) -> bool:
    # For a command, the box IS the description — repeating it in the header
    # just makes the same string appear twice on screen.
    if tool_name == "run_command":
        console.print("\n[yellow]▸ wants to run this command:[/yellow]")
        console.print(Panel(args.get("command", ""), border_style="yellow", expand=False))
    else:
        console.print(f"\n[yellow]▸ wants to {summary}[/yellow]")
    try:
        answer = console.input("[bold]allow?[/bold] [dim](y/n)[/dim] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes", "")


def print_events(sess: Session, message: str) -> None:
    """Drive one turn: render it as it happens, and mirror every event into
    the session log — so the web page can show this exact conversation."""
    sess.emit("user", {"text": message})
    final_text = ""
    try:
        for ev in sess.agent.run(message, ask=ask_permission):
            if ev.kind == "text" and ev.text.strip():
                final_text = ev.text
                console.print()
                console.print(Markdown(ev.text))
                sess.emit("text", {"text": ev.text})
            elif ev.kind == "tool_request":
                # A permission prompt is about to describe this action in
                # full — no need to also whisper it here first.
                if not ev.will_ask:
                    console.print(f"[dim]  · {escape(ev.summary)}[/dim]")
                sess.emit("tool", {"tool": ev.tool, "summary": ev.summary})
            elif ev.kind == "tool_result":
                sess.emit("result", {"tool": ev.tool,
                                     "text": (ev.text or "")[:2000]})
                if ev.text == "declined":
                    console.print("[red]  · declined[/red]")
                    continue
                lines = (ev.text or "").strip().splitlines()
                # Show a couple of lines of real output, not just the first —
                # for a command, line one is "[exit 0]" and the interesting
                # part is what follows.
                for ln in lines[:3]:
                    console.print(f"[dim]    {escape(ln[:110])}[/dim]")
                if len(lines) > 3:
                    console.print(f"[dim]    … {len(lines) - 3} more line(s)[/dim]")
            elif ev.kind == "note":
                console.print(f"[yellow italic]  {escape(ev.text)}[/yellow italic]")
                sess.emit("note", {"text": ev.text})
            elif ev.kind == "error":
                console.print(f"[red]{escape(ev.text)}[/red]")
                sess.emit("error", {"text": ev.text})
                # A dead model server is the single most common failure, and
                # the raw exception says nothing useful to someone new.
                if "connect" in ev.text.lower() or "refused" in ev.text.lower():
                    console.print("[dim]The model doesn't seem to be running. Try:[/dim] "
                                  "[cyan]merge on[/cyan]  "
                                  "[dim]or run[/dim] [cyan]/doctor[/cyan]")
            elif ev.kind == "done":
                sess.emit("done", {"usage": ev.usage or {}})
                u = ev.usage or {}
                if u.get("input_tokens") or u.get("output_tokens"):
                    console.print(
                        f"[dim]  ({u.get('input_tokens', 0)} in / "
                        f"{u.get('output_tokens', 0)} out)[/dim]"
                    )
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        sess.emit("note", {"text": "interrupted at the terminal"})
    finally:
        sess.last_used = time.time()
        sess.save()
        # the librarian files an index card in the background
        recall.remember_turn(message, final_text, str(sess.workspace), sess.id)


def show_sessions(current_id: str) -> list[Path]:
    """List saved chats, newest first. Returns the files in the order shown,
    so /resume <number> means the same thing the eye just read."""
    files = _session_files()[:15]
    if not files:
        console.print("[dim]no saved chats yet[/dim]")
        return files
    for i, f in enumerate(files, 1):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        first = next((e.get("text", "") for e in d.get("log", [])
                      if e.get("kind") == "user"), "")
        mark = "[bold green]●[/bold green]" if d.get("id") == current_id else " "
        when = time.strftime("%b %d %H:%M", time.localtime(d.get("last_used", 0)))
        folder = Path(d.get("workspace", "")).name or "?"
        console.print(f" {mark} [cyan]{i:>2}[/cyan] [dim]{when}[/dim] "
                      f"[bold]{folder[:18]:<18}[/bold] {escape(first[:46]) or '(empty)'}")
    console.print("[dim]same list as the web drawer — /resume <number> "
                  "picks one up[/dim]")
    return files


def handle_command(line: str, cfg: dict, workspace: Path, sess: Session) -> tuple[bool, Session]:
    """Returns (should_continue, session) — the session may be swapped by
    /resume or rebuilt on a model switch."""
    parts = line.strip().split()
    cmd, args = parts[0], parts[1:]
    agent = sess.agent

    if cmd in ("/quit", "/exit"):
        return False, sess

    if cmd == "/off":
        show_power_off()
        return False, sess

    if cmd == "/help":
        show_help(" ".join(args))

    elif cmd == "/doctor":
        show_doctor()

    elif cmd == "/config":
        console.print(f"[dim]{CONFIG_PATH}[/dim]")

    elif cmd == "/tools":
        for t in agent.tools.values():
            lock = "[yellow]asks first[/yellow]" if t.needs_permission else "[dim]safe[/dim]"
            console.print(f"  [cyan]{t.name}[/cyan] — {t.description.splitlines()[0]} ({lock})")

    elif cmd == "/clear":
        agent.history.clear()
        sess.log.clear()
        sess.log_base = 0
        sess.save()
        console.print("[dim]context cleared[/dim]")

    elif cmd == "/sessions":
        show_sessions(sess.id)

    elif cmd == "/resume":
        files = _session_files()[:15]
        if not args or not args[0].isdigit() or not (1 <= int(args[0]) <= len(files)):
            console.print("[dim]usage: /resume <number> — see the numbers "
                          "with /sessions[/dim]")
        else:
            picked = Session.load(files[int(args[0]) - 1], cfg)
            if not picked:
                console.print("[red]couldn't load that one — the file is "
                              "damaged[/red]")
            else:
                sess.save()
                console.print(f"[green]picked up the chat in "
                              f"{picked.workspace}[/green] "
                              f"[dim]({len(picked.agent.history)} earlier entries)[/dim]")
                return True, picked

    elif cmd == "/model":
        if not args:
            for name, m in cfg["models"].items():
                mark = "[bold green]●[/bold green]" if name == cfg["active_model"] else " "
                console.print(f" {mark} [cyan]{name}[/cyan] — {m.get('model')} "
                              f"[dim]({m.get('provider')})[/dim]")
        elif args[0] not in cfg["models"]:
            console.print(f"[red]No model named {args[0]!r}. "
                          f"Try: {', '.join(cfg['models'])}[/red]")
        else:
            cfg["active_model"] = args[0]
            save_config(cfg)
            sess.reload_model(cfg)
            console.print(f"[green]switched to {args[0]}[/green]")

    elif cmd == "/perm":
        if not args or args[0] not in ("ask", "auto", "deny"):
            console.print(f"[dim]permission mode is [bold]{agent.permission_mode}[/bold] "
                          f"— use ask, auto, or deny[/dim]")
        else:
            agent.permission_mode = args[0]
            cfg["agent"]["permission_mode"] = args[0]
            save_config(cfg)
            console.print(f"[green]permissions: {args[0]}[/green]")

    elif cmd == "/media":
        caps = capabilities(load_media_config(cfg))
        for name, c in caps.items():
            mark = "[green]●[/green]" if c["ok"] else "[dim]○[/dim]"
            console.print(f"  {mark} [cyan]{name}[/cyan] — {escape(c['detail'])}")

    elif cmd == "/flows":
        flows = load_pipelines()
        if not flows:
            console.print("[dim]no recipes defined[/dim]")
        for name, spec in flows.items():
            console.print(f"  [cyan]{name}[/cyan] — {spec.get('description', '')}")
            for st in spec.get("steps", []):
                k = st.get("kind", "agent")
                extra = (f" ×{st.get('max_rounds', 3)} until {st.get('until_ok') or st.get('until_text')}"
                         if k == "loop" else f" [{st.get('model', st.get('command', ''))[:40]}]")
                console.print(f"      [dim]{k}: {st.get('name')}{extra}[/dim]")

    elif cmd == "/run":
        flows = load_pipelines()
        if not args:
            console.print("[dim]usage: /run <recipe> <what you want done>[/dim]")
        elif args[0] not in flows:
            console.print(f"[red]No recipe {args[0]!r}. Try /flows.[/red]")
        elif len(args) < 2:
            console.print("[dim]tell it what to do: /run <recipe> <task>[/dim]")
        else:
            run_pipeline(args[0], flows[args[0]], " ".join(args[1:]), cfg, workspace)

    else:
        console.print(f"[red]Unknown command {cmd}. /help for the list.[/red]")

    return True, sess


def run_pipeline(name: str, spec: dict, task: str, cfg: dict, workspace: Path) -> None:
    """Run one orchestration recipe, printing progress as it goes."""
    spec = {**spec, "name": name}
    try:
        pipe = Pipeline(spec, cfg["models"], workspace,
                        permission_mode=cfg["agent"].get("permission_mode", "ask"))
    except Exception as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        return

    console.print(f"\n[bold cyan]▶ {name}[/bold cyan] [dim]{spec.get('description','')}[/dim]")
    try:
        for ev in pipe.run(task, ask=ask_permission):
            if ev.kind == "step_start":
                console.print(f"\n[bold]· {ev.step}[/bold] [dim]({ev.text})[/dim]")
            elif ev.kind == "loop_round":
                console.print(f"\n[yellow]  ↻ {ev.step} — round {ev.round}[/yellow]")
            elif ev.kind == "step_done":
                mark = "[green]✓[/green]" if ev.ok else "[red]✗[/red]"
                first = (ev.text or "").strip().splitlines()
                console.print(f"  {mark} [dim]{escape(first[0][:100]) if first else ''}[/dim]")
            elif ev.kind == "error":
                console.print(f"[red]{escape(ev.text)}[/red]")
                # A dead model server is the single most common failure, and
                # the raw exception says nothing useful to someone new.
                if "connect" in ev.text.lower() or "refused" in ev.text.lower():
                    console.print("[dim]The model doesn't seem to be running. Try:[/dim] "
                                  "[cyan]~/forge/start-model.sh big[/cyan]  "
                                  "[dim]or run[/dim] [cyan]/doctor[/cyan]")
            elif ev.kind == "done":
                mark = "[green]finished[/green]" if ev.ok else "[yellow]finished with problems[/yellow]"
                console.print(f"\n{mark}")
                if ev.text.strip():
                    console.print(Markdown(ev.text[:1500]))
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="forge",
        description="Your own coding agent. Try:  forge help",
        epilog="forge help          what everything does\n"
               "forge doctor        check your setup and name the fixes\n"
               "forge off           stop the local models, free the GPU\n"
               "forge on            start the big model again",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="*",
                    help="Run one message and exit, or: help / doctor")
    ap.add_argument("-w", "--workspace", default=".", help="Project directory")
    ap.add_argument("-m", "--model", help="Model config to use for this run")
    ap.add_argument("--auto", action="store_true", help="Skip permission prompts")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="Pick up your last chat in this folder")
    args = ap.parse_args()

    # `forge help` and `forge doctor` must work even when nothing else does —
    # they're what you reach for precisely when the setup is broken, so they
    # run before any model or workspace is touched.
    if args.message and args.message[0].lower() in ("help", "--topics"):
        show_help(" ".join(args.message[1:]))
        return
    if args.message and args.message[0].lower() == "doctor":
        show_doctor()
        return
    # Power belongs in this pre-flight group too: `forge off` must work
    # even when the model it would talk to is already broken or gone.
    if args.message and args.message[0].lower() in ("off", "sleep"):
        show_power_off()
        return
    if args.message and args.message[0].lower() in ("on", "wake"):
        show_power_on(args.message[1] if len(args.message) > 1 else "big")
        return

    first_run = not CONFIG_PATH.exists()
    cfg = load_config()
    if first_run:
        console.print(Panel(WELCOME, title="[bold cyan]Welcome to Forge[/bold cyan]",
                            border_style="cyan", expand=False))
    if args.model:
        if args.model not in cfg["models"]:
            console.print(f"[red]No model named {args.model!r}[/red]")
            sys.exit(1)
        cfg["active_model"] = args.model
    if args.auto:
        cfg["agent"]["permission_mode"] = "auto"

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        console.print(f"[red]Not a directory: {workspace}[/red]")
        sys.exit(1)
    # Running from the home folder fences the agent to everything you own —
    # which is how a chat about 3D printers left a stray file in ~. The web
    # chat already defaults to the Playground; the terminal now does too.
    # Saying -w ~ out loud still works for whoever really means it.
    if workspace == Path.home() and args.workspace == ".":
        workspace = Path.home() / "Merge"
        workspace.mkdir(exist_ok=True)
        console.print("[dim]the home folder is everything you own, so this "
                      "chat lives in ~/Merge (her workshop) instead "
                      "(run with -w ~ if you really mean home)[/dim]")

    mcfg = active_model_config(cfg)
    if mcfg.get("provider") == "anthropic" and not (mcfg.get("api_key")
                                                    or os.environ.get("ANTHROPIC_API_KEY")):
        console.print("[red]No Anthropic API key found.[/red]")
        console.print("Set one with:  [cyan]export ANTHROPIC_API_KEY=sk-ant-...[/cyan]")
        console.print(f"or put it in [dim]{CONFIG_PATH}[/dim] under models.claude.api_key")
        sys.exit(1)

    try:
        sess = open_session(cfg, workspace, args.cont)
    except Exception as e:
        console.print(f"[red]Couldn't start: {e}[/red]")
        sys.exit(1)

    # One-shot mode: `forge "fix the tests"` runs and exits — saved like
    # any other chat, so even a one-liner can be picked up later.
    if args.message:
        print_events(sess, " ".join(args.message))
        return

    console.print(Panel(
        BANNER.format(ws=workspace, model=cfg["active_model"],
                      provider=mcfg.get("provider"),
                      perm=cfg["agent"].get("permission_mode")),
        border_style="cyan", expand=False,
    ))

    while True:
        try:
            line = console.input("\n[bold cyan]›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not line:
            continue
        if line.startswith("/"):
            cont, sess = handle_command(line, cfg, workspace, sess)
            if not cont:
                console.print("[dim]bye[/dim]")
                break
            continue
        # Re-read config each turn so dashboard changes land without a restart.
        fresh = load_config()
        if fresh["active_model"] != cfg["active_model"]:
            cfg = fresh
            sess.reload_model(cfg)
            console.print(f"[dim](model changed to {cfg['active_model']} from the dashboard)[/dim]")
        print_events(sess, line)


if __name__ == "__main__":
    main()
