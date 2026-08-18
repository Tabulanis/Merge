"""A headless browser Merge can drive and SEE.

Playwright chromium, kept warm in ONE dedicated worker thread — Playwright's
sync objects are thread-bound, so every browser op runs in that single thread
and other threads post commands to it over a queue. That lets one browser
persist across turns (navigate in one message, click in the next) without
tripping over the "different thread" error.

The point is bug-finding the way a person does it: open the page, LOOK at it
(she screenshots and reads it with her own vision), read the console for the
errors that only show when something actually runs, poke it with JS.
"""
from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path

_SHOT_DIR = Path(tempfile.gettempdir()) / "merge-browser"


class _Browser:
    def __init__(self) -> None:
        self._cmds: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._lock = threading.Lock()
        self._console: list[str] = []

    def _start(self) -> None:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def call(self, op: str, *args, timeout: float = 60):
        """Post an op to the worker thread and wait for its result."""
        self._start()
        box: queue.Queue = queue.Queue(1)
        self._cmds.put((op, args, box))
        ok, val = box.get(timeout=timeout)
        if not ok:
            raise RuntimeError(val)
        return val

    # ---- everything below runs ONLY in the worker thread ----------------
    def _run(self) -> None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            self._pw = pw
            self._browser = None
            self._page = None
            while True:
                op, args, box = self._cmds.get()
                try:
                    box.put((True, getattr(self, "_op_" + op)(*args)))
                except Exception as e:  # never let one bad op kill the thread
                    box.put((False, f"{type(e).__name__}: {e}"))

    def _page_ready(self):
        # Relaunch if chromium died under us.
        if self._browser is not None and not self._browser.is_connected():
            self._browser, self._page = None, None
        if self._browser is None:
            self._browser = self._pw.chromium.launch(headless=True)
        if self._page is None:
            self._page = self._browser.new_page(ignore_https_errors=True)
            self._page.on("console", lambda m: self._console.append(f"{m.type}: {m.text}"[:600]))
            self._page.on("pageerror", lambda e: self._console.append(f"pageerror: {e}"[:600]))
        return self._page

    def _op_goto(self, url: str) -> dict:
        pg = self._page_ready()
        self._console.clear()
        pg.goto(url, wait_until="load", timeout=30000)
        pg.wait_for_timeout(500)                      # let scripts + first paint settle
        errs = [m for m in self._console if m.startswith(("pageerror", "error"))]
        return {"title": pg.title(), "url": pg.url,
                "text": (pg.inner_text("body") or "")[:3000],
                "console_errors": errs[:25]}

    def _op_shot(self, full: bool = False) -> str:
        pg = self._page_ready()
        _SHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = _SHOT_DIR / f"shot-{time.time_ns()}.png"
        pg.screenshot(path=str(path), full_page=bool(full))
        return str(path)

    def _op_js(self, code: str):
        return self._page_ready().evaluate(code)

    def _op_console(self):
        return list(self._console)[-60:]

    def _op_close(self) -> str:
        if self._page:
            self._page.close()
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        self._console.clear()
        return "browser closed"


_BROWSER = _Browser()


def goto(url: str) -> dict:
    return _BROWSER.call("goto", url)


def screenshot(full: bool = False) -> str:
    return _BROWSER.call("shot", full)


def run_js(code: str):
    return _BROWSER.call("js", code)


def console_log() -> list[str]:
    return _BROWSER.call("console")


def close() -> str:
    return _BROWSER.call("close")
