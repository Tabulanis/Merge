"""Everyday data chores — one-call versions of the operations agents
otherwise re-implement badly, over and over: sorting, deduping, diffing,
counting, stats, identifier case conversion, JSON checking, regex testing.

All pure and deterministic. Items travel as newline-separated text because
that's the shape models naturally produce. For anything bespoke (real
algorithms, transformations these don't cover) the compute tool remains
the right hammer.
"""
from __future__ import annotations

import datetime
import json
import re
import statistics

MAX_INPUT_CHARS = 200_000
MAX_OUTPUT_LINES = 500


def _lines(text: str) -> list[str]:
    return [ln for ln in (text or "").splitlines() if ln.strip()]


def _clip(lines: list[str]) -> str:
    if len(lines) > MAX_OUTPUT_LINES:
        omitted = len(lines) - MAX_OUTPUT_LINES
        lines = lines[:MAX_OUTPUT_LINES] + [f"... ({omitted} more lines omitted)"]
    return "\n".join(lines)


def _words(identifier: str) -> list[str]:
    """Split any identifier style into lowercase words."""
    s = identifier.strip()
    s = re.sub(r"[-_\s]+", " ", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w.lower() for w in s.split() if w]


def data_ops(operation: str, text: str = "", second_text: str = "",
             pattern: str = "", style: str = "snake",
             reverse: bool = False, unique: bool = False) -> str:
    if len(text) + len(second_text) > MAX_INPUT_CHARS:
        return f"Error: input too large (limit {MAX_INPUT_CHARS} characters)."

    if operation == "sort":
        items = _lines(text)
        if not items:
            return "Error: no items to sort (put one item per line in `text`)."
        if unique:
            items = list(dict.fromkeys(items))
        try:
            items.sort(key=lambda s: float(s), reverse=reverse)
            kind = "numeric"
        except ValueError:
            items.sort(key=str.lower, reverse=reverse)
            kind = "alphabetical (case-insensitive)"
        return _clip([f"# {len(items)} items, {kind}"
                      + (", duplicates removed" if unique else "")] + items)

    if operation == "dedupe":
        items = _lines(text)
        if not items:
            return "Error: no items (one per line in `text`)."
        kept = list(dict.fromkeys(items))
        return _clip([f"# {len(items)} -> {len(kept)} items "
                      f"({len(items) - len(kept)} duplicates removed, first "
                      f"occurrence kept, order preserved)"] + kept)

    if operation == "count":
        items = _lines(text)
        if not items:
            return "Error: no items (one per line in `text`)."
        freq: dict[str, int] = {}
        for it in items:
            freq[it] = freq.get(it, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return _clip([f"# {len(items)} items, {len(ranked)} distinct"]
                     + [f"{n:6d}  {item}" for item, n in ranked])

    if operation == "diff":
        a, b = _lines(text), _lines(second_text)
        if not a and not b:
            return "Error: both lists empty (text = first list, second_text = second)."
        sa, sb = set(a), set(b)
        only_a = [x for x in a if x not in sb]
        only_b = [x for x in b if x not in sa]
        both = len([x for x in dict.fromkeys(a) if x in sb])
        out = [f"# first list: {len(a)} items, second: {len(b)}, shared: {both}",
               f"# only in FIRST ({len(only_a)}):"] + [f"  - {x}" for x in only_a]
        out += [f"# only in SECOND ({len(only_b)}):"] + [f"  + {x}" for x in only_b]
        return _clip(out)

    if operation == "stats":
        raw = _lines(text)
        try:
            nums = [float(x) for x in raw]
        except ValueError as e:
            return f"Error: stats needs one number per line ({e})."
        if not nums:
            return "Error: no numbers (one per line in `text`)."
        out = [f"count:  {len(nums)}",
               f"sum:    {sum(nums):g}",
               f"min:    {min(nums):g}",
               f"max:    {max(nums):g}",
               f"mean:   {statistics.fmean(nums):g}",
               f"median: {statistics.median(nums):g}"]
        if len(nums) > 1:
            out.append(f"stdev:  {statistics.stdev(nums):g} (sample)")
        return "\n".join(out)

    if operation == "case":
        items = _lines(text)
        if not items:
            return "Error: no identifiers (one per line in `text`)."
        def convert(name: str) -> str:
            w = _words(name)
            if not w:
                return name
            if style == "snake":
                return "_".join(w)
            if style == "kebab":
                return "-".join(w)
            if style == "constant":
                return "_".join(x.upper() for x in w)
            if style == "camel":
                return w[0] + "".join(x.capitalize() for x in w[1:])
            if style == "pascal":
                return "".join(x.capitalize() for x in w)
            return name
        if style not in ("snake", "kebab", "constant", "camel", "pascal"):
            return "Error: style must be snake, kebab, constant, camel, or pascal."
        return _clip([f"{name}  ->  {convert(name)}" for name in items])

    if operation == "json_check":
        if not text.strip():
            return "Error: nothing to check (JSON goes in `text`)."
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            snippet = text[max(0, e.pos - 40):e.pos + 40].replace("\n", "\\n")
            return (f"INVALID JSON: {e.msg} at line {e.lineno}, column {e.colno}.\n"
                    f"Near: ...{snippet}...")
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        if len(pretty) > 20_000:
            return f"Valid JSON ({type(parsed).__name__}); too large to echo back."
        return "Valid JSON. Pretty-printed:\n" + pretty

    if operation == "regex":
        if not pattern:
            return "Error: put the regular expression in `pattern` and the text to search in `text`."
        try:
            rx = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            return f"INVALID PATTERN: {e}"
        matches = list(rx.finditer(text))
        if not matches:
            return f"Pattern is valid; 0 matches in {len(text)} characters."
        out = [f"Pattern is valid; {len(matches)} match(es):"]
        for m in matches[:100]:
            line_no = text.count("\n", 0, m.start()) + 1
            shown = m.group(0)[:120]
            out.append(f"line {line_no}: {shown!r}"
                       + (f"  groups={m.groups()}" if m.groups() else ""))
        if len(matches) > 100:
            out.append(f"... ({len(matches) - 100} more)")
        return _clip(out)

    return ("Error: unknown operation. Choose: sort, dedupe, count, diff, "
            "stats, case, json_check, regex.")


def date_calc(operation: str, date: str = "", second_date: str = "",
              days: int = 0) -> str:
    """Calendar math — the classic confident-wrong-answer zone for models.
    Dates are ISO format (2026-08-12)."""
    def parse(s: str, which: str) -> datetime.date:
        try:
            return datetime.date.fromisoformat(s.strip())
        except ValueError:
            raise ValueError(f"{which} must be an ISO date like 2026-08-12, got {s!r}")

    today = datetime.date.today()
    try:
        if operation == "today":
            return (f"Today is {today.isoformat()} ({today.strftime('%A')}), "
                    f"day {today.timetuple().tm_yday} of {today.year}.")
        if operation == "weekday":
            d = parse(date, "date")
            return f"{d.isoformat()} is a {d.strftime('%A')}."
        if operation == "diff":
            a = parse(date, "date")
            b = parse(second_date, "second_date") if second_date.strip() else today
            n = (b - a).days
            direction = "later than" if n >= 0 else "earlier than"
            return (f"{b.isoformat()} is {abs(n)} days {direction} {a.isoformat()} "
                    f"({abs(n) / 7:.1f} weeks, {abs(n) / 365.2425:.2f} years).")
        if operation == "add":
            d = parse(date, "date") if date.strip() else today
            r = d + datetime.timedelta(days=int(days))
            sign = "+" if days >= 0 else ""
            return (f"{d.isoformat()} {sign}{days} days = {r.isoformat()} "
                    f"({r.strftime('%A')}).")
    except ValueError as e:
        return f"Error: {e}"
    return "Error: unknown operation. Choose: today, weekday, diff, add."
