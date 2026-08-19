"""
Legal citation verification — the hallucination guard for case law.

The thing that gets real lawyers sanctioned is citing cases that do not
exist. Language models invent them constantly and sound confident about it.
This module exists so that no case name ever leaves our output without first
being checked against a real database.

Backend: CourtListener REST API v4 (free, no key). US courts only.
The user has verified the endpoint behavior; the code below follows their
verified spec exactly:

  GET https://www.courtlistener.com/api/rest/v4/search/?q=<query>&type=o
  - MUST send a User-Agent header or the request 000s.
  - A real case returns {"count": N>0, "results":[{caseName, citation[],
    court, dateFiled, absolute_url, docketNumber}, ...]}.
  - An invented case ("Thompson v. Zephyr Dynamics") returns count: 0.
    That zero is the hallucination detector.

Rules this module enforces by construction:
  - NEVER guess. A network failure says "the check could not run" — it does
    not fall back to what the model thinks the answer is.
  - count: 0 means NOT FOUND and must not be cited.
  - This is legal INFORMATION, not advice. Jurisdiction matters; law
    changes, so anything time-sensitive gets checked fresh.
  - CourtListener covers US courts only. Foreign law = say so plainly, give
    general principles, and defer to a local source. Never fake confidence.
"""

from __future__ import annotations

import re

import httpx

_API = "https://www.courtlistener.com/api/rest/v4/search/"
_UA = "Mozilla/5.0 (research)"
_TIMEOUT = 20  # seconds; a hung connection means "could not run", not a guess
_URL_PREFIX = "https://www.courtlistener.com"

DISCLAIMER = (
    "Legal information, not legal advice. Nothing here creates an "
    "attorney-client relationship. Jurisdiction matters, and law changes — "
    "anything time-sensitive must be checked fresh against current law. "
    "This tool covers US courts (CourtListener) only; for foreign law, "
    "treat anything said as general principles and verify with a local "
    "source. Verify every case before citing it; never cite one that has "
    "not been verified."
)


def _search(query: str) -> dict:
    """One search round-trip. Returns {'ok': bool, 'data': dict|None, 'error': str}."""
    params = {"q": query, "type": "o"}
    try:
        resp = httpx.get(_API, params=params, headers={"User-Agent": _UA},
                         timeout=_TIMEOUT)
    except httpx.HTTPError as exc:
        return {"ok": False, "data": None, "error": f"{type(exc).__name__}: {exc}"}
    if resp.status_code != 200:
        return {"ok": False, "data": None,
                "error": f"HTTP {resp.status_code} from CourtListener"}
    try:
        return {"ok": True, "data": resp.json(), "error": ""}
    except ValueError as exc:
        return {"ok": False, "data": None, "error": f"unparseable response: {exc}"}


def _fmt_hit(i: int, r: dict) -> str:
    """One formatted result line."""
    name = r.get("caseName") or "(no name given)"
    cites = r.get("citation") or []
    cite = "; ".join(c for c in cites if c) if cites else "(no citation)"
    court = r.get("court") or "(court not stated)"
    date = r.get("dateFiled") or "(date not stated)"
    docket = r.get("docketNumber") or ""
    url = r.get("absolute_url") or ""
    if url and not url.startswith("http"):
        url = _URL_PREFIX + url
    line = f"{i}. {name} — {cite} — {court}, {date}"
    if docket:
        line += f" [docket {docket}]"
    if url:
        line += f"\n   {url}"
    return line


def _fmt_not_found(query: str) -> str:
    return (
        f"NOT FOUND — no case matching \"{query}\" came back from CourtListener "
        "(count: 0).\n"
        "This case must not be cited. It may not exist — invented case names "
        "return exactly this result, which is how the check catches "
        "hallucinated citations. Do not cite it, do not paraphrase it into "
        "existence, and do not soften this into 'a case like...'. If you "
        "believe a real case was intended, find its actual name and verify "
        "that instead.\n"
        + DISCLAIMER
    )


def _norm(s: str) -> str:
    """Normalize a case name for comparison.

    Lowercase, collapse all whitespace runs to a single space, strip
    punctuation (periods, commas, hyphens, apostrophes, quotes), and map
    every form of "versus" ("versus", "vs.", "vs", "v.") to the bare word
    "v". So "Miranda v. Arizona" and "Miranda v Arizona" normalize to the
    same string. Stdlib only.
    """
    s = s.lower()
    s = s.replace("versus", " v ")
    s = s.replace("vs.", " v ").replace("vs", " v ")
    s = s.replace("v.", " v ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _is_name_match(query: str, rec: dict) -> bool:
    """Does this record actually match the case name we asked for?

    CourtListener's q= field is fuzzy — even a wholly invented case name
    returns partial matches ("Thompson v. Zephyr Dynamics" comes back with
    Duckworth v. Thompson). So a nonzero count is NOT proof the case
    exists. A record only counts as a hit when the case name we asked for
    appears inside the record's caseName.
    """
    q = _norm(query)
    name = _norm(rec.get("caseName") or "")
    if not q or not name:
        return False
    # A bare fragment ("Smith", "In re", "United States") is a substring of
    # thousands of real case names but is NOT itself a case — accepting it
    # defeats the whole anti-hallucination guard. Require a real case-shaped
    # query: it carries a party separator ("v"), OR it covers most of the
    # matched name, OR it's an exact match.
    if q == name:
        return True
    has_vs = "v" in re.split(r"[^a-z]+", q.lower())  # a 'versus' token
    substantial = len(q) >= 8 and (q in name) and len(q) >= 0.6 * len(name)
    vs_match = has_vs and len(q) >= 8 and (q in name or name in q)
    return substantial or vs_match


def _first_exact(query: str, results: list) -> dict | None:
    """The first record that is genuinely the case named in the query."""
    for rec in results:
        if _is_name_match(query, rec):
            return rec
    return None


def verify_case(query: str) -> str:
    """Verify that a case exists before it is cited.

    VERIFIED   -> the real caseName / citation / court / date / url.
    NOT FOUND  -> a blunt line: must not be cited, may not exist.
    CHECK FAILED -> the network died; the check could not run. Never a guess.

    A nonzero fuzzy-match count from unrelated cases does NOT count as
    verification: the record's caseName must actually contain the case we
    asked for, otherwise the answer is NOT FOUND.
    """
    if not isinstance(query, str) or not query.strip():
        return ("CHECK FAILED — empty query. Give a case name or citation to "
                "verify; nothing was checked, and nothing is verified.")
    query = query.strip()
    res = _search(query)
    if not res["ok"]:
        return (
            "CHECK FAILED — the network check could not run "
            f"({res['error']}).\n"
            "No case has been verified here. Do not cite any case from memory "
            "in place of this check — either retry when connectivity is back "
            "or treat the citation as unverified.\n"
            + DISCLAIMER
        )
    data = res["data"]
    count = data.get("count", 0)
    results = data.get("results") or []
    exact = _first_exact(query, results)
    if exact is None:
        note = (
            f"Note: the search returned {count} record(s) that only partially "
            "match (e.g. sharing one party name) — none of them is the case "
            "you asked for."
            if count > 0
            else "No matching records at all."
        )
        return (
            f'NOT FOUND — no case named "{query}" exists in CourtListener. '
            f"{note} It must not be cited; it may not exist at all. Do not "
            "cite it, and do not soften this into a plausible-sounding "
            "citation from memory.\n"
            + DISCLAIMER
        )
    lines = [
        f'VERIFIED — "{query}" resolves to a real case in CourtListener '
        f"({count} matching record{'s' if count != 1 else ''}).",
        _fmt_hit(1, exact),
        "",
    ]
    if count > 1:
        lines.append(
            f"Note: {count - 1} further matching record(s) exist — if you "
            "meant a different court or year, say so and re-verify that one "
            "before citing it."
        )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)

def find_cases(topic: str, limit: int = 5) -> str:
    """Search by topic and list real hits with citations and urls."""
    if not isinstance(topic, str) or not topic.strip():
        return ("No results — empty topic. Give a topic or subject to search "
                "for; nothing was searched.")
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 5
    res = _search(topic.strip())
    if not res["ok"]:
        return (
            "SEARCH FAILED — the network check could not run "
            f"({res['error']}).\n"
            "Nothing was returned and nothing is verified. Retry when "
            "connectivity is back.\n"
            + DISCLAIMER
        )
    data = res["data"]
    count = data.get("count", 0)
    results = (data.get("results") or [])[:limit]
    if not results:
        return (
            f"NO RESULTS — no cases matching \"{topic}\" (count: 0).\n"
            "Nothing here is a verified case. Do not fill the gap with a "
            "plausible-sounding citation.\n"
            + DISCLAIMER
        )
    lines = [
        f"Found {count} matching record{'s' if count != 1 else ''}; showing "
        f"the first {len(results)}. Every one below is a real, verified hit — "
        "cite only what you actually need, and re-verify before relying on it."
        "",
    ]
    lines.extend(_fmt_hit(i, r) for i, r in enumerate(results, 1))
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def legal_notice() -> str:
    """The standing disclaimer text."""
    return DISCLAIMER


# ---------------------------------------------------------------------------
# Statute verification (US Code via Cornell LII) — CIVIL + CRIMINAL branches
# ---------------------------------------------------------------------------

_LII_BASE = "https://www.law.cornell.edu/uscode/text/{title}/{section}"
_LII_UA = {"User-Agent": "Mozilla/5.0 (research; citation-verification)"}
_H1_RE = re.compile(r'<h1 class="title" id="page_title">(.*?)</h1>', re.S)


def verify_statute(title, section) -> str:
    """Verify a US Code section EXISTS before you cite it.

    Fetches the Cornell LII page for the given title/section and parses the
    authoritative <h1 class="title" id="page_title"> heading as ground truth.
    A real section returns VERIFIED with the true section name and the LII
    URL; a bogus title/section 404s or lacks the heading and returns NOT
    FOUND with the same blunt no-citing language as verify_case.

    RULES (read before using):
    - This is the statute branch of the hallucination guard. NEVER state a
      statute, code section, or regulation you have not checked this way.
      Invented citations are how models fail at law.
    - CRIMINAL: penal statutes (title 18) + elements + defenses. NEVER tell
      someone whether they committed a crime or what to say to police - that
      is "get a lawyer, and anything you say matters" territory. This is the
      branch where bad information hurts real people most.
    - CIVIL: contracts / torts / property / procedure. State law varies HARD
      by state, and federal rules differ from state rules - always name the
      jurisdiction, or say plainly you don't know it.
    - US Code only. State statutes are out of reach - say so plainly and
      verify against a local source instead of faking confidence.
    - This is legal INFORMATION, not legal advice; no attorney-client
      relationship. Law changes; anything time-sensitive must be re-checked
      fresh. For anything real, get a licensed professional in that
      jurisdiction.
    """
    try:
        title_s, section_s = str(int(title)), str(int(section))
    except (TypeError, ValueError):
        title_s, section_s = str(title), str(section)
    url = _LII_BASE.format(title=title_s, section=section_s)
    try:
        r = httpx.get(url, headers=_LII_UA, timeout=20)
    except Exception as e:  # network failure is NOT a verdict
        return (f"ERROR: Could not reach Cornell LII ({e!r}). Do not cite "
                f"{title_s} U.S.C. {section_s} until it is verified.")
    if r.status_code == 404:
        return (f"NOT FOUND: No such section: {title_s} U.S. Code {section_s}. "
                "Do NOT cite it. Verify against an authoritative "
                "source before citing any statute.")
    if r.status_code != 200:
        return (f"NOT FOUND: Expected HTTP 200 from Cornell LII, got {r.status_code} "
                f"for {url}. Do NOT cite {title_s} U.S. Code {section_s}.")
    m = _H1_RE.search(r.text)
    if not m:
        return (f"NOT FOUND: Page loaded but carries no statute title heading - "
                f"{title_s} U.S. Code {section_s} is not confirmed. "
                "Do NOT cite it.")
    name = re.sub(r"\s+", " ", m.group(1)).strip()
    return (
        "VERIFIED: " + name + "\n"
        "URL: " + url + "\n"
        + ("Legal information, not legal advice; no attorney-client "
           "relationship. US Code only; state law varies by state and "
           "federal rules differ from state rules - name the "
           "jurisdiction or say you don't know it. Law changes - "
           "anything time-sensitive must be re-checked fresh. For "
           "anything real, get a licensed professional in that "
           "jurisdiction.")
    )


# ---------------------------------------------------------------------------
# Regulation search (eCFR JSON API) - FINANCIAL branch (SEC 17 CFR, banking
# 12 CFR, etc.)
# ---------------------------------------------------------------------------

_ECFR_SEARCH = "https://www.ecfr.gov/api/search/v1/results"
_ECFR_UA = {"User-Agent": "Mozilla/5.0 (research; regulation-verification)"}


def find_regulation(query, limit=5) -> str:
    """Find real regulations (eCFR) before you cite them.

    Searches the eCFR public search API (no key) and returns real hits with
    their title/part/section, a citation string like "17 CFR 229.408", and
    the eCFR current-text URL. If nothing comes back, says so plainly.

    RULES (read before using):
    - FINANCIAL branch: securities / tax / banking regulations. These rules
      change constantly - anything time-sensitive must be re-checked fresh.
    - NEVER state a regulation you have not checked this way. This ties to
      the trading rule: no investment advice ever, and any claim of the
      form "the regulation says X" needs the actual regulation cited.
    - US federal regulations only (eCFR). State rules and foreign regimes
      are out of reach - say so plainly and verify against a local source
      instead of faking confidence.
    - This is legal INFORMATION, not legal advice; no attorney-client
      relationship. For anything real, get a licensed professional in that
      jurisdiction.
    """
    if not query or not str(query).strip():
        return ("No regulation query given — nothing was searched. Name a topic "
                "or a CFR cite (e.g. 'insider trading' or '17 CFR 240').")
    query = str(query).strip()
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    try:
        r = httpx.get(_ECFR_SEARCH,
                      params={"query": str(query), "per_page": limit},
                      headers=_ECFR_UA, timeout=20)
    except Exception as e:  # network failure is NOT a verdict
        return f"ERROR: Could not reach the eCFR API ({e!r})."
    if r.status_code != 200:
        return (f"NOT FOUND: eCFR search returned HTTP {r.status_code}. "
                f"No regulation confirmed for {query!r} - do NOT cite one.")
    try:
        data = r.json()
    except Exception as e:
        return f"ERROR: eCFR returned non-JSON ({e!r})."
    results = data.get("results") or []
    if not results:
        return (f"NOT FOUND: No regulation hits for {query!r}. "
                "Say so plainly; do NOT invent a citation.")
    hits = []
    for item in results[:limit]:
        h = item.get("hierarchy") or {}
        title = h.get("title")
        part = h.get("part")
        section = h.get("section")
        citation = None
        if title and part and section:
            citation = f"{title} CFR {section}"
        elif title and part:
            citation = f"{title} CFR Part {part}"
        url = None
        if title and part and section:
            url = (f"https://www.ecfr.gov/current/title-{title}/"
                   f"part-{part}/section-{section}")
        elif title and part:
            url = f"https://www.ecfr.gov/current/title-{title}/part-{part}"
        heading = ""
        hh = item.get("hierarchy_headings") or {}
        for k in ("part", "subpart", "section"):
            if hh.get(k):
                heading = (heading + " " + str(hh[k])).strip()
        hits.append({
            "citation": citation,
            "heading": heading or None,
            "url": url,
            "excerpt": (item.get("full_text_excerpt") or "")[:300] or None,
        })
    # Dedupe by citation: the search API sometimes returns the same
    # section several times; five hits should be five DIFFERENT sections.
    seen = set()
    deduped = []
    for h in hits:
        key = h.get("citation")
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        deduped.append(h)
    if not deduped:
        return (f"NOT FOUND: No regulation hits for {query!r} after "
                "deduplication. Say so plainly; do NOT invent a citation.")
    lines = [f"FOUND {len(deduped)} regulation hit(s) for: {query}"]
    for i, h in enumerate(deduped, 1):
        lines.append("")
        lines.append(f"  [{i}] {h.get('citation') or '(uncited)'}")
        if h.get("heading"):
            lines.append(f"      {h['heading']}")
        if h.get("url"):
            lines.append(f"      {h['url']}")
        if h.get("excerpt"):
            lines.append(f"      excerpt: {h['excerpt']}")
    lines.append("")
    lines.append("Legal information, not legal advice; no attorney-client "
                 "relationship. US federal regulations (eCFR) only - foreign "
                 "or state-level rules are out of reach; say so plainly. No "
                 "investment advice ever. Rules change constantly - anything "
                 "time-sensitive must be re-checked fresh, and any claim of what "
                 "a rule says needs the actual reg cited. For anything real, get "
                 "a licensed professional in that jurisdiction.")
    return "\n".join(lines)
