"""Real-world news ingestion — the world-data firehose, structured.

Pulls free, timestamped RSS headlines into clean items she can reason over,
embed (text -> vectors), or tag (sentiment / entity / event). Read-only. This is
the raw material for turning world events into signals — the one frontier that
ISN'T derived from price. Honest caveat baked in: public headlines are usually
already priced in by the time you read them, so the edge (if any) is in
INTERPRETATION or NICHE feeds the crowd isn't watching, not the headline itself.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "theblock": "https://www.theblock.co/rss.xml",
}
_TAG = re.compile(r"<[^>]+>")


def fetch(sources: list[str] | None = None, per_feed: int = 15) -> list[dict]:
    """Structured, timestamped items from the requested feeds (all if None)."""
    out = []
    for name in (sources or list(FEEDS)):
        url = FEEDS.get(name)
        if not url:
            continue
        try:
            r = httpx.get(url, timeout=15, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(r.text)
            for it in root.findall(".//item")[:per_feed]:
                out.append({
                    "source": name,
                    "title": (it.findtext("title", "") or "").strip(),
                    "date": (it.findtext("pubDate", "") or "").strip(),
                    "summary": _TAG.sub("", it.findtext("description", "") or "")[:220].strip(),
                })
        except Exception:
            continue
    return out


def feed(sources: list[str] | None = None, limit: int = 25) -> str:
    """Recent headlines as clean text — paste-ready and ready to embed/tag."""
    items = fetch(sources)
    if not items:
        return "No news fetched (feeds down or blocked)."
    lines = [f"{len(items)} recent headlines from {len(set(i['source'] for i in items))} feeds:"]
    for it in items[:limit]:
        lines.append(f"[{it['date'][:22]} · {it['source']}] {it['title']}"
                     + (f"\n    {it['summary']}" if it['summary'] else ""))
    return "\n".join(lines)
