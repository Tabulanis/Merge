"""The pulse collectors — quietly accumulate the world-data her portals need.

Merge's flow-signal ideas (narrative drift, sentiment-lag, FOMO-without-flow,
dev/price decoupling) all need TIME-SERIES that don't exist yet on our disk.
This module takes one snapshot per run of every free source and appends it to
JSONL files under ~/forge/datasets/pulse/ — run on a schedule, it builds the
dataset that makes those tests possible in weeks, honestly, once each.

Collection only. No trading, no keys, no interpretation — just receipts piling
up on the abundant resource (disk) so the scarce one (a real test) gets fed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

PULSE_DIR = Path.home() / "forge" / "datasets" / "pulse"


def _append(name: str, rec: dict) -> None:
    PULSE_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"t": time.time(), **rec}
    with (PULSE_DIR / f"{name}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def collect_once() -> str:
    """One snapshot of every source. Safe to run any time; each failure is
    isolated (a dead source never blocks the others)."""
    got = []

    # 1) Fear & Greed (today's value — the history API exists, but our own log
    #    is what lets us measure the LAG between extremes and volume response)
    try:
        d = httpx.get("https://api.alternative.me/fng/?limit=1&format=json",
                      timeout=15).json()["data"][0]
        _append("fng", {"value": int(d["value"]), "label": d["value_classification"]})
        got.append("fng")
    except Exception:
        pass

    # 2) Global market state (BTC dominance, total mcap — rotation context)
    try:
        g = httpx.get("https://api.coingecko.com/api/v3/global", timeout=15).json()["data"]
        _append("global", {
            "btc_dominance": g["market_cap_percentage"].get("btc"),
            "eth_dominance": g["market_cap_percentage"].get("eth"),
            "total_mcap_usd": g["total_market_cap"]["usd"],
            "total_volume_usd": g["total_volume"]["usd"],
        })
        got.append("global")
    except Exception:
        pass

    # 3) News headlines (titles only — the narrative stream for drift/embedding)
    try:
        from . import news as _news
        items = _news.fetch(per_feed=10)
        _append("news", {"n": len(items),
                         "titles": [f"{i['source']}: {i['title']}" for i in items]})
        got.append("news")
    except Exception:
        pass

    # 4) Volumes per asset (Kraken 24h — which arteries are carrying the blood)
    try:
        pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD", "LINKUSD"]
        r = httpx.get("https://api.kraken.com/0/public/Ticker",
                      params={"pair": ",".join(pairs)}, timeout=15).json()["result"]
        vols = {k: {"vol24_base": float(v["v"][1]), "last": float(v["c"][0])}
                for k, v in r.items()}
        _append("volumes", {"assets": vols})
        got.append("volumes")
    except Exception:
        pass

    return f"pulse snapshot: {', '.join(got) or 'nothing reachable'}"


if __name__ == "__main__":
    print(collect_once())
