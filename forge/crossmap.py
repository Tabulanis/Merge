"""Cross-asset influence map — coupling first, then the residual 'map of cause'.

Crypto is heavily COUPLED: almost everything is BTC-beta, so raw correlations
just say 'it's all one trade'. So:

  1. Measure the coupling — the first principal component (the common 'market'
     mode) and how much of everything it explains.
  2. Strip it out — work on RESIDUAL returns (each asset's move AFTER the common
     one), which is where any real cross-structure hides.
  3. Map directed lead-lag on the residuals — does A's past move help predict B's
     future? — as a directed influence graph.
  4. OUT-OF-SAMPLE the whole map — keep only edges that hold on data they weren't
     found on. A wide scan is a spurious-pattern factory; this is the filter.

Honest label: this is PREDICTIVE influence, not proven cause. In a coupled
system most apparent A->B is really 'both driven by C' (the factor we removed
helps, but narratives/sectors still confound). Treat the map as hypotheses to
test, never as mechanism. Descriptive of the past; fragile; costs not modelled.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .paper_market import fetch_closes

UNIVERSE = ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD", "DOTUSD",
            "LINKUSD", "LTCUSD", "AVAXUSD", "ATOMUSD", "XLMUSD", "ALGOUSD"]
_SHORT = {"XBTUSD": "BTC", "ETHUSD": "ETH", "SOLUSD": "SOL", "ADAUSD": "ADA",
          "XRPUSD": "XRP", "DOTUSD": "DOT", "LINKUSD": "LINK", "LTCUSD": "LTC",
          "AVAXUSD": "AVAX", "ATOMUSD": "ATOM", "XLMUSD": "XLM", "ALGOUSD": "ALGO"}


def _returns_matrix(pairs):
    series = {}
    for p in pairs:
        try:
            series[p] = np.array(fetch_closes(p, 1440))
        except Exception:
            pass
    names = list(series)
    m = min(len(series[p]) for p in names)
    closes = np.column_stack([series[p][-m:] for p in names])
    rets = np.diff(closes, axis=0) / closes[:-1]          # T x N daily returns
    return names, rets


def _residuals(rets):
    """Z-score, pull out PC1 (the common market mode), return residuals + coupling."""
    z = (rets - rets.mean(0)) / (rets.std(0) + 1e-9)
    pca = PCA(n_components=min(z.shape))
    scores = pca.fit_transform(z)
    pc1_var = pca.explained_variance_ratio_[0]
    recon_pc1 = np.outer(scores[:, 0], pca.components_[0])  # what PC1 explains
    resid = z - recon_pc1
    return resid, pc1_var


def _leadlag(resid, lag=1):
    """Directed influence matrix: infl[i,j] = corr(resid_i[t], resid_j[t+lag])."""
    n = resid.shape[1]
    a, b = resid[:-lag], resid[lag:]
    infl = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            infl[i, j] = np.corrcoef(a[:, i], b[:, j])[0, 1]
    return infl


def run(lag: int = 1, min_edge: float = 0.12, top: int = 10) -> str:
    names, rets = _returns_matrix(UNIVERSE)
    short = [_SHORT.get(p, p) for p in names]
    resid, pc1_var = _residuals(rets)

    # out-of-sample: find edges on train, keep only those that survive on test
    cut = len(resid) // 2
    infl_tr = _leadlag(resid[:cut], lag)
    infl_te = _leadlag(resid[cut:], lag)

    edges = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tr, te = infl_tr[i, j], infl_te[i, j]
            # persistent = same sign both halves AND both meaningfully strong
            if abs(tr) >= min_edge and abs(te) >= min_edge and np.sign(tr) == np.sign(te):
                edges.append((short[i], short[j], (tr + te) / 2, tr, te))
    edges.sort(key=lambda e: -abs(e[2]))

    # who is the strongest 'leader' among persistent edges (out-degree of influence)
    lead_count = {}
    for a, b, s, *_ in edges:
        lead_count[a] = lead_count.get(a, 0) + 1

    lines = [f"[cross-asset map · {len(names)} crypto · daily · {len(resid)}d · lag {lag}d]",
             f"  COUPLING: the common market mode (PC1) explains {pc1_var*100:.0f}% of ALL the "
             f"movement — that's how 'not uncoupled' they are; almost everything is BTC-beta.",
             f"  (everything below is on the RESIDUALS — moves AFTER the common one is removed)",
             f"  --- persistent directed edges (held on train AND held-out test) ---"]
    if not edges:
        lines.append("  NONE survived out-of-sample. That's the honest, common result: the "
                     "residual lead-lag 'patterns' were noise — they didn't repeat on unseen data.")
    else:
        for a, b, s, tr, te in edges[:top]:
            lines.append(f"  {a:5} → {b:5}  influence {s:+.2f}   (train {tr:+.2f} / test {te:+.2f})")
        if lead_count:
            top_leader = max(lead_count, key=lead_count.get)
            lines.append(f"  strongest 'leader' (most persistent outgoing edges): {top_leader} "
                         f"({lead_count[top_leader]} edges)")
    lines.append("  NOTE: predictive influence, NOT proven cause — in a coupled market most A→B is "
                 "really 'both driven by C' (narratives, sectors, macro). These survived one "
                 "out-of-sample split; a real map needs many splits, and even then it's a set of "
                 "HYPOTHESES to test with costs, never a mechanism. Fragile by nature.")
    return "\n".join(lines)
