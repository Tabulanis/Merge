META = {"name": "prob_binomial",
        "description": "P(exactly k successes in n Bernoulli trials at prob p), plus mean and variance",
        "why": "Core discrete distribution; needed anywhere success-count probabilities come up"}

import math

def run(params):
    n = int(params["n"])
    k = int(params["k"])
    p = float(params["p"])
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0,1]")
    if not (0 <= k <= n):
        raise ValueError("k out of range")
    # log-space binomial pmf to avoid underflow
    log_c = math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
    pmf = math.exp(log_c + k*math.log(p) + (n-k)*math.log(1-p)) if p not in (0.0, 1.0) else (1.0 if k==n else 0.0)
    return {
        "n": n, "k": k, "p": p,
        "P_exact_k": pmf,
        "mean": n*p,
        "variance": n*p*(1-p),
    }

SELFTEST = {
    "params": {"n": 10, "k": 5, "p": 0.5},
    "expect": {"P_exact_k": 0.24609375},
    "tol": 0.000001
}
