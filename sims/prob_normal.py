META = {"name": "prob_normal",
        "description": "P(a <= Z <= b) for the standard normal (mean 0, sd 1)",
        "why": "Workhorse continuous probability lookup; closed form via erf"}

import math

def _phi(x):
    # standard normal CDF via erf
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))

def run(params):
    a = float(params["a"])
    b = float(params["b"])
    if a > b:
        a, b = b, a
    return {
        "a": a, "b": b,
        "P_a_le_Z_le_b": _phi(b) - _phi(a),
    }

SELFTEST = {
    "params": {"a": -1, "b": 1},
    "expect": {"P_a_le_Z_le_b": 0.6827},
    "tol": 0.0005
}
