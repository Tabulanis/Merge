META = {"name": "prob_birthday",
        "description": "Probability at least two of N people share a birthday (365-day year)",
        "why": "Standard closed-form birthday problem; intuition badly underestimates it"}

def run(params):
    N = int(params["N"])
    # A negative party size returned 0.0 — a real-looking probability for
    # an input that has no meaning. Refuse it instead.
    if N < 0:
        raise ValueError("N must be 0 or more — a negative number of people isn't a party.")

    # P(at least one shared birthday) = 1 - prod_{k=0}^{N-1} (1 - k/365)
    p_no_shared = 1.0
    for k in range(N):
        p_no_shared *= (1 - k/365.0)
    return {
        "N": N,
        "P_shared": 1 - p_no_shared,
        "P_no_shared": p_no_shared,
    }

SELFTEST = {
    "params": {"N": 23},
    "expect": {"P_shared": 0.5073},
    "tol": 0.0005
}
