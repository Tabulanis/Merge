META = {"name": "prob_bayes",
        "description": "Posterior P(condition|positive test) from prevalence, sensitivity, specificity",
        "why": "Base-rate fallacy: intuition says ~sensitivity when a test is '99% accurate', and is badly wrong for rare conditions"}

def run(params):
    prev = float(params["prevalence"])
    sens = float(params["sensitivity"])
    spec = float(params["specificity"])
    if not all(0 <= x <= 1 for x in (prev, sens, spec)):
        raise ValueError("inputs must be in [0,1]")
    # Bayes: P(C|+) = sens*prev / (sens*prev + (1-spec)*(1-prev))
    num = sens * prev
    den = num + (1 - spec) * (1 - prev)
    post = num / den if den > 0 else 0.0
    return {
        "prevalence": prev,
        "sensitivity": sens,
        "specificity": spec,
        "P_condition_given_positive": post,
        "PPV_note": "intuition ~ sensitivity (%.2f) is wrong; true posterior is %.4f" % (sens, post),
    }

SELFTEST = {
    "params": {"prevalence": 0.01, "sensitivity": 0.99, "specificity": 0.95},
    "expect": {"P_condition_given_positive": 0.1667},
    "tol": 0.0005
}
