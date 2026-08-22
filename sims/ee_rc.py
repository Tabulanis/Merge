import math

META = {
    "name": "ee_rc",
    "description": "RC circuit: time constant, cutoff frequency, charge voltage",
    "why": "Standard first-order RC analysis; needed repeatedly for filter design and transient questions"
}

def run(params):
    R = float(params["R"])
    C = float(params["C"])
    if R < 0:
        raise ValueError("negative resistance")
    if C < 0:
        raise ValueError("negative capacitance")
    if R == 0 or C == 0:
        raise ValueError("R or C is zero -> no RC time constant / cutoff defined")
    tau = R * C
    fc = 1.0 / (2.0 * math.pi * R * C)
    out = {"tau_s": tau, "cutoff_Hz": fc}
    if "t" in params and "V" in params:
        t = float(params["t"])
        V = float(params["V"])
        if t < 0:
            raise ValueError("negative time")
        out["V(t)"] = V * (1.0 - math.exp(-t / tau))
    return out

SELFTEST = {
    "params": {"R": 1000.0, "C": 1e-6},
    "expect": {"tau_s": 0.001, "cutoff_Hz": 159.155},
    "tol": 0.005
}
