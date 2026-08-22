import math

META = {
    "name": "ee_reactance",
    "description": "Capacitive/inductive reactance and LC resonant frequency",
    "why": "Recurring AC circuit analysis; selftest values are textbook (1/(2 pi f C) etc.)"
}

def run(params):
    f = float(params.get("f", 0.0))
    out = {}
    if "C" in params:
        C = float(params["C"])
        if C <= 0:
            raise ValueError("capacitance must be positive")
        if f <= 0:
            raise ValueError("frequency must be positive to compute Xc")
        out["Xc_ohm"] = 1.0 / (2.0 * math.pi * f * C)
    if "L" in params:
        L = float(params["L"])
        if L <= 0:
            raise ValueError("inductance must be positive")
        if f <= 0:
            raise ValueError("frequency must be positive to compute Xl")
        out["Xl_ohm"] = 2.0 * math.pi * f * L
    if "LC" in params:
        L = float(params["L"])
        C = float(params["C"])
        if L <= 0 or C <= 0:
            raise ValueError("L and C must be positive for resonance")
        out["f_res_Hz"] = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
    if not out:
        raise ValueError("no valid combination: need f with C or L, or L and C for resonance")
    return out

SELFTEST = {
    "params": {"f": 1000.0, "C": 1e-6, "L": 1e-3, "LC": True},
    "expect": {"Xc_ohm": 159.155, "Xl_ohm": 6.2832, "f_res_Hz": 5032.92},
    "tol": 0.005
}
