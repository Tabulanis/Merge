META = {
    "name": "carnot_cycle",
    "description": "Maximum (Carnot) thermal efficiency between two reservoirs",
    "why": "The upper bound every real heat engine is measured against; two numbers, one answer",
}


def run(params):
    Th = float(params["Th_k"])   # hot reservoir temperature, K
    Tc = float(params["Tc_k"])   # cold reservoir temperature, K
    if Th <= 0 or Tc <= 0:
        raise ValueError("temperatures must be in kelvin and > 0")
    if Tc >= Th:
        raise ValueError("cold reservoir must be colder than hot (Tc < Th)")

    eta = 1.0 - Tc / Th
    return {
        "Th_k": Th,
        "Tc_k": Tc,
        "eta_max": eta,
        "eta_max_pct": 100.0 * eta,
    }


# SELFTEST: classic worked example (Cengel & Boles, "Thermodynamics", and
# every standard thermo text): hot reservoir 1000 K, cold reservoir 300 K.
# eta_max = 1 - 300/1000 = 0.70  (70 %). Independent, textbook-known answer.
SELFTEST = {
    "params": {"Th_k": 1000.0, "Tc_k": 300.0},
    "expect": {"eta_max": 0.70, "eta_max_pct": 70.0},
    "tol": 0.005,
}
