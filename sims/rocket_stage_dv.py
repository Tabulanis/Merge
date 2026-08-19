"""Delta-v calculator for a single rocket stage (Tsiolkovsky equation)."""

import math

META = {
    "name": "rocket_stage_dv",
    "description": "Delta-v from wet mass, dry mass, and specific impulse",
    "why": "Avoid hand-computing the rocket equation for every stage design"
}

g0 = 9.80665  # standard gravity, m/s²


def run(params):
    """
    params: dict with
        isp_s      — specific impulse in seconds
        wet_kg     — fully-fueled mass in kg
        dry_kg     — dry (empty) mass in kg
    """
    isp = params["isp_s"]
    wet = params["wet_kg"]
    dry = params["dry_kg"]

    mass_ratio = wet / dry
    dv = isp * g0 * math.log(mass_ratio)

    return {
        "dv_m_s": round(dv, 1),
        "dv_km_s": round(dv / 1000, 3),
        "mass_ratio": round(mass_ratio, 3),
        "propellant_kg": round(wet - dry, 1),
    }


# Self-test: 300 s Isp, 1000 kg wet, 400 kg dry
# Expected: dv = 300 * 9.80665 * ln(2.5) ≈ 2688 m/s
SELFTEST = {
    "params": {"isp_s": 300, "wet_kg": 1000, "dry_kg": 400},
    "expect": {"dv_m_s": 2688.0},
    "tol": 0.05,
}