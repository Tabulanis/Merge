"""Escape velocity from a body's mass and radius."""

from math import sqrt

# Gravitational constant (SI)
G = 6.67430e-11  # m^3 kg^-1 s^-2

META = {
    "name": "escape_velocity",
    "description": "Escape velocity from a celestial body's mass and radius",
    "why": "Reusable tool for comparing escape velocities across planets and moons without re-deriving the formula each time"
}


def run(params):
    """
    params: dict with
        mass_kg   — body mass in kilograms
        radius_m  — body radius in metres
    returns: dict with escape velocity in m/s and km/s
    """
    mass = params["mass_kg"]
    radius = params["radius_m"]

    ve_ms = sqrt(2 * G * mass / radius)
    ve_kms = ve_ms / 1000

    return {
        "ve_ms": round(ve_ms, 2),
        "ve_kms": round(ve_kms, 3),
    }


# Self-test: Earth — known textbook value is 11.186 km/s
SELFTEST = {
    "params": {
        "mass_kg": 5.972e24,
        "radius_m": 6.371e6,
    },
    "expect": {
        "ve_kms": 11.186,
    },
    "tol": 0.005,  # 0.5% — textbook value is rounded to 3 decimals
}