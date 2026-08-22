"""Physics simulators — deterministic playground tools for the agent.

Small, exact, reproducible: every scenario is closed-form math plus (for the
double slit) a seeded Monte Carlo, so the same call always returns the same
answer. These exist so the model reasons ABOUT physics instead of doing
mental arithmetic mid-explanation.
"""
from __future__ import annotations

import math
import random

# Exact constants (SI). h and c are exact by SI definition since 2019.
H = 6.62607015e-34        # Planck, J*s
HBAR = H / (2 * math.pi)
C = 299_792_458.0         # m/s
EV = 1.602176634e-19      # J per electronvolt
LY_M = 9.4607304725808e15  # meters per light-year
YEAR_S = 31_557_600.0     # Julian year, seconds


def _gamma(v_c: float) -> float:
    return 1.0 / math.sqrt(1.0 - v_c * v_c)


def relativity_sim(scenario: str, v_c: float = 0.0,
                   duration_years: float = 1.0,
                   distance_ly: float = 1.0) -> str:
    """Closed-form special-relativity scenarios. Returns readable text."""
    if scenario in ("time_dilation", "twin_trip", "doppler") and not (-0.999999 < v_c < 0.999999):
        return "Error: v_c must be a fraction of light speed strictly between -0.999999 and 0.999999."

    if scenario == "time_dilation":
        if v_c <= 0:
            return "Error: time_dilation needs v_c > 0 (fraction of c)."
        g = _gamma(v_c)
        traveler = duration_years / g
        return (f"Speed: {v_c:g} c   ->   gamma = {g:.6f}\n"
                f"While {duration_years:g} years pass for a stationary observer,\n"
                f"the traveler's clock records {traveler:.6f} years\n"
                f"(difference: {duration_years - traveler:.6f} years).\n"
                f"Formula: t_traveler = t_observer * sqrt(1 - v^2/c^2)")

    if scenario == "twin_trip":
        if v_c <= 0:
            return "Error: twin_trip needs v_c > 0 (fraction of c)."
        # Found by Merge auditing this module: distance_ly was unguarded, so a
        # negative distance printed "Earth twin ages: -8.0000 years" and a
        # negative age gap — people aging backwards, with formula annotations
        # attached, which reads exactly like a real result.
        if distance_ly <= 0:
            return ("Error: twin_trip needs distance_ly > 0 — a round trip to a "
                    "negative distance isn't a journey, and the ages it produces "
                    "would be negative too.")
        g = _gamma(v_c)
        earth_years = 2.0 * distance_ly / v_c          # out and back
        traveler_years = earth_years / g
        return (f"Round trip to {distance_ly:g} light-years at {v_c:g} c   (gamma = {g:.6f})\n"
                f"Earth twin ages:    {earth_years:.4f} years\n"
                f"Traveling twin ages: {traveler_years:.4f} years\n"
                f"Age gap when they reunite: {earth_years - traveler_years:.4f} years\n"
                f"(Idealized: instant turnaround, no acceleration phases.)")

    if scenario == "doppler":
        if v_c == 0:
            return "Error: doppler needs v_c != 0 (positive = source approaching, negative = receding)."
        b = v_c
        factor = math.sqrt((1 + b) / (1 - b))
        direction = "approaching" if b > 0 else "receding"
        example = 540.0  # THz, green light
        return (f"Relativistic Doppler, source {direction} at {abs(b):g} c:\n"
                f"observed frequency = emitted * {factor:.6f}\n"
                f"Example: green light emitted at {example:g} THz is seen at "
                f"{example * factor:.1f} THz.\n"
                f"Formula: f_obs = f_emit * sqrt((1+beta)/(1-beta)), beta = v/c")

    if scenario == "photon_clock":
        our_years = distance_ly  # light covers 1 ly per year, by definition
        return (f"A photon crosses {distance_ly:g} light-years.\n"
                f"Time on our clocks: {our_years:g} years.\n"
                f"Time on the photon's own clock: 0 (exactly).\n"
                f"Proper time d_tau = dt * sqrt(1 - v^2/c^2) -> 0 as v -> c.\n"
                f"Emission and absorption are, for the photon, the same instant.")

    return ("Error: unknown scenario. Choose: time_dilation, twin_trip, "
            "doppler, photon_clock.")


def _photon_lines(wavelength_nm: float) -> str:
    lam = wavelength_nm * 1e-9
    f = C / lam
    e_j = H * f
    p = H / lam
    return (f"Photon, wavelength {wavelength_nm:g} nm:\n"
            f"frequency  f = c/lambda = {f:.4e} Hz\n"
            f"energy     E = h*f      = {e_j:.4e} J  =  {e_j / EV:.4f} eV\n"
            f"momentum   p = h/lambda = {p:.4e} kg*m/s")


def quantum_sim(scenario: str, wavelength_nm: float = 500.0,
                slit_separation_um: float = 50.0,
                screen_distance_m: float = 1.0,
                photons: int = 2000,
                delta_x_nm: float = 1.0,
                seed: int = 7) -> str:
    """Quantum scenarios: photon properties, uncertainty floor, double slit."""
    if scenario == "photon":
        if wavelength_nm <= 0:
            return "Error: wavelength_nm must be positive."
        return _photon_lines(wavelength_nm)

    if scenario == "uncertainty":
        if delta_x_nm <= 0:
            return "Error: delta_x_nm must be positive."
        dx = delta_x_nm * 1e-9
        dp = HBAR / (2 * dx)
        return (f"Confine position to delta_x = {delta_x_nm:g} nm.\n"
                f"Minimum momentum spread: delta_p >= hbar/(2*delta_x) = {dp:.4e} kg*m/s\n"
                f"For comparison, a {wavelength_nm:g} nm photon carries "
                f"p = {H / (wavelength_nm * 1e-9):.4e} kg*m/s.\n"
                f"Tighter position -> wider momentum. Not a measurement flaw; wave math.")

    if scenario == "double_slit":
        if wavelength_nm <= 0 or slit_separation_um <= 0 or screen_distance_m <= 0:
            return "Error: wavelength_nm, slit_separation_um, screen_distance_m must be positive."
        photons = max(100, min(int(photons), 200_000))
        lam = wavelength_nm * 1e-9
        d = slit_separation_um * 1e-6
        fringe_mm = (lam * screen_distance_m / d) * 1000.0
        # Sample individual photon arrivals from the cos^2 interference law
        # (single-slit envelope ignored — pure two-slit physics). Seeded so
        # the same call always draws the same dots.
        rng = random.Random(seed)
        bins = 41
        span_mm = 4.0 * fringe_mm            # show ~4 fringes either side of center
        counts = [0] * bins
        for _ in range(photons):
            while True:                       # rejection sampling against cos^2
                y = rng.uniform(-span_mm, span_mm)
                if rng.random() <= math.cos(math.pi * y / fringe_mm) ** 2:
                    break
            idx = int((y + span_mm) / (2 * span_mm) * bins)
            counts[min(idx, bins - 1)] += 1
        peak = max(counts) or 1
        width = 46
        rows = []
        for i, n in enumerate(counts):
            y_mm = -span_mm + (i + 0.5) * (2 * span_mm) / bins
            bar = "#" * round(n / peak * width)
            rows.append(f"{y_mm:+7.2f} mm |{bar}")
        return (f"Double slit: lambda = {wavelength_nm:g} nm, slits {slit_separation_um:g} um apart, "
                f"screen {screen_distance_m:g} m away.\n"
                f"Fringe spacing: delta_y = lambda*L/d = {fringe_mm:.3f} mm\n"
                f"{photons} photons arrived ONE AT A TIME; each row is a strip of screen,\n"
                f"each # is arrivals piling up (seed {seed}, reproducible):\n\n"
                + "\n".join(rows)
                + "\n\nEach photon landed at one point (particle); the stripes are the wave."
                + "\n\n[Note to you, the agent — not part of the chart: the user "
                  "CANNOT see this tool result. If they asked to see the pattern, "
                  "copy the chart rows above verbatim into your reply.]")

    return "Error: unknown scenario. Choose: photon, double_slit, uncertainty."
