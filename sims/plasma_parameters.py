import math

META = {
    "name": "plasma_parameters",
    "description": "Plasma frequency, Debye length, and Debye-sphere particle count from n_e and T_e",
    "why": "Core plasma diagnostics; every plasma physicist computes these from n_e and T_e"
}

# CODATA / standard constants (corroborated, see plasma_basics dataset)
m_e   = 9.1093837139e-31    # kg
e     = 1.602176634e-19     # C
eps0  = 8.8541878128e-12    # F/m

def run(params):
    n_e = float(params["n_e"])   # m^-3
    T_e = float(params["T_e"])   # eV

    # Plasma (electron Langmuir) frequency, rad/s
    omega_pe = math.sqrt(n_e * e**2 / (m_e * eps0))

    # Debye length, m   (T in J: T_e[eV] * e)
    lambda_D = math.sqrt(eps0 * (T_e * e) / (n_e * e**2))

    # Number of electrons in a Debye sphere
    N_D = (4.0/3.0) * math.pi * n_e * lambda_D**3

    return {
        "omega_pe_rad_s": omega_pe,
        "lambda_D_m": lambda_D,
        "N_D": N_D,
    }

# SELFTEST: textbook case — n_e = 1e19 m^-3 (1e13 cm^-3) hydrogen at 1 eV (Chen, "Fundamentals of
# Plasma Physics", 2nd ed., Table 2.1 / worked examples; same numbers in
# Stix & Fatenejder, "Plasma Physics for Nuclear Fusion", §2.2).
# Independent hand-evaluation:
#   omega_pe = sqrt(1e19 * (1.602e-19)^2 / (9.109e-31 * 8.854e-12))
#            = sqrt(1e19 * 2.566e-38 / 8.066e-42)
#            = sqrt(3.181e22) = 1.784e11 rad/s
#   lambda_D = sqrt(8.854e-12 * 1.602e-19 / (1e19 * 2.566e-38))
#            = sqrt(1.418e-30 / 2.566e-19) = sqrt(5.527e-12) = 2.351e-6 m
#   N_D = (4/3) pi * 1e19 * (2.351e-6)^3
#       = 4.18879e19 * 1.2997e-17 = 544.3
#   (an earlier hand-check wrote 4.189e12 here instead of 4.189e19 and landed on
#   5.44e-5. The CODE was right; the expectation was wrong, and the self-test
#   caught the disagreement and withheld the checkmark — which is the point.
#   Physics agrees with the code: a plasma requires N_D >> 1, so 5.44e-5 would
#   mean this isn't a plasma at all.)
SELFTEST = {
    "params": {"n_e": 1e19, "T_e": 1.0},
    "expect": {
        "omega_pe_rad_s": 1.784e11,
        "lambda_D_m": 2.351e-6,
        "N_D": 544.3,
    },
    "tol": 0.005,
}
