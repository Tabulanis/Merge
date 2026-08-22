import math

META = {
    "name": "phase_state",
    "description": "Classify a fluid's phase state (super-critical / liquid / gas / near-critical) from T and P vs. its critical point",
    "why": "Supercritical-fluid work needs an instant read on whether a given T,P is super-critical; the reduced coordinates (Tr, Pr) are the universal way to express it",
}

# Critical constants pulled from the corroborated 'critical_points' dataset
# (Engineering ToolBox + Wikipedia/peer-reviewed, two independent roots).
# Keyed by lowercased name; aliases map to the canonical key.
CP = {
    "carbon_dioxide": {"tc_k": 304.13, "pc_mpa": 7.377, "aliases": ["co2", "co_2", "carbon-dioxide", "carbon dioxide"]},
    "water":          {"tc_k": 647.096, "pc_mpa": 22.064, "aliases": ["h2o", "h_2o", "steam", "water"]},
    "nitrogen":       {"tc_k": 126.2,  "pc_mpa": 3.3958, "aliases": ["n2", "n_2"]},
    "methane":        {"tc_k": 190.56, "pc_mpa": 4.599,  "aliases": ["ch4", "ch_4"]},
    "ethane":         {"tc_k": 305.33, "pc_mpa": 4.872,  "aliases": ["c2h6", "c2h_6"]},
    "argon":          {"tc_k": 150.8,  "pc_mpa": 4.898,  "aliases": ["ar"]},
    "ammonia":        {"tc_k": 405.4,  "pc_mpa": 11.283, "aliases": ["nh3", "nh_3", "ammonia"]},
}
# Near-critical band: within this relative distance of BOTH critical coordinates.
NEAR_BAND = 0.10   # +/-10 % of Tc and Pc


def _lookup(name):
    key = str(name).strip().lower().replace(" ", "_")
    if key in CP:
        return key, CP[key]
    for canon, rec in CP.items():
        if key in rec["aliases"]:
            return canon, rec
    raise KeyError("unknown fluid %r; known: %s" % (name, ", ".join(sorted(CP))))


def run(params):
    name = params["fluid"]
    T = float(params["T_k"])
    P = float(params["P_mpa"])

    canon, rec = _lookup(name)
    Tc, Pc = rec["tc_k"], rec["pc_mpa"]

    Tr = T / Tc
    Pr = P / Pc

    # Supercritical: above BOTH critical coordinates.
    if Tr > 1.0 and Pr > 1.0:
        state = "SUPERCRITICAL"
    else:
        # Below one or both: classify by reduced-temperature side.
        # Tr < 1 and Pr > 1  -> compressed (liquid-like)
        # Tr < 1 and Pr < 1  -> gas (could be liquid if above sat. pressure;
        #                      without a saturation curve we call it gas/vapor)
        # Tr > 1 and Pr < 1  -> gas (superheated / supercritical-temperature gas)
        if Tr >= 1.0 and Pr < 1.0:
            state = "gas"
        elif Tr < 1.0 and Pr >= 1.0:
            state = "liquid"
        else:
            # Tr < 1 AND Pr < 1: reduced properties genuinely cannot separate
            # liquid from gas here — that needs the vapor-pressure curve. The
            # old default of "gas" called room-temperature water a gas, stated
            # as flatly as a real answer. Say what's actually known instead.
            state = "SUBCRITICAL (liquid or gas — undetermined)"

    # Near-critical override: within NEAR_BAND of BOTH coordinates.
    near = (abs(Tr - 1.0) <= NEAR_BAND) and (abs(Pr - 1.0) <= NEAR_BAND)
    if near:
        state = "NEAR-CRITICAL"

    return {
        "fluid": canon,
        "T_k": T,
        "P_mpa": P,
        "Tc_k": Tc,
        "Pc_mpa": Pc,
        "Tr": Tr,
        "Pr": Pr,
        "state": state,
        "note": ("Below BOTH critical coordinates (Tr<1 and Pr<1) reduced "
                 "properties cannot tell liquid from gas — that requires the "
                 "vapor-pressure curve, so this reports UNDETERMINED rather "
                 "than guessing. Use a real EOS (REFPROP/NIST/Antoine) to "
                 "resolve the subcritical region."),
    }


# SELFTEST: CO2 at 320 K / 10 MPa.
# CO2 critical point: Tc = 304.13 K, Pc = 7.377 MPa (NIST / standard literature).
# Tr = 320/304.13 = 1.0522 (>1), Pr = 10/7.377 = 1.3556 (>1)  =>  SUPERCRITICAL.
# This is the textbook supercritical-CO2 condition (the basis of every sCO2 power
# cycle, e.g. the Rankine cycles in Poling, Prausnitz, O'Connell, "The Properties
# of Gases and Liquids", 5th ed., and the sCO2 cycle literature).
SELFTEST = {
    "params": {"fluid": "co2", "T_k": 320.0, "P_mpa": 10.0},
    "expect": {
        "state": "SUPERCRITICAL",
        "Tr": 320.0 / 304.13,
        "Pr": 10.0 / 7.377,
    },
    "tol": 0.005,
}
