"""Electric / plasma rocket thrust calculator.

Given input power, efficiency, and specific impulse, computes:
  - exhaust velocity
  - thrust
  - propellant mass flow rate
  - thrust-to-power ratio
"""

g0 = 9.80665  # standard gravity, m/s^2

META = {
    "name": "plasma_propulsion",
    "description": "Electric/plasma rocket thrust from power, efficiency, and Isp",
    "why": "Need validated numbers for ion-drive performance without hand-calculating every time"
}


def run(params):
    """
    params:
        power_kw      — input electrical power in kW
        efficiency    — fraction of input power that becomes jet power (0–1)
        isp_s         — specific impulse in seconds
    """
    P_in = params["power_kw"] * 1000          # W
    eff  = params["efficiency"]
    Isp  = params["isp_s"]

    P_jet = eff * P_in                        # useful jet power, W
    v_ex  = Isp * g0                          # exhaust velocity, m/s
    Thrust = 2 * P_jet / v_ex                 # N
    mdot  = Thrust / v_ex                     # kg/s
    T_P   = Thrust / P_in                     # N/W

    return {
        "exhaust_velocity_ms": round(v_ex, 2),
        "thrust_N": round(Thrust, 4),
        "mass_flow_kg_s": round(mdot, 6),
        "thrust_to_power_N_per_W": round(T_P, 6),
        "jet_power_W": round(P_jet, 2),
    }


# SELFTEST — hand-verified:
#   P_in = 1 kW, eff = 0.5, Isp = 1000 s
#   P_jet = 500 W
#   v_ex  = 1000 * 9.80665 = 9806.65 m/s
#   Thrust = 2*500 / 9806.65 = 0.1019716... N
#   mdot   = Thrust / v_ex = 1.0398e-5 kg/s
#   T/P    = Thrust / 1000 = 0.00010197 N/W
SELFTEST = {
    "params": {"power_kw": 1, "efficiency": 0.5, "isp_s": 1000},
    "expect": {
        "exhaust_velocity_ms": 9806.65,
        "thrust_N": 0.1020,
        "mass_flow_kg_s": 0.000010,
        "thrust_to_power_N_per_W": 0.000102,
        "jet_power_W": 500.0,
    },
    "tol": 0.02,   # 2 % tolerance
}