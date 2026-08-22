import math

META = {
    "name": "ee_power",
    "description": "Power from any two of V/I/R (incl. I^2R) and decibels for power/voltage ratios",
    "why": "Ubiquitous EE arithmetic; selftest values are textbook (12V*2A=24W, 10log10(2)=3.0103 dB, 20log10(2)=6.0206 dB)"
}

def run(params):
    out = {}
    keys = [k for k in ("V", "I", "R") if k in params]
    if len(keys) >= 2:
        V = float(params.get("V", 0.0))
        I = float(params.get("I", 0.0))
        R = float(params.get("R", 0.0))
        if R < 0:
            raise ValueError("negative resistance")
        if I < 0:
            raise ValueError("negative current")
        if V < 0:
            raise ValueError("negative voltage")
        if "V" in keys and "I" in keys:
            out["P_W"] = V * I
        elif "I" in keys and "R" in keys:
            out["P_W"] = I * I * R
        elif "V" in keys and "R" in keys:
            if R == 0:
                raise ValueError("zero resistance with only V and R")
            out["P_W"] = V * V / R
        # consistency: if all three given, prefer V*I and note others
        if len(keys) == 3:
            out["P_W_from_IR"] = I * I * R
            out["P_W_from_VR"] = V * V / R
    if "P_ratio" in params:
        r = float(params["P_ratio"])
        if r <= 0:
            raise ValueError("power ratio must be positive")
        out["dB_power"] = 10.0 * math.log10(r)
    if "V_ratio" in params:
        r = float(params["V_ratio"])
        if r <= 0:
            raise ValueError("voltage ratio must be positive")
        out["dB_voltage"] = 20.0 * math.log10(r)
    if not out:
        raise ValueError("no valid combination: need two of V/I/R, or a ratio")
    return out

SELFTEST = {
    "params": {"V": 12.0, "I": 2.0, "P_ratio": 2.0, "V_ratio": 2.0},
    "expect": {"P_W": 24.0, "dB_power": 3.0103, "dB_voltage": 6.0206},
    "tol": 0.005
}
