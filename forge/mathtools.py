"""
The compute tool: exact math for models that would otherwise guess.

Language models drop minus signs and hallucinate arithmetic — even good
ones. Physics and geometry answers that must be RIGHT get computed here:
the model writes a short Python program using sympy/math, the program runs
in a subprocess, and whatever it prints is the answer. The model does the
translating (words → equations); the machine does the math.

Every run is also appended to ~/.forge/physics-dataset.jsonl — the seed
corpus for someday fine-tuning a small local specialist. Real problems,
real programs, real answers, collected passively from normal use.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

DATASET = Path.home() / ".forge" / "physics-dataset.jsonl"
COMPUTE_TIMEOUT = 15          # seconds; sympy can chew on a bad integral forever
MAX_OUTPUT = 4000

# Imported into every snippet so the model never burns a turn on imports.
PREAMBLE = """\
import math
from math import pi, e, sqrt, sin, cos, tan, asin, acos, atan, atan2, radians, degrees, log, exp
import sympy
from sympy import (symbols, Symbol, solve, simplify, expand, factor, integrate,
                   diff, limit, series, Eq, Rational, nsimplify, sqrt as ssqrt,
                   sin as ssin, cos as scos, tan as stan, pi as spi, oo,
                   Matrix, det, trigsimp, solveset, linsolve, nonlinsolve)
from sympy.geometry import Point, Line, Segment, Circle, Triangle, Polygon, Ray
from sympy.physics.units import (meter, second, kilogram, newton, joule, watt,
                                 convert_to, speed_of_light, gravitational_constant)
"""


def run_compute(code: str) -> str:
    """Execute a math snippet, return what it printed (or a readable error)."""
    program = PREAMBLE + "\n" + code
    try:
        r = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=COMPUTE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return (f"Error: computation exceeded {COMPUTE_TIMEOUT}s. Simplify the "
                f"problem — use evalf()/nsimplify for numerics, or break it "
                f"into smaller steps.")
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if r.returncode != 0:
        tail = err.splitlines()[-1] if err else "unknown error"
        return f"Error: the computation crashed: {tail}"
    if not out:
        return ("Error: the program printed nothing. print() the result — "
                "the printed text is the answer you get back.")
    result = out[:MAX_OUTPUT]
    _log_dataset(code, result)
    return result


def _log_dataset(code: str, result: str) -> None:
    """Append (program, answer) to the training-data seed file. Best effort —
    dataset collection must never break a computation that succeeded."""
    try:
        DATASET.parent.mkdir(parents=True, exist_ok=True)
        with DATASET.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "code": code,
                                "result": result[:1000]}) + "\n")
    except Exception:
        pass


COMPUTE_DESCRIPTION = (
    "Run exact math in Python and get back what it prints. USE THIS FOR ALL "
    "physics, geometry, and numeric work — never do arithmetic or algebra in "
    "your head; you make sign and rounding mistakes, this tool does not. "
    "sympy is preloaded (symbols, solve, integrate, diff, Eq, Matrix, "
    "geometry: Point/Line/Circle/Triangle/Polygon, physics units: meter, "
    "second, newton, convert_to...) plus python math. Write a short program, "
    "print() the result. Exact forms preferred; .evalf() for decimals. "
    "Example: x=symbols('x'); print(solve(Eq(x**2-5*x+6,0), x))"
)
