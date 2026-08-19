# her sims land here — see `forge/sims.py`

A sim is deterministic code Merge writes once and reuses forever — but it only
earns its ✓ by reproducing an **independently-known answer** within tolerance.
Otherwise it's marked EXPERIMENTAL and says so every time it runs.

The three here are validated examples (each carries a `SELFTEST` against a known
reference, and passes):

- **`escape_velocity`** — √(2GM/r); self-tests against Earth's 11.186 km/s.
- **`rocket_stage_dv`** — the Tsiolkovsky rocket equation (single stage).
- **`plasma_propulsion`** — electric/plasma thrust from power, efficiency, Isp.

Everything Merge builds beyond these is her own; only proven, non-personal
examples ship here.
