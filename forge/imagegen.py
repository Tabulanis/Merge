"""Image generation on CPU via a tiny distilled diffusion model (SD-Turbo).

Runs entirely on CPU so Merge's GPU stays hers — SD-Turbo is distilled to
1-4 steps, so even on CPU (32 cores here) it's seconds, not the minutes a
full SDXL run would cost. Quality is concept/mood/sketch tier, not photoreal;
for finished renders there's ChoreographerStudio on the GPU.

Loaded lazily and kept warm in-process. Also runnable as a CLI:
    python -m forge.imagegen "a lighthouse in a storm" out.png
"""
from __future__ import annotations

import os

# Decisive: hide the GPU so torch can't touch the card Merge owns. Must be set
# before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

_PIPE = None
_MODEL_ID = "stabilityai/sd-turbo"


def _pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        # float32 on CPU (no fp16 speedup without a GPU). Auto-downloads to the
        # HF cache on first use (~2.5GB).
        pipe = AutoPipelineForText2Image.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float32, safety_checker=None)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        _PIPE = pipe
    return _PIPE


def generate(prompt: str, out_path: str, steps: int = 3,
             seed: int | None = None, size: int = 512) -> str:
    """Generate one image from `prompt`, save to `out_path`, return the path."""
    import torch
    steps = max(1, min(int(steps), 6))          # SD-Turbo: 1-4 is the sweet spot
    size = max(256, min(int(size), 768))
    gen = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
    # SD-Turbo is trained for guidance_scale=0.0 (no classifier-free guidance).
    image = _pipe()(prompt=prompt, num_inference_steps=steps,
                    guidance_scale=0.0, height=size, width=size,
                    generator=gen).images[0]
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return str(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m forge.imagegen <prompt> <out.png> [steps] [seed]")
        sys.exit(1)
    prompt, out = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    path = generate(prompt, out, steps=steps, seed=seed)
    print(f"saved: {path}")
