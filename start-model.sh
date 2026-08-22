#!/usr/bin/env bash
# Start a local model (or the image server) for Forge to talk to.
#   ./start-model.sh          -> the 30B (main workhorse, port 8080)
#   ./start-model.sh tiny     -> TinyLlama (small + fast, port 8081)
#   ./start-model.sh little   -> Qwen 3B on CPU (routing/titles/summaries, 8083)
#   ./start-model.sh coder14  -> Qwen2.5-Coder 14B (port 8082)
#   ./start-model.sh vision   -> shared 7B vision/mmproj on CPU (port 8090)
#   ./start-model.sh merge    -> Merge's sighted 27B (port 8085)
#   ./start-model.sh embed    -> nomic embedding organ, CPU (port 8086)
#   ./start-model.sh imagegen -> SD-Turbo image server, CPU (port 8771)
set -e
LLAMA=~/llama.cpp/build/bin/llama-server

# The image generator is a python server, not a llama model — handle it
# before the llama case. CPU only (GPU stays Merge's); holds SD-Turbo warm
# on :8771 so each image skips the model reload.
if [ "${1:-big}" = "imagegen" ]; then
  echo "starting image-gen warm server on :8771 (CPU, SD-Turbo) ..."
  exec env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$HOME/forge" python3 -m forge.imagegen_server
fi

case "${1:-big}" in
  big)
    MODEL=~/llama-agent/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-Q4_K_M.gguf
    # 32k context in the same VRAM as the old 16k: the KV cache (her
    # working memory on the card) is stored at 8-bit instead of 16-bit.
    # Needs flash attention on. Measured 2026-08-11 before keeping.
    PORT=8084; CTX=32768; NGL=99  # moved off 8080: Mote's harness server owns 8080 now (2026-08-22)
    EXTRA="-fa on -ctk q8_0 -ctv q8_0" ;;
  tiny)
    # -ngl 0 keeps this one entirely in system RAM on the CPU, leaving the
    # whole GPU for the big model. That's the point of a small router model:
    # it costs nothing the big one needs.
    MODEL=~/aidojo/shared/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    PORT=8081; CTX=4096; NGL=0 ;;
  little)
    # The little brain: modern 3B on CPU only, leaving the whole GPU to
    # the big model. Fast enough for short prompts (titles, routing,
    # summaries); not meant for real coding.
    MODEL=~/forge/models/qwen2.5-3b-instruct-q4.gguf
    PORT=8083; CTX=8192; NGL=0 ;;
  coder14)
    MODEL=~/llmmodels/Qwen2.5-Coder-14B-Instruct-abliterated-Q8_0.gguf
    PORT=8082; CTX=8192; NGL=99 ;;
  vision)
    # The eyes. Needs BOTH files: the language model and the mmproj, which
    # is the part that turns pixels into something the model can read.
    # A few GPU layers only — the big coder model owns most of the VRAM.
    # -ngl 0: CPU only. Measured, not guessed — with the 30B holding 22GB of
    # a 24GB card there is ~1.7GB free, and this needs ~2GB for weights plus
    # ~1.3GB for the vision encoder. Vision gets used in bursts (look at one
    # screenshot) rather than continuously, so trading speed for coexistence
    # is the right call. Raise NGL if you run vision without the big model.
    MODEL=~/forge/models/qwen2.5-vl-7b-q4.gguf
    MMPROJ=~/forge/models/qwen2.5-vl-7b-mmproj.gguf
    PORT=8090; CTX=4096; NGL=0 ;;
  merge)
    # Merge's sighted brain: Qwen3.6 27B dense, abliterated (Heretic), with
    # native vision — the mmproj here is her own eyes, not the shared 7B on
    # 8090. Dense means the whole model fires per token: smarter and sighted
    # than the 30B MoE, but slower per word.
    # VRAM math: she does NOT fit alongside the big 30B. Stop that first
    # (it's the llama-server on port 8080), then start her — she gets the
    # whole card. 16.5GB weights + ~1.5GB vision + KV cache ≈ 21GB of 24.
    # CTX=16384 is the safe start; try 32768 later and watch nvidia-smi.
    MODEL=~/forge/models/Qwen3.6-27B-Abliterated-Heretic-Q4_K_M.gguf
    MMPROJ=~/forge/models/Qwen3.6-27B-mmproj-F16.gguf
    PORT=8085; CTX=24576; NGL=99
    # --reasoning-budget 1024: her thinking-token headroom. It's now GENEROUS
    # on purpose — only Precise/Deep turn thinking on (Flash/Muse/Balanced run
    # with it off), so a big budget costs the fast modes nothing and just lets
    # the deep gears finish reasoning through hard problems instead of getting
    # guillotined mid-thought (which was producing empty answers). Per-mode
    # budgets aren't possible — llama takes this globally at startup. Lower it
    # if Precise/Deep feel too slow; -1 = unlimited, 0 = no thinking.
    EXTRA="-fa on -ctk q8_0 -ctv q8_0 --reasoning-budget 1024" ;;
  merge38)
    # CANDIDATE brain: Qwen3.8 27B abliterated (Blackfrost) WITH native vision
    # (mmproj ships in the same repo — the hard rule held). Same port as merge:
    # they can never run together on the 24GB card, so it's stop-one-start-other.
    # Same launch recipe as 3.6 until testing says otherwise. 3.6 stays on disk
    # and in its own case until this one has EARNED the seat (battery + probes).
    MODEL=~/forge/models/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf
    MMPROJ=~/forge/models/mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf
    PORT=8085; CTX=32768; NGL=99
    EXTRA="-fa on -ctk q8_0 -ctv q8_0 --reasoning-budget 1024" ;;
  embed)
    # Her associative sense-organ: nomic-embed-text on CPU, embedding-only.
    # Turns memory into vectors so recall finds things by MEANING, not just
    # matching words. Tiny + fast; the GPU stays entirely Merge's. Started
    # separately (llama-server can't both generate and embed on one port).
    MODEL=~/forge/models/nomic-embed-text-v1.5.f16.gguf
    PORT=8086; CTX=2048; NGL=0; EMBED=1 ;;
  *) echo "unknown model: $1  (try: big, tiny, little, coder14, vision, merge, merge38, embed, imagegen)"; exit 1 ;;
esac

if [ -n "${EMBED:-}" ]; then
  # Embedding-only server: --pooling mean is what nomic-embed expects, and no
  # --jinja (there's no chat here, just text -> vector). CUDA hidden so it can
  # never touch the card even if a GPU build tries a cudaMalloc.
  echo "starting $(basename "$MODEL") as an embedding organ on port $PORT (CPU) ..."
  exec env CUDA_VISIBLE_DEVICES="" "$LLAMA" -m "$MODEL" --host 127.0.0.1 \
    --port "$PORT" -c "$CTX" --embedding --pooling mean
fi

if [ -n "${MMPROJ:-}" ]; then
  echo "starting $(basename "$MODEL") with vision on port $PORT ..."
  if [ "$NGL" = "0" ]; then
    # CUDA_VISIBLE_DEVICES="" is the only thing that actually keeps this off
    # the GPU. Both -ngl 0 and --device none still let the vision encoder
    # try a cudaMalloc, which dies (and core-dumps) when the big coder model
    # already owns the card. Hiding the GPU at the driver level is decisive.
    echo "  (CPU only — the GPU is spoken for; expect ~2 min per image)"
    exec env CUDA_VISIBLE_DEVICES="" "$LLAMA" -m "$MODEL" --mmproj "$MMPROJ" \
      --host 127.0.0.1 --port "$PORT" -c "$CTX" --jinja
  fi
  exec "$LLAMA" -m "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port "$PORT" \
    -ngl "$NGL" -c "$CTX" --jinja ${EXTRA:-}
fi

echo "starting $(basename "$MODEL") on port $PORT ..."
# --jinja is what makes tool calling work: it uses the model's own chat
# template, which is where the tool-call format lives.
if [ "$NGL" = "0" ]; then
  # Same lesson as the vision model: -ngl 0 alone still lets the CUDA
  # build touch the card. Hiding the GPU entirely is what actually keeps
  # a CPU model off it — vital when the big model owns nearly all VRAM.
  exec env CUDA_VISIBLE_DEVICES="" "$LLAMA" -m "$MODEL" --host 127.0.0.1 \
    --port "$PORT" -c "$CTX" --jinja
fi
exec "$LLAMA" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -ngl "$NGL" -c "$CTX" --jinja ${EXTRA:-}
