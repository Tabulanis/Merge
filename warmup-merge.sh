#!/usr/bin/env bash
# Warm Merge's GPU kernels the moment her model finishes loading, so the FIRST
# real reply is as snappy as the rest instead of paying the cold-start tax
# (CUDA context + kernel init + compute-buffer alloc all happen on the first
# inference). Fire-and-forget, run detached by the model service's ExecStartPost.
set +e

# Wait for her model to be up (weights loaded into VRAM) — up to ~2 min.
for i in $(seq 1 60); do
  if curl -s -o /dev/null --max-time 3 http://127.0.0.1:8085/v1/models 2>/dev/null; then
    break
  fi
  sleep 2
done

# A tiny throwaway inference. Warms the CUDA kernels and primes the pipeline; a
# short prompt + a handful of tokens is enough to pay the cold-start cost here,
# not on the user's first message.
curl -s -o /dev/null --max-time 60 http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Warmup ping. Reply with just: ok"}],"max_tokens":8}' \
  2>/dev/null

# Second step: prime the prompt cache through the REAL path, so the first user
# message doesn't pay to process her whole system prompt from scratch. One
# off-the-record Flash turn (leaves no trace) via the dashboard. Best-effort —
# if the dashboard isn't up yet or the token isn't readable, we still got the
# kernel warm above.
TOKEN="$("$HOME/forge/.venv/bin/python" -c "import yaml,os; print((yaml.safe_load(open(os.path.expanduser('~/.forge/config.yaml'))).get('server') or {}).get('token',''))" 2>/dev/null)"
if [ -n "$TOKEN" ]; then
  for i in $(seq 1 20); do
    curl -sk -o /dev/null --max-time 3 "https://127.0.0.1:8770/api/voices?token=$TOKEN" 2>/dev/null && break
    sleep 2
  done
  curl -sk -o /dev/null --max-time 30 "https://127.0.0.1:8770/api/chat?token=$TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"message":"warmup","mode":"flash","privacy":"offrec"}' 2>/dev/null
fi

logger -t merge-warmup "warmed the merge model on :8085 (kernels + prompt cache)" 2>/dev/null || true
