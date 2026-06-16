#!/usr/bin/env bash
set -euo pipefail
MODEL="/models/smoke.gguf"
FLOOR="${MIN_DECODE_TOKS:?}"

echo "== list-devices =="
dev="$(/app/llama-server --list-devices 2>&1)"; echo "$dev"
echo "$dev" | grep -q 'Vulkan0' || { echo "FAIL: no Vulkan0 device"; exit 1; }

echo "== serve =="
/app/llama-server -m "$MODEL" -ngl 99 --host 127.0.0.1 --port 8080 >/tmp/srv.log 2>&1 &
for i in $(seq 1 60); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
grep -q 'offloaded .* layers to GPU' /tmp/srv.log || { echo "FAIL: no layer offload"; cat /tmp/srv.log; exit 1; }

echo "== completion =="
resp="$(curl -fsS http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":48,"temperature":0}')"
echo "$resp"
toks="$(printf '%s' "$resp" | sed -E 's/.*"predicted_per_second":([0-9.]+).*/\1/')"
echo "decode ${toks} tok/s (floor ${FLOOR})"
awk -v t="$toks" -v f="$FLOOR" 'BEGIN{exit !(t+0>=f+0)}' || { echo "FAIL: decode below floor"; exit 1; }
echo "PASS"
