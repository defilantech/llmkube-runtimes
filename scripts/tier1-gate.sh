#!/usr/bin/env bash
#
# Tier-1 runtime gate (cheap, no GPU required).
#
# The primary #725 guard is in the build stage: an RTLD_NOW dlopen of
# libggml-vulkan.so that fails the build on any unresolved symbol. This runtime
# gate is the complementary check that the slim runtime image can actually
# launch llama-server and load its backends without error (e.g. a libvulkan
# missing from the runtime layer would surface here).
#
# A GPU-less CI runner legitimately enumerates no accelerator device and emits
# no info-level "loaded backend" lines, so we gate on the ABSENCE of load
# errors, not on a positive device. Real GPU offload is verified by the
# out-of-band promoter (Tier-2).
set -euo pipefail

IMAGE="${1:?usage: tier1-gate.sh <image-ref>}"

echo "== llama-server --list-devices =="
if ! out="$(docker run --rm --entrypoint /app/llama-server "${IMAGE}" --list-devices 2>&1)"; then
  echo "${out}"
  echo "FAIL: llama-server --list-devices exited non-zero"
  exit 1
fi
echo "${out}"

echo "== verdict =="
if echo "${out}" | grep -qiE 'symbol lookup error|failed to load|undefined symbol|cannot (open|read)'; then
  echo "FAIL: backend load error in the runtime image"
  exit 1
fi
echo "PASS: server launches and backends load without error"

# The tools image bundles llama-bench alongside the same backends. Confirm it is
# present and launches (links its backends) too. No-op for the server image,
# which ships no llama-bench.
if docker run --rm --entrypoint test "${IMAGE}" -f /app/llama-bench 2>/dev/null; then
  echo "== llama-bench --help (tools image) =="
  if ! docker run --rm --entrypoint /app/llama-bench "${IMAGE}" --help >/dev/null 2>&1; then
    echo "FAIL: llama-bench is present but failed to launch"
    exit 1
  fi
  echo "PASS: llama-bench launches and links its backends"
fi
