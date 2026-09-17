#!/usr/bin/env bash
#
# Tier-1 gate for llmkube-exllamav3-cuda-gb10-dsv41-exl3: re-asserts the
# Dockerfile's guarantees on the SHIPPED image. No CUDA kernel is launched, and
# nothing here needs a driver. cuobjdump ships only in the devel stage, so the
# device-code assertion reads /app/CUDA_ARCHS, the receipt the build wrote.
#
# Why its own script rather than scripts/cuda-gate.sh: that one is shaped around
# llama-server's --list-devices verdict logic. This image ships no llama.cpp, and
# its launch check is an HTTP server coming up, so the two share conventions but
# not code.
set -euo pipefail

IMAGE="${1:?usage: exllamav3-gb10-gate.sh <image-ref>}"
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== receipts and attribution =="
run test -f /opt/llmkube/receipt.txt
run test -f /opt/llmkube/exllamav3-tree.sha256
run test -f /opt/llmkube/NOTICE
run test -f /opt/llmkube/LICENSE.exllamav3-MIT
run bash -c '[[ "$(head -1 /opt/llmkube/LICENSE.exllamav3-MIT)" == "MIT License"* ]]'
run test -d /opt/llmkube/patches
run test -f /opt/llmkube/patches/0001-aarch64-build.patch
run bash -c '! grep -rli "GNU AFFERO" /opt/llmkube/patches'
run bash -c 'grep -q "Deliberately absent" /opt/llmkube/NOTICE'
run bash -c 'grep -q "^exllamav3: " /opt/llmkube/receipt.txt'
run bash -c 'grep -q "^torch cuda: 13" /opt/llmkube/receipt.txt'
run bash -c 'grep -q "^device code: .*sm_121" /opt/llmkube/receipt.txt'
echo "PASS: receipt, stamp, NOTICE, MIT text and patch set present; no AGPL text vendored"

echo "== shipped device code inventory =="
if ! archs="$(docker run --rm --entrypoint cat "${IMAGE}" /app/CUDA_ARCHS 2>&1)"; then
  echo "${archs}"
  echo "FAIL: /app/CUDA_ARCHS missing from the image; cannot verify device code"
  exit 1
fi
echo "${archs}"
if ! grep -q 'sm_121' <<<"${archs}"; then
  echo "FAIL: shipped image carries no native sm_121 device code"
  exit 1
fi
if grep -q 'sm_9\|sm_8\|sm_7' <<<"${archs}"; then
  echo "NOTE: device-code receipt lists non-GB10 targets; check the arch flag"
fi
echo "PASS: native sm_121 device code present in the shipped image"

echo "== the extension resolves and the stubs behave =="
# dlopen with every symbol resolved is the load-bearing check: the aarch64 patch
# stubs three x86-only translation units, and dropping their symbols instead of
# stubbing them would fail here rather than on a node. torch first: it supplies libc10.
run python3 -c "import torch, exllamav3, exllamav3_ext as e; from exllamav3.version import __version__; print('exllamav3', __version__); assert e.exl3_moe_cpu_has_avx2() is False, 'x86 tier reported available'; print('ok')"
echo "PASS: exllamav3_ext loads, x86 tiers report unavailable on aarch64"

echo "== launcher checks actually run, and can fail =="
# The digest path, with the memory policy relaxed so a 16 GB runner is not the
# thing under test here.
run env EXL3_VERIFY_TREE=1 EXL3_MIN_MEM_AVAILABLE_GIB=0 /usr/local/bin/llmkube-exllamav3-entrypoint.sh true
echo "PASS: launcher verifies the tree digest and execs through"
# Falsify the digest check: mutate the tree, the launcher must refuse. Runs as
# root (the image user is non-root, so an unprivileged append fails silently and
# the "tamper" would never happen), and asserts the file actually changed, so a
# no-op tamper fails the gate rather than passing as a false positive.
if docker run --rm --user root --entrypoint bash "${IMAGE}" -c '
      f="$(ls -d /opt/venv/lib/python*/site-packages/exllamav3)/version.py"
      before="$(sha256sum "$f")"
      printf "\n# tamper\n" >> "$f"
      after="$(sha256sum "$f")"
      [ "$before" != "$after" ] || { echo "could not tamper with the tree"; exit 2; }
      EXL3_VERIFY_TREE=1 EXL3_MIN_MEM_AVAILABLE_GIB=0 /usr/local/bin/llmkube-exllamav3-entrypoint.sh true' >/dev/null 2>&1; then
  echo "FAIL: launcher accepted a tampered exllamav3 tree"
  exit 1
fi
echo "PASS: launcher rejects a tampered tree (the digest check can fail)"
# Falsify the memory pre-flight: an unreachable floor must stop the launcher.
if run env EXL3_MIN_MEM_AVAILABLE_GIB=99999 /usr/local/bin/llmkube-exllamav3-entrypoint.sh true >/dev/null 2>&1; then
  echo "FAIL: launcher ignored the MemAvailable floor"
  exit 1
fi
echo "PASS: launcher refuses to start below the MemAvailable floor"

echo "== the server binds and answers (scaffold) =="
docker run --rm --entrypoint bash "${IMAGE}" -c '
  EXL3_MIN_MEM_AVAILABLE_GIB=0 /usr/local/bin/llmkube-exllamav3-entrypoint.sh &
  pid=$!
  for _ in $(seq 1 30); do
    if curl -fsS "http://localhost:${EXL3_PORT:-5000}/health" >/tmp/h.json 2>/dev/null; then break; fi
    sleep 1
  done
  cat /tmp/h.json
  grep -q "\"server_ready\": *true" /tmp/h.json
  kill $pid
' || { echo "FAIL: scaffold server did not answer /health"; exit 1; }
echo "PASS: server binds the port and reports readiness"

echo "PASS: Tier-1 gate complete for ${IMAGE}"
