#!/usr/bin/env bash
#
# Tier-1 runtime gate for the GB10 CUDA image (cheap, no GPU required).
#
# The build stage already carries the two hard guards: native sm_121 device code
# must exist in libggml-cuda.so, and that .so must dlopen under RTLD_NOW against
# the driver stub. This script is the complementary check on the SHIPPED image:
# that the slim runtime layer can actually launch the binaries, and that the
# device code survived the COPY from the build stage.
#
# Why this is not scripts/tier1-gate.sh (the Vulkan/ROCm one): that script fails
# on any "failed to load" line. A GPU-less runner legitimately produces one for
# CUDA, because there is no device for cuInit to find. Gating on the absence of
# ALL load errors would either fail every CUDA build or, if loosened globally,
# stop catching real Vulkan defects. So CUDA gets its own verdict logic, which
# allowlists exactly the no-device case and nothing else.
set -euo pipefail

IMAGE="${1:?usage: cuda-gate.sh <image-ref>}"

# --- 1. Device code survived the build -> runtime COPY -------------------------
#
# /app/CUDA_ARCHS is the receipt cuobjdump wrote in the build stage. cuobjdump
# itself ships only in the CUDA devel image, so this file is how anyone holding
# the final runtime image can confirm what device code it contains. It also
# catches the specific mistake of copying the wrong stage's /out, which would
# otherwise produce a plausible-looking image with no GB10 kernels.
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
echo "PASS: native sm_121 device code present in the shipped image"

# --- 2. The server launches and links its backends ----------------------------
echo "== llama-server --list-devices =="
if ! out="$(docker run --rm --entrypoint /app/llama-server "${IMAGE}" --list-devices 2>&1)"; then
  echo "${out}"
  echo "FAIL: llama-server --list-devices exited non-zero"
  exit 1
fi
echo "${out}"

echo "== verdict =="
# Real defects: an unresolved symbol, an ABI/version mismatch, or a missing
# library. The one expected absence on a GPU-less runner is the driver
# (libcuda.so.1), which the node supplies at runtime via the NVIDIA container
# runtime and which no CUDA image may ship. Strip only that line, then apply the
# same strictness the Vulkan gate uses to whatever remains.
# grep -v returns 1 when it filters everything out, which is a legitimate
# result here, so it must not trip `set -e`. Here-strings rather than pipes
# throughout: `... | grep -q` lets grep exit first, sends SIGPIPE upstream, and
# under `set -o pipefail` turns a passing check into a failing one.
defects='symbol lookup error|undefined symbol|version .* not found|cannot (open|read) shared object'
residual="$(grep -viE 'libcuda\.so\.1|no CUDA devices|no usable GPU|cuInit|CUDA driver' <<<"${out}" || true)"
if grep -qiE "${defects}" <<<"${residual}"; then
  echo "FAIL: backend load error in the runtime image (not attributable to the absent driver)"
  grep -iE "${defects}" <<<"${residual}" || true
  exit 1
fi
echo "PASS: server launches with no load error beyond the expected absent driver"

# --- 3. The tools image's extra binaries launch too ---------------------------
# No-op for the server image, which ships no llama-bench.
if docker run --rm --entrypoint test "${IMAGE}" -f /app/llama-bench 2>/dev/null; then
  echo "== llama-bench --help (tools image) =="
  if ! docker run --rm --entrypoint /app/llama-bench "${IMAGE}" --help >/dev/null 2>&1; then
    echo "FAIL: llama-bench is present but failed to launch"
    exit 1
  fi
  echo "PASS: llama-bench launches and links its backends"
fi
