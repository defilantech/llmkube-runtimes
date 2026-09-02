#!/usr/bin/env bash
#
# Tier-1 gate for cuda-gb10-vllm-dsv4vision (cheap, no GPU required).
#
# The image exists for exactly two facts, so the gate asserts exactly those two
# on the SHIPPED image rather than trusting the Dockerfile's own RUN guards:
#   1. FlashInfer's sm120 sparse-MLA prefill dispatcher carries the DeepSeek-V4
#      Vision arms (flashinfer#4850's 512/512 page-64 arm and the compress-128
#      512 page-2 arm), for both 32 and 64 query heads.
#   2. No prebuilt AOT artifact shadows that source: FlashInfer JIT-compiles a
#      module only when flashinfer_jit_cache/jit_cache/<name>/<name>.so is absent.
# Both are string/path checks; nothing here needs CUDA. GPU serving is Tier-2.
set -euo pipefail

IMAGE="${1:?usage: vllm-dsv4vision-gate.sh <image-ref>}"
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

CU=/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu
AOT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so

echo "== dispatcher arms in the shipped source =="
if ! src="$(run cat "${CU}" 2>&1)"; then
  echo "${src}"; echo "FAIL: ${CU} missing; the FlashInfer layout changed"; exit 1
fi
n64="$(printf '%s\n' "${src}" | grep -c 'DISPATCH_DUAL_MG_CM(BF16, [36][24], 512, 64, 2)' || true)"
n2="$(printf '%s\n' "${src}" | grep -c 'DISPATCH_DUAL_MG_CM(BF16, [36][24], 512, 2, 2)' || true)"
echo "page-64 Vision dual arms (compress-4):  ${n64}  (want 2: NH 32 and 64)"
echo "page-2 Vision dual arms (compress-128): ${n2}   (want 2: NH 32 and 64)"
if [ "${n64}" -lt 2 ] || [ "${n2}" -lt 2 ]; then
  echo "FAIL: the Vision prefill arms are not in the shipped dispatcher"; exit 1
fi
echo "PASS: both Vision dual-cache arms present"

echo "== AOT artifact must be absent so the source is compiled =="
if run test -e "${AOT}" 2>/dev/null; then
  echo "FAIL: ${AOT} still exists; FlashInfer would load it and ignore the patched source"; exit 1
fi
echo "PASS: no AOT sparse_mla_sm120 artifact"

echo "== loader agrees (AOT path absent under FLASHINFER_AOT_DIR) and nvcc is present for the JIT =="
# Path test only; the module generator probes for a CUDA arch and raises on a GPU-less runner.
if ! out="$(run python3 -c 'import os,sys; from flashinfer.jit.env import FLASHINFER_AOT_DIR; p=os.path.join(str(FLASHINFER_AOT_DIR),"sparse_mla_sm120","sparse_mla_sm120.so"); print("aot_path", p, "exists", os.path.exists(p)); sys.exit(1 if os.path.exists(p) else 0)' 2>&1)"; then
  echo "${out}"; echo "FAIL: the loader would still resolve sparse_mla_sm120 to an AOT artifact"; exit 1
fi
echo "${out}"
if ! run /usr/local/cuda/bin/nvcc --version >/dev/null 2>&1; then
  echo "FAIL: nvcc missing; the JIT build at first start would fail"; exit 1
fi
echo "PASS: loader will JIT from source; nvcc available"
