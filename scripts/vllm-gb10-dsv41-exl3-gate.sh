#!/bin/bash
# Tier-1 gate for llmkube-vllm-cuda-gb10-dsv41-exl3: re-asserts the Dockerfile's guarantees on the
# SHIPPED image. No CUDA kernel is launched; everything here runs on a driverless runner.
set -euo pipefail
IMAGE="${1:?usage: vllm-gb10-dsv41-exl3-gate.sh <image-ref>}"
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== receipts and attribution =="
run test -f /opt/llmkube/receipt.txt
run test -f /opt/llmkube/vllm-tree.sha256
run test -f /opt/llmkube/NOTICE
run bash -c '[[ "$(head -1 /opt/llmkube/LICENSE.tonyd2wild-MIT)" == "MIT License"* ]]'
run bash -c '[[ "$(head -1 /opt/llmkube/LICENSE.cuda-exl3-MIT)" == "MIT License"* ]]'
run test -f /opt/llmkube/LICENSE.cutlass-BSD-3-Clause
run test -f /opt/llmkube/LICENSE.spdlog-MIT
run test -f /opt/llmkube/LICENSE.cccl-Apache-2.0-LLVM
run bash -c "grep -q 'Redistribution and use in source and binary forms' /opt/llmkube/LICENSE.cutlass-BSD-3-Clause"
run bash -c "grep -q 'MIT License' /opt/llmkube/LICENSE.spdlog-MIT"
run bash -c "grep -q 'Apache License' /opt/llmkube/LICENSE.cccl-Apache-2.0-LLVM"
run test -d /opt/llmkube/patches
run bash -c '! grep -rli "GNU AFFERO" /opt/llmkube/patches'
echo "PASS: receipt, stamp, NOTICE, all five license texts present; no AGPL text vendored"

echo "== FlashInfer is the in-place 0.7.0rc1 source build =="
run python3 -c 'import flashinfer, sys; sys.exit(0 if flashinfer.__version__ == "0.7.0rc1" else "wrong flashinfer: " + flashinfer.__version__)'
run bash -c "! pip show flashinfer-jit-cache >/dev/null 2>&1 && ! pip show flashinfer-cubin >/dev/null 2>&1"
echo "PASS: flashinfer 0.7.0rc1, AOT and cubin packages absent"

echo "== prewarmed JIT objects present =="
run bash -c 'ls /opt/llmkube/flashinfer-jit/.cache/flashinfer/*/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so'
run bash -c 'ls /opt/llmkube/flashinfer-jit/.cache/flashinfer/*/121a/cached_ops/mxfp8_gemm_cutlass_sm120/mxfp8_gemm_cutlass_sm120.so'
echo "PASS: sparse_mla_sm120 and mxfp8_gemm_cutlass_sm120 prewarmed"

echo "== patched tree is the pinned tree =="
run python3 -c "
import importlib
e = importlib.import_module('vllm.models.deepseek_v4_1.common.engram')
assert not hasattr(e, 'gather_engram_hashes'), 'base engram.py survived'
importlib.import_module('vllm.models.deepseek_v4_1.virtual_heads')
print('ok')"
echo "PASS: overlay tree"

run bash -c 'DSV41_VERIFY_TREE=1 /usr/local/bin/dsv41-entrypoint.sh true'
echo "PASS: entrypoint verifies the vLLM tree digest"

run python3 -c 'import torch, cuda_exl3, cuda_exl3._C; print("cuda_exl3 kernel loads")'

echo "== in-image test suites =="
run bash -c 'DSV41_IN_IMAGE=1 python3 -m pytest -q /opt/llmkube/tests'
echo "PASS: Tier-1 gate complete for ${IMAGE}"
