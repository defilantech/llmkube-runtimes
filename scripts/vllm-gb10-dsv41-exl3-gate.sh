#!/bin/bash
# Tier-1 gate for llmkube-vllm-cuda-gb10-dsv41-exl3: re-asserts the Dockerfile's guarantees on the
# SHIPPED image. No CUDA kernel is launched; everything here runs on a driverless runner.
set -euo pipefail
IMAGE="${1:?usage: vllm-gb10-dsv41-exl3-gate.sh <image-ref>}"
D=/usr/local/lib/python3.12/dist-packages
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

echo "== receipts and attribution =="
run test -f /opt/llmkube/receipt.txt
run test -f /opt/llmkube/vllm-tree.sha256
run test -f /opt/llmkube/NOTICE
run bash -c '[[ "$(head -1 /opt/llmkube/LICENSE.tonyd2wild-MIT)" == "MIT License"* ]]'
run bash -c '[[ "$(head -1 /opt/llmkube/LICENSE.cuda-exl3-MIT)" == "MIT License"* ]]'
run bash -c '! grep -rl "GNU AFFERO" /opt/llmkube/patches'
echo "PASS: receipt, stamp, NOTICE, both MIT texts present; no AGPL text vendored"

echo "== FlashInfer is the in-place 0.7.0rc1 source build =="
run python3 -c 'import flashinfer, sys; sys.exit(0 if flashinfer.__version__ == "0.7.0rc1" else "wrong flashinfer: " + flashinfer.__version__)'
run bash -c "! pip show flashinfer-jit-cache >/dev/null 2>&1 && ! pip show flashinfer-cubin >/dev/null 2>&1"
echo "PASS: flashinfer 0.7.0rc1, AOT and cubin packages absent"

echo "== prewarmed JIT objects present =="
run bash -c 'ls /opt/llmkube/flashinfer-jit/.cache/flashinfer/*/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so'
echo "PASS: sparse_mla_sm120 prewarmed"

echo "== patched tree is the pinned tree =="
run python3 -c "
import importlib
e = importlib.import_module('vllm.models.deepseek_v4_1.common.engram')
assert not hasattr(e, 'gather_engram_hashes'), 'base engram.py survived'
importlib.import_module('vllm.models.deepseek_v4_1.virtual_heads')
print('ok')"
echo "PASS: overlay tree"

echo "== in-image test suites =="
run python3 -m pytest -q /opt/llmkube/tests
echo "PASS: Tier-1 gate complete for ${IMAGE}"
