#!/usr/bin/env bash
#
# Tier-1 runtime gate for the GB10 vLLM GLM-5.3-Flash EXL3 image (no GPU).
#
# This image IS built from source here, unlike its NVFP4 sibling, so the gate
# asks build-outcome questions rather than base-drift ones:
#
#   1. the EXL3 quantization method is not actually registered, in which case
#      routed experts expand to BF16 and the model cannot fit at all;
#   2. the sparse-MLA NoPE padding did not land, so the first forward dies on
#      `pe_dim must be 64 for fp8_ds_mla`;
#   3. the CUDA extension built but is missing the fused/fat MoE entry points,
#      so it serves at a fraction of the measured prefill;
#   4. AGPL-era material or abliteration artifacts crept into an Apache-2.0
#      image;
#   5. the build stamp does not describe the shipped tree.
#
# Everything runs against the SHIPPED image, not the build stage, which is the
# check that survives a cache hit or a reordered layer.
#
# NOTHING HERE LAUNCHES A KERNEL. Importing torch and exllamav3_ext loads
# shared objects and is safe without a device, but no CUDA work is dispatched: a
# hosted runner has no GPU and a failure there would say nothing about the
# image. The extension must be imported AFTER torch or it cannot find libc10.so.
set -euo pipefail

IMAGE="${1:?usage: vllm-gb10-glm53-exl3-gate.sh <image-ref>}"

VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm

# The image ENTRYPOINT expects a command to exec, so probes override it.
run() { docker run --rm --entrypoint "$1" "${IMAGE}" "${@:2}"; }

# --- 1. EXL3 is registered, not merely present -------------------------------
echo "== EXL3 quantization method =="
if ! run test -f "${VLLM_SITE}/model_executor/layers/quantization/exl3.py"; then
  echo "FAIL: exl3.py is not installed into vLLM's quantization package."
  exit 1
fi
# Registering the file is not enough; vLLM only offers a method that appears in
# the QuantizationMethods Literal and in the lazy method_to_config map.
if ! run grep -q '"exl3"' "${VLLM_SITE}/model_executor/layers/quantization/__init__.py"; then
  echo "FAIL: \"exl3\" is not in vLLM's QuantizationMethods registry."
  echo "      --quantization exl3 would be rejected and routed experts would"
  echo "      fall back to BF16, which does not fit on 2x GB10."
  exit 1
fi
echo "PASS: exl3.py installed and registered in QuantizationMethods"

# --- 2. The NoPE sparse-MLA padding landed -----------------------------------
# This is the edit that makes the checkpoint loadable at all. Assert on the
# patched source rather than on a version string.
echo "== sparse-MLA NoPE padding =="
MLA="${VLLM_SITE}/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
if ! run grep -q 'self.rope_pad = 64' "${MLA}"; then
  echo "FAIL: the NoPE rope_pad edit is not in ${MLA}."
  echo "      GLM-5.3-Flash would die on the first forward with"
  echo "      'pe_dim must be 64 for fp8_ds_mla'."
  exit 1
fi
if ! run grep -q 'supports_dense_mha_prefill = False' "${MLA}"; then
  echo "FAIL: supports_dense_mha_prefill was not set False; prefill would take"
  echo "      a dense path this backend cannot serve."
  exit 1
fi
echo "PASS: sparse-MLA padding and prefill flag are present"

# --- 3. The CUDA extension carries every fused/fat entry point ---------------
echo "== exllamav3_ext symbols =="
# `import torch` FIRST is required, not stylistic. exllamav3_ext links against
# libtorch, so importing it in a bare interpreter dies with
# `ImportError: libc10.so: cannot open shared object file` before any symbol is
# looked at. Importing torch loads that library into the process. Upstream's
# build-time check has the same two-step; dropping it here produced a gate
# failure that read exactly like a missing-kernel failure on an image whose
# kernels were fine.
if ! out="$(run python3 -c '
import torch  # noqa: F401  loads libc10.so so the extension can link
import exllamav3_ext, sys
want = ("exl3_moe", "exl3_fat_gemm", "exl3_fat_gemm_scatter",
        "exl3_fat_moe_gateup", "exl3_fat_moe_down", "exl3_fat_moe_gather")
missing = [s for s in want if not hasattr(exllamav3_ext, s)]
sys.exit("missing: " + ",".join(missing) if missing else 0)
' 2>&1)"; then
  echo "${out}"
  echo "FAIL: the compiled extension is missing fused/fat MoE entry points."
  echo "      E2/E3 prefill would not run; this is not the validated runtime."
  exit 1
fi
echo "PASS: exllamav3_ext exposes all six fused/fat MoE entry points"

# --- 3b. GB10 persistent_topk is disabled --------------------------------------
# This one is not theoretical. persistent_topk needs >=128KB shared memory per
# block; GB10 has 101376 B. With it enabled the engine loads all 120 shards,
# forms TP=2, and then dies inside determine_available_memory, which reads as a
# late mystery crash rather than an unsupported kernel. The patch that disables
# it runs at build; this asserts the result on the shipped image, because the
# patch was absent from the build for a while and every other check stayed green.
echo "== GB10 persistent_topk disabled =="
KPOOL="${VLLM_SITE}/model_executor/layers/sparse_attn_indexer_kpool.py"
if ! run grep -q 'if False and current_platform.is_cuda() and select_k in (512, 1024, 2048)' "${KPOOL}"; then
  echo "FAIL: persistent_topk is still enabled in ${KPOOL}."
  echo "      This image dies on a GB10 during determine_available_memory."
  exit 1
fi
if ! run test -f "${VLLM_SITE%/vllm}/glm53_video.pth"; then
  echo "FAIL: glm53_video.pth is missing; the video placeholder import hook"
  echo "      will not load in the serving interpreter."
  exit 1
fi
echo "PASS: persistent_topk disabled and the video hook .pth is installed"

# --- 4. License and policy boundary ------------------------------------------
# The Apache-2.0 story of this image depends on AGPL-era files never appearing.
# That is testable, so test it on the artifact rather than trusting the build.
echo "== license boundary =="
if run bash -c 'ls /opt/glm53/patch_adaptive_k.py /opt/glm53/patch_dense_fp8.py 2>/dev/null | grep -q .'; then
  echo "FAIL: AGPL-3.0 overlay patches are present in an Apache-2.0 image."
  exit 1
fi
if run bash -c 'ls -d /opt/glm53/ablit /opt/glm53/patch_ablit.py /opt/glm53/ablit_runtime.py 2>/dev/null | grep -q .'; then
  echo "FAIL: abliteration artifacts are present; this image ships without them."
  exit 1
fi
if ! run test -f /opt/llmkube/LICENSE.upstream-MIT; then
  echo "FAIL: upstream MIT notice is not in the image; attribution is a license condition."
  exit 1
fi
if ! run test -f /opt/llmkube/NOTICE; then
  echo "FAIL: NOTICE is missing from the image."
  exit 1
fi
echo "PASS: no AGPL-era patches, no abliteration, attribution notices shipped"

# --- 5. The build stamp describes the shipped tree ---------------------------
# Drives the verifier rather than reading it: recompute in the container and
# require agreement. A stamp that does not match its own image is worse than no
# stamp, because the runtime check would then reject a good image.
echo "== build stamp matches the shipped tree =="
if ! out="$(run bash -c '
set -eu
want="$(cat /opt/llmkube/vllm-tree.sha256)"
got="$(find '"${VLLM_SITE}"' -name "*.py" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d" " -f1)"
[ "$want" = "$got" ] || { echo "stamp=$want tree=$got"; exit 1; }
echo "$got"
' 2>&1)"; then
  echo "${out}"
  echo "FAIL: the recorded vLLM tree digest does not match the shipped image."
  exit 1
fi
echo "PASS: tree digest ${out}"

# --- 6. ABLIT is refused ------------------------------------------------------
# Behavioural, not textual: set it and require a non-zero exit.
echo "== ABLIT refused =="
if docker run --rm -e ABLIT=1 "${IMAGE}" true >/dev/null 2>&1; then
  echo "FAIL: ABLIT=1 was accepted; the guard is decorative."
  exit 1
fi
if ! docker run --rm "${IMAGE}" true >/dev/null 2>&1; then
  echo "FAIL: the entrypoint refused to start in the default configuration."
  exit 1
fi
echo "PASS: ABLIT=1 refused, default start works"

echo
echo "Tier-1 gate PASSED for ${IMAGE}"
echo "Still unproven without a GB10: that the checkpoint loads, that TP=2 forms"
echo "over CX7, and the decode/prefill numbers. That is the out-of-band GPU smoke."
