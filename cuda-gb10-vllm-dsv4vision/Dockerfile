# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) vLLM runtime for DeepSeek-V4-Flash-Vision-Exp.
# DRAFT 2026-09-02, from the ring measurement in
# llmkube-internal/research/2026-09-01-gb10-ring-flash-next-and-dsv4-vision-config-study.md
#
# WHY THIS IMAGE EXISTS
#
# The first-party vllm/vllm-openai:deepseekv4-flash-vision image (built from the
# open PR vllm-project/vllm#54566 branch) serves this model on two Sparks over
# RoCE at TP2, EXCEPT that its FlashInfer (0.6.18) sm120 sparse-MLA prefill
# dispatcher has no instantiation for the Vision variant's widened 512-entry
# causal-SWA candidate buffer. Every prefill dies with
#   Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=32 topk=512
#   page_block_size=64 topk_extra=512 extra_page_block_size=64        (compress-4 layers)
#   ... topk_extra=128 extra_page_block_size=2                        (compress-128 layers)
#
# Two facts make this a 400-byte fix rather than a fork:
#   1. FlashInfer ships the kernel SOURCE in the wheel (flashinfer/data/csrc) and
#      JIT-compiles it with the image's nvcc 13.0 at first use.
#   2. It only does so when no prebuilt AOT artifact exists at
#      flashinfer_jit_cache/jit_cache/<module>/<module>.so. The image ships one
#      for sparse_mla_sm120, which silently shadows any source change.
#
# So: patch the source with flashinfer-ai/flashinfer#4850 (adds the 512/512/page-64
# arm, validated upstream on two GB10s) plus the compress-128 arm it is missing
# (512 primary, page-2 extra; topk_extra is a runtime argument of that launcher),
# and delete the one AOT .so. Measured on the ring 2026-09-02, TP2 + EP, fp8 KV,
# DSpark k=3: prefill 2,047 tok/s, decode 39.1 tok/s single-stream (llama.cpp RPC
# ring: 357 / 19.4), vision verified end to end.
#
# WHAT THIS IMAGE IS NOT. It tracks an unmerged vLLM branch and an open FlashInfer
# PR. When #54566 and #4850 merge and a tagged vllm-openai image carries them, this
# variant should be deleted and spec.image pointed at that tag. Same discipline as
# cuda-gb10-qwen4exp (retired when llama.cpp#27742 merged at b10660).
#
# Base pinned by digest: vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130 as of
# 2026-09-01 (vllm 0.28.1rc1.dev137+g5ab628dd1, flashinfer 0.6.18+cu130, torch
# 2.13.0+cu130, NCCL 2.29.7). arm64 only, by design.
ARG BASE_DIGEST=sha256:8568b4bbc821903d93a0a9c17dd80382fdc0ba78eaa128e3eb5cb71c3bf06b79
FROM docker.io/vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130@${BASE_DIGEST}

ARG FI_CSRC=/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc
ARG FI_AOT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so

# The patched dispatcher: upstream main at 2026-09-01 + PR #4850 + the compress-128
# arm. Kept as a whole file, not a diff, so the build never depends on fuzz.
COPY sparse_mla_sm120_prefill.cu ${FI_CSRC}/sparse_mla_sm120_prefill.cu

# PATCH GUARD. Assert the two arms this image exists for are in the shipped
# source, and that the shadowing AOT artifact is gone. A base bump that moves the
# file or renames the module must fail here, not on a Spark at profile time.
RUN set -e; \
    test -f "${FI_CSRC}/sparse_mla_sm120_prefill.cu"; \
    n=$(grep -c 'topk == 512 && topk_extra == 512 && extra_page_block_size == 64' "${FI_CSRC}/sparse_mla_sm120_prefill.cu"); \
    m=$(grep -c 'topk == 512 && extra_page_block_size == 2' "${FI_CSRC}/sparse_mla_sm120_prefill.cu"); \
    if [ "$n" -lt 2 ] || [ "$m" -lt 2 ]; then echo "ERROR: expected PR4850 (${n}) and compress-128 (${m}) dual arms for NH 32 and 64"; exit 1; fi; \
    test -f "${FI_AOT}" || { echo "ERROR: AOT artifact not at ${FI_AOT}; base layout changed, re-verify JitSpec.is_aot"; exit 1; }; \
    rm -f "${FI_AOT}"; \
    python3 - <<'PY'
import flashinfer, os
from flashinfer.jit.mla import gen_sparse_mla_sm120_module
spec = gen_sparse_mla_sm120_module()
assert not spec.is_aot, "sparse_mla_sm120 still resolves to an AOT artifact"
print("PASS: sparse_mla_sm120 will JIT from patched source; flashinfer", flashinfer.__version__)
PY

# Optional, deliberately NOT done here: pre-building the module at image build
# time (FLASHINFER_CUDA_ARCH_LIST=12.1a, no GPU needed for nvcc) would remove the
# ~2 min first-start JIT. Left for a follow-up once the CI runner's nvcc path is
# verified, because a silent build-time miss would just fall back to JIT anyway.

LABEL org.opencontainers.image.title="llmkube-vllm-cuda-gb10-dsv4vision" \
      org.opencontainers.image.description="vLLM runtime for DeepSeek-V4-Flash-Vision-Exp on GB10 (TP2 over RoCE): first-party deepseekv4-flash-vision base + FlashInfer sm120 sparse-MLA prefill arms (PR #4850 + compress-128). Tracks unmerged vllm#54566 / flashinfer#4850." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.base.digest="${BASE_DIGEST}" \
      io.llmkube.flashinfer.patch="flashinfer#4850+compress128"
