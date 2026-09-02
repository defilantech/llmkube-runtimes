# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) vLLM runtime for Qwen3.8-Flash-Next (NVFP4).
# DRAFT 2026-09-02, from the ring measurement in
# llmkube-internal/research/2026-09-01-gb10-ring-flash-next-and-dsv4-vision-config-study.md
#
# WHY THIS IMAGE EXISTS
#
# The first-party vllm/vllm-openai:qwen38-flash-next image serves
# RadixArk/Qwen3.8-Flash-Next-NVFP4 across two Sparks (TP2 + expert parallel over
# RoCE) at prefill 2,789 tok/s / decode 48 tok/s single-stream with MTP 3, EXCEPT
# that it refuses to load the checkpoint's FP8 PLE n-gram shards: the PLE
# quant-method resolver only picks the FP8 path when the WHOLE checkpoint is an
# Fp8Config, and a ModelOpt-NVFP4 checkpoint excludes *.ple.* from its quant
# config (vllm-project/vllm#54765; still true on merged main). Without the shim it
# builds a BF16 table and dies.
#
# The community fix (MiaAI-Lab / getrefined dual-Spark recipes) is a 7-line resolver
# shim: with PLE_QUANT_OVERRIDE=fp8 in the environment, resolve the PLE embedding
# method to Qwen3_8FlashNextPLEFp8EmbeddingMethod regardless of the parent config.
# They bind-mount a patched ple_layer.py; here it is applied to the image's own
# file at build time and asserted, so the runtime has no host-path dependency.
#
# WHAT THIS IMAGE IS NOT. It tracks a pre-release model tag of vllm-openai whose
# vLLM is 0.1.dev20073+g8e685d198; upstream merged Qwen3.8-Flash-Next support in
# vllm#53896 (2026-08-31) but no tagged release carries it yet and the PLE gate is
# unfixed there too. When a tagged image loads FP8 PLE unaided, delete this variant.
#
# Base pinned by digest: vllm/vllm-openai:qwen38-flash-next-arm64-cu130 (2026-08-26,
# arm64 only, torch 2.13 cu130, NCCL 2.30.7).
ARG BASE_DIGEST=sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
FROM docker.io/vllm/vllm-openai:qwen38-flash-next-arm64-cu130@${BASE_DIGEST}

ARG PLE=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

# Apply the resolver shim to the shipped file. Anchored on the class that follows
# the stock resolver; a base bump that moves it fails the build here.
RUN python3 - "${PLE}" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = 'class Qwen3_8FlashNextNGramEmbedding(PleOffloadLayer):'
assert '_get_ple_embedding_quant_method' in s, 'stock PLE resolver not found; base changed'
assert anchor in s, 'shim anchor not found; base changed'
assert '_ORIG_PLE_QUANT_RESOLVER' not in s, 'shim already present in base (drop this layer)'
shim = ('_ORIG_PLE_QUANT_RESOLVER = _get_ple_embedding_quant_method\n\n'
        'def _get_ple_embedding_quant_method(quant_config, prefix):\n'
        '    """LLMKube shim (MiaAI-Lab recipe, vllm#54765): honor FP8 PLE shards under an NVFP4 parent config."""\n'
        '    from os import getenv as _ple_getenv\n'
        '    if _ple_getenv("PLE_QUANT_OVERRIDE", "").strip().lower() == "fp8":\n'
        '        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n'
        '    return _ORIG_PLE_QUANT_RESOLVER(quant_config, prefix)\n\n\n')
i = s.find(anchor); open(p, 'w').write(s[:i] + shim + s[i:])
print('PASS: PLE resolver shim applied to', p)
PY

# SHIM GUARD: the symbol the shim depends on must exist in the module, and the
# module must still import. Import is CPU-only for this file's module path.
RUN python3 -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('ple', '${PLE}'); src=open('${PLE}').read(); assert 'Qwen3_8FlashNextPLEFp8EmbeddingMethod' in src and '_ORIG_PLE_QUANT_RESOLVER' in src; print('PASS: shim symbols present')"

# The override is opt-in at runtime so the same image also serves FP8/BF16
# checkpoints unchanged. InferenceService spec sets: env PLE_QUANT_OVERRIDE=fp8,
# VLLM_USE_DEEP_GEMM=0; flags --enable-expert-parallel --kv-cache-dtype auto
# --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'.

LABEL org.opencontainers.image.title="llmkube-vllm-cuda-gb10-qwen38fn" \
      org.opencontainers.image.description="vLLM runtime for Qwen3.8-Flash-Next NVFP4 on GB10 (TP2+EP over RoCE): first-party qwen38-flash-next base + FP8-PLE resolver shim (vllm#54765), opt-in via PLE_QUANT_OVERRIDE=fp8." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.base.digest="${BASE_DIGEST}" \
      io.llmkube.vllm.patch="ple-fp8-resolver-shim"
