# syntax=docker/dockerfile:1
#
# EXPERIMENT image. NOT a production runtime, NOT Tier-1/Tier-2 gated, NOT
# consumed by the operator. The supported runtimes are the server/tools images
# in vulkan/. This builds the third-party TurboQuant llama.cpp fork
# (github.com/TheTom/llama-cpp-turboquant) for Vulkan/RADV so we can benchmark,
# on AMD Strix Halo (gfx1151):
#   - TurboQuant asymmetric KV-cache quant (K q8 / V turbo3-4)
#   - MTP (the fork wires the path upstream llama-bench does not)
#   - EAGLE3 speculative decoding
#
# The fork tip is fetched live (the point is the just-synced EAGLE3/MTP/cohere2
# support); the resolved commit is recorded at /app/TURBO_SHA in the image.

# TEMPORARY: repointed at our fork to benchmark the mul_mat_vec_id work on
# gfx1151 before proposing it upstream. RESTORE THE TheTom REPO/REF BELOW ONCE
# THAT BENCHMARK IS DONE.
ARG TURBO_REPO=https://github.com/Defilan/llama.cpp.git
# feature/turboquant-kv-cache tip. Our two Vulkan turbo4 fixes are merged
# upstream (PR #204) -- the 66/68-byte block_turbo4_0 layout mismatch (drops the
# removed rnorm field) and the stale Lloyd-Max centroid tables (sync to the
# re-derived C reference), proven by test-backend-ops on gfx1151 (528/528 turbo4
# FA FAIL -> 9509/9509 sweep PASS).
#
# Bumped to the merge of TheTom#259 (2026-08-04), which brings the TurboQuant
# WEIGHT types up on Vulkan: TQ4_1S host wiring restored (it was lost in a
# rebase -- the shaders from 2a716ac4 were present but never passed to glslc,
# so every TQ4_1S MUL_MAT fell back to CPU) and TQ3_1S dequant + mul_mat_vec
# kernels added. Validated on this node: full test-backend-ops 24496/24496,
# and prefill on a TQ4_1S GGUF goes 1.19 -> 461 t/s because the matmuls are no
# longer CPU-resident. This tip also carries DeepSeek-V4 arch support
# (PR #224/#254), so a Config-I GGUF is now servable here.
#
# TEMPORARY: vulkan/tq-mul-mat-vec-id on Defilan/llama.cpp. Adds
# mul_mat_vec_id for TQ3_1S/TQ4_1S so MoE decode runs on the GPU, and replaces
# the blanket MUL_MAT_ID reject with a maxStorageBufferRange size gate (prompt
# processing still has no TQ mul_mm_id and stages the whole expert tensor as
# f16). Upstream ref to restore: 21eab71bc6c5f6c8194e6494006f25d979d8b6c4.
# set_rows/cpy wiring is still absent either way.
ARG TURBO_REF=86bf90f004eb11bbc8f6fe899a6eaeebe8feba8a
# ubuntu:26.04 pinned by digest, same base as the production Vulkan image
# (Mesa new enough for gfx1151 RADV).
ARG BASE_DIGEST=sha256:f3d28607ddd78734bb7f71f117f3c6706c666b8b76cbff7c9ff6e5718d46ff64

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS build
ARG TURBO_REPO
ARG TURBO_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build ca-certificates \
      libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q && git remote add origin "${TURBO_REPO}" \
    && git fetch -q --depth 1 origin "${TURBO_REF}" \
    && git checkout -q FETCH_HEAD \
    && git rev-parse HEAD > /tmp/TURBO_SHA \
    && echo "TurboQuant fork at $(cat /tmp/TURBO_SHA)"
# Same Vulkan build switches as the production image: dynamic backend loading,
# dlopen-able libggml-vulkan.so, single generic x86-64 CPU backend.
#
# LLAMA_BUILD_UI=OFF + LLAMA_USE_PREBUILT_UI=OFF are mandatory here for the same
# reason as the production image: this is a --depth 1 fetch, so there is no tag
# history for the UI's git-describe build number and the build hard-fails
# without them. (The original version of this file predated that fix.)
#
# LLAMA_BUILD_TESTS=ON so test-backend-ops exists. It is the only tool that
# compares a backend's kernels against the CPU reference per-op, which is how
# PR #204's turbo4 layout/centroid bugs were found and is the gate for the
# TQ3_1S/TQ4_1S work. Note upstream Vulkan CI explicitly skips it
# (`ctest -E test-backend-ops`, too slow on llvmpipe), so a real GPU is the only
# place it ever runs. The explicit --target list keeps ON from building the
# whole test suite.
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build -j"$(nproc)" \
         --target llama-bench llama-server llama-perplexity test-backend-ops \
    && mkdir -p /out \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-bench build/bin/llama-server build/bin/llama-perplexity /out/ \
    && cp build/bin/test-backend-ops /out/ \
    && cp /tmp/TURBO_SHA /out/TURBO_SHA
# #725-class guard (GPU-independent): force every relocation to resolve at
# dlopen so a broken libggml-vulkan.so fails the build instead of shipping.
RUN printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out /tmp/dlcheck /out/libggml-vulkan.so

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS tools
LABEL org.opencontainers.image.title="llmkube-llama-vulkan-turboquant" \
      org.opencontainers.image.description="EXPERIMENT: TurboQuant llama.cpp fork (Vulkan) for KV-quant/MTP/EAGLE3 benchmarking on gfx1151. Not a supported runtime." \
      org.opencontainers.image.source="https://github.com/TheTom/llama-cpp-turboquant" \
      org.opencontainers.image.licenses="MIT" \
      io.llmkube.experiment="turboquant-vulkan"
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libvulkan1 mesa-vulkan-drivers curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r llama && useradd -r -g llama llama
COPY --from=build /out/ /app/
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
EXPOSE 8080
# Non-root; GPU via securityContext.supplementalGroups + the /dev/dri device
# plugin. Entrypoint is llama-server so the LLMKube operator (which sets args
# only, not command, relying on the image entrypoint) can serve from this image
# via an image override + extraArgs such as --spec-type draft-mtp. The benchmark
# harness calls /app/llama-bench and /app/llama-perplexity by explicit path, so
# this entrypoint change does not affect benchmarking.
USER llama
ENTRYPOINT ["/app/llama-server"]
