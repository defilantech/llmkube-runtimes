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

ARG TURBO_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
# feature/turboquant-kv-cache tip. Now back on TheTom's canonical repo: our two
# Vulkan turbo4 fixes are merged upstream (PR #204, TheTom@337f08e8) -- the
# 66/68-byte block_turbo4_0 layout mismatch (drops the removed rnorm field) and
# the stale Lloyd-Max centroid tables (sync to the re-derived C reference),
# proven by test-backend-ops on gfx1151 (528/528 turbo4 FA FAIL -> 9509/9509
# sweep PASS). This tip also carries the server context-checkpoint sidecar for
# slot save/restore (PR #206), relevant to long-context GLM re-prefill latency.
ARG TURBO_REF=3d42fa1eb6e9f35c80ac86084637f574c3c764db
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
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    && cmake --build build -j"$(nproc)" \
         --target llama-bench llama-server llama-perplexity \
    && mkdir -p /out \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-bench build/bin/llama-server build/bin/llama-perplexity /out/ \
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
