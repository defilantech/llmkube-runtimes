# syntax=docker/dockerfile:1
#
# LLMKube AMD/Vulkan llama.cpp server runtime.
#
# Built from source so we own the Vulkan shader-gen step. The upstream floating
# :server-vulkan tag shipped a libggml-vulkan.so with an undefined shader symbol
# that silently fell back to CPU (defilantech/LLMKube#725); owning the build lets
# us catch and fix that class of break. See
# defilantech/LLMKube docs/proposals/697-amd-vulkan-runtime-image.md.
#
# Pinned inputs are bumped via reviewed PR. The CI Tier-1 gate (this image must
# load the Vulkan backend) and the out-of-band GPU promoter (real gfx1151 smoke)
# decide whether a bump is safe before anything consumes it.

# llama.cpp release tag + the exact commit it must resolve to (immutable pin).
ARG LLAMACPP_REF=b9663
ARG LLAMACPP_SHA=fdd109883df94789729cf3c411b709e0ecb129cc
# ubuntu:26.04 pinned by digest. 26.04's Mesa is new enough for gfx1151 (Strix
# Halo) RADV; Debian stable / older Ubuntu ship Mesa too old to enumerate it.
ARG BASE_DIGEST=sha256:f3d28607ddd78734bb7f71f117f3c6706c666b8b76cbff7c9ff6e5718d46ff64

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS build
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates \
      libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" https://github.com/ggml-org/llama.cpp.git . \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: ${LLAMACPP_REF} resolved to ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
       fi
# Mirror upstream's proven Vulkan recipe: dynamic backend loading + all CPU
# variants so the same image runs on any x86-64, with the Vulkan backend as a
# dlopen-able libggml-vulkan.so. LLAMA_CURL=OFF: models are staged locally by
# the LLMKube Model download Job, and the smoke gate runs fully offline.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/
# #725 guard, GPU-independent. The upstream break was an undefined shader symbol
# in libggml-vulkan.so that bound lazily, so it only crashed at shader-use time
# on a real GPU. RTLD_NOW forces every relocation to resolve at dlopen, so a
# build with that defect fails the build here instead of shipping.
RUN printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out /tmp/dlcheck /out/libggml-vulkan.so

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS server
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
LABEL org.opencontainers.image.title="llmkube-llama-vulkan" \
      org.opencontainers.image.description="LLMKube AMD/Vulkan llama.cpp server runtime" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}"
# RADV (mesa-vulkan-drivers) + the Vulkan loader. curl is only for the
# container HEALTHCHECK. No ROCm.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libvulkan1 mesa-vulkan-drivers curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r llama && useradd -r -g llama llama
COPY --from=build /out/ /app/
WORKDIR /app
# GGML_BACKEND_DL looks beside the executable; /app holds llama-server and the
# libggml-*.so backends.
ENV LLAMA_ARG_HOST=0.0.0.0
EXPOSE 8080
# Non-root. GPU access needs the deployment to grant the host render group via
# securityContext.supplementalGroups (the device plugin exposes /dev/dri).
USER llama
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["/app/llama-server"]
