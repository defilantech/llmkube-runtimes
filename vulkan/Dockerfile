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
# b10103 (2026-07-23) is the first release after poolside Laguna support merged
# upstream (2026-07-22), so this image can serve Laguna-family GGUFs. The build
# asserts the laguna arch actually compiled in (below); the on-hardware Laguna
# smoke on gfx1151 is the real proof that its op sequence runs on Vulkan.
ARG LLAMACPP_REF=b10103
ARG LLAMACPP_SHA=c588c4f47683e73ad2d69f50480bec6cc85fd0f7
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
# Dynamic backend loading, Vulkan as a dlopen-able libggml-vulkan.so, and a
# single generic-x86-64 CPU backend (GGML_NATIVE=OFF). We intentionally do NOT
# set GGML_CPU_ALL_VARIANTS: this is a GPU image (the Vulkan backend does the
# work), and the ~15 extra per-microarch CPU variants only add build time,
# image size, and peak memory, the last of which exhausted the CI runner and
# made vulkan-shaders-gen fail to fork. LLAMA_CURL=OFF: models are staged
# locally by the LLMKube Model download Job, and the smoke gate runs offline.
# LLAMA_BUILD_UI/LLAMA_USE_PREBUILT_UI=OFF: llama.cpp otherwise embeds the
# browser chat UI, fetched as a prebuilt tarball keyed by a git-describe build
# number. Our --depth 1 clone has no tag history, so that number resolves to a
# nonexistent version and hard-fails the build (tools/ui: llama-ui-embed). These
# runtime images are consumed as inference backends (clients hit /v1/*, not a
# browser), so skip the UI: fewer build-time network deps, no drift exposure.
# Mirrors rocm/Dockerfile, which already carries this fix.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out /out-tools \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/ \
    && cp build/bin/llama-bench build/bin/llama-cli /out-tools/
# Laguna guard, GPU-independent. The llama.cpp arch-name table compiles the
# string "laguna" into the binary only when the pin includes upstream laguna
# support. A pin that predates the 2026-07-22 merge would build cleanly but
# silently lack the arch; this fails the build instead of shipping an image
# that cannot load Laguna GGUFs.
RUN grep -qa laguna /out/llama-server \
    || { echo "ERROR: laguna arch not compiled into llama-server; LLAMACPP pin predates upstream laguna support"; exit 1; }
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

# Tooling image: the same Vulkan build, plus llama-bench and llama-cli for
# benchmarking and diagnostics on real hardware. Built from the SAME build stage
# (identical llama.cpp commit + backends as the server image) so its numbers are
# directly comparable to what the server runtime does. Kept as a separate image
# so the production server image stays minimal (server binary only). Placed
# before the server stage so the server image remains the default build target.
FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS tools
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
LABEL org.opencontainers.image.title="llmkube-llama-vulkan-tools" \
      org.opencontainers.image.description="LLMKube AMD/Vulkan llama.cpp tooling (llama-bench, llama-cli, llama-server) for benchmarking and diagnostics" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}"
# Same RADV (mesa-vulkan-drivers) + Vulkan loader runtime as the server image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libvulkan1 mesa-vulkan-drivers curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r llama && useradd -r -g llama llama
# /out = libggml-*.so backends + llama-server; /out-tools = llama-bench + llama-cli.
# GGML_BACKEND_DL looks beside the executable, so all binaries and the backends
# share /app.
COPY --from=build /out/ /app/
COPY --from=build /out-tools/ /app/
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
# Non-root, same as the server image. GPU access needs the deployment to grant
# the host render group via securityContext.supplementalGroups (the device
# plugin exposes /dev/dri). One-shot tools: no server port or HEALTHCHECK.
USER llama
ENTRYPOINT ["/app/llama-bench"]

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
