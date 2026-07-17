# syntax=docker/dockerfile:1
#
# LLMKube AMD/ROCm (HIP) llama.cpp server runtime for gfx1151 (Strix Halo).
# Per-model opt-in tier alongside the Vulkan default (defilantech/LLMKube#701).
# Same llama.cpp pin as the Vulkan image so benchmarks compare like vs like.
# Tier-1 gate: libggml-hip.so must dlopen(RTLD_NOW) in CI (no GPU needed).
# Tier-2: the out-of-band promoter smokes it on real gfx1151.

ARG LLAMACPP_REF=b9663
ARG LLAMACPP_SHA=fdd109883df94789729cf3c411b709e0ecb129cc
ARG ROCM_VERSION=7.2.4
# Digests recorded in Step 2 (replace before commit):
ARG ROCM_DEV_DIGEST=sha256:92f309c51b52cef8762867848f1529dee821624f23cd5df38455e819538f762f
ARG UBUNTU_DIGEST=sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
ARG GPU_TARGETS=gfx1151

FROM docker.io/rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete@${ROCM_DEV_DIGEST} AS build
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG GPU_TARGETS
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates rocwmma-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" https://github.com/ggml-org/llama.cpp.git . \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: ${LLAMACPP_REF} resolved to ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
       fi
# GGML_HIP_NO_VMM is mandatory on gfx1151 (HIP VMM is broken there and failures
# masquerade as driver errors). rocWMMA FlashAttention is the long-context win.
# LLAMA_BUILD_UI/LLAMA_USE_PREBUILT_UI=OFF: llama.cpp otherwise embeds the
# browser chat UI, fetched as a prebuilt tarball from an HF bucket keyed by a
# git-describe build number. Our --depth 1 clone below has no tag history, so
# that number resolves to a nonexistent bucket version; the fallback to
# "latest" then mismatches this pinned commit's asset manifest and hard-fails
# the build. These runtime images are consumed as inference backends (the
# operator/API clients hit /v1/*, not a browser), so skip the UI entirely:
# fewer build-time network dependencies and no drift exposure.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_HIP=ON -DGGML_BACKEND_DL=ON \
      -DGPU_TARGETS="${GPU_TARGETS}" \
      -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out /out-tools \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/ \
    && cp build/bin/llama-bench build/bin/llama-cli /out-tools/
# RTLD_NOW guard (same rationale as the Vulkan #725 guard): force every
# relocation at dlopen so a broken HIP backend fails the BUILD, not the node.
RUN printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out:/opt/rocm/lib /tmp/dlcheck /out/libggml-hip.so

# Shared runtime base: ubuntu:24.04 + only the ROCm runtime libraries, from
# AMD's apt repo pinned to the same ROCM_VERSION as the builder.
FROM docker.io/ubuntu:24.04@${UBUNTU_DIGEST} AS runtime-base
ARG ROCM_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
       > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      hip-runtime-amd rocblas hipblaslt rocm-device-libs \
    && apt-get purge -y curl gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
ENV ROCBLAS_USE_HIPBLASLT=1 \
    LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib

FROM runtime-base AS tools
COPY --from=build /out/ /usr/local/lib/
COPY --from=build /out-tools/ /usr/local/bin/
RUN mv /usr/local/lib/llama-server /usr/local/bin/llama-server
ENTRYPOINT ["/usr/local/bin/llama-bench"]

FROM runtime-base AS server
COPY --from=build /out/ /usr/local/lib/
RUN mv /usr/local/lib/llama-server /usr/local/bin/llama-server
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
