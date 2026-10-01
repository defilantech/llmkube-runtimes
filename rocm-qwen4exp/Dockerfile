# syntax=docker/dockerfile:1
#
# LLMKube AMD/ROCm (HIP) llama.cpp server runtime for gfx1151 (Strix Halo),
# Qwen3.8-Flash-Next (qwen4exp) with MTP plus the open upstream depth fixes
# that apply to our pin.
#
# WHY THIS IMAGE EXISTS
#
# It is the HIP half of a HIP-vs-Vulkan comparison on one Strix Halo. The
# Vulkan half is vulkan-qwen4exp: same llama.cpp tag, same vendored
# ggml-org#28243 MTP patch. This image adds, on top of that:
#
#   * 0010, ggml-org#28213: gather-based QSA sparse attention for decode
#     (attention cost bounded by the top-k width instead of context depth).
#   * 0011, ggml-org#28699: incremental pooled-key cache for the QSA indexer.
#   * 0012, ggml-org#29030: direct-read gather of lazy tensor rows (the PLE /
#     n-gram table), backported with context-only changes.
#
# All three are backend-agnostic src/ changes, so they reach the HIP backend.
# Which PRs were considered, which applied, their head commits, and exactly
# what the 0012 backport changes are recorded in patches/APPLIED.md. The
# HIP-relevant #29751 does NOT apply to b10794 and is not here; rebasing the
# pin for it is separate work. Because the depth patches are here and not in
# vulkan-qwen4exp, a HIP-vs-Vulkan number from these two images compares
# backends AND patch sets; the depth patches carry their own runtime kill
# switches (QWEN4EXP_QSA_GATHER=0, LLAMA_QSA_NO_POOLED_CACHE=1) for an
# MTP-only arm on this same binary.
#
# KNOWN DEFECT IN 0011, READ BEFORE SERVING MORE THAN ONE SLOT
#
# ggml-org#28699 is a draft. Its pooled rows are keyed by position block with
# no sequence identity, so two live sequences sharing ONE unified KV stream
# overwrite each other's rows and one of them answers from the other's
# context (reported on the PR 2026-09-20, unfixed at the vendored head).
# llama-server's default (--parallel auto) is exactly that: 4 slots with
# kv_unified on. Run this image with --parallel 1, or --no-kv-unified (the
# pooled cache then disables itself on multi-stream), or
# LLAMA_QSA_NO_POOLED_CACHE=1. llama-perplexity's PPL/KLD path does not set
# kv_unified, so it is unaffected unless -kvu is passed.
#
# TEMPLATE: this file is rocm/Dockerfile (base images, ROCm version, HIP
# flags, runtime libraries) plus the vulkan-qwen4exp patch step and guards.
# Layout follows vulkan-qwen4exp (/app) so scripts/qwen4exp-gate.sh runs
# against this image unchanged.

ARG LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMACPP_REF=b10794
ARG LLAMACPP_SHA=f9f09f02cc44d87d842dbd2d578857d92d4bb63b
ARG ROCM_VERSION=7.2.4
# Same digests as rocm/Dockerfile.
ARG ROCM_DEV_DIGEST=sha256:92f309c51b52cef8762867848f1529dee821624f23cd5df38455e819538f762f
ARG UBUNTU_DIGEST=sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
ARG AMDGPU_TARGETS=gfx1151

FROM docker.io/rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete@${ROCM_DEV_DIGEST} AS build
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG AMDGPU_TARGETS
# No rocwmma-dev, unlike rocm/Dockerfile: b10794 removed the
# GGML_HIP_ROCWMMA_FATTN option and ggml-cuda no longer includes rocWMMA.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates binutils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" "${LLAMACPP_REPO}" . \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: ${LLAMACPP_REF} resolved to ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
       fi
# The vendored patches, applied in order to the tag they were checked against.
# --check first so a base drift fails on the guard's message rather than
# mid-apply, and --whitespace=nowarn because 0001 has a whitespace-only hunk
# that some git configs refuse. Each later patch was checked against the tree
# the earlier ones leave, so the order is load-bearing.
COPY patches/ /src/patches/
RUN git apply --check --whitespace=nowarn patches/0001-qwen4exp-mtp.patch \
    && git apply --whitespace=nowarn patches/0001-qwen4exp-mtp.patch \
    && echo "PASS: ggml-org#28243 (qwen4exp MTP) applied to ${LLAMACPP_REF}"
RUN git apply --check --whitespace=nowarn patches/0010-qwen4exp-qsa-gather-decode.patch \
    && git apply --whitespace=nowarn patches/0010-qwen4exp-qsa-gather-decode.patch \
    && echo "PASS: ggml-org#28213 (QSA gather decode) applied"
RUN git apply --check --whitespace=nowarn patches/0011-qwen4exp-pooled-key-cache.patch \
    && git apply --whitespace=nowarn patches/0011-qwen4exp-pooled-key-cache.patch \
    && echo "PASS: ggml-org#28699 (QSA pooled-key cache) applied"
RUN git apply --check --whitespace=nowarn patches/0012-lazy-direct-row-reads.patch \
    && git apply --whitespace=nowarn patches/0012-lazy-direct-row-reads.patch \
    && echo "PASS: ggml-org#29030 (lazy direct row reads, backport) applied"
# HIP flags from rocm/Dockerfile, minus GGML_HIP_ROCWMMA_FATTN (gone in
# b10794). GGML_HIP_NO_VMM is mandatory on gfx1151 (HIP VMM is broken there
# and failures masquerade as driver errors); it is the default at b10794 but
# stays explicit so a default flip cannot silently re-enable VMM.
# GGML_HIP_MMQ_MFMA only affects CDNA and is kept for parity with rocm/.
# AMDGPU_TARGETS is forwarded to GPU_TARGETS by ggml-hip's CMakeLists.
# LLAMA_BUILD_UI/LLAMA_USE_PREBUILT_UI=OFF: see rocm/Dockerfile; the --depth 1
# clone has no tag history, so the prebuilt UI lookup hard-fails the build.
# Targets: llama-server (the runtime) and llama-perplexity (KL divergence
# for the bake-off's quality arm), from the same build.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_HIP=ON -DGGML_BACKEND_DL=ON \
      -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
      -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-perplexity \
    && mkdir -p /out \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server build/bin/llama-perplexity /out/ \
    && test -f /out/libggml-hip.so
# QWEN4EXP GUARD, the arch half: the architecture name and a PLE tensor name,
# so a partial build cannot pass. Same guard as vulkan-qwen4exp.
RUN found_arch=0; found_ple=0; \
    for lib in /out/libllama.so /out/libggml-base.so /out/libggml.so; do \
      [ -f "$lib" ] || continue; \
      if strings -a "$lib" | grep -qx 'qwen4exp'; then found_arch=1; fi; \
      if strings -a "$lib" | grep -q 'ple_conv1d'; then found_ple=1; fi; \
    done; \
    if ! strings -a /out/libllama.so | grep -q .; then \
      echo "ERROR: strings produced no output for /out/libllama.so; failing rather than passing blind."; \
      exit 1; \
    fi; \
    if [ "$found_arch" != "1" ]; then \
      echo "ERROR: architecture name 'qwen4exp' absent from the shipped libraries."; \
      exit 1; \
    fi; \
    if [ "$found_ple" != "1" ]; then \
      echo "ERROR: arch 'qwen4exp' is registered but PLE tensor names are absent."; \
      exit 1; \
    fi; \
    echo "PASS: qwen4exp architecture and PLE tensors registered"
# PATCH GUARD: every vendored patch leaves a string in libllama.so that stock
# b10794 does not have (listed in patches/APPLIED.md). A missing one means the
# build tree is not the patched tree, and the MTP one in particular fails
# silently at serve time: the model runs at the target's own speed with no
# error. Checked here on the build tree and again on the final image by
# scripts/rocm-qwen4exp-gate.sh.
RUN strings -a /out/libllama.so > /tmp/libllama.strings; \
    rc=0; \
    for m in 'QWEN4EXP MTP:' 'QWEN4EXP_QSA_GATHER' 'LLAMA_QSA_NO_POOLED_CACHE' 'row reads enabled'; do \
      if grep -q -- "$m" /tmp/libllama.strings; then echo "PASS: marker '$m' present"; \
      else echo "ERROR: marker '$m' absent from libllama.so; its patch is not in this build."; rc=1; fi; \
    done; \
    exit $rc
# ARG-SURFACE GUARD, as in vulkan-qwen4exp: --help is read from a file, never
# a pipe, so grep's early exit cannot SIGPIPE the server under pipefail.
RUN /out/llama-server --help > /tmp/help.txt 2>&1 || true; \
    if [ ! -s /tmp/help.txt ]; then \
      echo "ERROR: llama-server --help produced no output; cannot verify the arg surface."; \
      exit 1; \
    fi; \
    for flag in --spec-type --spec-draft-n-max; do \
      if ! grep -q -- "$flag" /tmp/help.txt; then \
        echo "ERROR: llama-server does not accept ${flag}."; \
        exit 1; \
      fi; \
    done \
    && echo "PASS: --spec-type and --spec-draft-n-max accepted by the parser"
# RTLD_NOW guard (the #725 guard, adapted to the HIP backend as rocm/ does):
# force every relocation at dlopen so a broken HIP backend fails the BUILD,
# not the node. The binary is also shipped (see the server stage) so the gate
# script can repeat the check against the slim runtime stage, which is where a
# missing ROCm runtime library would actually show up.
RUN printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out:/opt/rocm/lib /tmp/dlcheck /out/libggml-hip.so

# Runtime: ubuntu:24.04 plus only the ROCm runtime libraries, from AMD's apt
# repo pinned to the builder's ROCM_VERSION. Identical to rocm/Dockerfile's
# runtime-base; see that file for why hipblas is required.
FROM docker.io/ubuntu:24.04@${UBUNTU_DIGEST} AS server
ARG ROCM_VERSION
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
LABEL org.opencontainers.image.title="llmkube-llama-rocm-qwen4exp" \
      org.opencontainers.image.description="LLMKube AMD/ROCm (HIP, gfx1151) llama.cpp server for Qwen3.8-Flash-Next (qwen4exp). Pinned upstream tag plus vendored ggml-org#28243 MTP and the #28213, #28699, #29030 depth patches; ships llama-server and llama-perplexity." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.llamacpp.repo="${LLAMACPP_REPO}" \
      dev.llmkube.patchset="ggml-org/llama.cpp#28243 #28213 #28699 #29030(backport)"
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
       > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      hip-runtime-amd rocblas hipblas hipblaslt rocm-device-libs \
      libgomp1 \
    && apt-get purge -y gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
ENV ROCBLAS_USE_HIPBLASLT=1 \
    LD_LIBRARY_PATH=/opt/rocm/lib:/app \
    LLAMA_ARG_HOST=0.0.0.0
# GGML_BACKEND_DL looks beside the executable: /app holds llama-server,
# llama-perplexity and the libggml-*.so backends together.
COPY --from=build /out/ /app/
# The patch set and its manifest, so the gate (and anyone holding only the
# image) can check what was applied without the source repo.
COPY --from=build /src/patches/ /usr/share/llmkube/rocm-qwen4exp/patches/
# The RTLD_NOW checker from the build stage (a few KB, same glibc as this
# base), so the gate can re-run the dlopen guard against THIS stage's ROCm
# runtime libraries from any docker host, with no compiler on the host.
COPY --from=build /tmp/dlcheck /usr/libexec/llmkube/dlcheck
WORKDIR /app
EXPOSE 8080
# Runs as root, like rocm/Dockerfile, the hardware-validated ROCm image: HIP
# needs /dev/kfd as well as /dev/dri, and the non-root group mapping for kfd
# is unproven on our nodes. Going non-root is a follow-up, not part of this
# comparison.
ENTRYPOINT ["/app/llama-server"]
