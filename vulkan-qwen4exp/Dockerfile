# syntax=docker/dockerfile:1
#
# LLMKube AMD/Vulkan llama.cpp server runtime, Qwen3.8-Flash-Next (qwen4exp)
# with MTP speculative decoding.
#
# WHY THIS IMAGE EXISTS
#
# Qwen3.8-Flash-Next is arch `qwen4exp`, and the qwen4exp MTP draft graph is not
# upstream. Both statements were true when this variant was written; only the
# second still is.
#
#   * The ARCHITECTURE is upstream as of ggml-org/llama.cpp#27742, merged
#     2026-08-27 (commit 6c84c7d5), and the QSA correctness follow-ups from
#     #27941 are upstream too (merged 2026-09-01, commit 36b10154). Both are
#     ancestors of the pin below, so nothing here is a fork of the arch.
#   * The MTP DRAFT GRAPH is still an open pull request, ggml-org/llama.cpp
#     #28243, "models: Qwen3.8-Flash-Next MTP" (danielhanchen). Upstream's
#     common/ MTP machinery (`--spec-type draft-mtp`, the draft-mtp spec
#     implementation, `llama_set_embeddings_nextn`) is already present; what
#     #28243 adds is the qwen4exp-specific draft graph and its converter.
#
# So this image is a PINNED UPSTREAM TAG PLUS OUR OWN PATCH. It is not built
# from a fork, a third-party branch, or an open pull-request ref. The pull
# request's cumulative diff is vendored at patches/0001-qwen4exp-mtp.patch and
# the tag below is the exact commit that diff was authored against, which is
# why the patch applies cleanly and why the pin is a specific tag rather than
# the newest release. Bump both together, or neither.
#
# WHEN THIS IMAGE GOES AWAY
#
# When #28243 merges: delete patches/, delete this variant directory, and bump
# LLAMACPP_REF/LLAMACPP_SHA in vulkan/Dockerfile to a tag at or after the
# merge. Nothing else in this recipe is variant-specific.
#
# WHICH DRAFT HEAD, WHICH IS LOAD-BEARING
#
# Unsloth ship three families of MTP head for this model. USE THE SELF-CONTAINED
# ONE: MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf (~3.85 GiB).
#
# The `shared-` heads carry their own embeddings and borrow the rest from the
# running target. #28243 does not carry the loader-side borrowing path
# (`borrow_shared_tensor`), so a `shared-` head will not load on this build.
# Its absence is what the guard below is shaped to catch, not an assumption
# about the file layout.
#
# PROVENANCE AND LICENSE
#
# llama.cpp is MIT throughout and the vendored patch is a contribution to
# llama.cpp itself, so unlike the glm53-exl3 variant nothing here crosses a
# license boundary. See NOTICE for attribution and LICENSE.upstream-MIT.
#
# TEMPLATE: this file is vulkan/Dockerfile, plus the patch step. Keep the
# toolchain flags and their rationale in step with that file when either moves.

# Upstream llama.cpp, pinned by tag AND the exact commit it must resolve to.
# b10794 (2026-09-04) is the base ggml-org#28243 was authored against, so it is
# the only tag the vendored patch is known to apply to. It already contains
# both #27742 and #27941.
ARG LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMACPP_REF=b10794
ARG LLAMACPP_SHA=f9f09f02cc44d87d842dbd2d578857d92d4bb63b
# ubuntu:26.04 pinned by digest. 26.04's Mesa is new enough for gfx1151 (Strix
# Halo) RADV; Debian stable / older Ubuntu ship Mesa too old to enumerate it.
ARG BASE_DIGEST=sha256:f3d28607ddd78734bb7f71f117f3c6706c666b8b76cbff7c9ff6e5718d46ff64

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS build
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates \
      libvulkan-dev glslc spirv-headers \
      patch \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" "${LLAMACPP_REPO}" . \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: ${LLAMACPP_REF} resolved to ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
       fi
# The vendored MTP patch, applied to the tag it was authored against. --check
# first so a base drift fails on the guard's message rather than mid-apply with
# a reject file, and --whitespace=nowarn because the diff has a whitespace-only
# hunk (a blank line in llama-model-loader.h) that some git configs refuse.
COPY patches/0001-qwen4exp-mtp.patch /src/patches/0001-qwen4exp-mtp.patch
RUN git apply --check --whitespace=nowarn patches/0001-qwen4exp-mtp.patch \
    && git apply --whitespace=nowarn patches/0001-qwen4exp-mtp.patch \
    && echo "PASS: ggml-org#28243 (qwen4exp MTP) applied to ${LLAMACPP_REF}"
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
# nonexistent version and hard-fails the build (tools/ui: llama-ui-embed).
# Mirrors vulkan/Dockerfile and rocm/Dockerfile.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out /out-tools \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/ \
    && cp build/bin/llama-bench build/bin/llama-cli /out-tools/
# QWEN4EXP GUARD, the arch half. Same reasoning as cuda-gb10-qwen4exp: #27742
# adds no backend kernels, only an architecture definition over existing ops,
# so the question is host-side: did the arch compile into the shipped library,
# or did we build stock llama.cpp? The two are indistinguishable from the
# outside until a GGUF is rejected at load as an unknown architecture.
#
# Two independent strings so a partial build cannot pass: the arch name the
# loader matches the GGUF header against, and a PLE tensor name from the same
# PR (blk.%d.ple_conv1d), which is the half of this architecture that is
# genuinely new.
RUN found_arch=0; found_ple=0; \
    for lib in /out/libllama.so /out/libggml-base.so /out/libggml.so; do \
      [ -f "$lib" ] || continue; \
      if strings -a "$lib" | grep -qx 'qwen4exp'; then found_arch=1; fi; \
      if strings -a "$lib" | grep -q 'ple_conv1d'; then found_ple=1; fi; \
    done; \
    if ! strings -a /out/libllama.so | grep -q .; then \
      echo "ERROR: strings produced no output for /out/libllama.so."; \
      echo "       This guard cannot tell whether qwen4exp is present, so it is"; \
      echo "       failing rather than passing blind."; \
      exit 1; \
    fi; \
    if [ "$found_arch" != "1" ]; then \
      echo "ERROR: architecture name 'qwen4exp' absent from the shipped libraries."; \
      echo "       LLAMACPP_REF (${LLAMACPP_REF}) predates ggml-org#27742."; \
      exit 1; \
    fi; \
    if [ "$found_ple" != "1" ]; then \
      echo "ERROR: arch 'qwen4exp' is registered but PLE tensor names are absent."; \
      exit 1; \
    fi; \
    echo "PASS: qwen4exp architecture and PLE tensors registered"
# QWEN4EXP MTP GUARD, THE REASON THIS VARIANT EXISTS. This is the guard that
# has to be right, because its failure mode is silent: a llama.cpp without the
# qwen4exp draft graph loads the model, serves it correctly, and generates at
# the target's own speed with no error, no warning, and no log line. Nothing
# downstream can tell that apart from "this model is not much faster than the
# dense baseline", which on a first boot reads as a hardware result rather than
# a packaging bug.
#
# "QWEN4EXP MTP:" is a throw message #28243 adds to src/models/qwen4exp.cpp,
# so it lands in libllama.so only when that PR's draft graph is compiled in. It
# is deliberately preferable to a `--spec-type` check: `--spec-type draft-mtp`
# and the draft-mtp implementation are ALREADY upstream without #28243 (they
# serve other architectures), so that flag being accepted proves nothing about
# this model.
RUN if ! strings -a /out/libllama.so | grep -q 'QWEN4EXP MTP:'; then \
      echo "ERROR: the qwen4exp MTP draft graph is absent from libllama.so."; \
      echo "       This image would serve Qwen3.8-Flash-Next at the target's own"; \
      echo "       speed with no error. patches/0001-qwen4exp-mtp.patch did not"; \
      echo "       apply, or the pin moved off ${LLAMACPP_SHA}."; \
      exit 1; \
    fi \
    && echo "PASS: qwen4exp MTP draft graph present"
# ARG-SURFACE GUARD. The InferenceService will pass `--spec-type draft-mtp` and
# `-md`; a build whose argument parser rejects either aborts llama-server at
# startup, which shows up as a crashloop on a node rather than as a build
# failure here. --help is the parser's own opinion, so ask it.
# Output is captured to a file and read, not piped: `llama-server --help | grep
# -q` exits at the first match and takes the server's SIGPIPE, so under any
# pipefail the pipeline reports failure on a PASSING build. Docker RUN uses
# plain `sh` with no pipefail today, but a guard whose correctness depends on
# which shell it inherits is one SHELL directive away from inverting.
RUN /out/llama-server --help > /tmp/help.txt 2>&1 || true; \
    if [ ! -s /tmp/help.txt ]; then \
      echo "ERROR: llama-server --help produced no output; cannot verify the arg surface."; \
      exit 1; \
    fi; \
    for flag in --spec-type --spec-draft-n-max; do \
      if ! grep -q -- "$flag" /tmp/help.txt; then \
        echo "ERROR: llama-server does not accept ${flag}."; \
        echo "       The InferenceService sets spec.speculativeDecoding, which renders"; \
        echo "       these flags; without them the server aborts at startup."; \
        exit 1; \
      fi; \
    done \
    && echo "PASS: --spec-type and --spec-draft-n-max accepted by the parser"
# #725 guard, GPU-independent, identical to vulkan/Dockerfile. The upstream
# break was an undefined shader symbol in libggml-vulkan.so that bound lazily,
# so it only crashed at shader-use time on a real GPU. RTLD_NOW forces every
# relocation to resolve at dlopen, so a build with that defect fails the build
# here instead of shipping.
RUN printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out /tmp/dlcheck /out/libggml-vulkan.so

FROM docker.io/ubuntu:26.04@${BASE_DIGEST} AS server
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
LABEL org.opencontainers.image.title="llmkube-llama-vulkan-qwen4exp" \
      org.opencontainers.image.description="LLMKube AMD/Vulkan llama.cpp server runtime for Qwen3.8-Flash-Next (qwen4exp). Pinned upstream tag plus the vendored ggml-org#28243 MTP draft graph; uses the self-contained MTP head." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.llamacpp.repo="${LLAMACPP_REPO}" \
      dev.llmkube.patchset="ggml-org/llama.cpp#28243 (qwen4exp MTP)"
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
