# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) llama.cpp server runtime.
#
# WHY THIS IMAGE EXISTS
#
# Upstream's prebuilt ghcr.io/ggml-org/llama.cpp:server-cuda-* has an arm64
# manifest, so a DGX Spark pulls it and serves happily. It is also, on GB10,
# entirely JIT-compiled. Inspecting the shipped libggml-cuda.so from
# server-cuda-b10068 with cuobjdump (2026-08-10):
#
#   native cubins : sm_86, sm_89, sm_120a
#   PTX           : sm_50, sm_61, sm_70, sm_75, sm_80, sm_90
#
# GB10 is sm_121. The "a" in sm_120a means architecture-SPECIFIC: a 120a cubin
# does not load on 121. So there is no native code for this GPU at all, and
# every kernel is JIT-compiled at startup from the sm_90 (Hopper) PTX.
#
# MEASURED on a GB10, 2026-08-11, rather than assumed. A controlled run isolated
# codegen from everything else: one pod, one CUDA 13 toolchain, one llama.cpp
# commit (b10355), one GPU, one 16.8GB model, swapping only libggml-cuda.so
# between 121a-real and 90-virtual, CUDA JIT cache cleared before every run:
#
#   native sm_121a :  2.54 / 2.54 / 2.33 s   to "model loaded"
#   sm_90 PTX      : 20.14 / 21.21 / 20.12 s
#
# So JIT costs ~18s on every process start. It does NOT cost steady-state
# throughput, and an earlier draft of this comment was wrong to imply it did:
# against upstream's own image at a near-identical llama.cpp pin, prefill
# differed by +0.6% and decode by -0.1%, both inside run-to-run noise. JIT'd
# SASS runs at full speed once compiled. What you pay for is compiling it, on
# every pod start. The first request pays again, because CUDA loads modules
# lazily: first-request prefill measured 239 tok/s on JIT vs 785 native, roughly
# 14s of extra latency on a 4700-token prompt.
#
# The cause is upstream's toolkit version, not an oversight: llama.cpp's default
# arch list only appends 121a-real at CUDA >= 12.9 (ggml/src/ggml-cuda/
# CMakeLists.txt), and the upstream image is built on CUDA 12.8.x, so the branch
# never fires. Building on CUDA 13 with an explicit target fixes it.
#
# This image therefore targets exactly one GPU and asserts it (see the codegen
# guard below). The rest of the fleet is unaffected: the RTX 5060 Ti pair in
# shadowstack is sm_120 on amd64, which upstream already covers natively with
# its sm_120a cubin and which could not pull an arm64 image anyway.
#
# Gates, same two tiers as the Vulkan and ROCm images: Tier-1 is CI (no GPU
# needed) and is the codegen + dlopen guards below plus scripts/cuda-gate.sh;
# Tier-2 is the out-of-band promoter smoking the candidate on a real GB10.

# llama.cpp release tag + the exact commit it must resolve to (immutable pin).
# Deliberately the SAME pin as vulkan/Dockerfile so GB10 and Strix Halo numbers
# compare like vs like; rocm/Dockerfile carries the same rule for the same
# reason. Bump both together.
ARG LLAMACPP_REF=b10355
ARG LLAMACPP_SHA=dd1ea524333b1e697489067d7a4c39c60d32beee

# GB10 is sm_121. "121a-real" = architecture-specific device code, no PTX.
# Deliberately no virtual/PTX target: a PTX fallback would let this image load
# on non-GB10 hardware by silently JIT-compiling, which is the exact failure
# mode the image exists to remove. Better to fail loudly at load.
ARG CUDA_ARCH=121a-real

# CUDA 13.0.1 pinned by index digest (resolves per-platform; this image is
# built arm64-only). 13.0 is required, not merely convenient: 121a needs
# CUDA >= 12.9. Verified on a Spark that nvcc 13.0.88 accepts -arch=sm_121a and
# emits an sm_121a cubin. Node driver at time of writing is 580.173.02.
ARG CUDA_DEVEL_DIGEST=sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d
ARG CUDA_RUNTIME_DIGEST=sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

FROM docker.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@${CUDA_DEVEL_DIGEST} AS build
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG CUDA_ARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" https://github.com/ggml-org/llama.cpp.git . \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: ${LLAMACPP_REF} resolved to ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
       fi
# CMAKE_CUDA_ARCHITECTURES is set explicitly so llama.cpp's toolkit-version
# heuristic never runs: that heuristic is what produces a JIT-only GB10 build,
# and letting it decide is how this class of bug ships silently.
#
# GGML_NATIVE=OFF keeps the CPU backend generic (the build runner is not the
# serving node). GGML_BACKEND_DL=ON matches the Vulkan and ROCm images: the CUDA
# backend becomes a dlopen-able libggml-cuda.so found beside the executable.
# LLAMA_CURL=OFF because models are staged locally by the LLMKube Model download
# Job and the gate runs offline. LLAMA_BUILD_UI/LLAMA_USE_PREBUILT_UI=OFF
# because llama.cpp otherwise fetches a prebuilt browser-chat-UI tarball keyed
# by a git-describe build number, which a --depth 1 clone cannot produce; the
# Vulkan and ROCm images carry the same two flags for the same reason. These
# runtimes are consumed as inference backends over /v1/*, never via a browser.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out /out-tools \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/ \
    && cp build/bin/llama-bench build/bin/llama-cli /out-tools/

# CODEGEN GUARD. The whole point of this image, asserted rather than assumed.
#
# A wrong toolkit, a dropped -DCMAKE_CUDA_ARCHITECTURES, or an upstream change
# to the default arch list all produce a build that succeeds, an image that
# runs, and a GB10 quietly back on JIT. None of that is visible without looking
# at the fatbin, so look at the fatbin: fail the build unless native sm_121
# device code is actually present. GPU-independent, so it runs in CI.
#
# Also asserts the inverse: no PTX. PTX in this image would mean some target
# reintroduced a silent JIT fallback. The inventory is echoed either way so the
# build log records exactly what shipped.
# cuobjdump output is captured to files and the assertions read the files, not a
# pipeline. `cuobjdump ... | grep -q` looks equivalent and is not: grep exits at
# the first match, cuobjdump takes SIGPIPE, and under `set -o pipefail` the
# pipeline reports failure on a PASSING build. Docker RUN uses plain `sh` with
# no pipefail so the pipeline form happens to work here, but a guard whose
# correctness depends on which shell it inherits is one `SHELL` directive away
# from inverting itself, and this is the guard the image exists for.
# Also note `cuobjdump --list-ptx` exits non-zero when there is no PTX, which is
# the PASSING case; `|| true` keeps that from failing the build on its own.
RUN cuobjdump --list-elf /out/libggml-cuda.so > /tmp/elf.txt 2>/dev/null || true; \
    cuobjdump --list-ptx /out/libggml-cuda.so > /tmp/ptx.txt 2>/dev/null || true; \
    echo "== device code inventory =="; \
    grep -oE 'sm_[0-9]+[a-z]*' /tmp/elf.txt | sort | uniq -c; \
    echo "== PTX inventory (expected: empty) =="; \
    grep -oE 'sm_[0-9]+[a-z]*' /tmp/ptx.txt | sort | uniq -c; \
    if ! grep -q 'sm_121' /tmp/elf.txt; then \
      echo "ERROR: no native sm_121 device code in libggml-cuda.so."; \
      echo "       GB10 would JIT-compile every kernel at startup, which is exactly"; \
      echo "       what this image exists to prevent. Check CUDA_ARCH (${CUDA_ARCH})"; \
      echo "       and that the CUDA toolkit is >= 12.9."; \
      exit 1; \
    fi; \
    if grep -q 'sm_[0-9]' /tmp/ptx.txt; then \
      echo "ERROR: PTX present in libggml-cuda.so. A virtual/PTX target reintroduces"; \
      echo "       the silent-JIT path on non-GB10 hardware. Use *-real targets only,"; \
      echo "       or update this guard deliberately."; \
      exit 1; \
    fi; \
    echo "PASS: native sm_121 device code present, no PTX fallback"

# RTLD_NOW guard, same rationale as the Vulkan #725 guard and its ROCm twin:
# force every relocation to resolve at dlopen so a backend with an unresolved
# symbol fails the BUILD rather than binding lazily and crashing on a node at
# shader/kernel-use time.
#
# libggml-cuda.so links libcuda.so.1, which is the DRIVER and is installed by
# the host, not by any CUDA image: `ldconfig -p` in this base lists no libcuda at
# all, so without help the dlopen below fails for an uninteresting reason and
# proves nothing. The devel image ships a link stub at
# /usr/local/cuda/lib64/stubs/libcuda.so for exactly this purpose, but under the
# bare name, so symlink the SONAME the loader actually searches for. Verified
# present in 13.0.1-devel; if a future base drops it the ln fails and this guard
# fails loudly rather than quietly passing on a dangling link.
#
# The stub lives only in this build stage and is never copied into the shipped
# runtime image, where a fake driver would satisfy the loader and then fail in
# confusing ways at cuInit.
RUN test -f /usr/local/cuda/lib64/stubs/libcuda.so \
    && mkdir -p /tmp/stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 \
    && printf '%s\n' \
      '#include <dlfcn.h>' '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);' \
      '  if (!h) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }' \
      '  printf("dlopen OK: %s\n", argv[1]); return 0; }' > /tmp/dlcheck.c \
    && gcc /tmp/dlcheck.c -o /tmp/dlcheck -ldl \
    && LD_LIBRARY_PATH=/out:/tmp/stub:/usr/local/cuda/lib64 /tmp/dlcheck /out/libggml-cuda.so

# Record the shipped device-code inventory as a file so the runtime gate and
# anyone holding only the final image can verify what it contains without a
# CUDA toolkit. cuobjdump exists only in the devel stage; this is its receipt.
RUN cuobjdump --list-elf /out/libggml-cuda.so | grep -oE 'sm_[0-9]+[a-z]*' | sort -u > /out/CUDA_ARCHS \
    && cat /out/CUDA_ARCHS

# Tooling image: the same build, plus llama-bench and llama-cli. Built from the
# SAME build stage (identical llama.cpp commit and codegen as the server image)
# so its benchmark numbers describe what the server runtime actually does.
# Placed before the server stage so the server image stays the default target.
FROM docker.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@${CUDA_RUNTIME_DIGEST} AS tools
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG CUDA_ARCH
LABEL org.opencontainers.image.title="llmkube-llama-cuda-gb10-tools" \
      org.opencontainers.image.description="LLMKube NVIDIA GB10 (DGX Spark) llama.cpp tooling (llama-bench, llama-cli, llama-server), native sm_121" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.cuda.arch="${CUDA_ARCH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r llama && useradd -r -g llama llama
# /out = libggml-*.so backends + llama-server; /out-tools = llama-bench +
# llama-cli. GGML_BACKEND_DL looks beside the executable, so all binaries and
# the backends share /app.
COPY --from=build /out/ /app/
COPY --from=build /out-tools/ /app/
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
# Non-root. GPU access comes from the NVIDIA container runtime on the node.
# One-shot tools: no server port or HEALTHCHECK.
USER llama
ENTRYPOINT ["/app/llama-bench"]

FROM docker.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@${CUDA_RUNTIME_DIGEST} AS server
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG CUDA_ARCH
LABEL org.opencontainers.image.title="llmkube-llama-cuda-gb10" \
      org.opencontainers.image.description="LLMKube NVIDIA GB10 (DGX Spark) llama.cpp server runtime, native sm_121 (no JIT)" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.cuda.arch="${CUDA_ARCH}"
# The CUDA runtime base already carries libcudart and the math libraries
# (cuBLAS/cuBLASLt) that libggml-cuda.so links. curl is only for the container
# HEALTHCHECK. No driver: the node supplies libcuda.so.1 via the NVIDIA
# container runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r llama && useradd -r -g llama llama
COPY --from=build /out/ /app/
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
EXPOSE 8080
USER llama
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["/app/llama-server"]
