# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) llama.cpp server runtime, TurboQuant variant.
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

# TheTom's fork, branch laguna/port, pinned by immutable commit SHA. Same source
# as vulkan-laguna/Dockerfile, built here for CUDA instead of Vulkan.
#
# Why a fork rather than upstream: TurboQuant (TQ3_1S at 4.0 bpw, TQ4_1S at 5.0
# bpw) exists only here, and unlike the Vulkan side its CUDA kernels are real
# rather than orphans. That distinction is the entire reason this variant is
# worth having, and the TQ guard below is what holds the build to it.
#
# The branch is 165 commits behind the fork's own master and carries GB10 work
# of its own ("cuda: first-class MoE path + plain f16 FA hygiene on GB10",
# "dflash: fix DSpark tensor-meta accessor"), so it is not a stale side branch.
#
# NOTE this pin deliberately does NOT track cuda-gb10/Dockerfile's upstream pin.
# The two images exist to be compared: same GPU, same CUDA, different llama.cpp.
# Moving them in lockstep would defeat that, so they are bumped independently.
ARG LLAMACPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG LLAMACPP_REF=laguna/port
ARG LLAMACPP_SHA=bbf69962f74fed397d6583c9b1f686b3166dcbfa

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
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG CUDA_ARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# A branch head moves, so --depth 1 on the branch is not reproducible on its own.
# Fetch the pinned commit directly and verify it, which pins the build even after
# laguna/port advances.
RUN git init -q . \
    && git remote add origin "${LLAMACPP_REPO}" \
    && git fetch -q --depth 1 origin "${LLAMACPP_SHA}" \
    && git checkout -q FETCH_HEAD \
    && got="$(git rev-parse HEAD)" \
    && if [ "${got}" != "${LLAMACPP_SHA}" ]; then \
         echo "ERROR: checked out ${got}, expected ${LLAMACPP_SHA}"; exit 1; \
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
# GGML_RPC=ON builds the RPC backend and the rpc-server binary, which is how one
# model spans two Sparks: ggml-rpc-server exposes a remote host's devices and the
# main process offloads layers onto them, so the model is limited by the SUM of
# both machines' memory rather than either one's. NVIDIA builds their Spark
# llama.cpp the same way (nvidia/dgx-spark-playbooks, nvidia/llama-cpp).
#
# It is off in upstream's default build, and its absence is silent: a runtime
# without it simply rejects --rpc, which reads as a bad flag rather than a
# missing feature.
#
# Two things to be clear about, since neither is obvious:
#   - Enabling it changes nothing for single-node serving. It adds a backend and
#     a binary; llama-server behaves identically unless --rpc is passed.
#   - Upstream calls the RPC backend a proof of concept and warns it is insecure
#     ("never run the RPC server on an open network"). That is a statement about
#     exposure, not correctness: on a dedicated point-to-point cable between two
#     machines under the same owner it is a much smaller concern than on a shared
#     network. Do not publish rpc-server on a routable interface.
RUN cmake -B build \
      -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_RPC=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /out /out-tools \
    && find build -name '*.so*' -exec cp -P {} /out/ \; \
    && cp build/bin/llama-server /out/ \
    && cp build/bin/ggml-rpc-server /out/ \
    && cp build/bin/llama-bench build/bin/llama-cli /out-tools/

# RPC GUARD. Same discipline as the codegen guard: the feature this build turned
# on is asserted against the artifact, not assumed from the flag. A cmake option
# that silently stops taking effect (renamed, moved behind another condition)
# would otherwise ship an image whose only symptom is that --rpc is rejected at
# runtime, on a node, during a test you thought was measuring something else.
# The binary is `ggml-rpc-server` (tools/rpc/CMakeLists.txt sets that TARGET),
# not `rpc-server`; the subdirectory is added only `if (GGML_RPC)`, so its
# absence is exactly the signal we want to catch.
RUN test -x /out/ggml-rpc-server \
      || { echo "ERROR: ggml-rpc-server missing; GGML_RPC=ON did not take effect."; \
           echo "       tools/CMakeLists.txt adds tools/rpc only under if(GGML_RPC),"; \
           echo "       so this means the option stopped applying."; exit 1; }; \
    ls /out/libggml-rpc.so >/dev/null 2>&1 \
      && echo "NOTE: libggml-rpc.so present (dlopen-able RPC backend)" \
      || echo "NOTE: no separate libggml-rpc.so (RPC linked into the core build)"; \
    echo "PASS: ggml-rpc-server present"

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

# TURBOQUANT GUARD. The reason this variant exists, asserted rather than assumed.
#
# This is the lesson from the Vulkan side of the same fork, where TQ4_1S shader
# sources were present in the tree, looked complete on inspection, and were never
# compiled or registered. Reading the source proves nothing; a file can be in the
# repo and absent from the binary. So interrogate the built artifact.
#
# Two separate things have to be true, and each fails differently:
#
#   1. TQ device kernels are in the fatbin. --dump-resource-usage enumerates the
#      kernels actually compiled into the binary, which is the check that would
#      have caught the Vulkan orphans. Host-side strings would not: the launcher
#      symbols exist even when no device code was generated.
#   2. The types are registered in ggml itself, so a TQ GGUF is recognised at
#      load rather than rejected as an unknown type.
#
# Failing here means the fork built but its TurboQuant support did not, i.e. the
# image is an expensive way to run stock llama.cpp.
RUN cuobjdump --dump-resource-usage /out/libggml-cuda.so > /tmp/kernels.txt 2>/tmp/kernels.err || true; \
    # Distinguish "the dump did not work" from "the dump worked and TQ is absent".
    # Conflating them would fail the build with a confidently wrong reason, which
    # is the failure mode these guards exist to avoid. A working dump always
    # names some kernels, so an empty result means the tool, not the code.
    if ! grep -qE '[Ff]unction|[Rr]eg[Cc]ount|\.text\.' /tmp/kernels.txt; then \
      echo "ERROR: cuobjdump --dump-resource-usage produced no kernel listing."; \
      echo "       This guard cannot tell whether TurboQuant is present, so it is"; \
      echo "       failing rather than passing blind. stderr was:"; \
      sed 's/^/         /' /tmp/kernels.err; \
      echo "       Check the flag against this CUDA toolkit's cuobjdump."; \
      exit 1; \
    fi; \
    echo "== kernels compiled: $(grep -cE '[Ff]unction' /tmp/kernels.txt) =="; \
    echo "== TurboQuant device kernels =="; \
    grep -oiE '[a-z0-9_]*tq[0-9]_[0-9]s[a-z0-9_]*' /tmp/kernels.txt | sort -u | head -20; \
    if ! grep -qi 'tq[0-9]_[0-9]s' /tmp/kernels.txt; then \
      echo "ERROR: the kernel listing is populated but contains no TurboQuant kernels."; \
      echo "       The fork built and its CUDA backend compiled, but the TQ path did"; \
      echo "       not, which makes this image stock llama.cpp with extra steps."; \
      echo "       Check that LLAMACPP_SHA is a laguna/port commit and that"; \
      echo "       ggml/src/ggml-cuda/*tq*.cu and turbo-*.cu were compiled."; \
      exit 1; \
    fi; \
    echo "PASS: TurboQuant CUDA kernels present in device code"

# Type registration, checked against the shipped libraries rather than the tree.
# TQ3_1S is enum 45 and TQ4_1S is 46 in this fork; both must be known to ggml or
# a TQ GGUF fails to load no matter how good the kernels are.
RUN found=0; \
    for lib in /out/libggml-base.so /out/libggml.so /out/libllama.so; do \
      [ -f "$lib" ] || continue; \
      if strings -a "$lib" | grep -qiE 'tq[34]_1s'; then found=1; fi; \
    done; \
    if [ "$found" != "1" ]; then \
      echo "ERROR: TQ3_1S/TQ4_1S type names absent from the ggml libraries."; \
      echo "       The CUDA kernels exist but the types are not registered, so a"; \
      echo "       TurboQuant GGUF would be rejected at load as an unknown type."; \
      exit 1; \
    fi; \
    echo "PASS: TurboQuant types registered in ggml"

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
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG CUDA_ARCH
LABEL org.opencontainers.image.title="llmkube-llama-cuda-gb10-turbo-tools" \
      org.opencontainers.image.description="LLMKube NVIDIA GB10 (DGX Spark) TurboQuant llama.cpp tooling (llama-bench, llama-cli, llama-server), native sm_121" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.llamacpp.repo="${LLAMACPP_REPO}" \
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
ARG LLAMACPP_REPO
ARG LLAMACPP_REF
ARG LLAMACPP_SHA
ARG CUDA_ARCH
LABEL org.opencontainers.image.title="llmkube-llama-cuda-gb10-turbo" \
      org.opencontainers.image.description="LLMKube NVIDIA GB10 (DGX Spark) TurboQuant llama.cpp server runtime, native sm_121 (no JIT)" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.llamacpp.ref="${LLAMACPP_REF}" \
      io.llmkube.llamacpp.sha="${LLAMACPP_SHA}" \
      io.llmkube.llamacpp.repo="${LLAMACPP_REPO}" \
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
