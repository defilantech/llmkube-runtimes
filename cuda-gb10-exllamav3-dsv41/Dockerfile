# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) native ExLlamaV3 runtime for
# DeepSeek-V4.1-Flash EXL3 (SAGE 1.59bpw), tensor-parallel 1, one Spark.
#
# WHY THIS IMAGE EXISTS
#
# The other EXL3-on-Spark images here run vLLM, and between the stage and the
# serve they DROP page cache, because a vLLM worker refuses to start when
# CUDA-free memory at init is below gpuMemoryUtilization x total. Native
# ExLlamaV3 wants the opposite: on a GB10 it runs the GPU in ATS addressing mode
# and aliases the safetensors straight out of an mmap instead of copying them
# into CUDA allocations, which is the only reason ~107 GiB of weights fits a
# 128 GB box whose memory is shared between CPU and GPU. Evicting page cache
# evicts the pages this loader is trying to alias. The two paths disagree about
# the most important resource on the machine.
#
# WHY IT IS BUILT FROM SOURCE, AND WHY THE HTTP SERVER IS OURS
#
# The community recipe (vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe) runs
# TabbyAPI, which is AGPL-3.0, and its own LICENSE is AGPL-3.0-only. This repo is
# Apache-2.0 and its policy is that every layer is pinned and none is AGPL.
# ExLlamaV3 upstream (turboderp-org) and the fork (vcruz305, branch
# feat/gb10-ats-load) are MIT, so the engine is usable and only the server was
# the blocker. Hence: build the MIT engine, ship our own server. Full licence
# table, with SPDX IDs and pinned commits: docs/exllamav3-gb10-runtime.md.
#
# AARCH64
#
# The pinned branch is x86-oriented: it includes <immintrin.h>, applies
# target("avx2") attributes and calls __builtin_cpu_supports / _mm_pause. None of
# those exist on aarch64, so patches/0001-aarch64-build.patch (OUR OWN, written
# from the actual compile failures, not derived from any AGPL helper) makes the
# extension build. It is applied with git apply --check first, so a base that
# moves fails the build rather than shipping a silently wrong tree.
#
# GATES
#
# Two tiers, same as the Vulkan, ROCm and other CUDA images. Tier-1 is CI (no GPU
# needed): the guards below plus scripts/exllamav3-gb10-gate.sh on the shipped
# image. Tier-2 is an out-of-band GPU smoke on a real Spark; the existing
# promoter is Vulkan/gfx1151-shaped and does not cover GB10.

# CUDA 13.0.1 pinned by index digest. 13.0 is required, not merely convenient:
# GB10 is sm_121 and torch's cu130 wheels are what the extension links against.
# Same digests as cuda-gb10/ so the two GB10 images share a base.
ARG CUDA_DEVEL_DIGEST=sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d
ARG CUDA_RUNTIME_DIGEST=sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

FROM docker.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@${CUDA_DEVEL_DIGEST} AS build

# The fork branch is active, so the pin is the commit, not the ref. The ref is
# only how the clone is asked; the SHA check below is what makes it immutable.
ARG EXLLAMAV3_REF=feat/gb10-ats-load
ARG EXLLAMAV3_SHA=954a8ca6e59d48c3e3462068ecf083fe9990f4dc

# GB10 is sm_121. 12.1a is architecture-specific device code, no PTX, so this
# image cannot silently JIT itself onto other hardware; that is asserted below.
ARG CUDA_ARCH=12.1a
ARG TORCH_VERSION=2.14.0
# The arm64 runner has 4 vCPU and 16 GB; nvcc peaks at multiple GB per process.
# CI caps this at nproc; the default is the conservative fallback for a
# workstation Docker daemon.
ARG MAX_JOBS=3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential python3 python3-dev python3-venv python3-pip \
      cmake ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip wheel setuptools ninja

# torch from PyTorch's cu130 index. Verified on linux/arm64 during the aarch64
# build spike: 2.14.0+cu130 resolves and installs for aarch64.
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu130

# ExLlamaV3's own runtime requirements. Installed explicitly because the wheel
# below is built with --no-deps: the build must not resolve the dependency graph
# under a CUDA stack that is already correct for sm_121.
RUN pip install --no-cache-dir \
      "tokenizers>=0.21.1" "numpy>=2.1.0" "safetensors>=0.3.2" \
      rich typing_extensions pyyaml marisa_trie pydantic "llguidance>=1.7.0" pillow

RUN git clone --branch "${EXLLAMAV3_REF}" https://github.com/vcruz305/exllamav3.git /src \
    && got="$(git -C /src rev-parse HEAD)" \
    && if [ "${got}" != "${EXLLAMAV3_SHA}" ]; then \
         echo "ERROR: ${EXLLAMAV3_REF} resolved to ${got}, expected ${EXLLAMAV3_SHA}"; \
         exit 1; \
       fi \
    && echo "PASS: exllamav3 pinned to ${got}"

COPY patches /opt/llmkube/patches
# git apply --check is the drift gate. If the pinned source moves, the context
# of the patch no longer lines up and this fails the build with the file named,
# rather than applying something subtly different.
RUN cd /src \
    && git apply --check /opt/llmkube/patches/0001-aarch64-build.patch \
    && git apply /opt/llmkube/patches/0001-aarch64-build.patch \
    && echo "PASS: aarch64 build patch applied"

# Assert the patch took EFFECT, not merely that it was attempted. A patch that
# applied to the wrong tree, or a transform that silently stopped matching,
# would otherwise ship an x86-only tree compiled for aarch64. Each marker is a
# string only the patch produces, so none of them passes on the unpatched tree.
# Split on the FIRST '|' only: the header marker itself contains '||'.
RUN cd /src && rc=0; \
    for pair in \
      "exllamav3/exllamav3_ext/avx2_target.cpp|no cpuid here" \
      "exllamav3/exllamav3_ext/avx2_target.h|defined(__x86_64__) || defined(__i386__)" \
      "exllamav3/exllamav3_ext/avx512_target.h|defined(__x86_64__) || defined(__i386__)" \
      "exllamav3/exllamav3_ext/cpu/moe_mul1.cpp|cpu_moe_unavailable_" \
      "exllamav3/exllamav3_ext/cpu/moe_handoff.cu|__asm__ __volatile__(\"yield\")" \
      "exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu|__asm__ __volatile__(\"yield\")" \
      "exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx2.cpp|x86-only AVX2 kernel" \
      "exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx512.cpp|AVX-512 bf16 add is unavailable on aarch64" \
    ; do \
      f="${pair%%|*}"; m="${pair#*|}"; \
      if grep -qF -- "${m}" "${f}"; then \
        echo "PASS: ${f} carries its patch marker"; \
      else \
        echo "FAIL: ${f} is missing patch marker '${m}'; the patch did not take effect"; \
        rc=1; \
      fi; \
    done; \
    exit ${rc}

RUN cd /src \
    && CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" MAX_JOBS="${MAX_JOBS}" \
       pip install --no-build-isolation --no-deps . \
    && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# CODEGEN GUARD. The whole point of this image, asserted against the artifact
# rather than assumed from the arch flag. A wrong toolkit, a dropped
# TORCH_CUDA_ARCH_LIST, or an upstream change to the default list all produce a
# build that succeeds, an image that runs, and a GB10 quietly back on JIT.
# cuobjdump exists only in the devel stage, so the inventory is written to
# /app/CUDA_ARCHS as the receipt the runtime gate reads.
#
# cuobjdump output is captured to files and the assertions read the files, not a
# pipeline: `cuobjdump ... | grep -q` lets grep exit at the first match, sends
# SIGPIPE upstream, and under pipefail turns a PASSING build into a failing one.
# Docker RUN is plain `sh` with no pipefail, so the pipeline form happens to
# work, but a guard whose correctness depends on which shell it inherits is one
# SHELL directive from inverting itself. `--list-ptx` also exits non-zero when
# there is no PTX, which is the PASSING case, hence the `|| true`.
RUN mkdir -p /app \
    && SO="$(find /opt/venv -name 'exllamav3_ext*.so' | head -1)" \
    && test -n "${SO}" || { echo "ERROR: exllamav3_ext .so not found after build"; exit 1; }; \
    echo "ext: ${SO}"; \
    cuobjdump --list-elf "${SO}" > /tmp/elf.txt 2>/dev/null || true; \
    cuobjdump --list-ptx "${SO}" > /tmp/ptx.txt 2>/dev/null || true; \
    echo "== device code inventory =="; \
    grep -oE 'sm_[0-9]+[a-z]*' /tmp/elf.txt | sort | uniq -c; \
    echo "== PTX inventory (expected: empty) =="; \
    grep -oE 'sm_[0-9]+[a-z]*' /tmp/ptx.txt | sort | uniq -c; \
    if ! grep -q 'sm_121' /tmp/elf.txt; then \
      echo "ERROR: no native sm_121 device code in exllamav3_ext."; \
      echo "       GB10 would JIT every kernel, which this image exists to prevent."; \
      exit 1; \
    fi; \
    if grep -q 'sm_[0-9]' /tmp/ptx.txt; then \
      echo "ERROR: PTX present. A virtual target reintroduces the silent-JIT path."; \
      exit 1; \
    fi; \
    grep -oE 'sm_[0-9]+[a-z]*' /tmp/elf.txt | sort -u > /app/CUDA_ARCHS; \
    echo "PASS: native sm_121 device code present, no PTX fallback"

# The stub must resolve symbols AND report the x86 tiers unavailable. Proving
# the extension dlopens is the load-bearing part: cpu/moe_handoff.cu and
# parallel/all_reduce_cpu.cu resolve exl3_moe_cpu_* unconditionally, so a patch
# that dropped those symbols instead of stubbing them would fail here rather
# than on a node. torch must be imported first: it supplies libc10.
RUN python3 -c "import torch, exllamav3, exllamav3_ext as e; \
from exllamav3.version import __version__; \
print('exllamav3', __version__); \
assert e.exl3_moe_cpu_has_avx2() is False, 'x86 tier reported available on aarch64'; \
print('PASS: exllamav3_ext dlopens, stubs report x86 tiers unavailable')"

# Server-only dependencies, AFTER the extension build on purpose: editing the
# server must not invalidate the multi-minute CUDA compile layer. The server is
# copied into the runtime stage, so a server change rebuilds in seconds.
RUN pip install --no-cache-dir fastapi uvicorn jinja2

COPY build /opt/llmkube/build
RUN bash /opt/llmkube/build/receipt.sh

FROM docker.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@${CUDA_RUNTIME_DIGEST} AS runtime

ARG EXLLAMAV3_REF=feat/gb10-ats-load
ARG EXLLAMAV3_SHA=954a8ca6e59d48c3e3462068ecf083fe9990f4dc
LABEL org.opencontainers.image.title="llmkube-exllamav3-cuda-gb10-dsv41-exl3" \
      org.opencontainers.image.description="LLMKube NVIDIA GB10 (DGX Spark) native ExLlamaV3 runtime for DeepSeek-V4.1-Flash EXL3, sm_121 aarch64" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.llmkube.exllamav3.ref="${EXLLAMAV3_REF}" \
      io.llmkube.exllamav3.sha="${EXLLAMAV3_SHA}"

# Triton compiles its CUDA driver utilities from C at first use and needs a host
# compiler and the CPython headers to do it (python3-dev supplies Python.h), so
# both stay in the runtime stage: exllamav3's attention path runs through Triton.
# libgomp1 for torch's OpenMP, curl for the healthcheck. The CUDA runtime base
# already carries libcudart/cuBLAS that the extension links. No driver: the node
# supplies libcuda.so.1 via the NVIDIA container runtime.
# /home/exl3 is the runtime user's HOME and some libraries (Triton's JIT cache)
# write under it, so it is created and owned here rather than left absent.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev libgomp1 curl ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r exl3 && useradd -r -g exl3 exl3 \
    && mkdir -p /home/exl3 && chown exl3:exl3 /home/exl3

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/llmkube /opt/llmkube
COPY --from=build /app/CUDA_ARCHS /app/CUDA_ARCHS
# The patch set travels with the image so the Tier-1 gate can assert on the
# shipped artifact that nothing AGPL was vendored, and so anyone holding the
# image can read what was applied.
COPY patches /opt/llmkube/patches
COPY LICENSE.exllamav3-MIT NOTICE /opt/llmkube/
# The HTTP server. Loads the checkpoint from EXL3_MODEL_DIR in a background
# thread while the port is already bound, serves the OpenAI subset an agent loop
# uses, and reports load failures through /health and /ready rather than by
# dying. WHY it is ours rather than TabbyAPI: the licence boundary above.
COPY server /app/server
COPY entrypoint.sh /usr/local/bin/llmkube-exllamav3-entrypoint.sh
RUN chmod +x /usr/local/bin/llmkube-exllamav3-entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH \
    EXL3_MODEL_DIR=/models \
    EXL3_PORT=5000
WORKDIR /app
EXPOSE 5000
USER exl3
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS "http://localhost:${EXL3_PORT}/health" || exit 1
ENTRYPOINT ["/usr/local/bin/llmkube-exllamav3-entrypoint.sh"]
