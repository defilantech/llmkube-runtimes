# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) vLLM runtime for DeepSeek-V4.1-Flash EXL3 3.5bpw, tensor-parallel 3
# across three Sparks cabled in a ring (TP4 on a switch uses the same image).
#
# Built from the OFFICIAL vLLM base pinned by digest plus overlays pinned by SHA. Every vendored file is
# MIT, BSD-3-Clause (cutlass) or Apache-2.0; nothing AGPL. Read SPEC-runtimes-image.md (llmkube-internal) for the why.
#
#   base       vllm/vllm-openai deepseekv41-flash-0909-arm64      Apache-2.0
#   FlashInfer flashinfer-ai/flashinfer 07869c61 (v0.7.0rc1)      Apache-2.0   replaces 0.6.18 in place
#   cuda-exl3  Zeuss5/cuda-exl3 6a1ffc34                          MIT          sm_121a
#   patches    tonyd2wild DeepSeek-V4.1-Flash-vLLM-DGX-Spark fc725ecf patch/exl3-tp3   MIT (Kai, bot-lab-21 credited)
#   model.py   vllm-project/vllm e47aa780b nvidia/model.py        Apache-2.0   restores the tree the patches target
#
# arm64 only: GB10 is aarch64 Grace and the base publishes no amd64 manifest.

ARG BASE=vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
FROM ${BASE}

ARG MAX_JOBS=3
ENV D=/usr/local/lib/python3.12/dist-packages

COPY patches /opt/llmkube/patches
COPY build /opt/llmkube/build
COPY tests /opt/llmkube/tests

# ---- 5.1 drift gate: nothing is overwritten unless the base is the one the patch set targets ----
RUN python3 /opt/llmkube/build/drift_gate.py "$D/vllm" /opt/llmkube/patches/MD5SUMS.image-baseline-e47aa780b.txt

# ---- 5.2 FlashInfer 0.7.0rc1 from source, in place ----
RUN pip uninstall -y -q flashinfer-jit-cache flashinfer-cubin flashinfer-python || true
ENV VLLM_HAS_FLASHINFER_CUBIN=1 FLASHINFER_DISABLE_VERSION_CHECK=1
RUN <<'EOT'
set -eu
FI=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1; CUT=b46b16d003484063bca4ed365e44095c4c6ed633
CCCL=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116; SPD=c3aed4b68373955e1cc94307683d44dca1515d2b
mkdir -p /opt/fi-src/3rdparty/cutlass /opt/fi-src/3rdparty/cccl /opt/fi-src/3rdparty/spdlog
curl -fsSL -m 900 "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/$FI" | tar xz -C /opt/fi-src --strip-components=1
curl -fsSL -m 900 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/$CUT" | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1
curl -fsSL -m 900 "https://codeload.github.com/NVIDIA/cccl/tar.gz/$CCCL" | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1
curl -fsSL -m 900 "https://codeload.github.com/gabime/spdlog/tar.gz/$SPD" | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1
ls /opt/fi-src/3rdparty/cutlass/include/cutlass/cutlass.h /opt/fi-src/3rdparty/cccl/README.md /opt/fi-src/3rdparty/spdlog/include/spdlog/spdlog.h >/dev/null
cd /opt/fi-src
BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 MAX_JOBS="${MAX_JOBS}" pip install --no-deps --no-build-isolation -q .
cd / && rm -rf /opt/fi-src
python3 - <<'PY'
import flashinfer
from flashinfer.mla import supported_sparse_mla_sm120_configs as f
assert flashinfer.__version__ == "0.7.0rc1", flashinfer.__version__
c = f()["dsv4"]
for h in (16, 24, 32):
    for k in (1152, 640):
        assert c.supports_decode(num_heads=h, topk=k), (h, k)
print("flashinfer 0.7.0rc1 with TP2/TP3/TP4 sparse-MLA decode shapes")
PY
EOT

# ---- 5.3 cuda-exl3 for sm_121a, then bot-lab-21's V4.1 overlay on top ----
RUN <<'EOT'
set -eu
SHA=6a1ffc34866e23f484574ce1922a8bca93eb33b2
mkdir -p /opt/exl3-src
curl -fsSL -m 600 "https://codeload.github.com/Zeuss5/cuda-exl3/tar.gz/$SHA" | tar xz -C /opt/exl3-src --strip-components=1
cd /opt/exl3-src
TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="${MAX_JOBS}" pip install --no-deps --no-build-isolation -q .
cd / && rm -rf /opt/exl3-src
EOT

# ---- 5.4 vLLM patches (and the two cuda_exl3 overlays) copied to their destinations, then verified ----
RUN python3 /opt/llmkube/build/apply_patches.py /opt/llmkube/patches "$D/vllm" "$D/cuda_exl3" \
      --extra=nvidia_model.py=models/deepseek_v4_1/nvidia/model.py:4df8a4e2
RUN python3 - <<'PY'
import importlib.metadata as md
import torch  # noqa: F401  (loads libc10/libtorch; cuda_exl3._C links against them)
import cuda_exl3, cuda_exl3._C  # noqa: F401
eps = {(e.name, e.value) for e in md.entry_points(group="vllm.general_plugins")}
assert ("cuda_exl3", "cuda_exl3:register") in eps, eps
print("cuda_exl3 kernel loads; vllm.general_plugins registers cuda_exl3:register")
PY

# ---- 5.5 prewarm the JIT modules under the runtime env ----
# MAX_JOBS=1 here and in the shipped env: the mxfp8 cutlass GEMM compiles six variants and each nvcc front end
# takes ~6 GB of host RAM, so four in parallel exceed the 16 GB arm runner (and, on a serving node, sit on top of
# 79.5 GiB of pinned weights). Serial is slower and always fits.
ENV FLASHINFER_WORKSPACE_BASE=/opt/llmkube/flashinfer-jit \
    TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a FLASHINFER_NVCC_THREADS=1 MAX_JOBS=1
RUN mkdir -p /opt/llmkube/flashinfer-jit && python3 /opt/llmkube/build/prewarm.py && chmod -R a+rwX /opt/llmkube/flashinfer-jit

# ---- 6 in-image tests ----
RUN pip install -q pytest && python3 -m pytest -q /opt/llmkube/tests

# ---- 5.6 receipts, attribution, entrypoint ----
COPY NOTICE LICENSE.tonyd2wild-MIT LICENSE.cuda-exl3-MIT /opt/llmkube/
RUN bash /opt/llmkube/build/receipt.sh
COPY --chmod=0755 entrypoint.sh /usr/local/bin/dsv41-entrypoint.sh

LABEL org.opencontainers.image.title="llmkube-vllm-cuda-gb10-dsv41-exl3" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="vLLM for DeepSeek-V4.1-Flash EXL3 3.5bpw at TP3 across a three-DGX-Spark ring (GB10, sm_121a)"

ENTRYPOINT ["/usr/local/bin/dsv41-entrypoint.sh"]
