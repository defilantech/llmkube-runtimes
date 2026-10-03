# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
#
# LLMKube NVIDIA GB10 (DGX Spark) vLLM runtime for DeepSeek-V4.1-Flash on local-inference-lab's vLLM fork plus
# the b12x kernel library (CuTe-DSL, JIT), tensor-parallel 3 across three Sparks cabled in a ring.
#
# Unlike the EXL3 sibling (stock vLLM plus a small overlay), the fork is BUILT FROM SOURCE at a pinned commit and
# replaces the base's vLLM outright. The fork pads V4.1's 64 heads / 8 output groups to 72 / 9 for TP3 by itself
# (vllm/model_executor/models/config.py, DeepseekV41ForCausalLMConfig.update_model_config_for_parallelism). The source
# patches carried on the pins (patches/<name>/APPLIED.md) are not yet upstream: b12x's switchless-ring RoCE routing, and
# a matching pair (one for the fork, one for b12x) that loads exllamav3 mcg K3..K6 trellis checkpoints.
# What is built or vendored here is Apache-2.0, BSD-3-Clause or MIT, and nothing is AGPL (build/deps_gate.py scans
# every installed distribution). The image ALSO carries NVIDIA-proprietary components under NVIDIA's licenses: the
# CUDA runtime and toolkit pieces from the base, and nvidia-cutlass-dsl 4.7.1 (+ libs), which the fork and b12x
# require and the base already ships at 4.6.2. See NOTICE.
#
#   base        vllm/vllm-openai deepseekv41-flash-0909-arm64, by digest   Apache-2.0   torch 2.13.0+cu130, CUDA 13.0
#   vLLM fork   local-inference-lab/vllm ec49d578                          Apache-2.0   compiled here for sm_121a
#               + Defilan/vllm 9036f769 (DS4.1 config accepts mcg K3..K6 trellis manifests), carried
#   b12x        local-inference-lab/b12x 6380e581 (1.3.0)                  Apache-2.0   pure Python, kernels JIT at serve
#               (head of upstream's evidence/ds41-x4t-serving-20260929, the only ref with the trellis-dense reader)
#               + local-inference-lab/b12x#457 (switchless-ring RoCE routing), carried until it merges
#               + Defilan/b12x f0997577 (trellis-dense-checkpoint/1 accepts mcg K3..K6), carried
#   FlashInfer  0.6.18.post1 wheels by sha256 (the fork's pin)             Apache-2.0   replaces the base's 0.6.18
#
# Two ways to get the fork's vLLM wheel, one gate for both (build/pins_gate.py):
#   default     compile /src/vllm with the fork's own recipe (setup.py bdist_wheel after use_existing_torch.py)
#   fallback    VLLM_WHEEL_URL + VLLM_WHEEL_SHA256: a wheel this same stage produced on a Spark from the same commit,
#               hash-verified. Used only if the compile does not fit the hosted runner (see README).
# Either way the final stage's /src/vllm is a checkout of the pinned commit plus exactly the carried patches in
# patches/vllm (the compile applies the same patches first), and every tracked vllm/**/*.py must be byte-identical in
# the installed package, so a wheel from any other commit, or without the patches, fails the build.
#
# Stages: src (checkouts) -> vllm-build (the wheel) -> vllm-wheel (export only: `--target vllm-wheel` with a local
# output yields the wheel plus its sha256, which is how the fallback wheel is made) -> the image (last, the default).
#
# arm64 only: GB10 is aarch64 Grace and the base publishes no amd64 manifest.

ARG BASE=vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
ARG VLLM_FORK_COMMIT=ec49d5781a0ebf7960c7711ac3e0f42376b1a127
ARG B12X_COMMIT=6380e581f075df7f01a72d4b67f016225f260c86

# ---- 1 sources at their pinned commits (the base has no git; setuptools-scm and the pins gate need one) ----
FROM ${BASE} AS src
ARG VLLM_FORK_COMMIT
ARG B12X_COMMIT
RUN apt-get update -q && apt-get install -y -q --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN <<'EOT'
set -eu
fetch() {  # <dir> <repo> <commit>
  git clone -q --filter=blob:none --no-checkout "https://github.com/$2.git" "$1"
  git -C "$1" -c advice.detachedHead=false checkout -q "$3"
  test "$(git -C "$1" rev-parse HEAD)" = "$3" || { echo "$2: checkout is not $3" >&2; exit 1; }
}
fetch /src/vllm local-inference-lab/vllm "${VLLM_FORK_COMMIT}"
fetch /src/b12x local-inference-lab/b12x "${B12X_COMMIT}"
EOT

# ---- 2 the fork's vLLM wheel: compiled for sm_121a, or the hash-pinned fallback ----
# use_existing_torch.py is the fork's own tool for building against a preinstalled torch: it strips the torch pins
# from requirements/ and pyproject.toml, so the wheel builds against the base's 2.13.0+cu130 and its metadata never
# asks pip for another torch. The Rust frontend (vllm-rs, _rust_tool_parser) is optional in setup.py, off by default
# at runtime (VLLM_USE_RUST_FRONTEND=0) and not used by the DeepSeek V4.1 tool parser, so no Rust toolchain.
# Compile parallelism: the workflow derives MAX_JOBS from the runner's RAM; 2 is upstream's own default
# (docker/versions.json) and fits 16 GB. NVCC_THREADS parallelises across gencode targets and this builds exactly
# one (12.1a), so more threads only add memory.
# CUDA dev headers: the base is a runtime image (nvcc, cudart, curand-dev, cublas-dev; no unversioned libnvrtc.so; the fork's own
# image compiles in a full-toolkit builder). torch's ATen/cuda/CUDAContextLight.h includes <cusparse.h> and
# <cusolverDn.h>, which the fork's csrc pulls in via ATen/cuda/CUDAContext.h. The compile path installs the minimal
# -dev set, pinned to CUDA 13.0 Update 3 in NVIDIA's ubuntu2404/sbsa repo, in this stage only (not shipped).
FROM src AS vllm-build
ARG MAX_JOBS=2
ARG NVCC_THREADS=1
ARG VLLM_WHEEL_URL=""
ARG VLLM_WHEEL_SHA256=""
# The carried vLLM patches go into the compiled tree, so the wheel is the pin plus exactly those patches (they touch
# Python only; the compiled extensions are unchanged). The image stage applies them again to its own /src/vllm.
COPY patches/vllm /opt/llmkube/patches/vllm
COPY build/apply_carried_patches.sh /opt/llmkube/build/apply_carried_patches.sh
RUN bash /opt/llmkube/build/apply_carried_patches.sh /opt/llmkube/patches/vllm /src/vllm
RUN <<'EOT'
set -eu
mkdir -p /tmp/dist
if [ -n "${VLLM_WHEEL_URL}" ]; then
  [ -n "${VLLM_WHEEL_SHA256}" ] || { echo "VLLM_WHEEL_URL is set without VLLM_WHEEL_SHA256" >&2; exit 1; }
  curl -fsSL -m 1800 -o "/tmp/dist/${VLLM_WHEEL_URL##*/}" "${VLLM_WHEEL_URL}"
  echo "${VLLM_WHEEL_SHA256}  /tmp/dist/${VLLM_WHEEL_URL##*/}" | sha256sum -c -
  mode=wheel; origin="${VLLM_WHEEL_URL}"
else
  /usr/local/cuda/bin/nvcc --version | grep -q 'release 13\.0,' || { echo "base CUDA is not 13.0; re-pin the -dev set" >&2; exit 1; }
  # nvrtc-dev must match the base's nvrtc runtime exactly (the -dev package depends on that version), so its
  # pin is read from dpkg rather than hard-coded. CMake's CUDA::nvrtc needs the unversioned libnvrtc.so that only
  # the -dev package ships; without it configure fails with CUDA_nvrtc_LIBRARY NOTFOUND (first CI run, 2026-09-30).
  nvrtc_ver="$(dpkg-query -W -f='${Version}' cuda-nvrtc-13-0)"
  apt-get update -q
  apt-get install -y -q --no-install-recommends --allow-change-held-packages \
    libcusparse-dev-13-0=12.6.3.3-1 libcusolver-dev-13-0=12.0.4.66-1 libcufft-dev-13-0=12.0.0.61-1 \
    cuda-culibos-dev-13-0=13.0.85-1 cuda-nvtx-13-0=13.0.85-1 cuda-profiler-api-13-0=13.0.85-1 \
    "cuda-nvrtc-dev-13-0=${nvrtc_ver}"
  # DeepGEMM's DeepJIT includes <elfutils/libdwfl.h> (third-party/deep_jit/include/deep_jit/utils/exception.hpp) and
  # dlopen()s libdw.so.1 at runtime only for stack traces, so the build needs the headers and the runtime needs
  # nothing. Ubuntu drops superseded versions from noble-updates, so a version pin here would break the build on the
  # next security update; this stage is not shipped, and the installed version is printed for the build log.
  apt-get install -y -q --no-install-recommends libdw-dev
  dpkg-query -W -f='libdw-dev ${Version}\n' libdw-dev
  rm -rf /var/lib/apt/lists/*
  test -f /usr/include/elfutils/libdwfl.h || { echo "elfutils/libdwfl.h missing after libdw-dev" >&2; exit 1; }
  test -f /usr/local/cuda/include/cusparse.h && test -f /usr/local/cuda/include/cusolverDn.h
  ls /usr/local/cuda/lib64/libnvrtc.so /usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so 2>/dev/null | grep -q . \
    || { echo "libnvrtc.so (unversioned) still missing after cuda-nvrtc-dev-13-0=${nvrtc_ver}" >&2; exit 1; }
  cd /src/vllm
  python3 use_existing_torch.py --prefix
  pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > /tmp/keep.txt
  grep -q '^torch==' /tmp/keep.txt || { echo "constraints lack torch==; pip list printed it another way" >&2; exit 1; }
  pip install -q -c /tmp/keep.txt -r requirements/build/cuda.txt
  VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="${MAX_JOBS}" NVCC_THREADS="${NVCC_THREADS}" \
    python3 setup.py bdist_wheel --dist-dir=/tmp/dist --py-limited-api=cp38
  mode=source; origin=/src/vllm
fi
whl="$(ls /tmp/dist/vllm-*.whl)"
sha="$(sha256sum "$whl" | cut -d' ' -f1)"
echo "$mode $sha $origin" > /tmp/dist/vllm-install.txt
(cd /tmp/dist && sha256sum "${whl##*/}" > "${whl##*/}.sha256")
cat /tmp/dist/vllm-install.txt
EOT

FROM scratch AS vllm-wheel
COPY --from=vllm-build /tmp/dist/ /

# ---- 3 the image ----
FROM src
ARG VLLM_FORK_COMMIT
ARG B12X_COMMIT
ARG VLLM_WHEEL_URL=""
ARG VLLM_WHEEL_SHA256=""
ENV D=/usr/local/lib/python3.12/dist-packages

# b12x's checkpoint loader compiles its C extension at first use (b12x/loader/_native.py) and turns on io_uring only
# when pkg-config finds liburing; --load-format b12x then needs that io_uring bounce path ("io_uring bounce support is
# unavailable" on the first TP3 smoke run, 2026-10-01). Unpinned for the same reason as libdw-dev above; the installed
# versions are printed for the build log, and the in-image tests compile the loader and check liburing is found.
RUN <<EOT
set -eu
apt-get update -q
apt-get install -y -q --no-install-recommends liburing-dev pkg-config
dpkg-query -W -f='${Package} ${Version}\n' liburing-dev pkg-config
rm -rf /var/lib/apt/lists/*
pkg-config --exists liburing || { echo "pkg-config cannot find liburing" >&2; exit 1; }
EOT

COPY patches /opt/llmkube/patches
COPY build /opt/llmkube/build
COPY tests /opt/llmkube/tests
# Attribution first, so the in-image tests can check it: the four top-level texts plus licenses/ (every component
# the fork's CMake compiles into vllm/_C, at the revision it pins).
COPY NOTICE LICENSE.vllm-Apache-2.0 LICENSE.b12x-Apache-2.0 LICENSE.flashinfer-Apache-2.0 LICENSE.cutlass-BSD-3-Clause /opt/llmkube/
COPY licenses /opt/llmkube/licenses

# ---- 3.1 FlashInfer 0.6.18.post1, the fork's pin (requirements/cuda.txt, docker/Dockerfile FLASHINFER_VERSION) ----
# All three are wheels already built for aarch64/cu130 (python and cubin are py3-none-any), so nothing compiles here.
# Installed --no-deps from patches/PINNED_DISTS.txt after a sha256 check; the base's 0.6.18 set goes first so no
# 0.6.18 cubin or AOT package is left to trip FlashInfer's version match.
RUN <<'EOT'
set -eu
pip uninstall -y -q flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
mkdir -p /tmp/fi
awk '!/^#/ && NF == 4 { print $3, $4 }' /opt/llmkube/patches/PINNED_DISTS.txt | while read -r sha url; do
  f="/tmp/fi/${url##*/}"
  curl -fsSL -m 1800 -o "$f" "$url"
  echo "$sha  $f" | sha256sum -c -
done
pip install -q --no-deps /tmp/fi/*.whl
rm -rf /tmp/fi
python3 -c 'import flashinfer; assert flashinfer.__version__ == "0.6.18.post1", flashinfer.__version__'
EOT

# ---- 3.2 the fork's vLLM replaces the base's; its requirements resolve around a held torch/triton/FlashInfer ----
# The constraint file keeps torch, triton and FlashInfer exactly where they are while pip resolves the rest of the
# fork's requirements (tilelang, quack-kernels, humming-kernels, tokenspeed-mla, cutlass-dsl 4.7.1...). The wheel is
# re-hashed here and must match what the build stage recorded.
RUN --mount=type=bind,from=vllm-build,source=/tmp/dist,target=/tmp/dist <<'EOT'
set -eu
pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|triton|flashinfer[-_][a-z_-]+)==' > /opt/llmkube/constraints.txt
cat /opt/llmkube/constraints.txt
if ! grep -qi '^torch==' /opt/llmkube/constraints.txt || ! grep -qiE '^flashinfer[-_]python==' /opt/llmkube/constraints.txt; then
  echo "constraints file is missing torch== or flashinfer-python==" >&2; exit 1
fi
whl="$(ls /tmp/dist/vllm-*.whl)"
read -r _mode sha _origin < /tmp/dist/vllm-install.txt
test "$(sha256sum "$whl" | cut -d' ' -f1)" = "$sha" || { echo "vllm wheel does not match the build stage's record" >&2; exit 1; }
pip uninstall -y -q vllm
pip install -q -c /opt/llmkube/constraints.txt "$whl"
cp /tmp/dist/vllm-install.txt /opt/llmkube/vllm-install.txt
rm -rf /root/.cache/pip
EOT

# ---- 3.3a patches carried on the pins (patches/<name>/APPLIED.md): the fork's onto this stage's /src/vllm (the wheel
# above was compiled from the same patched tree), b12x's before b12x is installed ----
# build/apply_carried_patches.sh records a patch's new files with `git add -N` (intent to add), so the clean after
# install keeps them and the pins gate sees them as part of the checkout; the gate proves each checkout is the pin plus
# exactly these patches. Not `git apply --intent-to-add`: git 2.43 (this base) drops every other index entry when it
# is used without --index.
RUN bash /opt/llmkube/build/apply_carried_patches.sh /opt/llmkube/patches/vllm /src/vllm \
 && bash /opt/llmkube/build/apply_carried_patches.sh /opt/llmkube/patches/b12x /src/b12x

# ---- 3.3 b12x at its pinned commit plus the carried patches ----
# --no-deps: the 6380e581 pin still declares nvidia-cutlass-dsl (+ libs) ==4.6.2, while the fork requires 4.7.1 and
# upstream b12x moved to 4.7.1 with no source change (4bacd509). The fork's install above already brought 4.7.1 and
# b12x's other requirements (apache-tvm-ffi, transformers, safetensors...); the pins gate holds cutlass-dsl at 4.7.1,
# and the deps gate fails on any b12x requirement that is missing, accepting only the five cutlass-dsl ==4.6.2 lines
# (patches/PIP_CHECK_ALLOW.txt).
RUN pip install -q --no-deps /src/b12x && git -C /src/b12x clean -fdxq && rm -rf /root/.cache/pip

# ---- 3.3b NCCL 2.30.7 back in place of torch's 2.29.7 pin ----
# The base ships NCCL 2.30.7 even though torch 2.13.0+cu130 declares ==2.29.7; installing the fork's wheel lets pip
# "repair" that and downgrade to 2.29.7. The three-Spark ring needs 2.30.7: its NCCL_IB_SUBNET_AWARE_ROUTING picks
# the right NIC per peer on the odd-cycle P0->P1 cabling, and without it every channel's QP times out at INIT->RTR
# (observed on the first TP3 smoke run, 2026-10-01). Hash-checked, --no-deps, same wheel the base ships.
ARG NCCL_WHEEL_SHA256=ca786ffa5a647c75d4d1f5cc72a6c4f537947e2ba8823d7c8aaf768e7a7b9f77
ARG NCCL_WHEEL_URL=https://files.pythonhosted.org/packages/d1/21/a73174c6157101bdf1ffc22b517f76ff0082613989dd9bc8f43e8034caac/nvidia_nccl_cu13-2.30.7-py3-none-manylinux_2_18_aarch64.whl
RUN <<EOT
set -eu
mkdir -p /tmp/nccl
curl -fsSL -m 1800 -o "/tmp/nccl/$(basename "$NCCL_WHEEL_URL")" "$NCCL_WHEEL_URL"
echo "$NCCL_WHEEL_SHA256  /tmp/nccl/$(basename "$NCCL_WHEEL_URL")" | sha256sum -c -
pip install -q --no-deps --force-reinstall /tmp/nccl/*.whl
lib="$(python3 -c 'import nvidia.nccl, os; print(os.path.join(nvidia.nccl.__path__[0], "lib", "libnccl.so.2"))')"
grep -aq NCCL_IB_SUBNET_AWARE_ROUTING "$lib" || { echo "NCCL at $lib has no subnet-aware routing" >&2; exit 1; }
rm -rf /tmp/nccl /root/.cache/pip
EOT

# ---- 3.4 pins gate (checkouts, installed trees, dist versions, install record; build ARGs are in the env here) and
# deps gate (pip check conflicts touching a pinned package, AGPL in any installed distribution's license metadata) ----
RUN python3 /opt/llmkube/build/pins_gate.py --site "$D" && python3 /opt/llmkube/build/deps_gate.py

# ---- 3.5 prewarm: import the V4.1 serve path, provision the b12x compile cache ----
# b12x compiles its CuTe-DSL kernels per shape on the GPU at first serve; they land in B12X_COMPILE_CACHE_DIR, which
# an InferenceService can back with a volume so a restart does not recompile. The directory is world-writable because
# the serving user is not root on every cluster. CUTE_DSL_ARCH and TRITON_PTXAS_PATH are what upstream's Spark launcher
# sets (scripts/serve-ds4-flash-dspark-tp4-rdma.sh:335,337) and are facts about this image, so they are defaults here;
# CUTE_DSL_ARCH is part of b12x's compile-cache key, so it is set before prewarm records that key. The rest of the
# launcher's serving env is per-deployment and belongs to the InferenceService (README lists it).
ENV B12X_COMPILE_CACHE_DIR=/opt/llmkube/b12x-cache \
    CUTE_DSL_ARCH=sm_121a \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# No GPU exists during docker build, so vLLM sees 0 active Triton drivers and disables Triton, and the V4.1 kernels
# that build Triton constants at import time then fail (`tl.constexpr` is None). An EMPTY CUDA_VISIBLE_DEVICES is
# vLLM's own lenient case for driverless initialisation (vllm/triton_utils/importing.py, `is_distributed_env`):
# Triton stays enabled with 0 drivers. Set on these RUN lines only, never as ENV, so the image serves normally.
RUN mkdir -p "$B12X_COMPILE_CACHE_DIR" && chmod 1777 "$B12X_COMPILE_CACHE_DIR" && CUDA_VISIBLE_DEVICES= python3 /opt/llmkube/build/prewarm.py

# ---- 3.6 in-image tests (ours, plus the fork's own TP3 padding test from /src/vllm) ----
RUN pip install -q pytest==9.1.1 && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 DSV41_IN_IMAGE=1 python3 -m pytest -q -p no:cacheprovider /opt/llmkube/tests

# ---- 3.7 receipts, entrypoint ----
RUN bash /opt/llmkube/build/receipt.sh
COPY --chmod=0755 entrypoint.sh /usr/local/bin/dsv41-entrypoint.sh

LABEL org.opencontainers.image.title="llmkube-vllm-cuda-gb10-dsv41-b12x" \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND MIT AND LicenseRef-NVIDIA-Proprietary" \
      org.opencontainers.image.description="local-inference-lab vLLM fork + b12x for DeepSeek-V4.1-Flash at TP3 across a three-DGX-Spark ring (GB10, sm_121a)"

ENTRYPOINT ["/usr/local/bin/dsv41-entrypoint.sh"]
