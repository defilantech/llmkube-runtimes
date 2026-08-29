# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) vLLM runtime, NVFP4 MoE variant.
#
# The first vLLM image in this repo. Every other runtime here is llama.cpp,
# which is the right engine for single-stream work: on a GB10 serving
# Ornith-1.5-35B-A3B at Q4_K_M it reaches ~97 tok/s for one stream. What it
# does not do is scale with concurrency. vLLM with an NVFP4 checkpoint and
# in-checkpoint MTP is reported at ~86 tok/s single-stream but ~440 tok/s
# aggregate across 24 concurrent streams with sub-second TTFT, on one Spark.
#
# A fleet of coder agents is an aggregate workload, not a single-stream one,
# which is the whole reason this image exists.
#
# WHY WE DO NOT BUILD vLLM OURSELVES
#
# Building vLLM for sm_121 is a slow, finicky compile, and the community has
# already converged on eugr/spark-vllm-docker (2.2k stars) as the shared GB10
# stack. Rebuilding it would buy us nothing but a maintenance burden: a vLLM
# pin to track, a CUDA 13 aarch64 toolchain to keep working, and a multi-hour
# image build in a repo whose other images build in minutes.
#
# So we pin their image by DIGEST and add exactly one thing on top.
#
# WHY THE b12x UPGRADE IS THE WHOLE POINT
#
# b12x is the GB10-native MoE backend (a PyPI package, not part of vLLM). The
# base image below is nightly-20260823. The newest b12x on PyPI on that date was
# 1.2.6 (published 2026-08-20), so the base necessarily ships something older
# than 1.3.0 -- which has two bugs that crash startup the moment MTP and CUDA
# graphs are both enabled:
#
#   1. an undefined `metadata_row` in the W4A16 output-drain path, and
#   2. a route-packing workspace allocated with torch.empty on every kernel
#      call, which is illegal under CUDA graph capture, so
#      pack_topk_routes_by_expert raises during capture.
#
# The published recipes for this model work around both by bind-mounting
# patched copies of two b12x files over the container's site-packages. That is
# fine for `docker run` on one box and unusable under a Kubernetes scheduler:
# there is no host-path guarantee, and a pod that lands on a node without the
# files starts and then fails at capture.
#
# b12x 1.3.0 (2026-08-25, two days after this base image) fixes both upstream.
# Verified against the published wheel rather than assumed:
#
#   - `metadata_row` no longer appears anywhere in kernel.py (0 occurrences).
#   - route_pack.py now takes a PREALLOCATED workspace. The torch.empty call
#     survives only as the non-capture fallback, and under capture the code
#     raises a named error instead of silently allocating:
#
#       if torch.cuda.is_current_stream_capturing():
#           raise RuntimeError(f"{name} is not initialized for CUDA graph
#                                capture; provide a preallocated ... workspace")
#
# So pinning b12x removes the need for any patch, which is why this image
# carries none. If a future base image already ships >= 1.3.0 this layer
# becomes a no-op, which is the correct outcome rather than something to
# delete.
#
# ARCHITECTURE: arm64 only. GB10 is aarch64 Grace and the base publishes no
# amd64 manifest. There is nothing to gain from a multi-arch build.

# eugr/spark-vllm-b12x:latest == nightly-20260823, resolved 2026-08-29.
# Pinned by digest, not by tag: `latest` on that repo moves nightly, and an
# image whose contents change under a fixed name is not a runtime we can
# reason about when a serving regression appears.
FROM eugr/spark-vllm-b12x@sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd

# Pinned exactly, not >=. b12x shipped ten releases in the nineteen days to
# 2026-08-25 (1.1.0 on the 6th through 1.3.0 on the 25th); a floating constraint
# here would reintroduce the drift the digest pin above exists to remove.
ARG B12X_VERSION=1.3.0

# --no-deps so the upgrade cannot pull a different torch, cutlass-dsl or vllm
# into an image whose CUDA stack is already correct for sm_121.
#
# Safe here for a specific, checked reason rather than as a general habit:
# b12x's requirement set is BYTE-IDENTICAL across 1.2.6, 1.2.8 and 1.3.0 (same
# torch>=2.12.0, the same exact nvidia-cutlass-dsl==4.6.2 and its four libs
# packages, same apache-tvm-ffi range, same transformers/safetensors floors).
# The upgrade therefore needs nothing resolved -- there is no new dependency to
# skip and no changed pin to violate. Had 1.3.0 moved the cutlass-dsl pin,
# --no-deps would be the wrong tool and the honest fix would be to install the
# matching version explicitly, never to drop the flag and let pip swap torch.
#
# The gate runs `pip check` against the shipped image precisely so that a FUTURE
# bump which does add a dependency fails the build instead of failing on a GB10.
# The wheel is py3-none-any, so there is nothing to compile either.
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages \
        "b12x==${B12X_VERSION}" \
 && python3 -c "import importlib.metadata as m; print('b12x', m.version('b12x'))"

# Fail the BUILD, not a serving run at 3am, if the upgrade did not land the
# fixes this image exists to deliver. Both checks read the installed files
# rather than trusting the version string, because a version number is a claim
# and the file contents are the fact.
RUN python3 - <<'PY'
import importlib.util, pathlib, sys

# find_spec LOCATES the package without EXECUTING it. b12x is a CUDA MoE
# backend and importing it on a GPU-less build runner can fail on device init,
# which would turn this guard into a build break with an unrelated message.
# The files are what we are checking; we never need the module live.
spec = importlib.util.find_spec("b12x")
if spec is None or not spec.origin:
    sys.exit("b12x is not importable after install; the pip layer did not land.")
root = pathlib.Path(spec.origin).parent / "moe/_shared/kernels/w4a16"
if not root.is_dir():
    sys.exit(f"b12x w4a16 kernel directory missing at {root}; the package "
             "layout changed and these guards no longer check anything.")

kernel = (root / "kernel.py").read_text(errors="replace")
if "metadata_row" in kernel:
    sys.exit("b12x kernel.py still references metadata_row; the W4A16 "
             "output-drain bug is present and MTP + CUDA graphs will crash.")

route = (root / "route_pack.py").read_text(errors="replace")
if "is_current_stream_capturing" not in route:
    sys.exit("b12x route_pack.py has no CUDA-graph capture guard; the "
             "workspace is still allocated per call and capture will raise.")

print("b12x w4a16 guards OK: no metadata_row, capture guard present")
PY

LABEL org.opencontainers.image.title="llmkube-vllm-cuda-gb10-nvfp4" \
      org.opencontainers.image.description="vLLM on NVIDIA GB10 (sm_121) for NVFP4 MoE checkpoints with in-checkpoint MTP. Pinned eugr/spark-vllm-b12x base plus b12x>=1.3.0, which removes the W4A16 CUDA-graph patches other recipes bind-mount." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0" \
      dev.llmkube.gpu.arch="sm_121" \
      dev.llmkube.engine="vllm" \
      dev.llmkube.b12x.version="${B12X_VERSION}"

# Entrypoint deliberately inherited from the base. vLLM's server flags belong in
# InferenceService.spec.extraArgs, where an operator can change them without a
# rebuild, not baked in here.
