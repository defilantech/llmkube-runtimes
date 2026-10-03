# cuda-gb10-vllm-dsv41-b12x

`ghcr.io/defilantech/llmkube-vllm-cuda-gb10-dsv41-b12x:candidate-<sha>` (arm64): local-inference-lab's vLLM fork
plus the b12x kernel library, serving DeepSeek-V4.1-Flash at tensor-parallel 3 across three DGX Sparks (GB10,
sm_121a). Every input is pinned: the vLLM base by digest, the fork and b12x by commit
(`patches/UPSTREAM_COMMITS.txt`), FlashInfer and the other load-bearing packages by version and, where downloaded
directly, by sha256 (`patches/PINNED_DISTS.txt`). `build/pins_gate.py` fails the build on any drift.

The fork pads V4.1 attention from 64 heads / 8 output groups to 72 / 9 for TP3 itself
(`vllm/model_executor/models/config.py`, `DeepseekV41ForCausalLMConfig.update_model_config_for_parallelism`), and
its own test for that hook runs in the image.

b12x is pinned at `6380e581`, the head of upstream's `evidence/ds41-x4t-serving-20260929` branch and the only ref
with `b12x/moe/checkpoints/independent.py` (the trellis-dense-checkpoint/1 reader). Three source patches are carried
on the pins until they land upstream, each recorded in `patches/<name>/APPLIED.md` and drift-gated by
`build/pins_gate.py`: b12x#457 (switchless-ring RoCE routing), and a pair that lets the fork's DS4.1 trellis config
(`patches/vllm/0001`) and b12x's checkpoint reader (`patches/b12x/0002`) load exllamav3 mcg K3..K6 trellis
checkpoints as well as lut_e4m3 K2. The vLLM patch is applied before the compile, so the wheel is the pin plus
exactly that patch. That pin still declares `nvidia-cutlass-dsl==4.6.2` (upstream master moved to 4.7.1 with no
source change), so b12x is installed `--no-deps` against the fork's 4.7.1.

Moving from the `0d6600e6` master line to the evidence branch drops three upstream master commits the evidence
branch does not have:

- `ba090286` "Restore tensor-based inference API compatibility": legacy tensor-based GEMM, MoE and attention entry
  points over the typed preparation APIs (it adds `b12x/moe/fused_moe/_compat.py`, absent at `6380e581`), and
  hasattr guards in `b12x/integration/vllm/loader.py` and `plugin.py`. The fork does not need it: its DS4.1 MoE
  and attention paths build `b12x.preparation.PreparedCall` objects (`vllm/model_executor/layers/fused_moe/b12x.py`,
  `vllm/models/deepseek_v4_1/attention.py`), its tensor-FP8 linear always passes an explicit `plan`, every
  `from b12x... import` in the fork's `vllm/` resolves at `6380e581`, and the evidence loader's direct imports
  (`file_source_tensor`, `safetensors_file_sources`, `enable_tqdm` from `weight_utils.py`) exist in the fork.
- `0d6600e6` "Derive block-quantized launch heuristics from reuse and occupancy": launch tuning only
  (`gemm/blockscaled/_tuning.py`, the W4A16 kernel, `fused_moe/_impl.py`).
- `4bacd509` "Upgrade CUTLASS DSL dependencies to 4.7.1": pins and tests only, handled above.

## Build route

**Default: compile the fork from source** at the pinned commit for `TORCH_CUDA_ARCH_LIST=12.1a`, with the fork's
own recipe (`use_existing_torch.py --prefix`, then `setup.py bdist_wheel --py-limited-api=cp38`, as its
`docker/Dockerfile` does), against the base's torch 2.13.0+cu130.

**Fallback: install a wheel built on a Spark** from the same commit by the same Dockerfile stage
(`--target vllm-wheel`), published with its sha256, by setting `VLLM_WHEEL_URL` and `VLLM_WHEEL_SHA256` in the
workflow env. The gate then requires that exact hash, and the pristine `/src/vllm` checkout still has to match
the installed Python tree file for file, so a wheel from another commit cannot pass.

**Decision: pending the first Spark build.** Use the hosted runner if the compile fits in about 4 hours at
MAX_JOBS=2 (the workflow's 350-minute timeout sits under the runner's 360-minute cap); otherwise take the wheel
route. Record the measurement here:

| build | node | MAX_JOBS | wheel stage | full image | peak memory | route chosen |
|-------|------|----------|-------------|------------|-------------|--------------|
| first | ahazidgx3 | 8 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

The Spark build is a one-off BuildKit Job: `docs/cuda-gb10-vllm-dsv41-b12x-spark-build.yaml` (draft; fill in the
namespace, registry and push secret). It builds `--target vllm-wheel` first (exported to
`/var/tmp/dsv41-b12x-wheel` on the node) and then the full image to a scratch tag, and prints `MEASURE` lines.
Gate the scratch image from any machine with docker: `./scripts/vllm-gb10-dsv41-b12x-gate.sh <scratch-ref>`.

### Why not vLLM's precompiled route (`VLLM_USE_PRECOMPILED=1`)

It would drop the compile by grafting an upstream vllm-project wheel's compiled extensions onto the fork's Python
tree. An aarch64 cu130 upstream wheel does exist at the fork's merge base (`0f8fa53a`, 2026-09-16), but the fork
has changed compiled code since then: `csrc/libtorch_stable/torch_bindings.cpp` adds a
`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.out` overload (with a new `kv_mxfp8` argument),
`csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` changes the kernel, and FlashKDA is patched
through CMake. The V4 attention and the DSpark drafter call that overload
(`vllm/models/deepseek_v4/attention.py:873`, `vllm/models/deepseek_v4/nvidia/dspark.py:280`), so upstream's `_C`
would be missing an op the serve path needs. The fork's own release wheels
(`vllm-jovian-cu134-beta-*`) are x86_64, CUDA 13.4, SM120a, so they do not apply either. Upstream's Spark launcher
does the same thing this image does in effect: it runs the fork from a source tree with its own `.venv`
(`scripts/serve-ds4-flash-dspark-tp4-rdma.sh:6,37-38,333`), that is, a local build of this exact source.

### Why not upstream's container

Upstream serves from eugr's `spark-vllm-docker` image with the fork and b12x bind-mounted
(`scripts/serve-ds4-flash-dspark-tp4-rdma.sh:34,301-305`). This image is built instead from pinned sources on the
pinned vLLM base, per the runtime supply-chain policy.

## Serving env (for the InferenceService)

The image sets `B12X_COMPILE_CACHE_DIR=/opt/llmkube/b12x-cache` (world-writable; back it with a volume so a restart
does not recompile), `CUTE_DSL_ARCH=sm_121a` and `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`. Everything else
upstream's launcher sets is per-deployment. From `scripts/serve-ds4-flash-dspark-tp4-rdma.sh` at the pinned
commit (lines 333-384, 394-400) and its TP3 wrapper `scripts/serve-ds41-flash-dspark-tp3-rdma.sh:5-21`:

- b12x: `VLLM_PLUGINS=b12x_loader`, `B12X_WEIGHTS_COMPILE_WORKERS=4`, `B12X_STATE_COMPILE_WORKERS=4`,
  `B12X_BIND_COMPILE_WORKERS=4`, `VLLM_USE_B12X_WO_PROJECTION=1`, `VLLM_USE_B12X_MHC=1`,
  `VLLM_USE_B12X_FP8_GEMM=1`, `VLLM_USE_B12X_MOE=1`, `VLLM_USE_B12X_SPARSE_INDEXER=1`,
  `B12X_MLA_SM120_UNIFIED=1`, `B12X_DENSE_SPLITK_TURBO=1`, `B12X_W4A16_TC_DECODE=1`, `B12X_MOE_FORCE_A8=1`
- vLLM: `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_MEGA_AOT_ARTIFACT=1`,
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, `VLLM_USE_FLASHINFER_SAMPLER=1`,
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_ENABLE_PCIE_ALLREDUCE=0`,
  `DG_JIT_USE_NVRTC=0`, `USE_CUDNN=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `SAFETENSORS_FAST_GPU=1`, `OMP_NUM_THREADS=16`
- all-reduce over RoCE with b12x's collectives (the launcher's default `ALLREDUCE=rocenante`):
  `VLLM_ENABLE_ROCE_ALLREDUCE=1`, `VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`, `VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB`,
  `B12X_ROCE_CACHE_DIR=<compile cache>/roce`, `B12X_ROCE_TRAFFIC_CLASS=106`
- NCCL: `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA=<RoCE devices>`, `NCCL_IB_GID_INDEX=3`, `NCCL_IB_TC=106`,
  `NCCL_IB_MERGE_NICS=1`, `NCCL_NET_MERGE_POLICY=ALL`, `NCCL_NET_MERGE_LEVEL=SYS`, `NCCL_CUMEM_ENABLE=0`,
  `NCCL_RUNTIME_CONNECT=1`, `NCCL_P2P_LEVEL=SYS`, `NCCL_IGNORE_CPU_AFFINITY=1`, `NCCL_NET_PLUGIN=none`, and for
  TP3 `NCCL_MAX_NCHANNELS=8`
- TP3 wrapper sizing: `KV_CACHE_MEMORY_BYTES=6442450944`, `MAX_NUM_SEQS=2`, `MAX_NUM_BATCHED_TOKENS=1024`,
  Engram `{"table_memory":"disk","disk_resident_scales":false}`, `--limit-mm-per-prompt {"image":0}`

The serve flags (`--load-format b12x --moe-backend b12x --linear-backend b12x --attention-backend B12X`, DSpark
speculative config, parsers `deepseek_v41`) are at lines 468-512.

The launcher also `LD_PRELOAD`s a locally patched NCCL 2.30.7 (lines 10-11, 186-189, 370-371). This image does not
build NCCL. The fork's CUDA path requires NCCL >= 2.29.7 (`vllm/distributed/device_communicators/pynccl_wrapper.py`,
`ncclCommSuspend`/`ncclCommResume`), which torch 2.13.0's own pin (`nvidia-nccl-cu13==2.29.7`) meets; newer
properties (2.31) are optional. The shipped version is in `/opt/llmkube/pip-freeze.txt`.

## Publishing and licensing

CI builds and gates on every PR, push and manual dispatch, but pushes `candidate-<sha>` and attests provenance only
from `refs/heads/main` and release tags (`cuda-gb10-vllm-dsv41-b12x-v*`). A `workflow_dispatch` on any other branch
builds and gates without pushing.

The image is not all Apache-2.0/BSD/MIT: besides the Apache-2.0, BSD-3-Clause and MIT components (texts in the
top-level `LICENSE.*` files and in `licenses/`, one per component the fork's CMake compiles into `vllm/_C`, at the
revision it pins), it carries NVIDIA-proprietary Python packages (`nvidia-cutlass-dsl` 4.7.1 and its libs, which
the fork and b12x require and the base already ships at 4.6.2) and the base's CUDA runtime, under NVIDIA's licenses.
NOTICE says so, and the OCI label is `Apache-2.0 AND BSD-3-Clause AND MIT AND LicenseRef-NVIDIA-Proprietary`.

## What the build checks

- The compile stage adds CUDA 13.0 Update 3 `-dev` packages (cusparse, cusolver, cufft, culibos, nvtx,
  profiler-api) pinned by version from NVIDIA's ubuntu2404/sbsa repo. The base is a runtime image and torch's
  `ATen/cuda/CUDAContextLight.h` includes `cusparse.h` and `cusolverDn.h`. That stage is not shipped.
- `build/deps_gate.py`: `pip check` conflicts fail the build when either side is vllm, b12x or a
  `PINNED_DISTS.txt` package, unless a commented regex in `patches/PIP_CHECK_ALLOW.txt` accepts it.
  Other inherited conflicts are printed. It also scans every installed distribution's License,
  License-Expression and classifiers for AGPL/Affero.

- `build/pins_gate.py`: `/src/vllm` and `/src/b12x` are checkouts at the pinned commits of the pinned repos, clean
  or differing by exactly the carried patches (recorded sha256, changed files equal the patches' files, reversing
  them restores the pin, each patched file installed byte-identical, each marker present).
  - Every tracked `vllm/**/*.py` and `b12x/**/*.py` is byte-identical in dist-packages.
  - One vllm is installed and its version carries the fork commit; each `PINNED_DISTS.txt` package is installed
    exactly once at its version; the vLLM install record matches the wheel ARGs; vendored files match
    `MD5SUMS.txt` (none today).
  - Both pin files must name vllm and b12x (commits) and torch, flashinfer-python and b12x (versions), so an
    emptied file fails, and origins must be exactly the pinned GitHub repo.
  - `tests/test_pins.py` breaks one pin at a time against real git fixtures.
- `tests/test_imports_in_image.py`: the V4.1 import line, pinned versions, b12x's `vllm.general_plugins` entries
  (including upstream launcher's `b12x_loader` check), `sm_121` in `vllm/_C`, and the fork's
  `tests/v1/attention/test_b12x_sparse_mla_api.py::test_deepseek_v41_tp3_padding_uses_generic_parallel_hook`.
  That hook is gated on a CUDA SM12x platform, which a docker build does not have, so without a visible GB10 the
  test runs under `tests/plugins/sm12x_platform_shim.py` (replaces only the two platform queries the guard reads);
  on a Spark it runs unshimmed.
- `build/prewarm.py`: imports the V4.1 serve path and records b12x's compile-cache key. b12x's kernels themselves
  compile per shape on the GPU at first serve.

Locally, without docker: `python3 -m pytest -q -p no:cacheprovider cuda-gb10-vllm-dsv41-b12x/tests` (the in-image
suite skips).
