# cuda-gb10-vllm-dsv41-kk

`ghcr.io/defilantech/llmkube-vllm-cuda-gb10-dsv41-kk:candidate-<sha>` (arm64): local-inference-lab's
`integration/karmic-kraken-beta` line ("KK") of their vLLM fork and the b12x kernel library, for DeepSeek-V4.1-Flash at
tensor-parallel 3 across three DGX Sparks (GB10, sm_121a) on the official MXFP4/FP8 weights, with every B12X backend
(attention, linear, MoE, mHC), RoCEnante all-reduce over the switchless ring, DSpark, and the lossless MXFP4-CSF
checkpoint reader. Every input is pinned: the vLLM base by digest, the fork, b12x and NCCL by commit
(`patches/UPSTREAM_COMMITS.txt`), FlashInfer and the other load-bearing packages by version and, where downloaded
directly, by sha256 (`patches/PINNED_DISTS.txt`). `build/pins_gate.py` fails the build on any drift.

It is built like the `cuda-gb10-vllm-dsv41-b12x` sibling (same base, same compile recipe, same gates); what differs is
the pins, the carried patches, and a source-built NCCL. The sibling stays as it is (it serves the mcg trellis
checkpoints on the evidence branch).

## Pins

| component | pin | why this one |
|-----------|-----|--------------|
| vLLM fork | `local-inference-lab/vllm` `1286ae9c9a3fd9376b82da39f7bc87f6a99037d9` (KK beta, tag `vllm-jovian-cu134-beta-1286ae9c...`, 2026-10-03T16:57Z) | the KK beta head chosen in the 2026-10-03 pivot research; the branch has moved on since (`7f172870` on 2026-10-04), and the pin moves deliberately, not by following the branch |
| b12x | `local-inference-lab/b12x` `78ee52c302abcef019d7ce17634c80b76f0975c9` (1.3.0, KK beta, tag `b12x-cu134-beta-78ee52c3...`, 17:14Z the same day) | the fork does not pin b12x by SHA (`setup.py` extra `b12x==1.3.0`); it names the b12x changes it needs in its release records (`.lil/changes/*.json`, "requires"). 78ee52c3 is the first beta b12x release after the vLLM pin and has every one of them (`patches/b12x/APPLIED.md`), including CSF #456, deterministic MoE preparation #464 and collective deadlines #467 |
| NCCL | `NVIDIA/nccl` `73cf112295c33aee2b895f329f592f2a9b4b0f97` (tag `v2.30.7-1`) plus NVIDIA/nccl#2393 | see "NCCL" below |
| CUTLASS DSL | `nvidia-cutlass-dsl[cu13]` 4.7.1 (+ libs) | the fork (`requirements/cuda.txt`) and b12x 78ee52c3 (`pyproject.toml`) both require exactly 4.7.1, so unlike the sibling there is no b12x line in `patches/PIP_CHECK_ALLOW.txt` |
| FlashInfer | 0.6.18.post1 | the fork's pin, unchanged from the sibling |

The fork's build inputs (`csrc/`, `CMakeLists.txt`, `cmake/`, `setup.py`, `requirements/`) are byte-identical at
1286ae9c and at the sibling's ec49d578, so the compile, its CUDA `-dev` set, and every component compiled into
`vllm/_C` (NOTICE, `licenses/`) are the same; none of the 71 fork commits between them touches compiled sources. The
DS4.1-relevant ones include the compressor-state ring fixes (vllm-926, vllm-943: on DGX Spark the model "no longer
drifts away" as a generation gets longer), the CSF loader (vllm-956-*), and the admission-stall fix.

## Carried patches (`patches/<name>/APPLIED.md`, drift-gated)

- **b12x/0001**, local-inference-lab/b12x#457 (switchless-ring RoCE routing), carried until it merges. Regenerated
  against 78ee52c3: every hunk is byte-identical to the sibling's copy and applies with no offset; only one `index`
  line differs because upstream changed `_oneshot_cute.py` outside the patch's hunks.
- **vllm/0001**, DS4.1 MXFP4-CSF at TP3 (authored here, not yet proposed upstream). See "MXFP4-CSF" below.
- **nccl/0001**, NVIDIA/nccl#2393 (AArch64 IB send-path fence). See "NCCL" below.

## MXFP4-CSF

The checkpoint is `local-inference-lab/DeepSeek-V4.1-Flash-MXFP4-CSF` on Hugging Face (MIT; revision
`872da235166458bd6ffa9ee3f3c5c4771b63159c`, last modified 2026-10-01): schema `lil-mxfp4-csf-checkpoint/1`, codec
`row-base-offset1-u24-exceptions/1`, 48 shards under `tensors/` totalling 495.6 GB (the routed-expert scales are
2.3 GB instead of 17.0 GB), plus `manifest.json`, `build-contract.json` and the source metadata under `metadata/`.
Nothing in this image downloads it.

What the image needs, and has:

- the fork's reader: `--load-format mxfp4_csf` (`vllm/model_executor/model_loader/mxfp4_csf_loader.py`, which
  validates `manifest.json` and `build-contract.json`) and `--quantization mxfp4_csf`
  (`vllm/models/deepseek_v4_1/mxfp4_csf.py`, routed experts only; MXFP8 activations by default, BF16 with
  `VLLM_B12X_MOE_FP4_FORCE_A16=1`);
- b12x's CSF preparation and scale-expansion kernels (`b12x/_lib/quant/mxfp4_csf.py`, `CsfScalePlanes`,
  `Mxfp4CsfWeights`), the b12x-450-csf-* changes at 78ee52c3;
- **TP3**: at the pin (and at the branch head) the reader allowlists TP 1, 2, 4 and 8 for DS4.1. Every other
  constraint holds at TP3: 2304 / 3 = 768 local channels is 32-aligned, the rank boundaries are whole 16-row scale
  slabs, the slicer takes any column range, and b12x's compact geometry needs N64/K64 (1536 x 5120 and 5120 x 768 per
  rank). `patches/vllm/0001` adds 3 to the allowlist and nothing else. The in-image tests slice a synthetic CSF expert
  set at TP3 through the fork's own loader on CPU and check every rank's nibbles and decoded scales, and check the
  guard admits exactly TP 1, 2, 3, 4 and 8. **Not qualified on a GB10 ring yet**; if it misbehaves, serve the plain
  official checkpoint with `--load-format b12x` (as the reference TP3 setup does), at about 4.6 GiB less headroom per
  rank (derived).

Serving it is a deployment concern, per the fork's `docs/features/quantization/fp4_csf.md`: the model path is a serve
directory holding the source tokenizer files and a `config.json` whose `quantization_config` has
`quant_method: mxfp4_csf`, `format_version: 1` and an absolute `checkpoint_root` pointing at the CSF directory (the one
with `manifest.json` and `tensors/`), visible in the pod. The published `metadata/config.json` still says
`quant_method: fp8`, so the serve directory's `config.json` is ours to write (copy it and replace
`quantization_config`). The reader rejects expert parallel, data parallel, pipeline parallel and ubatching.

## NCCL

The image keeps NCCL 2.30.7 (the base ships it; installing the fork's wheel lets pip downgrade to torch's 2.29.7 pin,
whose lack of `NCCL_IB_SUBNET_AWARE_ROUTING` times out every queue pair on the ring), and rebuilds its library from
the v2.30.7-1 tag with one carried change: NVIDIA/nccl#2393, an acquire fence in `ncclIbIsend` between the CTS slot's
`idx` check and its `nreqs` load. Without it, on AArch64 the `nreqs` load can see the previous round's value and the
proxy thread can spin forever (NVIDIA/nccl#1983). The fix landed on NCCL's `dev` branch on 2026-10-01 as
`05d026f4685c1c15f53f8f76af91766e3e0eb4b6` (identical diff); no release has it yet.

Decision: include it, because the build is pinned (a release tag plus a two-line diff, both recorded), license-clean
(NCCL is Apache-2.0 with BSD-licensed parts; `LICENSE.nccl`), and cheap: the `nccl-build` stage compiles one
architecture (sm_120 cubins plus compute_120 PTX, both loadable on GB10's sm_121) with NCCL's own Makefile and
defaults, in parallel with the vLLM compile. Building `dev` instead would pull in 651 unreleased commits past the
version the ring was qualified with. The reference TP3 setup runs the same combination (2.30.7 rebuilt with #2393).

How it is installed and checked: the image installs the hash-pinned 2.30.7 wheel as before, then replaces that
wheel's `nvidia/nccl/lib/libnccl.so.2` (the file torch and vLLM's pynccl load) with the stage's library, after
checking its sha256 against the stage's record. `/opt/llmkube/nccl-build.txt` records the commit, the patch sha256
and the library sha256; `build/pins_gate.py` requires the record to name the pinned commit and exactly the carried
patches, the installed library to be the recorded one, and `/src/nccl` to be the pin plus exactly the patch. The
in-image test also calls `ncclGetVersion` (23007). The wheel's metadata still says 2.30.7, which is true of the
library. Drop the patch when a release that contains it is pinned instead.

## Serving env (for the InferenceService)

The image sets `B12X_COMPILE_CACHE_DIR=/opt/llmkube/b12x-cache` (world-writable; back it with a volume),
`CUTE_DSL_ARCH=sm_121a` and `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`. The rest is per deployment. The fork's
launchers at the pin, `scripts/serve-ds41-flash-dspark-tp3-rdma.sh` (TP3 wrapper) and
`scripts/serve-ds4-flash-dspark-tp4-rdma.sh` (lines 333-400 and 468-512), set the env the sibling's README lists (the
b12x switches, `VLLM_USE_V2_MODEL_RUNNER=1`, RoCEnante via `VLLM_ENABLE_ROCE_ALLREDUCE=1`, the NCCL IB settings, and
`NCCL_MAX_NCHANNELS=8` for TP3) and the serve flags `--moe-backend b12x --linear-backend b12x --attention-backend B12X`,
with `--load-format b12x` (plain checkpoint) or `--quantization mxfp4_csf --load-format mxfp4_csf` (CSF). The launcher
`LD_PRELOAD`s a locally patched NCCL; this image needs no preload, because the wheel's library is already the fenced
build.

Settings from the reference TP3 baseline (christopherowen/spark3-vllm-ds41f, a repository without a license: facts
only, nothing copied), to evaluate on the ring rather than bake in as image defaults:

- `B12X_W4A8_TINY_DECODE=0`: that repository reports the tiny-decode W4A8 path omits the SwiGLU clamp and produced
  incoherent output. At 78ee52c3 the path is still on by default (`b12x/moe/fused_moe/_impl.py`, kill switch
  `B12X_W4A8_TINY_DECODE=0`). Unverified by us; the correctness oracle decides.
- `B12X_AUTOTUNE=0`, or a prebuilt compile cache with `cache_only` boots: the autotune race budget defaults to half of
  free device memory, which on unified memory is a likely contributor to the 2026-10-02 first-boot wedge.
- DSpark 5 tokens, `max_num_batched_tokens` 4096, fail-closed memory guards (5 GiB at start, 3 GiB steady).

Not in this image (that baseline's own patches, unlicensed, reachable only through upstream PRs if LIL takes them):
sequence-parallel prefill, dead verification rows skipping routed experts, indexer top-k tie-break by position,
async-proxy fences before TMA stage release, Engram projection sharding, display carve-out weights.

## Build route

Same as the sibling. **Default: compile the fork from source** at the pinned commit for `TORCH_CUDA_ARCH_LIST=12.1a`
with the fork's own recipe (`use_existing_torch.py --prefix`, then `setup.py bdist_wheel --py-limited-api=cp38`),
against the base's torch 2.13.0+cu130; the sibling's identical compile fits the hosted arm runner (whole job 4 h 10 to
4 h 25). **Fallback: a wheel built on a Spark** from the same commit by the same Dockerfile stage
(`--target vllm-wheel`), published with its sha256, via `VLLM_WHEEL_URL` and `VLLM_WHEEL_SHA256` in the workflow env
(the sibling's `docs/cuda-gb10-vllm-dsv41-b12x-spark-build.yaml` shows the BuildKit Job; point it at this directory).
The gate then requires that hash and the pinned commit's tree, so a wheel from another commit cannot pass.

## Publishing and licensing

CI builds and gates on every PR, push and manual dispatch, but pushes `candidate-<sha>` and attests provenance only
from `refs/heads/main` and release tags (`cuda-gb10-vllm-dsv41-kk-v*`).

What is compiled or vendored here is Apache-2.0, BSD-3-Clause or MIT (texts in the top-level `LICENSE.*` files and in
`licenses/`), and nothing is AGPL (`build/deps_gate.py`). The image also carries NVIDIA-proprietary components under
NVIDIA's licenses: `nvidia-cutlass-dsl` 4.7.1 and its libs, a vendor runtime that the fork and b12x require (like
CUDA, nothing here modifies or compiles it), the base's CUDA runtime, and the static CUDA runtime linked into the
source-built NCCL. NOTICE says so, and the OCI label is
`Apache-2.0 AND BSD-3-Clause AND MIT AND LicenseRef-NVIDIA-Proprietary`.

## What the build checks

- `build/pins_gate.py`: `/src/vllm`, `/src/b12x` and `/src/nccl` are checkouts at the pinned commits of the pinned
  repos, clean or differing by exactly the carried patches (recorded sha256, changed files equal the patches' files,
  reversing them restores the pin, each marker present); every tracked `vllm/**/*.py` and `b12x/**/*.py` is
  byte-identical in dist-packages; NCCL's installed library is the recorded source build of the pin plus its patch;
  one vllm is installed and carries the fork commit; each `PINNED_DISTS.txt` package is installed once at its
  version; all three commits and the torch, flashinfer-python and b12x versions must be named, so an emptied file
  fails. `tests/test_pins.py` breaks one pin at a time against real git fixtures (including the NCCL record,
  library, patch and marker).
- `build/deps_gate.py`: `pip check` conflicts on vllm, b12x or a pinned package fail unless a commented regex in
  `patches/PIP_CHECK_ALLOW.txt` accepts them (two entries, both inherited from the base: cuSPARSELt's platform tag and
  torch's NCCL 2.29.7 pin); AGPL in any installed distribution's license metadata fails.
- `tests/test_imports_in_image.py`: the V4.1 import line, pinned versions, b12x's vLLM plugins, GB10-loadable cubins
  in `vllm/_C`, the fork's TP3 padding test, the b12x loader's io_uring build, the carried RoCE ring routing, the
  fenced NCCL (record, sha256, `ncclGetVersion`), and MXFP4-CSF at TP3 (byte-exact slicing per rank, the guard, b12x's
  CSF modules).
- `build/prewarm.py`: imports the V4.1 serve path, the CSF reader and kernels, and RoCEnante's routing, and records
  b12x's compile-cache key. b12x's kernels themselves compile per shape on the GPU at first serve.

Locally, without docker: `python3 -m pytest -q -p no:cacheprovider cuda-gb10-vllm-dsv41-kk/tests` (the in-image suite
skips). Tier-1 gate on a built image: `./scripts/vllm-gb10-dsv41-kk-gate.sh <ref>`.

## Next (needs the ring)

Prebuild the b12x compile cache on one Spark (`B12X_COMPILE_WORKERS=1`, capped race budget) and copy it to the other
two; then boot TP3 with the plain checkpoint first and CSF second, memory guards armed, and measure against the EXL3
ring baseline.
