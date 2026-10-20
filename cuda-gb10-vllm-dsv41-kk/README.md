# cuda-gb10-vllm-dsv41-kk

`ghcr.io/defilantech/llmkube-vllm-cuda-gb10-dsv41-kk:candidate-<sha>` (arm64): local-inference-lab's
`integration/karmic-kraken-beta` line ("KK") of their vLLM fork and the b12x kernel library, for DeepSeek-V4.1-Flash at
tensor-parallel 3 across three DGX Sparks (GB10, sm_121a) on the official MXFP4/FP8 weights, with every B12X backend
(attention, linear, MoE, mHC), RoCEnante all-reduce over the switchless ring, DSpark, and the lossless MXFP4-CSF
checkpoint reader. Every input is pinned: the vLLM base by digest, the fork, b12x and NCCL by commit
(`patches/UPSTREAM_COMMITS.txt`), FlashInfer and the other load-bearing packages by version and, where downloaded
directly, by sha256 (`patches/PINNED_DISTS.txt`). `build/pins_gate.py` fails the build on any drift.

It is built like the `cuda-gb10-vllm-dsv41-b12x` sibling (same base, same compile recipe, same gates); what differs is
the pins, the carried patches (this repository's own, plus 24 verbatim patches from `christopherowen/spark-ds41f`, see
"Carried from spark-ds41f"), and a source-built NCCL. The sibling stays as it is (it serves the mcg trellis
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
- **vllm/0002-0004**, unified-memory headroom for DSpark on three Sparks (authored here, not proposed upstream; see
  `patches/vllm/APPLIED.md` for each one's evidence): 0002 caps DeepSeek-V4.1's fp32 RoPE tables at `max_model_len`
  when `LLMKUBE_DSV41_ROPE_ROWS=max_model_len` (-384 MiB/rank at 262K, kept rows byte-identical, default unchanged);
  0003 stops allocating `candidate_block_buffer`, which the b12x attention never reads (-32 MiB/rank); 0004 writes
  phase-tagged PyTorch allocator snapshots when `LLMKUBE_MEMORY_STATS_DIR` is set (inert otherwise), for the memory
  census. `_mtp_hidden_buffer` is deliberately not gated: the model runner slices it on every DSpark step.
- **nccl/0001**, NVIDIA/nccl#2393 (AArch64 IB send-path fence). See "NCCL" below.
- **vllm/spark-ds41f/** (21) and **b12x/spark-ds41f/** (3): the DSv4.1 Lane A series from
  `christopherowen/spark-ds41f` at `a2a3abae7b`, applied after the patches above. See "Carried from spark-ds41f".
- **vllm/stacked/0001**, Engram next-chunk prefetch (authored here, not proposed upstream): our own patch on top of
  spark-ds41f 0002, applied after that series. See "Engram prefetch".

## Carried from spark-ds41f

The DSv4.1 Lane A serving image is this image's published `8fe1e18d...` digest plus three in-cluster lab layers
(b1, b2, b3) that applied 24 patches from `christopherowen/spark-ds41f` at
`a2a3abae7b395f98ef64dcc77beef1584d756cd6` (Apache-2.0 since `ad4605b9`; NOTICE, `LICENSE.spark-ds41f-Apache-2.0`).
They are carried here instead, so CI builds and attests what is served. Each file is byte-for-byte what spark-ds41f
publishes (headers, authorship and commit messages kept), so their numbering repeats ours (their vLLM 0001 is not
ours): they live in `patches/<name>/spark-ds41f/`, with a `series` file giving the apply order, an `APPLIED.md` in the
same format as ours (spark-ds41f path, head SHA, author, scope, default, marker, sha256), and an md5 line each in
`patches/MD5SUMS.txt`. `build/apply_carried_patches.sh` applies `patches/<name>/*.patch` in filename order and then
the series strictly in listed order, in the compile stage and in the image's checkouts alike; `build/pins_gate.py`
checks the checkouts against the same order. Strict-apply check: on vLLM 1286ae9c plus our 0001-0004, the 21 vLLM
patches apply with `git apply` in series order, and on b12x 78ee52c3 plus our 0001 the three b12x patches do, all with
no fuzz (eleven vLLM patches land at line offsets with their full context matching; the b12x ones at none). All 24
change Python only (plus upstream test files), so the compiled extensions are the same.

The order is the lab layers' order: b1, then b2, then b3. Which patches are always on and which need a switch, from the
lab READMEs (llmkube-internal `plans/dsv41-3spark/manifests/lane-a-b1/README.md`, "The patches", and
`plans/dsv41-3spark/manifests/lane-a-b23/README.md`, "The patches and their switches"):

| Patch | Layer | What it does | Default |
|-------|-------|--------------|---------|
| vLLM 0024 | b1 | the ratio-2 compressor ring is written during decode (it was never written, so steps starting at an odd position built compressed KV entries from a zeroed predecessor, reused through prefix caching) | always on |
| vLLM 0178 | b1 | DeepSeek's 2026-10-16 V4.1 encoding fix: the current turn's reasoning survives its tool results, tool-result images follow their placeholders in call order (new `DeepseekV41Renderer`), image tokens in text are refused | always on with `--tokenizer-mode deepseek_v41` |
| b12x 0003 | b1 | DSA radix top-k breaks exact score ties by lowest logical position instead of thread timing | always on |
| b12x 0004 | b1 | dense GEMM fences generic-proxy stage reads before the TMA refill (shared-expert down projection returned wrong columns beside the routed MoE) | always on |
| b12x 0005 | b1 | the same fence in the BF16 prefill projection, the mHC TF32 and BF16 TMA projections and the contiguous attention forward | always on |
| vLLM 0001 | b2 | pads the Engram WKV projection so `projection_tp` works at TP3 | off: `--engram-config` with `"projection_tp":true` |
| vLLM 0002 | b2 | reads disk Engram rows on a reader thread beside the graph launch (carried because 0015 builds on it) | off: `SPARK3_ENGRAM_ASYNC=1` |
| vLLM 0005 | b2 | pins adaptive-verification cost curves to files per shape set | off: `SPARK3_DSPARK_COST_DIR=<dir>` |
| vLLM 0007 | b2 | gathered Markov bias for the V4.1 drafter (carried because 0012 builds on it) | off: `dspark_draft_topk` in the speculative config |
| vLLM 0010 | b2 | dead verification rows below a survival cut read no expert weights | off: `SPARK3_DSPARK_DEAD_ROWS_TAU` (e.g. `0.2`) |
| vLLM 0012 | b2 | vocabulary-parallel greedy drafts | off: `SPARK3_DSPARK_VOCAB_PARALLEL=1`, with `"draft_sample_method":"greedy"` |
| vLLM 0013 | b2 | column-shards the drafter's replicated `main_proj` | off: `SPARK3_DSPARK_MAIN_PROJ_TP=1` |
| vLLM 0019 | b2 | reads the custom-op schema once in `fill_defaults` (host overhead, short-prompt TTFT) | always on |
| vLLM 0020 | b2 | compiles the dead-row kernels at warmup | always on (matters only with 0010 on) |
| vLLM 0039, 0041 | b2 | disjoint fabricated-context and dummy blocks per KV group (non-finite rows at TP3 startup) | always on |
| vLLM 0042 | b2 | marks the drafter's padding rows for the MoE routers | always on (`VLLM_MOE_SKIP_PADDING`, the fork's default 1) |
| vLLM 0144 | b2 | runs the adaptive-verification draft budget eagerly; reports JIT compiles | always on |
| vLLM 0014-0017 | b3 | sequence-parallel (SP) prefill: state, row-wise dataflow, CED encoder, local attention front | always on at TP > 1 (see 0018) |
| vLLM 0018 | b3 | SP whenever a prompt-processing forward's padded hidden message exceeds the one-shot all-reduce limit (205 tokens at TP3); removes the earlier `SPARK3_DS41_PREFILL_SP_MIN_ROWS` switch | always on at TP > 1, no switch |
| vLLM 0025 | b3 | splits the indexer's rows across ranks under SP | always on at TP > 1 |
| vLLM stacked/0001 (ours) | after b3 | Engram next-chunk prefetch (see "Engram prefetch") | off: `SPARK3_ENGRAM_PREFETCH=1`, needs `SPARK3_ENGRAM_ASYNC=1` |

So with no switch set, this image behaves like the Lane A b1 lab image plus the b2 fixes marked "always on" and SP
prefill. SP prefill has no off switch: it turns itself off only at tensor parallel size 1, with pipeline parallelism,
static sequence parallelism or request boundary checkpoints. The rest of b2 (Engram projection TP, async Engram rows,
the DSpark set) is off until an InferenceService sets its switch.

Left out of the lab layers and so not carried: vLLM 0027 (explicit RoCE collective policy) and b12x 0010, which it
needs and which conflicts with our b12x 0001 in `_roce_proxy.c`; and 0144's sibling 0184 (linear draft budget).

## Engram prefetch

`patches/vllm/stacked/0001-engram-prefetch-next-chunk.patch`, ours. It changes code spark-ds41f 0002 adds, so it cannot
live in `patches/vllm/` (applied first): it is in its own series directory, `stacked/`, which applies after
`spark-ds41f/` because series directories apply in directory-name order (`tests/test_pins.py` holds it to the rules
for our own patches and checks that order). It keeps 0002's `stage_disk_async` signature line verbatim, as a wrapper,
because that line is 0002's marker and the pins gate requires every carried patch's marker in the installed files.

With `SPARK3_ENGRAM_ASYNC=1` (0002) the reader thread reads a step's disk Engram rows beside the forward, but Engram
layer 1 needs them almost at once, so a 4,096-token prefill step on three Sparks still waited on that read. With
`SPARK3_ENGRAM_PREFETCH=1` as well, while one chunk of a long prompt runs, the host hashes the next chunk's tokens
(the same n-gram hash as the GPU; `tests/test_imports_in_image.py` checks it against b12x's CPU reference) and reads
its rows into a second staging set. The next step uses that set only if its own IDs match the prefetched ones exactly,
and reads as usual otherwise, so a wrong guess costs time, never correctness. It acts only on a batch of one request
that is still prefilling, the long-prompt agent case, and skips a final chunk short enough to run padded through a
CUDA graph. `SPARK3_ENGRAM_PREFETCH_VERIFY=1` re-reads every prefetched set and compares it byte for byte before use
(a correctness switch: it costs more than prefetch saves).

Measured in the lab on the Lane A configuration (both switches on, against the same configuration without
prefetch): cold prefill +5-7% at 10K, 40K and 100K tokens (3.40K, 3.50K, 3.54K tok/s), a 336K-token prompt 97.4 s
instead of 104.5 s and a 515K-token one 156.2 s instead of 165.5 s, decode unchanged, long-context recall 6/6;
every step of a long prompt used prefetched rows (1,024 steps, no wrong guess). With verify on, 577 prefetched sets
per rank were byte-identical to a fresh read.

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

Nothing is taken from that baseline, which still has no license. The same mechanisms (sequence-parallel prefill, dead
verification rows skipping routed experts, indexer top-k tie-break by position, async-proxy fences before TMA stage
release, Engram projection sharding) are now carried from `christopherowen/spark-ds41f`, a separate repository by the
same author that is Apache-2.0 (see "Carried from spark-ds41f"). Display carve-out weights are not in this image.

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
  repos, clean or differing by exactly the carried patches, ours, then the spark-ds41f series in its listed order,
  then our `stacked/` series
  (recorded sha256, changed files equal the patches' files, reversing them restores the pin, each marker present,
  every series patch listed once); every tracked `vllm/**/*.py` and `b12x/**/*.py` is
  byte-identical in dist-packages; NCCL's installed library is the recorded source build of the pin plus its patch;
  one vllm is installed and carries the fork commit; each `PINNED_DISTS.txt` package is installed once at its
  version; all three commits and the torch, flashinfer-python and b12x versions must be named, so an emptied file
  fails. `tests/test_pins.py` breaks one pin at a time against real git fixtures (including the NCCL record,
  library, patch and marker, and the series rules: listed order, unlisted or missing files, rows). The committed
  series is checked against the Lane A order, its rows, head SHAs, markers and scope (its package tree and upstream's
  `tests/`), and `MD5SUMS.txt`. Two rules for this repository's own patches (scope to the package tree; no two patches
  change one file) do not apply to the series, which changes upstream tests and is a stack; the Lane A lab layers had
  to deselect those two tests for that reason, and `tests/test_pins.py` scopes them instead.
- `build/deps_gate.py`: `pip check` conflicts on vllm, b12x or a pinned package fail unless a commented regex in
  `patches/PIP_CHECK_ALLOW.txt` accepts them (two entries, both inherited from the base: cuSPARSELt's platform tag and
  torch's NCCL 2.29.7 pin); AGPL in any installed distribution's license metadata fails.
- `tests/test_imports_in_image.py`: the V4.1 import line, pinned versions, b12x's vLLM plugins, GB10-loadable cubins
  in `vllm/_C`, the fork's TP3 padding test, the b12x loader's io_uring build, the carried RoCE ring routing, the
  fenced NCCL (record, sha256, `ncclGetVersion`), MXFP4-CSF at TP3 (byte-exact slicing per rank, the guard, b12x's
  CSF modules), and spark-ds41f 0178 (`--tokenizer-mode deepseek_v41` loads `DeepseekV41Renderer`, and the fork's
  `tests/tokenizers_/test_deepseek_v41.py`, which 0178 updates, passes against the installed package). The fork tests
  the b2 and b3 patches add or change are not run at build (four of the nine files need CUDA); the lab ran them in a
  GPU test pod.
- `build/prewarm.py`: imports the V4.1 serve path, the CSF reader and kernels, and RoCEnante's routing, and records
  b12x's compile-cache key. b12x's kernels themselves compile per shape on the GPU at first serve.

Locally, without docker: `python3 -m pytest -q -p no:cacheprovider cuda-gb10-vllm-dsv41-kk/tests` (the in-image suite
skips). Tier-1 gate on a built image: `./scripts/vllm-gb10-dsv41-kk-gate.sh <ref>`.

## Next (needs the ring)

Prebuild the b12x compile cache on one Spark (`B12X_COMPILE_WORKERS=1`, capped race budget) and copy it to the other
two; then boot TP3 with the plain checkpoint first and CSF second, memory guards armed, and measure against the EXL3
ring baseline.
