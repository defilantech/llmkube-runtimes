# GB10 native ExLlamaV3 runtime (`cuda-gb10-exllamav3-dsv41/`)

Planned image: `ghcr.io/defilantech/llmkube-exllamav3-cuda-gb10-dsv41-exl3`

Serves DeepSeek-V4.1-Flash EXL3 (SAGE, 1.59 bpw) on **one** DGX Spark with
**native ExLlamaV3**, tensor-parallel 1. It is the one-box counterpart to the
`vllm` ring image: the same checkpoint geometry, a different engine, and a
different memory strategy.

**Status: server written, not yet measured.** The aarch64 engine build is proven
on real hardware (see "Verified on linux/aarch64" below); the image, its build
guards, the launcher, the Tier-1 gate and the CI workflow are in place and green
locally; and the OpenAI-compatible server is implemented. What no page here has
yet is a number from the target Spark, because the checkpoint is not staged
there. Everything below that cites throughput is the community recipe author's.

## Why this image exists

LLMKube's other EXL3-on-Spark builds run **vLLM**, and between the stage and the
serve they **drop page cache** (`config/samples/jobs/drop-page-cache.yaml` in
the LLMKube repo) because a vLLM worker refuses to start when CUDA-free memory at
init sits below `gpuMemoryUtilization x total`, and a freshly staged checkpoint
leaves that memory counted as page cache.

Native ExLlamaV3 wants the opposite. On a GB10 it runs the GPU in **ATS
addressing mode** and aliases the safetensors straight out of an `mmap` instead
of copying them into CUDA allocations. That alias is the only reason roughly
107 GiB of weights fit a 128 GB box whose memory is shared between CPU and GPU.
Evicting page cache evicts the pages the loader is trying to alias.

So this image exists to serve the one-Spark path that the ring cannot, and the
two paths disagree about the single most important resource on the machine.
Anything copied between them, including well-meant cleanup, breaks this one.

## The licence boundary (why this is built from source)

The community recipe this follows,
[`vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe`](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/tree/main/one-spark-tp1),
is a well-run reference. Its server stack is not usable here, and the reason is
licensing, not quality. Every component, checked against its own `LICENSE`:

| Component | Licence | Can it ship here (Apache-2.0, none AGPL)? |
|---|---|---|
| ExLlamaV3 upstream (`turboderp-org/exllamav3`) | MIT | Yes, with attribution |
| ExLlamaV3 fork `vcruz305/exllamav3`, `feat/gb10-ats-load` @ `954a8ca6e59d48c3e3462068ecf083fe9990f4dc` | MIT, inherited | Yes, with attribution |
| Recipe repository | **AGPL-3.0-only** (its `LICENSE`: `SPDX-License-Identifier: AGPL-3.0-only`) | No: reference it, never vendor it |
| `vcruz305/vllm-exl3` | **AGPL-3.0** | No |
| `tools/patch_exllamav3_aarch64.py` (fetched from `vllm-exl3`) | **AGPL-3.0** | No: we write our own |
| TabbyAPI (`theroyallab/tabbyAPI`) | **AGPL-3.0** | No |
| vLLM | Apache-2.0 | not used on this path |
| NVIDIA CUDA base images | NVIDIA container terms | Yes, as `cuda-gb10/` already does |

**TabbyAPI is the binding one.** It is the "official API server for ExLlamaV3"
and the server the recipe runs, and it is AGPL-3.0. This repo's policy is that
every layer is pinned and **none is AGPL**, and CI enforces it (the
`build-cuda-gb10-vllm-dsv41-exl3.yml` licence step greps the tree for AGPL text
and asserts the licence files). An AGPL server cannot live here without
changing that policy.

The recipe's own `THIRD_PARTY_NOTICES.md` agrees on every pinned revision and
licence above, and pins `vllm-exl3` as AGPL-3.0-only itself.

Two things are worth separating, because they are different risks:

- **Legal exposure.** AGPL obligations attach on distribution and on offering a
  modified network service. Running TabbyAPI internally to measure a model does
  not trigger them. Nothing here is a claim that the recipe is unsafe to run.
- **Repo policy.** Publishing an AGPL image from this Apache-2.0 repository
  breaks its stated boundary. That is a repo-identity decision, and the answer
  taken here is to rebuild rather than pull someone else's AGPL image, which is
  exactly what the GLM-5.3 build did.

So: **the server is ours.** Everything else on the path is MIT or Apache-2.0 and
vendors cleanly.

## What the upstream recipe does, and what we reuse

Reused (MIT, from the fork at the pinned commit):

- `DeepseekV41ForCausalLM` (`exllamav3/architecture/deepseek_v41.py`) and the
  DSpark / MTP drafter.
- The GB10 ATS zero-copy loader.
- `util/align_safetensors.py`, the 64-byte re-lay. Safetensors writers pack
  tensors back to back, so most shards land `int16` trellis data on odd offsets
  and the loader must copy those into CUDA memory instead of aliasing them. The
  recipe author measures that at **67.41 GiB** without the re-lay, which is the
  difference between fitting and not. The re-lay only moves tensors onto a grid
  and inserts filler; no weight value changes.
- The EXL3 attention / MTP overlay (12 files, **10.61 GiB**) is a model artifact,
  not code, and is staged with the pack, never baked into the image.

Runtime knobs the library reads (passed through by our entrypoint, not
reimplemented): `EXL3_ATS_MMAP=1`, `EXL3_ATS_COPY` (a regex over tensor names:
matches are copied into CUDA memory, the rest stay aliased; the recipe's
negative lookahead keeps the MTP drafter aliased, which is the whole trick),
`EXL3_DSPARK_CONF`, `CHUNK`, `CTX`.

Not reused: the recipe's orchestration scripts (`AGPL-3.0-only`) and the
`vllm-exl3` build helper (`AGPL-3.0`). We reimplement both.

## The server (our code)

ExLlamaV3 ships no HTTP server; TabbyAPI is that layer. We write a small
Apache-2.0 server over the MIT library's own API (`Config.from_directory` →
`Model.from_config` → `Cache` → `model.load` → `Tokenizer.from_config` →
`Generator`, as the `examples/` confirm) and keep the scope deliberately thin:

- **Load:** the checkpoint from `EXL3_MODEL_DIR`, plus `EXL3_DRAFT_MODEL_DIR` as
  the `Generator(draft_model=...)` drafter, on `EXL3_DEVICE`. The `EXL3_ATS_*`
  and `EXL3_DSPARK_*` knobs the loader reads are passed through untouched. The
  load runs in a background thread while the port is already bound, so a client
  watches the transition instead of seeing connection refused for a minute.
- **Endpoints:** `GET /v1/models`, `POST /v1/completions`,
  `POST /v1/chat/completions` (streaming and non-streaming). Two health
  surfaces, deliberately split: `/health` is liveness and always 200 while the
  process is up, reporting load state and error; `/ready` is readiness and 503
  until a model is loaded. TCP being open is not readiness, and a failed load
  must not look like a dead process.
- **Chat templating:** the checkpoint's own template, read from
  `chat_template.jinja` or `tokenizer_config.json` and rendered with Jinja2.
  Hand-formatting a chat prompt fails silently (the model still answers, badly),
  so a missing template raises rather than guesses.
- **Out of scope:** multi-model routing, LoRA, auth, `config.yml`
  compatibility, concurrent streams. `Generator` is serialized behind one lock.

**Measurement parity is a requirement, not a nicety.** To reproduce the
recipe's published numbers the server must default to the same protocol the
recipe used: temperature 0, thinking off, and DSpark MTP drafting at the same
block size and confidence gate. A server with different defaults produces
incomparable numbers, and an incomparable number is worse than none.

## The aarch64 build (highest risk, spiked)

The extension is compiled for `sm_121` aarch64 with CUDA 13.0 and
`TORCH_CUDA_ARCH_LIST=12.1a`. The recipe makes it compile on aarch64 by patching
out x86 intrinsics, and the file it fetches for that is **AGPL-3.0**, so it
cannot be vendored here. We author our own.

**Spiked, not assumed.** The compile was run unpatched on a real `linux/aarch64`
engine (Docker, CUDA 13.0.1-devel base, same digest as `cuda-gb10/`), against the
pinned fork commit. Findings, in the order the build hits them:

1. **`torch` installs cleanly for aarch64.** `torch 2.14.0+cu130`, `cuda 13.0`
   from PyTorch's `cu130` index on `linux/arm64`. This was the dependency
   question, and it has a yes.
2. **`avx2_target.cpp` / `avx512_target.cpp`: `__builtin_cpu_supports` is
   x86-only.** The unpatched build fails at 16 seconds, in the C++ host compile
   rather than nvcc:
   `error: '__builtin_cpu_supports' was not declared in this scope`.
   The build stops at the first failing translation unit, so each fix reveals
   the next.
3. **`avx2_target.h` / `avx512_target.h`: the `target(...)` attributes are
   x86-only.** They are applied under `#ifdef __linux__`, and GCC on aarch64 does
   not accept `target("avx2")` / `target("avx512f,avx512bw")`.
4. **`cpu/moe_handoff.cu`: `__builtin_ia32_pause()` is x86-only.** `cpu_pause_()`
   compiles on aarch64 with nvcc only once it has an ARM branch.
5. **`cpu/moe_mul1.cpp` includes `immintrin.h` unconditionally.** This is the
   x86 CPU MoE offload GEMM (AVX-512 VNNI). The header does not exist on
   aarch64, and the translation unit is x86 by construction.

**The shape of our patch** follows directly, and point 5 is the one that needs a
decision rather than a substitution. `moe_handoff.cu` calls
`exl3_moe_cpu_forward_raw` and `exl3_moe_cpu_stage_experts` **unconditionally**
(its own comment says symbol resolution does not depend on the runtime toggle), so
the TU cannot simply be excluded: on aarch64 it is replaced by a stub that
defines the exported `exl3_moe_cpu_*` symbols, reports no AVX tier, and does no
work. That is safe for this workload rather than merely convenient: the CPU MoE
offload path is not exercised by a single-Spark GPU run, which is the only thing
this image serves. The GPU and ATS paths, which are what the model actually uses,
are untouched by any of the five points.

This is an Apache-2.0 patch written from the compile failures above, not ported
from the AGPL file, and it lives in `cuda-gb10-exllamav3-dsv41/patches/` as
`0001-aarch64-build.patch`. If a future build reaches a point where the aarch64
path must do real CPU MoE work, this document gets rewritten rather than worked
around.

### Verified on linux/aarch64

Not asserted, run. Against the pinned commit, on a real `linux/aarch64` engine:

| Check | Result |
|---|---|
| `git apply --check` on the pinned pristine tree | applies cleanly |
| `git apply --check` after the patch is already applied | **rejects** with `patch failed: exllamav3/exllamav3_ext/avx2_target.cpp:5` |
| `pip install --no-build-isolation --no-deps .` | exit 0, wheel `exllamav3-1.4.9-cp312-cp312-linux_aarch64.whl` |
| `exllamav3_ext` dlopen | loads, every symbol resolved |
| Device code (`cuobjdump --list-elf`) | **131 native `sm_121a` cubins** |
| PTX (`cuobjdump --list-ptx`) | **0** |
| `exl3_moe_cpu_has_avx2()` at runtime | `False`, as the stub intends |

The double-apply row is the one that matters for the guard: a drift check that
cannot fail is not a guard, and this one demonstrably can. The codegen rows are
the whole reason the image exists, asserted on the artifact rather than assumed
from the arch flag. `torch 2.14.0+cu130` installs cleanly for aarch64 from
PyTorch's `cu130` index, so the dependency question is closed.

## Gates

Same two-tier model as the other runtime images (see the README). Build-stage
guards, all GPU-independent, all failing the build rather than shipping a
quietly broken image:

1. The pinned ExLlamaV3 commit is the one that was checked out.
2. **Codegen guard.** `cuobjdump` finds native `sm_121` device code and no PTX,
   exactly as `cuda-gb10/` asserts. A wrong toolkit, a dropped arch flag, or an
   upstream change to the default list all produce a build that succeeds, an
   image that runs, and a GB10 quietly back on JIT. This is the guard the image
   exists for.
3. **Licence guard.** The MIT text and `NOTICE` are present, and a tree grep
   fails on any AGPL text. Because the whole reason this is built from source
   is the licence boundary, the guard is checked on the source tree before the
   expensive build, not only inside it.
4. The server imports and reports a version; the codegen inventory is written to
   `/app/CUDA_ARCHS` as a receipt.

CI then runs `scripts/exllamav3-gb10-gate.sh` against the **final image**. Like
the vLLM gates, it must **not** import `exllamav3_ext` on a GPU-less host: the
extension initializes CUDA on import, so on CI it would fail for a reason that
has nothing to do with the image. File, receipt, codegen and licence assertions
only.

A real GPU round trip stays Tier-2. The existing promoter is Vulkan / `gfx1151`
shaped and does not cover GB10, so the Spark smoke is out of band by hand until
the promoter is extended.

### The scaffold, verified locally

Built and gated on a real `linux/aarch64` engine (Docker, arm64 native):

| Check | Result |
|---|---|
| Image build, `MAX_JOBS=2` | green; pin verified, patch applied, 8 patch markers asserted, extension compiled |
| Build-stage codegen guard | 131 native `sm_121a` cubins, 0 PTX |
| Build-stage dlopen guard | `exllamav3_ext` loads, `exl3_moe_cpu_has_avx2()` is `False` |
| `scripts/exllamav3-gb10-gate.sh` | PASS, end to end |
| Tampered-tree falsification | launcher **rejects** it (and the test asserts the tamper happened) |
| Memory-floor falsification | launcher **refuses** to start below the floor |
| Scaffold server | binds 5000, `GET /health` returns 200 |

`MAX_JOBS` matters more than it looks: 4 concurrent nvcc processes on an 8 GB
Docker VM thrashed badly enough to stall the build, where 2 finished it in about
9 minutes. The CI workflow caps it at `nproc` on the 4-vCPU arm runner.

Two fakes were caught by writing the checks honestly, and both are worth
remembering: the first tamper test ran as the image's non-root user, so the write
was silently denied and the launcher "accepted" an unmodified tree; and two
patch-marker strings asserted text the patch never produces. A guard that cannot
fail is worse than no guard, because it reads as coverage.

## Bumping

Edit the fork commit in `cuda-gb10-exllamav3-dsv41/Dockerfile`. It is a bare
commit SHA on purpose: `feat/gb10-ats-load` is an active branch whose HEAD moves,
and a branch clone would silently build different source over time. Re-read the
arch class and the ATS loader when the pin moves; both are why the pin is where
it is.

The base CUDA image is pinned by digest and the workflow's `BASE_DIGEST` is
checked against the Dockerfile, as the other GB10 images do.

## Scope

Server image only, and it bakes no weights: the pack and the EXL3 attention /
MTP overlay are staged by the InferenceService, and the re-lay is a node-side
step. arm64 only, and a hard constraint rather than a preference: GB10 is
aarch64 Grace, the extension is compiled for `sm_121`, and no other fleet
hardware could pull this image.

Throughput numbers, when they come, will be measured on the target Spark with
the recipe's protocol and reported with the client path and prompt shape stated.

## Credit

The architecture class, the GB10 ATS zero-copy loader, the re-lay utility and
the recipe are [Victor Cruz](https://github.com/vcruz305)'s work. ExLlamaV3 is
Turboderp's, MIT. This repo packages that build with its own server and gates
it.
