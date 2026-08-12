# llmkube-runtimes

Inference runtime container images for [LLMKube](https://github.com/defilantech/LLMKube), built from source and gated on real hardware.

Today this repo builds the **AMD/Vulkan** llama.cpp runtime as two images from one build: a minimal **server** image (what the operator runs) and a **tools** image (`llama-bench` + `llama-cli`, for hardware benchmarking and diagnostics). The layout (`vulkan/`) is set up so other backends (CUDA, Intel, CPU) can be added as sibling directories later without restructuring.

It also builds a **Laguna variant** of that Vulkan server runtime (`vulkan-laguna/`), for serving poolside Laguna models with working tool calling. See [Laguna variant](#laguna-variant) below.

It also builds the **NVIDIA GB10** llama.cpp runtime (`cuda-gb10/`), for DGX Spark. Upstream's CUDA image runs on GB10 but JIT-compiles every kernel there; this one ships native `sm_121`. See [GB10 (DGX Spark) CUDA runtime](#gb10-dgx-spark-cuda-runtime) below.

It also builds a **TurboQuant variant** of that GB10 runtime (`cuda-gb10-turbo/`), from the same fork the Laguna variant uses, for `TQ3_1S` / `TQ4_1S` weights on CUDA. See [GB10 TurboQuant variant](#gb10-turboquant-variant) below.

It also builds the **Foreman coder-agent** toolchain image (`coder/`): the foreman-agent binary plus the Go toolchain its in-workspace self-gate needs. Same discipline as the runtime images: pinned inputs, built here, owned. See [Coder agent image](#coder-agent-image) below.

## Why this repo exists

LLMKube previously inherited its entire serving runtime from upstream floating image tags. That made the load-bearing part of the product an uncontrolled supply chain: when upstream's `:server-vulkan` tag shipped a `libggml-vulkan.so` with an undefined shader symbol, the backend silently failed to load and fell back to CPU, and we could neither fix nor detect it without a hand-run on a GPU (see [defilantech/LLMKube#725](https://github.com/defilantech/LLMKube/issues/725)).

Building from source here means we own the Vulkan shader-gen step, the base image, and dependency/CVE patching, and we gate every build on hardware before anything trusts it.

Design reference: [`docs/proposals/697-amd-vulkan-runtime-image.md`](https://github.com/defilantech/LLMKube/blob/main/docs/proposals/697-amd-vulkan-runtime-image.md) in the LLMKube repo.

## Images

Both images come from the same `vulkan/Dockerfile` build stage, so they carry the identical llama.cpp commit and Vulkan backends.

`ghcr.io/defilantech/llmkube-llama-vulkan` — the server runtime.

- Ubuntu 26.04 base (Mesa new enough for `gfx1151` / Strix Halo RADV), pinned by digest.
- `cmake -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON` with `GGML_NATIVE=OFF` (a single generic x86-64 CPU backend, not `GGML_CPU_ALL_VARIANTS`), llama.cpp pinned by tag + commit SHA.
- Runs the OpenAI-compatible `llama-server`. No ROCm.

`ghcr.io/defilantech/llmkube-llama-vulkan-tools` — benchmarking + diagnostics.

- Same backends and commit as the server image, plus `llama-bench` and `llama-cli` (it also carries `llama-server`). Default entrypoint is `llama-bench`.
- Run off-cluster to benchmark hardware (e.g. Strix Halo `gfx1151`) with numbers directly comparable to the server runtime. The operator never consumes this image.

Either pod consumes the GPU by mounting `/dev/dri` device nodes (both `renderD128` and `card1`) via a generic device-plugin resource; it requests no `nvidia.com/gpu`. Non-root: the deployment grants the host render group via `securityContext.supplementalGroups`.

## The two-tier gate

A built image is a **candidate**. Only an image a real GPU host has verified and signed is promoted to a tag the operator consumes.

1. **Tier 1, in CI (this repo, free runners, no GPU).** Build, then run `llama-server --list-devices` under the image's software Vulkan (lavapipe). The Vulkan backend must dlopen and register; a #725-class undefined-symbol break fails here before the image ever leaves CI. On pass, push `:candidate-<sha>` with an SBOM and build provenance.
2. **Tier 2, out-of-band on a self-hosted `gfx1151` host.** A promoter verifies the candidate's build provenance, runs a sandboxed offline GPU smoke (real device + layer offload + a throughput floor), then promotes to `:stable` / `:b<upstream>-llmkube<N>` and applies a smoke-passed signature. The host is never a CI runner, so fork-PR code never touches it.

Tier 2 (the promoter) lands in a follow-up; this bootstrap is Tier 1.

## Laguna variant

`ghcr.io/defilantech/llmkube-llama-vulkan-laguna` is the Vulkan server runtime, built from a fork, for serving [poolside](https://huggingface.co/poolside) Laguna models with working tool calling.

Upstream llama.cpp can load Laguna (the arch merged 2026-07-22, first released in `b10103`) but cannot parse its tool calls: Laguna emits poolside's native format and upstream's `common/chat.cpp` has no handler for it, so a tool call arrives as raw text in `message.content` with `finish_reason: stop` and an OpenAI-API client sees no `tool_calls` at all. That breaks any agent loop, and it breaks quietly, looking like a model quality problem.

This variant builds from [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) (branch `laguna/port`, pinned by commit SHA), whose **differential autoparser** derives a tool-call parser from the model's chat template instead of hand-writing one per model. It also ships that fork's corrected chat templates and defaults to one, because poolside's released GGUF embeds a template that silently cannot parse tool calls. Full detail, including how to diagnose a template problem: [`docs/laguna-variant.md`](docs/laguna-variant.md).

The production `vulkan/` image stays pure upstream and is unaffected. The two pins move independently. Server image only; benchmark numbers come from the pure-upstream tools image so they stay comparable across models.

## GB10 (DGX Spark) CUDA runtime

`ghcr.io/defilantech/llmkube-llama-cuda-gb10` is the llama.cpp server runtime for NVIDIA GB10, built with native `sm_121` device code. `-tools` is the matching `llama-bench` / `llama-cli` image from the same build stage.

A DGX Spark can pull upstream's `ghcr.io/ggml-org/llama.cpp:server-cuda-*` today: it has an arm64 manifest and it serves. It is also, on GB10, entirely JIT-compiled. Inspecting the shipped `libggml-cuda.so` from `server-cuda-b10068` with `cuobjdump` (2026-08-10):

| Kind | Architectures |
| --- | --- |
| Native cubins | `sm_86`, `sm_89`, `sm_120a` |
| PTX | `sm_50`, `sm_61`, `sm_70`, `sm_75`, `sm_80`, `sm_90` |

GB10 is `sm_121`, and the `a` in `sm_120a` means architecture-**specific**: a `120a` cubin does not load on `121`. So there is no native code for this GPU, and every kernel is JIT-compiled at startup from the `sm_90` (Hopper) PTX.

**What that actually costs, measured on a GB10 (2026-08-11).** A controlled run isolated codegen from every other variable: one pod, one CUDA 13 toolchain, one llama.cpp commit (`b10355`), one GPU, one 16.8 GiB model, swapping only `libggml-cuda.so` between `121a-real` and `90-virtual`, with the CUDA JIT cache cleared before each run.

| | native `sm_121a` | `sm_90` PTX (JIT) |
| --- | --- | --- |
| time to `model loaded` | 2.54 / 2.54 / 2.33 s | 20.14 / 21.21 / 20.12 s |
| first-request prefill | 785 tok/s | 239 tok/s |
| steady-state prefill | 787 tok/s | 782 tok/s |
| steady-state decode | 31.4 tok/s | 31.4 tok/s |

JIT costs about **18 s on every process start**, plus roughly 14 s more on the first request (CUDA loads modules lazily, so the first kernel launch pays again). It costs essentially **nothing in steady state**: +0.6% prefill and -0.1% decode against upstream's own image at a near-identical llama.cpp pin, both inside run-to-run noise. JIT'd SASS runs at full speed once compiled; what you pay for is compiling it, every time a pod starts.

This is upstream's toolkit version, not an oversight. llama.cpp only appends `121a-real` to its default architecture list at CUDA >= 12.9 ([`ggml/src/ggml-cuda/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt)), and the upstream image is built on CUDA 12.8.x, so that branch never fires. Building on CUDA 13 with an explicit `-DCMAKE_CUDA_ARCHITECTURES=121a-real` fixes it.

**Scope is one GPU, deliberately.** No virtual/PTX target is compiled in, so the image cannot silently JIT itself onto other hardware; it fails loudly instead, which is the whole point. The rest of the fleet is unaffected: `sm_120` consumer Blackwell is already covered natively by upstream's `sm_120a` cubin, and is x86 anyway, so it could never pull this arm64 image.

**The guard is the interesting part.** A wrong toolkit, a dropped `CMAKE_CUDA_ARCHITECTURES`, or an upstream change to the default list all produce a build that succeeds, an image that runs, and a GB10 quietly back on JIT. None of that is visible without looking at the fatbin, so the build stage looks at the fatbin: it fails unless `cuobjdump` finds native `sm_121`, and fails again if any PTX is present. Both checks are GPU-independent, so they run in CI. The shipped inventory is written to `/app/CUDA_ARCHS` as a receipt, because `cuobjdump` exists only in the CUDA devel image and anyone holding the final runtime image needs a way to verify what it contains.

Built on GitHub's arm64 hosted runners (GB10 is aarch64 Grace, and a QEMU cross-build would turn a ~30 minute CUDA compile into an overnight job). It pins the same llama.cpp ref as `vulkan/`, so GB10 and Strix Halo numbers compare like vs like.

**Multi-Spark serving.** The image is built with `GGML_RPC=ON` and ships `ggml-rpc-server`, so one model can span two Sparks over the ConnectX-7 link: the remote host exposes its devices, the main process offloads layers onto them, and the model is bounded by the sum of both machines' memory rather than either one's. NVIDIA builds their Spark llama.cpp the same way. Enabling it changes nothing for single-node serving, which is why it is on by default here. Note upstream considers the RPC backend a proof of concept and insecure on untrusted networks: keep `ggml-rpc-server` on the point-to-point fabric, never on a routable interface.

## GB10 TurboQuant variant

`ghcr.io/defilantech/llmkube-llama-cuda-gb10-turbo` (`cuda-gb10-turbo/`) is the same GB10 runtime built from [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) branch `laguna/port` instead of upstream, for TurboQuant weight support on CUDA.

| Format | Bytes per 32 weights | bits per weight |
| --- | --- | --- |
| `TQ3_1S` | 16 | 4.0 |
| `TQ4_1S` | 20 | 5.0 |
| `Q4_K_M` (for comparison) | ~19.4 | ~4.85 |

Note what that table does and does not say. TurboQuant is a **quality-per-bit** format, not a compression one: `TQ4_1S` is slightly *larger* than `Q4_K_M`, and `TQ3_1S` is about 18% smaller. It buys fidelity at a given size, not the ability to fit a much bigger model.

**Why a separate image rather than a swap.** The two exist to be compared: same GPU, same CUDA toolkit, same codegen, different llama.cpp. Their pins therefore move independently, unlike `cuda-gb10/` and `vulkan/` which are deliberately kept in lockstep. Keeping the upstream image untouched also means a fork regression cannot take the fleet's serving runtime with it.

**The TurboQuant guard is the interesting part here**, and it exists because of how the Vulkan side of the same fork failed: `TQ4_1S` shader sources were present in the tree, looked complete on inspection, and were never compiled or registered. Reading source proves nothing, so the build interrogates the artifact instead. It fails unless `cuobjdump` finds TurboQuant kernels in the compiled device code, and fails separately if the TQ type names are missing from the ggml libraries (kernels without registered types means a TQ GGUF is rejected at load). The guard also distinguishes "the dump did not work" from "the dump worked and TQ is absent", so a tooling problem cannot masquerade as a code problem.

It carries the same `GGML_RPC=ON` build and `ggml-rpc-server` as `cuda-gb10/`, so the two variants differ only in llama.cpp source, which is what makes them comparable.

`laguna/port` is an actively developed branch rather than a tag, so the build fetches the pinned commit directly and verifies it. The weekly canary runs on a different day from `cuda-gb10`'s so the two never contend for the shared Actions cache, and it doubles as a watch on the fork: a rebase there that drops the TQ CUDA path turns into a build failure rather than a silently stock image.

## Coder agent image

`ghcr.io/defilantech/llmkube-foreman-agent-coder` — a Foreman agent that can run its own coder gate.

The published `foreman-agent` image (in the LLMKube repo) is intentionally minimal: git plus the binary, no language toolchain. That is right for the reviewer and scheduler roles, but a **coder** agent runs an in-workspace self-gate against the target repo (gofmt, go vet, go build, golangci-lint, changed-package `go test`, codegen-drift), which needs that repo's language toolchain inside the agent container. Baking a Go toolchain into the published agent image would not generalise (a Python or Rust target repo needs a different toolchain), so the coder capability is a per-language BYO toolchain image, parallel to `AgenticTask.gateProfile.image`. This is LLMKube's own (Go) one. Design context: [defilantech/LLMKube#835](https://github.com/defilantech/LLMKube/issues/835).

It is the `golang:1.26` base (go + git) plus `make`, `helm`, `golangci-lint`, and `controller-gen`, pinned to the same versions the LLMKube Makefile uses, with the `foreman-agent` binary resolved from a pinned `github.com/defilantech/llmkube` module ref via `go install` (verified against the Go checksum database). Nothing is compiled under emulation: `foreman-agent` and `controller-gen` are cross-compiled in the `$BUILDPLATFORM` stage and `helm` + `golangci-lint` are prebuilt per-arch downloads, so the **arm64** image (Apple Silicon, arm edge) builds as fast as **amd64**.

The image is hardened for the chart's agent Deployment, which runs it non-root with a read-only root filesystem: the binary is at `/foreman-agent` (the Deployment's `command`) with a PATH symlink for the gate Job's bare-name invocation, the user is the numeric non-root `65532` (satisfies `runAsNonRoot: true`), and `HOME` / `GOPATH` / `GOCACHE` point at the writable `/tmp` mount. To use it, point a coder-role agent at this image in the `charts/foreman` values (set `agent.image` and add `"coder"` to `agent.roles`).

**Single-tier gate.** Unlike the GPU runtimes, the coder image needs no hardware: the CI smoke (`scripts/coder-smoke.sh`) runs the image as its non-root user and execs every gate tool, which is the full gate. There is no out-of-band GPU promoter, so consume an immutable `:candidate-<gitsha>` directly. The amd64 leg is CI-smoked; the arm64 leg is build-verified only (no arm CI runner), so smoke it by hand on Apple Silicon before trusting it there.

```bash
docker build -t llmkube-foreman-agent-coder:dev coder/
./scripts/coder-smoke.sh llmkube-foreman-agent-coder:dev
```

Bump the pinned foreman-agent by editing `LLMKUBE_REF` in `coder/Dockerfile` (a release tag or commit SHA); the build derives the binary's reported version from that ref (the tag, or the module proxy's pseudo-version for a commit) and stamps it via `-ldflags`, so the FleetNode reports a real version rather than `dev`. Bump the toolchain pins (`HELM_VERSION`, `GOLANGCI_LINT_VERSION`, `CONTROLLER_TOOLS_VERSION`) in lockstep with the LLMKube Makefile.

## Build locally

```bash
# server (default final stage)
docker build -t llmkube-llama-vulkan:dev vulkan/
./scripts/tier1-gate.sh llmkube-llama-vulkan:dev

# tools (llama-bench + llama-cli)
docker build --target tools -t llmkube-llama-vulkan-tools:dev vulkan/
./scripts/tier1-gate.sh llmkube-llama-vulkan-tools:dev
```

Bump the pinned llama.cpp ref by editing `LLAMACPP_REF` + `LLAMACPP_SHA` in `vulkan/Dockerfile` (the SHA check fails the build if they disagree); both images move together.

```bash
# Laguna variant (server only)
docker build -t llmkube-llama-vulkan-laguna:dev vulkan-laguna/
./scripts/tier1-gate.sh llmkube-llama-vulkan-laguna:dev
./scripts/laguna-gate.sh llmkube-llama-vulkan-laguna:dev
```

```bash
# GB10 / DGX Spark CUDA (arm64 only; build on an arm64 host)
docker build -t llmkube-llama-cuda-gb10:dev cuda-gb10/
./scripts/cuda-gate.sh llmkube-llama-cuda-gb10:dev

docker build --target tools -t llmkube-llama-cuda-gb10-tools:dev cuda-gb10/
./scripts/cuda-gate.sh llmkube-llama-cuda-gb10-tools:dev

# GB10 TurboQuant variant (same gate; adds the TQ guards at build time)
docker build -t llmkube-llama-cuda-gb10-turbo:dev cuda-gb10-turbo/
./scripts/cuda-gate.sh llmkube-llama-cuda-gb10-turbo:dev
```

`cuda-gate.sh` rather than `tier1-gate.sh`: a GPU-less host legitimately produces a CUDA backend load error (no device for `cuInit`), which `tier1-gate.sh` treats as failure. The CUDA gate allowlists exactly that one case, and additionally asserts the shipped image carries native `sm_121`.

Bump the Laguna variant by editing `LLAMACPP_SHA` in `vulkan-laguna/Dockerfile`. It pins a bare commit SHA with no companion ref (the branch it tracks moves), and it is independent of the production pin.

## Tags

Both images use the same tag scheme:

- `:candidate-<gitsha>` — built + Tier-1 passed, not yet GPU-verified. Do not run in production.
- `:b<upstream-build>-llmkube<N>` — immutable, GPU-smoke-passed.
- `:stable` — moving, advanced by the promoter.

The operator pins an explicit immutable tag or digest of the server image, never `:stable`. The tools image is run by hand for benchmarking; pin a `:candidate-<gitsha>` for a reproducible benchmark.

## Contributing

Commits must be signed off ([DCO](https://developercertificate.org/)): `git commit -s`. Licensed under [Apache-2.0](LICENSE).
