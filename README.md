# llmkube-runtimes

Inference runtime container images for [LLMKube](https://github.com/defilantech/LLMKube), built from source and gated on real hardware.

Today this repo builds the **AMD/Vulkan** llama.cpp runtime as two images from one build: a minimal **server** image (what the operator runs) and a **tools** image (`llama-bench` + `llama-cli`, for hardware benchmarking and diagnostics). The layout (`vulkan/`) is set up so other backends (CUDA, Intel, CPU) can be added as sibling directories later without restructuring.

It also builds a **Laguna variant** of that Vulkan server runtime (`vulkan-laguna/`), for serving poolside Laguna models with working tool calling. See [Laguna variant](#laguna-variant) below.

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

Bump the Laguna variant by editing `LLAMACPP_SHA` in `vulkan-laguna/Dockerfile`. It pins a bare commit SHA with no companion ref (the branch it tracks moves), and it is independent of the production pin.

## Tags

Both images use the same tag scheme:

- `:candidate-<gitsha>` — built + Tier-1 passed, not yet GPU-verified. Do not run in production.
- `:b<upstream-build>-llmkube<N>` — immutable, GPU-smoke-passed.
- `:stable` — moving, advanced by the promoter.

The operator pins an explicit immutable tag or digest of the server image, never `:stable`. The tools image is run by hand for benchmarking; pin a `:candidate-<gitsha>` for a reproducible benchmark.

## Contributing

Commits must be signed off ([DCO](https://developercertificate.org/)): `git commit -s`. Licensed under [Apache-2.0](LICENSE).
