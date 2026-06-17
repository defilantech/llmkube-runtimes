# llmkube-runtimes

Inference runtime container images for [LLMKube](https://github.com/defilantech/LLMKube), built from source and gated on real hardware.

Today this repo builds the **AMD/Vulkan** llama.cpp runtime as two images from one build: a minimal **server** image (what the operator runs) and a **tools** image (`llama-bench` + `llama-cli`, for hardware benchmarking and diagnostics). The layout (`vulkan/`) is set up so other backends (CUDA, Intel, CPU) can be added as sibling directories later without restructuring.

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

## Tags

Both images use the same tag scheme:

- `:candidate-<gitsha>` — built + Tier-1 passed, not yet GPU-verified. Do not run in production.
- `:b<upstream-build>-llmkube<N>` — immutable, GPU-smoke-passed.
- `:stable` — moving, advanced by the promoter.

The operator pins an explicit immutable tag or digest of the server image, never `:stable`. The tools image is run by hand for benchmarking; pin a `:candidate-<gitsha>` for a reproducible benchmark.

## Contributing

Commits must be signed off ([DCO](https://developercertificate.org/)): `git commit -s`. Licensed under [Apache-2.0](LICENSE).
