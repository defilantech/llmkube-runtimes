# llmkube-runtimes

Inference runtime container images for [LLMKube](https://github.com/defilantech/LLMKube), built from source and gated on real hardware.

Today this repo builds one image: the **AMD/Vulkan** llama.cpp server. The layout (`vulkan/`) is set up so other backends (CUDA, Intel, CPU) can be added as sibling directories later without restructuring.

## Why this repo exists

LLMKube previously inherited its entire serving runtime from upstream floating image tags. That made the load-bearing part of the product an uncontrolled supply chain: when upstream's `:server-vulkan` tag shipped a `libggml-vulkan.so` with an undefined shader symbol, the backend silently failed to load and fell back to CPU, and we could neither fix nor detect it without a hand-run on a GPU (see [defilantech/LLMKube#725](https://github.com/defilantech/LLMKube/issues/725)).

Building from source here means we own the Vulkan shader-gen step, the base image, and dependency/CVE patching, and we gate every build on hardware before anything trusts it.

Design reference: [`docs/proposals/697-amd-vulkan-runtime-image.md`](https://github.com/defilantech/LLMKube/blob/main/docs/proposals/697-amd-vulkan-runtime-image.md) in the LLMKube repo.

## Image

`ghcr.io/defilantech/llmkube-llama-vulkan`

- Ubuntu 26.04 base (Mesa new enough for `gfx1151` / Strix Halo RADV), pinned by digest.
- `cmake -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`, llama.cpp pinned by tag + commit SHA.
- Runs the OpenAI-compatible `llama-server`. No ROCm.

The pod consumes the GPU by mounting `/dev/dri` device nodes (both `renderD128` and `card1`) via a generic device-plugin resource; it requests no `nvidia.com/gpu`. Non-root: the deployment grants the host render group via `securityContext.supplementalGroups`.

## The two-tier gate

A built image is a **candidate**. Only an image a real GPU host has verified and signed is promoted to a tag the operator consumes.

1. **Tier 1, in CI (this repo, free runners, no GPU).** Build, then run `llama-server --list-devices` under the image's software Vulkan (lavapipe). The Vulkan backend must dlopen and register; a #725-class undefined-symbol break fails here before the image ever leaves CI. On pass, push `:candidate-<sha>` with an SBOM and build provenance.
2. **Tier 2, out-of-band on a self-hosted `gfx1151` host.** A promoter verifies the candidate's build provenance, runs a sandboxed offline GPU smoke (real device + layer offload + a throughput floor), then promotes to `:stable` / `:b<upstream>-llmkube<N>` and applies a smoke-passed signature. The host is never a CI runner, so fork-PR code never touches it.

Tier 2 (the promoter) lands in a follow-up; this bootstrap is Tier 1.

## Build locally

```bash
docker build -t llmkube-llama-vulkan:dev vulkan/
./scripts/tier1-gate.sh llmkube-llama-vulkan:dev
```

Bump the pinned llama.cpp ref by editing `LLAMACPP_REF` + `LLAMACPP_SHA` in `vulkan/Dockerfile` (the SHA check fails the build if they disagree).

## Tags

- `:candidate-<gitsha>` — built + Tier-1 passed, not yet GPU-verified. Do not run in production.
- `:b<upstream-build>-llmkube<N>` — immutable, GPU-smoke-passed.
- `:stable` — moving, advanced by the promoter.

The operator pins an explicit immutable tag or digest, never `:stable`.

## Contributing

Commits must be signed off ([DCO](https://developercertificate.org/)): `git commit -s`. Licensed under [Apache-2.0](LICENSE).
