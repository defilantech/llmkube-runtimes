# Qwen3.8-Flash-Next (qwen4exp) with MTP on a Strix Halo (`vulkan-qwen4exp/`)

`ghcr.io/defilantech/llmkube-llama-vulkan-qwen4exp`

An AMD/Vulkan runtime for Qwen3.8-Flash-Next with MTP speculative decoding, for
gfx1151 (Strix Halo) unified memory. This is the only variant image here that
carries a vendored patch rather than a fork or an upstream tag alone, so the
first half of this document is about the patch and the second half about the
settings that make the model actually serve.

## Why the variant exists, and how little of it is ours

Two things about this model were non-upstream when the experiment started. Only
one still is.

| | where it lives | since |
|---|---|---|
| the `qwen4exp` architecture | upstream, PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) | merged 2026-08-27, commit `6c84c7d5` |
| the QSA correctness follow-ups | upstream, PR [#27941](https://github.com/ggml-org/llama.cpp/pull/27941) | merged 2026-09-01, commit `36b10154` |
| the qwen4exp MTP draft graph | **our patch**, PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) | open |

So this image is a pinned upstream tag plus one patch, not a fork. The upstream
half matters more than it looks: the QSA defects #27941 fixes are exactly the
kind that appear under long context, which is the whole use case here, and they
are in the base rather than in anything we carry.

The MTP machinery is already upstream too (`--spec-type draft-mtp`, the
draft-mtp implementation, `llama_set_embeddings_nextn`). What #28243 adds is the
qwen4exp-specific draft graph, its `graph_mtp` builder, and the converter flags
that export a head with shared embeddings.

## The pin, and the one rule

```
LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git
LLAMACPP_REF=b10794
LLAMACPP_SHA=f9f09f02cc44d87d842dbd2d578857d92d4bb63b
```

**The tag is the commit #28243 was authored against, and that is the only reason
it is `b10794` rather than the newest release.** The patch does not apply to a
later tag: `src/models/qwen4exp.cpp` has moved since. Verified both ways, in a
clean clone of `b10794` and inside the build stage, with `git apply --check`.

Bump the tag and the patch together or neither. When #28243 merges, this variant
deletes itself: `patches/` goes, and `vulkan/Dockerfile` moves to a tag at or
after the merge.

## Use the self-contained draft head

Unsloth ship the heads in three families. Use:

```
MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf     3.85 GiB   self-contained   <- this one
MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf  2.60 GiB   shared
```

The `shared-` heads carry their own embeddings and borrow everything else from
the running target. That borrow path (`borrow_shared_tensor`) is loader-side
code that #28243 does not carry, and it is absent from the upstream base. A
`shared-` head on this image does not load. The `Q8_0` head costs 1.25 GiB more
and accepts slightly fewer drafts; it is the only one that works here.

## The two settings that are load-bearing

**`env: LLAMA_ATTN_ROT_DISABLE=1`.** Upstream's quantized-KV activation rotation
(#21038) is not supported by the qwen4exp attention path. Without this variable
the server aborts at load, or serves corrupted output. Every source that serves
this model on gfx1151 sets it, and we hit the same wall independently: our own
MinIO preload note records `GGML_ASSERT(inp->self_k_rot == nullptr ...)` for this
architecture. Set it through `spec.env`; the operator passes it straight to the
container.

**KV cache in `f16`.** Quantized KV is what triggers the assert above. `f16` is
the value that has actually been run against this model on our hardware, so the
first boot uses it. Quantized KV is a measurement worth taking *after* the
endpoint is proven, with the env var set.

## Recommended InferenceService shape

```yaml
runtime: llamacpp
image: ghcr.io/defilantech/llmkube-llama-vulkan-qwen4exp:candidate-<sha>
modelRef: qwen38-flash-next-strix
nodeSelector: {accelerator: amd}
podSecurityContext: {supplementalGroups: [991]}     # shadowstrix render gid
env:
  - {name: LLAMA_ATTN_ROT_DISABLE, value: "1"}
contextSize: 65536
parallelSlots: 1
flashAttention: true
jinja: true
cacheTypeK: f16
cacheTypeV: f16
speculativeDecoding:
  type: draft-mtp                                   # literal; `mtp` is rejected with draftModel set
  draftModel: MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
  nDraftMax: 2
resources:
  cpu: "8"
  gpu: 1
  memory: 48Gi
  memoryLimit: 105Gi
extraArgs: ["--alias", "qwen38-flash-next-strix", "--no-mmap", "-fit", "off"]
```

Four of those deserve a note.

**`draftModel` rather than a hand-written `-md`.** The operator resolves the
staged companion path against the directory holding the primary file and appends
`-md` itself. Writing the path by hand means writing a content-hash cache
directory that is unknowable before the first reconcile.

**`--no-mmap`.** With mmap, loading a ~90 GiB GGUF fills page cache and the pod
is OOM-killed mid-load. The same lesson cost us a boot on GB10: 22 s to load
with `--no-mmap`, 224 s with it.

**`-fit off`.** The shared draft head defeats llama.cpp's automatic memory
fitting, which measures the draft on its own before the target exists. Left on,
the server may pick a context that does not fit.

**`memoryLimit`.** `resources.memory` is both the request and the limit on this
operator version, so a 48Gi memory alone caps the cgroup at 48Gi for a workload
that grows past 90 GiB. The explicit limit is what lets the request stay small
(the scheduler reserves less) while the pod is still allowed to reach its
working set.

Everything else is the operator's own field: no `-ngl`, no `--ctx-size`, no
`--parallel`, no `--cache-type-*`, no `--spec-type`, no `-md` in `extraArgs`.

## Memory math

| item | GiB |
|---|---|
| UD-Q3_K_XL weights (3 shards) | 83.8 |
| self-contained MTP head `Q8_0` | 3.85 |
| KV at `f16`, 65536 tokens | ~1.6 |
| Vulkan compute buffers at 64K | 4 to 8 |
| serving total | ~93 to 97 |
| kubelite + dqlite + OS | ~4 |

Node allocatable on shadowstrix is 113.2 GiB. With nine Foreman agent pods at
1Gi requests each, the serving request plus those fits; the node-level OOM
killer is the only real limit, which is why the quant ladder is
`UD-Q3_K_XL` (83.8, on disk) then `UD-IQ3_XXS` (76.3). The other tenant on that
box, `nemotron-49b-strix`, is scaled to 0 for the duration.

## Guards

The Dockerfile asserts, and the CI gate re-checks against the final image:

- `qwen4exp` (exact line) and `ple_conv1d` present in the shipped libraries, so
  the arch registered rather than us having built stock llama.cpp;
- `"QWEN4EXP MTP:"`, a throw message #28243 adds to `src/models/qwen4exp.cpp`.
  **This is the guard that matters**, because its failure is silent: without the
  draft graph the server loads the model, generates correctly, and runs at the
  target's own speed with no error and no log line. A `--spec-type` check would
  not catch it, since that flag and the draft-mtp implementation are upstream
  for other architectures already;
- `--spec-type` and `--spec-draft-n-max` listed by `llama-server --help`, so the
  flags the InferenceService renders cannot abort the server at startup;
- an `RTLD_NOW` dlopen of `libggml-vulkan.so` (the #725 class).

## Measured

Not yet measured on shadowstrix. The throughput A/B (512 / 8192 / 32768 prompt
tokens, MTP on against MTP off, three runs each) and the soak results are
recorded in `llmkube-internal/research/qwen38fn-strix-reviewer/`, not here.

Expectations to compare against, both from other people's Strix numbers rather
than ours: mainline llama.cpp at IQ4_XS and 32K context runs about 71 t/s prefill
and 12 t/s decode. MTP on gfx1151 Vulkan is worth **18 to 26 percent** on decode
in community measurements, and one report has it *three times slower* at
untuned defaults, so `nDraftMax 2` is a starting point to be measured rather than
a setting to be trusted. The 1.3x to 1.7x figures that circulate for this head
are a B200 measurement and do not transfer.
