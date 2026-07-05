# turbo4 KV on gfx1151: runaway generation + SIGSEGV under sustained load

**Status:** fixed. Repro captured 2026-07-03; root cause confirmed and fixed
2026-07-04 (two Vulkan-only bugs: block-layout mismatch + stale centroid
tables, see "Root cause (confirmed)" below); soak-verified under the exact
crashing config (see the dated soak section below). Filed upstream on
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant):
issue [#203](https://github.com/TheTom/llama-cpp-turboquant/issues/203), PR
[#204](https://github.com/TheTom/llama-cpp-turboquant/pull/204).

## TL;DR

Serving GLM-4.7-REAP-218B with `--cache-type-k/v turbo4` at large context on
AMD Strix Halo (gfx1151, Vulkan/RADV) is unstable under a sustained multi-turn
agent workload. Two failure symptoms, reproduced across two independent ~1-2h
coder runs:

1. **Runaway generation** — deep into the session (as the prompt/KV context
   grows) a single completion stops honoring the stop token and decodes
   12-16k+ tokens monotonically with no EOS.
2. **SIGSEGV** — `llama-server` then crashes (container exit code **139**),
   dropping the SSE stream (`unexpected EOF` at the client).

The **same image + same model + same node** on `--cache-type-k/v f16 @ ctx
32768` is stable: clean `finish=stop`, short turns, 20+ turns cycled, 0
restarts. So this is the **turbo4 KV path**, not the model, the harness, or the
GPU/driver.

## Environment

- **Fork:** `github.com/TheTom/llama-cpp-turboquant`, branch
  `feature/turboquant-kv-cache`, `TURBO_REF=558c6b78e4f8cf92ec19539ff89b6d13f4183feb`
  (fused TurboQuant K/V dequant in the Vulkan flash-attention shaders, PR #184,
  + the merged turbo4 KV cache type).
- **Image:** `ghcr.io/defilantech/llmkube-llama-vulkan-turboquant:experiment-b07685b5f851a683cd88ddd3d836c6e8e99053ca`
  (built from `vulkan-turboquant/Dockerfile`: `GGML_VULKAN=ON`,
  `GGML_BACKEND_DL=ON`, ubuntu:26.04 / Mesa RADV).
- **Hardware:** AMD Ryzen AI Max (Strix Halo), **gfx1151**, 128GB unified
  memory (BIOS-split ~64GB system / ~64GB GPU; Vulkan reports ~112GB
  GTT-addressable). Vulkan backend, no ROCm/HIP.
- **Model:** GLM-4.7-REAP-218B-A32B (`glm4_moe`, 218B/32B-active MoE,
  unsloth UD-Q3_K_XL, ~98GB, 2-part split GGUF).

## Repro (crashing config)

```
llama-server \
  --flash-attn on \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --spec-type none \
  --ubatch-size 2048 --parallel 1 \
  --ctx-size 131072 \
  --no-mmap --jinja
# (--reasoning off vs template-default made NO difference; both crash)
```

Drive it with a sustained multi-turn agent loop (OpenAI `/v1/chat/completions`,
streaming) whose prompt grows each turn (tool outputs appended). A short
one-shot prompt (`max_tokens: 16`) returns clean `content` on turbo4 — the
failure is **load- and context-dependent**, emerging only as the prompt/KV
grows over many turns.

Observed:
- Run 1 (reasoning on): runaway turns of 12-16k tokens, **SIGSEGV ~76 min in**.
- Run 2 (reasoning off): identical runaway turns in `content`, **SIGSEGV ~114
  min in** (restartCount reached 2).

Server log signature at the runaway: a single `slot ... task N` with
`n_decoded` climbing monotonically past 12-16k, no `stop`/EOS, no new
`launch_slot_`, until the process dies (139).

## Control (stable config, same image/model/node)

```
llama-server \
  --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type none \
  --ubatch-size 2048 --parallel 1 \
  --ctx-size 32768 \
  --no-mmap --jinja
```

Result: PONG test returns `finish=stop`; the coder loop cycled **44 clean short
turns** over 81 min with **no runaway and no SIGSEGV** — confirming the model
emits EOS correctly, the harness is fine, and the runaway generation is specific
to the turbo4 KV path.

**Caveat (for accuracy):** the f16 control was itself cut short (run 3: turn 44 /
81 min; run 4 at 48Gi: turn 35 / 76 min) by a DIFFERENT and non-fork cause —
container **exit 137 `OOMKilled`** from **APU unified-memory exhaustion**, NOT a
segfault and NOT the fork. Live `kubectl top` proved it: the container's CPU RSS
stayed FLAT at ~11GB the whole run, never near any pod limit. The ~98GB model
lives in GPU/GTT device memory (`--no-mmap`), which cgroup metrics don't count;
as the coder context grows over turns, the attention/compute buffers for the
longer context grow in the same 128GB unified pool until model + KV + buffers
cross 128GB and the kernel OOM-kills llama-server. Raising the pod memory
request (32Gi→48Gi) did nothing (RSS never approached it). So f16 is a clean
control for *runaway + SIGSEGV* (neither ever occurred — 44/35 clean turns) and
the OOM is a hardware capacity wall specific to running a ~98GB model + long
context on 128GB unified, entirely independent of the turbo4 fork. The turbo4
SIGSEGV (139, runaway-driven) is the fork bug this doc reports; the f16 OOM (137)
is unified-memory capacity.

## Root cause (confirmed)

Two independent Vulkan-only bugs, both stemming from 77ab7e98's turbo4 rework
never reaching the Vulkan shaders:

1. **Block layout: 68 bytes (Vulkan) vs 66 bytes (C).** 77ab7e98 shrank
   `block_turbo4_0` to 66 bytes (`norm` + `qs[64]`, dropping a dead `rnorm`
   field) in `ggml-common.h` and updated the C, CUDA, and Metal sides;
   b01afefe later mopped up two Metal call sites 77ab7e98's own edit had
   missed. The Vulkan shaders (`types.glsl`, `copy_to_quant.comp`) were never
   touched by either commit and kept the legacy 68-byte struct and the write
   to the removed `rnorm` field. Every Vulkan shader indexing
   `block_turbo4_0` therefore strode the KV cache 2 bytes long per block
   while the C side allocated and byte-copied at 66, corrupting turbo4 KV
   data wherever a C-layout accessor (uploads, state save/restore, defrag
   copies) met shader-side addressing, and running out of bounds near the
   top of the buffer at large `n_kv`. Fixed in `03ff84818` (drop the `rnorm`
   field from the GLSL struct and the write to it), mirroring the Metal fix
   in b01afefe.
2. **Centroid tables: stale pre-77ab7e98 values.** The same 77ab7e98 commit
   re-derived the turbo4 Lloyd-Max centroids (a KLD/PPL fix) for C, CUDA, and
   Metal, but the Vulkan shaders kept the old table (+-0.173926 vs the new
   +-0.241529, a 1.389x ratio). After the layout fix alone, every turbo4
   `FLASH_ATTN_EXT` case on Vulkan still failed with a uniform ~40% relative
   error, matching that ratio. Fixed in `175a652dd` (synced the
   `FA_DEQUANT4_TURBO4_0` table in `flash_attn_dequant.glsl` and the
   `TC4`/`TM4` tables in `copy_to_quant.comp` to the C reference
   `CENTROIDS_4BIT` / `nearest_centroid_4bit` in `ggml-turbo-quant.c`).

Test evidence (`test-backend-ops`, gfx1151 / Radeon 8060S Graphics, RADV
STRIX_HALO): 0/528 turbo4 `FLASH_ATTN_EXT` cases pass before either fix (444
`inf mismatch`, 84 `ERR ~= 1.0`); 0/528 pass with the layout fix alone (all
528 now a uniform ~40% relative error); 528/528 pass with both fixes (full
sweep 9509/9509). turbo3 control unaffected throughout (528/528 `OK`). Also
added: turbo4 coverage in the `SET_ROWS` round-trip and `FLASH_ATTN_EXT`
sweep in `test-backend-ops.cpp` (`bc5c9c891`) — turbo3 already had both,
turbo4 had neither. `SET_ROWS_TURBO4` reports "not supported" on Vulkan in
every state (Vulkan has no `CPY` turbo4-to-f32 conversion; the new test still
covers CPU/Metal/CUDA), symmetric with turbo3 and unrelated to either bug.

Adjacent, not fixed: Vulkan's turbo3 centroid tables have also drifted
slightly from the C reference (-0.190685 vs -0.190207), within
`test-backend-ops` tolerance (turbo3 passes 528/528 throughout); left
untouched, offered as a follow-up in the upstream issue.

Both bugs are gfx1151/RADV-specific only as far as tested (no other Vulkan
GPU exercised); neither is gfx1151-specific in principle, since both are
struct-layout and constant-table mismatches hit by every Vulkan shader
invocation regardless of device.

## What would help isolate it

- A `-fit off` / verbose (`GGML_VK_FA_LOG=1` is already set) capture right at
  the runaway → crash boundary.
- Whether turbo3 (vs turbo4) shows the same runaway at the same context depth
  (isolates turbo4-specific dequant vs the general TurboQuant FA path).
- Whether a smaller `--ctx-size` on turbo4 (e.g. 32768) still crashes given
  enough turns (isolates "large n_ctx allocation" from "sustained load").
- Perplexity on a long context with turbo4 vs f16 (quantifies the coherence
  degradation in hypothesis 1).

## Upstream issue and PR

Filed on [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant):

- Issue [#203](https://github.com/TheTom/llama-cpp-turboquant/issues/203) —
  full writeup: symptom, repro/control, root cause with file/line
  references, test evidence, soak evidence, turbo3 drift note.
- PR [#204](https://github.com/TheTom/llama-cpp-turboquant/pull/204) — the
  two fix commits (`03ff84818`, `175a652dd`) plus the test commit
  (`bc5c9c891`), targeting `feature/turboquant-kv-cache`. Fixes #203.

## Related

- LLMKube memory: `reference_strix_large_model_serving` (stability note),
  `reference_turbo3_prefill_penalty` (the earlier gfx1151 turbo-KV prefill
  regression that #184 fixed — this crash is a *separate*, later-surfacing bug).
- The turbo4 image remains fine for **short-context / benchmarking** use; this
  bug only bites sustained long-context agent loops. f16/32K is the stable
  serving fallback for coder workloads until turbo4 is fixed.

## 2026-07-04 soak repro with the block-layout + centroid-table fix: runaway/SIGSEGV NOT reproduced

**Image:** `ghcr.io/defilantech/llmkube-llama-vulkan-turboquant:experiment-e377994dcc5d96065354e7546d0b3762d9ed4a14`
(`/app/TURBO_SHA` = `175a652ddd30a6ec3a743fa7498abff517b35b84`, branch
`fix/vulkan-turbo4-block-layout`: 66/68-byte `block_turbo4_0` layout fix +
centroid-table sync fix, both proven by `test-backend-ops` on gfx1151,
9509/9509).

**Config:** the exact crashing repro above: `--cache-type-k/v turbo4
--ctx-size 131072 --flash-attn on --spec-type none --ubatch-size 2048
--parallel 1 --no-mmap --jinja --reasoning off`, memory 48Gi,
`GGML_VK_FA_LOG=1`. Note the fork's auto-asymmetric guard upgraded K to q8_0
at load (GQA 12:1), so KV ran as q8_0 K + turbo4 V; the same guard applied to
the July 3 crashing runs. Driven by the same #921 Foreman coder loop
(`coder-glm`, contextWindowTokens 120000).

**Result: the fork bug is fixed. Neither failure symptom of this report
reproduced.**

- **117 consecutive clean coder turns** over 63m55s (dispatch 22:43:12Z,
  server death 23:47:05Z). Every completion released with
  `stop processing ... truncated = 0`. Zero runaways: the largest single
  generations were 1,463 / 1,317 / 951 tokens, all with clean EOS, versus
  the baseline's 12-16k monotonic no-EOS runaways. `n_decoded` traces stayed
  bounded (hundreds) at a steady ~6.4-7 t/s all the way down.
- **Context reached 57,742 tokens**, roughly 2x the depth the f16/32K
  control could even hold (n_ctx 32768), with coherent tool-calling
  throughout. The July 3 runaways surfaced well before this depth.
- restartCount stayed 0 for the whole 117-turn window. No
  `forcing full prompt re-processing` lines occurred.

**Terminal event (66 min container uptime): unified-memory capacity wall,
not the fork bug.** While re-evaluating the full 57,742-token prompt for
turn 118 (prompt cache at 8187.6 MiB of its 8192 MiB limit), the node hit
kernel `SystemOOM` (victims: llama-server, node_exporter, nginx-ingress) and
the GPU submission failed:

```
radv/amdgpu: Not enough memory for command submission.
terminate called after throwing an instance of 'vk::DeviceLostError'
  what():  vk::Queue::submit: ErrorDeviceLost
```

Container exit 139 (C++ terminate/abort on `DeviceLostError`), coder task
turn-118 SSE 503, Workload INCOMPLETE. This is Problem 2 from the memory
spike doc (98GB model + long-context KV/compute buffers exhausting the 128GB
unified pool), the same wall that 137-OOMKilled the f16 runs at 76-81 min /
~28k tokens; here it surfaced GPU-side (DeviceLost -> abort -> 139) instead
of cgroup-side (137) because the allocation that failed was a Vulkan command
submission mid-prompt-eval at 57.7k tokens. Time-to-wall was 66 min vs f16's
76-81 min, but at ~2x the context depth and with an 8GB prompt cache the f16
config never accumulated.

**Verdict:** PASS on the bug this document reports (runaway generation +
SIGSEGV in the turbo4 KV path): fixed by the block-layout + centroid-table
fixes and not reproduced under the exact crashing config. The remaining
crash is the pre-existing unified-memory capacity wall, which needs its own
mitigation (smaller ctx, `--cache-ram 0`, or admission headroom), tracked
separately from the fork fix.

**Run artifacts:** turns 1-117 all `finish=stop`; crash tail captured in the
Task 6 report (`llama-cpp-turboquant/.superpowers/sdd/task-6-report.md`).

## Correction after adversarial review (2026-07-04)

This corrects overstated causal claims in the soak-results section above; the
timestamps, counts, and logs are unchanged.

- **Trigger claim withdrawn.** The framing above that turn 118's full
  prompt re-evaluation "triggered" the OOM is not supported by the
  timestamps: kernel `SystemOOM` events began **23:45:37Z**, about 21 seconds
  before the cache-limit log line and turn-118 launch (**23:45:58.7Z**). The
  node was already at its memory ceiling during turn 117's ordinary,
  clean-EOS decode; the 57.7k-token re-evaluation then ran into a wall that
  was already closing, and the server died mid-prefill (last progress line:
  `prompt processing n_tokens = 6144, progress = 0.11`).
- **Exit-code caveat.** Container exit 139 (SIGSEGV) does not match the
  visible failure path (`vk::DeviceLostError` -> `std::terminate` -> `abort`),
  which would nominally produce 134 (SIGABRT). The exact terminal signal path
  (e.g. a segfault during unwind/cleanup after device loss under node-wide
  memory exhaustion) is plausible but was not captured or characterized. What
  distinguishes this event from the July 3 fork-bug crashes despite the
  matching exit code is the surrounding evidence, not the exit code itself:
  kernel SystemOOM killing unrelated node processes (node_exporter,
  nginx-ingress), an explicit `radv/amdgpu: Not enough memory for command
  submission` driver error, death during prefill rather than decode, and zero
  runaway generation anywhere in the 117-turn run (largest generation 1,463
  tokens, clean stop) versus the 12-16k-token runaways that preceded both
  July 3 crashes.
- **Headroom, not turbo4-specific waste.** turbo4 reached ~57.7k context
  before hitting the wall versus the f16 control's ~28k at its 137 OOM; that
  is consistent with turbo4's roughly 4x-smaller KV buying more context
  depth, not any turbo4-specific memory overhead.
- **Verdict, stated precisely.** The two fork bugs (runaway generation and
  its SIGSEGV) are fixed: neither recurred under the exact reproduction
  config, and the hardware test suite passed 9509/9509. The terminal event
  here is attributed to the pre-existing unified-memory capacity wall with
  high confidence, but its exact signal path is uncharacterized. Follow-ups
  belong to the capacity investigation, not the fork PR: capture the raw
  terminal signal (dmesg/coredump), per-turn VRAM/GTT telemetry (`rocm-smi` /
  amdgpu sysfs), an f16 @ ctx 131072 control on the same node, and
  prompt-cache sizing (`--cache-ram`) tuning.
