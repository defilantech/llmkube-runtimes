# Patches applied to llama.cpp b10794 in this image

Base: `ggml-org/llama.cpp` tag `b10794`, commit
`f9f09f02cc44d87d842dbd2d578857d92d4bb63b`. The Dockerfile applies the files
below in filename order, each with `git apply --check` first. Checked on
2026-09-30 against a clean `--depth 1` clone of that tag.

`scripts/rocm-qwen4exp-gate.sh` reads the table below. Every `*.patch` file
in this directory must have exactly one row, the sha256 must match the file,
and the marker string must be present in the image's `libllama.so`. Keep one
row per line and do not put a `|` inside a cell.

## Applied

| File | Upstream | Head SHA | Author | Backend scope | Form | Marker | sha256 |
|------|----------|----------|--------|---------------|------|--------|--------|
| 0001-qwen4exp-mtp.patch | ggml-org/llama.cpp#28243 | (vendored earlier, see vulkan-qwen4exp) | danielhanchen | common (model graph, converter) | exact PR diff, byte-identical to vulkan-qwen4exp/patches | QWEN4EXP MTP: | 82d13e06d2f259b587b9ac8c8b91f34d633b6089006da88bd3422a44541db3b6 |
| 0010-qwen4exp-qsa-gather-decode.patch | ggml-org/llama.cpp#28213 | beed2f78ac42cf16710b763e6f3ba20665c6d233 | abdel-darwish-27 | common (src/, backend-agnostic graph) | exact PR diff | QWEN4EXP_QSA_GATHER | 657f615e33294f99c6d1d9738b0382f9078f2fc81aad4540ed0e5cd24520cbf1 |
| 0011-qwen4exp-pooled-key-cache.patch | ggml-org/llama.cpp#28699 | 141f3f5646aa15e88d53198610a7540f4f4b0d71 | Rhonstin | common (src/, backend-agnostic memory + graph) | exact PR diff (PR is a DRAFT) | LLAMA_QSA_NO_POOLED_CACHE | 8de1ec95e28282464a6ae0d9182f859f68133176b55d4b8ac33f2d772cf6ffef |
| 0012-lazy-direct-row-reads.patch | ggml-org/llama.cpp#29030 | 7dc9a32df45c56ec48071de0d0d34466c6ce345c | pwilkin | common (src/ loader, graph, new lazy reader) | minimal backport, see below | row reads enabled | a1db927d2fbd114cb4b5342d5a126231f316d9bdb1dbd22747c84625735155d8 |

The markers are strings each patch introduces into `libllama.so`. None of
them exist in stock b10794 (checked by grepping the tag's `src/`, `common/`,
`include/` and `tools/`).

## Not applied

| Upstream | Head SHA | Author | Backend scope | Result on b10794 + 0001 |
|----------|----------|--------|---------------|-------------------------|
| ggml-org/llama.cpp#29751 ("llama: fix qwen4exp") | f72ec5aaf359532d7ef588af9ddfb73e6e6e97b7 | am17an | common (src/, backend-agnostic, so HIP-relevant) | does not apply: 29 of 38 hunks fail (llama-hparams.h 1/1, llama-memory-hybrid-idx.cpp 21/21, llama-memory-hybrid-idx.h 2/3, qwen4exp.cpp 5/11). Written against a master whose QSA pooling code has moved well past b10794. Not a context conflict. |
| ggml-org/llama.cpp#29639 ("vulkan: sparse flash attention for quantized K/V") | 61be16b616e49775d1be74f220b0a1a915414d9a | fxgsell | Vulkan only (ggml-vulkan.cpp, flash_attn.comp, test-backend-ops) | does not apply (ggml-vulkan.cpp 1/1, test-backend-ops.cpp 2/2 fail), and irrelevant to a HIP image either way |

Rebasing the pin to pick up #29751 is project-2 work, not this image.

## 0012 backport, exactly what differs from the PR

`gh pr diff 29030` does not `git apply` to b10794 (6 hunks fail). The vendored
file is the PR's diff regenerated with `git diff` against b10794 + 0001 +
0010 + 0011, so hunk headers and `index` lines differ. Its added and removed
lines are identical to the PR's except for these, checked line by line:

1. `src/CMakeLists.txt`: the PR adds `llama-lazy-reader.cpp` to the source
   list with 4-space indent; b10794's list is indented 12 spaces, so the
   same line is added at 12. Whitespace only.
2. `src/llama-context.cpp`: the PR adds `#include <thread>` with
   `<unordered_map>` as trailing context; b10794 has no `<unordered_map>`
   include there. Same line added after `<string>`. Context only.
3. `src/llama-model.cpp`: the PR's hunk that deletes the "resolve AUTO on
   systems without mmap support ... see #28160" block is dropped. That block
   landed on master after b10794 and does not exist in the base, so the
   PR's end state (no such fallback) is already true. No code added.
4. `tools/cli/README.md`, `tools/completion/README.md`,
   `tools/server/README.md`: dropped. Docs only; not compiled and not shipped
   in the image.

Every other hunk applies to the base with `git apply` unchanged (no fuzz).
