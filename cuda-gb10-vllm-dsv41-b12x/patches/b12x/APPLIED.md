# Patches carried on b12x 6380e581

Base: `local-inference-lab/b12x` commit `6380e581f075df7f01a72d4b67f016225f260c86`, the head of upstream's
`evidence/ds41-x4t-serving-20260929` branch (`patches/UPSTREAM_COMMITS.txt`); it is the only upstream ref that has
`b12x/moe/checkpoints/independent.py`, the trellis-dense-checkpoint/1 reader.
The move from the `0d6600e6` master line drops three master commits the evidence branch lacks: `ba090286` (legacy
tensor-based API compatibility, including `b12x/moe/fused_moe/_compat.py` and the hasattr guards in
`b12x/integration/vllm/loader.py` and `plugin.py`), `0d6600e6` (launch heuristics) and `4bacd509` (CUTLASS DSL 4.7.1
pins). That is acceptable because the fork's DS4.1 path uses the `b12x.preparation.PreparedCall` APIs and explicit
plans, every b12x import in the fork resolves at `6380e581`, and the loader's `weight_utils` imports
(`file_source_tensor`, `safetensors_file_sources`) exist in the fork; the CUTLASS pin is handled in
`PIP_CHECK_ALLOW.txt`. See the image README.
The Dockerfile applies every `*.patch` here in filename order with `git apply --check` then `git apply`,
records each patch's new files with `git add -N`, and then installs b12x. `build/pins_gate.py` checks each file against the table
below (sha256), that the checkout differs from the pin by exactly these patches, that every patched file under
`b12x/` is installed byte-identical, and that each marker appears in an installed patched file. Keep one row per
line and do not put a `|` inside a cell. Drop a patch (and its row) when the pin moves past its merge.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-roce-switchless-ring-routes.patch | local-inference-lab/b12x#457 | 4eccc7022b19ac01d19f10f1fc080196fe793dc6 | Defilan | b12x/comm/roce (RoCEnante setup and RDMA proxy) | PR diff restricted to b12x/ (tests, docs and evidence left out); b12x/comm/roce is identical at 0d6600e6 and 6380e581, so the diff regenerated against 6380e581 is byte-identical | three DGX Sparks cabled in a ring | c28859bef3e62ab6c6f32aaa795538692564e2788ae8591483053bac236b54ee |
| 0002-independent-trellis-mcg.patch | Defilan/b12x feat/independent-trellis-mcg (no upstream PR yet) | f09975779faa742f84142b8bf2f22f6b5c821ba6 | Defilan | b12x/moe/checkpoints/independent.py (trellis-dense-checkpoint/1 reader) | commit diff 6380e581..f0997577 restricted to b12x/ (its tests left out) | mcg trellis requires codebook_seed | 579ebb1a8ef9fdd56a6d756c9bfe8ccee86e6df1945f0ea770526d44381d2396 |

Why 0001: RoCEnante pairs local HCA h with every peer's HCA h, which only holds on a switched fabric. On the
three-Spark ring every queue pair timed out at RTR and vLLM fell back to NCCL for the tensor-parallel
all-reduce. With this patch RoCEnante connects on the ring; the evidence (GPU suite, standalone collectives
receipt, serving A/B) is in the PR.

Why 0002: trellis-dense-checkpoint/1 was hardcoded to the lut_e4m3 codebook at K2. The patch also accepts
exllamav3's mcg codebook at K3..K6 (gated on its fixed multiplier seed 0xCBAC1FED), sizes the tile words from the
manifest's bits, and passes the codebook and K through to TrellisSource. lut_e4m3 K2 checkpoints load exactly as
before. It pairs with `patches/vllm/0001-ds41-trellis-mcg.patch`, which lets the fork's DS4.1 config accept the same
manifests.
