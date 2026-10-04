# Patches carried on b12x 78ee52c3

Base: `local-inference-lab/b12x` commit `78ee52c302abcef019d7ce17634c80b76f0975c9` (`patches/UPSTREAM_COMMITS.txt`),
upstream's `integration/karmic-kraken-beta` line, tagged `b12x-cu134-beta-78ee52c302abcef019d7ce17634c80b76f0975c9`.
It is the first beta b12x release after the vLLM pin (`1286ae9c`, 2026-10-03T16:57Z; b12x 78ee52c3, 17:14Z), and it
carries every b12x change the vLLM pin's release records require (`.lil/changes/*.json` "requires": b12x-401, -402,
-408, -409, -420, -423, -433, -452, -454, the four b12x-450-csf-* changes, dense-mla-fp32-partials and the four
master syncs), each present as `.lil/changes/<id>.json` at 78ee52c3. 78ee52c3 is docs-only on top of 566d70cf (the
merge of #468, after #456 CSF, #461, #464 deterministic MoE preparation and #467 collective deadlines).
The Dockerfile applies every `*.patch` here in filename order with `git apply --check` then `git apply`,
records each patch's new files with `git add -N`, and then installs b12x. `build/pins_gate.py` checks each file against the table
below (sha256), that the checkout differs from the pin by exactly these patches, that every patched file under
`b12x/` is installed byte-identical, and that each marker appears in an installed patched file. Keep one row per
line and do not put a `|` inside a cell. Drop a patch (and its row) when the pin moves past its merge.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-roce-switchless-ring-routes.patch | local-inference-lab/b12x#457 | 4eccc7022b19ac01d19f10f1fc080196fe793dc6 | Defilan | b12x/comm/roce (RoCEnante setup and RDMA proxy) | PR diff restricted to b12x/ (tests, docs and evidence left out), regenerated against 78ee52c3; every hunk is byte-identical to the 6380e581 copy in cuda-gb10-vllm-dsv41-b12x and only the _oneshot_cute.py index line differs, because upstream changed that file outside the patch's hunks (torch.dtype names in get_launcher) | three DGX Sparks cabled in a ring | 8a6252ff5de34ee607e14f011e3463c5e1168f823d3bf4b455800d454aff70ca |

Why 0001: RoCEnante pairs local HCA h with every peer's HCA h, which only holds on a switched fabric. On the
three-Spark ring every queue pair timed out at RTR and vLLM fell back to NCCL for the tensor-parallel
all-reduce. With this patch RoCEnante connects on the ring; the evidence (GPU suite, standalone collectives
receipt, serving A/B) is in the PR. `git apply` reports every hunk applying cleanly at 78ee52c3 with no offset.
