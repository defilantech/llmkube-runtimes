# Patches carried on NCCL 2.30.7

Base: `NVIDIA/nccl` commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`, the `v2.30.7-1` release tag
(`patches/UPSTREAM_COMMITS.txt`). The Dockerfile's `nccl-build` stage applies every `*.patch` here in filename order
(`build/apply_carried_patches.sh`), builds `libnccl.so.2.30.7` from that tree for sm_120 (cubins that load on GB10's
sm_121) and records the commit, the patches' sha256 and the library's sha256 in `/opt/llmkube/nccl-build.txt`. The
image stage applies the same patches to its own `/src/nccl`, installs that library as the `nvidia-nccl-cu13` 2.30.7
wheel's `libnccl.so.2`, and `build/pins_gate.py` checks the checkout is the pin plus exactly these patches, each
marker is in the patched source, and the installed library is the recorded one. Keep one row per line and do not put
a `|` inside a cell. Drop a patch (and its row) when the pin moves to a release that contains it.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-ib-isend-cts-nreqs-acquire-fence.patch | NVIDIA/nccl#2393 (landed on dev as 05d026f4685c1c15f53f8f76af91766e3e0eb4b6) | b85c5d9c4aa5b46c737d4eaf1940ee917d158e5f | kodlan (Stanislav Bardyuk) | src/transport/net_ib/p2p.cc (ncclIbIsend) | exact PR diff; identical to the dev commit's diff; applies to v2.30.7-1 at the same line | Order the nreqs load after the idx load above | d6cb34f4ad68b40dcfe3f0c3c17ca47b3757d42ccc264f4f05d056ef71e22ffa |

Why: `ncclIbIsend` checks a CTS fifo slot's `idx` and then loads its `nreqs`, both written by the peer's NIC, with
nothing ordering the second load after the first. On AArch64 (Grace) the `nreqs` load can return the previous
round's value; a stale larger value spins the proxy thread forever (the hang in NVIDIA/nccl#1983). The fix is one
acquire fence (one `dmb ishld` on AArch64, nothing on x86). It landed on NCCL's `dev` branch on 2026-10-01 (the PR
shows as closed because it was mirrored to dev, not merged to master), and no release contains it yet. Carrying the
two-line diff on the 2.30.7 release tag, rather than building `dev` (651 commits past v2.30.7-1), keeps every other
NCCL byte at the version the base image and the ring were qualified with.
