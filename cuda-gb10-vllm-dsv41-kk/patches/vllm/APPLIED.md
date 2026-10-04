# Patches carried on the vLLM fork 1286ae9c

Base: `local-inference-lab/vllm` commit `1286ae9c9a3fd9376b82da39f7bc87f6a99037d9`, branch
`integration/karmic-kraken-beta`, tagged `vllm-jovian-cu134-beta-1286ae9c9a3fd9376b82da39f7bc87f6a99037d9`
(`patches/UPSTREAM_COMMITS.txt`).
The Dockerfile applies every `*.patch` here in filename order with `git apply --check` then `git apply` (new files
recorded with `git add -N`) twice: in the compile stage, before the wheel is built, and in the image's `/src/vllm`
checkout, which the pins gate compares the installed package against. `build/pins_gate.py` checks each file against
the table below (sha256), that the checkout differs from the pin by exactly these patches, that every patched file
under `vllm/` is installed byte-identical, and that each marker appears in an installed patched file. Keep one row
per line and do not put a `|` inside a cell. Drop a patch (and its row) when the pin moves past its merge.
A patch authored in this repository and not yet proposed upstream records `none` as its head SHA.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-ds41-mxfp4-csf-tp3.patch | carried here (not yet proposed upstream) | none | Defilan | vllm/model_executor/model_loader/mxfp4_csf_loader.py (read_mxfp4_csf_layer TP guard) | one-line allowlist change plus a comment, against 1286ae9c | DS4.1 TP3 (three DGX Sparks) | 18be69ad65b2d96e57f843eac7187c0e03ec26de965e222b05ce7528ffbe58b1 |

Why: the fork's MXFP4-CSF reader admits DeepSeek-V4.1 only at TP 1, 2, 4 or 8, by an explicit allowlist in
`read_mxfp4_csf_layer`; every other constraint on the path already holds at TP3. The routed-expert intermediate size
is 2304, so a TP3 rank owns 768 channels: `tp_extent` needs 32-channel alignment (2304 % 96 == 0), the compressed
scale-plane slicer needs whole 16-row slabs (rank boundaries 0, 768, 1536 are multiples of 16) and accepts any column
range, and b12x's compact W4A8 CSF geometry needs N64/K64 (w13 1536 x 5120, w2 5120 x 768 per rank). The Kimi family
in the same function already allows TP 12. The patch adds 3 for DS4.1 and nothing else. It touches Python only, so
the compiled extensions are the same as an unpatched build. `tests/test_imports_in_image.py` slices a synthetic CSF
expert set at TP3 through the fork's own loader and checks the bytes; serving it on a GB10 ring is not yet qualified
(see the image README). Upstream's branch head still has the same allowlist (checked at `7f172870`, 2026-10-04).
