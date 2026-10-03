# Patches carried on the vLLM fork ec49d578

Base: `local-inference-lab/vllm` commit `ec49d5781a0ebf7960c7711ac3e0f42376b1a127` (`patches/UPSTREAM_COMMITS.txt`).
The Dockerfile applies every `*.patch` here in filename order with `git apply --check` then `git apply` (new files
recorded with `git add -N`) twice: in the compile stage, before the wheel is built, and in the image's `/src/vllm`
checkout, which the pins gate compares the installed package against. `build/pins_gate.py` checks each file against
the table below (sha256), that the checkout differs from the pin by exactly these patches, that every patched file
under `vllm/` is installed byte-identical, and that each marker appears in an installed patched file. Keep one row
per line and do not put a `|` inside a cell. Drop a patch (and its row) when the pin moves past its merge.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-ds41-trellis-mcg.patch | Defilan/vllm feat/ds41-trellis-mcg (no upstream PR yet) | 9036f769a16cd8a174310d19bcfe2cb4bdb90a56 | Defilan | vllm/models/deepseek_v4_1/trellis.py (DS4.1 independent trellis config and MoE method) | commit diff ec49d578..9036f769 restricted to vllm/ (its test left out) | lut_e4m3 K2 or mcg K3..K6 | d613e43193f3187dfe488823902bb1930cb58f12c8544e7ac0819d56cfadc2f1 |

Why: `DeepseekV41TrellisConfig.from_config` accepted only the lut_e4m3 K2 trellis manifest. The patch also accepts
exllamav3's mcg codebook at K3..K6, and the per-layer log line names the codebook and K. It pairs with
`patches/b12x/0002-independent-trellis-mcg.patch`, which teaches b12x's checkpoint reader the same manifests.
The patch touches Python only, so the compiled extensions are the same as an unpatched build.
