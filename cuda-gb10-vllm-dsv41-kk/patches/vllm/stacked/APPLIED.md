# Patches carried on the vLLM fork 1286ae9c after the spark-ds41f series

This repository's own patches that change code added by `spark-ds41f/` patches, so they apply after that series, in
the order of `series` (`build/apply_carried_patches.sh`). `build/pins_gate.py` reads this table exactly as it reads
`patches/vllm/APPLIED.md`: sha256, that reversing every carried patch restores the pin, and that each marker appears
in an installed patched file. The marker of every earlier patch must survive these patches too, so a patch here keeps
a series patch's marker line verbatim (0001 keeps spark-ds41f 0002's `stage_disk_async` signature as a wrapper).
Keep one row per line and do not put a `|` inside a cell.

## Applied

| File | Upstream | Head SHA | Author | Scope | Form | Marker | sha256 |
|------|----------|----------|--------|-------|------|--------|--------|
| 0001-engram-prefetch-next-chunk.patch | carried here (not yet proposed upstream) | none | Defilan | vllm/models/deepseek_v4_1/common/engram.py vllm/models/deepseek_v4_1/nvidia/model.py vllm/models/deepseek_v4_1/nvidia/model_state.py | on spark-ds41f 0002 (async disk Engram rows); default off: SPARK3_ENGRAM_PREFETCH=1 with SPARK3_ENGRAM_ASYNC=1, SPARK3_ENGRAM_PREFETCH_VERIFY=1 re-reads and compares | def _engram_prefetch_setup(self, engrams, next_chunk, row_set): | 597cdb084e28701badc38365e8a4b3ddcb709902d3cf5b6839dfde1b2c08879e |
