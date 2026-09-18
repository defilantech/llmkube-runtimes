# aarch64 build patches

Applied to the pinned ExLlamaV3 checkout before `pip install`:

```sh
git apply --check patches/0001-aarch64-build.patch   # drift gate: fails loudly if the pin moved
git apply       patches/0001-aarch64-build.patch
```

`0001-aarch64-build.patch` is **our own** patch (Apache-2.0, this repo), written
from the actual compile failures on `linux/aarch64`. It is not derived from the
recipe's AGPL-3.0-only build helper; see
[`docs/exllamav3-gb10-runtime.md`](../../docs/exllamav3-gb10-runtime.md) for the
licence boundary that forces this.

It does four things:

1. `avx2_target.cpp` / `avx512_target.cpp`: `__builtin_cpu_supports` is x86-only,
   so the support probes report unsupported on aarch64.
2. `avx2_target.h` / `avx512_target.h`: `target("avx2")` /
   `target("avx512f,avx512bw")` attributes are x86-only; the target macros become
   empty on aarch64.
3. `cpu/moe_handoff.cu` and `parallel/all_reduce_cpu.cu`: `__builtin_ia32_pause()`
   / `_mm_pause()` are x86-only; the spin idiom gets an ARM `yield` branch.
4. Three x86-only host-kernel translation units include `immintrin.h`, which does
   not exist on aarch64, and are stubbed symbol-for-symbol because portable TUs
   reference them unconditionally: `cpu/moe_mul1.cpp`,
   `parallel/all_reduce_cpu_avx2.cpp`, `parallel/all_reduce_cpu_avx512.cpp`. The
   stubs keep the exported symbols resolvable and throw on the compute paths, so
   misuse is loud rather than silently wrong. None of these paths is exercised by
   a single-GB10 TP1 GPU run.

Verified on `linux/aarch64` against the pinned commit: the patch applies cleanly
to a pristine checkout, `git apply --check` rejects a double-apply (so the drift
gate can actually fail), the extension builds green, and the shipped
`exllamav3_ext` carries **131 native `sm_121a` cubins and zero PTX** and dlopens
with every symbol resolved.

`UPSTREAM_COMMITS.txt` records the pinned source commit. Bumping the pin means
re-reading this patch against the new source and re-running the aarch64 build.
