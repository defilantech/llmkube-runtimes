#!/usr/bin/env python3
"""Pre-warm the FlashInfer JIT modules the first serve needs, under the RUNTIME env (spec 5.5).

A prewarm run with FLASHINFER_JIT_VERBOSE/DEBUG/LINEINFO set compiles with different nvcc flags and
is a different cache entry, so serve time would JIT again (tonyd2wild's overlay5 lesson). nvcc needs
no GPU; loading the built .so needs no GPU either.
"""
from __future__ import annotations
import os
import sys
import time

DEBUG_FLAGS = ("FLASHINFER_JIT_VERBOSE", "FLASHINFER_JIT_DEBUG", "FLASHINFER_JIT_LINEINFO")


def scrub_env(env: dict) -> dict:
    return {k: v for k, v in env.items() if k not in DEBUG_FLAGS}


def _build_sparse_mla() -> None:
    from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module
    t = time.time()
    get_sparse_mla_sm120_module()
    print(f"prewarm: sparse_mla_sm120 built and loaded in {time.time() - t:.1f}s", flush=True)


def _build_mxfp8_gemm() -> None:
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8
    spec = gen_gemm_sm120_module_cutlass_mxfp8()
    t = time.time()
    if hasattr(spec, "build_and_load"):
        spec.build_and_load()
    else:
        spec.build()
    compiled = spec.is_compiled() if callable(spec.is_compiled) else spec.is_compiled
    if not compiled:
        raise SystemExit("prewarm: mxfp8 gemm did not compile")
    print(f"prewarm: gemm_sm120_cutlass_mxfp8 built in {time.time() - t:.1f}s", flush=True)


def main() -> int:
    kept = scrub_env(dict(os.environ))
    os.environ.clear()
    os.environ.update(kept)
    _build_sparse_mla()
    _build_mxfp8_gemm()
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    print(f"prewarm: workspace {base}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
