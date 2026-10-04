#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build-time prewarm for the karmic-kraken-beta image, under the RUNTIME env.

What can be warmed without a GPU is the import graph of the V4.1 serve path: every module below imports, b12x's
lazy op registry resolves the ops the fork calls, and a broken import fails the build instead of the first pod.
What cannot: b12x's CuTe-DSL kernels are planned and compiled per shape by a PreparationSession on the device at
first serve. They are cached in B12X_COMPILE_CACHE_DIR, keyed on the compiler toolchain env plus every non-operational
B12X_*/CUTE_*/CUTLASS_* variable, so that key is recorded here: a serve env that differs from it compiles again.
"""
from __future__ import annotations
import importlib
import os
import sys
import time

DEBUG_FLAGS = ("FLASHINFER_JIT_VERBOSE", "FLASHINFER_JIT_DEBUG", "FLASHINFER_JIT_LINEINFO")

V41_MODULES = (
    "b12x",
    "vllm.models.deepseek_v4_1.attention",
    "vllm.models.deepseek_v4_1.b12x_layers",
    "vllm.models.deepseek_v4_1.compressor",
    "vllm.models.deepseek_v4_1.sparse_mla",
    "vllm.models.deepseek_v4_1.nvidia.model",
    "vllm.models.deepseek_v4_1.nvidia.b12x_attention",
    "vllm.v1.attention.backends.mla.b12x_mla_sparse",
    "vllm.model_executor.models.config",
    # MXFP4-CSF (lossless scale compression): the fork's reader and DS4.1 quantization method, and b12x's CSF kernels.
    "vllm.model_executor.model_loader.mxfp4_csf_loader",
    "vllm.models.deepseek_v4_1.mxfp4_csf",
    "vllm.model_executor.layers.quantization.mxfp4_csf",
    "b12x._lib.quant.mxfp4_csf",
    # RoCEnante all-reduce (b12x collectives) with the carried switchless-ring routing.
    "b12x.comm.roce._routes",
)
B12X_OPS = ("attention.compressed_sparse_mla", "attention.dsa_indexer", "attention.mla_compress",
            "gemm.wo_projection", "gemm.block_fp8_linear", "gemm.bf16_gemv", "norm.mhc", "norm.hyperconnection")


def scrub_env(env: dict) -> dict:
    return {k: v for k, v in env.items() if k not in DEBUG_FLAGS}


def main() -> int:
    kept = scrub_env(dict(os.environ))
    os.environ.clear()
    os.environ.update(kept)
    for name in V41_MODULES:
        t = time.time()
        importlib.import_module(name)
        print(f"prewarm: import {name} {time.time() - t:.1f}s", flush=True)
    for op in B12X_OPS:
        importlib.import_module(f"b12x.{op}")
    print(f"prewarm: b12x ops resolve: {' '.join(B12X_OPS)}", flush=True)
    from b12x._lib.compiler import _compile_environment_key
    key = _compile_environment_key()
    cache = os.environ.get("B12X_COMPILE_CACHE_DIR", "")
    with open("/opt/llmkube/b12x-compile-env.txt", "w") as f:
        f.write(f"B12X_COMPILE_CACHE_DIR={cache}\n")
        for k, v in key:
            f.write(f"{k}={v}\n")
    print(f"prewarm: b12x compile cache {cache}; compile env key has {len(key)} entries", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
