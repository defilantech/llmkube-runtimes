from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prewarm", HERE.parent / "build" / "prewarm.py")
prewarm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prewarm)


def test_scrub_env_drops_the_three_debug_flags_and_keeps_the_rest():
    env = {"FLASHINFER_JIT_VERBOSE": "1", "FLASHINFER_JIT_DEBUG": "1", "FLASHINFER_JIT_LINEINFO": "1",
           "FLASHINFER_NVCC_THREADS": "1", "MAX_JOBS": "4"}
    out = prewarm.scrub_env(env)
    assert "FLASHINFER_JIT_VERBOSE" not in out and "FLASHINFER_JIT_DEBUG" not in out and "FLASHINFER_JIT_LINEINFO" not in out
    assert out["FLASHINFER_NVCC_THREADS"] == "1" and out["MAX_JOBS"] == "4"
