"""Every layer of the overlay imports in the shipped image, without a GPU (spec 6)."""
from __future__ import annotations
import importlib
import importlib.metadata as md

SHAPES = [(h, k) for h in (16, 24, 32) for k in (1152, 640)]


def test_flashinfer_is_the_source_build_with_the_ring_shapes():
    import flashinfer
    from flashinfer.mla import supported_sparse_mla_sm120_configs
    assert flashinfer.__version__ == "0.7.0rc1"
    cfg = supported_sparse_mla_sm120_configs()["dsv4"]
    for h, k in SHAPES:
        assert cfg.supports_decode(num_heads=h, topk=k), (h, k)


def test_cuda_exl3_kernel_and_plugin_entry_point():
    import torch  # noqa: F401  (cuda_exl3._C links against libc10)
    import cuda_exl3
    import cuda_exl3._C  # noqa: F401
    from cuda_exl3 import config, moe  # noqa: F401  (bot-lab-21's V4.1 overlay landed here)
    eps = {(e.name, e.value) for e in md.entry_points(group="vllm.general_plugins")}
    assert ("cuda_exl3", "cuda_exl3:register") in eps


def test_patched_vllm_tree_imports_and_has_no_base_engram():
    mods = [
        "vllm.models.deepseek_v4_1.attention",
        "vllm.models.deepseek_v4_1.virtual_heads",
        "vllm.models.deepseek_v4_1.common.engram",
        "vllm.models.deepseek_v4_1.nvidia.model",
        "vllm.models.deepseek_v4_1.nvidia.vl_model",
        "vllm.models.deepseek_v4_1.nvidia.model_state",
        "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse",
        "vllm.models.deepseek_v4.nvidia.model",
        "vllm.model_executor.layers.vocab_parallel_embedding",
        "vllm.model_executor.layers.sparse_attn_indexer",
        "vllm.model_executor.model_loader.weight_utils",
        "vllm.v1.attention.backends.mla.sparse_swa",
    ]
    for m in mods:
        importlib.import_module(m)
    engram = importlib.import_module("vllm.models.deepseek_v4_1.common.engram")
    assert not hasattr(engram, "gather_engram_hashes"), "base image's DP-Engram engram.py survived the overlay"
    vh = importlib.import_module("vllm.models.deepseek_v4_1.virtual_heads")
    assert hasattr(vh, "wrap_attention") and hasattr(vh, "_virtual_from")
