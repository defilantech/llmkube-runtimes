"""CPU-only reproduction of vLLM's KV cache grouping for the three-Spark DeepSeek-V4.1-Flash ring, and of the
DSV41_KV_GROUPING=fine hunk (issue #46). Adapted from bot-lab-21's recipe/patches/kvgroup/sim_kvgroup.py (MIT);
the geometry is the sm12x-pages patch this image ships (SWA 64-token pages, compressed/indexer 64 states per page,
4 kv-source layers, 43 SWA caches incl. the 3 DSpark draft layers). Runs in the image, no GPU.

The ring's served configuration: 524,288 context, 8,192 batched tokens, KV pinned at 4,509,715,660 bytes per rank,
DSpark k=5. vLLM logged "GPU KV cache size: 1,057,457 tokens" for it on 2026-09-20 (generations 43, 46, 47, 48).
Default mode must reproduce that number with whichever async-scheduling setting the build resolves to; fine mode
must return more tokens for the same bytes.
"""
from __future__ import annotations
import os
import types

import pytest

pytest.importorskip("vllm")
import torch  # noqa: E402
from vllm.v1.core import kv_cache_utils as U  # noqa: E402
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec, get_kv_quant_mode  # noqa: E402
from vllm.v1.kv_cache_layout import KVCacheLayout  # noqa: E402
from vllm.utils.math_utils import cdiv  # noqa: E402

PIN = 4509715660
CR = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
KV_SRC = [2, 8, 14, 20]
LOGGED_TOKENS = 1_057_457
MAXLEN, BATCHED = 524288, 8192


def specs(swa_bs=64, n_draft=3, draft_win=192):
    d, q = {}, get_kv_quant_mode("fp8_ds_mla")
    for i in range(40 + n_draft):
        cr, p = CR[i], f"model.layers.{i}.self_attn"
        if i < 40 and i in KV_SRC:
            d[p] = MLAAttentionSpec(block_size=64 * cr, num_kv_heads=1, head_size=512, dtype=torch.uint8,
                                    tokens_per_state=cr, cache_dtype_str="fp8_ds_mla", alignment=576,
                                    model_version="deepseek_v4", kv_quant_mode=q, state_content_bytes=584)
            d[p + ".indexer.k_cache"] = MLAAttentionSpec(block_size=64 * cr, num_kv_heads=1, head_size=132,
                                                         dtype=torch.uint8, tokens_per_state=cr, alignment=576)
        d[p + ".swa_cache"] = SlidingWindowMLASpec(block_size=swa_bs, num_kv_heads=1, head_size=512,
                                                   dtype=torch.uint8, sliding_window=(draft_win if i >= 40 else 128),
                                                   cache_dtype_str="fp8_ds_mla", state_content_bytes=584,
                                                   alignment=576, model_version="deepseek_v4", kv_quant_mode=q)
    return d


def cfg(L, mnbt, async_on):
    ns = types.SimpleNamespace
    return ns(scheduler_config=ns(disable_hybrid_kv_cache_manager=False, async_scheduling=async_on,
                                  max_num_batched_tokens=mnbt),
              model_config=ns(max_model_len=L, original_max_model_len=L, hf_config=ns(model_type="deepseek_v41")),
              parallel_config=ns(decode_context_parallel_size=1, pipeline_parallel_size=1),
              max_in_flight_tokens=(2 if async_on else 1) * mnbt,
              cache_config=ns(block_size=128, num_gpu_blocks_override=None, prefix_cache_retention_interval=None,
                              cache_dtype="fp8_ds_mla", get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC),
              speculative_config=ns(method="dspark", num_speculative_tokens=5, use_eagle=lambda: True,
                                    use_eagle_block_drop=lambda: True, use_multi_module_mtp=lambda: False))


def tokens(mode, async_on):
    os.environ["DSV41_KV_GROUPING"] = "fine" if mode == "fine" else ""
    try:
        c = cfg(MAXLEN, BATCHED, async_on)
        groups = U.get_kv_cache_groups(c, specs())
        kvc = U.get_kv_cache_config_from_groups(c, groups, PIN)
        t, _conc = U.get_kv_cache_capacity(c, kvc)
        return int(t), len(groups)
    finally:
        os.environ.pop("DSV41_KV_GROUPING", None)


def test_default_mode_reproduces_the_pool_the_ring_logged():
    got = {a: tokens("default", a) for a in (False, True)}
    assert LOGGED_TOKENS in {t for t, _ in got.values()}, got
    print("default mode reproduces the logged pool with async_scheduling =",
          [a for a, (t, _) in got.items() if t == LOGGED_TOKENS])


def test_fine_mode_packs_more_tokens_into_the_same_pin():
    for async_on in (False, True):
        d, dg = tokens("default", async_on)
        f, fg = tokens("fine", async_on)
        assert fg > dg, (dg, fg)
        assert f > d * 1.25, f"fine {f:,} vs default {d:,} (async={async_on}); bot-lab-21 simulated +43..70%"
        print(f"async={async_on}: default {d:,} ({dg} groups) -> fine {f:,} ({fg} groups), {(f / d - 1) * 100:+.0f}%")
