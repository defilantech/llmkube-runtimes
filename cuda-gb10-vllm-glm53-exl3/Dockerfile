# syntax=docker/dockerfile:1
#
# LLMKube NVIDIA GB10 (DGX Spark) vLLM runtime, GLM-5.3-Flash EXL3 variant.
#
# Serves Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw across two GB10s at TP=2 over
# CX7. Built from source rather than pulled, for a licensing reason that is the
# whole story of this file.
#
# PROVENANCE AND LICENSE -- READ BEFORE BUMPING ANYTHING
#
# This recipe derives from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks at commit
# 9cdf84570117a6afc203125a5c01ec61978c4e60, the LAST COMMIT BEFORE THAT PROJECT
# RELICENSED TO AGPL-3.0 on 2026-09-07 (relicense commit cf0f430e, 13:43 UTC).
# At 9cdf8457 the project's LICENSE is MIT, retained here as
# LICENSE.upstream-MIT, and its notice must travel with these files.
#
# That boundary is deliberate and load-bearing. llmkube-runtimes is Apache-2.0;
# MIT vendors into that cleanly with attribution, AGPL-3.0 does not. Upstream's
# own relicense notice preserves MIT for contributions made before that date,
# and a granted license cannot be revoked, so this snapshot is usable on MIT
# terms indefinitely.
#
# DO NOT "update" the overlay or the base image by pulling from upstream main.
# Anything committed after 2026-09-07 13:43 UTC is AGPL-3.0 and cannot ship in
# this repo. That also rules out the upstream prebuilt image
# ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3, whose binaries are built
# from post-relicense source. Building from the vLLM base is not a preference
# here; it is the only license-clean path.
#
# WHAT THE MIT BOUNDARY COSTS, WHICH IS LESS THAN IT SOUNDS
#
# Exactly two overlay patches are AGPL-only and therefore absent:
# patch_adaptive_k.py (adaptive verification length) and patch_dense_fp8.py
# (FP8 weight-only dense projections). BOTH ARE OPT-IN AND OFF BY DEFAULT
# upstream, and the final AGPL-era change to exl3.py only wires the second one
# in. So the default serving configuration is complete here.
#
# The E3 grouped MoE kernel -- the +37-45% cold prefill win -- is NOT lost. It
# was committed between 04:44 and 06:15 UTC on 2026-09-07, hours before the
# 13:43 relicense, so exl3_fat_moe.cu is MIT and is vendored.
#
# WHY THE OVERLAY EXISTS AT ALL
#
# Stock vLLM cannot serve this checkpoint. It dies on the first forward with
# `pe_dim must be 64 for fp8_ds_mla`: GLM-5.3-Flash is NoPE MLA
# (qk_rope_head_dim=0, kv_lora_rank=512) and the only sparse-MLA backend on
# SM12x is FLASHINFER_MLA_SPARSE_SM120, whose packed record is 512 NoPE + 16 B
# scales + 128 B RoPE. The first RUN below zero-pads the 512-d latent into that
# GLM_NSA geometry; the RUN after it asserts the edit landed. exl3.py then
# registers a real EXL3 quantization method so routed experts stay packed as
# trellis + suh + svh + mcg. Registering the NAME "exl3" is not enough.
#
# WHAT THIS FILE CHANGES VERSUS UPSTREAM 9cdf8457
#
#   1. Abliteration removed. Upstream ships an opt-in refusal-direction
#      ablation (ABLIT=1) with donor artifacts. Neither the artifacts, the
#      runtime hook, nor its test are copied here, so the capability is absent
#      rather than merely defaulted off.
#   2. MAX_JOBS is a build ARG. Upstream fixes it at 8 for a workstation; this
#      builds on a hosted arm64 runner with 4 vCPU and 16 GB RAM, where 8
#      concurrent nvcc processes get OOM-killed.
#   3. Header, receipts and labels are ours.
#
# Everything else, including the two long inline Python blocks, is upstream's
# MIT code carried verbatim rather than retyped, so it cannot drift by
# transcription.
#
# ARCHITECTURE: arm64 only. GB10 is aarch64 Grace and the base publishes no
# amd64 manifest, so an x86 runner cannot build this with or without QEMU.

ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
FROM ${BASE}

RUN python3 - <<'PY'
from pathlib import Path

target = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "flashinfer_mla_sparse_sm120.py"
)
source = target.read_text()


def replace_once(old: str, new: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch target: {old!r}")
    source = source.replace(old, new)


replace_once(
    '    """SM120 FlashInfer sparse-MLA implementation."""\n\n    is_sparse = True\n',
    '    """SM120 FlashInfer sparse-MLA implementation."""\n\n'
    "    is_sparse = True\n"
    "    supports_dense_mha_prefill = False\n",
)

replace_once(
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        from vllm.config import get_current_vllm_config\n",
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        self.rope_pad = 0\n"
    "        if self.qk_rope_head_dim == 0:\n"
    "            if self.kv_lora_rank != 512:\n"
    "                raise NotImplementedError(\n"
    '                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "\n'
    '                    "576-wide GLM_NSA geometry, which requires "\n'
    '                    f"kv_lora_rank=512; got {self.kv_lora_rank}."\n'
    "                )\n"
    "            self.rope_pad = 64\n"
    "        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad\n"
    "        from vllm.config import get_current_vllm_config\n",
)

replace_once(
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "\n"
    "        num_actual_toks = q.shape[0]\n",
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "        if self.rope_pad:\n"
    "            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n"
    "\n"
    "        num_actual_toks = q.shape[0]\n",
)

replace_once(
    "            qk_rope_head_dim=self.qk_rope_head_dim,\n",
    "            qk_rope_head_dim=self.kernel_qk_rope_head_dim,\n",
)

replace_once(
    "        topk_indices_physical = cast(\n"
    "            torch.Tensor,\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "            ),\n"
    "        )\n",
    "        topk_indices_physical, topk_lengths = cast(\n"
    "            tuple[torch.Tensor, torch.Tensor],\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "                return_valid_counts=True,\n"
    "            ),\n"
    "        )\n"
    "        sparse_topk_capacity = topk_indices_physical.shape[1]\n"
    "        empty_rows = topk_lengths == 0\n"
    "        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
    "            empty_rows, 0\n"
    "        )\n"
    "        topk_lengths = topk_lengths.clamp(min=1)\n",
)

replace_once(
    "            seq_lens=None,\n            max_seq_len=attn_metadata.topk_tokens,\n",
    "            seq_lens=topk_lengths,\n            max_seq_len=sparse_topk_capacity,\n",
)

replace_once(
    "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
    "            sparse_mla_top_k=sparse_topk_capacity,\n",
)

replace_once(
    "        return out.squeeze(1), None\n",
    "        out = out.squeeze(1)\n"
    "        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
    "        return out, None\n"
    "\n"
    "    def do_kv_cache_update(\n"
    "        self,\n"
    "        kv_c_normed: torch.Tensor,\n"
    "        k_pe: torch.Tensor,\n"
    "        kv_cache: torch.Tensor,\n"
    "        slot_mapping: torch.Tensor,\n"
    "        kv_cache_dtype: str,\n"
    "        k_scale: torch.Tensor,\n"
    "    ) -> None:\n"
    "        if self.rope_pad:\n"
    "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
    "        super().do_kv_cache_update(\n"
    "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
    "        )\n",
)

target.write_text(source)

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
old_width = "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)"
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    path = site / rel
    text = path.read_text()
    if text.count(old_width) != 1:
        raise RuntimeError(f"expected one buffer_width target in {rel}")
    path.write_text(text.replace(old_width, "buffer_width = topk_tokens"))

sparse_backend = site / "v1/attention/backends/mla/flashinfer_mla_sparse.py"
text = sparse_backend.read_text()
old_block_sizes = (
    "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
    "        return [64, 256]\n"
)
if text.count(old_block_sizes) != 1:
    raise RuntimeError("expected one SM120 kernel-block-size target")
sparse_backend.write_text(
    text.replace(
        old_block_sizes,
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [64]\n",
    )
)

platform = site / "platforms/cuda.py"
text = platform.read_text()
old_align = "        return index_kpool * min(PAGED_MQA_PAGE_SIZES)\n"
if text.count(old_align) != 1:
    raise RuntimeError("expected one indexer block alignment target")
platform.write_text(
    text.replace(
        old_align,
        "        page_sizes = PAGED_MQA_PAGE_SIZES\n"
        "        capability = cls.get_device_capability()\n"
        "        if capability is not None and capability.major == 12:\n"
        "            page_sizes = tuple(p for p in page_sizes if p == 64)\n"
        "        return index_kpool * min(page_sizes)\n",
    )
)

indexer = site / "model_executor/layers/sparse_attn_indexer_kpool.py"
text = indexer.read_text()
for old, new in (
    (
        "                    expanded = expand_pools_and_append_tail(\n"
        "                        pool_ids, q_seq, index_kpool\n"
        "                    )\n",
        "                    expanded = expand_pools_and_append_tail(\n"
        "                        pool_ids[:, : select_k - 1], q_seq, index_kpool\n"
        "                    )\n",
    ),
    (
        "            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
        "            out = expand_pools_and_append_tail(\n"
        "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
        "            )\n",
    ),
):
    if text.count(old) != 1:
        raise RuntimeError(f"expected one expand-pools target: {old!r}")
    text = text.replace(old, new)
indexer.write_text(text)

warmup = site / "model_executor/warmup/kernel_warmup.py"
text = warmup.read_text()
old_sparse_warmup = (
    "    flashinfer_sparse_mla_decode_autotune_warmup(worker)\n"
    "    deepseek_v4_sparse_mla_attention_warmup(worker)\n"
)
if text.count(old_sparse_warmup) != 1:
    raise RuntimeError("expected one FlashInfer sparse-MLA warmup target")
text = text.replace(
    old_sparse_warmup,
    "    # GLM53_SKIP_FI_SPARSE_WARMUP: SM120 autotune wedges rank 0 on GB10.\n"
    "    deepseek_v4_sparse_mla_attention_warmup(worker)\n",
)
old_autotune = "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n"
if text.count(old_autotune) != 1:
    raise RuntimeError("expected one FlashInfer autotuner import")
text = text.replace(
    old_autotune,
    "    logger.info_once(\"Skipping FlashInfer autotune on SM121\")\n"
    "    return\n"
    "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n",
)
warmup.write_text(text)

platform = site / "platforms/cuda.py"
text = platform.read_text()
old_pdl = (
    "            return False\n"
    "        return major >= 9\n"
)
if text.count(old_pdl) != 1:
    raise RuntimeError("expected one PDL capability gate")
platform.write_text(
    text.replace(
        old_pdl,
        "            return False\n"
        "        # PDL lowering races KDA state kernels on SM12x (GB10).\n"
        "        return major in (9, 10)\n",
    )
)
PY

RUN python3 - <<'PY'
import inspect

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl as impl,
)

assert impl.supports_dense_mha_prefill is False
assert "do_kv_cache_update" in impl.__dict__
init_src = inspect.getsource(impl.__init__)
assert "self.rope_pad = 64" in init_src
fwd_src = inspect.getsource(impl.forward_mqa)
assert "torch.nn.functional.pad(q, (0, self.rope_pad))" in fwd_src
assert "qk_rope_head_dim=self.kernel_qk_rope_head_dim" in fwd_src
assert "return_valid_counts=True" in fwd_src
assert "sparse_mla_top_k=sparse_topk_capacity" in fwd_src
assert "seq_lens=topk_lengths" in fwd_src
assert "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)" in fwd_src
assert "attn_metadata.topk_tokens" not in fwd_src

from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    src = (site / rel).read_text()
    compile(src, rel, "exec")
    assert "buffer_width = topk_tokens\n" in src
    assert "kpool - 1 if kpool > 1" not in src
kpool_src = (site / "model_executor/layers/sparse_attn_indexer_kpool.py").read_text()
compile(kpool_src, "sparse_attn_indexer_kpool.py", "exec")
assert kpool_src.count("pool_ids[:, : select_k - 1]") == 2

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend as sm120_backend,
)

assert sm120_backend.get_supported_kernel_block_sizes() == [64]
align_src = (site / "platforms/cuda.py").read_text()
compile(align_src, "cuda.py", "exec")
assert "if capability is not None and capability.major == 12:" in align_src
assert "return index_kpool * min(page_sizes)" in align_src
assert "return major in (9, 10)" in align_src
warmup_src = (site / "model_executor/warmup/kernel_warmup.py").read_text()
compile(warmup_src, "kernel_warmup.py", "exec")
assert "flashinfer_sparse_mla_decode_autotune_warmup(worker)" not in warmup_src
assert "Skipping FlashInfer autotune on SM121" in warmup_src
print("glm53 NoPE sparse-MLA overlay verify OK")
PY

# EXL3/MCG trellis for routed experts (compiled for SM121, not SM120 cubins).
# Python overlay (exl3.py) is copied AFTER the CUDA compile so edits do not
# rebuild exllamav3_ext. The aarch64 stub patch must stay in this layer.
COPY overlay/patch_exl3_ext_aarch64.py /opt/glm53/patch_exl3_ext_aarch64.py
COPY overlay/patch_exl3_fat_kernel.py /opt/glm53/patch_exl3_fat_kernel.py
COPY overlay/exl3_fat_gemm.cu /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu
COPY overlay/exl3_fat_gemm.cuh /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cuh
COPY overlay/exl3_fat_moe.cu /opt/glm53/exl3-fat-kernel/exl3_fat_moe.cu
COPY overlay/exl3_fat_moe.cuh /opt/glm53/exl3-fat-kernel/exl3_fat_moe.cuh

ARG EXLLAMAV3_COMMIT=c5d9c657966ffeeaa9353f0cc899f18629da4a13
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
# MAX_JOBS is a BUILD ARG, not a fixed 8. Upstream targets a workstation;
# this builds on a GitHub hosted arm64 runner with 4 vCPU and 16 GB RAM,
# where 8 concurrent nvcc processes exhaust memory and the build dies with
# a bare "Killed". The workflow passes nproc; the default here is the
# conservative value for that runner.
ARG MAX_JOBS=3
ENV MAX_JOBS=${MAX_JOBS}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN python3 - <<'PY'
from pathlib import Path

init = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/__init__.py")
text = init.read_text()
old = 'QuantizationMethods = Literal[\n    "awq",\n'
new = 'QuantizationMethods = Literal[\n    "exl3",\n    "awq",\n'
if text.count(old) != 1:
    raise RuntimeError("expected one QuantizationMethods Literal header")
text = text.replace(old, new)
lazy = (
    "    # Update the `method_to_config` with customized quantization methods.\n"
    "    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n"
    "\n"
    "    return method_to_config[quantization]\n"
)
lazy_new = (
    "    from .exl3 import Exl3Config\n"
    "    method_to_config[\"exl3\"] = Exl3Config\n"
    "    # Update the `method_to_config` with customized quantization methods.\n"
    "    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n"
    "\n"
    "    return method_to_config[quantization]\n"
)
if text.count(lazy) != 1:
    raise RuntimeError("expected one get_quantization_config trailer")
text = text.replace(lazy, lazy_new)
init.write_text(text)
print("registered exl3 in QUANTIZATION_METHODS (lazy import)")
PY

# The vLLM image keeps CUDA toolkit headers under nvidia/cu13 (cusparse.h), not
# /usr/local/cuda/include. ExLlamaV3 also ships AVX2/AVX512 CPU targets that
# do not compile on aarch64; stub them so the SM121 GEMM still builds.
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 -c "from pathlib import Path; assert (Path('/tmp/exllamav3')/'exllamav3/modules/quant/exl3.py').is_file()"; \
    python3 /opt/glm53/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/glm53/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/glm53/exl3-fat-kernel; \
    export CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    cd /tmp/exllamav3; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="${MAX_JOBS}" \
      pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch; import exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_gemm'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_gemm_scatter'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_moe_gateup'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_moe_down'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_moe_gather'), dir(exllamav3_ext); print('exllamav3_ext', exllamav3_ext.__file__, 'exl3_moe=yes fat_gemm=yes fat_moe=yes')"; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

# Keep this AFTER the CUDA compile layer so Python-only hook edits do not
# rebuild exllamav3_ext. Exl3Config.override_quantization_method requires
# "exl3" in ModelConfig's ordered overrides list.
COPY overlay/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
COPY overlay/patch_model_overrides.py /opt/glm53/patch_model_overrides.py
COPY overlay/qwen3_dflash2.py /opt/glm53/qwen3_dflash2.py
COPY overlay/dflash2_speculator.py /opt/glm53/dflash2_speculator.py
COPY overlay/patch_dflash2.py /opt/glm53/patch_dflash2.py
COPY overlay/patch_glm_eagle3.py /opt/glm53/patch_glm_eagle3.py
COPY overlay/patch_glm5_drafter_group.py /opt/glm53/patch_glm5_drafter_group.py
COPY tests/test_exl3_overlay.py /opt/glm53/test_exl3_overlay.py
COPY files/chat_template.jinja /opt/glm53/chat_template.jinja
COPY overlay/patch_glm_video_placeholders.py /opt/glm53/patch_glm_video_placeholders.py
COPY overlay/patch_suppress_stops_in_reasoning.py /opt/glm53/patch_suppress_stops_in_reasoning.py
COPY tests/test_suppress_stops.py /opt/glm53/test_suppress_stops.py
COPY overlay/patch_scheduler_decode_floor.py /opt/glm53/patch_scheduler_decode_floor.py
COPY tests/test_scheduler_decode_floor.py /opt/glm53/test_scheduler_decode_floor.py
COPY overlay/patch_hybrid_prefix_hit.py /opt/glm53/patch_hybrid_prefix_hit.py
COPY tests/test_hybrid_prefix_hit.py /opt/glm53/test_hybrid_prefix_hit.py
COPY overlay/patch_xgrammar_termination.py /opt/glm53/patch_xgrammar_termination.py
COPY tests/test_xgrammar_termination.py /opt/glm53/test_xgrammar_termination.py
COPY overlay/patch_kpool_tail_slotmap.py /opt/glm53/patch_kpool_tail_slotmap.py
COPY tests/test_kpool_tail_slotmap.py /opt/glm53/test_kpool_tail_slotmap.py
COPY overlay/patch_spinwait.py /opt/glm53/patch_spinwait.py
COPY tests/test_spinwait_patch.py /opt/glm53/test_spinwait_patch.py
COPY overlay/patch_indexer_workspace.py /opt/glm53/patch_indexer_workspace.py
COPY tests/test_indexer_workspace.py /opt/glm53/test_indexer_workspace.py
RUN python3 /opt/glm53/patch_model_overrides.py
RUN python3 /opt/glm53/patch_dflash2.py
RUN python3 /opt/glm53/patch_glm_eagle3.py
RUN python3 /opt/glm53/patch_glm5_drafter_group.py
RUN python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
RUN python3 /opt/glm53/patch_scheduler_decode_floor.py
RUN python3 /opt/glm53/patch_hybrid_prefix_hit.py
RUN python3 /opt/glm53/patch_xgrammar_termination.py
RUN python3 /opt/glm53/patch_kpool_tail_slotmap.py
# Applied unconditionally; the injected sizing reads GLM53_INDEXER_WORKSPACE
# at runtime and returns the stock expression unless it is "rightsize".
RUN python3 /opt/glm53/patch_indexer_workspace.py
RUN python3 /opt/glm53/patch_spinwait.py --preflight

RUN EXL3_SELFCHECK_GPU=0 python3 /opt/glm53/test_exl3_overlay.py \
    && python3 /opt/glm53/test_suppress_stops.py \
    && python3 /opt/glm53/test_scheduler_decode_floor.py \
    && python3 /opt/glm53/test_hybrid_prefix_hit.py \
    && python3 /opt/glm53/test_xgrammar_termination.py \
    && python3 /opt/glm53/test_kpool_tail_slotmap.py \
    && python3 /opt/glm53/test_spinwait_patch.py \
    && python3 /opt/glm53/test_indexer_workspace.py

# Baked by start.sh --build-arg so a git pull that changes overlay/Dockerfile
# misses this label and rebuilds once. Keep last so stamp-only rebuilds are cheap.
ARG GLM53_RECIPE_STAMP=unknown
LABEL glm53.recipe.stamp=${GLM53_RECIPE_STAMP}

# ---------------------------------------------------------------------------
# LLMKube additions beyond the upstream MIT recipe.
# ---------------------------------------------------------------------------

# Fail the build if AGPL-era material ever lands in this image. The license
# boundary is a property we can actually test, so test it rather than rely on
# whoever edits overlay/ next remembering the date. Also asserts abliteration
# stayed out, since that is a capability decision and not a default.
RUN <<'EOT'
set -eu
bad=0
for f in /opt/glm53/patch_adaptive_k.py /opt/glm53/patch_dense_fp8.py; do
    [ -e "$f" ] && { echo "LICENSE VIOLATION: $f is AGPL-3.0 and cannot ship in an Apache-2.0 repo"; bad=1; }
done
for f in /opt/glm53/patch_ablit.py /opt/glm53/ablit_runtime.py /opt/glm53/ablit; do
    [ -e "$f" ] && { echo "POLICY: $f (abliteration) is deliberately not shipped"; bad=1; }
done
if [ "$bad" -ne 0 ]; then
    echo
    echo "See this Dockerfile's header: anything committed upstream after"
    echo "2026-09-07 13:43 UTC is AGPL-3.0. Do not vendor it here."
    exit 1
fi
echo "license boundary OK: no AGPL-era patches, no abliteration artifacts"
EOT

# Record the patched vLLM tree so a running container can prove nothing edited
# site-packages after the build (GLM53_VERIFY_TREE=1). Written last so it
# covers every patch layer above.
RUN <<'EOT'
set -eu
mkdir -p /opt/llmkube
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.py' -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 \
  > /opt/llmkube/vllm-tree.sha256
echo "vllm tree sha256: $(cat /opt/llmkube/vllm-tree.sha256)"
EOT

# Upstream's MIT notice must travel with the vendored overlay.
COPY LICENSE.upstream-MIT /opt/llmkube/LICENSE.upstream-MIT
COPY NOTICE /opt/llmkube/NOTICE

COPY entrypoint.sh /usr/local/bin/glm53-entrypoint.sh
RUN chmod 0755 /usr/local/bin/glm53-entrypoint.sh \
 && bash -n /usr/local/bin/glm53-entrypoint.sh

# Serving flags belong in InferenceService.spec.extraArgs, not baked here, so an
# operator can change them without a rebuild. ABLIT is the exception: it is a
# capability decision, and this image has no artifacts for it.
ENV ABLIT=0

LABEL org.opencontainers.image.title="llmkube-vllm-cuda-gb10-glm53-exl3" \
      org.opencontainers.image.description="vLLM on NVIDIA GB10 (sm_121) for GLM-5.3-Flash EXL3/TR3 4bpw across two DGX Sparks at TP=2. Built from the Apache-2.0 vLLM base plus MIT ExLlamaV3 and the pre-relicense MIT overlay. No AGPL material, no abliteration." \
      org.opencontainers.image.source="https://github.com/defilantech/llmkube-runtimes" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      dev.llmkube.gpu.arch="sm_121" \
      dev.llmkube.engine="vllm" \
      dev.llmkube.quantization="exl3" \
      dev.llmkube.upstream.kit="MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@9cdf84570117a6afc203125a5c01ec61978c4e60" \
      dev.llmkube.upstream.license="MIT (pre-2026-09-07 relicense)" \
      dev.llmkube.ablit="absent"

ENTRYPOINT ["/usr/local/bin/glm53-entrypoint.sh"]
