"""vLLM resolves the TP3-edited checkpoint config the way the ring boot needs (spec 6)."""
from __future__ import annotations
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_model_config_resolves_tp3_tokenizer_and_exl3(tmp_path):
    (tmp_path / "config.json").write_bytes((FIXTURES / "config.tp3.json").read_bytes())
    (tmp_path / "quantization_config.json").write_bytes((FIXTURES / "quantization_config.json").read_bytes())
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import ModelConfig
    m = ModelConfig(model=str(tmp_path), tokenizer=str(tmp_path), skip_tokenizer_init=True)
    hc, tc = m.hf_config, m.hf_text_config
    assert tc.num_attention_heads == 72 and tc.o_groups == 9
    assert getattr(hc, "virtual_heads_from", None) == {"num_attention_heads": 64, "o_groups": 8}
    assert tc.n_routed_experts == 384 and tc.vocab_size == 129280 and tc.num_hidden_layers == 40
    assert m.tokenizer_mode == "deepseek_v41"
    assert m.quantization == "exl3"
    from vllm.models.deepseek_v4_1.virtual_heads import _virtual_from
    assert _virtual_from(hc) == {"num_attention_heads": 64, "o_groups": 8}
