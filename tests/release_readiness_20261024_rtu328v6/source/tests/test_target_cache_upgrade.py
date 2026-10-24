"""HF 5 prefill, append and speculative rollback with stable KV storage."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from litespec.modeling.kv_cache import make_target_cache


def test_preallocated_target_cache_prefill_decode_crop():
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    cache = make_target_cache(model, 32)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        model(tokens[:, :3], past_key_values=cache, use_cache=True)
        pointers = cache.storage_ptrs()
        model(tokens[:, 3:4], past_key_values=cache, use_cache=True)
        cache.crop(3)
        actual = model(tokens[:, 3:], past_key_values=cache, use_cache=True).logits
        expected = model(tokens, use_cache=False).logits[:, 3:]
    assert cache.get_seq_length() == 5
    assert cache.storage_ptrs() == pointers
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_native_hybrid_cache_selection():
    from types import SimpleNamespace

    from transformers import Qwen3_5Config
    from transformers.cache_utils import DynamicCache

    config = Qwen3_5Config(
        text_config={
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
        }
    )
    cache = make_target_cache(SimpleNamespace(config=config), 32)
    assert isinstance(cache, DynamicCache)
    assert len(cache.layers) == 2
