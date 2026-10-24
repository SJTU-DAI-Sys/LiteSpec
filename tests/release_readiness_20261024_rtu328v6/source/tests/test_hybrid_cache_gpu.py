"""Native HF hybrid recurrent-cache rollback on the upgraded CUDA runtime."""

import pytest
import torch
from transformers import AutoModelForImageTextToText, Qwen3_5Config

from litespec.modeling.draft.dlite import (
    _begin_recurrent_cache_transaction,
    _crop_target_cache,
    _restore_recurrent_cache_transaction,
)
from litespec.modeling.kv_cache import make_target_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_hybrid_cache_restore_and_replay():
    torch.manual_seed(42)
    text = {
        "vocab_size": 32,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
    }
    config = Qwen3_5Config(
        text_config=text,
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "out_hidden_size": 128,
            "patch_size": 2,
        },
    )
    model = (
        AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa")
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    cache = make_target_cache(model, 32)
    tokens = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.no_grad():
        model(input_ids=tokens[:, :3], past_key_values=cache, use_cache=True)
        snapshot = _begin_recurrent_cache_transaction(cache)
        assert snapshot is not None
        model(input_ids=tokens[:, 3:], past_key_values=cache, use_cache=True)
        _crop_target_cache(cache, current_length=5, tokens_to_remove=2)
        _restore_recurrent_cache_transaction(snapshot)
        model(input_ids=tokens[:, 3:4], past_key_values=cache, use_cache=True)
        _crop_target_cache(cache, current_length=4, tokens_to_remove=0)
        actual = model(
            input_ids=tokens[:, 4:], past_key_values=cache, use_cache=True
        ).logits
        expected = model(input_ids=tokens, use_cache=False).logits[:, 4:]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.005)
