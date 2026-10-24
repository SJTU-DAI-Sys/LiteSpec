"""Actual FLASH FlexAttention training for all DFlash-family registrations."""

import pytest
import torch
from transformers import Qwen3Config

from litespec.core.dflash import OnlineDFlashModel
from litespec.core.dflash2 import OnlineDFlash2Model
from litespec.core.dspark import OnlineDSparkModel
from litespec.modeling.draft import (
    DFlash2DraftModel,
    DFlashDraftModel,
    DSparkDraftModel,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "model_class", [DFlashDraftModel, DFlash2DraftModel, DSparkDraftModel]
)
def test_registered_flash_objective_backward(model_class):
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
    )
    config._attn_implementation = "flex_attention"
    config.architectures = [model_class.__name__]
    config.num_target_layers, config.block_size = 4, 4
    method = {"target_layer_ids": [0, 3], "mask_token_id": 31}
    if model_class is DFlash2DraftModel:
        method.update(
            conv_kernel_size=2, conv_group_size=4, selector_rank=4, selector_top_k=4
        )
    setattr(config, model_class.method_config_key, method)
    draft = model_class(config).cuda().to(torch.bfloat16)
    head = (
        torch.nn.Linear(128, 32, bias=False)
        .cuda()
        .to(torch.bfloat16)
        .requires_grad_(False)
    )
    embed = torch.nn.Embedding(32, 128).cuda().to(torch.bfloat16).requires_grad_(False)
    wrapper = {
        DFlashDraftModel: OnlineDFlashModel,
        DFlash2DraftModel: OnlineDFlash2Model,
        DSparkDraftModel: OnlineDSparkModel,
    }[model_class]
    online = wrapper(
        draft,
        head,
        embed,
        31,
        block_size=4,
        flex_attention_backend="FLASH",
        num_anchors=4,
        objective_chunk_blocks=64,
    )
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-4)
    tokens = torch.randint(0, 30, (1, 32), device="cuda")
    hidden = torch.randn(1, 32, 256, device="cuda", dtype=torch.bfloat16)
    final = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = online(
            tokens,
            hidden,
            torch.ones_like(tokens),
            target_last_hidden_states=final,
            collect_detailed_metrics=False,
        )
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert any(parameter.grad is not None for parameter in draft.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in draft.parameters()
            if parameter.grad is not None
        )
        optimizer.step()
