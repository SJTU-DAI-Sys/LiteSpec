"""Tiny registered draft forward/backward and checkpoint loading on upgraded HF."""

import pytest
import torch
from transformers import Qwen3Config

from litespec.modeling.draft import (
    DFlash2DraftModel,
    DFlashDraftModel,
    DLiteDraftModel,
    DSparkDraftModel,
)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "model_class",
    [DFlashDraftModel, DFlash2DraftModel, DSparkDraftModel, DLiteDraftModel],
)
def test_registered_draft_backward_and_reload(
    model_class, implementation, tmp_path, tmp_path_factory, monkeypatch
):
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
    config._attn_implementation = implementation
    config.architectures = [model_class.__name__]
    config.num_target_layers = 4
    config.block_size = 4
    method = {"target_layer_ids": [0, 3], "mask_token_id": 31}
    if model_class is DFlash2DraftModel:
        method.update(
            conv_kernel_size=2, conv_group_size=4, selector_rank=4, selector_top_k=4
        )
    if model_class is DLiteDraftModel:
        method.update(
            architecture_version="dlite_v2",
            model_role="student",
            chs_num_layers=2,
            sequential_head="rnn",
            sequential_rank=4,
        )
    setattr(config, model_class.method_config_key, method)
    model = model_class(config)
    noise = torch.randn(1, 4, 16)
    if model_class is DLiteDraftModel:
        kwargs = {
            "position_ids": torch.arange(4).unsqueeze(0),
            "rotary_position_ids": torch.arange(6).unsqueeze(0),
            "noise_embedding": noise,
            "target_hidden": torch.randn(1, 1, 2, 16),
            "history_hidden": torch.empty(1, 1, 0, 16),
        }
    else:
        kwargs = {
            "position_ids": torch.arange(12).unsqueeze(0),
            "noise_embedding": noise,
            "target_hidden": torch.randn(1, 8, 32),
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad()
        value = model(**kwargs)
        loss = value.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
    model.eval()
    expected = model(**kwargs).detach()
    import transformers.dynamic_module_utils as dynamic_modules
    from transformers import AutoModel

    from litespec.training.checkpoint import export_draft

    monkeypatch.setattr(
        dynamic_modules,
        "HF_MODULES_CACHE",
        str(tmp_path_factory.getbasetemp() / "modules"),
    )
    export_draft(model, str(tmp_path))
    exported = AutoModel.from_pretrained(
        tmp_path, trust_remote_code=True, attn_implementation=implementation
    ).eval()
    torch.testing.assert_close(exported(**kwargs), expected, rtol=0, atol=0)
    restored = model_class.from_pretrained(
        tmp_path, attn_implementation=implementation
    ).eval()
    torch.testing.assert_close(restored(**kwargs), expected, rtol=0, atol=0)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": 2}, tmp_path / "optimizer.pt"
    )
    reloaded_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    state = torch.load(tmp_path / "optimizer.pt", weights_only=True)
    reloaded_optimizer.load_state_dict(state["optimizer"])
    assert state["step"] == 2 and len(reloaded_optimizer.state) == len(optimizer.state)


def test_rotary_stays_fp32_during_bfloat16_cast():
    from litespec.modeling.draft.rotary import FP32RotaryEmbedding

    config = Qwen3Config(
        hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64
    )
    rotary = FP32RotaryEmbedding(config)
    expected = rotary.inv_freq.clone()
    rotary.to(torch.bfloat16)
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
