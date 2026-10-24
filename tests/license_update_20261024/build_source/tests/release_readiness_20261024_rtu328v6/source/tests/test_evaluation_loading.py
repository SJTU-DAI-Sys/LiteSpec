"""Load public benchmark target/draft/tokenizer bundles under Transformers 5."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3Config,
)

from evaluation.dflash_model_loading import load_dflash_benchmark_models
from evaluation.model_loading import has_flash_attention, load_dlite_benchmark_models
from litespec.modeling.draft import (
    DFlash2DraftModel,
    DFlashDraftModel,
    DLiteDraftModel,
    DSparkDraftModel,
)


def test_cute_namespace_is_not_flash_attention_2(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    assert not has_flash_attention()


@pytest.mark.parametrize(
    "model_class,hybrid",
    [
        (DLiteDraftModel, False),
        (DLiteDraftModel, True),
        (DFlashDraftModel, False),
        (DFlash2DraftModel, False),
        (DSparkDraftModel, False),
    ],
)
def test_public_evaluation_bundle(tmp_path, monkeypatch, model_class, hybrid):
    # FlashAttention 4 supplies a flash_attn namespace without FA2 symbols. The
    # public loader must recognize that and use its documented SDPA fallback.
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    fields = dict(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
    )
    if hybrid:
        text = dict(
            fields,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
        )
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
        target = AutoModelForImageTextToText.from_config(config)
    else:
        target = AutoModelForCausalLM.from_config(Qwen3Config(**fields))
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target.save_pretrained(target_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"<unk>": 0, **{f"v{i}": i for i in range(1, 32)}}, unk_token="<unk>"
            )
        ),
        unk_token="<unk>",
        pad_token="<unk>",
        mask_token="v31",
    )
    tokenizer.save_pretrained(target_path)
    config = Qwen3Config(**fields)
    config.architectures = [model_class.__name__]
    config.num_target_layers, config.block_size = 2, 4
    method = {"target_layer_ids": [0, 1], "mask_token_id": 31}
    if model_class is DLiteDraftModel:
        method.update(
            architecture_version="dlite_v2",
            model_role="student",
            chs_num_layers=2,
            sequential_head="rnn",
            sequential_rank=4,
        )
    if model_class is DFlash2DraftModel:
        method.update(
            conv_kernel_size=2, conv_group_size=4, selector_rank=4, selector_top_k=4
        )
    setattr(config, model_class.method_config_key, method)
    model_class(config).save_pretrained(draft_path)
    args = SimpleNamespace(
        model_name_or_path=str(target_path),
        draft_name_or_path=str(draft_path),
        trust_remote_code=False,
        mask_token_id=31,
        algorithm=model_class.method_config_key.removesuffix("_config"),
    )
    loader = (
        load_dlite_benchmark_models
        if model_class is DLiteDraftModel
        else load_dflash_benchmark_models
    )
    loaded_target, loaded_draft, loaded_tokenizer, *metadata = loader(
        args, torch.device("cpu")
    )
    assert type(loaded_target) is type(target) and type(loaded_draft) is model_class
    assert loaded_tokenizer.mask_token_id == 31
    assert loaded_target.config._attn_implementation == "sdpa" or hybrid
