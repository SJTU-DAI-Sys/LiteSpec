"""Public launcher parsing and native HF target loading on Transformers 5."""

from pathlib import Path

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen3_5Config,
    Qwen3Config,
)

from litespec.modeling.config_utils import load_hf_target_model
from litespec.training.algorithms import get_training_adapter
from scripts.config.launch_dflash_family import load_config as load_dflash
from scripts.config.launch_dlite import load_config as load_dlite
from scripts.config.training_launcher import training_base_cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "algorithm,mode",
    [
        ("dflash", None),
        ("dflash2", None),
        ("dspark", None),
        ("dlite", "sft"),
        ("dlite", "teacher"),
        ("dlite", "two_stage"),
    ],
)
def test_public_yaml_and_recipe_parsing(algorithm, mode):
    name = algorithm if mode is None else f"dlite_{mode}"
    path = ROOT / "scripts/config" / f"{name}.yaml"
    env, cli = load_dflash(path, algorithm) if mode is None else load_dlite(path, mode)
    if env.get("TEACHER_DRAFT_PATH"):
        cli += ["--teacher-draft-path", env["TEACHER_DRAFT_PATH"]]
    args = get_training_adapter(algorithm).parse_args(
        mode=mode, argv=training_base_cli(env) + cli + ["--report-to", "none"]
    )
    assert args.target_model_path and args.output_dir


@pytest.mark.parametrize("native_hybrid", [False, True])
def test_native_hf_loading(tmp_path, native_hybrid):
    text = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
    }
    if native_hybrid:
        text.update(
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
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
                "out_hidden_size": 16,
                "patch_size": 2,
            },
        )
        model = AutoModelForImageTextToText.from_config(config)
    else:
        config = Qwen3Config(**text)
        model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(tmp_path)
    restored = load_hf_target_model(
        str(tmp_path), dtype=torch.float32, attn_implementation="eager"
    )
    assert type(restored) is type(model)
    assert len(restored.state_dict()) == len(model.state_dict())
    if native_hybrid:
        assert restored.config.text_config.layer_types == [
            "linear_attention",
            "full_attention",
        ]
    else:
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(restored(tokens).logits, model(tokens).logits)
