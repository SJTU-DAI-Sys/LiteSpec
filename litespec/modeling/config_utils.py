"""Helpers for loading text-model configs used by DLite training."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig, Qwen3Config

QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}


def get_pretrained_config_dict(
    pretrained_model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Read raw config JSON even when AutoConfig lacks the model class."""
    config_dict, _ = PretrainedConfig.get_config_dict(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return config_dict


def is_qwen35_model_type(model_type: str | None) -> bool:
    return str(model_type or "") in QWEN35_MODEL_TYPES


def _qwen35_text_dict_to_qwen3_config(
    outer_config: dict[str, Any],
) -> Qwen3Config:
    """Convert Qwen3.5 target metadata to the dense Qwen3 DLite backbone."""
    text_config = outer_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen3.5 config is missing the required text_config object.")

    # Dense Qwen3.5 checkpoints expose ``intermediate_size``.  MoE variants do
    # not have a single dense FFN width, so use the shared expert width for the
    # dense DLite draft (and fall back to an individual routed expert width).
    # This choice affects only the trainable draft backbone; the target keeps
    # its native MoE configuration inside SGLang.
    intermediate_size = text_config.get("intermediate_size")
    if intermediate_size is None:
        intermediate_size = text_config.get("shared_expert_intermediate_size")
    if intermediate_size is None:
        intermediate_size = text_config.get("moe_intermediate_size")

    required_field_names = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    missing_field_names = [name for name in required_field_names if text_config.get(name) is None]
    if intermediate_size is None:
        missing_field_names.append(
            "intermediate_size/shared_expert_intermediate_size/" "moe_intermediate_size"
        )
    if missing_field_names:
        raise ValueError(
            "Qwen3.5 text_config is missing required DLite fields: "
            + ", ".join(missing_field_names)
        )

    rope_parameters = text_config.get("rope_parameters") or {}
    num_hidden_layers = int(text_config["num_hidden_layers"])
    kwargs: dict[str, Any] = {
        "vocab_size": int(text_config["vocab_size"]),
        "hidden_size": int(text_config["hidden_size"]),
        "intermediate_size": int(intermediate_size),
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(text_config["num_attention_heads"]),
        "num_key_value_heads": int(text_config["num_key_value_heads"]),
        "head_dim": int(text_config["head_dim"]),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "max_position_embeddings": int(
            text_config.get("max_position_embeddings", 32768)
        ),
        "initializer_range": float(text_config.get("initializer_range", 0.02)),
        "rms_norm_eps": float(text_config.get("rms_norm_eps", 1e-6)),
        "use_cache": bool(text_config.get("use_cache", True)),
        "tie_word_embeddings": bool(
            outer_config.get(
                "tie_word_embeddings",
                text_config.get("tie_word_embeddings", False),
            )
        ),
        "rope_theta": float(
            text_config.get("rope_theta", rope_parameters.get("rope_theta", 10000.0))
        ),
        "attention_bias": bool(text_config.get("attention_bias", False)),
        "attention_dropout": float(text_config.get("attention_dropout", 0.0)),
        "use_sliding_window": False,
        "layer_types": ["full_attention"] * num_hidden_layers,
    }
    for token_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = text_config.get(token_name, outer_config.get(token_name))
        if value is not None:
            kwargs[token_name] = value

    config = Qwen3Config(**kwargs)
    config.target_source_model_type = str(outer_config.get("model_type", "qwen3_5"))
    config.dlite_target_architectures = list(outer_config.get("architectures", []))
    return config


def load_text_model_config(
    pretrained_model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    """Load target text metadata, normalizing Qwen3.5 for the DLite draft."""
    raw_config = get_pretrained_config_dict(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if is_qwen35_model_type(raw_config.get("model_type")):
        return _qwen35_text_dict_to_qwen3_config(raw_config)

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return getattr(config, "text_config", config)
