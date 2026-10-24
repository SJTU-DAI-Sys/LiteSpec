from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from litespec.modeling.draft.dlite import (
    DLiteDraftModel,
    _begin_recurrent_cache_transaction,
    _crop_target_cache,
    _restore_recurrent_cache_transaction,
)
from litespec.modeling.draft.registry import draft_source_files
from litespec.training.checkpoint import export_draft
from litespec.training.recipes.dlite import build_draft_config


def make_config(version="dlite_v2", role="student"):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    config.num_target_layers = 4
    config.block_size = 4
    config.dlite_config = {
        "model_role": role,
        "chs_num_layers": 2,
        "target_layer_ids": [0, 3],
        "sequential_head": "rnn",
        "sequential_rank": 8,
        "mask_token_id": 31,
    }
    if version is not None:
        config.dlite_config["architecture_version"] = version
    return config


@pytest.mark.parametrize(
    ("version", "expected"),
    [("dlite_v1", [0, 1, 2, 3]), ("dlite_v2", [1, 2, 3, 4]), (None, [1, 2, 3, 4])],
)
def test_versioned_student_positions(version, expected):
    model = DLiteDraftModel(make_config(version))
    anchor = torch.tensor([[7]])
    context, queries = model.build_block_position_ids(
        anchor_positions=anchor,
        token_position_ids=anchor.unsqueeze(-1),
        token_keep_mask=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    assert context.tolist() == [[0, 0]]
    assert queries.tolist() == [expected]


def test_versions_share_identical_state_dict_layout():
    v1 = DLiteDraftModel(make_config("dlite_v1"))
    v2 = DLiteDraftModel(make_config("dlite_v2"))
    assert list(v1.state_dict()) == list(v2.state_dict())
    assert {key: tuple(value.shape) for key, value in v1.state_dict().items()} == {
        key: tuple(value.shape) for key, value in v2.state_dict().items()
    }


def test_fresh_training_config_defaults_to_v2():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    config.num_target_layers = 4
    config.block_size = 4
    args = SimpleNamespace(
        target_layer_ids="0,3",
        chs_num_layers=2,
        sequential_head="rnn",
        sequential_rank=8,
        mask_token_id=31,
        attention_backend="flex_attention",
    )
    result = build_draft_config(args, model_role="student", source_config=config)
    assert result.dlite_config["architecture_version"] == "dlite_v2"


def test_export_has_auto_map_and_complete_code_closure(tmp_path):
    model = DLiteDraftModel(make_config("dlite_v2"))
    state_before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    export_draft(model, tmp_path)

    assert model.config.auto_map == {"AutoModel": "dlite.DLiteDraftModel"}
    assert (tmp_path / "model.safetensors").is_file()
    for filename in draft_source_files("DLiteDraftModel"):
        assert (tmp_path / filename).is_file()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])


def test_old_role_and_invalid_layers_are_rejected():
    with pytest.raises(ValueError, match="student"):
        DLiteDraftModel(make_config(role="pivot_q_student"))
    config = make_config()
    config.dlite_config["target_layer_ids"] = [3, 0]
    with pytest.raises(ValueError, match="strictly increasing"):
        DLiteDraftModel(config)


class NewCropCache:
    is_croppable = True

    def __init__(self):
        self.calls = []

    def crop(self, value):
        self.calls.append(value)


class OldCropCache:
    def __init__(self):
        self.calls = []

    def crop(self, value):
        self.calls.append(value)


def test_crop_api_compatibility():
    new = NewCropCache()
    old = OldCropCache()
    _crop_target_cache(new, current_length=20, tokens_to_remove=3)
    _crop_target_cache(old, current_length=20, tokens_to_remove=3)
    assert new.calls == [-3]
    assert old.calls == [17]


def test_recurrent_cache_transaction_restores_state():
    layer = SimpleNamespace(
        recurrent_states={0: torch.tensor([1.0, 2.0])},
        is_recurrent_states_initialized={0: True},
    )

    class Cache:
        layers = [layer]

        def activate_past_recording(self):
            self.activated = True

    cache = Cache()
    snapshot = _begin_recurrent_cache_transaction(cache)
    assert cache.activated
    layer.recurrent_states[0].add_(10)
    _restore_recurrent_cache_transaction(snapshot)
    torch.testing.assert_close(layer.recurrent_states[0], torch.tensor([1.0, 2.0]))
