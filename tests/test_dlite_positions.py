from types import SimpleNamespace

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel


@pytest.mark.parametrize("anchor", [0, 1, 5, 100])
@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("architecture_version", ["dlite_v1", "dlite_v2"])
def test_training_and_inference_rope_positions(anchor, teacher, architecture_version):
    model = SimpleNamespace(
        architecture_version=architecture_version,
        is_teacher=teacher,
        chs_num_layers=2,
        condition_slot_count=2,
        block_size=4,
        token_prefix_count=1,
        swa_window_size=1,
        fuse_slot_count=0,
    )
    model.build_block_position_ids = (
        lambda **kwargs: DLiteDraftModel.build_block_position_ids(model, **kwargs)
    )
    anchors = torch.tensor([[anchor], [anchor]])
    context, queries = model.build_block_position_ids(
        anchor_positions=anchors,
        token_position_ids=anchors.unsqueeze(-1),
        token_keep_mask=torch.ones(2, 1, 1, dtype=torch.bool),
    )
    local_offset = 0 if architecture_version == "dlite_v1" else 1
    expected_queries = (
        list(range(anchor, anchor + 4))
        if teacher
        else list(range(local_offset, local_offset + 4))
    )
    expected_context = [max(anchor - 1, 0)] * 2 if teacher else [0, 0]
    assert queries.tolist() == [expected_queries] * 2
    assert context.tolist() == [expected_context] * 2

    # A teacher needs a preceding pivot; student local positions also cover 0.
    if teacher and anchor == 0:
        return
    _, inference_context, inference_queries = DLiteDraftModel.build_inference_context(
        model,
        recent_condition_hidden=torch.zeros(2, 1, 3),
        current_target_hidden=torch.zeros(2, 1, 2, 3),
        anchor_position=anchor,
    )
    torch.testing.assert_close(inference_context, context)
    torch.testing.assert_close(inference_queries, queries)
