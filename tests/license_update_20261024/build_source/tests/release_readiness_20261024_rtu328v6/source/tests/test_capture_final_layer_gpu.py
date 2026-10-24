"""Final-only capture accepts native absent/mirrored auxiliary outputs."""

from types import SimpleNamespace

import pytest
import torch

from litespec.modeling.target.training_target_model import SGLangTrainingTargetModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mirror_aux", [False, True])
def test_final_only_packed_output(monkeypatch, mirror_aux):
    target = SGLangTrainingTargetModel(
        SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=12, num_hidden_layers=4)
        )
    )
    target.capture_layer_ids = [3]
    total = 8
    final = (
        torch.arange(total * 12, device="cuda", dtype=torch.float32)
        .reshape(total, 12)
        .to(torch.bfloat16)
    )

    def extend(requests, **kwargs):
        assert [request.origin_input_ids for request in requests] == [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
        ]
        assert [request.extend_input_len for request in requests] == [3, 5]
        return (final if mirror_aux else None), final

    monkeypatch.setattr(target, "_extend", extend)
    result = target.generate_capture_data([[1, 2, 3], [4, 5, 6, 7, 8]])
    assert result.offsets == [0, 3, 8] and result.lengths == [3, 5]
    assert result.layer_ids == [] and result.final_layer_id == 3
    assert result.aux_hidden.shape == (8, 0)
    assert torch.equal(result.final_hidden, final)
