"""Final-only capture accepts native absent/mirrored auxiliary outputs."""

from types import SimpleNamespace

import pytest
import torch

from litespec.modeling.target.training_target_model import SGLangTrainingTargetModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("pad_token_id", [None, 0])
@pytest.mark.parametrize("mirror_aux", [False, True])
def test_final_only_packed_output(monkeypatch, pad_token_id, mirror_aux):
    target = SGLangTrainingTargetModel(
        SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=12, num_hidden_layers=4)
        )
    )
    target.capture_layer_ids = [3]
    total = 8 if pad_token_id is None else 10
    final = (
        torch.arange(total * 12, device="cuda", dtype=torch.float32)
        .reshape(total, 12)
        .to(torch.bfloat16)
    )

    def extend(requests, **kwargs):
        assert sum(len(request.origin_input_ids) for request in requests) == total
        return (final if mirror_aux else None), final

    monkeypatch.setattr(target, "_extend", extend)
    result = target.generate_capture_data(
        [[1, 2, 3], [4, 5, 6, 7, 8]], pad_token_id=pad_token_id
    )
    assert result.offsets == [0, 3, 8] and result.lengths == [3, 5]
    assert result.layer_ids == [] and result.final_layer_id == 3
    assert result.aux_hidden.shape == (8, 0)
    expected = final if pad_token_id is None else torch.cat([final[:3], final[5:]])
    assert torch.equal(result.final_hidden, expected)
