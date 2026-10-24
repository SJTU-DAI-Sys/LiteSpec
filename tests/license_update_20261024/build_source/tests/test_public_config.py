import unittest
from unittest import mock

try:
    import torch
    from torch import nn
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"Transformers runtime is unavailable: {exc}") from exc

from litespec.core.dlite import OnlineDLiteModel
from litespec.modeling.draft.dlite import DLiteDraftModel, build_target_layer_ids


class PublicConfigTest(unittest.TestCase):
    def test_student_uses_minimal_dlite_config(self):
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
            "architecture_version": "dlite_v1",
            "model_role": "student",
            "chs_num_layers": 2,
            "target_layer_ids": build_target_layer_ids(4, 2),
            "sequential_head": "rnn",
            "sequential_rank": 8,
            "mask_token_id": 31,
        }
        model = DLiteDraftModel(config)
        self.assertTrue(model.is_student)
        self.assertEqual(model.sequential_head_type, "rnn")
        self.assertEqual(model.draft_query_length, config.block_size)
        self.assertNotIn("anchor_group_size", config.dlite_config)
        self.assertNotIn("sequential_output_mode", config.dlite_config)
        self.assertNotIn("mask_embedding_mode", config.dlite_config)

        embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        online = OnlineDLiteModel(
            draft_model=model,
            target_lm_head=nn.Linear(config.hidden_size, config.vocab_size),
            target_embed_tokens=embed_tokens,
            mask_token_id=31,
            block_size=config.block_size,
        )
        input_ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
        hidden_states = {
            layer_id: torch.randn(1, input_ids.size(1), config.hidden_size)
            for layer_id in model.target_layer_ids
        }
        prepared = online.prepare_batch(
            input_ids,
            hidden_states,
            torch.ones_like(input_ids),
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.tensor([[True]]),
        )
        self.assertEqual(prepared.query_embeddings.shape, (1, 1, 4, 16))
        torch.testing.assert_close(
            prepared.query_embeddings[0, 0, 0], embed_tokens(input_ids[0, 2])
        )
        torch.testing.assert_close(
            prepared.query_embeddings[0, 0, 1:],
            embed_tokens(torch.tensor([31, 31, 31])),
        )
        self.assertEqual(prepared.token_position_ids.tolist(), [[[2]]])
        training_output = online.compute_supervised_loss(
            torch.randn(1, 1, 3, config.hidden_size),
            prepared,
            torch.randn(1, 1, 3, config.vocab_size),
        )
        expected_terms = training_output.ratio_metrics["expected_accept_len"]
        expected_value = expected_terms.numerator / expected_terms.denominator
        self.assertGreaterEqual(expected_value.item(), 1.0)
        self.assertLessEqual(expected_value.item(), config.block_size)
        zero_anchor, zero_keep = online.sample_anchor_positions(
            input_ids.size(1), torch.tensor([[1, 1, 0, 0, 0, 0]])
        )
        self.assertEqual(zero_anchor.tolist(), [[0]])
        self.assertEqual(zero_keep.tolist(), [[True]])

        with mock.patch("litespec.core.dlite.flex_kernel_options", return_value={}):
            flash_online = OnlineDLiteModel(
                draft_model=model,
                target_lm_head=online.lm_head,
                target_embed_tokens=embed_tokens,
                mask_token_id=31,
                block_size=config.block_size,
                flex_attention_backend="FLASH",
            )
        flash_prepared = flash_online.prepare_batch(
            input_ids,
            hidden_states,
            torch.ones_like(input_ids),
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.tensor([[True]]),
        )
        self.assertEqual(flash_prepared.anchor_positions.shape, (1, 64))
        self.assertEqual(int(flash_prepared.block_keep_mask.sum()), 1)
        inference_query = model.build_inference_query_embeddings(
            embed_tokens,
            torch.tensor([[6, 31, 31, 31]]),
        )
        self.assertEqual(inference_query.shape, (1, 4, 16))
        torch.testing.assert_close(inference_query[0, 0], embed_tokens.weight[6])

    def test_v2_and_missing_version_default_to_v2(self):
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
            "architecture_version": "dlite_v2",
            "model_role": "student",
            "chs_num_layers": 2,
            "target_layer_ids": build_target_layer_ids(4, 2),
            "sequential_head": "rnn",
            "sequential_rank": 8,
            "mask_token_id": 31,
        }
        model = DLiteDraftModel(config)
        self.assertEqual(model.architecture_version, "dlite_v2")

        del config.dlite_config["architecture_version"]
        model = DLiteDraftModel(config)
        self.assertEqual(model.architecture_version, "dlite_v2")
        self.assertEqual(config.dlite_config["architecture_version"], "dlite_v2")

    def test_invalid_version_is_rejected(self):
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
            "architecture_version": "unknown",
            "model_role": "student",
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
            "sequential_head": "rnn",
            "sequential_rank": 8,
            "mask_token_id": 31,
        }
        with self.assertRaisesRegex(ValueError, "dlite_v2"):
            DLiteDraftModel(config)


if __name__ == "__main__":
    unittest.main()
