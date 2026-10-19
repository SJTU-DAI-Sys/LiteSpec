# coding=utf-8
"""Training primitives for the current SWA-teacher/PivotQ-student architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from litespec.core.attention_masks import (
    create_dlite_flex_mask,
    flash_kv_padding,
    flash_padded_block_width,
)
from litespec.core.block_metrics import (
    block_accepted_length_terms,
    distribution_acceptance_probability,
    expected_accept_len_terms,
    fsdp_global_weighted_mean,
)
from litespec.core.block_training import (
    apply_exponential_position_decay,
    sample_sorted_anchors,
)
from litespec.core.losses import weighted_ce_l1, weighted_cross_entropy, weighted_kl
from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.modeling.draft.flex_attention import flex_kernel_options

HiddenStatesInput = Union[
    tuple[torch.Tensor, ...], list[torch.Tensor], Dict[int, torch.Tensor]
]


def infer_hidden_states_embedding_offset(
    hidden_states: tuple | list, num_transformer_layers: int
) -> int:
    if len(hidden_states) == num_transformer_layers:
        return 0
    if len(hidden_states) == num_transformer_layers + 1:
        return 1
    return 1 if len(hidden_states) > num_transformer_layers else 0


def _hidden_at_layer(
    hidden_states: HiddenStatesInput,
    layer_id: int,
    num_transformer_layers: int,
) -> torch.Tensor:
    if isinstance(hidden_states, dict):
        return hidden_states[layer_id]
    offset = infer_hidden_states_embedding_offset(hidden_states, num_transformer_layers)
    return hidden_states[layer_id + offset]


def prepare_target_hidden(
    hidden_states: HiddenStatesInput,
    anchor_positions: torch.Tensor,
    target_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Gather CHS at ``anchor-1`` as ``(batch, anchors, layers, hidden)``."""
    positions = (anchor_positions - 1).clamp(min=0)
    pieces = []
    for layer_id in target_layer_ids:
        layer_hidden = _hidden_at_layer(hidden_states, layer_id, num_transformer_layers)
        pieces.append(
            torch.gather(
                layer_hidden,
                1,
                positions.unsqueeze(-1).expand(-1, -1, layer_hidden.size(-1)),
            )
        )
    return torch.stack(pieces, dim=2)


def prepare_history_hidden_states(
    hidden_states: HiddenStatesInput,
    history_layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Stack first/middle/last target layers as ``(batch, seq, 3, hidden)``."""
    return torch.stack(
        [
            _hidden_at_layer(hidden_states, layer_id, num_transformer_layers)
            for layer_id in history_layer_ids
        ],
        dim=2,
    )


def gather_target_prefill_logits(
    target_logits: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather causal logits at ``a..a+B-2`` for labels ``a+1..a+B-1``."""
    offsets = torch.arange(int(block_size) - 1, device=anchor_positions.device).view(
        1, 1, -1
    )
    positions = anchor_positions.unsqueeze(-1) + offsets
    if bool((positions >= target_logits.size(1)).any()):
        raise ValueError("Target prefill logits do not cover all prediction positions.")
    expanded = target_logits.unsqueeze(1).expand(-1, anchor_positions.size(1), -1, -1)
    return torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, target_logits.size(-1)),
    )


@dataclass
class PreparedDLiteBatch:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    target_hidden: torch.Tensor
    shared_fused_history: Optional[torch.Tensor]
    query_embeddings: torch.Tensor
    token_keep_mask: torch.Tensor
    token_position_ids: torch.Tensor
    labels: torch.Tensor
    prev_token_ids: torch.Tensor
    raw_weight_mask: torch.Tensor
    binary_eval_mask: torch.Tensor


class OnlineDLiteModel(nn.Module):
    """One-pass backbone/logits/loss wrapper for the current architecture."""

    def __init__(
        self,
        draft_model: DLiteDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int,
        attention_backend: str = "flex_attention",
        flex_attention_backend: str = "TRITON",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        final_ce_weight: float = 1.0,
        l1_loss_weight: float = 1.0,
        base_lm_ce_weight: float = 0.0,
        base_lm_ce_decay_gamma: Optional[float] = None,
        use_target_greedy_ce_labels: bool = False,
        process_group=None,
    ) -> None:
        super().__init__()
        if attention_backend != "flex_attention":
            raise ValueError("The current architecture supports flex_attention only.")
        flex_attention_backend = str(flex_attention_backend).upper()
        if flex_attention_backend not in {"TRITON", "FLASH"}:
            raise ValueError("flex_attention_backend must be TRITON or FLASH")
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = int(mask_token_id)
        self.block_size = int(block_size)
        if self.block_size <= 1:
            raise ValueError(
                f"block_size must be at least 2 for next-token loss, got {block_size}."
            )
        self.attention_backend = attention_backend
        self.flex_attention_backend = flex_attention_backend
        self.flex_kernel_options = flex_kernel_options(flex_attention_backend)
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = loss_decay_gamma
        self.final_ce_weight = float(final_ce_weight)
        self.l1_loss_weight = float(l1_loss_weight)
        self.base_lm_ce_weight = float(base_lm_ce_weight)
        for name, value in (
            ("final_ce_weight", self.final_ce_weight),
            ("l1_loss_weight", self.l1_loss_weight),
            ("base_lm_ce_weight", self.base_lm_ce_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.final_ce_weight + self.l1_loss_weight + self.base_lm_ce_weight == 0:
            raise ValueError("At least one DLite loss weight must be positive.")
        self.base_lm_ce_decay_gamma = base_lm_ce_decay_gamma
        self.use_target_greedy_ce_labels = bool(use_target_greedy_ce_labels)
        self.process_group = process_group

    def sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_anchor = max(seq_len - self.block_size, 0)
        valid_anchor_mask = loss_mask[:, : max_anchor + 1] > 0.5
        valid_anchor_mask = valid_anchor_mask & (loss_mask[:, 1 : max_anchor + 2] > 0.5)
        counts = valid_anchor_mask.sum(dim=1)
        sampled_width = min(self.num_anchors, int(counts.max().item()))
        if sampled_width <= 0:
            raise ValueError("No valid anchor positions in this batch.")
        query_len = int(self.draft_model.draft_query_length)
        chs_slots = int(self.draft_model.chs_num_layers)
        output_width = (
            flash_padded_block_width(sampled_width, query_len, chs_slots)
            if self.flex_attention_backend == "FLASH"
            else sampled_width
        )
        return sample_sorted_anchors(
            valid_anchor_mask,
            self.num_anchors,
            output_width=output_width,
            max_valid_anchors=int(counts.max().item()),
        )

    def _build_labels_and_weights(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        offsets = torch.arange(self.block_size, device=input_ids.device).view(1, 1, -1)
        block_positions = anchor_positions.unsqueeze(-1) + offsets
        valid = block_positions < seq_len
        gather_positions = block_positions.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            gather_positions,
        )
        gathered_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            gather_positions,
        )
        weights = (
            block_keep_mask.unsqueeze(-1).float() * valid.float() * gathered_loss_mask
        )[:, :, 1:]
        return target_ids[:, :, 1:], target_ids[:, :, :-1], weights, weights > 0

    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[HiddenStatesInput],
        loss_mask: torch.Tensor,
        *,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        shared_query_embeddings: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        raw_history_hidden: Optional[torch.Tensor] = None,
    ) -> PreparedDLiteBatch:
        if anchor_positions is None or block_keep_mask is None:
            anchor_positions, block_keep_mask = self.sample_anchor_positions(
                input_ids.size(1), loss_mask
            )
        elif self.flex_attention_backend == "FLASH":
            natural_width = anchor_positions.size(1)
            padded_width = flash_padded_block_width(
                natural_width,
                self.draft_model.draft_query_length,
                self.draft_model.chs_num_layers,
            )
            pad = padded_width - natural_width
            if pad:
                anchor_positions = F.pad(anchor_positions, (0, pad))
                block_keep_mask = F.pad(block_keep_mask, (0, pad), value=False)
                if target_hidden is not None:
                    target_hidden = F.pad(target_hidden, (0, 0, 0, 0, 0, pad))
                if shared_query_embeddings is not None:
                    shared_query_embeddings = F.pad(
                        shared_query_embeddings, (0, 0, 0, 0, 0, pad)
                    )
        if target_hidden is None:
            if hidden_states is None:
                raise ValueError("hidden_states or prepared target_hidden is required")
            target_hidden = prepare_target_hidden(
                hidden_states,
                anchor_positions,
                self.draft_model.target_layer_ids,
                self.draft_model.config.num_target_layers,
            )
        prefix_count = self.draft_model.token_prefix_count
        prefix_offsets = torch.arange(
            1 - prefix_count,
            1,
            device=anchor_positions.device,
        ).view(1, 1, -1)
        token_positions = (anchor_positions.unsqueeze(-1) + prefix_offsets).clamp(min=0)
        token_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            token_positions,
        )
        token_keep = block_keep_mask.unsqueeze(-1).expand(-1, -1, prefix_count)
        if shared_query_embeddings is None:
            token_embeddings = self.embed_tokens(token_ids)
            token_embeddings = token_embeddings * token_keep.unsqueeze(-1).to(
                token_embeddings.dtype
            )
            mask_ids = torch.full(
                (*token_ids.shape[:2], self.block_size - 1),
                self.mask_token_id,
                device=input_ids.device,
                dtype=torch.long,
            )
            query_embeddings = torch.cat(
                [token_embeddings, self.embed_tokens(mask_ids)], dim=2
            )
        else:
            query_embeddings = shared_query_embeddings
        shared_fused_history = None
        if self.draft_model.is_teacher:
            if raw_history_hidden is None:
                if hidden_states is None:
                    raise ValueError(
                        "teacher requires hidden_states or raw_history_hidden"
                    )
                raw_history = prepare_history_hidden_states(
                    hidden_states,
                    self.draft_model.history_layer_ids,
                    self.draft_model.config.num_target_layers,
                )
            else:
                raw_history = raw_history_hidden
            shared_fused_history = self.draft_model.fuse_history_hidden(raw_history)
        labels, prev_ids, weights, binary = self._build_labels_and_weights(
            input_ids, loss_mask, anchor_positions, block_keep_mask
        )
        return PreparedDLiteBatch(
            anchor_positions,
            block_keep_mask,
            target_hidden,
            shared_fused_history,
            query_embeddings,
            token_keep,
            token_positions,
            labels,
            prev_ids,
            weights,
            binary,
        )

    def forward_backbone(
        self, batch: PreparedDLiteBatch, *, seq_len: int
    ) -> torch.Tensor:
        bsz, num_blocks = batch.anchor_positions.shape
        query_len = self.draft_model.draft_query_length
        query_embeddings = batch.query_embeddings.reshape(
            bsz, num_blocks * query_len, -1
        )
        context_pos, draft_pos = self.draft_model.build_block_position_ids(
            batch.anchor_positions, batch.token_position_ids, batch.token_keep_mask
        )
        if self.draft_model.is_teacher:
            if batch.shared_fused_history is None:
                raise ValueError("Teacher requires shared fused history.")
            history = batch.shared_fused_history
            mask_seq_len = int(seq_len)
            natural_kv = (
                mask_seq_len
                + num_blocks * self.draft_model.chs_num_layers
                + num_blocks * query_len
            )
            history_pad = (
                flash_kv_padding(natural_kv)
                if self.flex_attention_backend == "FLASH"
                else 0
            )
            if history_pad:
                history = F.pad(history, (0, 0, 0, history_pad))
                mask_seq_len += history_pad
            attention_mask = create_dlite_flex_mask(
                model_role=self.draft_model.model_role,
                anchor_positions=batch.anchor_positions,
                block_keep_mask=batch.block_keep_mask,
                token_keep_mask=batch.token_keep_mask,
                seq_len=mask_seq_len,
                swa_window_size=self.draft_model.swa_window_size,
                chs_slots=self.draft_model.chs_num_layers,
                query_len=query_len,
                device=query_embeddings.device,
                flex_backend=self.flex_attention_backend,
            )
            shared_pos_ids = (
                torch.arange(mask_seq_len, device=query_embeddings.device)
                .view(1, -1)
                .expand(bsz, -1)
            )
            # The logical per-block context is [W-1 fuse, S CHS], but the fuse
            # sequence is physically stored once and selected per anchor by the
            # flex mask. Its RoPE ids are therefore ``shared_pos_ids`` exactly once.
            logical_context = context_pos.view(
                bsz, num_blocks, self.draft_model.chs_len_per_block
            )
            chs_pos_ids = logical_context[:, :, -self.draft_model.chs_num_layers :]
            rotary = torch.cat(
                [shared_pos_ids, chs_pos_ids.reshape(bsz, -1), draft_pos], dim=-1
            )
            output = self.draft_model(
                position_ids=draft_pos,
                noise_embedding=query_embeddings,
                target_hidden=batch.target_hidden,
                shared_history=history,
                attention_mask=attention_mask,
                rotary_position_ids=rotary,
                kernel_options=self.flex_kernel_options,
            )
        else:
            attention_mask = create_dlite_flex_mask(
                model_role=self.draft_model.model_role,
                anchor_positions=batch.anchor_positions,
                block_keep_mask=batch.block_keep_mask,
                token_keep_mask=batch.token_keep_mask,
                seq_len=seq_len,
                swa_window_size=self.draft_model.swa_window_size,
                chs_slots=self.draft_model.chs_num_layers,
                query_len=query_len,
                device=query_embeddings.device,
                flex_backend=self.flex_attention_backend,
            )
            empty_history = query_embeddings.new_empty(
                bsz, num_blocks, 0, query_embeddings.size(-1)
            )
            rotary = torch.cat([context_pos, draft_pos], dim=-1)
            output = self.draft_model(
                position_ids=draft_pos,
                noise_embedding=query_embeddings,
                target_hidden=batch.target_hidden,
                history_hidden=empty_history,
                attention_mask=attention_mask,
                rotary_position_ids=rotary,
                kernel_options=self.flex_kernel_options,
            )
        output = output.view(bsz, num_blocks, query_len, output.size(-1))
        return output[:, :, -self.draft_model.proposal_length :, :]

    @staticmethod
    def _position_weights(raw: torch.Tensor, gamma: Optional[float]) -> torch.Tensor:
        positions = torch.arange(1, raw.size(-1) + 1, device=raw.device)
        return apply_exponential_position_decay(
            raw, gamma, positions, first_prediction_index=1
        )

    def compute_supervised_loss(
        self,
        prediction_hidden: torch.Tensor,
        batch: PreparedDLiteBatch,
        target_prefill_logits: torch.Tensor,
        *,
        serial_logits: Optional[torch.Tensor] = None,
        collect_detailed_metrics: bool = True,
    ) -> TrainingOutput:
        """Compute supervised loss and its full-vocabulary projections.

        With ``use_target_greedy_ce_labels``, CE uses the target model's greedy
        prefill tokens.  Independently, the serial head always updates its RNN
        state with ground-truth previous tokens from the training sequence
        (``batch.prev_token_ids``).
        """
        active_positions = batch.binary_eval_mask.bool()
        if not active_positions.any():
            raise ValueError("DLite loss has no supervised label positions.")

        # Backend attention can return arbitrary values for padded query rows.
        # Clear them before the serial head; masking the eventual scalar loss is
        # insufficient because NaN * 0 remains NaN.
        prediction_hidden = torch.where(
            active_positions.unsqueeze(-1),
            prediction_hidden,
            torch.zeros_like(prediction_hidden),
        )
        prev_token_ids = torch.where(
            active_positions,
            batch.prev_token_ids,
            torch.zeros_like(batch.prev_token_ids),
        )
        final_weights = self._position_weights(
            batch.raw_weight_mask, self.loss_decay_gamma
        )
        base_weights = self._position_weights(
            batch.raw_weight_mask, self.base_lm_ce_decay_gamma
        )
        active_hidden = prediction_hidden[active_positions]
        active_target_logits = target_prefill_logits[active_positions]
        active_ce_labels = (
            active_target_logits.argmax(dim=-1)
            if self.use_target_greedy_ce_labels
            else batch.labels[active_positions]
        )
        active_final_weights = final_weights[active_positions].float()
        active_base_weights = base_weights[active_positions].float()
        sequential_head = self.draft_model.sequential_head
        use_base_logits = self.base_lm_ce_weight > 0
        base_logits = self.lm_head(active_hidden).float() if use_base_logits else None
        if serial_logits is None:
            latent = sequential_head.forward_teacher_forcing(
                hidden_states=prediction_hidden,
                prev_token_ids=prev_token_ids,
            )
            final_logits = sequential_head.project_logits(
                latent[active_positions]
            ).float()
        else:
            # The transition objective shares this graph with Stage-1 KL.
            final_logits = serial_logits
        if (active_ce_labels < 0).any() or (
            active_ce_labels >= final_logits.size(-1)
        ).any():
            label_source = (
                "Target-prefill greedy labels"
                if self.use_target_greedy_ce_labels
                else "Supervised labels"
            )
            raise ValueError(
                f"{label_source} must be within the output vocabulary: "
                f"min={int(active_ce_labels.min().item())}, "
                f"max={int(active_ce_labels.max().item())}, "
                f"vocab_size={final_logits.size(-1)}."
            )
        final_denominator = active_final_weights.sum()
        if active_target_logits.size(-1) < final_logits.size(-1):
            active_target_logits = F.pad(
                active_target_logits,
                (0, final_logits.size(-1) - active_target_logits.size(-1)),
                value=torch.finfo(active_target_logits.dtype).min,
            )
        elif active_target_logits.size(-1) != final_logits.size(-1):
            raise ValueError("Target and draft vocab sizes do not match.")
        if self.base_lm_ce_weight > 0:
            assert base_logits is not None
            base_ce_numerator, _ = weighted_cross_entropy(
                base_logits.float(), active_ce_labels, active_base_weights
            )
            base_denominator = active_base_weights.sum()
        else:
            base_ce_numerator = prediction_hidden.new_zeros((), dtype=torch.float32)
            base_denominator = final_denominator
        if self.l1_loss_weight > 0:
            final_ce_numerator, l1_numerator, _, l1_values = weighted_ce_l1(
                final_logits.float(),
                active_ce_labels,
                active_target_logits,
                active_final_weights,
            )
            if collect_detailed_metrics:
                active_acceptance = (1.0 - 0.5 * l1_values).clamp(0.0, 1.0)
        else:
            final_ce_numerator, _ = weighted_cross_entropy(
                final_logits.float(), active_ce_labels, active_final_weights
            )
            l1_numerator = prediction_hidden.new_zeros((), dtype=torch.float32)
            if collect_detailed_metrics:
                active_acceptance = distribution_acceptance_probability(
                    active_target_logits, final_logits
                )
        # The optimizer-window objective needs one additive denominator so
        # FSDP's rank-averaged gradients can be rescaled after accumulation.
        # When final/base decay schedules differ, use the loss-coefficient-
        # weighted mean of their effective-weight sums. This equals the shared
        # denominator exactly in the common case, preserves the absolute scale
        # of a lone loss coefficient, and remains additive across ranks/windows.
        final_objective_weight = self.final_ce_weight + self.l1_loss_weight
        objective_weight_sum = final_objective_weight + self.base_lm_ce_weight
        loss_denominator = (
            final_objective_weight * final_denominator
            + self.base_lm_ce_weight * base_denominator
        ) / objective_weight_sum
        loss_numerator = (
            self.final_ce_weight * final_ce_numerator
            + self.l1_loss_weight * l1_numerator
            + self.base_lm_ce_weight * base_ce_numerator
        )
        total = fsdp_global_weighted_mean(
            loss_numerator, loss_denominator, self.process_group
        )
        if not collect_detailed_metrics:
            return TrainingOutput(
                loss=total,
                loss_terms=RatioTerms(loss_numerator, loss_denominator.detach()),
            )
        with torch.no_grad():
            predictions = torch.zeros_like(batch.labels)
            predictions[active_positions] = final_logits.argmax(dim=-1)
            ce_labels = torch.zeros_like(batch.labels)
            ce_labels[active_positions] = active_ce_labels
            valid = batch.binary_eval_mask
            accepted = block_accepted_length_terms(predictions, ce_labels, valid)
            acceptance = torch.zeros_like(batch.raw_weight_mask, dtype=torch.float32)
            acceptance[active_positions] = active_acceptance
            expected_accept_len = expected_accept_len_terms(acceptance, valid)
        ratios = {
            "hard_label/token_accuracy": RatioTerms(
                ((predictions == ce_labels) & valid).sum().detach(),
                valid.sum().detach(),
            ),
            "hard_label/block_accepted_length": accepted.detached(),
            "expected_accept_len": expected_accept_len.detached(),
            "objective/final_ce": RatioTerms(
                final_ce_numerator.detach(), final_denominator.detach()
            ),
        }
        if self.base_lm_ce_weight > 0:
            ratios["objective/base_ce"] = RatioTerms(
                base_ce_numerator.detach(), base_denominator.detach()
            )
        if self.l1_loss_weight > 0:
            ratios["objective/l1_distance"] = RatioTerms(
                l1_numerator.detach(), final_denominator.detach()
            )
        return TrainingOutput(
            loss=total,
            loss_terms=RatioTerms(loss_numerator, loss_denominator.detach()),
            ratio_metrics=ratios,
        )

    def compute_serial_logits(
        self,
        prediction_hidden: torch.Tensor,
        batch: PreparedDLiteBatch,
    ) -> torch.Tensor:
        """Return active final logits produced through the serial head.

        Stage 1 uses this path for teacher/student distribution distillation.
        Keeping it on ``OnlineDLiteModel`` also ensures the student's frozen
        serial-head parameters are accessed from inside ``FSDP.forward``.
        """
        active_positions = batch.binary_eval_mask.bool()
        if not active_positions.any():
            raise ValueError("Serial-head distillation has no supervised positions.")

        prediction_hidden = torch.where(
            active_positions.unsqueeze(-1),
            prediction_hidden,
            torch.zeros_like(prediction_hidden),
        )
        prev_token_ids = torch.where(
            active_positions,
            batch.prev_token_ids,
            torch.zeros_like(batch.prev_token_ids),
        )
        sequential_head = self.draft_model.sequential_head
        latent = sequential_head.forward_teacher_forcing(
            hidden_states=prediction_hidden,
            prev_token_ids=prev_token_ids,
        )
        serial_logits = sequential_head.project_logits(latent[active_positions]).float()
        return serial_logits

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[HiddenStatesInput] = None,
        target_hidden: Optional[torch.Tensor] = None,
        raw_history_hidden: Optional[torch.Tensor] = None,
        target_prefill_logits: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        prepared_batch: Optional[PreparedDLiteBatch] = None,
        seq_len: Optional[int] = None,
        return_backbone: bool = False,
        return_backbone_and_serial_logits: bool = False,
        return_transition_outputs: bool = False,
        target_logits_are_gathered: bool = False,
        collect_detailed_metrics: bool = True,
    ):
        if prepared_batch is None:
            if input_ids is None or loss_mask is None:
                raise ValueError("input_ids and loss_mask are required")
            prepared_batch = self.prepare_batch(
                input_ids,
                hidden_states,
                loss_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                target_hidden=target_hidden,
                raw_history_hidden=raw_history_hidden,
            )
            seq_len = input_ids.size(1)
        if seq_len is None:
            raise ValueError("seq_len is required with prepared_batch")
        prediction_hidden = self.forward_backbone(prepared_batch, seq_len=seq_len)
        if return_transition_outputs:
            if target_prefill_logits is None:
                raise ValueError(
                    "target_prefill_logits are required for transition loss"
                )
            gathered_target_logits = (
                target_prefill_logits
                if target_logits_are_gathered
                else gather_target_prefill_logits(
                    target_prefill_logits,
                    prepared_batch.anchor_positions,
                    self.block_size,
                )
            )
            student_serial_logits = self.compute_serial_logits(
                prediction_hidden, prepared_batch
            )
            supervised = self.compute_supervised_loss(
                prediction_hidden,
                prepared_batch,
                gathered_target_logits,
                serial_logits=student_serial_logits,
                collect_detailed_metrics=collect_detailed_metrics,
            )
            return (
                prediction_hidden,
                student_serial_logits,
                supervised,
            )
        if return_backbone_and_serial_logits:
            return prediction_hidden, self.compute_serial_logits(
                prediction_hidden, prepared_batch
            )
        if return_backbone:
            return prediction_hidden
        if target_prefill_logits is None:
            raise ValueError("target_prefill_logits are required for supervised loss")
        gathered_target_logits = (
            target_prefill_logits
            if target_logits_are_gathered
            else gather_target_prefill_logits(
                target_prefill_logits,
                prepared_batch.anchor_positions,
                self.block_size,
            )
        )
        return self.compute_supervised_loss(
            prediction_hidden,
            prepared_batch,
            gathered_target_logits,
            collect_detailed_metrics=collect_detailed_metrics,
        )


def compute_stage1_distillation_loss(
    *,
    student_serial_logits: torch.Tensor,
    teacher_serial_logits: torch.Tensor,
    raw_weight_mask: torch.Tensor,
    kl_weight: float,
    loss_decay_gamma: Optional[float],
    process_group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Stage-1 metrics plus an additive numerator/denominator pair."""
    offsets = torch.arange(raw_weight_mask.size(-1), device=raw_weight_mask.device)
    decay = (
        torch.ones_like(offsets, dtype=torch.float32)
        if loss_decay_gamma is None or loss_decay_gamma <= 0
        else torch.exp(-offsets.float() / float(loss_decay_gamma))
    )
    weights = raw_weight_mask * decay.view(1, 1, -1)
    active_positions = weights > 0
    if not active_positions.any():
        raise ValueError("Stage-1 distillation has no supervised positions.")
    active_count = int(active_positions.sum().item())
    if student_serial_logits.ndim != 2 or student_serial_logits.size(0) != active_count:
        raise ValueError(
            "Student serial logits must contain one row per active Stage-1 "
            f"position; got {tuple(student_serial_logits.shape)} for {active_count} positions."
        )
    if teacher_serial_logits.shape != student_serial_logits.shape:
        raise ValueError(
            "Teacher/student serial-logit shapes must match, got "
            f"{tuple(teacher_serial_logits.shape)} and "
            f"{tuple(student_serial_logits.shape)}."
        )
    active_weights = weights[active_positions].float()
    denominator = active_weights.sum()
    kl_numerator = weighted_kl(
        student_serial_logits, teacher_serial_logits, active_weights
    )
    kl_loss = fsdp_global_weighted_mean(kl_numerator, denominator, process_group)
    loss_numerator = float(kl_weight) * kl_numerator
    return float(kl_weight) * kl_loss, kl_loss, loss_numerator, denominator.detach()


__all__ = [
    "HiddenStatesInput",
    "OnlineDLiteModel",
    "PreparedDLiteBatch",
    "compute_stage1_distillation_loss",
    "gather_target_prefill_logits",
    "prepare_history_hidden_states",
    "prepare_target_hidden",
]
