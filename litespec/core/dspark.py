"""DSpark Markov and confidence training objectives."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from litespec.core.attention_masks import FLEX_ATTENTION_AVAILABLE
from litespec.core.block_metrics import (
    block_accepted_length_terms,
    expected_accept_len_terms,
)
from litespec.core.chunking import checkpointed_chunk_reduce, is_chunk_recomputing
from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.core.losses import (
    weighted_cross_entropy,
    weighted_ce_l1,
    weighted_binary_cross_entropy,
)
from litespec.modeling.draft.dflash import DFlashDraftModel

from ._dflash_common import DFlashTrainingBase


class OnlineDSparkModel(DFlashTrainingBase):
    """DSpark online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        flex_attention_backend: str = "TRITON",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dspark_ce_loss_alpha: float = 0.1,
        dspark_l1_loss_alpha: float = 0.9,
        dspark_confidence_head_alpha: float = 1.0,
        objective_chunk_blocks: int = 128,
        process_group=None,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            flex_attention_backend=flex_attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            process_group=process_group,
        )
        if dspark_ce_loss_alpha < 0:
            raise ValueError("dspark_ce_loss_alpha must be >= 0")
        if dspark_l1_loss_alpha < 0:
            raise ValueError("dspark_l1_loss_alpha must be >= 0")
        if dspark_confidence_head_alpha < 0:
            raise ValueError("dspark_confidence_head_alpha must be >= 0")
        self.dspark_ce_loss_alpha = float(dspark_ce_loss_alpha)
        self.dspark_l1_loss_alpha = float(dspark_l1_loss_alpha)
        self.dspark_confidence_head_alpha = float(dspark_confidence_head_alpha)

    def _build_dspark_labels_and_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        # Anchor-inclusive layout shared with DFlash/DLite. Output slot zero is
        # the clean anchor and is intentionally never supervised.
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        target_valid = label_indices < seq_len
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
            2,
            safe_label_indices,
        )
        suffix_mask = target_valid[..., 1:] & (target_loss_mask[..., 1:] > 0.5)
        suffix_mask = suffix_mask & block_keep_mask.unsqueeze(-1)
        suffix_mask = suffix_mask.to(torch.int32).cumprod(dim=-1).bool()
        eval_mask = torch.cat(
            [torch.zeros_like(suffix_mask[..., :1]), suffix_mask], dim=-1
        )
        return target_ids, eval_mask, safe_label_indices

    def _dspark_loss_weight_mask(
        self,
        eval_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight_mask = eval_mask.to(torch.float32)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=eval_mask.device).view(
                1, 1, -1
            )
            # Position one is DSpark's first supervised prediction. Keep it at
            # unit weight, matching the pre-alignment objective's first slot.
            decay_positions = (positions - 1).clamp_min(0)
            decay_weights = torch.exp(
                -decay_positions.float() / float(self.loss_decay_gamma)
            )
            loss_weight_mask = loss_weight_mask * decay_weights
        return loss_weight_mask

    def _aligned_target_hidden(
        self,
        target_last_hidden_states: torch.Tensor,
        safe_label_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the target state that predicts each DSpark label token."""

        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        bsz = target_last_hidden_states.shape[0]
        hidden_size = target_last_hidden_states.shape[-1]
        gather_indices = target_pred_indices.reshape(bsz, -1, 1).expand(
            -1, -1, hidden_size
        )
        return torch.gather(
            target_last_hidden_states,
            1,
            gather_indices,
        ).reshape(*safe_label_indices.shape, hidden_size)

    def _dspark_objective_chunk_terms(
        self,
        draft_block_hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        eval_mask: torch.Tensor,
        aligned_target_hidden: Optional[torch.Tensor],
        *,
        collect_detailed_metrics: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive loss and telemetry numerators for one block slice."""

        bsz, num_blocks, block_size, hidden_size = draft_block_hidden_states.shape
        base_logits = self.lm_head(
            draft_block_hidden_states.reshape(bsz, num_blocks * block_size, hidden_size)
        ).reshape(bsz, num_blocks, block_size, -1)
        draft_logits = self.draft_model.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=draft_block_hidden_states,
        )
        zero = loss_weights.new_zeros(())
        l1_num = zero
        confidence_num = zero
        confidence_error_num = zero
        teacher_agreement_num = zero
        teacher_top1_num = zero
        draft_top1_num = zero
        tau_num = zero
        tau_den = zero
        accept_probability = None
        collect_metrics = collect_detailed_metrics and not is_chunk_recomputing()
        correct_num = eval_den = prefix_num = prefix_den = zero

        teacher_ids = None
        if aligned_target_hidden is not None:
            with torch.no_grad():
                target_logits = self.lm_head(
                    aligned_target_hidden.reshape(
                        bsz, num_blocks * block_size, hidden_size
                    )
                ).reshape_as(draft_logits)
                if collect_metrics:
                    teacher_ids = target_logits.argmax(dim=-1)
            distribution_terms = weighted_ce_l1(
                draft_logits,
                target_ids,
                target_logits,
                loss_weights,
                diagnostics=collect_metrics,
            )
            ce_num, l1_objective, _, l1_per_token = distribution_terms[:4]
            if collect_metrics:
                target_top1, draft_top1 = distribution_terms[4:]
            # This probability remains a required, detached confidence target
            # on non-log steps. Only its logging-only consumers are skipped.
            accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
            if self.dspark_l1_loss_alpha > 0:
                l1_num = l1_objective
        else:
            ce_num, _ = weighted_cross_entropy(draft_logits, target_ids, loss_weights)

        confidence_pred = self.draft_model.predict_confidence(
            draft_block_hidden_states,
            prev_token_ids=prev_token_ids,
        )
        if confidence_pred is not None and self.dspark_confidence_head_alpha > 0:
            if accept_probability is None:
                raise ValueError(
                    "DSpark confidence loss requires target_last_hidden_states"
                )
            confidence_num = weighted_binary_cross_entropy(
                confidence_pred.float(),
                accept_probability.detach(),
                loss_weights,
            )
            if collect_metrics:
                with torch.no_grad():
                    confidence_error_num = (
                        (confidence_pred.float().sigmoid() - accept_probability).abs()
                        * loss_weights
                    ).sum()

        if collect_metrics:
            with torch.no_grad():
                predicted_ids = draft_logits.argmax(dim=-1)
                correct = ((predicted_ids == target_ids) & eval_mask).float()
                correct_num = correct.sum()
                eval_den = eval_mask.float().sum()
                # Slot zero is the unsupervised anchor. Start every acceptance
                # chain at slot one so that the anchor mask cannot terminate it.
                prefix = block_accepted_length_terms(
                    predicted_ids[..., 1:], target_ids[..., 1:], eval_mask[..., 1:]
                )
                prefix_num, prefix_den = prefix.numerator, prefix.denominator
                if aligned_target_hidden is not None:
                    assert teacher_ids is not None
                    teacher_agreement_num = (
                        (predicted_ids == teacher_ids).float() * eval_mask
                    ).sum()
                    teacher_top1_num = (target_top1 * eval_mask).sum()
                    draft_top1_num = (draft_top1 * eval_mask).sum()
                    expected_terms = expected_accept_len_terms(
                        accept_probability[..., 1:], eval_mask[..., 1:]
                    )
                    tau_num = expected_terms.numerator
                    tau_den = expected_terms.denominator
        return (
            ce_num,
            l1_num,
            confidence_num,
            confidence_error_num,
            correct_num,
            eval_den,
            teacher_agreement_num,
            teacher_top1_num,
            draft_top1_num,
            tau_num,
            tau_den,
            prefix_num,
            prefix_den,
        )

    def _compute_dspark_loss(
        self,
        *,
        draft_hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        prev_token_ids: torch.Tensor,
        safe_label_indices: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor],
        anchor_positions: torch.Tensor,
        collect_detailed_metrics: bool,
    ) -> TrainingOutput:
        """Token-pooled DSpark objective with bounded vocab-logit memory."""

        bsz, num_blocks, block_size = target_ids.shape
        draft_block_hidden_states = draft_hidden_states.reshape(
            bsz,
            num_blocks,
            block_size,
            -1,
        )
        loss_weights = self._dspark_loss_weight_mask(eval_mask)
        local_loss_den = loss_weights.sum()
        if target_last_hidden_states is None:
            raise ValueError(
                "DSpark expected_accept_len requires target_last_hidden_states"
            )
        aligned_target_hidden = self._aligned_target_hidden(
            target_last_hidden_states,
            safe_label_indices,
        )

        def objective_chunk(*chunk_args):
            return self._dspark_objective_chunk_terms(
                *chunk_args, collect_detailed_metrics=collect_detailed_metrics
            )

        totals = checkpointed_chunk_reduce(
            objective_chunk,
            draft_block_hidden_states,
            prev_token_ids,
            target_ids,
            loss_weights,
            eval_mask,
            aligned_target_hidden,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        (
            ce_num,
            l1_num,
            confidence_num,
            confidence_error_num,
            correct_num,
            eval_den,
            teacher_agreement_num,
            teacher_top1_num,
            draft_top1_num,
            tau_num,
            tau_den,
            prefix_num,
            prefix_den,
        ) = totals

        objective_num = (
            self.dspark_ce_loss_alpha * ce_num
            + self.dspark_l1_loss_alpha * l1_num
            + self.dspark_confidence_head_alpha * confidence_num
        )
        global_loss_den = local_loss_den.detach().clone()
        world_size = 1
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(self.process_group)
            if world_size > 1:
                dist.all_reduce(
                    global_loss_den, op=dist.ReduceOp.SUM, group=self.process_group
                )
        # Device-side assert: a host-side float() here drains the stream per
        # microbatch right after a collective, serializing all ranks.
        torch._assert_async(
            (global_loss_den > 0).any(),
            "DSpark objective has no supervised target tokens",
        )
        loss = world_size * objective_num / global_loss_den

        ratio_metrics = {
            "hard_label/token_accuracy": (correct_num, eval_den),
            "hard_label/block_accepted_length": (prefix_num, prefix_den),
            "objective/ce": (ce_num.detach(), local_loss_den.detach()),
            "objective/l1_distance": (l1_num.detach(), local_loss_den.detach()),
            "confidence_loss": (
                confidence_num.detach(),
                local_loss_den.detach(),
            ),
            "confidence_abs_error": (
                confidence_error_num.detach(),
                local_loss_den.detach(),
            ),
        } if collect_detailed_metrics else {}
        if collect_detailed_metrics and aligned_target_hidden is not None:
            ratio_metrics.update(
                {
                    "teacher_agreement": (teacher_agreement_num, eval_den),
                    "teacher_top1_prob": (teacher_top1_num, eval_den),
                    "draft_top1_prob": (draft_top1_num, eval_den),
                    "tau_probabilistic": (tau_num, tau_den),
                    "expected_accept_len": (tau_num, tau_den),
                }
            )
        return TrainingOutput(
            loss=loss,
            loss_terms=RatioTerms(objective_num, local_loss_den.detach()),
            ratio_metrics={
                name: RatioTerms(numerator.detach(), denominator.detach())
                for name, (numerator, denominator) in ratio_metrics.items()
            },
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        max_valid_anchors: Optional[int] = None,
        collect_detailed_metrics: bool = True,
    ) -> TrainingOutput:
        """Parallel DSpark training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        anchor_positions, block_keep_mask, draft_hidden_states = (
            self._forward_draft_blocks(
                input_ids=input_ids,
                target_context_hidden=hidden_states,
                loss_mask=loss_mask,
                max_valid_anchors=max_valid_anchors,
            )
        )

        (
            target_ids,
            eval_mask,
            safe_label_indices,
        ) = self._build_dspark_labels_and_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        output = self._compute_dspark_loss(
            draft_hidden_states=draft_hidden_states,
            target_ids=target_ids,
            eval_mask=eval_mask,
            prev_token_ids=prev_token_ids,
            safe_label_indices=safe_label_indices,
            target_last_hidden_states=target_last_hidden_states,
            anchor_positions=anchor_positions,
            collect_detailed_metrics=collect_detailed_metrics,
        )
        return output
