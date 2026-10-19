"""DFlash block prediction objective."""

from typing import Optional

import torch

from litespec.core.attention_masks import FLEX_ATTENTION_AVAILABLE
from litespec.core.chunking import checkpointed_chunk_reduce, is_chunk_recomputing
from litespec.core.outputs import RatioTerms, TrainingOutput

from ._dflash_common import DFlashMetricTerms, DFlashTrainingBase


class OnlineDFlashModel(DFlashTrainingBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        max_valid_anchors: Optional[int] = None,
        selector_loss_alpha: Optional[float] = None,
        collect_detailed_metrics: bool = True,
    ) -> TrainingOutput:
        """Parallel block-wise training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, draft_hidden_states = self._forward_draft_blocks(
            input_ids=input_ids,
            target_context_hidden=hidden_states,
            loss_mask=loss_mask,
            max_valid_anchors=max_valid_anchors,
        )

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        predecessor_ids = torch.cat(
            [target_ids[:, :, :1], target_ids[:, :, :-1]],
            dim=-1,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        draft_block_hidden_states = draft_hidden_states.reshape(
            bsz,
            anchor_positions.shape[1],
            self.block_size,
            -1,
        )
        aligned_target_hidden = (
            self._aligned_target_hidden(
                target_last_hidden_states,
                safe_label_indices,
            )
            if target_last_hidden_states is not None and collect_detailed_metrics
            else None
        )
        metric_totals = None

        def objective_chunk(hidden, ids, weights, predecessors, teacher_hidden):
            def collect(logits):
                nonlocal metric_totals
                # Reuse each chunk's projection before it is released. Logging
                # never retains its graph or runs again during checkpoint replay.
                with torch.no_grad():
                    terms = self._dflash_metric_chunk_terms(
                        hidden, ids, weights, predecessors, teacher_hidden,
                        logits=logits,
                    )
                    metric_totals = terms if metric_totals is None else DFlashMetricTerms(
                        *(a + b for a, b in zip(metric_totals, terms))
                    )

            return self._dflash_objective_chunk_terms(
                hidden, ids, weights, predecessors,
                collect_detailed_metrics=(
                    collect_detailed_metrics and not is_chunk_recomputing()
                ),
                metrics_callback=(
                    collect if collect_detailed_metrics and not is_chunk_recomputing()
                    else None
                ),
            )

        (
            ce_loss_num,
            loss_den,
            target_probability_num,
            correct_num,
            accuracy_denom,
            selector_ce_num,
            selector_probability_num,
            selector_correct_num,
            selector_weight_den,
            selector_covered_num,
        ) = checkpointed_chunk_reduce(
            objective_chunk,
            draft_block_hidden_states,
            target_ids,
            weight_mask,
            predecessor_ids,
            aligned_target_hidden,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        loss_num = ce_loss_num
        effective_selector_alpha = (
            self.selector_loss_alpha
            if selector_loss_alpha is None
            else float(selector_loss_alpha)
        )
        if effective_selector_alpha < 0:
            raise ValueError("selector_loss_alpha must be >= 0")
        selector_loss_num = loss_num.new_zeros(())
        has_selector_objective = self._selector_objective_enabled
        if has_selector_objective:
            # The selector is a categorical distribution over the serving
            # top-k. Keep its proper, calibrated CE independent of the base CE.
            selector_loss_num = selector_ce_num
            loss_num = loss_num + effective_selector_alpha * selector_loss_num

        loss_denominator = loss_den
        ratio_metrics = {
            "hard_label/token_accuracy": (
                correct_num.detach(),
                accuracy_denom.detach(),
            ),
            "hard_label/gold_probability": (
                target_probability_num.detach(),
                accuracy_denom.detach(),
            ),
            "objective/ce": (
                ce_loss_num.detach(),
                loss_denominator.detach(),
            ),
        } if collect_detailed_metrics else {}
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        sum_metrics = {}
        if collect_detailed_metrics:
            assert metric_totals is not None
            terms = metric_totals
            ratio_metrics["hard_label/block_accepted_length"] = (
                (
                    terms.selector_accepted_length_num
                    if candidate_selector is not None
                    else terms.unary_accepted_length_num
                ).detach(),
                terms.block_den.detach(),
            )
            top_k = self._metric_top_k()
            for name, numerators in (
                (
                    "dflash/hard_label/unary_top1_accuracy",
                    terms.unary_correct_num,
                ),
                (
                    "dflash/hard_label/unary_probability",
                    terms.hard_label_probability_num,
                ),
                (
                    f"dflash/hard_label/unary_top{top_k}_recall",
                    terms.unary_topk_recall_num,
                ),
                (
                    f"dflash/hard_label/unary_top{top_k}_mass",
                    terms.unary_topk_mass_num,
                ),
            ):
                self._add_ratio_family(
                    ratio_metrics,
                    name,
                    numerators,
                    terms.position_den,
                )
            ratio_metrics[
                f"dflash/hard_label/unary_top{top_k}_oracle_accepted_length"
            ] = (terms.oracle_accepted_length_num.detach(), terms.block_den.detach())
            ratio_metrics["dflash/hard_label/unary_gold_probability_chain_length"] = (
                terms.expected_accepted_length_num.detach(),
                terms.block_den.detach(),
            )
            self._add_prefix_metrics(
                ratio_metrics,
                sum_metrics,
                terms,
                top_k,
                has_selector=candidate_selector is not None,
            )
            if aligned_target_hidden is not None:
                ratio_metrics["expected_accept_len"] = (
                    terms.final_expected_accepted_length_num.detach(),
                    terms.block_den.detach(),
                )
                ratio_metrics["dflash/teacher/unary_overlap_chain_length"] = (
                    terms.teacher_expected_accepted_length_num.detach(),
                    terms.block_den.detach(),
                )
                for name, numerators in (
                    (
                        "dflash/teacher/unary_distribution_overlap",
                        terms.teacher_expected_acceptance_num,
                    ),
                    (
                        "dflash/teacher/unary_top1_agreement",
                        terms.teacher_unary_top1_agreement_num,
                    ),
                    (
                        f"dflash/teacher/unary_top{top_k}_mass",
                        terms.teacher_unary_topk_mass_num,
                    ),
                ):
                    self._add_ratio_family(
                        ratio_metrics,
                        name,
                        numerators,
                        terms.position_den,
                    )
            if candidate_selector is not None:
                self._add_ratio_family(
                    ratio_metrics,
                    "dflash2/selector/self_conditioned_marginal_accuracy",
                    terms.selector_greedy_correct_num,
                    terms.position_den,
                )
                self._add_ratio_family(
                    ratio_metrics,
                    "dflash2/selector/teacher_forced_covered_accuracy",
                    terms.selector_conditional_correct_num,
                    terms.selector_covered_den,
                )
                ratio_metrics["dflash2/selector/greedy_accepted_length"] = (
                    terms.greedy_accepted_length_num.detach(),
                    terms.block_den.detach(),
                )
                if aligned_target_hidden is not None:
                    self._add_ratio_family(
                        ratio_metrics,
                        "dflash2/selector/self_conditioned_teacher_argmax_agreement",
                        terms.teacher_selector_greedy_agreement_num,
                        terms.position_den,
                    )
        if has_selector_objective and collect_detailed_metrics:
            ratio_metrics.update(
                {
                    "dflash2/objective/weighted_covered_accuracy": (
                        selector_correct_num.detach(),
                        selector_weight_den.detach(),
                    ),
                    "dflash2/selector/teacher_forced_covered_gold_probability": (
                        selector_probability_num.detach(),
                        selector_covered_num.detach(),
                    ),
                    "dflash2/selector/loss": (
                        selector_loss_num.detach(),
                        selector_weight_den.detach(),
                    ),
                }
            )
        # Reduce all chunks before flooring the denominator so the result does
        # not depend on how many chunks happen to contain no effective weight.
        denominator_floor = torch.finfo(loss_denominator.dtype).tiny
        loss = loss_num / loss_denominator.clamp_min(denominator_floor)
        return TrainingOutput(
            loss=loss,
            loss_terms=RatioTerms(loss_num, loss_denominator.detach()),
            ratio_metrics={
                name: RatioTerms(*terms) for name, terms in ratio_metrics.items()
            },
            sum_metrics=sum_metrics,
        )
