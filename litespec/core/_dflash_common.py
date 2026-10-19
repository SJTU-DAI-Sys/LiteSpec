# coding=utf-8
"""DFlash-family training models and shared masking helpers."""

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from litespec.core.attention_masks import BlockMask
from litespec.core.attention_masks import (
    create_dflash_flex_mask as shared_create_dflash_flex_mask,
)
from litespec.core.attention_masks import (
    create_dflash_sdpa_mask as shared_create_dflash_sdpa_mask,
)
from litespec.core.block_metrics import (
    block_accepted_length_terms,
    expected_accept_len_terms,
    hard_label_prefix_counts,
    sparse_distribution_acceptance_probability,
)
from litespec.core.losses import weighted_cross_entropy
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.modeling.draft.flex_attention import flex_kernel_options


class SelectorTerms(NamedTuple):
    """Additive selector objective and metric terms for one objective chunk."""

    ce_num: torch.Tensor
    probability_num: torch.Tensor
    correct_num: torch.Tensor
    weight_den: torch.Tensor
    covered_num: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "SelectorTerms":
        return cls(
            reference.new_zeros(()),
            reference.new_zeros(()),
            reference.new_zeros(()),
            reference.new_zeros(()),
            reference.new_zeros(()),
        )


class DFlashObjectiveTerms(NamedTuple):
    """Additive terms for ``OnlineDFlashModel``'s block objective.

    This tuple covers the standard DFlash loss and optional DFlash2 selector
    terms. DSpark defines a separate objective reduction. The fields stay flat
    and tensor-only to satisfy ``checkpointed_chunk_reduce``'s chunk-function
    contract.
    """

    ce_loss_num: torch.Tensor
    loss_den: torch.Tensor
    target_probability_num: torch.Tensor
    correct_num: torch.Tensor
    accuracy_den: torch.Tensor
    selector_ce_num: torch.Tensor
    selector_probability_num: torch.Tensor
    selector_correct_num: torch.Tensor
    selector_weight_den: torch.Tensor
    selector_covered_num: torch.Tensor


class DFlashMetricTerms(NamedTuple):
    """Additive DFlash2 diagnostics for one metric chunk.

    Per-position fields hold one entry per block position; the accepted-length
    fields and ``block_den`` are additive scalars. No per-anchor reconstruction
    or cross-anchor traversal is needed when combining chunks.
    """

    hard_label_probability_num: torch.Tensor
    unary_correct_num: torch.Tensor
    position_den: torch.Tensor
    unary_topk_recall_num: torch.Tensor
    unary_topk_mass_num: torch.Tensor
    selector_conditional_correct_num: torch.Tensor
    selector_covered_den: torch.Tensor
    selector_greedy_correct_num: torch.Tensor
    teacher_expected_acceptance_num: torch.Tensor
    teacher_unary_top1_agreement_num: torch.Tensor
    teacher_unary_topk_mass_num: torch.Tensor
    teacher_selector_greedy_agreement_num: torch.Tensor
    block_den: torch.Tensor
    oracle_accepted_length_num: torch.Tensor
    greedy_accepted_length_num: torch.Tensor
    expected_accepted_length_num: torch.Tensor
    teacher_expected_accepted_length_num: torch.Tensor
    final_expected_accepted_length_num: torch.Tensor
    prefix_reached_num: torch.Tensor
    prefix_accepted_num: torch.Tensor
    selector_prefix_covered_num: torch.Tensor
    selector_prefix_unary_correct_num: torch.Tensor
    prefix_eligible_num: torch.Tensor
    unary_accepted_length_num: torch.Tensor
    selector_accepted_length_num: torch.Tensor


def _expected_accepted_length_num(
    acceptance: torch.Tensor,
    supervised: torch.Tensor,
    block_valid: torch.Tensor,
) -> torch.Tensor:
    """Sum ``1 + sum_k prod_{j<=k} a_j`` over valid blocks.

    ``acceptance`` holds the per-slot acceptance event or probability for the
    predicted slots; the chain stops at the first unsupervised slot.
    """

    return expected_accept_len_terms(acceptance, supervised, block_valid).numerator


class _DFlashUnaryDiagnostics(NamedTuple):
    """Full-vocabulary unary outputs reused by all DFlash2 diagnostics."""

    probabilities: torch.Tensor
    hard_label_probability: torch.Tensor
    predicted_ids: torch.Tensor
    topk_logits: torch.Tensor
    candidate_ids: torch.Tensor
    target_is_candidate: torch.Tensor


class _DFlashSelectorDiagnostics(NamedTuple):
    """Teacher-forced selector outputs and their effective metric masks."""

    selected_ids: torch.Tensor
    covered_mask: torch.Tensor
    logits: torch.Tensor


class _DFlashGreedyDiagnostics(NamedTuple):
    """Per-block acceptance chains plus the greedy selector path when present."""

    selected_ids: Optional[torch.Tensor]
    correct_num: torch.Tensor
    block_den: torch.Tensor
    oracle_accepted_length_num: torch.Tensor
    accepted_length_num: torch.Tensor
    expected_accepted_length_num: torch.Tensor


class _DFlashTeacherTerms(NamedTuple):
    """Additive target-distribution diagnostics for one metric chunk."""

    expected_acceptance_num: torch.Tensor
    unary_top1_agreement_num: torch.Tensor
    unary_topk_mass_num: torch.Tensor
    selector_greedy_agreement_num: torch.Tensor
    expected_accepted_length_num: torch.Tensor
    final_expected_accepted_length_num: torch.Tensor

    @classmethod
    def zeros(cls, position_den: torch.Tensor) -> "_DFlashTeacherTerms":
        return cls(
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(()),
            position_den.new_zeros(()),
        )


@torch.no_grad()
def _block_accepted_numerator(predicted_ids, target_ids, valid_mask):
    if predicted_ids is None:
        return target_ids.new_zeros((), dtype=torch.float32)
    return block_accepted_length_terms(predicted_ids, target_ids, valid_mask).numerator


class DFlashTrainingBase(nn.Module):
    """DFlash online training wrapper."""

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
        objective_chunk_blocks: int = 128,
        selector_loss_alpha: float = 1.0,
        selector_warmup_ratio: float = 0.0,
        selector_ramp_ratio: float = 0.0,
        selector_stop_gradient: bool = False,
        metric_top_k: int = 16,
        process_group=None,
    ):
        super().__init__()
        if metric_top_k <= 0:
            raise ValueError(f"metric_top_k must be > 0, got {metric_top_k}")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")
        if selector_loss_alpha < 0:
            raise ValueError("selector_loss_alpha must be >= 0")
        if not 0.0 <= selector_warmup_ratio <= 1.0:
            raise ValueError("selector_warmup_ratio must be in [0, 1]")
        if not 0.0 <= selector_ramp_ratio <= 1.0:
            raise ValueError("selector_ramp_ratio must be in [0, 1]")
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.flex_attention_backend = str(flex_attention_backend).upper()
        self.flex_kernel_options = (
            flex_kernel_options(self.flex_attention_backend)
            if self.attention_backend == "flex_attention"
            else None
        )
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.selector_warmup_ratio = float(selector_warmup_ratio)
        self.selector_ramp_ratio = float(selector_ramp_ratio)
        self.selector_stop_gradient = bool(selector_stop_gradient)
        # Candidate-set width for top-K diagnostics on drafts without a selector.
        self.metric_top_k = int(metric_top_k)
        self.process_group = process_group

        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        self._selector_objective_enabled = (
            candidate_selector is not None and self.selector_loss_alpha > 0
        )
        if (
            candidate_selector is not None
            and not self._selector_objective_enabled
            and isinstance(candidate_selector, nn.Module)
        ):
            # A zero configured weight statically disables selector training.
            # Freezing keeps those parameters out of BF16Optimizer and prevents
            # DDP(find_unused_parameters=False) from waiting for their gradients.
            candidate_selector.requires_grad_(False)

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchors whose clean token and first target are supervised."""

        num_candidates = max(seq_len - 1, 0)
        valid_anchor_mask = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid_anchor_mask.sum(dim=1)
        if max_valid_anchors is None:
            # Direct model callers may supply an already-device-resident mask.
            # Training strategies pass the CPU-computed value and avoid this
            # synchronizing fallback on CUDA.
            max_valid_anchors = int(valid_counts.max().item())
        width = min(self.num_anchors, max(0, int(max_valid_anchors)))
        if width == 0:
            raise ValueError(
                "DFlash-family training requires two consecutive supervised tokens"
            )

        random_values = torch.rand(valid_anchor_mask.shape, device=device)
        random_values.masked_fill_(~valid_anchor_mask, 2.0)
        candidates = random_values.argsort(dim=1)[:, :width]
        block_keep_mask = torch.arange(width, device=device).unsqueeze(
            0
        ) < valid_counts.clamp(max=width).unsqueeze(1)

        sentinel = valid_anchor_mask.shape[1]
        anchor_positions = torch.where(
            block_keep_mask,
            candidates,
            torch.full_like(candidates, sentinel),
        )
        anchor_positions = anchor_positions.sort(dim=1).values
        block_keep_mask = anchor_positions < sentinel
        return torch.where(block_keep_mask, anchor_positions, 0), block_keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, num_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        position_ids = anchor_positions.unsqueeze(-1) + offsets
        return position_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        num_blocks = anchor_positions.shape[1]
        block_size = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, num_blocks * block_size), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(num_blocks, device=device) * block_size
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, num_blocks)
        # masked_fill with a scalar avoids a pageable H2D copy per microbatch.
        noise_ids[flat_batch_idx, block_starts] = anchor_tokens.masked_fill(
            ~block_keep_mask, self.mask_token_id
        )

        return self.embed_tokens(noise_ids)

    def _aligned_target_hidden(
        self,
        target_last_hidden_states: torch.Tensor,
        safe_label_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the frozen target state that predicts each hard label."""

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

    @staticmethod
    def _add_position_ratios(
        metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        name: str,
        numerators: torch.Tensor,
        denominators: torch.Tensor,
    ) -> None:
        """Add one ratio per predicted block position.

        Per-position keys drop the algorithm prefix and live under their own
        ``position_<k>/`` tracker section so each block position renders as one
        dashboard group.
        """

        metric_path = name.split("/", 1)[1]
        for position in range(1, numerators.numel()):
            metrics[f"position_{position}/{metric_path}"] = (
                numerators[position].detach(),
                denominators[position].detach(),
            )

    @classmethod
    def _add_ratio_family(
        cls,
        metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        name: str,
        numerators: torch.Tensor,
        denominators: torch.Tensor,
    ) -> None:
        """Add an aggregate ratio and the sole retained position metric."""

        metrics[name] = (
            numerators.sum().detach(),
            denominators.sum().detach(),
        )
        if name == "dflash/hard_label/unary_top16_recall":
            cls._add_position_ratios(metrics, name, numerators, denominators)

    @classmethod
    def _add_prefix_metrics(
        cls,
        ratios: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        counts: Dict[str, torch.Tensor],
        terms: DFlashMetricTerms,
        top_k: int,
        *,
        has_selector: bool,
    ) -> None:
        """Expose survival and first-failure diagnostics with exact counts."""
        block_den = terms.block_den.detach()
        counts["dflash/hard_label/block_count"] = block_den
        ratios["dflash/hard_label/supervised_prefix_length"] = (
            block_den + terms.prefix_eligible_num.sum().detach(),
            block_den,
        )
        ratios["dflash/hard_label/unary_greedy_accepted_length"] = (
            block_den + terms.prefix_accepted_num[0].sum().detach(),
            block_den,
        )
        paths = [
            "dflash/hard_label/unary_greedy",
            f"dflash/hard_label/unary_top{top_k}_oracle",
        ]
        if has_selector:
            paths.append("dflash2/selector/greedy")
        for index, path in enumerate(paths):
            reached = terms.prefix_reached_num[index]
            accepted = terms.prefix_accepted_num[index]
            cls._add_ratio_family(
                ratios, f"{path}_prefix_acceptance", accepted, reached
            )

        if has_selector:
            reached = terms.prefix_reached_num[2]
            accepted = terms.prefix_accepted_num[2]
            covered = terms.selector_prefix_covered_num
            for name, numerator, denominator in (
                (
                    "unary_top1_accuracy",
                    terms.selector_prefix_unary_correct_num,
                    reached,
                ),
                (f"top{top_k}_recall", covered, reached),
                ("covered_accuracy", accepted, covered),
                ("coverage_miss_rate", reached - covered, reached),
                ("ranking_error_rate", covered - accepted, reached),
            ):
                cls._add_ratio_family(
                    ratios,
                    f"dflash2/selector/greedy_prefix_{name}",
                    numerator,
                    denominator,
                )

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        target_context_hidden: torch.Tensor,
        loss_mask: torch.Tensor,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            max_valid_anchors=max_valid_anchors,
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        mask_builder = (
            shared_create_dflash_flex_mask
            if self.attention_backend == "flex_attention"
            else shared_create_dflash_sdpa_mask
        )
        mask_args = {
            "anchor_positions": anchor_positions,
            "block_keep_mask": block_keep_mask,
            "S": seq_len,
            "block_size": self.block_size,
            "device": device,
        }
        if self.attention_backend == "flex_attention":
            mask_args["flex_backend"] = self.flex_attention_backend
        full_attn_mask = mask_builder(**mask_args)
        sliding_window = self.draft_model.sliding_window
        dflash_attn_mask = full_attn_mask
        if sliding_window is not None:
            dflash_attn_mask = {
                "full_attention": full_attn_mask,
                "sliding_attention": mask_builder(
                    **mask_args,
                    sliding_window=sliding_window,
                ),
            }

        draft_kwargs = {}
        if self.attention_backend == "flex_attention":
            draft_kwargs["kernel_options"] = self.flex_kernel_options
        draft_hidden_states = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_context_hidden,
            attention_mask=dflash_attn_mask,
            **draft_kwargs,
        )
        return anchor_positions, block_keep_mask, draft_hidden_states

    def _dflash_objective_chunk_terms(
        self,
        draft_block_hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        predecessor_ids: torch.Tensor,
        *,
        metrics_callback=None,
        collect_detailed_metrics: bool = True,
    ) -> DFlashObjectiveTerms:
        """Return a flat tuple of additive objective and metric tensors."""

        bsz, num_blocks, block_size, hidden_size = draft_block_hidden_states.shape
        logits = self.lm_head(
            draft_block_hidden_states.reshape(bsz, num_blocks * block_size, hidden_size)
        ).reshape(bsz, num_blocks, block_size, -1)
        if metrics_callback is not None:
            metrics_callback(logits)
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        objective_logits = (
            self.draft_model.transform_unary_logits(logits)
            if candidate_selector is not None
            else logits
        )
        loss_weights = weight_mask
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(
                self.block_size,
                device=draft_block_hidden_states.device,
            ).view(1, 1, -1)
            decay_weights = torch.exp(
                -(positions - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            loss_weights = loss_weights * decay_weights
        loss_den = loss_weights.sum()

        ce_loss_num, neg_log_q = weighted_cross_entropy(
            objective_logits, target_ids, loss_weights
        )
        zero = ce_loss_num.new_zeros(())
        target_probability_num = zero
        correct_num = zero
        accuracy_den = zero
        if collect_detailed_metrics:
            target_probability = torch.exp(-neg_log_q)
            target_probability_num = (target_probability.detach() * weight_mask).sum()

        selector_terms = SelectorTerms.zeros(ce_loss_num)
        if self._selector_objective_enabled:
            selector_terms = self._selector_chunk_terms(
                candidate_selector=candidate_selector,
                objective_logits=objective_logits,
                draft_block_hidden_states=draft_block_hidden_states,
                target_ids=target_ids,
                predecessor_ids=predecessor_ids,
                loss_weights=loss_weights,
                weight_mask=weight_mask,
                collect_detailed_metrics=collect_detailed_metrics,
            )

        if collect_detailed_metrics:
            with torch.no_grad():
                predicted_ids = objective_logits.argmax(dim=-1)
                correct_num = (
                    ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
                )
                accuracy_den = weight_mask.sum()
        return DFlashObjectiveTerms(
            ce_loss_num=ce_loss_num,
            loss_den=loss_den,
            target_probability_num=target_probability_num,
            correct_num=correct_num,
            accuracy_den=accuracy_den,
            selector_ce_num=selector_terms.ce_num,
            selector_probability_num=selector_terms.probability_num,
            selector_correct_num=selector_terms.correct_num,
            selector_weight_den=selector_terms.weight_den,
            selector_covered_num=selector_terms.covered_num,
        )

    @torch.no_grad()
    def _metric_top_k(self) -> int:
        """Candidate-set width: the selector's top-k, else ``metric_top_k``.

        Never wider than the vocabulary, so tiny test vocabularies stay valid.
        """

        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        top_k = (
            int(candidate_selector.top_k)
            if candidate_selector is not None
            else self.metric_top_k
        )
        vocab_size = getattr(self.embed_tokens, "num_embeddings", None)
        return top_k if vocab_size is None else min(top_k, int(vocab_size))

    def _dflash_metric_chunk_terms(
        self,
        draft_block_hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        predecessor_ids: torch.Tensor,
        aligned_target_hidden: Optional[torch.Tensor] = None,
        *,
        logits: Optional[torch.Tensor] = None,
    ) -> DFlashMetricTerms:
        """Return additive unary, selector, and teacher diagnostics for one chunk.

        The unary, objective-weight, and teacher families apply to every
        DFlash-family draft; the selector and greedy-path families are only
        computed when the draft carries a DFlash2 candidate selector.
        """

        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        unary = self._dflash_unary_diagnostics(
            draft_block_hidden_states,
            target_ids,
            top_k=self._metric_top_k(),
            logits=logits,
        )
        selector = None
        if candidate_selector is not None:
            selector = self._dflash2_selector_diagnostics(
                candidate_selector=candidate_selector,
                unary=unary,
                draft_block_hidden_states=draft_block_hidden_states,
                predecessor_ids=predecessor_ids,
                weight_mask=weight_mask,
            )
        position_den = weight_mask.sum(dim=(0, 1))
        greedy = self._dflash_greedy_diagnostics(
            candidate_selector=candidate_selector,
            unary=unary,
            draft_block_hidden_states=draft_block_hidden_states,
            target_ids=target_ids,
            weight_mask=weight_mask,
            position_den=position_den,
        )
        teacher = self._dflash_teacher_terms(
            unary=unary,
            selector=selector,
            greedy_ids=greedy.selected_ids,
            aligned_target_hidden=aligned_target_hidden,
            weight_mask=weight_mask,
            position_den=position_den,
        )
        return self._reduce_dflash_metric_terms(
            unary=unary,
            selector=selector,
            greedy=greedy,
            teacher=teacher,
            target_ids=target_ids,
            weight_mask=weight_mask,
            position_den=position_den,
        )

    def _dflash_unary_diagnostics(
        self,
        draft_block_hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        top_k: int,
        logits: Optional[torch.Tensor] = None,
    ) -> _DFlashUnaryDiagnostics:
        """Project draft states once and derive the strict unary top-k view."""

        bsz, num_blocks, block_size, hidden_size = draft_block_hidden_states.shape
        if logits is None:
            logits = self.lm_head(
                draft_block_hidden_states.reshape(bsz, num_blocks * block_size, hidden_size)
            ).reshape(bsz, num_blocks, block_size, -1)
        transform = getattr(self.draft_model, "transform_unary_logits", None)
        objective_logits = logits if transform is None else transform(logits)
        probabilities = torch.softmax(objective_logits.float(), dim=-1)
        hard_label_probability = probabilities.gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        topk_logits, candidate_ids = objective_logits.topk(
            min(top_k, objective_logits.shape[-1]),
            dim=-1,
        )
        target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
        return _DFlashUnaryDiagnostics(
            probabilities=probabilities,
            hard_label_probability=hard_label_probability,
            predicted_ids=objective_logits.argmax(dim=-1),
            topk_logits=topk_logits,
            candidate_ids=candidate_ids,
            target_is_candidate=target_matches.any(dim=-1),
        )

    @staticmethod
    def _dflash_greedy_diagnostics(
        *,
        candidate_selector: Optional[nn.Module],
        unary: _DFlashUnaryDiagnostics,
        draft_block_hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        position_den: torch.Tensor,
    ) -> _DFlashGreedyDiagnostics:
        """Reduce per-block acceptance chains; walk the selector path if any."""

        bsz, num_blocks, block_size, hidden_size = draft_block_hidden_states.shape
        zero = position_den.new_zeros(())
        if block_size <= 1:
            return _DFlashGreedyDiagnostics(
                selected_ids=None,
                correct_num=position_den.new_zeros(position_den.shape),
                block_den=zero,
                oracle_accepted_length_num=zero.clone(),
                accepted_length_num=zero.clone(),
                expected_accepted_length_num=zero.clone(),
            )

        # A block accepts its anchor, then only its leading run of supervised
        # slots: covered (oracle), hit by the greedy path (realized), or weighted
        # by the gold-token probability.
        supervised = weight_mask[:, :, 1:] > 0.5
        block_valid = supervised.any(dim=-1).float()
        oracle_accepted_length_num = _expected_accepted_length_num(
            unary.target_is_candidate[:, :, 1:], supervised, block_valid
        )
        expected_accepted_length_num = _expected_accepted_length_num(
            unary.hard_label_probability[:, :, 1:], supervised, block_valid
        )
        if candidate_selector is None:
            return _DFlashGreedyDiagnostics(
                selected_ids=None,
                correct_num=position_den.new_zeros(position_den.shape),
                block_den=block_valid.sum(),
                oracle_accepted_length_num=oracle_accepted_length_num,
                accepted_length_num=zero,
                expected_accepted_length_num=expected_accepted_length_num,
            )

        greedy_ids = candidate_selector.greedy_path(
            candidate_ids=unary.candidate_ids[:, :, 1:].reshape(
                bsz * num_blocks,
                block_size - 1,
                candidate_selector.top_k,
            ),
            unary_logits=unary.topk_logits[:, :, 1:].reshape(
                bsz * num_blocks,
                block_size - 1,
                candidate_selector.top_k,
            ),
            hidden_states=draft_block_hidden_states[:, :, 1:].reshape(
                bsz * num_blocks,
                block_size - 1,
                hidden_size,
            ),
            anchor_token_ids=target_ids[:, :, 0].reshape(-1),
        ).reshape(bsz, num_blocks, block_size - 1)
        greedy_hit = greedy_ids == target_ids[:, :, 1:]
        return _DFlashGreedyDiagnostics(
            selected_ids=greedy_ids,
            correct_num=torch.cat(
                (
                    position_den.new_zeros(1),
                    (greedy_hit.float() * weight_mask[:, :, 1:]).sum(dim=(0, 1)),
                )
            ),
            block_den=block_valid.sum(),
            oracle_accepted_length_num=oracle_accepted_length_num,
            accepted_length_num=_expected_accepted_length_num(
                greedy_hit, supervised, block_valid
            ),
            expected_accepted_length_num=expected_accepted_length_num,
        )

    def _dflash_teacher_terms(
        self,
        *,
        unary: _DFlashUnaryDiagnostics,
        selector: Optional[_DFlashSelectorDiagnostics],
        greedy_ids: Optional[torch.Tensor],
        aligned_target_hidden: Optional[torch.Tensor],
        weight_mask: torch.Tensor,
        position_den: torch.Tensor,
    ) -> _DFlashTeacherTerms:
        """Compare unary and greedy predictions with the frozen target head."""

        if aligned_target_hidden is None:
            return _DFlashTeacherTerms.zeros(position_den)

        bsz, num_blocks, block_size, hidden_size = aligned_target_hidden.shape
        teacher_logits = self.lm_head(
            aligned_target_hidden.reshape(
                bsz,
                num_blocks * block_size,
                hidden_size,
            )
        ).reshape_as(unary.probabilities)
        teacher_probabilities = torch.softmax(teacher_logits.float(), dim=-1)
        teacher_ids = teacher_logits.argmax(dim=-1)
        expected_acceptance = (
            1.0 - 0.5 * (unary.probabilities - teacher_probabilities).abs().sum(dim=-1)
        ).clamp(0.0, 1.0)

        selector_greedy_agreement_num = position_den.new_zeros(position_den.shape)
        expected_accepted_length_num = position_den.new_zeros(())
        final_expected_accepted_length_num = position_den.new_zeros(())
        if block_size > 1:
            if greedy_ids is not None:
                selector_greedy_agreement_num = torch.cat(
                    (
                        position_den.new_zeros(1),
                        (
                            (greedy_ids == teacher_ids[:, :, 1:]).float()
                            * weight_mask[:, :, 1:]
                        ).sum(dim=(0, 1)),
                    )
                )
            supervised = weight_mask[:, :, 1:] > 0.5
            expected_accepted_length_num = _expected_accepted_length_num(
                expected_acceptance[:, :, 1:],
                supervised,
                supervised.any(dim=-1).float(),
            )
            final_acceptance = expected_acceptance[:, :, 1:]
            if selector is not None:
                final_acceptance = sparse_distribution_acceptance_probability(
                    teacher_logits[:, :, 1:],
                    selector.logits[:, :, 1:],
                    unary.candidate_ids[:, :, 1:],
                )
            final_expected_accepted_length_num = expected_accept_len_terms(
                final_acceptance, supervised
            ).numerator
        return _DFlashTeacherTerms(
            expected_acceptance_num=(expected_acceptance * weight_mask).sum(dim=(0, 1)),
            unary_top1_agreement_num=(
                (unary.predicted_ids == teacher_ids).float() * weight_mask
            ).sum(dim=(0, 1)),
            unary_topk_mass_num=(
                teacher_probabilities.gather(-1, unary.candidate_ids).sum(dim=-1)
                * weight_mask
            ).sum(dim=(0, 1)),
            selector_greedy_agreement_num=selector_greedy_agreement_num,
            expected_accepted_length_num=expected_accepted_length_num,
            final_expected_accepted_length_num=final_expected_accepted_length_num,
        )

    @staticmethod
    def _reduce_dflash_metric_terms(
        *,
        unary: _DFlashUnaryDiagnostics,
        selector: Optional[_DFlashSelectorDiagnostics],
        greedy: _DFlashGreedyDiagnostics,
        teacher: _DFlashTeacherTerms,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        position_den: torch.Tensor,
    ) -> DFlashMetricTerms:
        """Reduce per-token diagnostics into the flat chunk-reducer contract."""

        covered_mask = weight_mask * unary.target_is_candidate.float()
        position_zeros = position_den.new_zeros(position_den.shape)
        if selector is None:
            selector_conditional_correct_num = position_zeros
        else:
            selector_conditional_correct_num = (
                (selector.selected_ids == target_ids).float() * selector.covered_mask
            ).sum(dim=(0, 1))
        prefix = hard_label_prefix_counts(
            unary_hit=unary.predicted_ids[:, :, 1:] == target_ids[:, :, 1:],
            covered=unary.target_is_candidate[:, :, 1:],
            selector_hit=(
                None
                if greedy.selected_ids is None
                else greedy.selected_ids == target_ids[:, :, 1:]
            ),
            supervised=weight_mask[:, :, 1:] > 0.5,
        )
        return DFlashMetricTerms(
            hard_label_probability_num=(unary.hard_label_probability * weight_mask).sum(
                dim=(0, 1)
            ),
            unary_correct_num=(
                (unary.predicted_ids == target_ids).float() * weight_mask
            ).sum(dim=(0, 1)),
            position_den=position_den,
            unary_topk_recall_num=covered_mask.sum(dim=(0, 1)),
            unary_topk_mass_num=(
                unary.probabilities.gather(-1, unary.candidate_ids).sum(dim=-1)
                * weight_mask
            ).sum(dim=(0, 1)),
            selector_conditional_correct_num=selector_conditional_correct_num,
            selector_covered_den=covered_mask.sum(dim=(0, 1)),
            selector_greedy_correct_num=greedy.correct_num,
            teacher_expected_acceptance_num=teacher.expected_acceptance_num,
            teacher_unary_top1_agreement_num=teacher.unary_top1_agreement_num,
            teacher_unary_topk_mass_num=teacher.unary_topk_mass_num,
            teacher_selector_greedy_agreement_num=teacher.selector_greedy_agreement_num,
            block_den=greedy.block_den,
            oracle_accepted_length_num=greedy.oracle_accepted_length_num,
            greedy_accepted_length_num=greedy.accepted_length_num,
            expected_accepted_length_num=greedy.expected_accepted_length_num,
            teacher_expected_accepted_length_num=teacher.expected_accepted_length_num,
            final_expected_accepted_length_num=(
                teacher.final_expected_accepted_length_num
            ),
            prefix_reached_num=prefix.reached,
            prefix_accepted_num=prefix.accepted,
            selector_prefix_covered_num=prefix.selector_covered,
            selector_prefix_unary_correct_num=prefix.selector_unary_correct,
            prefix_eligible_num=prefix.eligible,
            unary_accepted_length_num=_block_accepted_numerator(
                unary.predicted_ids[..., 1:],
                target_ids[..., 1:],
                weight_mask[..., 1:] > 0.5,
            ),
            selector_accepted_length_num=_block_accepted_numerator(
                greedy.selected_ids,
                target_ids[..., 1:],
                weight_mask[..., 1:] > 0.5,
            ),
        )
