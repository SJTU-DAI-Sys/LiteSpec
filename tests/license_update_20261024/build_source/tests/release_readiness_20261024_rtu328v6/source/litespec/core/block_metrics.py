"""Additive metrics shared by block-parallel draft training algorithms."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .outputs import RatioTerms


class PrefixCounts(NamedTuple):
    reached: torch.Tensor
    accepted: torch.Tensor
    selector_covered: torch.Tensor
    selector_unary_correct: torch.Tensor
    eligible: torch.Tensor


def _prefix_events(hit: torch.Tensor, supervised: torch.Tensor):
    accepted = (hit & supervised).long().cumprod(dim=-1).bool()
    previous = torch.cat(
        (torch.ones_like(accepted[..., :1]), accepted[..., :-1]), dim=-1
    )
    return previous & supervised, accepted


def hard_label_prefix_counts(
    unary_hit: torch.Tensor,
    covered: torch.Tensor,
    selector_hit: Optional[torch.Tensor],
    supervised: torch.Tensor,
) -> PrefixCounts:
    if selector_hit is None:
        selector_reached = selector_accepted = torch.zeros_like(supervised)
    else:
        selector_reached, selector_accepted = _prefix_events(selector_hit, supervised)
    unary_reached, unary_accepted = _prefix_events(unary_hit, supervised)
    oracle_reached, oracle_accepted = _prefix_events(covered, supervised)

    def count(events):
        return F.pad(events.float().sum(dim=(0, 1)), (1, 0))

    return PrefixCounts(
        reached=torch.stack(
            [count(unary_reached), count(oracle_reached), count(selector_reached)]
        ),
        accepted=torch.stack(
            [count(unary_accepted), count(oracle_accepted), count(selector_accepted)]
        ),
        selector_covered=count(selector_reached & covered),
        selector_unary_correct=count(selector_reached & unary_hit),
        eligible=count(supervised.long().cumprod(dim=-1).bool()),
    )


def block_accepted_length_terms(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    supervised: torch.Tensor,
) -> RatioTerms:
    valid_blocks = supervised.any(dim=-1)
    accepted = (
        ((predicted_ids == target_ids) & supervised).long().cumprod(dim=-1).sum(dim=-1)
    )
    lengths = accepted.float() + 1.0
    return RatioTerms((lengths * valid_blocks).sum(), valid_blocks.sum())


def distribution_acceptance_probability(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
) -> torch.Tensor:
    """Return speculative acceptance probability for two dense distributions."""

    if target_logits.shape != draft_logits.shape:
        raise ValueError("target_logits and draft_logits must have identical shapes")
    target_probs = torch.softmax(target_logits.detach().float(), dim=-1)
    draft_probs = torch.softmax(draft_logits.detach().float(), dim=-1)
    return torch.minimum(target_probs, draft_probs).sum(dim=-1)


def sparse_distribution_acceptance_probability(
    target_logits: torch.Tensor,
    draft_candidate_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Return acceptance probability for a normalized sparse draft proposal."""

    if draft_candidate_logits.shape != candidate_ids.shape:
        raise ValueError("draft_candidate_logits and candidate_ids must match")
    if target_logits.shape[:-1] != candidate_ids.shape[:-1]:
        raise ValueError("target and candidate batch dimensions must match")
    target_probs = torch.softmax(target_logits.detach().float(), dim=-1)
    target_candidate_probs = target_probs.gather(-1, candidate_ids)
    draft_probs = torch.softmax(draft_candidate_logits.detach().float(), dim=-1)
    return torch.minimum(target_candidate_probs, draft_probs).sum(dim=-1)


def expected_accept_len_terms(
    acceptance_probability: torch.Tensor,
    supervised: torch.Tensor,
    block_valid: Optional[torch.Tensor] = None,
) -> RatioTerms:
    """Return additive terms for ``1 + sum_k prod_{i<=k} acceptance_i``."""

    if acceptance_probability.shape != supervised.shape:
        raise ValueError("acceptance_probability and supervised must match")
    valid_blocks = supervised.any(dim=-1) if block_valid is None else block_valid
    alive = acceptance_probability.detach().float() * supervised.float()
    expected = 1.0 + alive.cumprod(dim=-1).sum(dim=-1)
    return RatioTerms(
        (expected * valid_blocks.float()).sum(), valid_blocks.float().sum()
    )


def fsdp_global_weighted_mean(
    numerator: torch.Tensor, denominator: torch.Tensor, group=None
):
    global_denominator = denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM, group=group)
        world_size = dist.get_world_size(group)
    return numerator * (world_size / global_denominator.clamp_min(1e-6))
