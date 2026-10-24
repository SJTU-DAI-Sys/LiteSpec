"""Additive training outputs; numerators retain their gradient graph."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class RatioTerms:
    numerator: torch.Tensor
    denominator: torch.Tensor

    def detached(self) -> "RatioTerms":
        return RatioTerms(self.numerator.detach(), self.denominator.detach())


@dataclass
class TrainingOutput:
    loss: torch.Tensor
    loss_terms: RatioTerms
    ratio_metrics: dict[str, RatioTerms] = field(default_factory=dict)
    sum_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
