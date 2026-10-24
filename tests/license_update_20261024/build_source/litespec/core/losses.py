"""Shared weighted losses with autograd-generated Triton forward/backward.

The eager functions are the numerical reference, including input-dtype CE
semantics and FP32 distribution losses. CPU and small workloads stay eager.
CE and L1 softmax stay native to preserve rounding/subgradients, including
FP32 logits cast from a BF16 head. Eligible distribution reductions use
Inductor-generated Triton and first-order autograd.
Set LITESPEC_LOSS_BACKEND=torch to compare an entire run against the reference,
or inductor to explicitly exercise compilation in numerical/throughput tests.
"""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F


def _ce(logits, labels, weights):
    values = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    return (values * weights).sum(), values.detach()


def _ce_l1(logits, labels, teacher_logits, weights, diagnostics=False):
    ce, values = _ce(logits, labels, weights)
    l1 = _l1(logits, teacher_logits, weights, diagnostics)
    return ce, l1[0], values, *l1[1:]


def _l1(logits, teacher_logits, weights, diagnostics=False):
    target = torch.softmax(teacher_logits.detach().float(), dim=-1)
    draft = torch.softmax(logits.float(), dim=-1)
    return _l1_probs(draft, target, weights, diagnostics)


def _l1_probs(draft, target, weights, diagnostics=False):
    distance = (draft - target).abs().sum(dim=-1)
    result = ((distance * weights).sum(), distance.detach())
    if diagnostics:
        # Detached diagnostics must not save autograd state: logging-only
        # maxima are skipped when the objective is checkpoint-recomputed.
        with torch.no_grad():
            target_top1 = target.max(dim=-1).values
            draft_top1 = draft.max(dim=-1).values
        return (*result, target_top1, draft_top1)
    return result


def _ce_l1_probs(logits, labels, draft, target, weights, diagnostics=False):
    ce, values = _ce(logits, labels, weights)
    l1 = _l1_probs(draft, target, weights, diagnostics)
    return ce, l1[0], values, *l1[1:]


def _kl(logits, teacher_logits, weights):
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    values = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (values * weights).sum()


def _kl_student(logits, teacher_log_probs, teacher_probs, weights):
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    values = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (values * weights).sum()


def _bce(logits, targets, weights):
    values = F.binary_cross_entropy_with_logits(
        logits.float(), targets.detach(), reduction="none"
    )
    return (values * weights).sum()


@lru_cache(None)
def _compiled(function):
    # AOTAutograd derives backward from the same reference expression, rather
    # than maintaining a separate handwritten gradient/normalization policy.
    return torch.compile(
        function,
        fullgraph=True,
        dynamic=True,
        backend="inductor",
        # Preserve eager BF16/FP16 rounding boundaries, particularly CE. The
        # default compiler may elide casts and silently change the objective.
        options={"emulate_precision_casts": True},
    )


def _should_compile(logits):
    backend = os.environ.get("LITESPEC_LOSS_BACKEND", "auto")
    if backend not in ("auto", "torch", "inductor"):
        raise ValueError(f"Unknown LITESPEC_LOSS_BACKEND: {backend}")
    enabled = logits.is_cuda and backend != "torch"
    # Amortize compilation/dispatch on vocab reductions. Tiny selector CE and
    # confidence heads remain eager unless explicitly requested for comparison.
    return enabled and (backend == "inductor" or logits.numel() >= 65536)


def _run(function, *args):
    if _should_compile(args[0]):
        return _compiled(function)(*args)
    return function(*args)


def weighted_cross_entropy(logits, labels, weights):
    """Return an additive numerator and detached, unweighted per-token CE."""
    # FP32 logits may have been cast from a BF16 projection. Compiling CE's
    # log-softmax can shift its FP32 gradient across BF16 rounding midpoints,
    # changing model gradients and optimizer updates despite a close loss.
    _should_compile(logits)  # Validate the configured backend on this path.
    return _ce(logits, labels, weights)


def weighted_ce_l1(logits, labels, teacher_logits, weights, *, diagnostics=False):
    """Return CE/L1 numerators, detached per-token CE and L1 distance."""
    if not _should_compile(logits):
        return _ce_l1(logits, labels, teacher_logits, weights, diagnostics)
    # Preserve native softmax rounding on both sides of abs(). Inductor can
    # round the two branches differently, making equal distributions nonzero
    # and changing L1's subgradient. Fuse the reduction after this boundary.
    target = torch.softmax(teacher_logits.detach().float(), dim=-1)
    draft = torch.softmax(logits.float(), dim=-1)
    # The dtype at this API cannot tell whether logits came from a BF16 head.
    # Keep native CE forward/backward for every dtype, and compile only L1's
    # reduction after the native probability boundary.
    ce, values = _ce(logits, labels, weights)
    # Keep the training kernel identical on log/non-log steps and checkpoint
    # replay. Logging-only maxima must not specialize its reduction/backward.
    l1 = _run(_l1_probs, draft, target, weights, False)
    result = (ce, l1[0], values, *l1[1:])
    if diagnostics:
        with torch.no_grad():
            return (*result, target.max(dim=-1).values, draft.max(dim=-1).values)
    return result


def weighted_kl(logits, teacher_logits, weights):
    """Additive teacher-to-student KL; teacher is always detached."""
    if _should_compile(logits):
        # Materialize the frozen teacher once so AOT does not replay its
        # normalization during backward.
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
            teacher_probs = teacher_log_probs.exp()
        return _run(_kl_student, logits, teacher_log_probs, teacher_probs, weights)
    return _run(_kl, logits, teacher_logits, weights)


def weighted_binary_cross_entropy(logits, targets, weights):
    if targets.dtype != torch.float32:
        return _bce(logits, targets, weights)
    return _run(_bce, logits, targets, weights)
