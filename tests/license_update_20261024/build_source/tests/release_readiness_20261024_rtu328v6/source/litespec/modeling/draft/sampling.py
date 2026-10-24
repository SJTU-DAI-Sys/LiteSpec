"""Shape-preserving categorical sampling primitives."""

import torch


def _sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    *,
    promote_logits: bool = True,
) -> torch.Tensor:
    """Return the target sampling distribution used by lossless verification."""

    if temperature <= 0:
        raise ValueError("temperature must be positive when computing probabilities")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    scores = (logits.float() if promote_logits else logits) / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None

    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def _sample_probs(probs: torch.Tensor) -> torch.Tensor:
    shape = probs.shape[:-1]
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(shape)


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    *,
    greedy_threshold: float = 0.0,
    promote_logits: bool = True,
) -> torch.Tensor:
    if temperature <= 0 or temperature < greedy_threshold:
        return torch.argmax(logits, dim=-1)
    return _sample_probs(
        _sampling_probs(
            logits, temperature, top_p, top_k, promote_logits=promote_logits
        )
    )
