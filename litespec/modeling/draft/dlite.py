import logging
import time
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from .flex_attention import compile_friendly_flex_attention
from .registry import method_config, register_draft
from .rotary import FP32RotaryEmbedding as Qwen3RotaryEmbedding
from .sampling import sample as sample_tokens
from .target_cache import (
    _begin_recurrent_cache_transaction,
    _crop_target_cache,
    _new_target_cache,
    _restore_recurrent_cache_transaction,
)

SEQUENTIAL_HEAD_TYPES = ("rnn",)


def _sample_sequential_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return sample_tokens(logits, temperature, greedy_threshold=1e-5)


class DLiteSequentialHead(nn.Module):
    """Low-rank serial vocabulary head for DLite block prediction."""

    def __init__(
        self,
        *,
        head_type: str,
        vocab_size: int,
        sequential_rank: int,
        hidden_size: int,
        max_prediction_length: int,
    ) -> None:
        super().__init__()
        self.head_type = str(head_type).lower()
        self.vocab_size = int(vocab_size)
        self.sequential_rank = int(sequential_rank)
        self.hidden_size = int(hidden_size)
        self.max_prediction_length = int(max_prediction_length)

        if self.head_type not in SEQUENTIAL_HEAD_TYPES:
            raise ValueError(
                f"Unknown sequential head type {self.head_type!r}; "
                f"expected one of {SEQUENTIAL_HEAD_TYPES}."
            )
        if self.sequential_rank <= 0:
            raise ValueError(
                f"sequential_rank must be positive, got {self.sequential_rank}."
            )
        if self.max_prediction_length <= 0:
            raise ValueError(
                "max_prediction_length must be positive, got "
                f"{self.max_prediction_length}."
            )
        self.prev_token_embedding = nn.Embedding(self.vocab_size, self.sequential_rank)
        self.output_proj = nn.Linear(self.sequential_rank, self.vocab_size, bias=False)
        self.state_proj = nn.Linear(2 * self.sequential_rank, 2 * self.sequential_rank)
        self.state_hidden_mlp = nn.Linear(
            self.sequential_rank + self.hidden_size,
            self.sequential_rank,
        )

    def project_logits(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project low-rank head states to full-vocabulary logits."""
        return self.output_proj(latent_states)

    def _compute_step_latent(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prev_embeddings = self.prev_token_embedding(prev_token_ids.long())
        if state is None:
            state = torch.zeros_like(prev_embeddings)
        mem_inputs = torch.cat([state, prev_embeddings], dim=-1)
        gate_raw, candidate_raw = self.state_proj(mem_inputs).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        fused_inputs = torch.cat([new_state, hidden_states], dim=-1)
        return self.state_hidden_mlp(fused_inputs), new_state

    def forward_teacher_forcing(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return low-rank states for teacher-forced block predictions."""
        if hidden_states.shape[:-1] != prev_token_ids.shape:
            raise ValueError(
                "hidden_states and prev_token_ids leading shapes must match, "
                f"got {tuple(hidden_states.shape)} and "
                f"{tuple(prev_token_ids.shape)}."
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.size(-1)}."
            )

        prediction_length = hidden_states.size(-2)
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds configured "
                f"maximum {self.max_prediction_length}."
            )
        batch_shape = hidden_states.shape[:-2]
        state = torch.zeros(
            *batch_shape,
            self.sequential_rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        outputs: list[torch.Tensor] = []
        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids[..., position],
                hidden_states=hidden_states[..., position, :],
                state=state,
            )
            outputs.append(latent.unsqueeze(-2))
        if not outputs:
            return hidden_states.new_empty(
                *hidden_states.shape[:-2], 0, self.sequential_rank
            )
        return torch.cat(outputs, dim=-2)

    def sample_block_tokens(
        self,
        *,
        hidden_states: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Serially sample a block and return token IDs and sampling logits."""
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, prediction_length, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )

        bsz, prediction_length = hidden_states.shape[:2]
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds configured "
                f"maximum {self.max_prediction_length}."
            )
        state = hidden_states.new_zeros(bsz, self.sequential_rank)
        prev_token_ids = first_prev_token_ids.long()
        sampled_tokens: list[torch.Tensor] = []
        final_logits: list[torch.Tensor] = []
        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_states[:, position, :],
                state=state,
            )
            step_logits = self.project_logits(latent)
            next_token_ids = _sample_sequential_tokens(step_logits, float(temperature))
            sampled_tokens.append(next_token_ids.unsqueeze(1))
            final_logits.append(step_logits.unsqueeze(1))
            prev_token_ids = next_token_ids

        if not sampled_tokens:
            return (
                torch.empty(
                    bsz,
                    0,
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
                hidden_states.new_empty(bsz, 0, self.vocab_size),
            )
        return torch.cat(sampled_tokens, dim=1), torch.cat(final_logits, dim=1)


def _cuda_sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    return sample_tokens(
        logits, temperature, greedy_threshold=1e-5, promote_logits=False
    )


STOCHASTIC_VERIFICATION_MODES = ("match", "rejection")


def _validate_stochastic_verification_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in STOCHASTIC_VERIFICATION_MODES:
        raise ValueError(
            f"Unknown stochastic_verification_mode={mode!r}; expected one of "
            f"{STOCHASTIC_VERIFICATION_MODES}."
        )
    return mode


def _logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1),
            num_classes=logits.shape[-1],
        ).float()
    return torch.softmax(logits.float() / float(temperature), dim=-1)


def _sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    return torch.multinomial(probs.reshape(-1, vocab_size), num_samples=1).reshape(
        *probs.shape[:-1]
    )


def _sample_residual(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(
        residual_mass > 1e-8,
        residual / residual_mass.clamp_min(1e-8),
        target_probs,
    )
    return _sample_from_probs(residual)


def rejection_sample_verify(
    *,
    proposed_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    """Verify one stochastic draft block using speculative rejection sampling.

    Returns the number of accepted draft tokens and the correction/bonus token.
    This first implementation intentionally supports batch size one, matching
    the sequential acceptance semantics used by DSpARK.
    """
    if temperature < 1e-5:
        raise ValueError("rejection sampling requires temperature > 0.")
    if proposed_tokens.ndim != 2 or proposed_tokens.shape[0] != 1:
        raise ValueError(
            "rejection sampling currently requires proposed_tokens shape [1, K]."
        )
    proposal_count = proposed_tokens.shape[1]
    if draft_logits.shape[:2] != (1, proposal_count):
        raise ValueError(
            "draft_logits must have shape [1, K, vocab] matching proposed_tokens."
        )
    if target_logits.shape[:2] != (1, proposal_count + 1):
        raise ValueError("target_logits must have shape [1, K + 1, vocab].")
    if draft_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError("draft and target logits must use the same vocabulary.")

    target_probs = _logits_to_probs(target_logits, temperature)
    if proposal_count == 0:
        return 0, _sample_from_probs(target_probs[:, 0, :])

    draft_probs = _logits_to_probs(draft_logits, temperature)
    token_index = proposed_tokens.unsqueeze(-1)
    selected_target = (
        target_probs[:, :proposal_count, :]
        .gather(dim=-1, index=token_index)
        .squeeze(-1)
    )
    selected_draft = draft_probs.gather(dim=-1, index=token_index).squeeze(-1)
    accept_probs = torch.minimum(
        torch.ones_like(selected_target),
        selected_target / selected_draft.clamp_min(1e-20),
    )
    accepted_prefix = (
        (torch.rand_like(accept_probs) < accept_probs).to(torch.int64).cumprod(dim=1)
    )
    accepted_count = int(accepted_prefix.sum().item())

    if accepted_count < proposal_count:
        next_token = _sample_residual(
            target_probs[:, accepted_count, :],
            draft_probs[:, accepted_count, :],
        )
    else:
        next_token = _sample_from_probs(target_probs[:, proposal_count, :])
    return accepted_count, next_token


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    k_len = k.size(-2)
    q_cos = cos[..., -q_len:, :]
    q_sin = sin[..., -q_len:, :]
    k_cos = cos[..., -k_len:, :]
    k_sin = sin[..., -k_len:, :]
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


DLITE_ARCHITECTURE_VERSIONS = ("dlite_v1", "dlite_v2")
DLITE_ARCHITECTURE_VERSION = "dlite_v2"
DLITE_MODEL_ROLES = ("swa_teacher", "student")


def _infer_hs_embedding_offset(
    hidden_states: tuple | list, num_transformer_layers: int
) -> int:
    lt = len(hidden_states)
    if lt == num_transformer_layers:
        return 0
    if lt == num_transformer_layers + 1:
        return 1
    return 1 if lt > num_transformer_layers else 0


def build_target_layer_ids(
    num_transformer_layers: int, chs_num_layers: int
) -> list[int]:
    """Select exactly ``chs_num_layers`` layers, including the first and last."""
    L = num_transformer_layers
    S = int(chs_num_layers)
    if L < 2:
        raise ValueError(f"DLite requires at least 2 target layers, got {L}.")
    if not 2 <= S <= L:
        raise ValueError(f"chs_num_layers must be in [2, {L}], got {chs_num_layers}.")
    selected_layer_ids: set[int] = {0, L - 1}
    n_middle = S - 2
    lo, hi = 1, L - 2
    if n_middle > 0 and lo <= hi:
        interior_span = hi - lo
        num_interior_layers = hi - lo + 1
        n_take = min(n_middle, num_interior_layers)
        if n_take == 1:
            selected_layer_ids.add((lo + hi) // 2)
        else:
            for i in range(n_take):
                if interior_span == 0:
                    idx = lo
                else:
                    idx = lo + int(round(i * interior_span / (n_take - 1)))
                selected_layer_ids.add(int(max(lo, min(hi, idx))))
    target_layer_ids = sorted(selected_layer_ids)
    if len(target_layer_ids) != S:
        raise RuntimeError(
            f"Failed to select exactly {S} CHS layers from target depth {L}: {target_layer_ids}"
        )
    return target_layer_ids


def gather_pivot_multilayer_inference(
    hidden_states: tuple | list,
    target_layer_ids: list[int],
    token_index: int,
    num_transformer_layers: int,
) -> torch.Tensor:
    """Return (B, 1, S, H) pivot features for inference."""
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
    for layer_id in target_layer_ids:
        layer_h = hidden_states[layer_id + off]
        pieces.append(layer_h[:, token_index, :].unsqueeze(1))
    return torch.stack(pieces, dim=2)


def gather_hidden_layers_inference(
    hidden_states: tuple | list,
    layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Return selected target layers as ``(B, T, S, H)``."""
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    return torch.stack([hidden_states[layer_id + off] for layer_id in layer_ids], dim=2)


class Qwen3DLiteAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        cos, sin = position_embeddings

        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.config._attn_implementation == "flex_attention":
            # Call FlexAttention directly.  Transformers requests LSE during
            # training, but the FLASH backend cannot backpropagate through the
            # auxiliary LSE output.  DLite never consumes it.
            attn_output = compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kwargs.get("kernel_options"),
            ).transpose(1, 2)
            attn_weights = None
        else:
            attn_fn: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DLiteDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DLiteAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@register_draft
class DLiteDraftModel(Qwen3PreTrainedModel):
    method_config_key = "dlite_config"
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DLiteDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        if config.architectures is None:
            config.architectures = [type(self).__name__]
        if config.architectures != [type(self).__name__]:
            raise ValueError(
                f"{type(self).__name__} requires its own architecture/config"
            )
        dlite_config = method_config(config)
        architecture_version = dlite_config.get(
            "architecture_version", DLITE_ARCHITECTURE_VERSION
        )
        if architecture_version not in DLITE_ARCHITECTURE_VERSIONS:
            raise ValueError(
                "Incompatible DLite checkpoint/config architecture_version: "
                f"expected one of {DLITE_ARCHITECTURE_VERSIONS}, got "
                f"{architecture_version!r}."
            )
        self.architecture_version = str(architecture_version)
        self.model_role = str(dlite_config.get("model_role", "")).lower()
        if self.model_role not in DLITE_MODEL_ROLES:
            raise ValueError(
                f"model_role must be one of {DLITE_MODEL_ROLES}, got "
                f"{self.model_role!r}."
            )
        required_field_names = {
            "model_role",
            "chs_num_layers",
            "target_layer_ids",
            "sequential_head",
            "sequential_rank",
        }
        missing_field_names = sorted(required_field_names - dlite_config.keys())
        if missing_field_names:
            raise ValueError(
                f"dlite_config is missing required fields: {missing_field_names}"
            )
        self.swa_window_size = int(
            dlite_config["swa_window_size"] if self.is_teacher else 1
        )
        if self.swa_window_size < 1:
            raise ValueError(
                f"swa_window_size must be >= 1, got {self.swa_window_size}."
            )
        self.chs_num_layers = int(dlite_config["chs_num_layers"])
        self.target_layer_ids = [
            int(value) for value in dlite_config["target_layer_ids"]
        ]
        if len(self.target_layer_ids) != self.chs_num_layers:
            raise ValueError(
                "target_layer_ids length must equal chs_num_layers: "
                f"{len(self.target_layer_ids)} != {self.chs_num_layers}."
            )
        if self.target_layer_ids != sorted(set(self.target_layer_ids)):
            raise ValueError("target_layer_ids must be unique and strictly increasing.")
        invalid_layer_ids = [
            layer_id
            for layer_id in self.target_layer_ids
            if not 0 <= layer_id < int(config.num_target_layers)
        ]
        if invalid_layer_ids:
            raise ValueError(
                f"target_layer_ids {invalid_layer_ids} are outside [0, "
                f"{int(config.num_target_layers) - 1}]."
            )
        default_history_ids = [
            0,
            config.num_target_layers // 2,
            config.num_target_layers - 1,
        ]
        self.history_layer_ids = list(
            dlite_config["history_layer_ids"]
            if self.is_teacher
            else default_history_ids
        )
        if self.history_layer_ids != default_history_ids:
            raise ValueError(
                "history_layer_ids are fixed to target first/middle/last layers "
                f"{default_history_ids}, got {self.history_layer_ids}."
            )

        dlite_config["architecture_version"] = self.architecture_version
        dlite_config["model_role"] = self.model_role
        self.sequential_head_type = str(dlite_config["sequential_head"]).lower()
        self.sequential_rank = int(dlite_config["sequential_rank"])
        if self.sequential_head_type not in SEQUENTIAL_HEAD_TYPES:
            raise ValueError(
                f"Unknown sequential_head={self.sequential_head_type!r}; expected "
                f"one of {SEQUENTIAL_HEAD_TYPES}."
            )
        if self.sequential_rank <= 0:
            raise ValueError(
                f"sequential_rank must be positive, got {self.sequential_rank}."
            )
        self.sequential_head = DLiteSequentialHead(
            head_type=self.sequential_head_type,
            vocab_size=config.vocab_size,
            sequential_rank=self.sequential_rank,
            hidden_size=config.hidden_size,
            max_prediction_length=config.block_size - 1,
        )
        self._compiled_serial_sampler_cache: dict[tuple[float, bool], Callable] = {}
        config.dlite_config = dlite_config

        self.layers = nn.ModuleList(
            [
                Qwen3DLiteDecoderLayer(
                    config,
                    layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = dlite_config.get("mask_token_id", None)
        self.mask_embedding_mode = "vocab_row"
        self._mask_embedding_cache: Optional[tuple[int, torch.Tensor]] = None
        self._last_decode_stats = {}

        hidden_size = config.hidden_size
        if self.is_teacher:
            self.history_fuse = nn.Linear(3 * hidden_size, hidden_size, bias=False)
            self.history_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.layer_depth_embedding = nn.Embedding(config.num_target_layers, hidden_size)
        self.context_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        logging.getLogger(__name__).debug(
            f"DLite: architecture_version={self.architecture_version}, "
            f"model_role={self.model_role}, "
            f"swa_window_size={self.swa_window_size}, "
            f"fuse_slots={self.fuse_slot_count}, "
            f"chs_num_layers={self.chs_num_layers}, "
            f"condition_slots={self.condition_slot_count}, "
            f"target_layer_ids={self.target_layer_ids}, "
            f"history_layer_ids={self.history_layer_ids}, "
            f"sequential_head={self.sequential_head_type}, "
            f"sequential_rank={self.sequential_rank}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        return self.condition_slot_count + (
            self.fuse_slot_count if self.is_teacher else 0
        )

    @property
    def condition_slot_count(self) -> int:
        return self.chs_num_layers

    @property
    def is_teacher(self) -> bool:
        return self.model_role == "swa_teacher"

    @property
    def is_student(self) -> bool:
        return self.model_role == "student"

    @property
    def fuse_slot_count(self) -> int:
        return self.swa_window_size - 1

    @property
    def token_prefix_count(self) -> int:
        return 1

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def build_block_position_ids(
        self,
        anchor_positions: torch.Tensor,
        token_position_ids: torch.Tensor,
        token_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return role-specific RoPE ids; student queries start at 1 after CHS 0."""
        if token_position_ids.shape != token_keep_mask.shape:
            raise ValueError("token positions and keep mask must have equal shapes")
        pivot = (anchor_positions - 1).clamp(min=0)
        if self.is_teacher:
            context_positions = torch.cat(
                [
                    torch.arange(
                        self.fuse_slot_count,
                        device=anchor_positions.device,
                    ).view(1, 1, -1)
                    + anchor_positions.unsqueeze(-1)
                    - self.swa_window_size,
                    pivot.unsqueeze(-1).expand(-1, -1, self.chs_num_layers),
                ],
                dim=-1,
            ).clamp(min=0)
            draft_positions = token_position_ids
            mask_offsets = torch.arange(
                1, self.block_size, device=anchor_positions.device
            ).view(1, 1, -1)
            draft_positions = torch.cat(
                [draft_positions, anchor_positions.unsqueeze(-1) + mask_offsets],
                dim=-1,
            )
        else:
            has_tokens = token_keep_mask.any(dim=-1)
            sentinel = torch.iinfo(token_position_ids.dtype).max
            first_valid = torch.where(
                token_keep_mask,
                token_position_ids,
                torch.full_like(token_position_ids, sentinel),
            ).amin(dim=-1)
            origin = torch.where(has_tokens, first_valid, pivot)
            local_offset = 0 if self.architecture_version == "dlite_v1" else 1
            local_tokens = torch.where(
                token_keep_mask,
                token_position_ids - origin.unsqueeze(-1) + local_offset,
                torch.zeros_like(token_position_ids),
            )
            context_positions = (
                (pivot - origin)
                .clamp(min=0)
                .unsqueeze(-1)
                .expand(-1, -1, self.chs_num_layers)
            )
            mask_offsets = torch.arange(
                1, self.block_size, device=anchor_positions.device
            ).view(1, 1, -1)
            local_masks = (
                anchor_positions.unsqueeze(-1)
                + mask_offsets
                - origin.unsqueeze(-1)
                + local_offset
            )
            draft_positions = torch.cat([local_tokens, local_masks], dim=-1)
        return (
            context_positions.reshape(anchor_positions.shape[0], -1),
            draft_positions.reshape(anchor_positions.shape[0], -1),
        )

    @property
    def draft_block_len(self) -> int:
        """Parallel draft slots per anchor (1 anchor + remaining MASK tokens)."""
        return self.block_size

    @property
    def draft_query_length(self) -> int:
        return self.token_prefix_count + self.proposal_length

    @property
    def proposal_length(self) -> int:
        """Draft tokens proposed after the anchor; total span is block_size."""
        return self.block_size - 1

    @property
    def max_verify_block_size(self) -> int:
        """Anchor-inclusive target verification window (equals config block_size)."""
        return self.proposal_length + 1

    def _prediction_hidden(self, block_hidden: torch.Tensor) -> torch.Tensor:
        """Return one hidden state for each proposed token."""
        return block_hidden[:, -self.proposal_length :, :]

    def fuse_history_hidden(self, history_hidden: torch.Tensor) -> torch.Tensor:
        """Fuse ``(..., 3, H)`` target states into one vector per token."""
        if not self.is_teacher:
            raise RuntimeError("History fusion is only available on teacher models.")
        if history_hidden.ndim < 3:
            raise ValueError(
                "history_hidden must have shape (..., 3, H); got "
                f"{tuple(history_hidden.shape)}."
            )
        if history_hidden.shape[-2] != 3:
            raise ValueError(
                "history_hidden must have exactly three selected layers; got shape "
                f"{tuple(history_hidden.shape)}."
            )
        expected_in = self.history_fuse.in_features
        if history_hidden.shape[-1] * 3 != expected_in:
            raise ValueError(
                "history_hidden feature dimension does not match history_fuse; "
                f"got H={history_hidden.shape[-1]} (flattened {history_hidden.shape[-1] * 3}), "
                f"expected {expected_in}."
            )
        flat = history_hidden.flatten(start_dim=-2)
        return self.history_norm(self.history_fuse(flat))

    def _apply_chs_depth_embedding(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Add target-layer identity to current CHS hidden states."""
        _, _, num_chs_slots, hidden_size = target_hidden.shape
        if num_chs_slots != self.chs_num_layers:
            raise ValueError(
                f"Expected {self.chs_num_layers} current CHS slots, got {num_chs_slots}."
            )
        depth_ids = torch.tensor(
            self.target_layer_ids, device=target_hidden.device, dtype=torch.long
        )
        depth_emb = self.layer_depth_embedding(depth_ids).view(
            1, 1, self.chs_num_layers, hidden_size
        )
        return self.context_norm(target_hidden + depth_emb)

    def build_inference_query_embeddings(
        self,
        embed_tokens: nn.Module,
        draft_input_ids: torch.Tensor,
        predecessor_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build versioned real-token queries followed by ``B-1`` MASK queries."""
        if draft_input_ids.ndim != 2 or draft_input_ids.shape[1] != self.block_size:
            raise ValueError(
                "draft_input_ids must have shape "
                f"(batch, {self.block_size}), got {tuple(draft_input_ids.shape)}."
            )
        anchor_ids = draft_input_ids[:, :1]
        if predecessor_token_ids is not None:
            raise ValueError(
                "DLite student queries do not accept predecessor_token_ids."
            )
        real_token_ids = anchor_ids
        real_embeddings = embed_tokens(real_token_ids)
        mask_ids = draft_input_ids[:, 1:]
        weight = getattr(embed_tokens, "weight", None)
        if weight is not None and bool((mask_ids >= weight.shape[0]).any()):
            if self.mask_token_id is None or bool(
                (mask_ids != self.mask_token_id).any()
            ):
                raise ValueError(
                    "Draft query contains an out-of-vocabulary non-MASK id."
                )
            if self.mask_embedding_mode == "vocab_row":
                raise ValueError(
                    "Checkpoint requires vocab_row MASK embeddings, but "
                    f"mask_token_id={self.mask_token_id} is outside the provided "
                    f"embedding with {weight.shape[0]} rows."
                )
            weight_ptr = weight.data_ptr()
            cached = self._mask_embedding_cache
            if (
                cached is None
                or cached[0] != weight_ptr
                or cached[1].device != weight.device
                or cached[1].dtype != weight.dtype
            ):
                cached = (weight_ptr, weight.detach().mean(dim=0))
                self._mask_embedding_cache = cached
            mask_embeddings = (
                cached[1]
                .view(1, 1, -1)
                .expand(mask_ids.shape[0], mask_ids.shape[1], -1)
            )
        else:
            mask_embeddings = embed_tokens(mask_ids)
        return torch.cat([real_embeddings, mask_embeddings], dim=1)

    def build_inference_current_chs(
        self,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build current CHS context for inference."""
        if target_hidden.ndim != 4 or target_hidden.shape[2] != self.chs_num_layers:
            raise ValueError(
                f"target_hidden must have shape (B,N,{self.chs_num_layers},H), got "
                f"{tuple(target_hidden.shape)}."
            )
        if target_hidden.shape[1] != 1:
            raise ValueError("Inference current CHS expects exactly one anchor block.")
        return target_hidden

    def _build_shared_context_kv(
        self,
        shared_history: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate per-anchor CHS slots and one shared history sequence."""
        if shared_history.ndim != 3:
            raise ValueError(
                "shared_history must have shape (B,T,H), got "
                f"{tuple(shared_history.shape)}."
            )
        current_chs_hidden = self._apply_chs_depth_embedding(target_hidden)
        chs_flat = current_chs_hidden.reshape(
            current_chs_hidden.shape[0], -1, current_chs_hidden.shape[-1]
        )
        if self.is_teacher:
            return torch.cat([shared_history, chs_flat], dim=1)
        return chs_flat

    def _fuse_target_hidden(
        self,
        target_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build the configured per-block CHS/history context."""
        bsz, num_blocks, _, _ = target_hidden.shape
        if history_hidden.shape[:2] != (bsz, num_blocks):
            raise ValueError(
                "history_hidden batch/block dimensions must match target_hidden: "
                f"{tuple(history_hidden.shape)} vs {tuple(target_hidden.shape)}."
            )
        current_chs_hidden = self._apply_chs_depth_embedding(target_hidden)
        context_hidden = (
            torch.cat([history_hidden, current_chs_hidden], dim=2)
            if self.is_teacher
            else current_chs_hidden
        )
        return context_hidden.reshape(
            bsz, num_blocks * context_hidden.shape[2], current_chs_hidden.shape[-1]
        )

    def fuse_target_output_history(self, hidden_states: tuple | list) -> torch.Tensor:
        """Fuse the three configured target layers for every returned token."""
        selected = gather_hidden_layers_inference(
            hidden_states,
            self.history_layer_ids,
            self.config.num_target_layers,
        )
        return self.fuse_history_hidden(selected)

    def build_inference_context(
        self,
        recent_condition_hidden: torch.Tensor,
        current_target_hidden: torch.Tensor,
        anchor_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build inference KV history and role-specific RoPE positions."""
        if recent_condition_hidden.ndim != 3:
            raise ValueError(
                "recent_condition_hidden must have shape (B,T,H), got "
                f"{tuple(recent_condition_hidden.shape)}."
            )
        if self.is_teacher and recent_condition_hidden.shape[1] < 1:
            raise ValueError(
                "recent_condition_hidden must include the current pivot in fuse mode."
            )
        if current_target_hidden.shape[:3] != (
            recent_condition_hidden.shape[0],
            1,
            self.condition_slot_count,
        ):
            raise ValueError(
                "current_target_hidden must have shape (B,1,S,H) matching the "
                f"condition tensor; got {tuple(current_target_hidden.shape)}."
            )
        anchor_position = int(anchor_position)
        if self.is_teacher:
            recent_condition_hidden = recent_condition_hidden[
                :, -self.swa_window_size :, :
            ]
            history_source = recent_condition_hidden[:, :-1, :]
        else:
            history_source = recent_condition_hidden[:, :0, :]
        history = history_source.unsqueeze(1)
        token_pos = (
            torch.arange(
                anchor_position - self.token_prefix_count + 1,
                anchor_position + 1,
                device=recent_condition_hidden.device,
                dtype=torch.long,
            )
            .view(1, 1, -1)
            .expand(recent_condition_hidden.shape[0], -1, -1)
        )
        token_keep = torch.ones(
            recent_condition_hidden.shape[0],
            1,
            self.token_prefix_count,
            dtype=torch.bool,
            device=recent_condition_hidden.device,
        )
        anchors = torch.full(
            (recent_condition_hidden.shape[0], 1),
            anchor_position,
            dtype=torch.long,
            device=recent_condition_hidden.device,
        )
        if self.is_teacher:
            history_len = history.shape[2]
            history_positions = (
                torch.arange(
                    anchor_position - history_len - 1,
                    anchor_position - 1,
                    device=recent_condition_hidden.device,
                )
                .view(1, -1)
                .expand(recent_condition_hidden.shape[0], -1)
            )
            chs_positions = torch.full(
                (recent_condition_hidden.shape[0], self.chs_num_layers),
                anchor_position - 1,
                dtype=torch.long,
                device=recent_condition_hidden.device,
            )
            context_positions = torch.cat([history_positions, chs_positions], dim=-1)
            mask_positions = (
                torch.arange(
                    anchor_position + 1,
                    anchor_position + self.block_size,
                    device=recent_condition_hidden.device,
                )
                .view(1, -1)
                .expand(recent_condition_hidden.shape[0], -1)
            )
            draft_positions = torch.cat(
                [
                    token_pos.reshape(recent_condition_hidden.shape[0], -1),
                    mask_positions,
                ],
                dim=-1,
            )
        else:
            context_positions, draft_positions = self.build_block_position_ids(
                anchor_positions=anchors,
                token_position_ids=token_pos,
                token_keep_mask=token_keep,
            )
        return history, context_positions, draft_positions

    def initialize_inference_condition(
        self,
        target_hidden_states: tuple | list,
        pivot_index: int = -1,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Retain fused target history for a teacher; students need no KV history."""
        if self.is_teacher:
            condition_source = self.fuse_target_output_history(target_hidden_states)
            keep_length = self.swa_window_size
        else:
            reference = target_hidden_states[-1]
            return reference[:, :0, :]
        seq_len = condition_source.shape[1]
        pivot_index = int(pivot_index)
        if pivot_index < 0:
            pivot_index += seq_len
        if not 0 <= pivot_index < seq_len:
            raise IndexError(
                f"pivot_index={pivot_index} is out of range for length {seq_len}."
            )
        through_pivot = condition_source[:, : pivot_index + 1, :]
        return (
            through_pivot[:, -keep_length:, :]
            if keep_length
            else through_pivot[:, :0, :]
        )

    def update_inference_condition(
        self,
        recent_condition_hidden: torch.Tensor,
        target_hidden_states: tuple | list,
        pivot_index: int,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append newly verified fused history for teacher inference."""
        if self.is_teacher:
            condition_new = self.fuse_target_output_history(target_hidden_states)
            keep_length = self.swa_window_size
        else:
            return recent_condition_hidden
        pivot_index = int(pivot_index)
        if not 0 <= pivot_index < condition_new.shape[1]:
            raise IndexError(
                f"pivot_index={pivot_index} is out of range for target output "
                f"length {condition_new.shape[1]}."
            )
        through_new_pivot = condition_new[:, : pivot_index + 1, :]
        condition = torch.cat([recent_condition_hidden, through_new_pivot], dim=1)
        return condition[:, -keep_length:, :] if keep_length else condition[:, :0, :]

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        history_hidden: Optional[torch.Tensor] = None,
        shared_history: Optional[torch.Tensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        if shared_history is not None:
            if target_hidden is None or target_hidden.ndim != 4:
                raise ValueError(
                    "shared_history training requires target_hidden shaped (B,N,S,H)."
                )
            target_hidden = self._build_shared_context_kv(shared_history, target_hidden)
        else:
            assert target_hidden is not None and target_hidden.ndim == 4
            assert history_hidden is not None and history_hidden.ndim == 4
            target_hidden = self._fuse_target_hidden(target_hidden, history_hidden)
        noise_len = hidden_states.shape[1]
        if position_ids.shape[1] != noise_len:
            draft_pos_ids = position_ids[:, -noise_len:]
        else:
            draft_pos_ids = position_ids

        rotary_position_ids = (
            rotary_position_ids if rotary_position_ids is not None else draft_pos_ids
        )
        # Qwen3RotaryEmbedding only reads x.device and x.dtype. Passing the
        # existing activation avoids allocating a dense (B, KV_LEN, H) dummy;
        position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=draft_pos_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def sample_draft_tokens(
        self,
        *,
        draft_hidden: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        compile_serial_head: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample standard DLite draft positions using configured head semantics."""
        if compile_serial_head:
            cache_key = (float(temperature), False)
            compiled_sampler = self._compiled_serial_sampler_cache.get(cache_key)
            if compiled_sampler is None:
                sequential_head = self.sequential_head
                fixed_temperature = float(temperature)

                def serial_sampler(
                    hidden_states: torch.Tensor,
                    previous_ids: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return sequential_head.sample_block_tokens(
                        hidden_states=hidden_states,
                        first_prev_token_ids=previous_ids,
                        temperature=fixed_temperature,
                    )

                compiled_sampler = torch.compile(
                    serial_sampler,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._compiled_serial_sampler_cache[cache_key] = compiled_sampler
            return compiled_sampler(draft_hidden, first_prev_token_ids)
        return self.sequential_head.sample_block_tokens(
            hidden_states=draft_hidden,
            first_prev_token_ids=first_prev_token_ids,
            temperature=temperature,
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        decode_timing_after_first_token: bool = False,
        verify_block_size: Optional[int] = None,
        stochastic_verification_mode: str = "match",
        compile_serial_head: bool = False,
        strict_greedy_verification: bool = False,
        target_cache: Optional[Cache] = None,
    ):
        self.eval()
        stochastic_verification_mode = _validate_stochastic_verification_mode(
            stochastic_verification_mode
        )
        self._last_decode_stats = {
            "accept_lengths": [],
            "decode_wall_time": 0.0,
            "steps": 0,
            "verification_mode": stochastic_verification_mode,
            "compile_serial_head": bool(compile_serial_head),
            "strict_greedy_verification": bool(strict_greedy_verification),
            "strict_verification_steps": 0,
        }
        bsz = input_ids.shape[0]
        use_rejection_sampling = (
            temperature >= 1e-5 and stochastic_verification_mode == "rejection"
        )
        if use_rejection_sampling and bsz != 1:
            raise ValueError(
                "stochastic rejection sampling currently requires batch size 1."
            )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        draft_block_len = self.draft_block_len
        proposal_length = self.proposal_length
        verify_block_size = (
            proposal_length + 1 if verify_block_size is None else int(verify_block_size)
        )
        if not 1 <= verify_block_size <= proposal_length + 1:
            raise ValueError(
                f"verify_block_size must be in [1, {proposal_length + 1}], got "
                f"{verify_block_size}"
            )
        output_ids = torch.full(
            (bsz, max_length + proposal_length + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = (
            torch.arange(output_ids.shape[1], device=target.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        past_key_values_target = (
            _new_target_cache(target) if target_cache is None else target_cache
        )

        # Prefill stage (not included in decode wall time)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
        )
        recent_condition_hidden = self.initialize_inference_condition(
            output.hidden_states,
        )

        # Decode stage: single cuda-synced wall clock (draft + target + bookkeeping)
        decode_start: float | None = (
            None if decode_timing_after_first_token else _cuda_sync_time(target.device)
        )
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            draft_input_ids = output_ids[:, start : start + draft_block_len].clone()
            noise_embedding = self.build_inference_query_embeddings(
                target.get_input_embeddings(),
                draft_input_ids,
            )
            current_target_hidden = self.build_inference_current_chs(
                target_hidden,
            )
            history_hidden, ctx_pos_part, draft_block_pos = (
                self.build_inference_context(
                    recent_condition_hidden,
                    current_target_hidden,
                    start,
                )
            )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_hidden = self(
                target_hidden=current_target_hidden,
                history_hidden=history_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_block_pos,
                rotary_position_ids=full_rotary,
                is_causal=False,
            )
            draft_hidden = self._prediction_hidden(block_hidden)
            draft_temperature = temperature if use_rejection_sampling else 0.0
            sampled_draft_tokens, draft_logits = self.sample_draft_tokens(
                draft_hidden=draft_hidden,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=draft_temperature,
                compile_serial_head=compile_serial_head,
            )
            all_verify_output_ids = torch.cat(
                [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
            )

            # The draft always consumes and predicts the full configured block.  Only
            # the requested prefix is sent to the target; the remaining draft tokens
            # are deliberately discarded before verification.
            verify_output_ids = all_verify_output_ids[:, :verify_block_size]
            verify_position_ids = position_ids[:, start : start + verify_block_size]
            use_strict_greedy = strict_greedy_verification and temperature < 1e-5
            if use_strict_greedy:
                recurrent_snapshot = None
                acceptance_length = 0
                for verify_index in range(verify_block_size):
                    output = target(
                        verify_output_ids[:, verify_index : verify_index + 1],
                        position_ids=verify_position_ids[
                            :, verify_index : verify_index + 1
                        ],
                        past_key_values=past_key_values_target,
                        use_cache=True,
                        logits_to_keep=1,
                        output_hidden_states=True,
                    )
                    next_token = sample(output.logits, temperature)
                    if verify_index == verify_block_size - 1:
                        break
                    if not bool(
                        (
                            verify_output_ids[:, verify_index + 1]
                            == next_token.reshape(-1)
                        ).all()
                    ):
                        break
                    acceptance_length += 1
                verification_cache_committed = True
                self._last_decode_stats["strict_verification_steps"] += 1
            else:
                recurrent_snapshot = _begin_recurrent_cache_transaction(
                    past_key_values_target
                )
                output = target(
                    verify_output_ids,
                    position_ids=verify_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                )
                verification_cache_committed = False

            if use_rejection_sampling:
                proposal_count = verify_block_size - 1
                acceptance_length, next_token = rejection_sample_verify(
                    proposed_tokens=verify_output_ids[:, 1:],
                    draft_logits=draft_logits[:, :proposal_count, :],
                    target_logits=output.logits,
                    temperature=temperature,
                )
            elif not use_strict_greedy:
                posterior = sample(output.logits, temperature)
                acceptance_lengths_per_row = (
                    (verify_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)
                )
                acceptance_length = int(acceptance_lengths_per_row.min().item())
                next_token = posterior[:, acceptance_length]
            output_ids[:, start : start + acceptance_length + 1] = verify_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = next_token
            kept_verify_tokens = acceptance_length + 1
            cache_length_after_verify = start + verify_block_size
            if verification_cache_committed:
                pass
            elif recurrent_snapshot is None:
                _crop_target_cache(
                    past_key_values_target,
                    current_length=cache_length_after_verify,
                    tokens_to_remove=verify_block_size - kept_verify_tokens,
                )
            elif kept_verify_tokens == verify_block_size:
                _crop_target_cache(
                    past_key_values_target,
                    current_length=cache_length_after_verify,
                    tokens_to_remove=0,
                )
            else:
                _crop_target_cache(
                    past_key_values_target,
                    current_length=cache_length_after_verify,
                    tokens_to_remove=verify_block_size,
                )
                _restore_recurrent_cache_transaction(recurrent_snapshot)
                target(
                    verify_output_ids[:, :kept_verify_tokens],
                    position_ids=verify_position_ids[:, :kept_verify_tokens],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=False,
                )
                _crop_target_cache(
                    past_key_values_target,
                    current_length=start + kept_verify_tokens,
                    tokens_to_remove=0,
                )
            start += kept_verify_tokens
            pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
            recent_condition_hidden = self.update_inference_condition(
                recent_condition_hidden,
                output.hidden_states,
                pivot_index,
            )
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
            )
            acceptance_lengths.append(acceptance_length + 1)
            self._last_decode_stats["accept_lengths"].append(acceptance_length + 1)
            self._last_decode_stats["steps"] += 1

            if decode_timing_after_first_token and decode_start is None:
                decode_start = _cuda_sync_time(target.device)

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        if decode_start is None:
            decode_start = _cuda_sync_time(target.device)
        decode_wall_time = _cuda_sync_time(target.device) - decode_start
        self._last_decode_stats["decode_wall_time"] = decode_wall_time

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
