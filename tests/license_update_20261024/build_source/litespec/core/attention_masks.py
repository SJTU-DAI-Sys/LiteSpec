"""Attention-mask builders and physical shape policy for draft training."""

from __future__ import annotations

import math

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask

    from litespec.modeling.draft.flex_attention import (
        FLASH_KV_TILE,
        FLASH_Q_TILE,
        create_flex_block_mask,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    BlockMask = None
    create_flex_block_mask = None
    FLASH_Q_TILE = 256
    FLASH_KV_TILE = 128
    FLEX_ATTENTION_AVAILABLE = False

if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


def flash_block_alignment(query_len: int, kv_per_block: int) -> int:
    q = FLASH_Q_TILE // math.gcd(FLASH_Q_TILE, int(query_len))
    kv = FLASH_KV_TILE // math.gcd(FLASH_KV_TILE, int(kv_per_block))
    return math.lcm(q, kv)


def flash_padded_block_width(width: int, query_len: int, chs_slots: int) -> int:
    alignment = flash_block_alignment(query_len, query_len + chs_slots)
    return math.ceil(int(width) / alignment) * alignment


def flash_kv_padding(kv_len: int) -> int:
    return (FLASH_KV_TILE - int(kv_len) % FLASH_KV_TILE) % FLASH_KV_TILE


# Q:  [block_0 .. block_{N-1}], each block has B tokens (anchor + MASK*)
# KV: [target 0..S-1 | draft_0 | ... | draft_{N-1}]
#
# Query in block i, offset o, anchor a:
#   target:  kv < a  (and kv < S)
#   draft:   only own block
#   other blocks / target[a:]: invisible
#   block_keep_mask=False: see nothing
#
# If sliding_window=W:
#   target further restricted to [a + o - (W-1), a)
#   draft becomes causal: draft_offset <= o
# Without W, intra-block draft is full (not causal).
def create_dflash_sdpa_mask(
    anchor_positions, block_keep_mask, S, block_size, device, sliding_window=None
):
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    bsz, num_blocks = anchor_positions.shape
    q_len, kv_len = num_blocks * block_size, S + num_blocks * block_size
    q = torch.arange(q_len, device=device).view(1, 1, -1, 1)
    kv = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
    q_block, q_offset = q // block_size, q % block_size
    anchors = anchor_positions.view(bsz, 1, num_blocks, 1).repeat_interleave(
        block_size, 2
    )
    context_mask = (kv < S) & (kv < anchors)
    if sliding_window is not None:
        context_mask &= kv >= anchors + q_offset - (sliding_window - 1)
    draft_mask = (kv >= S) & (q_block == (kv - S) // block_size)
    if sliding_window is not None:
        draft_mask &= (kv - S) % block_size <= q_offset
    valid = block_keep_mask.view(bsz, 1, num_blocks, 1).repeat_interleave(block_size, 2)
    return (context_mask | draft_mask) & valid


# Same visibility as create_dflash_sdpa_mask.
# Q_LEN = N*B, KV_LEN = S + N*B.
def create_dflash_flex_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    flex_backend="TRITON",
    sliding_window=None,
):
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention is not available")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    bsz, num_blocks = anchor_positions.shape

    def mask_mod(b, h, q_idx, kv_idx):
        qb, qo = q_idx // block_size, q_idx % block_size
        safe_block_indices = qb.clamp(max=num_blocks - 1)
        anchor = anchor_positions[b, safe_block_indices]
        context_mask = (kv_idx < S) & (kv_idx < anchor)
        if sliding_window is not None:
            context_mask &= kv_idx >= anchor + qo - (sliding_window - 1)
        draft_mask = (kv_idx >= S) & (qb == (kv_idx - S) // block_size)
        if sliding_window is not None:
            draft_mask &= (kv_idx - S) % block_size <= qo
        return (
            (context_mask | draft_mask)
            & block_keep_mask[b, safe_block_indices]
            & (qb < num_blocks)
        )

    return create_flex_block_mask(
        mask_mod,
        B=bsz,
        H=None,
        Q_LEN=num_blocks * block_size,
        KV_LEN=S + num_blocks * block_size,
        device=device,
        backend=flex_backend,
    )


# Q:  [Q_0 .. Q_{N-1}], each block has query_len = 1 anchor + (B-1) MASK
# teacher KV: [shared history 0..S-1 | CHS_0..CHS_{N-1} | Q_0..Q_{N-1}]
# student KV: [CHS_0..CHS_{N-1} | Q_0..Q_{N-1}]   # prefix=0, no history
#
# Query in block i, anchor a:
#   history (teacher only): [a-W, a-2]   # W-1 fused tokens; a-1 is CHS
#   CHS: own-block slots only
#   query: own-block, non-padded tokens (full, not causal)
#   other blocks: invisible
#
# Padded block (block_keep_mask=False): own CHS only, so the row is not empty.
def create_dlite_flex_mask(
    *,
    model_role,
    anchor_positions,
    block_keep_mask,
    token_keep_mask,
    seq_len,
    swa_window_size,
    chs_slots,
    query_len,
    device,
    flex_backend="TRITON",
):
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention is not available")
    batch, blocks = anchor_positions.shape
    total_chs, total_q = blocks * chs_slots, blocks * query_len
    prefix = int(seq_len) if model_role == "swa_teacher" else 0
    kv_len, chs_start, q_start = (
        prefix + total_chs + total_q,
        prefix,
        prefix + total_chs,
    )
    token_count, max_block = token_keep_mask.size(-1), max(blocks - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
        qb = q_idx // query_len
        safe_qb = qb.clamp(min=0, max=max_block)
        valid_block = block_keep_mask[b, safe_qb] & (qb <= max_block)
        anchor = anchor_positions[b, safe_qb]
        history = (
            (kv_idx < prefix)
            & (kv_idx >= anchor - int(swa_window_size))
            & (kv_idx <= anchor - 2)
        )
        is_chs = (kv_idx >= chs_start) & (kv_idx < q_start)
        chs_valid = is_chs & ((kv_idx - chs_start) // chs_slots == qb)
        is_q = kv_idx >= q_start
        kv_qb, slot = (kv_idx - q_start) // query_len, (kv_idx - q_start) % query_len
        safe_kb = kv_qb.clamp(min=0, max=max_block)
        safe_slot = slot.clamp(min=0, max=query_len - 1)
        padded_token = (safe_slot < token_count) & ~token_keep_mask[
            b, safe_kb, safe_slot.clamp(max=token_count - 1)
        ]
        query = is_q & (kv_qb == qb) & ~padded_token
        return (valid_block & (history | chs_valid | query)) | (
            chs_valid & ~valid_block & (qb <= max_block)
        )

    return create_flex_block_mask(
        mask_mod,
        B=batch,
        H=None,
        Q_LEN=total_q,
        KV_LEN=kv_len,
        device=device,
        backend=flex_backend,
    )


__all__ = [
    "BlockMask",
    "FLEX_ATTENTION_AVAILABLE",
    "create_dflash_flex_mask",
    "create_dflash_sdpa_mask",
    "create_dlite_flex_mask",
    "flash_block_alignment",
    "flash_kv_padding",
    "flash_padded_block_width",
]
