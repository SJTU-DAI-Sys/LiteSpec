# Modified for LiteSpec: integrate ring attention with LiteSpec attention interfaces.
# Upstream attribution and license terms: THIRD_PARTY_NOTICES.md.

# adapt from https://github.com/feifeibear/long-context-attention/tree/main/yunchang
from .ring_flash_attn import (
    ring_flash_attn_func,
    ring_flash_attn_kvpacked_func,
    ring_flash_attn_qkvpacked_func,
)

__all__ = [
    "ring_flash_attn_func",
    "ring_flash_attn_kvpacked_func",
    "ring_flash_attn_qkvpacked_func",
]
