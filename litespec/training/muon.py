"""FP32-sharded Muon with bounded, shape-batched all-to-all orthogonalization.

Math follows torch.optim.Muon in PyTorch 2.9.1 (including its actual lerp
momentum convention). FSDP private APIs belong in fsdp_layout, not here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import torch
import torch.distributed as dist

MUON_DEFAULTS = dict(
    lr=0.02,
    momentum=0.95,
    weight_decay=0.1,
    nesterov=True,
    ns_steps=5,
    ns_coefficients=(3.4445, -4.775, 2.0315),
    eps=1e-7,
    adjust_lr_fn="original",
    matrices_per_rank=1,
    low_memory=False,
    adamw_lr=0.0004,
    adamw_betas=(0.9, 0.999),
    adamw_weight_decay=0.1,
)
# Newton-Schulz / nesterov / shape-LR internals stay at MUON_DEFAULTS.
MUON_CLI_KEYS = (
    "lr",
    "momentum",
    "weight_decay",
    "matrices_per_rank",
    "low_memory",
    "adamw_lr",
    "adamw_weight_decay",
)


def validate_muon_config(config):
    config = {**MUON_DEFAULTS, **config}
    unknown = set(config) - set(MUON_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown Muon settings: {sorted(unknown)}")
    for key in ("lr", "adamw_lr", "eps"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"muon_{key} must be finite and positive")
    for key in ("weight_decay", "adamw_weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"muon_{key} must be finite and non-negative")
    if not 0 <= config["momentum"] < 1:
        raise ValueError("muon_momentum must be in [0, 1)")
    config["adamw_betas"] = tuple(config["adamw_betas"])
    if len(config["adamw_betas"]) != 2 or any(
        not 0 <= b < 1 for b in config["adamw_betas"]
    ):
        raise ValueError("muon_adamw_betas must contain two values in [0, 1)")
    config["ns_coefficients"] = tuple(config["ns_coefficients"])
    if len(config["ns_coefficients"]) != 3 or not all(
        map(math.isfinite, config["ns_coefficients"])
    ):
        raise ValueError("muon_ns_coefficients must contain three finite values")
    for key in ("ns_steps", "matrices_per_rank"):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] < 1
        ):
            raise ValueError(f"muon_{key} must be a positive integer")
    if config["ns_steps"] >= 100:
        raise ValueError("muon_ns_steps must be less than 100")
    if config["adjust_lr_fn"] not in ("original", "match_rms_adamw"):
        raise ValueError("Invalid muon_adjust_lr_fn")
    if not isinstance(config["nesterov"], bool):
        raise ValueError("muon_nesterov must be a boolean")
    if not isinstance(config["low_memory"], bool):
        raise ValueError("muon_low_memory must be a boolean")
    return config


def muon_config_from_args(args):
    if not getattr(args, "use_muon", False):
        return None
    return validate_muon_config(
        {key: getattr(args, f"muon_{key}") for key in MUON_CLI_KEYS}
    )


def newton_schulz(update, coefficients=(3.4445, -4.775, 2.0315), steps=5, eps=1e-7):
    """Orthogonalize each trailing 2-D matrix independently, in BF16."""
    x = update.bfloat16()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        if x.ndim == 2:
            polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.addmm(x, polynomial, x, beta=a)
        else:
            polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.baddbmm(x, polynomial, x, beta=a)
    return x.transpose(-2, -1) if transpose else x


def adjusted_lr(lr, shape, rule):
    rows, cols = shape
    scale = (
        math.sqrt(max(1, rows / cols))
        if rule == "original"
        else 0.2 * math.sqrt(max(rows, cols))
    )
    return lr * scale


@dataclass
class Exchange:
    indices: list[int]
    owners: list[int]
    send: torch.Tensor | None
    receive: torch.Tensor | None
    send_sizes: list[int]
    receive_sizes: list[int]
    work: object = None


class ShardedMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        layouts,
        *,
        process_group=None,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
        nesterov=True,
        ns_steps=5,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        adjust_lr_fn="original",
        matrices_per_rank=1,
        low_memory=False,
    ):
        validate_muon_config(
            dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                ns_steps=ns_steps,
                ns_coefficients=ns_coefficients,
                eps=eps,
                adjust_lr_fn=adjust_lr_fn,
                matrices_per_rank=matrices_per_rank,
                low_memory=low_memory,
            )
        )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=tuple(ns_coefficients),
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("ShardedMuon expects one registered parameter group")
        self.layouts = list(layouts)
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        self.world = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.matrices_per_rank = matrices_per_rank
        self.low_memory = low_memory
        self.params = self.param_groups[0]["params"]
        if len(self.params) != len(self.layouts):
            raise ValueError("Muon parameter/layout count mismatch")
        for p, layout in zip(self.params, self.layouts, strict=True):
            if len(layout.spec.shape) != 2 or len(layout.shards) != self.world:
                raise ValueError(f"Invalid Muon matrix layout: {layout.spec.name}")
            if p.dtype != torch.float32 or p.numel() != layout.shards[self.rank][1]:
                raise ValueError("Muon requires FP32 local master shards")
        if matrices_per_rank < 1:
            raise ValueError("matrices_per_rank must be positive")
        self.communication_calls = 0
        self._prefetch_stream = None
        self._communication_ready = False

    @torch.no_grad()
    def prepare_communication(self):
        """Warm P2P connections before FP32 gradients/momentum consume headroom.

        NCCL allocations bypass PyTorch's caching allocator. Return unused cache
        once, then touch every peer while optimizer temporaries are still absent.
        This cannot remove NCCL's persistent connection buffers.
        """
        if self._communication_ready:
            return
        device = self.params[0].device
        if device.type == "cuda" and self.world > 1:
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                send = torch.zeros(self.world, device=device, dtype=torch.bfloat16)
                receive = torch.empty_like(send)
                dist.all_to_all_single(receive, send, group=self.process_group)
                torch.cuda.synchronize(device)
        self._communication_ready = True

    def _batches(self, active):
        buckets = defaultdict(list)
        for i, enabled in enumerate(active):
            if enabled:
                buckets[self.layouts[i].spec.shape].append(i)
        cursor = 0
        capacity = self.world * self.matrices_per_rank
        for indices in buckets.values():
            for start in range(0, len(indices), capacity):
                batch = indices[start : start + capacity]
                owners = [(cursor + i) % self.world for i in range(len(batch))]
                cursor = (cursor + len(batch)) % self.world
                yield batch, owners

    def _exchange(self, indices, owners, send, send_sizes, receive_sizes):
        receive = send.new_empty(sum(receive_sizes))
        work = None
        if self.world > 1:
            work = dist.all_to_all_single(
                receive,
                send,
                output_split_sizes=receive_sizes,
                input_split_sizes=send_sizes,
                group=self.process_group,
                async_op=True,
            )
            self.communication_calls += 1
        else:
            receive.copy_(send)
        return Exchange(indices, owners, send, receive, send_sizes, receive_sizes, work)

    def _collect(self, indices, owners, *, release_gradients=False):
        group = self.param_groups[0]
        selected_by_peer = [
            [i for i, owner in zip(indices, owners) if owner == peer]
            for peer in range(self.world)
        ]
        send_sizes, receive_sizes = [], []
        for peer in range(self.world):
            send_sizes.append(
                sum(
                    self.layouts[i].shards[self.rank][1] for i in selected_by_peer[peer]
                )
            )
            receive_sizes.append(
                sum(
                    self.layouts[i].shards[peer][1]
                    for i, owner in zip(indices, owners)
                    if owner == self.rank
                )
            )
        # Pack directly into one BF16 buffer instead of retaining per-matrix
        # BF16 updates and a second, concatenated copy at the same time.
        send = self.params[0].new_empty(sum(send_sizes), dtype=torch.bfloat16)
        cursor = 0
        for selected in selected_by_peer:
            for i in selected:
                p = self.params[i]
                n = p.numel()
                if n:
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    if p.is_cuda:
                        stream = torch.cuda.current_stream(p.device)
                        p.record_stream(stream)
                        g.record_stream(stream)
                        buf.record_stream(stream)
                    buf.lerp_(g, 1 - group["momentum"])
                    if group["nesterov"]:
                        # The hybrid wrapper discards master gradients after
                        # this update, so reuse that FP32 storage when allowed.
                        u = (
                            g.lerp_(buf, group["momentum"])
                            if release_gradients
                            else g.lerp(buf, group["momentum"])
                        )
                    else:
                        u = buf
                    send[cursor : cursor + n].copy_(u.flatten())
                    del g, u
                if release_gradients:
                    p.grad = None
                cursor += n
        return self._exchange(indices, owners, send, send_sizes, receive_sizes)

    def _orthogonalize_and_return(self, collected):
        if collected.work is not None:
            collected.work.wait()
        indices, owners = collected.indices, collected.owners
        owned = [i for i, owner in zip(indices, owners) if owner == self.rank]
        send = None
        if owned:
            shape = self.layouts[owned[0]].spec.shape
            full_matrices = collected.receive.new_empty((len(owned), *shape))
            cursor = 0
            for peer in range(self.world):
                for j, i in enumerate(owned):
                    offset, length = self.layouts[i].shards[peer]
                    full_matrices[j].view(-1)[offset : offset + length].copy_(
                        collected.receive[cursor : cursor + length]
                    )
                    cursor += length
            # Work.wait() orders communication before the copies above on the
            # current stream. Neither exchange buffer is needed by NS, even
            # when the next batch is being prefetched on another stream.
            collected.send = collected.receive = collected.work = None
            group = self.param_groups[0]
            orthogonalized_matrices = newton_schulz(
                full_matrices, group["ns_coefficients"], group["ns_steps"], group["eps"]
            ).contiguous()
            del full_matrices
            # Tall matrices leave NS transposed. Flattening each peer's slice
            # separately used to retain world_size full contiguous copies.
            flat = orthogonalized_matrices.view(len(owned), -1)
            send = flat.new_empty(sum(collected.receive_sizes))
            cursor = 0
            for peer in range(self.world):
                for j, i in enumerate(owned):
                    offset, length = self.layouts[i].shards[peer]
                    send[cursor : cursor + length].copy_(
                        flat[j, offset : offset + length]
                    )
                    cursor += length
            del flat, orthogonalized_matrices
        if send is None:
            send = self.params[0].new_empty(0, dtype=torch.bfloat16)
        collected.send = collected.receive = collected.work = None
        return self._exchange(
            indices, owners, send, collected.receive_sizes, collected.send_sizes
        )

    def _apply(self, returned):
        if returned.work is not None:
            returned.work.wait()
        group = self.param_groups[0]
        cursor = 0
        for peer in range(self.world):
            for i, owner in zip(returned.indices, returned.owners):
                if owner != peer:
                    continue
                p = self.params[i]
                n = p.numel()
                update = returned.receive[cursor : cursor + n].reshape(p.shape)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(
                    update,
                    alpha=-adjusted_lr(
                        group["lr"], self.layouts[i].spec.shape, group["adjust_lr_fn"]
                    ),
                )
                cursor += n

    @torch.no_grad()
    def step(self, closure=None, *, active=None, release_gradients=False):
        if closure is not None:
            raise ValueError("ShardedMuon does not support closures")
        self.prepare_communication()
        if active is None:
            mask = torch.tensor(
                [p.grad is not None for p in self.params],
                dtype=torch.int32,
                device=self.params[0].device,
            )
            if self.world > 1:
                dist.all_reduce(mask, op=dist.ReduceOp.MAX, group=self.process_group)
            active = mask.tolist()
        batches = iter(self._batches(active))
        first = next(batches, None)
        self.communication_calls = 0
        if first is None:
            return
        if self.low_memory:
            # Finish and release each batch before allocating the next one.
            # Standalone optimizer users retain gradients unless explicitly
            # opting into consumption; BF16Optimizer already clears them.
            for batch in chain((first,), batches):
                collected = self._collect(*batch, release_gradients=release_gradients)
                returned = self._orthogonalize_and_return(collected)
                del collected
                self._apply(returned)
                del returned
            return
        # A dedicated stream lets gather(k+1) run while NS(k) runs. Events join
        # the streams without host/device barriers. NCCL order is identical on
        # every rank: gather0, gather1, return0, gather2, return1, ... .
        device = self.params[0].device
        compute = torch.cuda.current_stream(device) if device.type == "cuda" else None
        if compute is not None and self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=device)
        prefetch = self._prefetch_stream
        current = self._collect(*first, release_gradients=release_gradients)
        while current is not None:
            next_batch = next(batches, None)
            upcoming = None
            event = None
            if next_batch is not None:
                if prefetch is not None:
                    prefetch.wait_stream(compute)
                    with torch.cuda.stream(prefetch):
                        upcoming = self._collect(
                            *next_batch, release_gradients=release_gradients
                        )
                        event = prefetch.record_event()
                else:
                    upcoming = self._collect(
                        *next_batch, release_gradients=release_gradients
                    )
            returned = self._orthogonalize_and_return(current)
            self._apply(returned)
            del returned
            if event is not None:
                compute.wait_event(event)
                # Transfers ownership of temporary allocations to this stream.
                upcoming.send.record_stream(compute)
                upcoming.receive.record_stream(compute)
            current = upcoming
