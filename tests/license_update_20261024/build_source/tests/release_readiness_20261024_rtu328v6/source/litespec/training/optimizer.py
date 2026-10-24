"""FP32-master AdamW/optional Muon, scheduling and global gradient clipping."""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler

from litespec.distributed import print_on_rank0


def register_optimizer_parameters(model, parameters=None):
    """Classify original parameters before FSDP destroys their matrix shapes."""
    from .fsdp_layout import ParameterSpec

    selected = list(
        dict.fromkeys(
            (p for p in model.parameters() if p.requires_grad)
            if parameters is None
            else parameters
        )
    )
    names = {id(p): name for name, p in model.named_parameters()}
    policies = {id(p): set() for p in selected}
    # Muon exceptions for optimizer ownership
    muon_exceptions = {
        "selector",
        "candidate_selector",
        "markov_head",
        "sequential_head",
        "confidence_head",
    }
    for path, module in model.named_modules(remove_duplicate=False):
        for name, p in module.named_parameters(recurse=False):
            if id(p) not in policies:
                continue
            if (
                muon_exceptions.intersection(path.split("."))
                or isinstance(module, torch.nn.Embedding)
                or "norm" in type(module).__name__.lower()
                or name in ("bias", "base_kernel")
            ):
                kind = "adamw"
            elif (
                isinstance(module, torch.nn.Linear) and name == "weight" and p.ndim == 2
            ):
                if not p.is_contiguous():
                    raise ValueError(
                        f"Muon requires contiguous matrix weights: {path}.{name}"
                    )
                kind = "muon"
            else:
                raise ValueError(
                    f"Unclassified optimizer parameter: {path}.{name} {tuple(p.shape)}"
                )
            policies[id(p)].add(kind)
    specs = []
    for p in selected:
        choices = policies[id(p)]
        if id(p) not in names or len(choices) != 1:
            raise ValueError(
                f"Missing/conflicting optimizer ownership: {names.get(id(p), '<external>')}"
            )
        specs.append(ParameterSpec(names[id(p)], tuple(p.shape), choices.pop()))
    return selected, specs


def optimizer_step_count(phases, steps_per_epoch, accumulation_steps):
    """Each phase flushes its own tail; epochs within a phase do not."""
    return sum(
        math.ceil(
            (
                phase.epochs * steps_per_epoch
                if phase.schedule_full_epochs
                else phase.batch_count(steps_per_epoch)
            )
            / accumulation_steps
        )
        for phase in phases
    )


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key not in "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        expected = type(self.after_scheduler).__name__
        if state_dict["after_scheduler_type"] != expected:
            raise ValueError(f"Scheduler type must be {expected}")
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class DelayerScheduler(TwoStageScheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau)

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, last_epoch=-1):
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        self.delay_epochs = delay_epochs
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(DelayerScheduler, self).step(epoch)


class WarmupScheduler(TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class WarmupDelayerScheduler(TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs and a flat lr schedule
    until it reaches M epochs then applies the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self, optimizer, warmup_epochs, delay_epochs, after_scheduler, last_epoch=-1
    ):
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must >= 0, got {warmup_epochs}")
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs + self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                # reset lr to base_lr
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base_lr
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()
        elif self.last_epoch >= self.warmup_epochs:
            return self.base_lrs

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingLR(_CosineAnnealingLR):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        eta_min: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        super().__init__(optimizer, total_steps, eta_min=eta_min, last_epoch=last_epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            max(total_steps - warmup_steps, 1),
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        parameters=None,
        process_group=None,
        muon_config=None,
        parameter_specs=None,
        parameter_layouts=None,
    ):
        self.model = model
        self.use_muon = muon_config is not None
        self.muon_optimizer = None
        self.muon_scheduler = None
        if self.use_muon:
            from .fsdp_layout import local_layouts
            from .muon import validate_muon_config

            muon_config = validate_muon_config(muon_config)
            if parameter_specs is None:
                parameters, parameter_specs = register_optimizer_parameters(
                    model, parameters
                )
            if parameter_layouts is None:
                if dist.is_initialized() and dist.get_world_size(process_group) > 1:
                    raise ValueError("Distributed Muon requires explicit FSDP layouts")
                parameter_layouts = local_layouts(parameter_specs)
            self.parameter_specs = list(parameter_specs)
            self.parameter_layouts = list(parameter_layouts)
        self.model_params = (
            [p for p in model.parameters() if p.requires_grad]
            if parameters is None
            else list(parameters)
        )
        if not self.model_params:
            raise ValueError("BF16Optimizer requires at least one model parameter.")
        parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self.parameter_names = [
            parameter_names[id(parameter)] for parameter in self.model_params
        ]
        self.optimizer_config = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_grad_norm=float(max_grad_norm),
            total_steps=int(total_steps),
            warmup_ratio=float(warmup_ratio),
        )
        if self.use_muon:
            # Legacy lr/weight_decay are deliberately irrelevant in hybrid mode.
            self.optimizer_config.update(
                lr=float(muon_config["adamw_lr"]),
                weight_decay=float(muon_config["adamw_weight_decay"]),
                kind="muon_adamw_v1",
                muon=muon_config,
            )
            if [s.name for s in self.parameter_specs] != self.parameter_names or [
                l.spec for l in self.parameter_layouts
            ] != self.parameter_specs:
                raise ValueError("Optimizer parameter registration/layout mismatch")
            rank = dist.get_rank(process_group) if dist.is_initialized() else 0
            world = dist.get_world_size(process_group) if dist.is_initialized() else 1
            if any(
                len(layout.shards) != world
                or parameter.numel() != layout.shards[rank][1]
                for parameter, layout in zip(self.model_params, self.parameter_layouts)
            ):
                raise ValueError(
                    "Optimizer local parameter shapes do not match shard layouts"
                )
        self.max_grad_norm = float(max_grad_norm)
        self.process_group = process_group
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}.")
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.adamw_indices = list(range(len(self.fp32_params)))
        self.muon_indices = []
        if self.use_muon:
            from .muon import ShardedMuon

            self.adamw_indices = [
                i for i, s in enumerate(self.parameter_specs) if s.optimizer == "adamw"
            ]
            self.muon_indices = [
                i for i, s in enumerate(self.parameter_specs) if s.optimizer == "muon"
            ]
            if len(self.adamw_indices) + len(self.muon_indices) != len(
                self.fp32_params
            ):
                raise ValueError("Unknown optimizer assignment")
            if self.muon_indices:
                self.muon_optimizer = ShardedMuon(
                    [self.fp32_params[i] for i in self.muon_indices],
                    [self.parameter_layouts[i] for i in self.muon_indices],
                    process_group=process_group,
                    **{
                        k: v
                        for k, v in muon_config.items()
                        if not k.startswith("adamw_")
                    },
                )
            lr, weight_decay = (
                muon_config["adamw_lr"],
                muon_config["adamw_weight_decay"],
            )
            for kind in ("adamw", "muon"):
                specs = [s for s in self.parameter_specs if s.optimizer == kind]
                if not dist.is_initialized() or dist.get_rank(process_group) == 0:
                    logging.getLogger(__name__).info(
                        "%s: %s tensors, %s parameters",
                        kind,
                        len(specs),
                        f"{sum(math.prod(s.shape) for s in specs):,}",
                    )
        self.optimizer = (
            torch.optim.AdamW(
                [self.fp32_params[i] for i in self.adamw_indices],
                lr=lr,
                weight_decay=weight_decay,
                betas=muon_config["adamw_betas"] if self.use_muon else (0.9, 0.999),
            )
            if self.adamw_indices
            else None
        )
        self.scheduler = (
            CosineAnnealingWarmupLR(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
            )
            if self.optimizer is not None
            else None
        )
        if self.muon_optimizer is not None:
            self.muon_scheduler = CosineAnnealingWarmupLR(
                self.muon_optimizer,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
            )

    def zero_grad(self) -> None:
        """Clear both FSDP model gradients and FP32 optimizer gradients."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=True)
        for param in self.model_params:
            param.grad = None
        for param in self.fp32_params:
            param.grad = None

    def scale_model_gradients(self, factor: float) -> None:
        """Scale accumulated FSDP gradients before they are copied to FP32."""
        if factor <= 0:
            raise ValueError(f"gradient scale must be positive, got {factor}.")
        with torch.no_grad():
            for param in self.model_params:
                if param.grad is not None:
                    param.grad.mul_(factor)

    def _global_grad_norm(self) -> torch.Tensor:
        """Return the global L2 norm using one scalar all-reduce.

        Gradients remain sharded and local. Only their local squared L2 norm is
        summed across ranks.
        """
        if not self.fp32_params:
            return torch.tensor(0.0)

        device = self.fp32_params[0].device
        gradients = [param.grad for param in self.fp32_params if param.grad is not None]
        if gradients:
            local_norms = torch._foreach_norm(gradients, 2)
            local_squared_norm = (
                torch.stack([norm.double() for norm in local_norms]).square().sum()
            )
        else:
            local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            # Communication payload: exactly one 0-D scalar, never gradients.
            dist.all_reduce(
                local_squared_norm,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
        return local_squared_norm.sqrt()

    def step(self) -> float:
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32)
                    if p.grad is not None and (not self.use_muon or p.requires_grad)
                    else None
                )
            if self.use_muon:
                mask = torch.tensor(
                    [p.grad is not None for p in self.fp32_params],
                    dtype=torch.int32,
                    device=self.fp32_params[0].device,
                )
                if dist.is_initialized():
                    dist.all_reduce(
                        mask, op=dist.ReduceOp.MAX, group=self.process_group
                    )
                active = mask.tolist()
                for enabled, p in zip(active, self.fp32_params):
                    if enabled and p.grad is None:
                        p.grad = torch.zeros_like(p)

        grad_norm = self._global_grad_norm()
        grad_norm_value = float(grad_norm.item())
        clip_coefficient = min(
            self.max_grad_norm / (grad_norm_value + 1e-6),
            1.0,
        )
        if clip_coefficient < 1.0:
            with torch.no_grad():
                for param in self.fp32_params:
                    if param.grad is not None:
                        param.grad.mul_(clip_coefficient)

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
        if self.muon_optimizer is not None:
            self.muon_optimizer.step(active=[active[i] for i in self.muon_indices])
            self.muon_optimizer.zero_grad(set_to_none=True)
            self.muon_scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.copy_(mp)
                p.grad = None
        return grad_norm_value

    def _param_state_compatible(self, param, param_state) -> bool:
        for key, value in param_state.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue
            if value.shape != param.shape:
                return False
        return True

    def _optimizer_state_compatible(self, optimizer_state_dict) -> bool:
        """Return True when every saved Adam moment matches current fp32 param shapes."""
        if self.use_muon:
            if optimizer_state_dict.get("kind") != "muon_adamw_v1":
                return False
            if optimizer_state_dict.get("layouts") != [
                l.state_dict() for l in self.parameter_layouts
            ]:
                return False
            for key, indices in (
                ("adamw", self.adamw_indices),
                ("muon", self.muon_indices),
            ):
                saved = optimizer_state_dict.get(key)
                if not indices:
                    if saved is not None:
                        return False
                    continue
                if not saved or len(saved["param_groups"]) != 1:
                    return False
                ids = saved["param_groups"][0]["params"]
                if len(ids) != len(indices):
                    return False
                for i, saved_id in zip(indices, ids):
                    state = saved["state"].get(saved_id, {})
                    if not self._param_state_compatible(self.fp32_params[i], state):
                        return False
                    if any(
                        v.dtype != torch.float32
                        for k, v in state.items()
                        if isinstance(v, torch.Tensor) and k != "step"
                    ):
                        return False
            return True
        ckpt_params = optimizer_state_dict["param_groups"][0]["params"]
        if len(ckpt_params) != len(self.fp32_params):
            print_on_rank0(
                "Optimizer checkpoint has "
                f"{len(ckpt_params)} params, expected {len(self.fp32_params)}."
            )
            return False

        for idx, (param, param_id) in enumerate(
            zip(self.fp32_params, ckpt_params, strict=True)
        ):
            param_state = optimizer_state_dict["state"].get(param_id)
            if param_state is None:
                continue
            if not self._param_state_compatible(param, param_state):
                print_on_rank0(
                    "Optimizer state shape mismatch at param index "
                    f"{idx}: checkpoint moments do not match model "
                    f"{tuple(param.shape)} ({param.numel()} elements)."
                )
                return False
        return True

    def load_state_dict(self, state_dict) -> bool:
        if state_dict["optimizer_config"] != self.optimizer_config:
            raise ValueError(
                "Optimizer/scheduler configuration does not match the checkpoint"
            )
        ckpt_opt = state_dict["optimizer_state_dict"]
        if state_dict["optimizer_parameter_names"] != self.parameter_names:
            raise ValueError(
                "Optimizer parameter names/order do not match the checkpoint"
            )
        if not self._optimizer_state_compatible(ckpt_opt):
            raise ValueError("Optimizer parameter shapes do not match the checkpoint")
        masters = state_dict["fp32_master_params"]
        if len(masters) != len(self.fp32_params) or any(
            source.shape != target.shape or source.dtype != torch.float32
            for source, target in zip(masters, self.fp32_params)
        ):
            raise ValueError(
                "FP32 master parameter shapes/dtypes do not match the checkpoint"
            )
        with torch.no_grad():
            for model_param, master, saved in zip(
                self.model_params, self.fp32_params, masters, strict=True
            ):
                master.copy_(saved)
                model_param.copy_(master)
        if self.use_muon:
            schedulers = state_dict["scheduler_state_dict"]
            for key, opt, scheduler in (
                ("adamw", self.optimizer, self.scheduler),
                ("muon", self.muon_optimizer, self.muon_scheduler),
            ):
                if opt is not None:
                    opt.load_state_dict(ckpt_opt[key])
                    scheduler.load_state_dict(schedulers[key])
        else:
            self.optimizer.load_state_dict(ckpt_opt)
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded optimizer and scheduler state.")
        return True

    def state_dict(self):
        if self.use_muon:
            optimizer_state = {
                "kind": "muon_adamw_v1",
                "layouts": [l.state_dict() for l in self.parameter_layouts],
                "adamw": self.optimizer.state_dict() if self.optimizer else None,
                "muon": (
                    self.muon_optimizer.state_dict() if self.muon_optimizer else None
                ),
            }
            scheduler_state = {
                "adamw": self.scheduler.state_dict() if self.scheduler else None,
                "muon": (
                    self.muon_scheduler.state_dict() if self.muon_scheduler else None
                ),
            }
        else:
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
        return {
            "optimizer_parameter_names": self.parameter_names,
            "optimizer_config": self.optimizer_config,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "fp32_master_params": [p.detach().cpu().clone() for p in self.fp32_params],
        }

    def get_learning_rate(self):
        return (self.optimizer or self.muon_optimizer).param_groups[0]["lr"]

    def get_learning_rates(self):
        return {
            name: opt.param_groups[0]["lr"]
            for name, opt in (("adamw", self.optimizer), ("muon", self.muon_optimizer))
            if opt is not None
        }


def normalize_accumulated_gradients(
    optimizer,
    local_denominator: torch.Tensor,
    accumulation_steps: int,
    group=None,
) -> float:
    """Normalize rank-averaged gradients by the global optimizer-window weight.

    Every microbatch backpropagates its additive loss numerator divided by the
    configured accumulation length. FSDP averages those gradients across ranks.
    Multiplying by ``world_size * accumulation_steps / global_denominator``
    therefore produces the gradient of the summed numerator divided by the sum
    of effective weights across every microbatch and DP rank in the window.
    """
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    denominator = local_denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise ValueError("global loss denominator must be finite and positive")
    scale = world_size * int(accumulation_steps) / denominator_value
    optimizer.scale_model_gradients(scale)
    return denominator_value
