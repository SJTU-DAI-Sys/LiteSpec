"""Additive window metrics, log layout and experiment trackers."""

from __future__ import annotations

import abc
import argparse
import hashlib
import logging
import netrc
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist

from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.distributed import print_on_rank0


@dataclass
class TrackerArgs:
    report_to: str = "none"
    wandb_project: str = None
    wandb_name: str = None
    wandb_key: str = None
    wandb_run_id: str = None  # 固定 run id：首次创建，已有则续写（wandb resume=allow）
    swanlab_project: str = None
    swanlab_name: str = None
    swanlab_key: str = None
    mlflow_experiment_id: str = None
    mlflow_run_name: str = None
    mlflow_run_id: str = None
    mlflow_tracking_uri: str = None
    mlflow_registry_uri: str = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--report-to",
            type=str,
            default="none",
            choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
            help="The integration to report results and logs to.",
        )
        # wandb-specific args
        parser.add_argument("--wandb-project", type=str, default=None)
        parser.add_argument("--wandb-name", type=str, default=None)
        parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
        parser.add_argument(
            "--wandb-run-id",
            type=str,
            default=None,
            help="Fixed W&B run id: resume logging if that run exists, otherwise start a new run (resume=allow).",
        )
        # swanlab-specific args
        parser.add_argument(
            "--swanlab-project",
            type=str,
            default=None,
            help="The project name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-name",
            type=str,
            default=None,
            help="The experiment name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-key",
            type=str,
            default=None,
            help="The API key for swanlab non-interactive login.",
        )
        # mlflow-specific args
        parser.add_argument(
            "--mlflow-tracking-uri",
            type=str,
            default=None,
            help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
        )
        parser.add_argument(
            "--mlflow-experiment-name",
            type=str,
            default=None,
            help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
        )
        parser.add_argument(
            "--mlflow-run-name",
            type=str,
            default=None,
            help="The MLflow run name. If not set, MLflow will auto-generate one.",
        )


logger = logging.getLogger("litespec.distributed")


_WANDB_STR_MAX_LEN = 128


_WANDB_NO_LOGIN_MODES = frozenset({"offline", "dryrun", "disabled"})


def _wandb_requires_login() -> bool:
    """Return whether the configured W&B mode needs online credentials."""
    return os.environ.get("WANDB_MODE", "online").strip().lower() not in (
        _WANDB_NO_LOGIN_MODES
    )


def _clip_wandb_identifier(value: Optional[str], field: str) -> Optional[str]:
    if not value or len(value) <= _WANDB_STR_MAX_LEN:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    clipped = f"{value[: _WANDB_STR_MAX_LEN - 9]}_{digest}"
    logger.warning(
        "WandB %s exceeds %d chars (%d); clipping to %r",
        field,
        _WANDB_STR_MAX_LEN,
        len(value),
        clipped,
    )
    return clipped


try:
    import wandb
except ImportError:
    wandb = None


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


try:
    import swanlab
except ImportError:
    swanlab = None


try:
    import mlflow
except ImportError:
    mlflow = None


class Tracker(abc.ABC):
    """
    Abstract Base Class for experiment trackers.

    Each tracker implementation should handle its own initialization, logging,
    and cleanup. It should also provide a class method to validate
    command-line arguments before initialization.
    """

    def __init__(self, args, output_dir: str):
        self.args = args
        self.output_dir = output_dir
        self.rank = dist.get_rank()
        self.coordinator_rank = int(getattr(args, "tracker_global_rank", 0))
        self.is_coordinator = self.rank == self.coordinator_rank
        self.is_initialized = False

    @classmethod
    @abc.abstractmethod
    def validate_args(cls, parser, args) -> None:
        """
        Validate necessary arguments for this tracker.
        This method is called during argument parsing.
        It should raise an error if required arguments are missing.
        """

    @abc.abstractmethod
    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to the tracker.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the tracker and clean up resources.
        """


class NoOpTracker(Tracker):
    """A tracker that does nothing, for when no tracking is desired."""

    @classmethod
    def validate_args(cls, parser, args):
        pass  # No arguments to validate

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        self.is_initialized = True  # Considered initialized to do nothing

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        pass  # Do nothing

    def close(self):
        pass  # Do nothing


class WandbTracker(Tracker):
    """Tracks experiments using Weights & Biases."""

    @classmethod
    def validate_args(cls, parser, args):
        if wandb is None:
            parser.error(
                "To use --report-to wandb, you must install wandb: 'pip install wandb'"
            )

        if not _wandb_requires_login():
            return

        if args.wandb_key is not None:
            return

        if "WANDB_API_KEY" in os.environ:
            args.wandb_key = os.environ["WANDB_API_KEY"]
            return

        try:
            netrc_path = os.path.expanduser("~/.netrc")
            if os.path.exists(netrc_path):
                netrc_file = netrc.netrc(netrc_path)
                if "api.wandb.ai" in netrc_file.hosts:
                    _, _, password = netrc_file.authenticators("api.wandb.ai")
                    if password:
                        args.wandb_key = password
                        return
        except (FileNotFoundError, netrc.NetrcParseError):
            pass

        if args.wandb_key is None:
            parser.error(
                "When --report-to is 'wandb', you must provide a wandb API key via one of:\n"
                "  1. --wandb-key argument\n"
                "  2. WANDB_API_KEY environment variable\n"
                "  3. `wandb login` command"
            )

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.is_coordinator:
            if _wandb_requires_login():
                wandb.login(key=args.wandb_key)
            # wandb_run_id: optional stable id from launch scripts; resume="allow" continues an
            # existing run or creates one (resume="must" would fail on first start).
            init_kwargs = {
                "project": args.wandb_project,
                "name": _clip_wandb_identifier(args.wandb_name, "name"),
                "config": vars(args),
            }
            if hasattr(args, "wandb_run_id") and args.wandb_run_id:
                init_kwargs["id"] = _clip_wandb_identifier(args.wandb_run_id, "id")
                # Resume if the id exists; otherwise create (stable ids from e.g. launch scripts).
                init_kwargs["resume"] = "allow"
            wandb.init(**init_kwargs)
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.is_coordinator and self.is_initialized:
            wandb.log(log_dict, step=step)

    def close(self):
        if self.is_coordinator and self.is_initialized and wandb.run:
            wandb.finish()
            self.is_initialized = False


class SwanlabTracker(Tracker):
    """Tracks experiments using SwanLab."""

    @classmethod
    def validate_args(cls, parser, args):
        if swanlab is None:
            parser.error(
                "To use --report-to swanlab, you must install swanlab: 'pip install swanlab'"
            )

        if args.swanlab_key is not None:
            return
        if "SWANLAB_API_KEY" in os.environ:
            args.swanlab_key = os.environ["SWANLAB_API_KEY"]
            return
        # Swanlab can run in anonymous mode if no key is provided in a non-distributed env.
        # However, a key is often required for distributed runs to sync correctly.
        if (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and args.swanlab_key is None
        ):
            parser.error(
                "In a distributed environment, when --report-to is 'swanlab', you must provide a swanlab API key via:\n"
                "  1. --swanlab-key argument\n"
                "  2. SWANLAB_API_KEY environment variable"
            )

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.is_coordinator:
            if args.swanlab_key:
                swanlab.login(api_key=args.swanlab_key)

            swanlog_dir = os.path.join(output_dir, "swanlog")
            os.makedirs(swanlog_dir, exist_ok=True)
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_name,
                config=vars(args),
                logdir=swanlog_dir,
            )
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.is_coordinator and self.is_initialized:
            swanlab.log(log_dict, step=step)

    def close(self):
        if (
            self.is_coordinator
            and self.is_initialized
            and swanlab.get_run() is not None
        ):
            swanlab.finish()
            self.is_initialized = False


class TensorboardTracker(Tracker):
    """Tracks experiments using TensorBoard."""

    @classmethod
    def validate_args(cls, parser, args):
        if SummaryWriter is None:
            parser.error(
                "To use --report-to tensorboard, you must have tensorboard installed: 'pip install tensorboard'"
            )

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.is_coordinator:
            log_dir = os.path.join(output_dir, "runs")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.is_coordinator and self.is_initialized:
            for key, value in log_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, global_step=step)

    def close(self):
        if self.is_coordinator and self.is_initialized:
            self.writer.close()
            self.is_initialized = False


class MLflowTracker(Tracker):
    """Tracks experiments using MLflow."""

    @classmethod
    def validate_args(cls, parser, args):
        if mlflow is None:
            parser.error(
                "To use --report-to mlflow, you must install mlflow: 'pip install mlflow'"
            )
        # Set tracking URI from environment variable if not explicitly provided
        if args.mlflow_tracking_uri is None and "MLFLOW_TRACKING_URI" in os.environ:
            args.mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        elif args.mlflow_tracking_uri is None:
            print(
                "Warning: MLflow tracking URI not set. Defaulting to local './mlruns'."
            )

        # Set experiment name from environment variable if not explicitly provided
        if (
            args.mlflow_experiment_name is None
            and "MLFLOW_EXPERIMENT_NAME" in os.environ
        ):
            args.mlflow_experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.is_coordinator:
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)

            # This will either use the set URI or the default
            mlflow.set_experiment(args.mlflow_experiment_name)
            mlflow.start_run(run_name=args.mlflow_run_name)
            mlflow.log_params(vars(args))
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.is_coordinator and self.is_initialized:
            # MLflow's log_metrics takes a dictionary directly
            mlflow.log_metrics(log_dict, step=step)

    def close(self):
        if self.is_coordinator and self.is_initialized:
            mlflow.end_run()
            self.is_initialized = False


TRACKER_REGISTRY = {
    "wandb": WandbTracker,
    "swanlab": SwanlabTracker,
    "tensorboard": TensorboardTracker,
    "mlflow": MLflowTracker,
    "none": NoOpTracker,
}


def get_tracker_class(report_to: str) -> Optional[Tracker]:
    """Returns the tracker class based on the name."""
    return TRACKER_REGISTRY.get(report_to)


def create_tracker(args, output_dir: str) -> Tracker:
    """Factory function to create an experiment tracker instance."""
    tracker_class = get_tracker_class(args.report_to)
    if not tracker_class:
        raise ValueError(f"Unsupported report_to type: {args.report_to}")
    return tracker_class(args, output_dir)


def log_cuda_peak(stage: str) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print_on_rank0(
        f"{stage} CUDA peak: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"
    )
    return {"allocated_gib": allocated, "reserved_gib": reserved}


_POSITION_METRIC_PREFIX = "position_"


_RETAINED_POSITION_METRIC_SUFFIX = "/hard_label/unary_top16_recall"


class TrainingMetricAccumulator:
    """Accumulate additive metrics over one optimizer window."""

    def __init__(self) -> None:
        self.ratio_metrics: dict[str, RatioTerms] = {}
        self.sum_metrics: dict[str, torch.Tensor] = {}

    def update(self, output: TrainingOutput) -> None:
        ratios = {"objective/loss": output.loss_terms, **output.ratio_metrics}
        for name, terms in ratios.items():
            numerator = terms.numerator.detach()
            denominator = terms.denominator.detach()
            previous = self.ratio_metrics.get(name)
            self.ratio_metrics[name] = (
                RatioTerms(numerator.clone(), denominator.clone())
                if previous is None
                else RatioTerms(
                    previous.numerator + numerator,
                    previous.denominator + denominator,
                )
            )
        for name, value in output.sum_metrics.items():
            value = value.detach()
            previous = self.sum_metrics.get(name)
            self.sum_metrics[name] = (
                value.clone() if previous is None else previous + value
            )

    def reset(self) -> None:
        self.ratio_metrics.clear()
        self.sum_metrics.clear()

    def reduce(self, group=None) -> dict[str, torch.Tensor]:
        if not self.ratio_metrics:
            raise ValueError("Cannot reduce an empty training metric window")
        return _reduce_additive_metrics(
            self.ratio_metrics,
            self.sum_metrics,
            group=group,
        )


def build_training_log_payload(
    metrics: Mapping[str, torch.Tensor | float],
    learning_rate: float,
    *,
    detail_metrics: Optional[Mapping[str, torch.Tensor | float | int | None]] = None,
    detail_prefix: Optional[str] = None,
) -> dict[str, float | int]:
    """Build the shared tracker layout for cross-algorithm comparison."""

    def scalar(value):
        return value.item() if isinstance(value, torch.Tensor) else value

    detail_root = "detail" if not detail_prefix else f"detail/{detail_prefix}"
    algorithm = detail_prefix.split("/", 1)[0] if detail_prefix else None

    def detail_key(name):
        if algorithm:
            name = name.removeprefix(algorithm + "/")
        return f"{detail_root}/{name}"

    payload = {detail_key(name): scalar(value) for name, value in metrics.items()}
    if detail_metrics:
        payload.update(
            {
                detail_key(name): scalar(value)
                for name, value in detail_metrics.items()
                if value is not None
            }
        )

    loss = metrics.get("objective/loss")
    if loss is not None:
        payload["train/loss"] = scalar(loss)
    payload["train/learning_rate"] = float(learning_rate)
    # The public key is retained, but all algorithms use block averaging.
    value = metrics.get("hard_label/block_accepted_length")
    if value is not None:
        payload["train/walk_accepted_length"] = scalar(value)
    expected = metrics.get("expected_accept_len")
    if expected is not None:
        payload["train/expected_accept_len"] = scalar(expected)
    return payload


def _reduce_additive_metrics(ratios, sums, *, group=None):
    """Globally reduce additive metric terms without local ratio averaging."""

    reduced: dict[str, torch.Tensor] = {}
    for name, terms in ratios.items():
        if name.startswith(_POSITION_METRIC_PREFIX) and not name.endswith(
            _RETAINED_POSITION_METRIC_SUFFIX
        ):
            continue
        pair = torch.stack(
            [terms.numerator.detach().float(), terms.denominator.detach().float()]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pair, group=group)
        reduced[name] = pair[0] / pair[1].clamp_min(1e-12)
    for name, value in sums.items():
        if name.startswith(_POSITION_METRIC_PREFIX) and not name.endswith(
            _RETAINED_POSITION_METRIC_SUFFIX
        ):
            continue
        value = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, group=group)
        reduced[name] = value
    return reduced


def reduce_training_metrics(
    output: TrainingOutput, group=None
) -> dict[str, torch.Tensor]:
    """Globally reduce one output's metrics.

    Global metrics are all retained. Per-position metrics are intentionally
    restricted to unary top-16 recall.
    """
    ratios = {"objective/loss": output.loss_terms.detached(), **output.ratio_metrics}
    return _reduce_additive_metrics(ratios, output.sum_metrics, group=group)
