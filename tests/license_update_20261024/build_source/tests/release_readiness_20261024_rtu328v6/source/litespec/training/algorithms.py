"""Small algorithm contract and the state consumed by the shared runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

    from torch.distributed import ProcessGroup
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.utils.data import DataLoader

    from litespec.core.dflash import OnlineDFlashModel
    from litespec.core.dflash2 import OnlineDFlash2Model
    from litespec.core.dlite import OnlineDLiteModel
    from litespec.core.dspark import OnlineDSparkModel
    from litespec.distributed.groups import DisaggregatedTopology
    from litespec.distributed.transport import RouteFragment
    from litespec.modeling.draft.dflash import DFlashDraftModel
    from litespec.modeling.draft.dflash2 import DFlash2DraftModel
    from litespec.modeling.draft.dlite import DLiteDraftModel
    from litespec.modeling.draft.dspark import DSparkDraftModel
    from litespec.modeling.target.target_utils import (
        SharedTargetEmbeddingsAndHead,
        TargetEmbeddingsAndHead,
    )
    from litespec.modeling.target.training_target_model import TrainingTargetModel
    from litespec.training.logging import Tracker
    from litespec.training.optimizer import BF16Optimizer
    from litespec.training.recipes._dflash_common import DFlashFamilyTrainingAdapter
    from litespec.training.recipes.dlite import DLiteTrainingAdapter


@dataclass(frozen=True)
class Phase:
    name: str
    phase_id: int
    epochs: int
    need_history: bool = False
    need_prediction_hidden: bool = False
    max_batches: int | None = None
    log_at_schedule_end: bool = True
    schedule_full_epochs: bool = False
    checkpoint_prefix: str = ""
    final_checkpoint: str = "final"
    start_checkpoint: str | None = None

    def batch_count(self, steps: int) -> int:
        count = self.epochs * steps
        return count if self.max_batches is None else min(count, self.max_batches)


@dataclass
class RunState:
    phase_index: int = 0
    epoch: int = 0
    next_batch: int = 0
    phase_step: int = 0
    global_step: int = 0
    optimizer_step: int = 0


@dataclass
class TrainingSession:
    """Runtime container for one training run.
    Holds CLI args, the algorithm adapter, phase cursor, models (draft /
    teacher / target / FSDP draft_online wrapper), data, optimizer, tracker, and
    disaggregate topology. Runner and recipes read dependencies from here
    instead of implicit globals.
    """

    args: Any
    adapter: DFlashFamilyTrainingAdapter | DLiteTrainingAdapter
    phases: list[Phase]
    state: RunState = field(default_factory=RunState)
    draft: (
        DFlashDraftModel | DFlash2DraftModel | DSparkDraftModel | DLiteDraftModel | None
    ) = None
    teacher: DLiteDraftModel | None = None
    draft_online: (
        OnlineDFlashModel
        | OnlineDFlash2Model
        | OnlineDSparkModel
        | OnlineDLiteModel
        | None
    ) = None
    teacher_online: OnlineDLiteModel | None = None
    fsdp_model: FullyShardedDataParallel | None = None
    components: TargetEmbeddingsAndHead | SharedTargetEmbeddingsAndHead | None = None
    target: TrainingTargetModel | None = None
    dataloader: DataLoader | None = None
    optimizer: BF16Optimizer | None = None
    tracker: Tracker | None = None
    topology: DisaggregatedTopology | None = None
    routes: list[RouteFragment] | None = None
    packet_metadata: dict[str, Any] = field(default_factory=dict)
    resume_state: dict[str, Any] | None = None
    steps_per_epoch: int = 0
    total_steps: int = 0
    tp_draft_rank: int | None = None
    teacher_identity: str | None = None
    # Capture once per run, including resume; checkpoint saves never reread YAML.
    training_yaml: str | None = field(
        default_factory=lambda: os.environ.get("LITESPEC_TRAINING_YAML")
    )

    @property
    def process_group(self) -> ProcessGroup | None:
        return self.topology.draft_group if self.topology is not None else None

    @property
    def coordinator_rank(self) -> int:
        return self.topology.draft_global_ranks[0] if self.topology is not None else 0

    @property
    def phase(self) -> Phase:
        return self.phases[self.state.phase_index]


@dataclass(frozen=True)
class TrainingAdapter:
    algorithm: str
    architecture: str
    modes: tuple[str, ...] = ()

    @property
    def model_class(self):
        from litespec.modeling.draft.registry import resolve_draft

        return resolve_draft(self.architecture)

    def parse_args(self, mode=None, argv=None):
        if self.modes:
            if mode not in self.modes:
                raise ValueError(f"{self.algorithm} requires mode in {self.modes}")
        elif mode is not None:
            raise ValueError(f"{self.algorithm} does not have training modes")
        module = import_module(f"litespec.training.recipes.{self.algorithm}")
        return module.parse_args(mode=mode, argv=argv)

    def build_config(self, args, *, source_config=None, model_role=None):
        raise NotImplementedError

    def build_model(self, args, *, source_config=None, model_role=None):
        import torch

        config = self.build_config(
            args, source_config=source_config, model_role=model_role
        )
        return self.model_class(config).cuda().to(torch.bfloat16)

    def load_model(self, path, attention_backend):
        import torch

        return self.model_class.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation=attention_backend
        ).cuda()

    def prepare_batch(self, draft_online, *args, **kwargs):
        return draft_online.prepare_batch(*args, **kwargs)

    def training_step(self, draft_online, batch):
        return draft_online(**batch)

    def required_target_layers(self, model):
        return set(model.target_layer_ids)

    def optimizer_parameters(self, session):
        return None

    def enter_phase(self, session, phase):
        session.draft.train()

    def detail_metrics(self, session):
        return {}

    def log_metrics(self, session, reduced):
        return reduced

    def training_metadata(self, session):
        return {}

    def checkpoint_metadata(self, model):
        return {
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "block_size": int(model.block_size),
            "num_draft_layers": int(model.config.num_hidden_layers),
            "target_layer_ids": list(model.target_layer_ids),
            "num_target_layers": int(model.config.num_target_layers),
            "target_hidden_size": int(model.config.hidden_size),
            model.method_config_key: dict(
                getattr(model.config, model.method_config_key)
            ),
        }


def get_training_adapter(algorithm):
    if algorithm not in ("dflash", "dflash2", "dspark", "dlite"):
        raise ValueError(f"Unknown training algorithm {algorithm!r}")
    return import_module(f"litespec.training.recipes.{algorithm}").ADAPTER
