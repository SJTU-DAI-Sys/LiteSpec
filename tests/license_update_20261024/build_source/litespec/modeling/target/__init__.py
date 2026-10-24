"""Training target-model backends.

SGLang is imported lazily by ``training_target_model`` so the HF backend remains
usable without the optional training dependency.
"""

from .training_target_model import (
    HFTrainingTargetModel,
    SGLangTrainingTargetModel,
    TargetTrainingOutput,
    TrainingTargetModel,
    get_training_target_model,
)

__all__ = [
    "TargetTrainingOutput",
    "TrainingTargetModel",
    "HFTrainingTargetModel",
    "SGLangTrainingTargetModel",
    "get_training_target_model",
]
