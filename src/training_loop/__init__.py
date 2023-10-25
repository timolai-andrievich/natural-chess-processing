"""Module containing the trainer class - TrainingLoop,
and config parsing function.
"""
from .training_loop import TrainingLoop
from . import config

__all__ = ["TrainingLoop", "config"]
