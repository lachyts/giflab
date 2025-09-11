"""Frame sampling module for efficient GIF validation."""

from .frame_sampler import (
    FrameSampler,
    SamplingResult,
    SamplingStrategy,
)
from .strategies import (
    UniformSampler,
    AdaptiveSampler,
    ProgressiveSampler,
    SceneAwareSampler,
)

__all__ = [
    "FrameSampler",
    "SamplingResult",
    "SamplingStrategy",
    "UniformSampler",
    "AdaptiveSampler",
    "ProgressiveSampler",
    "SceneAwareSampler",
]