"""Compat shim for GifLab experimental framework.

Re-exports the public API from the `giflab.experimental` package while
eliminating the previous module/package name collision.
"""

from .experimental import (
    ExperimentalRunner,
    ExperimentResult,
    ParetoAnalyzer,
    PipelineSampler,
    SAMPLING_STRATEGIES,
    SamplingStrategy,
    PipelineResultsCache,
    get_git_commit,
    ErrorTypes,
)

__all__: list[str] = [
    "ExperimentalRunner",
    "ExperimentResult",
    "ParetoAnalyzer",
    "PipelineSampler",
    "SAMPLING_STRATEGIES",
    "SamplingStrategy",
    "PipelineResultsCache",
    "get_git_commit",
    "ErrorTypes",
]
