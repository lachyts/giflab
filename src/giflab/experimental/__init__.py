"""Experimental pipeline testing framework - modular components.

This module provides the fully modularized experimental components:
- ExperimentalRunner: Core experimental runner for systematic pipeline testing
- ExperimentResult: Result dataclass for experimental analysis
- ParetoAnalyzer: Pareto frontier analysis for pipeline efficiency
- PipelineSampler: Intelligent sampling strategies
- SAMPLING_STRATEGIES: Available sampling strategy configurations

The ExperimentalRunner can now be imported from either:
- giflab.experimental (backwards compatibility)
- giflab.experimental.runner (direct access)
"""

# Import modular components
# Import cache and error components that were already extracted
from ..elimination_cache import PipelineResultsCache, get_git_commit
from ..elimination_errors import ErrorTypes

# Import synthetic GIF components for backward compatibility
from ..synthetic_gifs import SyntheticGifSpec
from .pareto import ParetoAnalyzer
from .runner import ExperimentalRunner, ExperimentResult
from .sampling import SAMPLING_STRATEGIES, PipelineSampler, SamplingStrategy

# Re-export key components
__all__ = [
    # Modular components
    "ParetoAnalyzer",
    "PipelineSampler",
    "SAMPLING_STRATEGIES",
    "SamplingStrategy",
    "ExperimentalRunner",
    "ExperimentResult",
    # Cache and error handling
    "PipelineResultsCache",
    "ErrorTypes",
    "get_git_commit",
    # Synthetic GIF components (backward compatibility)
    "SyntheticGifSpec",
]
