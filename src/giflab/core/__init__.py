"""Core pipeline testing framework - modular components.

This module provides the core pipeline testing and analysis components:
- GifLabRunner: Main pipeline runner for systematic testing and optimization
- AnalysisResult: Result dataclass for pipeline analysis
- ParetoAnalyzer: Pareto frontier analysis for pipeline efficiency
- PipelineSampler: Intelligent sampling strategies
- SAMPLING_STRATEGIES: Available sampling strategy configurations

The GifLabRunner provides comprehensive pipeline testing with:
- Multi-engine compression testing
- Quality metrics analysis
- Intelligent sampling strategies
- Result caching and resume functionality
"""

# Import modular components
# Import cache and error components that were already extracted
from ..elimination_cache import PipelineResultsCache, get_git_commit
from ..elimination_errors import ErrorTypes

# Import synthetic GIF components for backward compatibility
from ..synthetic_gifs import SyntheticGifSpec
from .pareto import ParetoAnalyzer
from .runner import GifLabRunner, AnalysisResult
from .sampling import SAMPLING_STRATEGIES, PipelineSampler, SamplingStrategy

# Re-export key components
__all__ = [
    # Core components
    "GifLabRunner",
    "AnalysisResult", 
    "ParetoAnalyzer",
    "PipelineSampler",
    "SAMPLING_STRATEGIES",
    "SamplingStrategy",
    # Cache and error handling
    "PipelineResultsCache",
    "ErrorTypes",
    "get_git_commit",
    # Synthetic GIF components
    "SyntheticGifSpec",
]
