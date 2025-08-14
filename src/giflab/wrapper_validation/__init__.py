"""Wrapper output validation system for ensuring pipeline data integrity.

This package provides comprehensive validation for GIF compression tool wrappers
to catch output corruption and parameter mismatch issues that could skew experimental results.

Main Components:
- WrapperOutputValidator: Core validation framework
- ValidationResult: Structured validation result dataclass
- PipelineStageValidator: Multi-stage pipeline validation
- Frame, color, and timing validation implementations
- Integration hooks for existing wrapper system
"""

from ..config import ValidationConfig
from .core import WrapperOutputValidator
from .pipeline_validation import PipelineStageValidator
from .types import ValidationResult

__all__ = [
    "WrapperOutputValidator",
    "PipelineStageValidator",
    "ValidationResult",
    "ValidationConfig",
]
