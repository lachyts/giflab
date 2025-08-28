"""
GifLab Optimization Validation System

Provides comprehensive validation for GIF optimization pipeline results with:
- Frame count and FPS optimization validation
- Quality preservation threshold checking 
- Content-type specific optimization rules
- Multi-metric validation combinations for optimization success
- Terminal-accessible validation results for automated optimization workflows
"""

from .validation_checker import ValidationChecker
from .data_structures import ValidationResult, ValidationStatus, ValidationIssue, ValidationWarning

__all__ = [
    "ValidationChecker",
    "ValidationResult", 
    "ValidationStatus",
    "ValidationIssue",
    "ValidationWarning",
]