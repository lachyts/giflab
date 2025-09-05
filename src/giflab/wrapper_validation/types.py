"""Validation type definitions and configuration."""

from dataclasses import dataclass, field
from typing import Any, Optional


# Timing validation exception hierarchy
class TimingValidationError(Exception):
    """Base exception for timing validation errors."""

    pass


class TimingImportError(TimingValidationError):
    """Exception raised when timing validation modules cannot be imported."""

    pass


class TimingCalculationError(TimingValidationError):
    """Exception raised when timing calculations fail."""

    pass


class TimingFileError(TimingValidationError):
    """Exception raised when timing validation cannot access GIF files."""

    pass


@dataclass
class ValidationResult:
    """Result of wrapper output validation.

    Provides structured information about validation success/failure with
    detailed context for debugging and analysis.
    """

    is_valid: bool
    validation_type: str  # "frame_count", "color_count", "timing", "quality", "integrity",
    # "timing_grid_validation", "timing_drift", "frame_delay_consistency",
    # "timing_validation_error"
    expected: Any
    actual: Any
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields and set default error message."""
        if not self.is_valid and not self.error_message:
            self.error_message = f"{self.validation_type} validation failed: expected {self.expected}, got {self.actual}"


# Timing Validation Type Documentation
"""
Timing-specific validation types and their expected data structures:

timing_grid_validation:
    expected: {
        "max_drift_ms": int,           # Maximum acceptable timing drift
        "duration_diff_threshold": int, # Maximum total duration difference
        "alignment_accuracy_min": float, # Minimum alignment accuracy (0.0-1.0)
        "grid_ms": int                 # Grid size used for alignment
    }
    actual: {
        "max_timing_drift_ms": int,    # Actual maximum timing drift
        "total_duration_diff_ms": int, # Actual total duration difference
        "alignment_accuracy": float,   # Actual alignment accuracy
        "timing_drift_score": float    # Normalized timing drift score (0.0-1.0)
    }
    details: {
        "timing_metrics": TimingMetrics,      # Complete timing metrics object
        "drift_analysis": dict,               # Detailed drift analysis
        "original_frame_count": int,          # Number of frames in original
        "compressed_frame_count": int,        # Number of frames in compressed
        "grid_length_original": int,          # Original animation grid length
        "grid_length_compressed": int,        # Compressed animation grid length
        "thresholds_used": dict               # Thresholds used for validation
    }

timing_validation_error:
    expected: "successful_timing_validation"
    actual: "validation_exception"
    details: {
        "exception": str,         # Exception message
        "original_gif": str,      # Path to original GIF
        "compressed_gif": str     # Path to compressed GIF
    }
"""
