"""Validation type definitions and configuration."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ValidationResult:
    """Result of wrapper output validation.
    
    Provides structured information about validation success/failure with
    detailed context for debugging and analysis.
    """
    
    is_valid: bool
    validation_type: str  # "frame_count", "color_count", "timing", "quality", "integrity"
    expected: Any
    actual: Any
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate required fields and set default error message."""
        if not self.is_valid and not self.error_message:
            self.error_message = f"{self.validation_type} validation failed: expected {self.expected}, got {self.actual}"


