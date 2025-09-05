"""Standardized Error Handling Utilities

Provides consistent error handling patterns across the GifLab codebase
to improve debugging and error reporting consistency.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any


class ErrorLevel(Enum):
    """Error severity levels for consistent logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class GifLabError(Exception):
    """Base exception class for all GifLab errors."""

    def __init__(
        self, message: str, cause: Exception | None = None, context: dict | None = None
    ):
        super().__init__(message)
        self.cause = cause
        self.context = context or {}
        self.timestamp = None  # Will be set when logged

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.cause:
            return f"{base_msg} (caused by: {self.cause})"
        return base_msg


class ValidationError(GifLabError):
    """Raised when input validation fails."""

    pass


class ProcessingError(GifLabError):
    """Raised when GIF processing operations fail."""

    pass


class EngineError(GifLabError):
    """Raised when external engine operations fail."""

    pass


class ConfigurationError(GifLabError):
    """Raised when configuration is invalid or missing."""

    pass


class MetricsError(GifLabError):
    """Raised when quality metrics calculation fails."""

    pass


def handle_error(
    error: Exception,
    operation: str,
    error_type: type[GifLabError] = ProcessingError,
    level: ErrorLevel = ErrorLevel.ERROR,
    context: dict | None = None,
    logger: logging.Logger | None = None,
    reraise: bool = True,
) -> GifLabError | None:
    """Standardized error handling with consistent logging and error transformation.

    Args:
        error: Original exception that occurred
        operation: Description of operation that failed
        error_type: Type of GifLabError to raise
        level: Logging level for the error
        context: Additional context information
        logger: Logger to use (defaults to module logger)
        reraise: Whether to reraise the transformed exception

    Returns:
        The transformed error if reraise=False, otherwise None

    Raises:
        GifLabError: Transformed error if reraise=True
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create standardized error message
    message = f"Failed to {operation}: {error}"

    # Create context with error details
    error_context = context or {}
    error_context.update(
        {
            "operation": operation,
            "original_error_type": type(error).__name__,
            "original_error_message": str(error),
            "has_traceback": True,
        }
    )

    # Create transformed error
    transformed_error = error_type(message, cause=error, context=error_context)

    # Log with appropriate level and context
    log_message = f"🚨 {operation.capitalize()} failed: {error}"
    if error_context:
        context_str = ", ".join(
            f"{k}={v}" for k, v in error_context.items() if k not in ["has_traceback"]
        )
        if context_str:
            log_message += f" (context: {context_str})"

    # Log at appropriate level
    log_func = getattr(logger, level.value)
    log_func(log_message)

    # Log traceback at debug level for investigation
    if level in [ErrorLevel.ERROR, ErrorLevel.CRITICAL]:
        logger.debug(f"Traceback for {operation}: {traceback.format_exc()}")

    if reraise:
        raise transformed_error from error
    else:
        return transformed_error


@contextmanager
def error_context(
    operation: str,
    error_type: type[GifLabError] = ProcessingError,
    level: ErrorLevel = ErrorLevel.ERROR,
    context: dict | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Context manager for standardized error handling.

    Usage:
        with error_context("process GIF file", ProcessingError, context={'file': 'test.gif'}):
            # operations that might fail
            risky_operation()

    Args:
        operation: Description of operation being performed
        error_type: Type of GifLabError to raise on failure
        level: Logging level for errors
        context: Additional context information
        logger: Logger to use
    """
    try:
        yield
    except GifLabError:
        # Re-raise GifLab errors unchanged
        raise
    except Exception as e:
        # Transform other exceptions into standardized format
        handle_error(e, operation, error_type, level, context, logger, reraise=True)


def log_warning_with_context(
    message: str, context: dict | None = None, logger: logging.Logger | None = None
) -> None:
    """Log a warning with standardized context formatting.

    Args:
        message: Warning message
        context: Additional context information
        logger: Logger to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    warning_msg = f"⚠️  {message}"
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        warning_msg += f" (context: {context_str})"

    logger.warning(warning_msg)


def log_info_with_context(
    message: str, context: dict | None = None, logger: logging.Logger | None = None
) -> None:
    """Log an info message with standardized context formatting.

    Args:
        message: Info message
        context: Additional context information
        logger: Logger to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    info_msg = f"ℹ️  {message}"
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        info_msg += f" (context: {context_str})"

    logger.info(info_msg)


def safe_operation(
    operation_func: Callable[[], Any],
    operation_name: str,
    default_return: Any = None,
    error_type: type[GifLabError] = ProcessingError,
    level: ErrorLevel = ErrorLevel.WARNING,
    context: dict | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Safely execute an operation with standardized error handling.

    Args:
        operation_func: Function to execute
        operation_name: Description of the operation
        default_return: Value to return if operation fails
        error_type: Type of error to handle
        level: Logging level for errors
        context: Additional context
        logger: Logger to use

    Returns:
        Result of operation_func or default_return if it fails
    """
    try:
        with error_context(operation_name, error_type, level, context, logger):
            return operation_func()
    except GifLabError as e:
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.warning(f"Safe operation '{operation_name}' failed, using default: {e}")
        return default_return


# Pre-configured error handlers for common use cases
def handle_validation_error(error: Exception, operation: str, **kwargs: Any) -> None:
    """Handle validation errors with consistent pattern."""
    handle_error(error, operation, ValidationError, ErrorLevel.ERROR, **kwargs)


def handle_processing_error(error: Exception, operation: str, **kwargs: Any) -> None:
    """Handle processing errors with consistent pattern."""
    handle_error(error, operation, ProcessingError, ErrorLevel.ERROR, **kwargs)


def handle_engine_error(error: Exception, operation: str, **kwargs: Any) -> None:
    """Handle external engine errors with consistent pattern."""
    handle_error(error, operation, EngineError, ErrorLevel.ERROR, **kwargs)


def handle_metrics_error(error: Exception, operation: str, **kwargs: Any) -> None:
    """Handle metrics calculation errors with consistent pattern."""
    handle_error(error, operation, MetricsError, ErrorLevel.WARNING, **kwargs)


# Backward compatibility aliases
def clean_error_message(error_msg: str) -> str:
    """Clean error message for CSV output by removing problematic characters.

    Handles all CSV-problematic characters including:
    - Newlines and carriage returns (replaced with spaces)
    - Quotes (replaced with apostrophes)
    - Commas (replaced with semicolons)
    - Tabs (replaced with spaces)
    - Control characters (removed)
    - Null bytes (removed)

    Args:
        error_msg: Raw error message string

    Returns:
        Cleaned error message safe for CSV output
    """
    import re

    # Convert to string and handle empty cases
    cleaned = str(error_msg)

    # Replace line breaks and carriage returns with spaces
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")

    # Replace quotes with apostrophes to avoid CSV escaping issues
    cleaned = cleaned.replace('"', "'").replace("`", "'")

    # Replace commas with semicolons to avoid CSV field separation issues
    cleaned = cleaned.replace(",", ";")

    # Replace tabs with spaces
    cleaned = cleaned.replace("\t", " ")

    # Remove null bytes and other control characters (except space)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)

    # Collapse multiple spaces into single spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Trim whitespace
    cleaned = cleaned.strip()

    # Limit length to prevent extremely long error messages
    max_length = 500
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 3] + "..."

    return cleaned
