"""Error handling and categorization for pipeline elimination.

This module provides error type constants and categorization logic
for pipeline elimination failures.
"""

from __future__ import annotations


class ErrorTypes:
    """Constants for error type categorization."""
    GIFSKI = 'gifski'
    FFMPEG = 'ffmpeg'
    IMAGEMAGICK = 'imagemagick'
    GIFSICLE = 'gifsicle'
    ANIMATELY = 'animately'
    TIMEOUT = 'timeout'
    COMMAND_EXECUTION = 'command_execution'
    OTHER = 'other'
    
    @classmethod
    def all_types(cls) -> list[str]:
        """Return all error type constants."""
        return [cls.GIFSKI, cls.FFMPEG, cls.IMAGEMAGICK, cls.GIFSICLE, 
                cls.ANIMATELY, cls.TIMEOUT, cls.COMMAND_EXECUTION, cls.OTHER]
    
    @classmethod
    def categorize_error(cls, error_msg: str) -> str:
        """Categorize an error message into error type constants.
        
        Args:
            error_msg: Error message string to categorize
            
        Returns:
            Error type constant string
        """
        error_msg_lower = error_msg.lower()
        
        if cls.GIFSKI in error_msg_lower:
            return cls.GIFSKI
        elif cls.FFMPEG in error_msg_lower:
            return cls.FFMPEG
        elif cls.IMAGEMAGICK in error_msg_lower:
            return cls.IMAGEMAGICK
        elif cls.GIFSICLE in error_msg_lower:
            return cls.GIFSICLE
        elif cls.ANIMATELY in error_msg_lower:
            return cls.ANIMATELY
        elif 'command failed' in error_msg_lower:
            return cls.COMMAND_EXECUTION
        elif 'timeout' in error_msg_lower:
            return cls.TIMEOUT
        else:
            return cls.OTHER


def clean_error_for_analysis(error_msg: str) -> str:
    """Clean error message for analysis and storage.
    
    Args:
        error_msg: Raw error message
        
    Returns:
        Cleaned error message suitable for analysis
    """
    # Import here to avoid circular imports
    from .error_handling import clean_error_message
    return clean_error_message(error_msg)


def extract_tool_from_error(error_msg: str) -> str:
    """Extract the tool name from an error message.
    
    Args:
        error_msg: Error message string
        
    Returns:
        Tool name or 'unknown' if not identifiable
    """
    error_msg_lower = error_msg.lower()
    
    # Check for specific tool mentions
    if 'gifski' in error_msg_lower:
        return 'gifski'
    elif 'ffmpeg' in error_msg_lower:
        return 'ffmpeg'
    elif 'imagemagick' in error_msg_lower or 'magick' in error_msg_lower:
        return 'imagemagick'
    elif 'gifsicle' in error_msg_lower:
        return 'gifsicle'
    elif 'animately' in error_msg_lower:
        return 'animately'
    else:
        return 'unknown'


def is_recoverable_error(error_msg: str) -> bool:
    """Determine if an error is potentially recoverable.
    
    Args:
        error_msg: Error message string
        
    Returns:
        True if error might be recoverable with retry or different parameters
    """
    error_msg_lower = error_msg.lower()
    
    # Non-recoverable errors (pipeline design issues)
    non_recoverable_patterns = [
        'only 1 valid frame',  # gifski single frame
        'dimension inconsistency is too severe',  # gifski dimension mismatch
        'invalid pipeline contains',  # validation errors
        'external-tool',  # invalid tool references
        'not found',  # missing tools/files
    ]
    
    for pattern in non_recoverable_patterns:
        if pattern in error_msg_lower:
            return False
    
    # Potentially recoverable errors
    recoverable_patterns = [
        'timeout',
        'temporary',
        'busy',
        'locked',
        'network',
        'connection',
    ]
    
    for pattern in recoverable_patterns:
        if pattern in error_msg_lower:
            return True
            
    # Default to non-recoverable for unknown errors
    return False 