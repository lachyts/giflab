"""Lossy compression functionality for GIF optimization."""

from pathlib import Path
from typing import Dict, Any
from enum import Enum


class LossyEngine(Enum):
    """Supported lossy compression engines."""
    GIFSICLE = "gifsicle"
    ANIMATELY = "animately"


def apply_lossy_compression(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    engine: LossyEngine = LossyEngine.GIFSICLE
) -> Dict[str, Any]:
    """Apply lossy compression to a GIF using the specified engine.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level (0 = lossless, higher = more lossy)
        engine: Compression engine to use
        
    Returns:
        Dictionary with compression metadata (render_ms, etc.)
        
    Raises:
        ValueError: If lossy_level is negative
        IOError: If input file cannot be read or output cannot be written
        RuntimeError: If compression engine fails
    """
    if lossy_level < 0:
        raise ValueError(f"lossy_level must be non-negative, got {lossy_level}")
    
    # TODO: Implement lossy compression
    # This will be implemented in Stage 4 (S4)
    raise NotImplementedError("Lossy compression not yet implemented")


def compress_with_gifsicle(
    input_path: Path,
    output_path: Path,
    lossy_level: int
) -> Dict[str, Any]:
    """Compress GIF using gifsicle with lossy options.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for gifsicle
        
    Returns:
        Dictionary with compression metadata
    """
    # TODO: Implement gifsicle compression
    # This will be implemented in Stage 4 (S4)
    raise NotImplementedError("Gifsicle compression not yet implemented")


def compress_with_animately(
    input_path: Path,
    output_path: Path,
    lossy_level: int
) -> Dict[str, Any]:
    """Compress GIF using animately CLI with lossy options.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for animately
        
    Returns:
        Dictionary with compression metadata
    """
    # TODO: Implement animately compression
    # This will be implemented in Stage 4 (S4)
    raise NotImplementedError("Animately compression not yet implemented") 