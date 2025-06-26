"""Lossy compression functionality for GIF optimization."""

import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

from .frame_keep import (
    build_gifsicle_frame_args,
    build_animately_frame_args,
    validate_frame_keep_ratio,
    get_frame_reduction_info
)
from .meta import extract_gif_metadata


class LossyEngine(Enum):
    """Supported lossy compression engines."""
    GIFSICLE = "gifsicle"
    ANIMATELY = "animately"


def apply_lossy_compression(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0,
    engine: LossyEngine = LossyEngine.GIFSICLE
) -> Dict[str, Any]:
    """Apply lossy compression to a GIF using the specified engine.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level (0 = lossless, higher = more lossy)
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0), default 1.0 (all frames)
        engine: Compression engine to use
        
    Returns:
        Dictionary with compression metadata (render_ms, etc.)
        
    Raises:
        ValueError: If lossy_level is negative or frame_keep_ratio is invalid
        IOError: If input file cannot be read or output cannot be written
        RuntimeError: If compression engine fails
    """
    if lossy_level < 0:
        raise ValueError(f"lossy_level must be non-negative, got {lossy_level}")
    
    validate_frame_keep_ratio(frame_keep_ratio)
    
    if not input_path.exists():
        raise IOError(f"Input file not found: {input_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dispatch to appropriate engine
    if engine == LossyEngine.GIFSICLE:
        return compress_with_gifsicle(input_path, output_path, lossy_level, frame_keep_ratio)
    elif engine == LossyEngine.ANIMATELY:
        return compress_with_animately(input_path, output_path, lossy_level, frame_keep_ratio)
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def compress_with_gifsicle(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0
) -> Dict[str, Any]:
    """Compress GIF using gifsicle with lossy and frame reduction options.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for gifsicle
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        
    Returns:
        Dictionary with compression metadata
        
    Raises:
        RuntimeError: If gifsicle command fails
    """
    # Get GIF metadata for frame information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
    except Exception as e:
        raise RuntimeError(f"Failed to extract metadata from {input_path}: {str(e)}")
    
    # Build gifsicle command
    cmd = ["gifsicle", "--optimize"]
    
    # Add lossy compression if level > 0
    if lossy_level > 0:
        cmd.extend([f"--lossy={lossy_level}"])
    
    # Add frame reduction arguments
    if frame_keep_ratio < 1.0:
        frame_args = build_gifsicle_frame_args(frame_keep_ratio, total_frames)
        cmd.extend(frame_args)
    
    # Add input and output
    cmd.extend([str(input_path), "--output", str(output_path)])
    
    # Execute command and measure time
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        render_ms = int((end_time - start_time) * 1000)
        
        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"Gifsicle failed to create output file: {output_path}")
        
        return {
            "render_ms": render_ms,
            "engine": "gifsicle",
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "original_frames": total_frames,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Gifsicle failed with exit code {e.returncode}: {e.stderr}"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Gifsicle timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Gifsicle execution failed: {str(e)}")


def compress_with_animately(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0
) -> Dict[str, Any]:
    """Compress GIF using animately CLI with lossy and frame reduction options.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for animately
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        
    Returns:
        Dictionary with compression metadata
        
    Raises:
        RuntimeError: If animately command fails
    """
    # Path to animately launcher
    animately_path = "/Users/lachlants/bin/launcher"
    
    # Check if animately is available
    if not Path(animately_path).exists():
        raise RuntimeError(f"Animately launcher not found at: {animately_path}")
    
    # Get GIF metadata for frame information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
    except Exception as e:
        raise RuntimeError(f"Failed to extract metadata from {input_path}: {str(e)}")
    
    # Build animately command
    cmd = [animately_path, "gif", "optimize"]
    
    # Add lossy compression if level > 0
    if lossy_level > 0:
        cmd.extend(["--lossy", str(lossy_level)])
    
    # Add frame reduction arguments
    if frame_keep_ratio < 1.0:
        frame_args = build_animately_frame_args(frame_keep_ratio, total_frames)
        cmd.extend(frame_args)
    
    # Add input and output
    cmd.extend([str(input_path), str(output_path)])
    
    # Execute command and measure time
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        render_ms = int((end_time - start_time) * 1000)
        
        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"Animately failed to create output file: {output_path}")
        
        return {
            "render_ms": render_ms,
            "engine": "animately",
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "original_frames": total_frames,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Animately failed with exit code {e.returncode}: {e.stderr}"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Animately timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Animately execution failed: {str(e)}")


def validate_lossy_level(lossy_level: int, engine: LossyEngine) -> None:
    """Validate that the lossy level is appropriate for the given engine.
    
    Args:
        lossy_level: Lossy compression level to validate
        engine: Engine to validate against
        
    Raises:
        ValueError: If lossy level is not supported by the engine
    """
    if lossy_level < 0:
        raise ValueError(f"Lossy level must be non-negative, got {lossy_level}")
    
    # Both engines support the same lossy levels in this project
    # but we could add engine-specific validation here if needed
    valid_levels = [0, 40, 120]
    if lossy_level not in valid_levels:
        raise ValueError(
            f"Lossy level {lossy_level} not in supported levels: {valid_levels}"
        )


def apply_compression_with_all_params(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float,
    color_keep_count: Optional[int] = None,
    engine: LossyEngine = LossyEngine.GIFSICLE
) -> Dict[str, Any]:
    """Apply compression with all parameters (lossy, frame, color) in a single pass.
    
    This function will be extended in stage 4 to include color reduction.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level
        frame_keep_ratio: Ratio of frames to keep
        color_keep_count: Number of colors to keep (future implementation)
        engine: Compression engine to use
        
    Returns:
        Dictionary with compression metadata
    """
    # For now, just call the lossy compression with frame reduction
    # Color reduction will be added in stage 4
    return apply_lossy_compression(
        input_path,
        output_path,
        lossy_level,
        frame_keep_ratio,
        engine
    ) 