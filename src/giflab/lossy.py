"""Lossy compression functionality for GIF optimization."""

import subprocess
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

from .frame_keep import (
    build_gifsicle_frame_args,
    build_animately_frame_args,
    validate_frame_keep_ratio,
    get_frame_reduction_info
)
from .color_keep import (
    build_gifsicle_color_args,
    build_animately_color_args,
    validate_color_keep_count,
    count_gif_colors
)
from .meta import extract_gif_metadata


def _find_animately_launcher() -> Optional[str]:
    """Find the animately launcher executable.
    
    Returns:
        Path to animately launcher or None if not found
    """
    # Check environment variable first
    env_path = os.environ.get('ANIMATELY_PATH')
    if env_path and Path(env_path).exists():
        return env_path
    
    # Common installation paths to check
    search_paths = [
        # Original hardcoded path for backwards compatibility
        "/Users/lachlants/bin/launcher",
        # Common Unix paths
        "/usr/local/bin/animately",
        "/usr/bin/animately",
        "/opt/animately/bin/launcher",
        # User paths
        os.path.expanduser("~/bin/animately"),
        os.path.expanduser("~/bin/launcher"),
        # Current directory (for development)
        "./animately",
        "./launcher"
    ]
    
    for path in search_paths:
        if Path(path).exists():
            return path
    
    # Try to find in PATH
    try:
        result = subprocess.run(['which', 'animately'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    
    return None


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
    frame_keep_ratio: float = 1.0,
    color_keep_count: Optional[int] = None
) -> Dict[str, Any]:
    """Compress GIF using gifsicle with lossy, frame reduction, and color reduction options.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for gifsicle
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (optional)
        
    Returns:
        Dictionary with compression metadata
        
    Raises:
        RuntimeError: If gifsicle command fails
    """
    # Get GIF metadata for frame and color information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
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
    
    # Add color reduction arguments
    if color_keep_count is not None:
        validate_color_keep_count(color_keep_count)
        color_args = build_gifsicle_color_args(color_keep_count, original_colors)
        cmd.extend(color_args)
    
    # Add input and output with path validation
    input_str = str(input_path.resolve())  # Resolve to absolute path
    output_str = str(output_path.resolve())  # Resolve to absolute path
    
    # Validate paths don't contain suspicious characters
    if any(char in input_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Input path contains potentially dangerous characters: {input_path}")
    if any(char in output_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Output path contains potentially dangerous characters: {output_path}")
    
    cmd.extend([input_str, "--output", output_str])
    
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
        elapsed_seconds = end_time - start_time
        # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
        render_ms = min(int(elapsed_seconds * 1000), 86400000)
        
        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"Gifsicle failed to create output file: {output_path}")
        
        return {
            "render_ms": render_ms,
            "engine": "gifsicle",
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "color_keep_count": color_keep_count,
            "original_frames": total_frames,
            "original_colors": original_colors,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Gifsicle failed with exit code {e.returncode}: {e.stderr}"
        )
    except subprocess.TimeoutExpired as e:
        # Ensure process is properly terminated on timeout
        if e.args and hasattr(e.args[0], 'kill'):
            try:
                e.args[0].kill()
            except Exception:
                pass
        raise RuntimeError(f"Gifsicle timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Gifsicle execution failed: {str(e)}")


def compress_with_animately(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0,
    color_keep_count: Optional[int] = None
) -> Dict[str, Any]:
    """Compress GIF using animately CLI with lossy, frame reduction, and color reduction options.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for animately
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (optional)
        
    Returns:
        Dictionary with compression metadata
        
    Raises:
        RuntimeError: If animately command fails
    """
    # Find animately launcher with configurable path
    animately_path = _find_animately_launcher()
    
    # Check if animately is available
    if not animately_path or not Path(animately_path).exists():
        raise RuntimeError(f"Animately launcher not found. Please install animately or set ANIMATELY_PATH environment variable.")
    
    # Get GIF metadata for frame and color information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
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
    
    # Add color reduction arguments
    if color_keep_count is not None:
        validate_color_keep_count(color_keep_count)
        color_args = build_animately_color_args(color_keep_count, original_colors)
        cmd.extend(color_args)
    
    # Add input and output with path validation
    input_str = str(input_path.resolve())  # Resolve to absolute path
    output_str = str(output_path.resolve())  # Resolve to absolute path
    
    # Validate paths don't contain suspicious characters
    if any(char in input_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Input path contains potentially dangerous characters: {input_path}")
    if any(char in output_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Output path contains potentially dangerous characters: {output_path}")
    
    cmd.extend([input_str, output_str])
    
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
        elapsed_seconds = end_time - start_time
        # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
        render_ms = min(int(elapsed_seconds * 1000), 86400000)
        
        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"Animately failed to create output file: {output_path}")
        
        return {
            "render_ms": render_ms,
            "engine": "animately",
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "color_keep_count": color_keep_count,
            "original_frames": total_frames,
            "original_colors": original_colors,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Animately failed with exit code {e.returncode}: {e.stderr}"
        )
    except subprocess.TimeoutExpired as e:
        # Ensure process is properly terminated on timeout
        if e.args and hasattr(e.args[0], 'kill'):
            try:
                e.args[0].kill()
            except Exception:
                pass
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
    color_keep_count: int,
    engine: LossyEngine = LossyEngine.GIFSICLE
) -> Dict[str, Any]:
    """Apply compression with all parameters (lossy, frame, color) in a single pass.
    
    This is the main function for single-pass compression with full parameter support.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level
        frame_keep_ratio: Ratio of frames to keep
        color_keep_count: Number of colors to keep
        engine: Compression engine to use
        
    Returns:
        Dictionary with compression metadata
        
    Raises:
        ValueError: If any parameter is invalid
        IOError: If input file cannot be read or output cannot be written
        RuntimeError: If compression engine fails
    """
    # Validate all parameters
    if lossy_level < 0:
        raise ValueError(f"lossy_level must be non-negative, got {lossy_level}")
    
    validate_frame_keep_ratio(frame_keep_ratio)
    validate_color_keep_count(color_keep_count)
    
    if not input_path.exists():
        raise IOError(f"Input file not found: {input_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dispatch to appropriate engine with all parameters
    if engine == LossyEngine.GIFSICLE:
        return compress_with_gifsicle(
            input_path, output_path, lossy_level, frame_keep_ratio, color_keep_count
        )
    elif engine == LossyEngine.ANIMATELY:
        return compress_with_animately(
            input_path, output_path, lossy_level, frame_keep_ratio, color_keep_count
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def get_compression_estimate(
    input_path: Path,
    lossy_level: int,
    frame_keep_ratio: float,
    color_keep_count: int
) -> Dict[str, Any]:
    """Estimate compression effects without actually compressing the file.
    
    Args:
        input_path: Path to input GIF file
        lossy_level: Lossy compression level
        frame_keep_ratio: Ratio of frames to keep
        color_keep_count: Number of colors to keep
        
    Returns:
        Dictionary with compression estimates
        
    Raises:
        IOError: If input file cannot be read
        ValueError: If parameters are invalid
    """
    validate_frame_keep_ratio(frame_keep_ratio)
    validate_color_keep_count(color_keep_count)
    
    if not input_path.exists():
        raise IOError(f"Input file not found: {input_path}")
    
    try:
        # Get original metadata
        metadata = extract_gif_metadata(input_path)
        
        # Calculate frame reduction estimate
        target_frames = max(1, int(metadata.orig_frames * frame_keep_ratio))
        frame_reduction = (metadata.orig_frames - target_frames) / metadata.orig_frames
        
        # Calculate color reduction estimate
        target_colors = min(color_keep_count, metadata.orig_n_colors)
        color_reduction = (metadata.orig_n_colors - target_colors) / metadata.orig_n_colors if metadata.orig_n_colors > 0 else 0.0
        
        # Rough size estimate (very approximate)
        size_reduction_estimate = frame_reduction * 0.6 + color_reduction * 0.3 + (lossy_level / 120) * 0.1
        estimated_size_kb = metadata.orig_kilobytes * (1.0 - size_reduction_estimate)
        
        return {
            "original_size_kb": metadata.orig_kilobytes,
            "estimated_size_kb": max(0.1, estimated_size_kb),  # Minimum 0.1KB
            "estimated_compression_ratio": metadata.orig_kilobytes / max(0.1, estimated_size_kb),
            "frame_reduction_percent": frame_reduction * 100.0,
            "color_reduction_percent": color_reduction * 100.0,
            "target_frames": target_frames,
            "target_colors": target_colors,
            "lossy_level": lossy_level,
            "quality_loss_estimate": min(100.0, lossy_level / 120 * 50 + frame_reduction * 30 + color_reduction * 20)
        }
        
    except Exception as e:
        raise IOError(f"Error estimating compression for {input_path}: {str(e)}") 