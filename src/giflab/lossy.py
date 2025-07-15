"""Lossy compression functionality for GIF optimization.

This module provides unified interfaces for both gifsicle and animately compression engines.
Each engine has different command-line syntax and capabilities, but this module provides
a consistent Python API for both.

Engine Documentation & Best Practices:

Gifsicle (https://www.lcdf.org/gifsicle/):
- Powerful command-line GIF manipulation tool
- Frame selection syntax: #0 #2 #4 (specify frames to keep)
- Optimization levels: --optimize, -O1, -O2, -O3
- Lossy compression: --lossy=LEVEL (0-200, higher = more compression)
- Color reduction: --colors N (reduce palette to N colors)
- Command structure: gifsicle [OPTIONS] INPUT [FRAME_SELECTION] --output OUTPUT

Animately Engine:
- Internal compression engine with streamlined CLI
- Frame reduction: --reduce RATIO (0.0-1.0, decimal format)
- Lossy compression: --lossy LEVEL (compression level)
- Color reduction: --colors N (reduce palette to N colors)
- Command structure: animately --input INPUT [OPTIONS] --output OUTPUT

Key Differences:
- Gifsicle: Frame selection by index (#0 #2 #4), input file before frame args
- Animately: Frame reduction by ratio (--reduce 0.5), input/output via flags
- Gifsicle: More complex but powerful optimization options
- Animately: Simpler syntax, consistent flag-based interface

Usage Examples:
    # Gifsicle frame reduction (keep every other frame)
    gifsicle --optimize input.gif #0 #2 #4 #6 --output output.gif

    # Animately frame reduction (keep 50% of frames)
    animately --input input.gif --reduce 0.50 --output output.gif

    # Both engines with lossy compression
    gifsicle --optimize --lossy=40 input.gif --output output.gif
    animately --input input.gif --lossy 40 --output output.gif
"""

import os
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

from .color_keep import (
    build_animately_color_args,
    build_gifsicle_color_args,
    validate_color_keep_count,
)
from .config import DEFAULT_ENGINE_CONFIG
from .frame_keep import (
    build_animately_frame_args,
    build_gifsicle_frame_args,
    validate_frame_keep_ratio,
)
from .meta import extract_gif_metadata


class LossyEngine(Enum):
    """Supported lossy compression engines."""
    GIFSICLE = "gifsicle"
    ANIMATELY = "animately"


def get_gifsicle_version() -> str:
    """Get the version of gifsicle.

    Returns:
        Version string of gifsicle (e.g., "1.94")

    Raises:
        RuntimeError: If gifsicle is not available or version cannot be determined
    """
    gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

    if not gifsicle_path or not Path(gifsicle_path).exists():
        raise RuntimeError(f"Gifsicle not found at {gifsicle_path}")

    try:
        result = subprocess.run(
            [gifsicle_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            raise RuntimeError(f"Gifsicle version check failed: {result.stderr}")

        # Parse version from output (e.g., "LCDF Gifsicle 1.94")
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if 'gifsicle' in line.lower() and any(char.isdigit() for char in line):
                # Extract version number (e.g., "1.94" from "LCDF Gifsicle 1.94")
                words = line.split()
                for word in words:
                    if '.' in word and any(char.isdigit() for char in word):
                        return word

        # Fallback - return first line if version parsing fails
        return output_lines[0] if output_lines else "unknown"

    except subprocess.TimeoutExpired:
        raise RuntimeError("Gifsicle version check timed out") from None
    except Exception as e:
        raise RuntimeError(f"Failed to get gifsicle version: {str(e)}") from e


def get_animately_version() -> str:
    """Get the version of animately.

    Returns:
        Version string of animately (e.g., "1.0.0" or "animately-engine" if version not detectable)

    Raises:
        RuntimeError: If animately is not available
    """
    animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

    if not animately_path or not Path(animately_path).exists():
        raise RuntimeError(f"Animately not found at {animately_path}")

    try:
        # Try --version first
        result = subprocess.run(
            [animately_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if any(char.isdigit() for char in line):
                    # Extract version number if found
                    words = line.split()
                    for word in words:
                        if '.' in word and any(char.isdigit() for char in word):
                            return word
            return output_lines[0] if output_lines else "animately-engine"

        # If --version fails, try --help and look for version info
        result = subprocess.run(
            [animately_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 or result.stderr:
            # Check both stdout and stderr for version information
            output_text = (result.stdout + result.stderr).lower()
            if 'version' in output_text:
                # Try to extract version from help text
                lines = output_text.split('\n')
                for line in lines:
                    if 'version' in line and any(char.isdigit() for char in line):
                        words = line.split()
                        for word in words:
                            if '.' in word and any(char.isdigit() for char in word):
                                return word

        # Fallback - return a generic identifier
        return "animately-engine"

    except subprocess.TimeoutExpired:
        raise RuntimeError("Animately version check timed out") from None
    except Exception as e:
        raise RuntimeError(f"Failed to get animately version: {str(e)}") from e


def get_engine_version(engine: LossyEngine) -> str:
    """Get the version of the specified engine.

    Args:
        engine: Engine to get version for

    Returns:
        Version string of the engine

    Raises:
        RuntimeError: If engine is not available or version cannot be determined
    """
    if engine == LossyEngine.GIFSICLE:
        return get_gifsicle_version()
    elif engine == LossyEngine.ANIMATELY:
        return get_animately_version()
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def apply_lossy_compression(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0,
    engine: LossyEngine = LossyEngine.GIFSICLE
) -> dict[str, Any]:
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
        raise OSError(f"Input file not found: {input_path}")

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
    color_keep_count: int | None = None
) -> dict[str, Any]:
    """Compress GIF using gifsicle with lossy, frame reduction, and color reduction options.

    Constructs and executes a gifsicle command following best practices from:
    https://www.lcdf.org/gifsicle/

    Command Construction:
    1. Base: gifsicle --optimize
    2. Lossy: --lossy=LEVEL (if level > 0)
    3. Colors: --colors N (if color_keep_count specified)
    4. Input: INPUT_FILE (must come before frame selection)
    5. Frames: #0 #2 #4 (frame selection, if frame_keep_ratio < 1.0)
    6. Output: --output OUTPUT_FILE

    Example Commands:
        # Lossless with frame reduction
        gifsicle --optimize input.gif #0 #2 #4 #6 --output output.gif

        # Lossy with color reduction
        gifsicle --optimize --lossy=40 --colors 64 input.gif --output output.gif

        # Full optimization
        gifsicle --optimize --lossy=40 --colors 64 input.gif #0 #2 #4 --output output.gif

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for gifsicle (0=lossless, higher=more lossy)
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (optional)

    Returns:
        Dictionary with compression metadata including:
        - render_ms: Processing time in milliseconds
        - engine: "gifsicle"
        - command: Full command that was executed
        - original_frames: Number of frames in input
        - original_colors: Number of colors in input

    Raises:
        RuntimeError: If gifsicle command fails or binary not found
        ValueError: If paths contain dangerous characters

    Note:
        Frame selection (#0 #2 #4) must come AFTER input file for gifsicle.
        This is different from --delete which conflicts with --optimize.
    """
    # Get GIF metadata for frame and color information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
    except Exception as e:
        raise RuntimeError(f"Failed to extract metadata from {input_path}: {str(e)}") from e

    # Build gifsicle command
    gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
    if not gifsicle_path or not _is_executable(gifsicle_path):
        raise RuntimeError(
            "gifsicle not found. Please install it or set GIFSICLE_PATH in EngineConfig."
        )
    cmd = [gifsicle_path, "--optimize"]

    # Add lossy compression if level > 0
    if lossy_level > 0:
        cmd.extend([f"--lossy={lossy_level}"])

    # Add color reduction arguments
    if color_keep_count is not None:
        validate_color_keep_count(color_keep_count)
        color_args = build_gifsicle_color_args(color_keep_count, original_colors, dithering=False)
        cmd.extend(color_args)

    # Add input and output with path validation
    input_str = str(input_path.resolve())  # Resolve to absolute path
    output_str = str(output_path.resolve())  # Resolve to absolute path

    # Validate paths don't contain suspicious characters
    if any(char in input_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Input path contains potentially dangerous characters: {input_path}")
    if any(char in output_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Output path contains potentially dangerous characters: {output_path}")

    # Add input file first (required for frame operations)
    cmd.append(input_str)

    # Add frame reduction arguments AFTER input file
    if frame_keep_ratio < 1.0:
        frame_args = build_gifsicle_frame_args(frame_keep_ratio, total_frames)
        cmd.extend(frame_args)

    # Add output
    cmd.extend(["--output", output_str])

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

        # Get engine version
        try:
            engine_version = get_gifsicle_version()
        except RuntimeError:
            engine_version = "unknown"

        return {
            "render_ms": render_ms,
            "engine": "gifsicle",
            "engine_version": engine_version,
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
        ) from e
    except subprocess.TimeoutExpired as e:
        # Ensure process is properly terminated on timeout
        if e.args and hasattr(e.args[0], 'kill'):
            try:
                e.args[0].kill()
            except Exception:
                pass
        raise RuntimeError("Gifsicle timed out after 5 minutes") from e
    except Exception as e:
        raise RuntimeError(f"Gifsicle execution failed: {str(e)}") from e


def compress_with_animately(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0,
    color_keep_count: int | None = None
) -> dict[str, Any]:
    """Compress GIF using animately CLI with lossy, frame reduction, and color reduction options.

    Constructs and executes an animately command using its flag-based interface.

    Animately CLI Reference:
    Usage: animately.exe [OPTION...]
      -i, --input arg        Path to input gif file
      -o, --output arg       Path to output gif file
      -l, --lossy arg        Lossy compression level
      -f, --reduce arg       Reduce frames (ratio 0.0-1.0)
      -p, --colors arg       Reduce palette colors
      -d, --delay arg        Delay between frames
      -t, --trim-frames arg  Trim frames
      -m, --trim-ms arg      Trim in milliseconds
      -s, --scale arg        Scale factor
      -c, --crop arg         Crop dimensions
      -z, --zoom arg         Zoom factor
      -u, --tone arg         Duotone effect
      -r, --repeated-frame   Repeated frame optimization
      -e, --meta arg         GIF metadata
      -y, --loops arg        Loop count
      -h, --help             Show help

    Command Construction:
    1. Base: animately
    2. Input: --input INPUT_FILE
    3. Output: --output OUTPUT_FILE
    4. Lossy: --lossy LEVEL (if level > 0)
    5. Frames: --reduce RATIO (if frame_keep_ratio < 1.0)
    6. Colors: --colors N (if color_keep_count specified)

    Example Commands:
        # Lossless with frame reduction
        animately --input input.gif --reduce 0.50 --output output.gif

        # Lossy with color reduction
        animately --input input.gif --lossy 40 --colors 64 --output output.gif

        # Full optimization
        animately --input input.gif --lossy 40 --reduce 0.50 --colors 64 --output output.gif

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for animately
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (optional)

    Returns:
        Dictionary with compression metadata including:
        - render_ms: Processing time in milliseconds
        - engine: "animately"
        - command: Full command that was executed
        - original_frames: Number of frames in input
        - original_colors: Number of colors in input

    Raises:
        RuntimeError: If animately command fails or binary not found
        ValueError: If paths contain dangerous characters

    Note:
        Animately uses decimal ratios (0.50 for 50%) and consistent flag syntax.
        All parameters are specified via flags, unlike gifsicle's mixed approach.
    """
    animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

    # Check if animately is available
    if not animately_path or not _is_executable(animately_path):
        raise RuntimeError(
            "Animately launcher not found. "
            "Please install animately or set ANIMATELY_PATH in EngineConfig."
        )

    # Get GIF metadata for frame and color information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
    except Exception as e:
        raise RuntimeError(f"Failed to extract metadata from {input_path}: {str(e)}") from e

    # Build animately command
    cmd = [animately_path]

    # Add input and output with path validation
    input_str = str(input_path.resolve())  # Resolve to absolute path
    output_str = str(output_path.resolve())  # Resolve to absolute path

    # Validate paths don't contain suspicious characters
    if any(char in input_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Input path contains potentially dangerous characters: {input_path}")
    if any(char in output_str for char in [';', '&', '|', '`', '$']):
        raise ValueError(f"Output path contains potentially dangerous characters: {output_path}")

    # Add input and output paths
    cmd.extend(["--input", input_str, "--output", output_str])

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

        # Get engine version
        try:
            engine_version = get_animately_version()
        except RuntimeError:
            engine_version = "unknown"

        return {
            "render_ms": render_ms,
            "engine": "animately",
            "engine_version": engine_version,
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
        ) from e
    except subprocess.TimeoutExpired as e:
        # Ensure process is properly terminated on timeout
        if e.args and hasattr(e.args[0], 'kill'):
            try:
                e.args[0].kill()
            except Exception:
                pass
        raise RuntimeError("Animately timed out after 5 minutes") from e
    except Exception as e:
        raise RuntimeError(f"Animately execution failed: {str(e)}") from e


def _is_executable(path: str) -> bool:
    """Check if a path is an executable file.

    Args:
        path: The file path to check.

    Returns:
        True if the path is an executable file, False otherwise.
    """
    # First check if it's an absolute path
    if Path(path).is_absolute():
        return Path(path).is_file() and os.access(path, os.X_OK)
    
    # Otherwise, check if it's available in PATH
    import shutil
    return shutil.which(path) is not None


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
) -> dict[str, Any]:
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
        raise OSError(f"Input file not found: {input_path}")

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
) -> dict[str, Any]:
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
        raise OSError(f"Input file not found: {input_path}")

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
        raise OSError(f"Error estimating compression for {input_path}: {str(e)}") from e
