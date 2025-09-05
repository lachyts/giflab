"""Extended lossy compression functionality with advanced gifsicle options.

This module extends the base lossy compression functionality to support
advanced gifsicle options including different optimization levels and
dithering modes. These options are useful for experimental testing
of compression effectiveness.

Extended Gifsicle Options:
- Optimization levels: --optimize, -O1, -O2, -O3
- Dithering modes: --no-dither, --dither, --dither=ordered
- Advanced color reduction methods
- Custom compression strategies
"""

import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

from .color_keep import validate_color_keep_count
from .config import DEFAULT_ENGINE_CONFIG
from .frame_keep import build_gifsicle_frame_args, validate_frame_keep_ratio
from .input_validation import validate_path_security
from .meta import extract_gif_metadata


def validate_lossy_level_for_engine(lossy_level: int, engine_name: str) -> None:
    """Validate lossy level for specific engine with engine-aware ranges.

    Engine ranges:
    - Gifsicle: 0-300
    - Animately: 0-100
    - FFmpeg: 0-100
    - Gifski: 0-100
    - ImageMagick: 0-100

    Args:
        lossy_level: Lossy compression level to validate
        engine_name: Name of the engine (case-insensitive)

    Raises:
        ValueError: If lossy level is outside valid range for the engine
    """
    engine_name_lower = engine_name.lower()

    if "gifsicle" in engine_name_lower:
        max_lossy = 300
        engine_display = "Gifsicle"
    else:
        # Most engines use 0-100 range
        max_lossy = 100
        engine_display = engine_name

    if lossy_level < 0 or lossy_level > max_lossy:
        raise ValueError(
            f"lossy_level must be between 0 and {max_lossy} for {engine_display}, got {lossy_level}"
        )


class GifsicleOptimizationLevel(Enum):
    """Gifsicle optimization levels."""

    BASIC = "basic"  # --optimize
    LEVEL1 = "level1"  # -O1
    LEVEL2 = "level2"  # -O2
    LEVEL3 = "level3"  # -O3


class GifsicleDitheringMode(Enum):
    """Gifsicle dithering modes."""

    NONE = "none"  # --no-dither
    FLOYD = "floyd"  # --dither (default Floyd-Steinberg)
    ORDERED = "ordered"  # --dither=ordered


def build_gifsicle_optimization_args(opt_level: GifsicleOptimizationLevel) -> list[str]:
    """Build gifsicle optimization arguments.

    Args:
        opt_level: Optimization level to use

    Returns:
        List of command line arguments for optimization
    """
    if opt_level == GifsicleOptimizationLevel.BASIC:
        return ["--optimize"]
    elif opt_level == GifsicleOptimizationLevel.LEVEL1:
        return ["-O1"]
    elif opt_level == GifsicleOptimizationLevel.LEVEL2:
        return ["-O2"]
    elif opt_level == GifsicleOptimizationLevel.LEVEL3:
        return ["-O3"]
    else:
        raise ValueError(f"Unknown optimization level: {opt_level}")


def build_gifsicle_dithering_args(dither_mode: GifsicleDitheringMode) -> list[str]:
    """Build gifsicle dithering arguments.

    Args:
        dither_mode: Dithering mode to use

    Returns:
        List of command line arguments for dithering
    """
    if dither_mode == GifsicleDitheringMode.NONE:
        return ["--no-dither"]
    elif dither_mode == GifsicleDitheringMode.FLOYD:
        return ["--dither"]
    elif dither_mode == GifsicleDitheringMode.ORDERED:
        return ["--dither=ordered"]
    else:
        raise ValueError(f"Unknown dithering mode: {dither_mode}")


def build_gifsicle_color_args_extended(
    color_count: int,
    original_colors: int,
    dither_mode: GifsicleDitheringMode = GifsicleDitheringMode.NONE,
    color_method: str | None = None,
) -> list[str]:
    """Build extended gifsicle color reduction arguments.

    Args:
        color_count: Target number of colors
        original_colors: Original number of colors
        dither_mode: Dithering mode to use
        color_method: Color reduction method (diversity, blend-diversity, etc.)

    Returns:
        List of command line arguments for color reduction
    """
    args = []

    # Add color reduction if needed
    if color_count < original_colors and color_count < 256:
        args.extend(["--colors", str(color_count)])

        # Add color method if specified
        if color_method:
            args.extend(["--color-method", color_method])

        # Add dithering options
        args.extend(build_gifsicle_dithering_args(dither_mode))

    return args


def compress_with_gifsicle_extended(
    input_path: Path,
    output_path: Path,
    lossy_level: int = 0,
    frame_keep_ratio: float = 1.0,
    color_keep_count: int | None = 256,
    optimization_level: GifsicleOptimizationLevel = GifsicleOptimizationLevel.BASIC,
    dithering_mode: GifsicleDitheringMode = GifsicleDitheringMode.NONE,
    color_method: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Compress GIF using gifsicle with extended options.

    This function provides access to advanced gifsicle options including:
    - Different optimization levels (-O1, -O2, -O3)
    - Dithering modes (none, floyd-steinberg, ordered)
    - Color reduction methods
    - Custom arguments

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level (0-200)
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (``None`` preserves original palette)
        optimization_level: Gifsicle optimization level
        dithering_mode: Dithering mode for color reduction
        color_method: Color reduction method
        extra_args: Additional command line arguments

    Returns:
        Dictionary with compression metadata

    Raises:
        RuntimeError: If gifsicle command fails
        ValueError: If parameters are invalid
    """
    # Validate parameters using engine-aware validation
    validate_lossy_level_for_engine(lossy_level, "gifsicle")

    validate_frame_keep_ratio(frame_keep_ratio)
    # Allow *None* to preserve existing palette (no validation or color args)
    if color_keep_count is not None:
        validate_color_keep_count(color_keep_count)

    if not input_path.exists():
        raise OSError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get GIF metadata
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
    except Exception as e:
        raise RuntimeError(f"Failed to extract metadata from {input_path}: {e}") from e

    # Build gifsicle command
    gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
    if not gifsicle_path:
        raise RuntimeError("gifsicle path not configured")

    # Validate paths
    validated_input = validate_path_security(input_path)
    validated_output = validate_path_security(output_path)

    input_str = str(validated_input.resolve())
    output_str = str(validated_output.resolve())

    # Build command
    cmd = [gifsicle_path]

    # Add optimization arguments
    cmd.extend(build_gifsicle_optimization_args(optimization_level))

    # Add lossy compression if level > 0
    if lossy_level > 0:
        cmd.extend([f"--lossy={lossy_level}"])

    # Add extended color reduction arguments
    if color_keep_count is not None:
        color_args = build_gifsicle_color_args_extended(
            color_keep_count, original_colors, dithering_mode, color_method
        )
        cmd.extend(color_args)

    # Add input file
    cmd.append(input_str)

    # Add frame reduction arguments (after input file)
    if frame_keep_ratio < 1.0:
        frame_args = build_gifsicle_frame_args(frame_keep_ratio, total_frames)
        cmd.extend(frame_args)

    # Add extra arguments if provided
    if extra_args:
        cmd.extend(extra_args)

    # Add output
    cmd.extend(["--output", output_str])

    # Execute command
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=300
        )

        end_time = time.time()
        render_ms = min(int((end_time - start_time) * 1000), 86400000)

        # Verify output
        if not output_path.exists():
            raise RuntimeError(f"Gifsicle failed to create output file: {output_path}")

        # Get version info
        try:
            version_result = subprocess.run(
                [gifsicle_path, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            engine_version = version_result.stdout.strip().split("\n")[0]
        except Exception:
            engine_version = "unknown"

        return {
            "render_ms": render_ms,
            "engine": "gifsicle",
            "engine_version": engine_version,
            "optimization_level": optimization_level.value,
            "dithering_mode": dithering_mode.value,
            "color_method": color_method,
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "color_keep_count": color_keep_count,
            "original_frames": total_frames,
            "original_colors": original_colors,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None,
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Gifsicle failed: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Gifsicle timed out") from e
    except Exception as e:
        raise RuntimeError(f"Gifsicle execution failed: {e}") from e


def apply_compression_strategy(
    input_path: Path,
    output_path: Path,
    strategy: str,
    lossy_level: int = 0,
    frame_keep_ratio: float = 1.0,
    color_keep_count: int | None = 256,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply compression using a named strategy.

    Supported strategies:
    - "pure_gifsicle": Basic gifsicle with --optimize
    - "gifsicle_dithered": Gifsicle with Floyd-Steinberg dithering
    - "gifsicle_optimized": Gifsicle with -O3 optimization
    - "gifsicle_ordered_dither": Gifsicle with ordered dithering
    - "gifsicle_custom": Custom gifsicle options via kwargs

    Args:
        input_path: Path to input GIF
        output_path: Path to output GIF
        strategy: Strategy name
        lossy_level: Lossy compression level
        frame_keep_ratio: Frame keep ratio
        color_keep_count: Number of colors to keep (``None`` preserves original palette)
        **kwargs: Additional strategy-specific arguments

    Returns:
        Dictionary with compression metadata

    Raises:
        ValueError: If strategy is not supported
    """
    if strategy == "pure_gifsicle":
        return compress_with_gifsicle_extended(
            input_path,
            output_path,
            lossy_level,
            frame_keep_ratio,
            color_keep_count,
            optimization_level=GifsicleOptimizationLevel.BASIC,
            dithering_mode=GifsicleDitheringMode.NONE,
        )

    elif strategy == "gifsicle_dithered":
        return compress_with_gifsicle_extended(
            input_path,
            output_path,
            lossy_level,
            frame_keep_ratio,
            color_keep_count,
            optimization_level=GifsicleOptimizationLevel.BASIC,
            dithering_mode=GifsicleDitheringMode.FLOYD,
        )

    elif strategy == "gifsicle_optimized":
        return compress_with_gifsicle_extended(
            input_path,
            output_path,
            lossy_level,
            frame_keep_ratio,
            color_keep_count,
            optimization_level=GifsicleOptimizationLevel.LEVEL3,
            dithering_mode=GifsicleDitheringMode.NONE,
        )

    elif strategy == "gifsicle_ordered_dither":
        return compress_with_gifsicle_extended(
            input_path,
            output_path,
            lossy_level,
            frame_keep_ratio,
            color_keep_count,
            optimization_level=GifsicleOptimizationLevel.BASIC,
            dithering_mode=GifsicleDitheringMode.ORDERED,
        )

    elif strategy == "gifsicle_custom":
        opt_level = kwargs.get("optimization_level", GifsicleOptimizationLevel.BASIC)
        dither_mode = kwargs.get("dithering_mode", GifsicleDitheringMode.NONE)
        color_method = kwargs.get("color_method", None)
        extra_args = kwargs.get("extra_args", None)

        return compress_with_gifsicle_extended(
            input_path,
            output_path,
            lossy_level,
            frame_keep_ratio,
            color_keep_count,
            optimization_level=opt_level,
            dithering_mode=dither_mode,
            color_method=color_method,
            extra_args=extra_args,
        )

    else:
        raise ValueError(f"Unknown compression strategy: {strategy}")


def compare_compression_strategies(
    input_path: Path,
    output_dir: Path,
    strategies: list[str],
    lossy_level: int = 0,
    frame_keep_ratio: float = 1.0,
    color_keep_count: int | None = 256,
) -> dict[str, dict[str, Any]]:
    """Compare multiple compression strategies on the same input.

    Args:
        input_path: Path to input GIF
        output_dir: Directory to save results
        strategies: List of strategy names to compare
        lossy_level: Lossy compression level
        frame_keep_ratio: Frame keep ratio
        color_keep_count: Color keep count

    Returns:
        Dictionary mapping strategy names to compression results
    """
    results = {}

    for strategy in strategies:
        try:
            output_path = output_dir / f"{strategy}_output.gif"
            result = apply_compression_strategy(
                input_path,
                output_path,
                strategy,
                lossy_level,
                frame_keep_ratio,
                color_keep_count,
            )
            results[strategy] = result
        except Exception as e:
            results[strategy] = {"error": str(e)}

    return results


def get_available_strategies() -> list[str]:
    """Get list of available compression strategies.

    Returns:
        List of strategy names
    """
    return [
        "pure_gifsicle",
        "gifsicle_dithered",
        "gifsicle_optimized",
        "gifsicle_ordered_dither",
        "gifsicle_custom",
    ]


def get_strategy_description(strategy: str) -> str:
    """Get description of a compression strategy.

    Args:
        strategy: Strategy name

    Returns:
        Description string
    """
    descriptions = {
        "pure_gifsicle": "Basic gifsicle with --optimize",
        "gifsicle_dithered": "Gifsicle with Floyd-Steinberg dithering",
        "gifsicle_optimized": "Gifsicle with -O3 optimization",
        "gifsicle_ordered_dither": "Gifsicle with ordered dithering",
        "gifsicle_custom": "Custom gifsicle options",
    }
    return descriptions.get(strategy, "Unknown strategy")
