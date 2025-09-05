from __future__ import annotations

import glob
import os
import tempfile
import time
from pathlib import Path
from shutil import copy
from typing import Any

from PIL import Image

from ..frame_keep import calculate_frame_indices
from ..system_tools import discover_tool
from .common import run_command

__all__ = [
    "color_reduce",
    "frame_reduce",
    "lossy_compress",
    "export_png_sequence",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _magick_binary() -> str:
    """Return the preferred ImageMagick binary (``magick`` or ``convert``)."""
    info = discover_tool("imagemagick")
    info.require()
    return info.name  # e.g. "magick" or "convert"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def color_reduce(
    input_path: Path,
    output_path: Path,
    *,
    colors: int = 32,
    dither: bool = False,
) -> dict[str, Any]:
    """Reduce color palette of *input_path* to *colors* using ImageMagick.

    Parameters
    ----------
    input_path
        Source GIF.
    output_path
        Destination GIF.
    colors
        Target palette size (1–256).
    dither
        If *True* apply dithering during quantisation.
    """
    if colors < 1 or colors > 256:
        raise ValueError("colors must be between 1 and 256 inclusive")

    cmd = [
        _magick_binary(),
        str(input_path),
    ]

    if not dither:
        cmd.append("+dither")

    cmd += ["-colors", str(colors), str(output_path)]

    return run_command(cmd, engine="imagemagick", output_path=output_path)


def frame_reduce(
    input_path: Path,
    output_path: Path,
    *,
    keep_ratio: float,
) -> dict[str, Any]:
    """Drop frames to achieve *keep_ratio* while preserving timing and looping."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")

    # Shortcut – no reduction needed.
    if keep_ratio == 1.0:
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copy(input_path, output_path)
        duration_ms = int((time.perf_counter() - start) * 1000)
        size_kb = int(os.path.getsize(output_path) / 1024)
        return {
            "render_ms": duration_ms,
            "engine": "imagemagick",
            "command": "cp",
            "kilobytes": size_kb,
        }

    # Import timing functions from frame_keep module
    from ..frame_keep import calculate_adjusted_delays, extract_gif_timing_info

    # Extract timing and loop information from original GIF
    try:
        timing_info = extract_gif_timing_info(input_path)
        original_delays = timing_info["frame_delays"]
        loop_count = timing_info["loop_count"]
        total_frames = timing_info["total_frames"]
    except Exception:
        # Fallback to old behavior if timing extraction fails
        with Image.open(input_path) as img:
            total_frames = 0
            try:
                while True:
                    img.seek(total_frames)
                    total_frames += 1
            except EOFError:
                pass
        original_delays = [100] * total_frames  # Default delays
        loop_count = 0  # Default to infinite loop

    # Calculate which frames to keep using standardized algorithm
    frames_to_keep = calculate_frame_indices(total_frames, keep_ratio)

    # Convert to ImageMagick deletion pattern
    # Keep frames by deleting all others
    all_frames = set(range(total_frames))
    frames_to_delete = sorted(all_frames - set(frames_to_keep))

    if not frames_to_delete:
        # Nothing to delete, just copy
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copy(input_path, output_path)
        duration_ms = int((time.perf_counter() - start) * 1000)
        size_kb = int(os.path.getsize(output_path) / 1024)
        return {
            "render_ms": duration_ms,
            "engine": "imagemagick",
            "command": "cp",
            "kilobytes": size_kb,
        }

    # Use simpler approach: delete frames and set uniform delay
    adjusted_delays = calculate_adjusted_delays(original_delays, frames_to_keep)

    # Build ImageMagick command with timing and loop preservation
    cmd = [_magick_binary(), str(input_path), "-coalesce"]

    # Delete unwanted frames
    delete_pattern = ",".join(str(frame) for frame in frames_to_delete)
    cmd.extend(["-delete", delete_pattern])

    # Set uniform delay based on average of adjusted delays
    if adjusted_delays:
        avg_delay = sum(adjusted_delays) / len(adjusted_delays)
        delay_ticks = max(2, int(avg_delay // 10))  # Convert ms to ticks (1/100s)
        cmd.extend(["-delay", str(delay_ticks)])

    # Preserve loop count
    if loop_count is not None:
        cmd.extend(["-loop", str(loop_count)])
    else:
        cmd.extend(["-loop", "0"])  # Default to infinite loop

    # Optimize and save
    cmd.extend(["-layers", "optimize", str(output_path)])

    return run_command(cmd, engine="imagemagick", output_path=output_path)


def lossy_compress(
    input_path: Path,
    output_path: Path,
    *,
    quality: int = 80,
) -> dict[str, Any]:
    """Lossy compression using ImageMagick's -quality flag."""
    if quality < 0 or quality > 100:
        raise ValueError("quality must be in 0–100 range")

    cmd = [
        _magick_binary(),
        str(input_path),
        "-quality",
        str(quality),
        str(output_path),
    ]

    return run_command(cmd, engine="imagemagick", output_path=output_path)


def export_png_sequence(
    input_path: Path,
    output_dir: Path,
    *,
    frame_pattern: str = "frame_%04d.png",
) -> dict[str, Any]:
    """Export GIF frames as PNG sequence for gifski pipeline optimization.

    Args:
        input_path: Input GIF file
        output_dir: Directory to store PNG sequence
        frame_pattern: Pattern for PNG filenames (default: frame_%04d.png)

    Returns:
        Metadata dict with execution info and PNG sequence details

    Note:
        ImageMagick uses -coalesce which properly handles frame timing and prevents
        the over-extraction issues that can occur with FFmpeg on animately-processed GIFs.
    """
    magick = _magick_binary()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output path with pattern
    output_pattern = output_dir / frame_pattern

    cmd = [
        magick,
        str(input_path),
        "-coalesce",  # Ensure frames are properly separated and handle timing correctly
        str(output_pattern),
    ]

    metadata = run_command(cmd, engine="imagemagick", output_path=output_pattern)

    # Count generated PNG files
    png_files = glob.glob(str(output_dir / "frame_*.png"))

    # Add PNG sequence info to metadata
    metadata.update(
        {
            "png_sequence_dir": str(output_dir),
            "frame_count": len(png_files),
            "frame_pattern": frame_pattern,
        }
    )

    return metadata
