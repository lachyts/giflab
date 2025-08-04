from __future__ import annotations

import glob
import os
from shutil import copy
import time
from pathlib import Path
from typing import Any

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
    """Drop frames to achieve *keep_ratio* using a simple “delete every Nth” rule."""
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

    # Very naive deletion strategy: drop every *step* frame where
    #   step = round(1/keep_ratio).
    # For example          keep_ratio=0.5 ⇒ step=2 ⇒ delete 1--2
    step = max(2, round(1 / keep_ratio))
    delete_pattern = f"1--{step}"  # “delete every <step> frames starting at 1”

    cmd = [
        _magick_binary(),
        str(input_path),
        "-coalesce",
        "-delete",
        delete_pattern,
        "-layers",
        "optimize",
        str(output_path),
    ]

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
    metadata.update({
        "png_sequence_dir": str(output_dir),
        "frame_count": len(png_files),
        "frame_pattern": frame_pattern,
    })
    
    return metadata
