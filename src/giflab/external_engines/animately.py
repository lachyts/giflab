from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .common import run_command

__all__ = [
    "export_png_sequence",
]


def export_png_sequence(
    input_path: Path,
    output_dir: Path,
    *,
    frame_pattern: str = "frame_%04d.png",
) -> dict[str, Any]:
    """Export GIF frames as PNG sequence for gifski pipeline optimization.
    
    Uses ImageMagick to extract PNG frames from Animately-processed GIF files.
    This allows Animately→gifski pipelines to work with PNG sequences instead
    of failing when gifski receives a single GIF file.
    
    Args:
        input_path: Input GIF file (processed by Animately)
        output_dir: Directory to store PNG sequence  
        frame_pattern: Pattern for PNG filenames (default: frame_%04d.png)
        
    Returns:
        Metadata dict with execution info and PNG sequence details
        
    Note:
        Since Animately doesn't have direct PNG export, we use ImageMagick
        with -coalesce to properly extract frames from Animately-processed GIFs.
        This prevents the "single frame" errors in gifski pipelines.
    """
    # Import ImageMagick binary from the ImageMagick engine
    from .imagemagick import _magick_binary
    
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
    
    metadata = run_command(cmd, engine="animately_png_export", output_path=output_pattern)
    
    # Count generated PNG files
    png_files = glob.glob(str(output_dir / "frame_*.png"))
    
    # Add PNG sequence info to metadata
    metadata.update({
        "png_sequence_dir": str(output_dir),
        "frame_count": len(png_files),
        "frame_pattern": frame_pattern,
    })
    
    return metadata 