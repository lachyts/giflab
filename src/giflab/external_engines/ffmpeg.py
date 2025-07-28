from __future__ import annotations

import glob
import tempfile
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


def _ffmpeg_binary() -> str:
    info = discover_tool("ffmpeg")
    info.require()
    return info.name


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def color_reduce(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Palette-based color reduction via two-pass FFmpeg."""
    ffmpeg = _ffmpeg_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        palette_path = Path(tmpdir) / "palette.png"

        # 1️⃣ generate palette (no fps filter to avoid pipeline conflicts)
        gen_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-filter_complex",
            "palettegen",
            str(palette_path),
        ]
        meta1 = run_command(gen_cmd, engine="ffmpeg", output_path=palette_path)

        # 2️⃣ apply palette (no fps filter to avoid pipeline conflicts)
        use_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-i",
            str(palette_path),
            "-filter_complex",
            "paletteuse",
            str(output_path),
        ]
        meta2 = run_command(use_cmd, engine="ffmpeg", output_path=output_path)

        # 3️⃣ Combine metadata from both passes
        return {
            "render_ms": meta1.get("render_ms", 0) + meta2.get("render_ms", 0),
            "engine": "ffmpeg",
            "command": f"{meta1.get('command', '')}\n{meta2.get('command', '')}",
            "kilobytes": meta2.get("kilobytes", 0),
        }


def frame_reduce(
    input_path: Path,
    output_path: Path,
    *,
    fps: float,
) -> dict[str, Any]:
    """Reduce frame-rate to *fps* using FFmpeg."""
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        f"fps={fps}",
        str(output_path),
    ]
    return run_command(cmd, engine="ffmpeg", output_path=output_path)


def lossy_compress(
    input_path: Path,
    output_path: Path,
    *,
    q_scale: int = 4,
) -> dict[str, Any]:
    """Lossy compression via FFmpeg."""
    if q_scale < 1 or q_scale > 31:
        raise ValueError("q_scale must be in 1–31 range")

    ffmpeg = _ffmpeg_binary()

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-q:v",
        str(q_scale),
        str(output_path),
    ]

    return run_command(cmd, engine="ffmpeg", output_path=output_path)


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
    """
    ffmpeg = _ffmpeg_binary()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build output path with pattern
    output_pattern = output_dir / frame_pattern
    
    # Get proper frame rate to avoid over-extraction with animately-processed GIFs
    # Some tools (like animately) create timing metadata that confuses FFmpeg
    import subprocess
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(input_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        avg_frame_rate = result.stdout.strip()
        
        # Use explicit frame rate if we got a valid one
        if avg_frame_rate and avg_frame_rate != "0/0":
            cmd = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-r", avg_frame_rate,  # Use detected frame rate
                "-i",
                str(input_path),
                str(output_pattern),
            ]
        else:
            # Fallback to original method if frame rate detection fails
            cmd = [
                ffmpeg,
                "-y",
                "-v",
                "error", 
                "-i",
                str(input_path),
                str(output_pattern),
            ]
    except Exception:
        # Fallback to original method if frame rate detection fails
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error", 
            "-i",
            str(input_path),
            str(output_pattern),
        ]
    
    metadata = run_command(cmd, engine="ffmpeg", output_path=output_pattern)
    
    # Count generated PNG files
    png_files = glob.glob(str(output_dir / "frame_*.png"))
    
    # Add PNG sequence info to metadata
    metadata.update({
        "png_sequence_dir": str(output_dir),
        "frame_count": len(png_files),
        "frame_pattern": frame_pattern,
    })
    
    return metadata
