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
    """Reduce frame-rate to *fps* using FFmpeg with timing and loop preservation."""
    ffmpeg = _ffmpeg_binary()
    
    # Import timing functions
    from ..frame_keep import extract_gif_timing_info
    
    # Extract original timing and loop information
    try:
        timing_info = extract_gif_timing_info(input_path)
        loop_count = timing_info["loop_count"]
    except Exception:
        loop_count = 0  # Default to infinite loop
    
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        f"fps={fps}"
    ]
    
    # Preserve loop count
    if loop_count is not None and loop_count >= 0:
        cmd.extend(["-loop", str(loop_count)])
    else:
        cmd.extend(["-loop", "0"])  # Infinite loop
    
    cmd.append(str(output_path))
    
    return run_command(cmd, engine="ffmpeg", output_path=output_path)


def frame_reduce_by_ratio(
    input_path: Path,
    output_path: Path,
    *,
    keep_ratio: float,
) -> dict[str, Any]:
    """Reduce frames by *keep_ratio* using FFmpeg with proper timing preservation."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    
    # Import timing and frame functions
    import os
    import time
    from shutil import copy

    from ..frame_keep import (
        calculate_adjusted_delays,
        calculate_frame_indices,
        extract_gif_timing_info,
    )
    
    # Shortcut – no reduction needed
    if keep_ratio == 1.0:
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copy(input_path, output_path)
        duration_ms = int((time.perf_counter() - start) * 1000)
        size_kb = int(os.path.getsize(output_path) / 1024)
        return {
            "render_ms": duration_ms,
            "engine": "ffmpeg",
            "command": "cp",
            "kilobytes": size_kb,
        }
    
    # Extract timing and loop information
    try:
        timing_info = extract_gif_timing_info(input_path)
        original_delays = timing_info["frame_delays"]
        loop_count = timing_info["loop_count"]
        total_frames = timing_info["total_frames"]
    except Exception:
        # Fallback: convert to fps-based approach
        original_fps = 10.0  # Default assumption
        target_fps = original_fps * keep_ratio
        target_fps = max(target_fps, 0.1)  # Minimum FPS
        return frame_reduce(input_path, output_path, fps=target_fps)
    
    # Calculate which frames to keep
    frames_to_keep = calculate_frame_indices(total_frames, keep_ratio)
    adjusted_delays = calculate_adjusted_delays(original_delays, frames_to_keep)
    
    # Create frame selection string for FFmpeg
    # FFmpeg select filter: select frames by index
    select_expr = "+".join([f"eq(n\\,{idx})" for idx in frames_to_keep])
    
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path)
    ]
    
    # Use select filter to choose specific frames
    filters = [f"select='{select_expr}'"]
    
    # Set frame delays using setpts filter
    # Calculate time base for frame delays
    if len(adjusted_delays) > 1:
        # Create PTS (Presentation Time Stamp) values for proper timing
        pts_values: list[float] = []
        current_pts: float = 0.0
        for delay in adjusted_delays[:-1]:  # All but last frame
            pts_values.append(current_pts)
            current_pts += delay / 1000.0  # Convert ms to seconds
        pts_values.append(current_pts)  # Last frame
        
        # Apply setpts to control timing
        filters.append("setpts=N*TB")
    
    cmd.extend(["-filter_complex", ",".join(filters)])
    
    # Preserve loop count
    if loop_count is not None and loop_count >= 0:
        cmd.extend(["-loop", str(loop_count)])
    else:
        cmd.extend(["-loop", "0"])  # Infinite loop
    
    cmd.append(str(output_path))
    
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
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "csv=p=0",
            str(input_path),
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
                "-r",
                avg_frame_rate,  # Use detected frame rate
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
    metadata.update(
        {
            "png_sequence_dir": str(output_dir),
            "frame_count": len(png_files),
            "frame_pattern": frame_pattern,
        }
    )

    return metadata
