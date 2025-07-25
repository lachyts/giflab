from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from giflab.system_tools import discover_tool

from .common import run_command

__all__ = [
    "color_reduce",
    "frame_reduce",
    "lossy_compress",
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
    *,
    fps: float = 15.0,
) -> dict[str, Any]:
    """Palette-based color reduction via two-pass FFmpeg.
    
    Note: fps parameter is deprecated and no longer used to avoid
    pipeline conflicts when used after frame reduction steps.
    """
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
    qv: int = 30,
    fps: float = 15.0,
) -> dict[str, Any]:
    """Single-pass lossy compression using quantiser *qv* and *fps* filter."""
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-lavfi",
        f"fps={fps}",
        "-q:v",
        str(qv),
        str(output_path),
    ]
    return run_command(cmd, engine="ffmpeg", output_path=output_path)
