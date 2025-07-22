from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Any
import glob

from giflab.system_tools import discover_tool

from .common import run_command

__all__ = ["lossy_compress"]


def _magick_binary() -> str:
    info = discover_tool("imagemagick")
    info.require()
    return info.name


def _gifski_binary() -> str:
    info = discover_tool("gifski")
    info.require()
    return info.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lossy_compress(
    input_path: Path,
    output_path: Path,
    *,
    quality: int = 60,
) -> dict[str, Any]:
    """Lossy compression via gifski (requires PNG frames)."""
    if quality < 0 or quality > 100:
        raise ValueError("quality must be in 0–100 range")

    magick = _magick_binary()
    gifski = _gifski_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = str(Path(tmpdir) / "frame_%04d.png")

        # 1️⃣ split GIF into PNG frames
        split_cmd = [magick, str(input_path), frame_pattern]
        # Use a dummy output path since ImageMagick creates multiple files
        dummy_output = Path(tmpdir) / "dummy"
        run_command(split_cmd, engine="imagemagick", output_path=dummy_output)

        # 2️⃣ find the frames that were just created
        frame_files = sorted(glob.glob(f"{tmpdir}/frame_*.png"))
        if not frame_files:
            raise RuntimeError(f"gifski failed: no PNG frames found in {tmpdir}")

        # 3️⃣ encode with gifski
        encode_cmd = [
            gifski,
            "--quality",
            str(quality),
            "-o",
            str(output_path),
        ] + frame_files
        return run_command(encode_cmd, engine="gifski", output_path=output_path) 