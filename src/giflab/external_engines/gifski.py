from __future__ import annotations

import glob
import subprocess
import tempfile
from pathlib import Path
from typing import Any

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

        # 🆕 3️⃣ Validate frame sizes and filter out invalid 1x1 frames
        valid_frame_files = []
        invalid_frames = []
        
        for frame_file in frame_files:
            try:
                from PIL import Image
                with Image.open(frame_file) as img:
                    if img.size == (1, 1):
                        invalid_frames.append(frame_file)
                    else:
                        valid_frame_files.append(frame_file)
            except Exception:
                # If we can't read the frame, skip it
                invalid_frames.append(frame_file)
        
        if invalid_frames:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"gifski: Filtered out {len(invalid_frames)} invalid 1x1 frames from {len(frame_files)} total frames")
        
        if not valid_frame_files:
            raise RuntimeError(f"gifski failed: no valid frames found (all {len(frame_files)} frames were 1x1 or invalid)")
        
        if len(valid_frame_files) < len(frame_files) / 2:
            # For elimination testing, log a warning but continue if we have at least 1 valid frame
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"gifski: Pipeline appears broken - {len(invalid_frames)} invalid out of {len(frame_files)} total frames. Proceeding with {len(valid_frame_files)} valid frames.")
            logger.warning("gifski: This pipeline combination (likely Animately frame reduction + FFmpeg enhanced) should be marked as problematic.")

        # 4️⃣ encode with gifski using only valid frames
        encode_cmd = [
            gifski,
            "--quality",
            str(quality),
            "-o",
            str(output_path),
        ] + valid_frame_files
        return run_command(encode_cmd, engine="gifski", output_path=output_path)
