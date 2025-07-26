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

        # 3️⃣ Validate frames with fail-fast approach
        valid_frame_files = _validate_and_prepare_frames(frame_files)

        # 5️⃣ encode with gifski using processed frames
        encode_cmd = [
            gifski,
            "--quality",
            str(quality),
            "-o",
            str(output_path),
        ] + valid_frame_files
        
        # Execute gifski with clean error handling
        try:
            return run_command(encode_cmd, engine="gifski", output_path=output_path)
        except Exception as e:
            # Fail fast with clear error message pointing to root cause
            if "wrong size" in str(e):
                raise RuntimeError(
                    f"gifski failed due to inconsistent frame dimensions. "
                    f"This indicates a problem in the pipeline steps before gifski. "
                    f"Original error: {e}"
                ) from e
            else:
                                 # Re-raise other errors unchanged
                 raise


def _validate_and_prepare_frames(frame_files: list) -> list:
    """Validate PNG frames for gifski processing with fail-fast approach.
    
    Args:
        frame_files: List of PNG frame file paths
        
    Returns:
        List of valid frame files ready for gifski
        
    Raises:
        RuntimeError: If frames are invalid or inconsistent
    """
    import logging
    from PIL import Image
    from collections import Counter
    
    logger = logging.getLogger(__name__)
    
    if not frame_files:
        raise RuntimeError("gifski: No frame files provided")
    
    # Quick validation pass
    valid_frames = []
    frame_dimensions = []
    invalid_count = 0
    
    for frame_file in frame_files:
        try:
            with Image.open(frame_file) as img:
                # Skip obviously invalid frames
                if img.size == (1, 1):
                    invalid_count += 1
                    continue
                    
                valid_frames.append(frame_file)
                frame_dimensions.append(img.size)
                
        except Exception as e:
            logger.warning(f"gifski: Could not read frame {frame_file}: {e}")
            invalid_count += 1
    
    # Fail fast if we don't have enough valid frames
    if not valid_frames:
        raise RuntimeError(
            f"gifski: No valid frames found. All {len(frame_files)} frames were invalid or 1x1 pixels. "
            f"This indicates a fundamental issue in earlier pipeline steps."
        )
    
    if len(valid_frames) < len(frame_files) * 0.5:
        raise RuntimeError(
            f"gifski: Too many invalid frames ({invalid_count} invalid out of {len(frame_files)} total). "
            f"This pipeline combination produces fundamentally broken results and should be eliminated."
        )
    
    # Check dimension consistency
    unique_dimensions = set(frame_dimensions)
    if len(unique_dimensions) > 1:
        dimension_counts = Counter(frame_dimensions)
        most_common_dim, count = dimension_counts.most_common(1)[0]
        
        logger.warning(
            f"gifski: Inconsistent frame dimensions detected. "
            f"Found {len(unique_dimensions)} different sizes. "
            f"Most common: {most_common_dim} ({count}/{len(valid_frames)} frames)"
        )
        
        # If more than 20% of frames have different dimensions, fail fast
        if count < len(valid_frames) * 0.8:
            raise RuntimeError(
                f"gifski: Frame dimension inconsistency is too severe for reliable processing. "
                f"Only {count}/{len(valid_frames)} frames have the most common dimension {most_common_dim}. "
                f"This indicates a fundamental issue in pipeline frame processing that should be fixed earlier."
            )
        
        # Log the issue but proceed if it's minor
        logger.info(f"gifski: Proceeding with minor dimension inconsistencies (mostly {most_common_dim})")
    
    logger.debug(f"gifski: Validated {len(valid_frames)} frames, {invalid_count} invalid frames filtered")
    return valid_frames
