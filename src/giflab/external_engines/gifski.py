from __future__ import annotations

import glob
import logging
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from ..system_tools import discover_tool

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
    png_sequence_dir: Path | None = None,
) -> dict[str, Any]:
    """Lossy compression via gifski (requires PNG frames).
    
    Args:
        input_path: Input GIF file (used if png_sequence_dir is None)
        output_path: Output GIF file path
        quality: Quality level 0-100 (higher = better quality)
        png_sequence_dir: Optional directory containing PNG sequence from previous pipeline step
    """
    if quality < 0 or quality > 100:
        raise ValueError("quality must be in 0–100 range")

    gifski = _gifski_binary()

    if png_sequence_dir is not None:
        # Use PNG sequence directly from previous pipeline step
        frame_files = sorted(glob.glob(str(png_sequence_dir / "frame_*.png")))
        if not frame_files:
            raise RuntimeError(f"gifski failed: no PNG frames found in {png_sequence_dir}")
        
        # 3️⃣ Runtime frame count guard - count actual PNG files
        if len(frame_files) < 2:
            raise RuntimeError(
                f"gifski: Found only {len(frame_files)} PNG frame file(s), but gifski requires at least 2 frames. "
                f"This occurs when frame reduction reduces the input to too few frames. "
                f"Consider using a different compression tool for single-frame outputs."
            )
        
        # 3️⃣ Normalize frame dimensions before validation
        normalized_frame_files = _normalize_frame_dimensions(frame_files)
        
        # 3️⃣ Validate frames with fail-fast approach
        valid_frame_files = _validate_and_prepare_frames(normalized_frame_files)

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
    else:
        # Original behavior: extract PNG frames from GIF input
        magick = _magick_binary()
        
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

            # 3️⃣ Runtime frame count guard - count actual PNG files
            if len(frame_files) < 2:
                raise RuntimeError(
                    f"gifski: Found only {len(frame_files)} PNG frame file(s), but gifski requires at least 2 frames. "
                    f"This occurs when frame reduction reduces the input to too few frames. "
                    f"Consider using a different compression tool for single-frame outputs."
                )

            # 3️⃣ Normalize frame dimensions before validation
            normalized_frame_files = _normalize_frame_dimensions(frame_files)
            
            # 3️⃣ Validate frames with fail-fast approach
            valid_frame_files = _validate_and_prepare_frames(normalized_frame_files)

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


def _normalize_frame_dimensions(frame_files: list) -> list:
    """Normalize frame dimensions to prevent gifski dimension inconsistency errors.
    
    All frames will be padded to the most common dimension using center alignment
    to avoid quality degradation from content shifting.
    
    Args:
        frame_files: List of PNG frame file paths
        
    Returns:
        List of normalized frame files (may be new files if padding was needed)
        
    Raises:
        RuntimeError: If frame normalization fails
    """
    import tempfile
    from collections import Counter
    from PIL import ImageOps
    
    logger = logging.getLogger(__name__)
    
    if not frame_files:
        return frame_files
    
    # First pass: collect all frame dimensions
    frame_dimensions = []
    for frame_file in frame_files:
        try:
            with Image.open(frame_file) as img:
                frame_dimensions.append(img.size)
        except Exception as e:
            logger.warning(f"Could not read frame {frame_file} for dimension analysis: {e}")
            # Keep original file if we can't read it
            frame_dimensions.append(None)
    
    # Check if normalization is needed
    valid_dimensions = [d for d in frame_dimensions if d is not None]
    if not valid_dimensions:
        return frame_files
    
    unique_dimensions = set(valid_dimensions)
    if len(unique_dimensions) <= 1:
        # All frames have same dimensions, no normalization needed
        return frame_files
    
    # Find most common dimension
    dimension_counts = Counter(valid_dimensions)
    target_dimension, count = dimension_counts.most_common(1)[0]
    
    logger.info(f"gifski: Normalizing frame dimensions to {target_dimension} "
               f"(most common size, {count}/{len(valid_dimensions)} frames)")
    
    # Second pass: normalize frames that need it
    normalized_files = []
    temp_dir = None
    
    for i, (frame_file, dimension) in enumerate(zip(frame_files, frame_dimensions)):
        if dimension == target_dimension or dimension is None:
            # Frame already has target dimension or couldn't be read
            normalized_files.append(frame_file)
        else:
            # Need to normalize this frame
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="gifski_normalized_")
            
            try:
                with Image.open(frame_file) as img:
                    # Use center-aligned padding to avoid content shifting
                    normalized_img = ImageOps.pad(img, target_dimension, method=Image.LANCZOS, centering=(0.5, 0.5))
                    
                    # Save normalized frame
                    normalized_path = Path(temp_dir) / f"normalized_{i:04d}.png"
                    normalized_img.save(normalized_path, format='PNG')
                    normalized_files.append(str(normalized_path))
                    
            except Exception as e:
                logger.warning(f"Failed to normalize frame {frame_file}: {e}")
                # Keep original if normalization fails
                normalized_files.append(frame_file)
    
    if temp_dir:
        logger.debug(f"gifski: Created {len([f for f in normalized_files if temp_dir in str(f)])} normalized frames in {temp_dir}")
    
    return normalized_files


def _validate_and_prepare_frames(
    frame_files: list, 
    min_valid_frame_ratio: float = 0.8,
    max_dimension_inconsistency_ratio: float = 0.2
) -> list:
    """Validate PNG frames for gifski processing with configurable validation.
    
    Args:
        frame_files: List of PNG frame file paths
        min_valid_frame_ratio: Minimum ratio of valid frames required (default: 0.8 = 80%)
        max_dimension_inconsistency_ratio: Maximum ratio of frames with inconsistent dimensions (default: 0.2 = 20%)
        
    Returns:
        List of valid frame files ready for gifski
        
    Raises:
        RuntimeError: If frames are invalid or inconsistent beyond configured thresholds
    """
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
    
    # CRITICAL: gifski requires at least 2 frames to create an animation
    if len(valid_frames) < 2:
        raise RuntimeError(
            f"gifski: Only {len(valid_frames)} valid frame(s) found, but gifski requires at least 2 frames to create an animation. "
            f"This usually occurs when frame reduction reduces the input to a single frame. "
            f"Consider using a different compression tool for single-frame outputs."
        )
    
    # Check if we have enough valid frames using configurable threshold
    required_valid_frames = len(frame_files) * min_valid_frame_ratio
    if len(valid_frames) < required_valid_frames:
        raise RuntimeError(
            f"gifski: Too many invalid frames ({invalid_count} invalid out of {len(frame_files)} total). "
            f"Only {len(valid_frames)} valid frames found, but {required_valid_frames:.1f} required "
            f"(min_valid_frame_ratio={min_valid_frame_ratio}). "
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
        
        # Check dimension consistency using configurable threshold
        required_consistent_frames = len(valid_frames) * (1.0 - max_dimension_inconsistency_ratio)
        if count < required_consistent_frames:
            raise RuntimeError(
                f"gifski: Frame dimension inconsistency is too severe for reliable processing. "
                f"Only {count}/{len(valid_frames)} frames have the most common dimension {most_common_dim}, "
                f"but {required_consistent_frames:.1f} required "
                f"(max_dimension_inconsistency_ratio={max_dimension_inconsistency_ratio}). "
                f"This indicates a fundamental issue in pipeline frame processing that should be fixed earlier."
            )
        
        # Log the issue but proceed if it's minor
        logger.info(f"gifski: Proceeding with minor dimension inconsistencies (mostly {most_common_dim})")
    
    logger.debug(f"gifski: Validated {len(valid_frames)} frames, {invalid_count} invalid frames filtered")
    return valid_frames
