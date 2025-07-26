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
        frame_dimensions = {}  # Track dimensions of all frames
        
        for frame_file in frame_files:
            try:
                from PIL import Image
                with Image.open(frame_file) as img:
                    if img.size == (1, 1):
                        invalid_frames.append(frame_file)
                    else:
                        valid_frame_files.append(frame_file)
                        frame_dimensions[frame_file] = img.size
            except Exception:
                # If we can't read the frame, skip it
                invalid_frames.append(frame_file)
        
        if invalid_frames:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"gifski: Filtered out {len(invalid_frames)} invalid 1x1 frames from {len(frame_files)} total frames")
        
        if not valid_frame_files:
            raise RuntimeError(f"gifski: no valid frames found (all {len(frame_files)} frames were 1x1 or invalid)")
        
        if len(valid_frame_files) < len(frame_files) / 2:
            # For elimination testing, log a warning but continue if we have at least 1 valid frame
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"gifski: Pipeline appears broken - {len(invalid_frames)} invalid out of {len(frame_files)} total frames. Proceeding with {len(valid_frame_files)} valid frames.")
            logger.warning("gifski: This pipeline combination (likely Animately frame reduction + FFmpeg enhanced) should be marked as problematic.")

        # 🆕 4️⃣ Check for frame dimension consistency and normalize if needed
        if len(frame_dimensions) > 1:
            unique_dimensions = set(frame_dimensions.values())
            if len(unique_dimensions) > 1:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"gifski: Found {len(unique_dimensions)} different frame dimensions: {unique_dimensions}")
                logger.warning("gifski: Normalizing all frames to consistent dimensions to prevent Gifski failures")
                
                # Find the most common dimension (mode)
                from collections import Counter
                dimension_counts = Counter(frame_dimensions.values())
                target_dimension = dimension_counts.most_common(1)[0][0]
                logger.info(f"gifski: Normalizing all frames to {target_dimension}")
                
                # Resize inconsistent frames to target dimension
                normalized_frame_files = []
                frames_normalized = 0
                frames_failed = 0
                
                for frame_file in valid_frame_files:
                    current_dimension = frame_dimensions[frame_file]
                    if current_dimension != target_dimension:
                        logger.debug(f"gifski: Resizing {frame_file} from {current_dimension} to {target_dimension}")
                        
                        # Create normalized frame file with a more unique name
                        frame_path = Path(frame_file)
                        normalized_frame_file = str(frame_path.parent / f"{frame_path.stem}_norm_{target_dimension[0]}x{target_dimension[1]}.png")
                        
                        try:
                            from PIL import Image
                            with Image.open(frame_file) as img:
                                # Ensure we're working with RGB mode for consistency
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Use high-quality resampling to maintain quality
                                resized_img = img.resize(target_dimension, Image.Resampling.LANCZOS)
                                resized_img.save(normalized_frame_file, format='PNG')
                                
                                # Verify the saved image has correct dimensions
                                with Image.open(normalized_frame_file) as verify_img:
                                    if verify_img.size == target_dimension:
                                        normalized_frame_files.append(normalized_frame_file)
                                        frames_normalized += 1
                                    else:
                                        logger.error(f"gifski: Verification failed for {normalized_frame_file}, expected {target_dimension}, got {verify_img.size}")
                                        normalized_frame_files.append(frame_file)  # Use original as fallback
                                        frames_failed += 1
                                        
                        except Exception as e:
                            logger.warning(f"gifski: Failed to resize {frame_file}: {e}, using original")
                            normalized_frame_files.append(frame_file)
                            frames_failed += 1
                    else:
                        normalized_frame_files.append(frame_file)
                
                # Log normalization results
                logger.info(f"gifski: Frame normalization complete - {frames_normalized} resized, {frames_failed} failed, {len(normalized_frame_files)} total frames")
                
                # Only use normalized frames if we have a reasonable success rate
                if frames_failed == 0 or frames_normalized > frames_failed:
                    valid_frame_files = normalized_frame_files
                    logger.info("gifski: Using normalized frames for encoding")
                else:
                    logger.error("gifski: Too many normalization failures, using original frames (this may cause Gifski to fail)")

        # 5️⃣ encode with gifski using processed frames
        encode_cmd = [
            gifski,
            "--quality",
            str(quality),
            "-o",
            str(output_path),
        ] + valid_frame_files
        
        # Add defensive error handling for Gifski failures
        try:
            return run_command(encode_cmd, engine="gifski", output_path=output_path)
        except Exception as e:
            # If Gifski fails with frame size errors, try one more normalization attempt
            if "wrong size" in str(e) and len(set(frame_dimensions.values())) > 1:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"gifski: Initial encoding failed with frame size error, attempting emergency normalization")
                
                # Emergency normalization - force all frames to the same size
                # Use the first frame's dimension as the target
                if valid_frame_files:
                    try:
                        from PIL import Image
                        with Image.open(valid_frame_files[0]) as first_img:
                            emergency_target = first_img.size
                        
                        logger.info(f"gifski: Emergency normalization to {emergency_target}")
                        
                        emergency_frame_files = []
                        for i, frame_file in enumerate(valid_frame_files):
                            frame_path = Path(frame_file)
                            emergency_frame_file = str(frame_path.parent / f"emergency_{i:04d}.png")
                            
                            try:
                                with Image.open(frame_file) as img:
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    
                                    # Force resize to emergency target
                                    resized_img = img.resize(emergency_target, Image.Resampling.LANCZOS)
                                    resized_img.save(emergency_frame_file, format='PNG')
                                    emergency_frame_files.append(emergency_frame_file)
                            except Exception as resize_error:
                                logger.error(f"gifski: Emergency resize failed for {frame_file}: {resize_error}")
                                # Skip this frame rather than fail completely
                                continue
                        
                        if emergency_frame_files:
                            logger.info(f"gifski: Retrying with {len(emergency_frame_files)} emergency-normalized frames")
                            emergency_cmd = [
                                gifski,
                                "--quality",
                                str(quality),
                                "-o",
                                str(output_path),
                            ] + emergency_frame_files
                            
                            return run_command(emergency_cmd, engine="gifski", output_path=output_path)
                        
                    except Exception as emergency_error:
                        logger.error(f"gifski: Emergency normalization failed: {emergency_error}")
            
            # If all else fails, re-raise the original error
            raise e
