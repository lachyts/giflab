"""Frame reduction functionality for GIF compression.

This module provides utilities for calculating frame selection parameters
for use with compression engines. Frame reduction is applied as part of
single-pass compression along with lossy and color reduction.

Engine Documentation:
- Gifsicle: https://www.lcdf.org/gifsicle/
- Animately: Internal engine with CLI options (see help output)

Frame Selection Best Practices:
- Gifsicle uses frame selection syntax: #0 #2 #4 (frames to keep)
- Anamely uses --reduce flag with ratio: --reduce 0.5 (keep 50% of frames)
- Both engines support even distribution of kept frames across the animation
"""

from pathlib import Path
from typing import Any

from PIL import Image

from .config import DEFAULT_COMPRESSION_CONFIG


def calculate_frame_indices(total_frames: int, keep_ratio: float) -> list[int]:
    """Calculate which frame indices to keep based on the keep ratio.

    Args:
        total_frames: Total number of frames in the GIF
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)

    Returns:
        List of frame indices to keep (0-based)

    Raises:
        ValueError: If keep_ratio is not between 0.0 and 1.0
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be between 0.0 and 1.0, got {keep_ratio}")

    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    if keep_ratio == 1.0:
        return list(range(total_frames))

    if keep_ratio == 0.0:
        return [0]  # Keep at least the first frame

    # Calculate step size to achieve desired ratio
    # Use evenly distributed sampling
    target_frame_count = max(1, int(total_frames * keep_ratio))

    if target_frame_count >= total_frames:
        return list(range(total_frames))

    # Calculate indices with even distribution
    indices = []
    step = total_frames / target_frame_count

    for i in range(target_frame_count):
        index = int(i * step)
        if index not in indices and index < total_frames:
            indices.append(index)

    # Ensure we have at least one frame and don't exceed total
    if not indices:
        indices = [0]

    return sorted(indices)


def calculate_target_frame_count(total_frames: int, keep_ratio: float) -> int:
    """Calculate the target number of frames after reduction.

    Args:
        total_frames: Total number of frames in the GIF
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)

    Returns:
        Target number of frames after reduction

    Raises:
        ValueError: If keep_ratio is not between 0.0 and 1.0
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be between 0.0 and 1.0, got {keep_ratio}")

    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    if keep_ratio == 1.0:
        return total_frames

    target_count = max(1, int(total_frames * keep_ratio))
    return min(target_count, total_frames)


def build_gifsicle_frame_args(keep_ratio: float, total_frames: int) -> list[str]:
    """Build gifsicle command arguments for frame reduction.

    Uses gifsicle's frame selection syntax to specify which frames to keep.
    This approach is compatible with --optimize and follows gifsicle best practices.

    Reference: https://www.lcdf.org/gifsicle/
    Frame selection syntax: #num, #num1-num2, #num1-, #name

    Example:
        For 12 frames with 0.5 ratio: ['#0', '#2', '#4', '#6', '#8', '#10']
        Command: gifsicle --optimize input.gif #0 #2 #4 #6 #8 #10 --output output.gif

    Args:
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        total_frames: Total number of frames in the source GIF

    Returns:
        List of command line arguments for gifsicle frame selection

    Note:
        - Input file must be specified BEFORE frame selection arguments
        - Frame selection is compatible with --optimize (unlike --delete)
        - Frames are selected with even distribution across the animation
    """
    if keep_ratio == 1.0:
        return []  # No frame reduction needed

    indices = calculate_frame_indices(total_frames, keep_ratio)

    # Build gifsicle frame selection arguments
    # Use frame selection syntax: #0 #2 #4 etc (frames to keep)
    # This is more compatible with --optimize than --delete
    frame_args = []
    for index in indices:
        frame_args.append(f"#{index}")

    return frame_args


def build_animately_frame_args(keep_ratio: float, total_frames: int) -> list[str]:
    """Build animately command arguments for frame reduction.

    Uses animately's --reduce flag to specify the ratio of frames to keep.
    This is simpler than gifsicle's frame selection approach.

    Animately CLI Reference:
    Usage: animately.exe [OPTION...]
      -f, --reduce arg       Reduce frames
      -i, --input arg        Path to input gif file
      -o, --output arg       Path to output gif file
      -l, --lossy arg        Lossy compression level
      -p, --colors arg       Reduce palette colors
      -d, --delay arg        Delay between frames
      -t, --trim-frames arg  Trim frames
      -m, --trim-ms arg      Trim in milliseconds

    Example:
        For 0.5 ratio: ['--reduce', '0.50']
        Command: animately --input input.gif --reduce 0.50 --output output.gif

    Args:
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        total_frames: Total number of frames in the source GIF

    Returns:
        List of command line arguments for animately frame reduction

    Note:
        - Uses decimal ratio format (0.50 for 50%)
        - More straightforward than gifsicle's frame selection
        - Automatically distributes kept frames evenly across animation
    """
    if keep_ratio == 1.0:
        return []  # No frame reduction needed

    # Anamely uses --reduce flag for frame reduction
    # The value should be the ratio of frames to keep
    target_frames = calculate_target_frame_count(total_frames, keep_ratio)

    if target_frames == total_frames:
        return []

    # Use --reduce argument with the keep ratio
    return ["--reduce", f"{keep_ratio:.2f}"]


def validate_frame_keep_ratio(keep_ratio: float) -> None:
    """Validate that the frame keep ratio is supported.

    Args:
        keep_ratio: Frame keep ratio to validate

    Raises:
        ValueError: If frame keep ratio is not supported
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"Frame keep ratio must be between 0.0 and 1.0, got {keep_ratio}")

    # Check against configured valid ratios
    valid_ratios = DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS

    # Allow small floating point differences
    tolerance = 1e-6
    is_valid = any(abs(keep_ratio - valid_ratio) < tolerance
                   for valid_ratio in valid_ratios)

    if not is_valid:
        raise ValueError(
            f"Frame keep ratio {keep_ratio} not in supported ratios: {valid_ratios}"
        )


def get_frame_reduction_info(input_path: Path, keep_ratio: float) -> dict[str, Any]:
    """Get information about frame reduction for a given GIF and keep ratio.

    Args:
        input_path: Path to input GIF file
        keep_ratio: Ratio of frames to keep

    Returns:
        Dictionary with frame reduction information

    Raises:
        IOError: If input file cannot be read
        ValueError: If keep_ratio is invalid
    """
    validate_frame_keep_ratio(keep_ratio)

    if not input_path.exists():
        raise OSError(f"Input file not found: {input_path}")

    try:
        with Image.open(input_path) as img:
            if img.format != 'GIF':
                raise ValueError(f"File is not a GIF: {input_path}")

            # Count frames using safer method
            frame_count = 0
            try:
                # Use PIL's built-in frame counting if available
                if hasattr(img, 'n_frames'):
                    frame_count = img.n_frames
                else:
                    # Fallback to manual counting with better error handling
                    current_frame = 0
                    while True:
                        try:
                            img.seek(current_frame)
                            frame_count = current_frame + 1
                            current_frame += 1
                        except EOFError:
                            break
                        except Exception:
                            # Stop on any other error to prevent infinite loops
                            break

                        # Safety limit to prevent infinite loops with corrupted files
                        if current_frame > 10000:  # Reasonable upper limit
                            raise ValueError(f"GIF appears to have excessive frames (>{current_frame}), possibly corrupted")
            except EOFError:
                pass  # Normal end of frames
            except Exception as e:
                raise ValueError(f"Error counting frames in GIF: {e}") from e

            # Validate frame count
            if frame_count <= 0:
                raise ValueError(f"Invalid frame count: {frame_count}")

            # Calculate reduction info
            target_frames = calculate_target_frame_count(frame_count, keep_ratio)
            frame_indices = calculate_frame_indices(frame_count, keep_ratio)

            return {
                "original_frames": frame_count,
                "target_frames": target_frames,
                "keep_ratio": keep_ratio,
                "frames_kept": len(frame_indices),
                "frame_indices": frame_indices,
                "reduction_percent": (1.0 - keep_ratio) * 100.0
            }

    except ValueError:
        # Re-raise ValueError as-is (e.g., "File is not a GIF")
        raise
    except Exception as e:
        raise OSError(f"Error reading GIF {input_path}: {str(e)}") from e
