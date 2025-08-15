"""Frame reduction functionality for GIF compression.

This module provides utilities for calculating frame selection parameters
for use with compression engines. Frame reduction is applied as part of
single-pass compression along with lossy and color reduction.

Engine Documentation:
- Gifsicle: https://www.lcdf.org/gifsicle/
- Animately: Internal engine with CLI options (see help output)

Frame Selection Best Practices:
- Gifsicle uses frame selection syntax: #0 #2 #4 (frames to keep)
- Animately uses --reduce flag with ratio: --reduce 0.5 (keep 50% of frames)
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


def build_gifsicle_timing_args(
    original_delays: list[int], frame_indices: list[int], loop_count: int | None
) -> list[str]:
    """Build gifsicle command arguments to preserve timing and looping.

    Args:
        original_delays: List of original frame delays in milliseconds
        frame_indices: List of frame indices that will be kept (0-based)
        loop_count: Original loop count (0 = infinite, None = not set)

    Returns:
        List of command line arguments for gifsicle timing preservation

    Example:
        For adjusted delays [200, 300] and infinite loop:
        Returns: ['--delay', '20', '--loopcount', '0']
    """
    timing_args = []
    
    # Calculate adjusted delays for remaining frames
    adjusted_delays = calculate_adjusted_delays(original_delays, frame_indices)
    
    # Use uniform delay based on average of adjusted delays (simpler approach)
    if adjusted_delays:
        avg_delay = sum(adjusted_delays) / len(adjusted_delays)
        # Convert milliseconds to centiseconds (gifsicle units)
        # Gifsicle uses centiseconds (1/100 second), minimum 2 (20ms)
        delay_cs = max(2, int(avg_delay // 10))
        timing_args.extend(["--delay", str(delay_cs)])
    
    # Add loop count preservation
    if loop_count is not None:
        timing_args.extend([f"--loopcount={loop_count}"])
    else:
        # Default to infinite loop if not specified
        timing_args.extend(["--loopcount=0"])
    
    return timing_args


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

    # Animately uses --reduce flag for frame reduction
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
        raise ValueError(
            f"Frame keep ratio must be between 0.0 and 1.0, got {keep_ratio}"
        )

    # Check against configured valid ratios
    valid_ratios = DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS

    # Allow small floating point differences
    tolerance = 1e-6
    is_valid = any(
        abs(keep_ratio - valid_ratio) < tolerance for valid_ratio in (valid_ratios or [])
    )

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
            if img.format != "GIF":
                raise ValueError(f"File is not a GIF: {input_path}")

            # Count frames using safer method
            frame_count = 0
            try:
                # Use PIL's built-in frame counting if available
                if hasattr(img, "n_frames"):
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
                            raise ValueError(
                                f"GIF appears to have excessive frames (>{current_frame}), possibly corrupted"
                            )
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
                "reduction_percent": (1.0 - keep_ratio) * 100.0,
            }

    except ValueError:
        # Re-raise ValueError as-is (e.g., "File is not a GIF")
        raise
    except Exception as e:
        raise OSError(f"Error reading GIF {input_path}: {str(e)}") from e


def extract_gif_timing_info(input_path: Path) -> dict[str, Any]:
    """Extract frame timing and loop information from a GIF file.

    Args:
        input_path: Path to input GIF file

    Returns:
        Dictionary containing:
        - frame_delays: List of frame delays in milliseconds
        - loop_count: Number of loops (0 = infinite, None = not set)
        - total_frames: Total number of frames
        - average_delay: Average frame delay

    Raises:
        IOError: If input file cannot be read
        ValueError: If file is not a valid GIF
    """
    if not input_path.exists():
        raise OSError(f"Input file not found: {input_path}")

    try:
        with Image.open(input_path) as img:
            if img.format != "GIF":
                raise ValueError(f"File is not a GIF: {input_path}")

            # Extract loop count (0 = infinite, None = not set)
            loop_count = img.info.get("loop", None)
            
            # Count frames and extract delays
            frame_delays = []
            frame_count = 0
            
            try:
                # Use PIL's built-in frame counting if available
                if hasattr(img, "n_frames"):
                    frame_count = img.n_frames
                else:
                    # Fallback to manual counting
                    current_frame = 0
                    while True:
                        try:
                            img.seek(current_frame)
                            frame_count = current_frame + 1
                            current_frame += 1
                        except EOFError:
                            break
                        except Exception:
                            break
                        
                        # Safety limit to prevent infinite loops
                        if current_frame > 10000:
                            raise ValueError(f"GIF appears to have excessive frames (>{current_frame})")

                # Extract delays for each frame
                for i in range(frame_count):
                    try:
                        img.seek(i)
                        duration = img.info.get("duration", 100)  # Default 100ms if not specified
                        # GIF standard minimum is 10ms, but browsers often enforce 20ms
                        frame_delays.append(max(20, duration))
                    except (EOFError, Exception):
                        frame_delays.append(100)  # Fallback delay
                        
            except EOFError:
                pass  # Normal end of frames
            except Exception as e:
                raise ValueError(f"Error extracting frame information: {e}") from e

            # Validate results
            if frame_count <= 0:
                raise ValueError(f"Invalid frame count: {frame_count}")

            if not frame_delays:
                frame_delays = [100] * frame_count

            # Calculate average delay
            average_delay = sum(frame_delays) / len(frame_delays) if frame_delays else 100

            return {
                "frame_delays": frame_delays,
                "loop_count": loop_count,
                "total_frames": frame_count,
                "average_delay": average_delay,
            }

    except ValueError:
        # Re-raise ValueError as-is (e.g., "File is not a GIF")
        raise
    except Exception as e:
        raise OSError(f"Error reading GIF timing info from {input_path}: {str(e)}") from e


def calculate_adjusted_delays(original_delays: list[int], frame_indices: list[int]) -> list[int]:
    """Calculate adjusted frame delays when frames are removed.
    
    When frames are removed, the timing of remaining frames needs to be adjusted
    to maintain the original animation speed. This function calculates the new
    delays for the kept frames.

    Args:
        original_delays: List of original frame delays in milliseconds
        frame_indices: List of frame indices that will be kept (0-based)

    Returns:
        List of adjusted delays for the kept frames

    Example:
        Original: [100, 100, 100, 100] (4 frames, 100ms each)
        Keep indices: [0, 2] (keep frames 0 and 2)
        Result: [200, 200] (each remaining frame gets 2x delay to maintain speed)
    """
    if not original_delays or not frame_indices:
        return []

    # Handle edge case: single frame
    if len(frame_indices) == 1:
        return [sum(original_delays)]  # Single frame gets total duration

    adjusted_delays = []
    
    for i, frame_idx in enumerate(frame_indices):
        if i == len(frame_indices) - 1:
            # Last frame: include all remaining delays from current frame to end
            remaining_delays = original_delays[frame_idx:]
            adjusted_delay = sum(remaining_delays)
        else:
            # Calculate delay as sum of all delays between this frame and next kept frame
            next_frame_idx = frame_indices[i + 1]
            frame_range_delays = original_delays[frame_idx:next_frame_idx]
            adjusted_delay = sum(frame_range_delays)
        
        # Ensure minimum delay (GIF standard minimum with browser compatibility)
        adjusted_delay = max(20, adjusted_delay)
        adjusted_delays.append(adjusted_delay)

    return adjusted_delays
