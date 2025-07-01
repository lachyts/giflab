"""Frame reduction functionality for GIF compression.

This module provides utilities for calculating frame selection parameters
for use with compression engines. Frame reduction is applied as part of
single-pass compression along with lossy and color reduction.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image

from .config import DEFAULT_COMPRESSION_CONFIG


def calculate_frame_indices(total_frames: int, keep_ratio: float) -> List[int]:
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


def build_gifsicle_frame_args(keep_ratio: float, total_frames: int) -> List[str]:
    """Build gifsicle command arguments for frame reduction.
    
    Args:
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        total_frames: Total number of frames in the source GIF
        
    Returns:
        List of command line arguments for gifsicle
    """
    if keep_ratio == 1.0:
        return []  # No frame reduction needed
    
    indices = calculate_frame_indices(total_frames, keep_ratio)
    
    # Build gifsicle frame selection arguments
    # Format: --delete "#0-2" --delete "#4-6" etc.
    args = []
    
    # Create ranges of frames to delete
    keep_set = set(indices)
    delete_ranges = []
    start_delete = None
    
    for i in range(total_frames):
        if i not in keep_set:
            if start_delete is None:
                start_delete = i
        else:
            if start_delete is not None:
                # End of delete range
                if start_delete == i - 1:
                    delete_ranges.append(f"#{start_delete}")
                else:
                    delete_ranges.append(f"#{start_delete}-{i-1}")
                start_delete = None
    
    # Handle final range if it extends to the end
    if start_delete is not None:
        if start_delete == total_frames - 1:
            delete_ranges.append(f"#{start_delete}")
        else:
            delete_ranges.append(f"#{start_delete}-{total_frames-1}")
    
    # Add delete arguments
    for delete_range in delete_ranges:
        args.extend(["--delete", delete_range])
    
    return args


def build_animately_frame_args(keep_ratio: float, total_frames: int) -> List[str]:
    """Build animately command arguments for frame reduction.
    
    Args:
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        total_frames: Total number of frames in the source GIF
        
    Returns:
        List of command line arguments for animately
    """
    if keep_ratio == 1.0:
        return []  # No frame reduction needed
    
    # Animately uses a simpler approach - specify frame reduction ratio
    # The exact format depends on animately's CLI, but typically:
    # --frame-skip or --frame-rate reduction
    
    # For now, use a generic approach that might need adjustment
    # based on actual animately CLI documentation
    target_frames = calculate_target_frame_count(total_frames, keep_ratio)
    
    if target_frames == total_frames:
        return []
    
    # Calculate skip factor
    skip_factor = total_frames / target_frames
    
    # Use frame reduction argument (format may need adjustment)
    return ["--frame-reduce", f"{keep_ratio:.2f}"]


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


def get_frame_reduction_info(input_path: Path, keep_ratio: float) -> Dict[str, Any]:
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
        raise IOError(f"Input file not found: {input_path}")
    
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
                raise ValueError(f"Error counting frames in GIF: {e}")
            
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
        raise IOError(f"Error reading GIF {input_path}: {str(e)}") 