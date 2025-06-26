"""Frame reduction functionality for GIF compression."""

from pathlib import Path
from typing import List, Tuple

from PIL import Image


def reduce_frames(
    input_path: Path, 
    output_path: Path, 
    keep_ratio: float
) -> Tuple[int, int]:
    """Reduce the number of frames in a GIF by keeping every nth frame.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save reduced GIF
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        
    Returns:
        Tuple of (original_frame_count, new_frame_count)
        
    Raises:
        ValueError: If keep_ratio is not between 0.0 and 1.0
        IOError: If input file cannot be read or output cannot be written
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be between 0.0 and 1.0, got {keep_ratio}")
    
    # TODO: Implement frame reduction
    # This will be implemented in Stage 2 (S2)
    raise NotImplementedError("Frame reduction not yet implemented")


def calculate_frame_indices(total_frames: int, keep_ratio: float) -> List[int]:
    """Calculate which frame indices to keep based on the keep ratio.
    
    Args:
        total_frames: Total number of frames in the GIF
        keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        
    Returns:
        List of frame indices to keep
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be between 0.0 and 1.0, got {keep_ratio}")
    
    if keep_ratio == 1.0:
        return list(range(total_frames))
    
    # Calculate step size to achieve desired ratio
    step = 1.0 / keep_ratio
    indices = []
    current = 0.0
    
    while int(current) < total_frames:
        indices.append(int(current))
        current += step
    
    return indices 