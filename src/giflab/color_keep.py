"""Color palette reduction functionality for GIF compression."""

from pathlib import Path
from typing import Tuple, List

from PIL import Image


def reduce_colors(
    input_path: Path,
    output_path: Path, 
    max_colors: int
) -> Tuple[int, int]:
    """Reduce the color palette of a GIF to the specified maximum colors.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save color-reduced GIF
        max_colors: Maximum number of colors to keep (1-256)
        
    Returns:
        Tuple of (original_color_count, new_color_count)
        
    Raises:
        ValueError: If max_colors is not between 1 and 256
        IOError: If input file cannot be read or output cannot be written
    """
    if not 1 <= max_colors <= 256:
        raise ValueError(f"max_colors must be between 1 and 256, got {max_colors}")
    
    # TODO: Implement color reduction
    # This will be implemented in Stage 3 (S3)
    raise NotImplementedError("Color reduction not yet implemented")


def count_gif_colors(image_path: Path) -> int:
    """Count the number of unique colors in a GIF.
    
    Args:
        image_path: Path to the GIF file
        
    Returns:
        Number of unique colors in the GIF
        
    Raises:
        IOError: If file cannot be read
    """
    # TODO: Implement color counting
    # This will be implemented in Stage 3 (S3)
    raise NotImplementedError("Color counting not yet implemented")


def extract_dominant_colors(image: Image.Image, n_colors: int) -> List[Tuple[int, int, int]]:
    """Extract the most dominant colors from an image.
    
    Args:
        image: PIL Image object
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of RGB tuples representing dominant colors
    """
    # TODO: Implement dominant color extraction
    # This will be implemented in Stage 3 (S3)
    raise NotImplementedError("Dominant color extraction not yet implemented") 