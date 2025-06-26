"""Color palette reduction functionality for GIF compression.

This module provides utilities for calculating color reduction parameters
for use with compression engines. Color reduction is applied as part of
single-pass compression along with lossy and frame reduction.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
from PIL import Image
import numpy as np
from collections import Counter

from .config import DEFAULT_COMPRESSION_CONFIG


def validate_color_keep_count(color_count: int) -> None:
    """Validate that the color keep count is supported.
    
    Args:
        color_count: Color count to validate
        
    Raises:
        ValueError: If color count is not supported
    """
    if not isinstance(color_count, int) or color_count <= 0:
        raise ValueError(f"Color count must be a positive integer, got {color_count}")
    
    # Check against configured valid counts
    valid_counts = DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS
    
    if color_count not in valid_counts:
        raise ValueError(
            f"Color count {color_count} not in supported counts: {valid_counts}"
        )


def build_gifsicle_color_args(color_count: int, original_colors: int) -> List[str]:
    """Build gifsicle command arguments for color reduction.
    
    Args:
        color_count: Target number of colors to keep
        original_colors: Original number of colors in the GIF
        
    Returns:
        List of command line arguments for gifsicle
    """
    # No reduction needed if target is >= original or target is max (256)
    if color_count >= original_colors or color_count >= 256:
        return []
    
    # Gifsicle uses --colors argument for palette reduction
    return ["--colors", str(color_count)]


def build_animately_color_args(color_count: int, original_colors: int) -> List[str]:
    """Build animately command arguments for color reduction.
    
    Args:
        color_count: Target number of colors to keep
        original_colors: Original number of colors in the GIF
        
    Returns:
        List of command line arguments for animately
    """
    # No reduction needed if target is >= original or target is max (256)
    if color_count >= original_colors or color_count >= 256:
        return []
    
    # Animately uses --colors argument for palette reduction
    return ["--colors", str(color_count)]


def count_gif_colors(image_path: Path) -> int:
    """Count the number of unique colors in a GIF.
    
    Args:
        image_path: Path to the GIF file
        
    Returns:
        Number of unique colors in the GIF
        
    Raises:
        IOError: If file cannot be read
        ValueError: If file is not a GIF
    """
    if not image_path.exists():
        raise IOError(f"File not found: {image_path}")
    
    try:
        with Image.open(image_path) as img:
            if img.format != 'GIF':
                raise ValueError(f"File is not a GIF: {image_path}")
            
            # Get the first frame for color analysis
            img.seek(0)
            
            if img.mode == 'P':
                # Palette mode - count unique palette entries
                palette = img.getpalette()
                if palette:
                    # Convert palette to RGB tuples and count unique colors
                    rgb_palette = [(palette[i], palette[i+1], palette[i+2]) 
                                   for i in range(0, len(palette), 3)]
                    unique_colors = len(set(rgb_palette))
                    return min(unique_colors, 256)  # GIF max is 256
                else:
                    return 256  # Default if no palette
            else:
                # Convert to palette mode to count colors
                quantized = img.quantize(colors=256)
                palette = quantized.getpalette()
                if palette:
                    rgb_palette = [(palette[i], palette[i+1], palette[i+2]) 
                                   for i in range(0, len(palette), 3)]
                    unique_colors = len(set(rgb_palette))
                    return min(unique_colors, 256)
                else:
                    return 256
                    
    except ValueError:
        # Re-raise ValueError as-is (e.g., "File is not a GIF")
        raise
    except Exception as e:
        raise IOError(f"Error reading GIF {image_path}: {str(e)}")


def get_color_reduction_info(input_path: Path, color_count: int) -> Dict[str, Any]:
    """Get information about color reduction for a given GIF and target count.
    
    Args:
        input_path: Path to input GIF file
        color_count: Target number of colors to keep
        
    Returns:
        Dictionary with color reduction information
        
    Raises:
        IOError: If input file cannot be read
        ValueError: If color_count is invalid
    """
    validate_color_keep_count(color_count)
    
    if not input_path.exists():
        raise IOError(f"Input file not found: {input_path}")
    
    try:
        original_colors = count_gif_colors(input_path)
        
        # Calculate reduction info
        target_colors = min(color_count, original_colors)
        reduction_needed = color_count < original_colors
        
        return {
            "original_colors": original_colors,
            "target_colors": target_colors,
            "color_keep_count": color_count,
            "reduction_needed": reduction_needed,
            "reduction_percent": ((original_colors - target_colors) / original_colors * 100.0) if original_colors > 0 else 0.0,
            "compression_ratio": original_colors / target_colors if target_colors > 0 else 1.0
        }
        
    except Exception as e:
        raise IOError(f"Error analyzing GIF {input_path}: {str(e)}")


def extract_dominant_colors(image: Image.Image, n_colors: int) -> List[Tuple[int, int, int]]:
    """Extract the most dominant colors from an image.
    
    Args:
        image: PIL Image object
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of RGB tuples representing dominant colors
        
    Raises:
        ValueError: If n_colors is not positive
    """
    if n_colors <= 0:
        raise ValueError(f"n_colors must be positive, got {n_colors}")
    
    # Convert image to RGB if needed
    if image.mode != 'RGB':
        rgb_image = image.convert('RGB')
    else:
        rgb_image = image
    
    # Convert to numpy array for faster processing
    img_array = np.array(rgb_image)
    
    # Reshape to get all pixels as RGB tuples
    pixels = img_array.reshape(-1, 3)
    
    # Convert to tuples for counting
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    
    # Count color frequencies
    color_counts = Counter(pixel_tuples)
    
    # Get the most common colors
    dominant_colors = color_counts.most_common(n_colors)
    
    # Return just the color tuples (without counts)
    return [color for color, count in dominant_colors]


def analyze_gif_palette(image_path: Path) -> Dict[str, Any]:
    """Analyze the color palette of a GIF file.
    
    Args:
        image_path: Path to the GIF file
        
    Returns:
        Dictionary with detailed palette analysis
        
    Raises:
        IOError: If file cannot be read
        ValueError: If file is not a GIF
    """
    if not image_path.exists():
        raise IOError(f"File not found: {image_path}")
    
    try:
        with Image.open(image_path) as img:
            if img.format != 'GIF':
                raise ValueError(f"File is not a GIF: {image_path}")
            
            # Analyze first frame
            img.seek(0)
            
            # Get color count
            color_count = count_gif_colors(image_path)
            
            # Get dominant colors (up to 10)
            dominant_colors = extract_dominant_colors(img, min(10, color_count))
            
            # Get palette info if available
            palette_info = {}
            if img.mode == 'P':
                palette = img.getpalette()
                if palette:
                    palette_info = {
                        "mode": "palette",
                        "palette_size": len(palette) // 3,
                        "has_transparency": "transparency" in img.info
                    }
            else:
                palette_info = {
                    "mode": img.mode,
                    "palette_size": None,
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
            
            return {
                "total_colors": color_count,
                "dominant_colors": dominant_colors,
                "palette_info": palette_info,
                "reduction_candidates": {
                    count: color_count > count 
                    for count in DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS
                }
            }
            
    except Exception as e:
        raise IOError(f"Error analyzing GIF palette {image_path}: {str(e)}")


def get_optimal_color_count(image_path: Path, quality_threshold: float = 0.95) -> int:
    """Suggest optimal color count for a GIF based on its content.
    
    Args:
        image_path: Path to the GIF file
        quality_threshold: Quality threshold for color reduction (0.0 to 1.0)
        
    Returns:
        Suggested optimal color count
        
    Raises:
        IOError: If file cannot be read
        ValueError: If quality_threshold is invalid
    """
    if not 0.0 <= quality_threshold <= 1.0:
        raise ValueError(f"Quality threshold must be between 0.0 and 1.0, got {quality_threshold}")
    
    try:
        # Analyze the GIF
        analysis = analyze_gif_palette(image_path)
        original_colors = analysis["total_colors"]
        
        # Find the largest supported color count that provides meaningful reduction
        valid_counts = sorted(DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS, reverse=True)
        
        for count in valid_counts:
            if count < original_colors:
                # Check if this reduction meets quality threshold
                retained_ratio = count / original_colors
                if retained_ratio >= quality_threshold:
                    return count
        
        # If no reduction meets threshold, return the smallest valid count
        return min(valid_counts)
        
    except Exception as e:
        raise IOError(f"Error determining optimal color count for {image_path}: {str(e)}")


def reduce_colors(
    input_path: Path,
    output_path: Path, 
    max_colors: int
) -> Tuple[int, int]:
    """Reduce the color palette of a GIF to the specified maximum colors.
    
    NOTE: This function is deprecated in favor of single-pass compression.
    Use build_gifsicle_color_args() or build_animately_color_args() instead.
    
    Args:
        input_path: Path to input GIF file
        output_path: Path to save color-reduced GIF
        max_colors: Maximum number of colors to keep (1-256)
        
    Returns:
        Tuple of (original_color_count, new_color_count)
        
    Raises:
        ValueError: If max_colors is not between 1 and 256
        IOError: If input file cannot be read or output cannot be written
        DeprecationWarning: This function breaks single-pass compression architecture
    """
    import warnings
    warnings.warn(
        "reduce_colors() is deprecated. Use single-pass compression with "
        "build_gifsicle_color_args() or build_animately_color_args() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not 1 <= max_colors <= 256:
        raise ValueError(f"max_colors must be between 1 and 256, got {max_colors}")
    
    # For compatibility, implement basic color reduction
    # But this should not be used in production due to quality loss from multiple passes
    try:
        original_colors = count_gif_colors(input_path)
        
        with Image.open(input_path) as img:
            # Reduce colors using PIL quantization
            if img.mode != 'P':
                img = img.convert('P', palette=Image.ADAPTIVE, colors=max_colors)
            elif original_colors > max_colors:
                # Re-quantize with fewer colors
                rgb_img = img.convert('RGB')
                img = rgb_img.quantize(colors=max_colors, method=Image.MEDIANCUT)
            
            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format='GIF', optimize=True)
            
            new_colors = count_gif_colors(output_path)
            return (original_colors, new_colors)
            
    except Exception as e:
        raise IOError(f"Error reducing colors: {str(e)}") 