"""Metadata extraction and hashing for GIF files."""

import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from PIL import Image


@dataclass
class GifMetadata:
    """Metadata extracted from a GIF file."""
    
    gif_sha: str
    orig_filename: str
    orig_kilobytes: float
    orig_width: int
    orig_height: int
    orig_frames: int
    orig_fps: float
    orig_n_colors: int
    entropy: Optional[float] = None


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_gif_metadata(file_path: Path) -> GifMetadata:
    """Extract metadata from a GIF file.
    
    Args:
        file_path: Path to the GIF file
        
    Returns:
        GifMetadata object with extracted information
        
    Raises:
        ValueError: If file is not a valid GIF
        IOError: If file cannot be read
    """
    # TODO: Implement GIF metadata extraction
    # This will be implemented in Stage 1 (S1)
    raise NotImplementedError("GIF metadata extraction not yet implemented")


def calculate_entropy(image: Image.Image) -> float:
    """Calculate entropy of an image for complexity measurement.
    
    Args:
        image: PIL Image object
        
    Returns:
        Entropy value as float
    """
    # TODO: Implement entropy calculation
    # This will be implemented in Stage 1 (S1)
    raise NotImplementedError("Entropy calculation not yet implemented") 