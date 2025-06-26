"""Configuration settings for GifLab."""

from typing import List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompressionConfig:
    """Configuration for GIF compression variants."""
    
    # Frame keep ratios to test
    FRAME_KEEP_RATIOS: List[float] = None
    
    # Color palette sizes to test
    COLOR_KEEP_COUNTS: List[int] = None
    
    # Lossy compression levels to test
    LOSSY_LEVELS: List[int] = None
    
    # Supported engines
    ENGINES: List[str] = None
    
    def __post_init__(self) -> None:
        if self.FRAME_KEEP_RATIOS is None:
            self.FRAME_KEEP_RATIOS = [1.00, 0.90, 0.80, 0.70, 0.50]
        
        if self.COLOR_KEEP_COUNTS is None:
            self.COLOR_KEEP_COUNTS = [256, 128, 64]
        
        if self.LOSSY_LEVELS is None:
            self.LOSSY_LEVELS = [0, 40, 120]
        
        if self.ENGINES is None:
            self.ENGINES = ["gifsicle", "animately"]


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    RAW_DIR: Path = Path("data/raw")
    RENDERS_DIR: Path = Path("data/renders") 
    CSV_DIR: Path = Path("data/csv")
    BAD_GIFS_DIR: Path = Path("data/bad_gifs")
    TMP_DIR: Path = Path("data/tmp")
    SEED_DIR: Path = Path("seed")
    LOGS_DIR: Path = Path("logs")


# Default configuration instances
DEFAULT_COMPRESSION_CONFIG = CompressionConfig()
DEFAULT_PATH_CONFIG = PathConfig() 