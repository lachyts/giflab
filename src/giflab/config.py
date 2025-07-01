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
class MetricsConfig:
    """Configuration for quality metrics calculation."""
    
    # SSIM calculation modes: fast, optimized, full, comprehensive
    SSIM_MODE: str = "comprehensive"
    
    # Maximum frames to sample for SSIM calculation (balance accuracy/speed)
    SSIM_MAX_FRAMES: int = 30
    
    # Frame alignment: content-based (most robust visual matching)
    # No configuration needed - always uses content-based alignment
    
    # Enable comprehensive multi-metric calculation
    USE_COMPREHENSIVE_METRICS: bool = True
    
    # Enable temporal consistency analysis
    TEMPORAL_CONSISTENCY_ENABLED: bool = True
    
    # Composite quality weights (must sum to 1.0)
    SSIM_WEIGHT: float = 0.30
    MS_SSIM_WEIGHT: float = 0.35
    PSNR_WEIGHT: float = 0.25
    TEMPORAL_WEIGHT: float = 0.10
    
    def __post_init__(self) -> None:
        # Validate weights sum to 1.0 with proper floating point tolerance
        total_weight = (self.SSIM_WEIGHT + self.MS_SSIM_WEIGHT + 
                       self.PSNR_WEIGHT + self.TEMPORAL_WEIGHT)
        tolerance = 1e-6  # More restrictive tolerance for configuration validation
        if abs(total_weight - 1.0) > tolerance:
            raise ValueError(f"Composite quality weights must sum to 1.0 (±{tolerance}), got {total_weight:.10f}")
        
        # Validate individual weights are non-negative
        weights = [self.SSIM_WEIGHT, self.MS_SSIM_WEIGHT, self.PSNR_WEIGHT, self.TEMPORAL_WEIGHT]
        if any(w < 0 for w in weights):
            raise ValueError(f"All weights must be non-negative, got {weights}")
        
        # Note: Frame alignment is always content-based (most robust approach)
        
        # Validate SSIM mode
        valid_modes = {"fast", "optimized", "full", "comprehensive"}
        if self.SSIM_MODE not in valid_modes:
            raise ValueError(f"Invalid SSIM mode: {self.SSIM_MODE}")
        
        # Validate frame limit is reasonable
        if self.SSIM_MAX_FRAMES <= 0 or self.SSIM_MAX_FRAMES > 1000:
            raise ValueError(f"SSIM_MAX_FRAMES must be between 1 and 1000, got {self.SSIM_MAX_FRAMES}")


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
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_PATH_CONFIG = PathConfig() 