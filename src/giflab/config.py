"""Configuration settings for GifLab."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class CompressionConfig:
    """Configuration for GIF compression variants."""

    # Frame keep ratios to test
    FRAME_KEEP_RATIOS: list[float] | None = None

    # Color palette sizes to test
    COLOR_KEEP_COUNTS: list[int] | None = None

    # Lossy compression levels to test
    LOSSY_LEVELS: list[int] | None = None

    # Supported engines
    ENGINES: list[str] | None = None

    def __post_init__(self) -> None:
        if self.FRAME_KEEP_RATIOS is None:
            self.FRAME_KEEP_RATIOS = [1.00, 0.90, 0.80, 0.70, 0.50]

        if self.COLOR_KEEP_COUNTS is None:
            self.COLOR_KEEP_COUNTS = [256, 128, 64, 32, 16, 8]

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

    # Enable raw (un-normalised) metric values alongside normalised ones
    RAW_METRICS: bool = False

    # Enable positional sampling (first, middle, last frame analysis)
    ENABLE_POSITIONAL_SAMPLING: bool = True
    POSITIONAL_METRICS: list[str] = None  # Will be set in __post_init__

    # Composite quality weights (must sum to 1.0)
    SSIM_WEIGHT: float = 0.30
    MS_SSIM_WEIGHT: float = 0.35
    PSNR_WEIGHT: float = 0.25
    TEMPORAL_WEIGHT: float = 0.10

    # PSNR normalisation upper bound (dB) – values ≥ this map to 1.0
    PSNR_MAX_DB: float = 50.0

    # Edge-similarity Canny thresholds
    EDGE_CANNY_THRESHOLD1: int = 50
    EDGE_CANNY_THRESHOLD2: int = 150

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

        # Validate PSNR max dB
        if self.PSNR_MAX_DB <= 0:
            raise ValueError("PSNR_MAX_DB must be positive")

        # Validate Canny thresholds
        if self.EDGE_CANNY_THRESHOLD1 <= 0 or self.EDGE_CANNY_THRESHOLD2 <= 0:
            raise ValueError("Canny thresholds must be positive")
        if self.EDGE_CANNY_THRESHOLD1 >= self.EDGE_CANNY_THRESHOLD2:
            raise ValueError("EDGE_CANNY_THRESHOLD1 must be < EDGE_CANNY_THRESHOLD2")

        # Set default positional metrics if not provided
        if self.POSITIONAL_METRICS is None:
            self.POSITIONAL_METRICS = ["ssim", "mse", "fsim", "chist"]


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
    
    def __post_init__(self) -> None:
        """Validate path configuration after initialization."""
        # Import here to avoid circular imports
        from .validation import validate_config_paths, ValidationError
        
        try:
            # Convert dataclass to dict for validation
            config_dict = {
                'RAW_DIR': self.RAW_DIR,
                'RENDERS_DIR': self.RENDERS_DIR,
                'CSV_DIR': self.CSV_DIR,
                'BAD_GIFS_DIR': self.BAD_GIFS_DIR,
                'TMP_DIR': self.TMP_DIR,
                'SEED_DIR': self.SEED_DIR,
                'LOGS_DIR': self.LOGS_DIR,
            }
            
            # Validate paths for security
            validate_config_paths(config_dict)
            
        except ValidationError as e:
            raise ValueError(f"Invalid path configuration: {e}") from e


@dataclass
class EngineConfig:
    """Configuration for compression engine paths."""

    # Path to the gifsicle executable.
    # On macOS/Linux, "gifsicle" should work if installed via package manager
    # On Windows, you might need to provide the full path to the .exe
    # e.g., "C:/Program Files/gifsicle/gifsicle.exe"
    GIFSICLE_PATH: str = "gifsicle"

    # Path to the animately executable.
    # Set this to the location of your animately binary.
    ANIMATELY_PATH: str = "animately"


# Default configuration instances
DEFAULT_COMPRESSION_CONFIG = CompressionConfig()
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_ENGINE_CONFIG = EngineConfig()
