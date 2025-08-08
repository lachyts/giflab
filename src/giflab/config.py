"""Configuration settings for GifLab."""

from dataclasses import dataclass
from pathlib import Path


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
    # Legacy 4-metric weights (for backward compatibility)
    SSIM_WEIGHT: float = 0.30
    MS_SSIM_WEIGHT: float = 0.35
    PSNR_WEIGHT: float = 0.25
    TEMPORAL_WEIGHT: float = 0.10
    
    # Enhanced 11-metric composite quality weights (comprehensive approach)
    # Core structural similarity metrics (40% total)
    ENHANCED_SSIM_WEIGHT: float = 0.18
    ENHANCED_MS_SSIM_WEIGHT: float = 0.22
    
    # Signal quality metrics (25% total)
    ENHANCED_PSNR_WEIGHT: float = 0.15
    ENHANCED_MSE_WEIGHT: float = 0.10
    
    # Advanced structural metrics (20% total)
    ENHANCED_FSIM_WEIGHT: float = 0.08
    ENHANCED_EDGE_WEIGHT: float = 0.07
    ENHANCED_GMSD_WEIGHT: float = 0.05
    
    # Perceptual quality metrics (10% total)
    ENHANCED_CHIST_WEIGHT: float = 0.04
    ENHANCED_SHARPNESS_WEIGHT: float = 0.03
    ENHANCED_TEXTURE_WEIGHT: float = 0.03
    
    # Temporal consistency (5% total)
    ENHANCED_TEMPORAL_WEIGHT: float = 0.05
    
    # Enable enhanced composite quality calculation
    USE_ENHANCED_COMPOSITE_QUALITY: bool = True

    # PSNR normalisation upper bound (dB) – values ≥ this map to 1.0
    PSNR_MAX_DB: float = 50.0

    # Edge-similarity Canny thresholds
    EDGE_CANNY_THRESHOLD1: int = 50
    EDGE_CANNY_THRESHOLD2: int = 150

    def __post_init__(self) -> None:
        # Validate legacy weights sum to 1.0 with proper floating point tolerance
        legacy_total = (self.SSIM_WEIGHT + self.MS_SSIM_WEIGHT +
                       self.PSNR_WEIGHT + self.TEMPORAL_WEIGHT)
        tolerance = 1e-6  # More restrictive tolerance for configuration validation
        if abs(legacy_total - 1.0) > tolerance:
            raise ValueError(f"Legacy composite quality weights must sum to 1.0 (±{tolerance}), got {legacy_total:.10f}")

        # Validate enhanced weights sum to 1.0
        enhanced_total = (self.ENHANCED_SSIM_WEIGHT + self.ENHANCED_MS_SSIM_WEIGHT +
                         self.ENHANCED_PSNR_WEIGHT + self.ENHANCED_MSE_WEIGHT +
                         self.ENHANCED_FSIM_WEIGHT + self.ENHANCED_EDGE_WEIGHT +
                         self.ENHANCED_GMSD_WEIGHT + self.ENHANCED_CHIST_WEIGHT +
                         self.ENHANCED_SHARPNESS_WEIGHT + self.ENHANCED_TEXTURE_WEIGHT +
                         self.ENHANCED_TEMPORAL_WEIGHT)
        if abs(enhanced_total - 1.0) > tolerance:
            raise ValueError(f"Enhanced composite quality weights must sum to 1.0 (±{tolerance}), got {enhanced_total:.10f}")

        # Validate all weights are non-negative
        legacy_weights = [self.SSIM_WEIGHT, self.MS_SSIM_WEIGHT, self.PSNR_WEIGHT, self.TEMPORAL_WEIGHT]
        enhanced_weights = [self.ENHANCED_SSIM_WEIGHT, self.ENHANCED_MS_SSIM_WEIGHT,
                          self.ENHANCED_PSNR_WEIGHT, self.ENHANCED_MSE_WEIGHT,
                          self.ENHANCED_FSIM_WEIGHT, self.ENHANCED_EDGE_WEIGHT,
                          self.ENHANCED_GMSD_WEIGHT, self.ENHANCED_CHIST_WEIGHT,
                          self.ENHANCED_SHARPNESS_WEIGHT, self.ENHANCED_TEXTURE_WEIGHT,
                          self.ENHANCED_TEMPORAL_WEIGHT]
        all_weights = legacy_weights + enhanced_weights
        if any(w < 0 for w in all_weights):
            raise ValueError(f"All weights must be non-negative, got negatives in {all_weights}")

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
        from .validation import ValidationError, validate_config_paths

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
    """Configuration for compression engine paths with environment variable overrides."""

    # Path to the gifsicle executable.
    # On macOS/Linux, "gifsicle" should work if installed via package manager
    # On Windows, you might need to provide the full path to the .exe
    # e.g., "C:/Program Files/gifsicle/gifsicle.exe"
    # Override with: GIFLAB_GIFSICLE_PATH
    GIFSICLE_PATH: str = "gifsicle"

    # Path to the animately executable.
    # Set this to the location of your animately binary.
    # Override with: GIFLAB_ANIMATELY_PATH
    ANIMATELY_PATH: str = "animately"

    # Path to ImageMagick executable (magick or convert).
    # On most systems, "magick" should work (ImageMagick 7.x)
    # On older systems or specific setups, may need "convert"
    # Override with: GIFLAB_IMAGEMAGICK_PATH
    IMAGEMAGICK_PATH: str = "magick"

    # Path to FFmpeg executable.
    # Usually "ffmpeg" works if installed via package manager
    # Override with: GIFLAB_FFMPEG_PATH
    FFMPEG_PATH: str = "ffmpeg"

    # Path to FFprobe executable (companion to FFmpeg).
    # Usually "ffprobe" works if FFmpeg is properly installed
    # Override with: GIFLAB_FFPROBE_PATH
    FFPROBE_PATH: str = "ffprobe"

    # Path to gifski executable.
    # Install from: https://gif.ski/
    # Override with: GIFLAB_GIFSKI_PATH
    GIFSKI_PATH: str = "gifski"

    def __post_init__(self) -> None:
        """Apply environment variable overrides after initialization."""
        import os
        
        # Environment variable mapping
        env_overrides = {
            'GIFSICLE_PATH': 'GIFLAB_GIFSICLE_PATH',
            'ANIMATELY_PATH': 'GIFLAB_ANIMATELY_PATH',
            'IMAGEMAGICK_PATH': 'GIFLAB_IMAGEMAGICK_PATH',
            'FFMPEG_PATH': 'GIFLAB_FFMPEG_PATH',
            'FFPROBE_PATH': 'GIFLAB_FFPROBE_PATH',
            'GIFSKI_PATH': 'GIFLAB_GIFSKI_PATH',
        }
        
        # Apply overrides from environment variables
        for attr_name, env_var_name in env_overrides.items():
            env_value = os.getenv(env_var_name)
            if env_value:
                setattr(self, attr_name, env_value)


# Default configuration instances
DEFAULT_COMPRESSION_CONFIG = CompressionConfig()
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_ENGINE_CONFIG = EngineConfig()
