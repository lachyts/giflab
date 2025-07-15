"""Tests for giflab.config module."""

from pathlib import Path

from giflab.config import (
    DEFAULT_COMPRESSION_CONFIG,
    DEFAULT_PATH_CONFIG,
    CompressionConfig,
    PathConfig,
)


class TestCompressionConfig:
    """Tests for CompressionConfig class."""

    def test_default_initialization(self):
        """Test that default values are set correctly."""
        config = CompressionConfig()

        assert config.FRAME_KEEP_RATIOS == [1.00, 0.90, 0.80, 0.70, 0.50]
        assert config.COLOR_KEEP_COUNTS == [256, 128, 64, 32, 16, 8]
        assert config.LOSSY_LEVELS == [0, 40, 120]
        assert config.ENGINES == ["gifsicle", "animately"]

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = CompressionConfig(
            FRAME_KEEP_RATIOS=[1.0, 0.5],
            COLOR_KEEP_COUNTS=[128, 64],
            LOSSY_LEVELS=[0, 80],
            ENGINES=["gifsicle"]
        )

        assert config.FRAME_KEEP_RATIOS == [1.0, 0.5]
        assert config.COLOR_KEEP_COUNTS == [128, 64]
        assert config.LOSSY_LEVELS == [0, 80]
        assert config.ENGINES == ["gifsicle"]


class TestPathConfig:
    """Tests for PathConfig class."""

    def test_default_paths(self):
        """Test default path configuration."""
        config = PathConfig()

        assert config.RAW_DIR == Path("data/raw")
        assert config.RENDERS_DIR == Path("data/renders")
        assert config.CSV_DIR == Path("data/csv")
        assert config.BAD_GIFS_DIR == Path("data/bad_gifs")
        assert config.TMP_DIR == Path("data/tmp")
        assert config.SEED_DIR == Path("seed")
        assert config.LOGS_DIR == Path("logs")

    def test_custom_paths(self):
        """Test custom path configuration."""
        config = PathConfig(
            RAW_DIR=Path("/custom/raw"),
            RENDERS_DIR=Path("/custom/renders")
        )

        assert config.RAW_DIR == Path("/custom/raw")
        assert config.RENDERS_DIR == Path("/custom/renders")
        # Other paths should remain default
        assert config.CSV_DIR == Path("data/csv")


def test_default_configurations():
    """Test that default configuration instances work."""
    assert isinstance(DEFAULT_COMPRESSION_CONFIG, CompressionConfig)
    assert isinstance(DEFAULT_PATH_CONFIG, PathConfig)

    # Verify defaults are properly set
    assert DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS is not None
    assert DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS is not None
    assert DEFAULT_COMPRESSION_CONFIG.LOSSY_LEVELS is not None
    assert DEFAULT_COMPRESSION_CONFIG.ENGINES is not None
    
    assert len(DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS) == 5
    assert len(DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS) == 6  # Updated from 3 to 6
    assert len(DEFAULT_COMPRESSION_CONFIG.LOSSY_LEVELS) == 3
    assert len(DEFAULT_COMPRESSION_CONFIG.ENGINES) == 2
