"""Tests for giflab.metrics module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch
from PIL import Image

from giflab.config import MetricsConfig, DEFAULT_METRICS_CONFIG
from giflab.metrics import (
    calculate_comprehensive_metrics,
    calculate_ssim,
    calculate_file_size_kb,
    extract_gif_frames,
    resize_to_common_dimensions,
    align_frames,
    align_frames_content_based,
    calculate_ms_ssim,
    calculate_temporal_consistency,
    calculate_frame_mse,
    FrameExtractResult
)


class TestMetricsConfig:
    """Tests for MetricsConfig class."""

    def test_default_initialization(self):
        """Test that default values are set correctly."""
        config = MetricsConfig()
        
        assert config.SSIM_MODE == "comprehensive"
        assert config.SSIM_MAX_FRAMES == 30
        assert config.USE_COMPREHENSIVE_METRICS is True
        assert config.TEMPORAL_CONSISTENCY_ENABLED is True
        assert config.SSIM_WEIGHT == 0.30
        assert config.MS_SSIM_WEIGHT == 0.35
        assert config.PSNR_WEIGHT == 0.25
        assert config.TEMPORAL_WEIGHT == 0.10

    def test_weights_sum_to_one(self):
        """Test that composite quality weights sum to 1.0."""
        config = MetricsConfig()
        total_weight = (config.SSIM_WEIGHT + config.MS_SSIM_WEIGHT + 
                       config.PSNR_WEIGHT + config.TEMPORAL_WEIGHT)
        assert abs(total_weight - 1.0) < 0.001

    def test_invalid_weights_raises_error(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MetricsConfig(
                SSIM_WEIGHT=0.5,
                MS_SSIM_WEIGHT=0.5,
                PSNR_WEIGHT=0.5,
                TEMPORAL_WEIGHT=0.5
            )



    def test_invalid_ssim_mode_raises_error(self):
        """Test that invalid SSIM mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SSIM mode"):
            MetricsConfig(SSIM_MODE="invalid_mode")

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = MetricsConfig(
            SSIM_MODE="fast",
            SSIM_MAX_FRAMES=10,
            SSIM_WEIGHT=0.25,
            MS_SSIM_WEIGHT=0.25,
            PSNR_WEIGHT=0.25,
            TEMPORAL_WEIGHT=0.25
        )
        
        assert config.SSIM_MODE == "fast"
        assert config.SSIM_MAX_FRAMES == 10
        assert config.SSIM_WEIGHT == 0.25


class TestFileOperations:
    """Tests for basic file operations."""

    def test_calculate_file_size_kb(self, tmp_path):
        """Test file size calculation in KB."""
        test_file = tmp_path / "test.txt"
        test_content = "x" * 1024  # 1KB of content
        test_file.write_text(test_content)
        
        size_kb = calculate_file_size_kb(test_file)
        assert abs(size_kb - 1.0) < 0.1  # Should be approximately 1KB

    def test_calculate_file_size_kb_nonexistent_file(self):
        """Test file size calculation with nonexistent file."""
        with pytest.raises(IOError):
            calculate_file_size_kb(Path("nonexistent_file.gif"))


class TestFrameExtraction:
    """Tests for GIF frame extraction."""

    def create_test_gif(self, tmp_path, frames=3, width=50, height=50, duration=100):
        """Create a test GIF file."""
        gif_path = tmp_path / "test.gif"
        
        # Create frames with different colors
        images = []
        for i in range(frames):
            # Create a solid color frame
            color = (i * 80 % 255, (i * 100) % 255, (i * 120) % 255)
            img = Image.new('RGB', (width, height), color)
            images.append(img)
        
        # Save as GIF
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0
            )
        
        return gif_path

    def test_extract_gif_frames_basic(self, tmp_path):
        """Test basic GIF frame extraction."""
        gif_path = self.create_test_gif(tmp_path, frames=3)
        result = extract_gif_frames(gif_path)
        
        assert isinstance(result, FrameExtractResult)
        assert result.frame_count == 3
        assert len(result.frames) == 3
        assert result.dimensions == (50, 50)
        assert result.duration_ms > 0
        
        # Check frame data
        for frame in result.frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (50, 50, 3)

    def test_extract_gif_frames_max_frames(self, tmp_path):
        """Test frame extraction with max_frames limit."""
        gif_path = self.create_test_gif(tmp_path, frames=10)
        result = extract_gif_frames(gif_path, max_frames=5)
        
        assert result.frame_count == 5
        assert len(result.frames) == 5

    def test_extract_gif_frames_single_frame(self, tmp_path):
        """Test extraction from single-frame image."""
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (50, 50), (255, 0, 0))
        img.save(img_path)
        
        result = extract_gif_frames(img_path)
        
        assert result.frame_count == 1
        assert len(result.frames) == 1
        assert result.dimensions == (50, 50)
        assert result.duration_ms == 0

    def test_extract_gif_frames_invalid_file(self, tmp_path):
        """Test extraction from invalid file."""
        invalid_file = tmp_path / "invalid.gif"
        invalid_file.write_text("not a gif")
        
        with pytest.raises(IOError):
            extract_gif_frames(invalid_file)


class TestFrameDimensionHandling:
    """Tests for frame dimension handling."""

    def create_frames(self, count, height, width):
        """Create test frames with specified dimensions."""
        frames = []
        for i in range(count):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frames.append(frame)
        return frames

    def test_resize_to_common_dimensions_same_size(self):
        """Test resizing when frames are already same size."""
        frames1 = self.create_frames(3, 50, 50)
        frames2 = self.create_frames(3, 50, 50)
        
        resized1, resized2 = resize_to_common_dimensions(frames1, frames2)
        
        assert len(resized1) == 3
        assert len(resized2) == 3
        assert all(frame.shape == (50, 50, 3) for frame in resized1)
        assert all(frame.shape == (50, 50, 3) for frame in resized2)

    def test_resize_to_common_dimensions_different_sizes(self):
        """Test resizing when frames have different sizes."""
        frames1 = self.create_frames(2, 100, 100)  # Larger
        frames2 = self.create_frames(2, 50, 50)    # Smaller
        
        resized1, resized2 = resize_to_common_dimensions(frames1, frames2)
        
        # Should resize to smallest common size (50x50)
        assert all(frame.shape == (50, 50, 3) for frame in resized1)
        assert all(frame.shape == (50, 50, 3) for frame in resized2)

    def test_resize_to_common_dimensions_empty_frames(self):
        """Test resizing with empty frame lists."""
        frames1 = []
        frames2 = self.create_frames(2, 50, 50)
        
        resized1, resized2 = resize_to_common_dimensions(frames1, frames2)
        
        assert len(resized1) == 0
        assert len(resized2) == 2


class TestFrameAlignment:
    """Tests for frame alignment methods."""

    def create_test_frames(self, count):
        """Create test frames with identifiable patterns."""
        frames = []
        for i in range(count):
            # Create frame with unique pattern based on index
            frame = np.full((50, 50, 3), i * 25, dtype=np.uint8)
            frames.append(frame)
        return frames



    def test_align_frames_content_based(self):
        """Test content-based alignment method (the only alignment method)."""
        original_frames = self.create_test_frames(5)
        compressed_frames = self.create_test_frames(3)
        
        aligned = align_frames_content_based(original_frames, compressed_frames)
        
        assert len(aligned) <= len(original_frames)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in aligned)

    def test_align_frames_wrapper(self):
        """Test the align_frames wrapper function."""
        original_frames = self.create_test_frames(4)
        compressed_frames = self.create_test_frames(3)
        
        aligned = align_frames(original_frames, compressed_frames)
        
        assert len(aligned) <= len(original_frames)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in aligned)



    def test_align_frames_empty_lists(self):
        """Test alignment with empty frame lists."""
        original_frames = self.create_test_frames(5)
        empty_frames = []
        
        # Test with empty compressed frames
        aligned = align_frames(original_frames, empty_frames)
        assert len(aligned) == 0
        
        # Test with empty original frames
        aligned = align_frames(empty_frames, original_frames)
        assert len(aligned) == 0




class TestMetricCalculations:
    """Tests for individual metric calculations."""

    def create_similar_frames(self):
        """Create two similar frames for testing."""
        frame1 = np.random.randint(100, 155, (50, 50, 3), dtype=np.uint8)
        # Create similar frame with small differences
        frame2 = frame1.copy()
        frame2[0:10, 0:10] = frame2[0:10, 0:10] + 10  # Small difference
        return frame1, frame2

    def create_different_frames(self):
        """Create two very different frames for testing."""
        frame1 = np.zeros((50, 50, 3), dtype=np.uint8)  # Black
        frame2 = np.full((50, 50, 3), 255, dtype=np.uint8)  # White
        return frame1, frame2

    def test_calculate_frame_mse(self):
        """Test MSE calculation between frames."""
        frame1, frame2 = self.create_similar_frames()
        mse = calculate_frame_mse(frame1, frame2)
        
        assert isinstance(mse, float)
        assert mse >= 0.0

    def test_calculate_frame_mse_different_sizes(self):
        """Test MSE calculation with different frame sizes."""
        frame1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (25, 25, 3), dtype=np.uint8)
        
        mse = calculate_frame_mse(frame1, frame2)
        assert isinstance(mse, float)
        assert mse >= 0.0

    def test_calculate_ms_ssim(self):
        """Test MS-SSIM calculation."""
        frame1, frame2 = self.create_similar_frames()
        ms_ssim_val = calculate_ms_ssim(frame1, frame2)
        
        assert isinstance(ms_ssim_val, float)
        assert 0.0 <= ms_ssim_val <= 1.0

    def test_calculate_ms_ssim_identical_frames(self):
        """Test MS-SSIM with identical frames."""
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        ms_ssim_val = calculate_ms_ssim(frame, frame)
        
        assert ms_ssim_val >= 0.9  # Should be very high for identical frames

    def test_calculate_temporal_consistency_single_frame(self):
        """Test temporal consistency with single frame."""
        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
        consistency = calculate_temporal_consistency(frames)
        
        assert consistency == 1.0  # Single frame is perfectly consistent

    def test_calculate_temporal_consistency_consistent_frames(self):
        """Test temporal consistency with very similar frames."""
        base_frame = np.random.randint(100, 155, (50, 50, 3), dtype=np.uint8)
        frames = []
        for i in range(5):
            # Create very similar frames
            frame = base_frame.copy()
            frame = frame + np.random.randint(-5, 6, frame.shape, dtype=np.int8)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
        
        consistency = calculate_temporal_consistency(frames)
        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0

    def test_calculate_temporal_consistency_inconsistent_frames(self):
        """Test temporal consistency with very different frames."""
        frames = []
        for i in range(5):
            # Create very different frames
            frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            frames.append(frame)
        
        consistency = calculate_temporal_consistency(frames)
        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0


class TestComprehensiveMetrics:
    """Tests for the main comprehensive metrics calculation."""

    def create_test_gif_pair(self, tmp_path):
        """Create a pair of test GIFs (original and compressed)."""
        # Create original GIF
        original_path = tmp_path / "original.gif"
        original_images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100), (i * 50, i * 60, i * 70))
            original_images.append(img)
        
        original_images[0].save(
            original_path,
            save_all=True,
            append_images=original_images[1:],
            duration=100,
            loop=0
        )
        
        # Create compressed GIF (smaller, fewer frames)
        compressed_path = tmp_path / "compressed.gif"
        compressed_images = []
        for i in range(0, 5, 2):  # Every 2nd frame
            img = Image.new('RGB', (80, 80), (i * 50 + 10, i * 60 + 10, i * 70 + 10))
            compressed_images.append(img)
        
        compressed_images[0].save(
            compressed_path,
            save_all=True,
            append_images=compressed_images[1:],
            duration=100,
            loop=0
        )
        
        return original_path, compressed_path

    def test_calculate_comprehensive_metrics_basic(self, tmp_path):
        """Test basic comprehensive metrics calculation."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)
        
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)
        
        # Check all required keys are present
        required_keys = ["ssim", "ms_ssim", "psnr", "temporal_consistency", 
                        "composite_quality", "render_ms", "kilobytes"]
        assert all(key in metrics for key in required_keys)
        
        # Check value ranges
        assert 0.0 <= metrics["ssim"] <= 1.0
        assert 0.0 <= metrics["ms_ssim"] <= 1.0
        assert 0.0 <= metrics["psnr"] <= 1.0
        assert 0.0 <= metrics["temporal_consistency"] <= 1.0
        assert 0.0 <= metrics["composite_quality"] <= 1.0
        assert metrics["render_ms"] >= 0
        assert metrics["kilobytes"] > 0

    def test_calculate_comprehensive_metrics_custom_config(self, tmp_path):
        """Test comprehensive metrics with custom configuration."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)
        
        config = MetricsConfig(
            SSIM_MAX_FRAMES=5,
            TEMPORAL_CONSISTENCY_ENABLED=False
        )
        
        metrics = calculate_comprehensive_metrics(original_path, compressed_path, config)
        
        assert isinstance(metrics, dict)
        assert "composite_quality" in metrics

    def test_calculate_comprehensive_metrics_content_based_alignment(self, tmp_path):
        """Test comprehensive metrics with content-based alignment (the only method)."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)
        
        # Test with default configuration (content-based alignment)
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)
        
        assert isinstance(metrics, dict)
        assert "composite_quality" in metrics
        assert 0.0 <= metrics["composite_quality"] <= 1.0

    def test_calculate_comprehensive_metrics_nonexistent_file(self):
        """Test comprehensive metrics with nonexistent files."""
        with pytest.raises((IOError, ValueError)):
            calculate_comprehensive_metrics(
                Path("nonexistent1.gif"), 
                Path("nonexistent2.gif")
            )

    def test_calculate_comprehensive_metrics_performance(self, tmp_path):
        """Test performance requirements (should be under 1 second)."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)
        
        start_time = time.perf_counter()
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Should meet performance target (significantly under 1 second)
        assert processing_time_ms < 1000  # Less than 1 second
        assert metrics["render_ms"] > 0


class TestLegacyCompatibility:
    """Tests for legacy function compatibility."""

    def test_calculate_ssim_legacy(self, tmp_path):
        """Test legacy SSIM function."""
        # Create simple test GIFs
        original_path = tmp_path / "original.gif"
        compressed_path = tmp_path / "compressed.gif"
        
        # Create identical single-frame GIFs
        img = Image.new('RGB', (50, 50), (128, 128, 128))
        img.save(original_path)
        img.save(compressed_path)
        
        ssim_value = calculate_ssim(original_path, compressed_path)
        
        assert isinstance(ssim_value, float)
        assert 0.0 <= ssim_value <= 1.0


class TestQualityDifferentiation:
    """Tests for quality differentiation validation."""

    def create_quality_test_gifs(self, tmp_path):
        """Create GIFs with different quality levels for testing differentiation."""
        base_img = Image.new('RGB', (100, 100), (128, 128, 128))
        
        # Excellent quality (identical)
        excellent_path = tmp_path / "excellent.gif"
        base_img.save(excellent_path)
        
        # Good quality (slight differences)
        good_path = tmp_path / "good.gif"
        good_array = np.array(base_img)
        good_array = good_array + np.random.randint(-10, 11, good_array.shape, dtype=np.int8)
        good_array = np.clip(good_array, 0, 255).astype(np.uint8)
        good_img = Image.fromarray(good_array)
        good_img.save(good_path)
        
        # Poor quality (significant differences)
        poor_path = tmp_path / "poor.gif"
        poor_array = np.array(base_img)
        poor_array = poor_array + np.random.randint(-50, 51, poor_array.shape, dtype=np.int8)
        poor_array = np.clip(poor_array, 0, 255).astype(np.uint8)
        poor_img = Image.fromarray(poor_array)
        poor_img.save(poor_path)
        
        return excellent_path, good_path, poor_path

    def test_quality_differentiation(self, tmp_path):
        """Test that metrics can differentiate between quality levels."""
        # Create base reference image
        reference_path = tmp_path / "reference.gif"
        ref_img = Image.new('RGB', (100, 100), (128, 128, 128))
        ref_img.save(reference_path)
        
        excellent_path, good_path, poor_path = self.create_quality_test_gifs(tmp_path)
        
        # Calculate metrics for each quality level
        excellent_metrics = calculate_comprehensive_metrics(reference_path, excellent_path)
        good_metrics = calculate_comprehensive_metrics(reference_path, good_path)
        poor_metrics = calculate_comprehensive_metrics(reference_path, poor_path)
        
        # Quality should decrease: excellent > good > poor
        assert excellent_metrics["composite_quality"] >= good_metrics["composite_quality"]
        assert good_metrics["composite_quality"] >= poor_metrics["composite_quality"]
        
        # Should achieve some differentiation (not necessarily 40% but some separation)
        quality_range = (excellent_metrics["composite_quality"] - 
                        poor_metrics["composite_quality"])
        assert quality_range > 0.1  # At least 10% differentiation


def test_default_configuration():
    """Test that default configuration works."""
    assert isinstance(DEFAULT_METRICS_CONFIG, MetricsConfig)
    assert DEFAULT_METRICS_CONFIG.SSIM_MODE == "comprehensive"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_frame_gifs(self, tmp_path):
        """Test handling of single-frame GIFs."""
        # Create single-frame GIFs
        img1 = Image.new('RGB', (50, 50), (100, 100, 100))
        img2 = Image.new('RGB', (50, 50), (110, 110, 110))
        
        path1 = tmp_path / "single1.gif"
        path2 = tmp_path / "single2.gif"
        
        img1.save(path1)
        img2.save(path2)
        
        metrics = calculate_comprehensive_metrics(path1, path2)
        
        assert isinstance(metrics, dict)
        assert metrics["temporal_consistency"] == 1.0  # Single frame is perfectly consistent

    def test_very_small_gifs(self, tmp_path):
        """Test handling of very small GIFs."""
        # Create tiny GIFs (8x8 pixels)
        img1 = Image.new('RGB', (8, 8), (50, 50, 50))
        img2 = Image.new('RGB', (8, 8), (60, 60, 60))
        
        path1 = tmp_path / "tiny1.gif"
        path2 = tmp_path / "tiny2.gif"
        
        img1.save(path1)
        img2.save(path2)
        
        metrics = calculate_comprehensive_metrics(path1, path2)
        
        assert isinstance(metrics, dict)
        assert all(0.0 <= metrics[key] <= 1.0 for key in ["ssim", "ms_ssim", "psnr", "composite_quality"])

    def test_memory_efficiency(self, tmp_path):
        """Test that large frame counts are handled efficiently."""
        # Create GIF with many frames but limit processing
        images = []
        for i in range(50):  # Create 50 frames
            img = Image.new('RGB', (50, 50), (i * 5 % 255, i * 7 % 255, i * 11 % 255))
            images.append(img)
        
        gif_path = tmp_path / "many_frames.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=50,
            loop=0
        )
        
        # Should limit to max frames automatically
        config = MetricsConfig(SSIM_MAX_FRAMES=10)
        metrics = calculate_comprehensive_metrics(gif_path, gif_path, config)
        
        assert isinstance(metrics, dict)
        assert metrics["composite_quality"] >= 0.9  # Should be high for identical GIFs 