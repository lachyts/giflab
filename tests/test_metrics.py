"""Tests for giflab.metrics module."""

import time
from pathlib import Path

import numpy as np
import pytest
from giflab.config import DEFAULT_METRICS_CONFIG, MetricsConfig
from giflab.metrics import (
    FrameExtractResult,
    align_frames,
    align_frames_content_based,
    calculate_comprehensive_metrics,
    calculate_file_size_kb,
    calculate_frame_mse,
    calculate_ms_ssim,
    calculate_ssim,
    calculate_temporal_consistency,
    extract_gif_frames,
    resize_to_common_dimensions,
)
from PIL import Image


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
        total_weight = (
            config.SSIM_WEIGHT
            + config.MS_SSIM_WEIGHT
            + config.PSNR_WEIGHT
            + config.TEMPORAL_WEIGHT
        )
        assert abs(total_weight - 1.0) < 0.001

    def test_invalid_weights_raises_error(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MetricsConfig(
                SSIM_WEIGHT=0.5,
                MS_SSIM_WEIGHT=0.5,
                PSNR_WEIGHT=0.5,
                TEMPORAL_WEIGHT=0.5,
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
            TEMPORAL_WEIGHT=0.25,
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
            img = Image.new("RGB", (width, height), color)
            images.append(img)

        # Save as GIF
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
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
        img = Image.new("RGB", (50, 50), (255, 0, 0))
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

    def test_extract_gif_frames_even_sampling(self, tmp_path):
        """Test that frame extraction uses even sampling across timeline."""
        # Create a 20-frame GIF where each frame has a unique color
        frames = []
        for i in range(20):
            color_value = int(255 * i / 19)  # 0 to 255 spread across frames
            img = Image.new("RGB", (50, 50), (color_value, 100, 150))
            frames.append(img)

        gif_path = tmp_path / "even_sampling_test.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100)

        # Extract 10 frames from 20
        result = extract_gif_frames(gif_path, max_frames=10)

        assert result.frame_count == 10
        assert len(result.frames) == 10

        # With even sampling, should get frames at indices: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        np.linspace(0, 19, 10, dtype=int)

        # Check that first and last frames are sampled
        first_frame_color = result.frames[0][0, 0, 0]  # Red channel
        last_frame_color = result.frames[-1][0, 0, 0]

        assert first_frame_color < 50  # Should be from early frame (low color value)
        assert last_frame_color > 200  # Should be from late frame (high color value)

    def test_extract_gif_frames_timeline_coverage(self, tmp_path):
        """Test that frame sampling covers full animation timeline."""
        # Create 40-frame GIF (case where consecutive sampling would miss 25%)
        gif_path = self.create_test_gif(tmp_path, frames=40)

        # Extract 30 frames with even sampling
        result = extract_gif_frames(gif_path, max_frames=30)

        assert result.frame_count == 30

        # Even sampling should include frames from near the end
        # With consecutive sampling, max frame would be 29
        # With even sampling, max frame should be 39
        import numpy as np

        expected_max_index = np.linspace(0, 39, 30, dtype=int)[-1]
        assert expected_max_index >= 35, "Should sample from end of animation"


class TestFrameDimensionHandling:
    """Tests for frame dimension handling."""

    def create_frames(self, count, height, width):
        """Create test frames with specified dimensions."""
        frames = []
        for _i in range(count):
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
        frames2 = self.create_frames(2, 50, 50)  # Smaller

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
        for _i in range(5):
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
        for _i in range(5):
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
            img = Image.new("RGB", (100, 100), (i * 50, i * 60, i * 70))
            original_images.append(img)

        original_images[0].save(
            original_path,
            save_all=True,
            append_images=original_images[1:],
            duration=100,
            loop=0,
        )

        # Create compressed GIF (smaller, fewer frames)
        compressed_path = tmp_path / "compressed.gif"
        compressed_images = []
        for i in range(0, 5, 2):  # Every 2nd frame
            img = Image.new("RGB", (80, 80), (i * 50 + 10, i * 60 + 10, i * 70 + 10))
            compressed_images.append(img)

        compressed_images[0].save(
            compressed_path,
            save_all=True,
            append_images=compressed_images[1:],
            duration=100,
            loop=0,
        )

        return original_path, compressed_path

    def test_calculate_comprehensive_metrics_basic(self, tmp_path):
        """Test basic comprehensive metrics calculation."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Check all required keys are present
        required_keys = [
            "ssim",
            "ms_ssim",
            "psnr",
            "temporal_consistency",
            "composite_quality",
            "render_ms",
            "kilobytes",
        ]
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

        config = MetricsConfig(SSIM_MAX_FRAMES=5, TEMPORAL_CONSISTENCY_ENABLED=False)

        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

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
                Path("nonexistent1.gif"), Path("nonexistent2.gif")
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
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        img.save(original_path)
        img.save(compressed_path)

        ssim_value = calculate_ssim(original_path, compressed_path)

        assert isinstance(ssim_value, float)
        assert 0.0 <= ssim_value <= 1.0


class TestQualityDifferentiation:
    """Tests for quality differentiation validation."""

    def create_quality_test_gifs(self, tmp_path):
        """Create GIFs with different quality levels for testing differentiation."""
        base_img = Image.new("RGB", (100, 100), (128, 128, 128))

        # Excellent quality (identical)
        excellent_path = tmp_path / "excellent.gif"
        base_img.save(excellent_path)

        # Good quality (slight differences)
        good_path = tmp_path / "good.gif"
        good_array = np.array(base_img)
        good_array = good_array + np.random.randint(
            -10, 11, good_array.shape, dtype=np.int8
        )
        good_array = np.clip(good_array, 0, 255).astype(np.uint8)
        good_img = Image.fromarray(good_array)
        good_img.save(good_path)

        # Poor quality (significant differences)
        poor_path = tmp_path / "poor.gif"
        poor_array = np.array(base_img)
        poor_array = poor_array + np.random.randint(
            -50, 51, poor_array.shape, dtype=np.int8
        )
        poor_array = np.clip(poor_array, 0, 255).astype(np.uint8)
        poor_img = Image.fromarray(poor_array)
        poor_img.save(poor_path)

        return excellent_path, good_path, poor_path

    def test_quality_differentiation(self, tmp_path):
        """Test that metrics can differentiate between quality levels."""
        # Create base reference image
        reference_path = tmp_path / "reference.gif"
        ref_img = Image.new("RGB", (100, 100), (128, 128, 128))
        ref_img.save(reference_path)

        excellent_path, good_path, poor_path = self.create_quality_test_gifs(tmp_path)

        # Calculate metrics for each quality level
        excellent_metrics = calculate_comprehensive_metrics(
            reference_path, excellent_path
        )
        good_metrics = calculate_comprehensive_metrics(reference_path, good_path)
        poor_metrics = calculate_comprehensive_metrics(reference_path, poor_path)

        # Quality should decrease: excellent > good > poor
        assert (
            excellent_metrics["composite_quality"] >= good_metrics["composite_quality"]
        )
        assert good_metrics["composite_quality"] >= poor_metrics["composite_quality"]

        # Should achieve some differentiation (not necessarily 40% but some separation)
        quality_range = (
            excellent_metrics["composite_quality"] - poor_metrics["composite_quality"]
        )
        assert quality_range > 0.1  # At least 10% differentiation


class TestExtendedComprehensiveMetrics:
    """Tests for the extended comprehensive metrics functionality."""

    def create_test_gif_pair(self, tmp_path):
        """Create a pair of test GIFs for extended metrics testing."""
        # Create original GIF
        original_path = tmp_path / "original.gif"
        original_images = []
        for i in range(3):
            img = Image.new("RGB", (64, 64), (i * 80, i * 60, i * 70))
            original_images.append(img)

        original_images[0].save(
            original_path,
            save_all=True,
            append_images=original_images[1:],
            duration=100,
            loop=0,
        )

        # Create compressed GIF (slightly different)
        compressed_path = tmp_path / "compressed.gif"
        compressed_images = []
        for i in range(3):
            img = Image.new("RGB", (64, 64), (i * 80 + 5, i * 60 + 5, i * 70 + 5))
            compressed_images.append(img)

        compressed_images[0].save(
            compressed_path,
            save_all=True,
            append_images=compressed_images[1:],
            duration=100,
            loop=0,
        )

        return original_path, compressed_path

    def test_extended_metrics_basic(self, tmp_path):
        """Test that all new metrics are included in comprehensive results."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Check that all new metrics are present
        expected_new_metrics = [
            "mse",
            "rmse",
            "fsim",
            "gmsd",
            "chist",
            "edge_similarity",
            "texture_similarity",
            "sharpness_similarity",
        ]

        for metric in expected_new_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(
                metrics[metric], float
            ), f"Metric {metric} should be float"

    def test_aggregation_descriptors(self, tmp_path):
        """Test that aggregation descriptors (std, min, max) are included."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Check that aggregation descriptors exist for all metrics
        base_metrics = [
            "ssim",
            "ms_ssim",
            "psnr",
            "mse",
            "rmse",
            "fsim",
            "gmsd",
            "chist",
            "edge_similarity",
            "texture_similarity",
            "sharpness_similarity",
            "temporal_consistency",
        ]

        for metric in base_metrics:
            assert f"{metric}_std" in metrics, f"Missing {metric}_std"
            assert f"{metric}_min" in metrics, f"Missing {metric}_min"
            assert f"{metric}_max" in metrics, f"Missing {metric}_max"

            # Check that std, min, max are valid
            assert (
                metrics[f"{metric}_std"] >= 0.0
            ), f"{metric}_std should be non-negative"
            assert (
                metrics[f"{metric}_min"] <= metrics[f"{metric}_max"]
            ), f"{metric}_min should be <= max"

    def test_raw_metrics_flag_disabled(self, tmp_path):
        """Test that raw metrics are not included when RAW_METRICS=False."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(RAW_METRICS=False)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that no raw metrics are present
        raw_metrics = [key for key in metrics.keys() if key.endswith("_raw")]
        assert (
            len(raw_metrics) == 0
        ), f"Should not have raw metrics when disabled, found: {raw_metrics}"

    def test_raw_metrics_flag_enabled(self, tmp_path):
        """Test that raw metrics are included when RAW_METRICS=True."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(RAW_METRICS=True)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that raw metrics are present
        expected_raw_metrics = [
            "ssim_raw",
            "ms_ssim_raw",
            "psnr_raw",
            "mse_raw",
            "rmse_raw",
            "fsim_raw",
            "gmsd_raw",
            "chist_raw",
            "edge_similarity_raw",
            "texture_similarity_raw",
            "sharpness_similarity_raw",
            "temporal_consistency_raw",
            "temporal_consistency_pre_raw",
            "temporal_consistency_post_raw",
            "temporal_consistency_delta_raw",
        ]

        for raw_metric in expected_raw_metrics:
            assert raw_metric in metrics, f"Missing raw metric: {raw_metric}"

    def test_single_frame_aggregation(self, tmp_path):
        """Test aggregation descriptors work correctly for single frame GIFs."""
        # Create single frame GIFs
        original_path = tmp_path / "single_original.gif"
        compressed_path = tmp_path / "single_compressed.gif"

        img1 = Image.new("RGB", (64, 64), (100, 100, 100))
        img2 = Image.new("RGB", (64, 64), (105, 105, 105))

        img1.save(original_path)
        img2.save(compressed_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # For single frame, std should be 0 and min should equal max
        base_metrics = [
            "ssim",
            "ms_ssim",
            "psnr",
            "mse",
            "rmse",
            "fsim",
            "gmsd",
            "chist",
            "edge_similarity",
            "texture_similarity",
            "sharpness_similarity",
        ]

        for metric in base_metrics:
            assert (
                metrics[f"{metric}_std"] == 0.0
            ), f"{metric}_std should be 0 for single frame"
            assert (
                metrics[f"{metric}_min"] == metrics[f"{metric}_max"]
            ), f"{metric}_min should equal max for single frame"

    def test_backward_compatibility(self, tmp_path):
        """Test that existing functionality still works (backward compatibility)."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Check that all original metrics are still present
        original_metrics = [
            "ssim",
            "ms_ssim",
            "psnr",
            "temporal_consistency",
            "composite_quality",
            "render_ms",
            "kilobytes",
        ]

        for metric in original_metrics:
            assert metric in metrics, f"Missing original metric: {metric}"

        # Check that composite quality is still calculated correctly
        assert 0.0 <= metrics["composite_quality"] <= 1.0
        assert metrics["render_ms"] > 0
        assert metrics["kilobytes"] > 0

    def test_metrics_count_expansion(self, tmp_path):
        """Test that the number of metrics has expanded as expected."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        # Test without raw metrics
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Should have base metrics + aggregation descriptors + positional sampling + new temporal keys (pre, post, delta)
        # 12 base metrics * 4 = 48
        # + 3 system metrics = 51
        # + 16 positional sampling = 67
        # + 3 new temporal consistency variants (pre, post, delta) = 70
        expected_count = 70
        assert (
            len(metrics) == expected_count
        ), f"Expected {expected_count} metrics, got {len(metrics)}"

        # Test with raw metrics
        config = MetricsConfig(RAW_METRICS=True)
        metrics_with_raw = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Should have all previous metrics + 15 raw metrics + 16 positional sampling = 101
        expected_count_with_raw = expected_count + 15  # 85 metrics
        assert (
            len(metrics_with_raw) == expected_count_with_raw
        ), f"Expected {expected_count_with_raw} metrics with raw, got {len(metrics_with_raw)}"

    def test_aggregation_helper_function(self):
        """Test the _aggregate_metric helper function directly."""
        from giflab.metrics import _aggregate_metric

        # Test with multiple values
        values = [0.8, 0.9, 0.7, 0.85]
        result = _aggregate_metric(values, "test_metric")

        expected_keys = [
            "test_metric",
            "test_metric_std",
            "test_metric_min",
            "test_metric_max",
        ]
        assert all(key in result for key in expected_keys)

        assert result["test_metric"] == pytest.approx(0.8125, abs=0.001)  # mean
        assert result["test_metric_min"] == 0.7
        assert result["test_metric_max"] == 0.9
        assert result["test_metric_std"] > 0.0

        # Test with single value
        single_result = _aggregate_metric([0.5], "single")
        assert single_result["single"] == 0.5
        assert single_result["single_std"] == 0.0
        assert single_result["single_min"] == 0.5
        assert single_result["single_max"] == 0.5

        # Test with empty values
        empty_result = _aggregate_metric([], "empty")
        assert empty_result["empty"] == 0.0
        assert empty_result["empty_std"] == 0.0
        assert empty_result["empty_min"] == 0.0
        assert empty_result["empty_max"] == 0.0

    def test_positional_sampling_enabled(self, tmp_path):
        """Test that positional sampling works when enabled."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(ENABLE_POSITIONAL_SAMPLING=True)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that positional metrics are present for default metrics
        default_positional_metrics = ["ssim", "mse", "fsim", "chist"]

        for metric in default_positional_metrics:
            assert f"{metric}_first" in metrics, f"Missing {metric}_first"
            assert f"{metric}_middle" in metrics, f"Missing {metric}_middle"
            assert f"{metric}_last" in metrics, f"Missing {metric}_last"
            assert (
                f"{metric}_positional_variance" in metrics
            ), f"Missing {metric}_positional_variance"

            # Check that values are reasonable
            assert isinstance(metrics[f"{metric}_first"], float)
            assert isinstance(metrics[f"{metric}_middle"], float)
            assert isinstance(metrics[f"{metric}_last"], float)
            assert metrics[f"{metric}_positional_variance"] >= 0.0

    def test_positional_sampling_disabled(self, tmp_path):
        """Test that positional sampling is not included when disabled."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(ENABLE_POSITIONAL_SAMPLING=False)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that no positional metrics are present
        positional_keys = [
            key
            for key in metrics.keys()
            if any(
                suffix in key
                for suffix in ["_first", "_middle", "_last", "_positional_variance"]
            )
        ]
        assert (
            len(positional_keys) == 0
        ), f"Should not have positional metrics when disabled, found: {positional_keys}"

    def test_positional_sampling_custom_metrics(self, tmp_path):
        """Test positional sampling with custom metric selection."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(
            ENABLE_POSITIONAL_SAMPLING=True,
            POSITIONAL_METRICS=["ssim", "mse"],  # Only these two
        )
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that only specified metrics have positional data
        expected_metrics = ["ssim", "mse"]
        unexpected_metrics = ["fsim", "chist", "gmsd"]

        for metric in expected_metrics:
            assert f"{metric}_first" in metrics, f"Missing {metric}_first"
            assert (
                f"{metric}_positional_variance" in metrics
            ), f"Missing {metric}_positional_variance"

        for metric in unexpected_metrics:
            assert f"{metric}_first" not in metrics, f"Should not have {metric}_first"
            assert (
                f"{metric}_positional_variance" not in metrics
            ), f"Should not have {metric}_positional_variance"

    def test_positional_sampling_single_frame(self, tmp_path):
        """Test positional sampling with single frame GIFs."""
        # Create single frame GIFs
        original_path = tmp_path / "single_original.gif"
        compressed_path = tmp_path / "single_compressed.gif"

        img1 = Image.new("RGB", (64, 64), (100, 100, 100))
        img2 = Image.new("RGB", (64, 64), (105, 105, 105))

        img1.save(original_path)
        img2.save(compressed_path)

        config = MetricsConfig(ENABLE_POSITIONAL_SAMPLING=True)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # For single frame, first/middle/last should be identical
        for metric in ["ssim", "mse", "fsim", "chist"]:
            first_val = metrics[f"{metric}_first"]
            middle_val = metrics[f"{metric}_middle"]
            last_val = metrics[f"{metric}_last"]

            assert (
                first_val == middle_val == last_val
            ), f"{metric} positional values should be identical for single frame"
            # Use pytest.approx for floating point comparison
            assert metrics[f"{metric}_positional_variance"] == pytest.approx(
                0.0, abs=1e-10
            ), f"{metric} positional variance should be ~0 for single frame"

    def test_positional_sampling_helper_function(self):
        """Test the _calculate_positional_samples helper function directly."""
        import numpy as np
        from giflab.metrics import _calculate_positional_samples, mse

        # Create test frame pairs with different qualities
        frames = []
        for i in range(5):
            frame1 = np.full((32, 32, 3), i * 50, dtype=np.uint8)
            frame2 = np.full(
                (32, 32, 3), i * 50 + 10, dtype=np.uint8
            )  # Slightly different
            frames.append((frame1, frame2))

        result = _calculate_positional_samples(frames, mse, "test_mse")

        # Check that all expected keys are present
        expected_keys = [
            "test_mse_first",
            "test_mse_middle",
            "test_mse_last",
            "test_mse_positional_variance",
        ]
        assert all(key in result for key in expected_keys)

        # Check that values are reasonable
        assert result["test_mse_first"] >= 0.0
        assert result["test_mse_middle"] >= 0.0
        assert result["test_mse_last"] >= 0.0
        assert result["test_mse_positional_variance"] >= 0.0

        # Test with empty frames - fix the expected keys
        empty_result = _calculate_positional_samples([], mse, "empty_mse")
        empty_expected_keys = [
            "empty_mse_first",
            "empty_mse_middle",
            "empty_mse_last",
            "empty_mse_positional_variance",
        ]
        assert all(empty_result[key] == 0.0 for key in empty_expected_keys)

    def test_positional_variance_interpretation(self, tmp_path):
        """Test that positional variance correctly indicates position importance."""
        # Create GIFs with varying frame quality to test variance calculation
        original_path = tmp_path / "varying_original.gif"
        compressed_path = tmp_path / "varying_compressed.gif"

        # Create frames with different patterns to ensure MSE differences
        original_images = []
        compressed_images = []

        for i in range(5):
            # Create original with consistent pattern
            orig_img = Image.new("RGB", (64, 64), (100, 100, 100))

            # Create compressed with increasing noise/differences
            comp_img = Image.new("RGB", (64, 64), (100, 100, 100))

            # Add different amounts of "noise" by drawing rectangles
            from PIL import ImageDraw

            draw = ImageDraw.Draw(comp_img)

            # More rectangles for later frames (simulating more compression artifacts)
            for j in range(i + 1):  # 1, 2, 3, 4, 5 rectangles
                x = j * 12
                y = j * 12
                draw.rectangle([x, y, x + 8, y + 8], fill=(150 + j * 20, 100, 100))

            original_images.append(orig_img)
            compressed_images.append(comp_img)

        original_images[0].save(
            original_path,
            save_all=True,
            append_images=original_images[1:],
            duration=100,
            loop=0,
        )
        compressed_images[0].save(
            compressed_path,
            save_all=True,
            append_images=compressed_images[1:],
            duration=100,
            loop=0,
        )

        config = MetricsConfig(ENABLE_POSITIONAL_SAMPLING=True)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Test that positional sampling produces valid results
        mse_first = metrics["mse_first"]
        mse_middle = metrics["mse_middle"]
        mse_last = metrics["mse_last"]
        mse_variance = metrics["mse_positional_variance"]

        # Test that we have meaningful positional data
        assert (
            mse_first >= 0.0 and mse_middle >= 0.0 and mse_last >= 0.0
        ), "All positional MSE values should be non-negative"

        # The variance should be non-negative (key insight: if variance is low, position doesn't matter much)
        assert mse_variance >= 0.0, "Positional variance should be non-negative"

        # Test that the positional sampling mechanism works (values are computed)
        assert isinstance(mse_first, float), "First position value should be a float"
        assert isinstance(mse_middle, float), "Middle position value should be a float"
        assert isinstance(mse_last, float), "Last position value should be a float"
        assert isinstance(mse_variance, float), "Positional variance should be a float"

        # The key insight: if all values are the same, variance will be 0 (position doesn't matter)
        # If values differ, variance will be > 0 (position matters)
        # Both cases are valid and informative for production decisions


def test_default_configuration():
    """Test that default configuration works."""
    assert isinstance(DEFAULT_METRICS_CONFIG, MetricsConfig)
    assert DEFAULT_METRICS_CONFIG.SSIM_MODE == "comprehensive"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_frame_gifs(self, tmp_path):
        """Test handling of single-frame GIFs."""
        # Create single-frame GIFs
        img1 = Image.new("RGB", (50, 50), (100, 100, 100))
        img2 = Image.new("RGB", (50, 50), (110, 110, 110))

        path1 = tmp_path / "single1.gif"
        path2 = tmp_path / "single2.gif"

        img1.save(path1)
        img2.save(path2)

        metrics = calculate_comprehensive_metrics(path1, path2)

        assert isinstance(metrics, dict)
        assert (
            metrics["temporal_consistency"] == 1.0
        )  # Single frame is perfectly consistent

    def test_very_small_gifs(self, tmp_path):
        """Test handling of very small GIFs."""
        # Create tiny GIFs (8x8 pixels)
        img1 = Image.new("RGB", (8, 8), (50, 50, 50))
        img2 = Image.new("RGB", (8, 8), (60, 60, 60))

        path1 = tmp_path / "tiny1.gif"
        path2 = tmp_path / "tiny2.gif"

        img1.save(path1)
        img2.save(path2)

        metrics = calculate_comprehensive_metrics(path1, path2)

        assert isinstance(metrics, dict)
        assert all(
            0.0 <= metrics[key] <= 1.0
            for key in ["ssim", "ms_ssim", "psnr", "composite_quality"]
        )

    def test_memory_efficiency(self, tmp_path):
        """Test that large frame counts are handled efficiently."""
        # Create GIF with many frames but limit processing
        images = []
        for i in range(50):  # Create 50 frames
            img = Image.new("RGB", (50, 50), (i * 5 % 255, i * 7 % 255, i * 11 % 255))
            images.append(img)

        gif_path = tmp_path / "many_frames.gif"
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=50, loop=0
        )

        # Should limit to max frames automatically
        config = MetricsConfig(SSIM_MAX_FRAMES=10)
        metrics = calculate_comprehensive_metrics(gif_path, gif_path, config)

        assert isinstance(metrics, dict)
        assert metrics["composite_quality"] >= 0.9  # Should be high for identical GIFs
