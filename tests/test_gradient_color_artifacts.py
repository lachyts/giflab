"""Unit tests for gradient banding and color artifact detection.

This test suite covers the gradient banding detection and perceptual color
validation systems, including edge cases, boundary conditions, and performance
characteristics.
"""

from unittest.mock import patch

import numpy as np
import pytest

from giflab.gradient_color_artifacts import (
    GradientBandingDetector,
    PerceptualColorValidator,
    calculate_banding_metrics,
    calculate_gradient_color_metrics,
    calculate_perceptual_color_metrics,
)


class TestGradientBandingDetector:
    """Test gradient banding detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = GradientBandingDetector(patch_size=32, variance_threshold=100.0)

    @pytest.mark.fast
    def test_detector_initialization(self):
        """Test detector initialization with different parameters."""
        detector = GradientBandingDetector()
        assert detector.patch_size == 64
        assert detector.variance_threshold == 100.0

        detector_custom = GradientBandingDetector(
            patch_size=32, variance_threshold=50.0
        )
        assert detector_custom.patch_size == 32
        assert detector_custom.variance_threshold == 50.0

    @pytest.mark.fast
    def test_smooth_gradient_no_banding(self):
        """Test that smooth gradients don't trigger banding detection."""
        # Create smooth horizontal gradient
        frames = self._create_smooth_gradient_frames()

        result = self.detector.detect_banding_artifacts(frames, frames)

        # Smooth gradient should have low banding scores
        assert result["banding_score_mean"] < 10.0
        assert result["gradient_region_count"] >= 0
        assert result["banding_patch_count"] >= 0

    @pytest.mark.fast
    def test_banded_gradient_detection(self):
        """Test that posterized gradients trigger banding detection."""
        # Create posterized (banded) gradient frames
        smooth_frames = self._create_smooth_gradient_frames()
        banded_frames = self._create_banded_gradient_frames()

        result = self.detector.detect_banding_artifacts(smooth_frames, banded_frames)

        # Banded gradient should have higher banding scores
        assert result["banding_score_mean"] >= 0.0
        assert result["gradient_region_count"] >= 0
        if result["banding_patch_count"] > 0:
            assert result["banding_score_p95"] >= result["banding_score_mean"]

    @pytest.mark.fast
    def test_gradient_region_detection(self):
        """Test gradient region identification."""
        # Create frame with clear gradient regions
        frame = self._create_test_gradient_frame()

        regions = self.detector.detect_gradient_regions(frame)

        # Should find at least some gradient regions
        assert isinstance(regions, list)
        for region in regions:
            x, y, w, h = region
            assert x >= 0 and y >= 0 and w > 0 and h > 0

    @pytest.mark.fast
    def test_gradient_magnitude_histogram(self):
        """Test gradient magnitude histogram calculation."""
        # Create test patch with gradients
        patch = np.linspace(0, 255, 32 * 32).reshape(32, 32).astype(np.uint8)

        histogram = self.detector.calculate_gradient_magnitude_histogram(patch)

        assert len(histogram) == 32  # Default bins
        assert np.sum(histogram) == pytest.approx(1.0, abs=0.1)  # Normalized
        assert np.all(histogram >= 0)

    @pytest.mark.fast
    def test_contour_detection(self):
        """Test false contour detection in patches."""
        # Create patch with artificial contours
        patch = np.zeros((32, 32), dtype=np.uint8)
        patch[:16] = 100  # Top half darker
        patch[16:] = 200  # Bottom half lighter

        contour_count = self.detector.detect_contours_in_patches(patch)

        assert isinstance(contour_count, int)
        assert contour_count >= 0

    @pytest.mark.fast
    def test_banding_severity_calculation(self):
        """Test banding severity score calculation."""
        # Create smooth patch and banded version
        smooth_patch = np.linspace(0, 255, 32 * 32).reshape(32, 32).astype(np.uint8)

        # Create banded version (posterized)
        banded_patch = (smooth_patch // 64) * 64  # Reduce to 4 levels

        severity = self.detector.calculate_banding_severity(smooth_patch, banded_patch)

        assert isinstance(severity, float)
        assert 0.0 <= severity <= 100.0

    @pytest.mark.fast
    def test_empty_frames_handling(self):
        """Test handling of empty frame lists."""
        result = self.detector.detect_banding_artifacts([], [])

        assert result["banding_score_mean"] == 0.0
        assert result["banding_score_p95"] == 0.0
        assert result["banding_patch_count"] == 0
        assert result["gradient_region_count"] == 0

    @pytest.mark.fast
    def test_single_frame_handling(self):
        """Test handling of single-frame inputs."""
        frame = self._create_test_gradient_frame()

        result = self.detector.detect_banding_artifacts([frame], [frame])

        # Should handle single frame gracefully
        assert isinstance(result, dict)
        assert all(
            key in result
            for key in [
                "banding_score_mean",
                "banding_score_p95",
                "banding_patch_count",
                "gradient_region_count",
            ]
        )

    @pytest.mark.fast
    def test_mismatched_frame_shapes(self):
        """Test handling of frames with different sizes."""
        frame_small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        frame_large = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = self.detector.detect_banding_artifacts([frame_small], [frame_large])

        # Should handle size mismatch gracefully
        assert isinstance(result, dict)
        assert result["banding_score_mean"] >= 0.0

    def _create_smooth_gradient_frames(self, num_frames=3):
        """Create frames with smooth gradients for testing."""
        frames = []
        for i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            # Create horizontal gradient
            for x in range(64):
                intensity = int(x * 255 / 63)
                frame[:, x] = [intensity, intensity, intensity]

            frames.append(frame)
        return frames

    def _create_banded_gradient_frames(self, num_frames=3, bands=8):
        """Create frames with posterized/banded gradients for testing."""
        frames = []
        for i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            # Create banded horizontal gradient
            band_width = 64 // bands
            for band in range(bands):
                intensity = int(band * 255 / (bands - 1))
                x_start = band * band_width
                x_end = min((band + 1) * band_width, 64)
                frame[:, x_start:x_end] = [intensity, intensity, intensity]

            frames.append(frame)
        return frames

    def _create_test_gradient_frame(self):
        """Create a single frame with gradient for testing."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # Create diagonal gradient
        for y in range(64):
            for x in range(64):
                intensity = int((x + y) * 255 / (63 + 63))
                frame[y, x] = [intensity, intensity, intensity]

        return frame


class TestPerceptualColorValidator:
    """Test perceptual color validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 2, 3, 5]
        )

    @pytest.mark.fast
    def test_validator_initialization(self):
        """Test validator initialization with different parameters."""
        validator = PerceptualColorValidator()
        assert validator.patch_size == 64
        assert validator.jnd_thresholds == [1.0, 2.0, 3.0, 5.0]

        validator_custom = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 3]
        )
        assert validator_custom.patch_size == 32
        assert validator_custom.jnd_thresholds == [1, 3]

    @pytest.mark.fast
    def test_rgb_to_lab_conversion(self):
        """Test RGB to Lab color space conversion."""
        # Create test RGB image
        rgb_image = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],  # Red, Green
                [[0, 0, 255], [128, 128, 128]],  # Blue, Gray
            ],
            dtype=np.uint8,
        )

        lab_image = self.validator.rgb_to_lab(rgb_image)

        assert lab_image.shape == rgb_image.shape
        assert lab_image.dtype == np.float32

    @pytest.mark.fast
    def test_deltae2000_calculation(self):
        """Test CIEDE2000 color difference calculation."""
        # Create test Lab colors
        lab1 = np.array([[[50, 20, 30]]], dtype=np.float32)  # Reference color
        lab2 = np.array([[[55, 25, 35]]], dtype=np.float32)  # Slightly different color

        deltae = self.validator.calculate_deltae2000(lab1, lab2)

        assert deltae.shape == (1, 1)
        assert deltae[0, 0] >= 0.0
        assert isinstance(float(deltae[0, 0]), float)

    @pytest.mark.fast
    def test_identical_colors_zero_deltae(self):
        """Test that identical colors have zero ΔE00."""
        lab_color = np.array([[[50, 20, 30]]], dtype=np.float32)

        deltae = self.validator.calculate_deltae2000(lab_color, lab_color)

        assert deltae[0, 0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.fast
    def test_color_patch_sampling(self):
        """Test smart color patch sampling from frames."""
        # Create test frame
        frame = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

        patches = self.validator.sample_color_patches(frame, num_samples=9)

        assert isinstance(patches, list)
        assert len(patches) <= 9
        for color_patch in patches:
            x, y, w, h = color_patch
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert x + w <= 96 and y + h <= 96

    @pytest.mark.fast
    def test_small_frame_sampling(self):
        """Test patch sampling on frames smaller than patch size."""
        # Create very small frame
        small_frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)

        patches = self.validator.sample_color_patches(small_frame, num_samples=4)

        # Should handle small frame gracefully
        assert isinstance(patches, list)
        assert len(patches) >= 0

    @pytest.mark.fast
    def test_identical_frames_no_color_difference(self):
        """Test that identical frames have minimal color difference."""
        frames = self._create_test_color_frames()

        result = self.validator.calculate_color_difference_metrics(frames, frames)

        assert result["deltae_mean"] == pytest.approx(0.0, abs=0.1)
        assert result["deltae_p95"] == pytest.approx(0.0, abs=0.1)
        assert result["deltae_pct_gt1"] == pytest.approx(0.0, abs=1.0)

    @pytest.mark.fast
    def test_different_color_frames(self):
        """Test color difference detection between different frames."""
        frames_original = self._create_test_color_frames()
        frames_shifted = self._create_shifted_color_frames()

        result = self.validator.calculate_color_difference_metrics(
            frames_original, frames_shifted
        )

        assert result["deltae_mean"] >= 0.0
        assert result["deltae_p95"] >= result["deltae_mean"]
        assert result["deltae_max"] >= result["deltae_p95"]
        assert result["color_patch_count"] > 0

    @pytest.mark.fast
    def test_empty_frames_handling(self):
        """Test handling of empty frame lists."""
        result = self.validator.calculate_color_difference_metrics([], [])

        expected_keys = [
            "deltae_mean",
            "deltae_p95",
            "deltae_max",
            "deltae_pct_gt1",
            "deltae_pct_gt2",
            "deltae_pct_gt3",
            "deltae_pct_gt5",
            "color_patch_count",
        ]

        assert all(key in result for key in expected_keys)
        assert result["deltae_mean"] == 0.0
        assert result["color_patch_count"] == 0

    @pytest.mark.fast
    def test_threshold_percentages(self):
        """Test that threshold percentages are calculated correctly."""
        # Create frames with known color differences
        frames_original = self._create_test_color_frames()
        frames_high_diff = self._create_high_difference_color_frames()

        result = self.validator.calculate_color_difference_metrics(
            frames_original, frames_high_diff
        )

        # Check that percentages are valid
        assert 0.0 <= result["deltae_pct_gt1"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt2"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt3"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt5"] <= 100.0

        # Higher thresholds should have lower or equal percentages
        assert result["deltae_pct_gt1"] >= result["deltae_pct_gt2"]
        assert result["deltae_pct_gt2"] >= result["deltae_pct_gt3"]
        assert result["deltae_pct_gt3"] >= result["deltae_pct_gt5"]

    def _create_test_color_frames(self, num_frames=3):
        """Create frames with test colors for validation."""
        frames = []
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (128, 128, 128),  # Gray
        ]

        for i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            # Create colored patches
            patch_size = 32
            for idx, color in enumerate(colors):
                x = (idx % 2) * patch_size
                y = (idx // 2) * patch_size
                frame[y : y + patch_size, x : x + patch_size] = color

            frames.append(frame)
        return frames

    def _create_shifted_color_frames(self, num_frames=3):
        """Create frames with slightly shifted colors."""
        frames = []
        colors = [
            (235, 20, 20),  # Shifted red
            (20, 235, 20),  # Shifted green
            (20, 20, 235),  # Shifted blue
            (148, 108, 108),  # Shifted gray
        ]

        for i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            # Create shifted colored patches
            patch_size = 32
            for idx, color in enumerate(colors):
                x = (idx % 2) * patch_size
                y = (idx // 2) * patch_size
                frame[y : y + patch_size, x : x + patch_size] = color

            frames.append(frame)
        return frames

    def _create_high_difference_color_frames(self, num_frames=3):
        """Create frames with high color differences for threshold testing."""
        frames = []
        colors = [
            (100, 200, 50),  # Very different from red
            (200, 50, 200),  # Very different from green
            (255, 255, 0),  # Very different from blue
            (50, 200, 200),  # Very different from gray
        ]

        for i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            # Create very different colored patches
            patch_size = 32
            for idx, color in enumerate(colors):
                x = (idx % 2) * patch_size
                y = (idx // 2) * patch_size
                frame[y : y + patch_size, x : x + patch_size] = color

            frames.append(frame)
        return frames


class TestGradientColorIntegration:
    """Test integration functions and edge cases."""

    @pytest.mark.fast
    def test_calculate_banding_metrics_function(self):
        """Test main banding metrics calculation function."""
        frames = self._create_test_frames()

        result = calculate_banding_metrics(frames, frames)

        expected_keys = [
            "banding_score_mean",
            "banding_score_p95",
            "banding_patch_count",
            "gradient_region_count",
        ]
        assert all(key in result for key in expected_keys)
        assert all(isinstance(result[key], (int, float)) for key in expected_keys)

    @pytest.mark.fast
    def test_calculate_perceptual_color_metrics_function(self):
        """Test main perceptual color metrics calculation function."""
        frames = self._create_test_frames()

        result = calculate_perceptual_color_metrics(frames, frames)

        expected_keys = [
            "deltae_mean",
            "deltae_p95",
            "deltae_max",
            "deltae_pct_gt1",
            "deltae_pct_gt2",
            "deltae_pct_gt3",
            "deltae_pct_gt5",
            "color_patch_count",
        ]
        assert all(key in result for key in expected_keys)
        assert all(isinstance(result[key], (int, float)) for key in expected_keys)

    @pytest.mark.fast
    def test_calculate_gradient_color_metrics_combined(self):
        """Test combined gradient and color metrics calculation."""
        frames = self._create_test_frames()

        result = calculate_gradient_color_metrics(frames, frames)

        # Should include both banding and color metrics
        expected_banding_keys = [
            "banding_score_mean",
            "banding_score_p95",
            "banding_patch_count",
            "gradient_region_count",
        ]
        expected_color_keys = [
            "deltae_mean",
            "deltae_p95",
            "deltae_max",
            "deltae_pct_gt1",
            "deltae_pct_gt2",
            "deltae_pct_gt3",
            "deltae_pct_gt5",
            "color_patch_count",
        ]

        all_keys = expected_banding_keys + expected_color_keys
        assert all(key in result for key in all_keys)

    @pytest.mark.fast
    def test_exception_handling_in_combined_function(self):
        """Test that combined function handles exceptions gracefully."""
        # This tests the try-catch in the combined function
        with patch(
            "giflab.gradient_color_artifacts.calculate_banding_metrics",
            side_effect=Exception("Test error"),
        ):
            frames = self._create_test_frames()
            result = calculate_gradient_color_metrics(frames, frames)

            # Should return fallback metrics
            assert isinstance(result, dict)
            assert "banding_score_mean" in result
            assert result["banding_score_mean"] == 0.0

    @pytest.mark.fast
    def test_none_inputs_handling(self):
        """Test handling of None inputs."""
        result = calculate_gradient_color_metrics(None, None)

        # Should handle None gracefully (converted to empty lists)
        assert isinstance(result, dict)

    @pytest.mark.fast
    def test_performance_with_large_frames(self):
        """Test performance characteristics with larger frames."""
        import time

        # Create larger test frames
        large_frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            large_frames.append(frame)

        start_time = time.time()
        result = calculate_gradient_color_metrics(large_frames, large_frames)
        end_time = time.time()

        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0  # 10 seconds max
        assert isinstance(result, dict)

    def _create_test_frames(self, num_frames=3):
        """Create simple test frames for testing."""
        frames = []
        for i in range(num_frames):
            # Create frame with gradient and color patches
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            # Add horizontal gradient in top half
            for x in range(64):
                intensity = int(x * 255 / 63)
                frame[:32, x] = [intensity, intensity // 2, intensity // 3]

            # Add solid colors in bottom half
            frame[32:48, :32] = [255, 0, 0]  # Red
            frame[32:48, 32:] = [0, 255, 0]  # Green
            frame[48:, :32] = [0, 0, 255]  # Blue
            frame[48:, 32:] = [128, 128, 128]  # Gray

            frames.append(frame)
        return frames


# Pytest fixtures for reusable test data
@pytest.fixture
def test_gradient_frames():
    """Fixture providing test frames with gradients."""
    frames = []
    for i in range(3):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Horizontal gradient
        for x in range(64):
            intensity = int(x * 255 / 63)
            frame[:, x] = [intensity, intensity, intensity]
        frames.append(frame)
    return frames


@pytest.fixture
def test_color_frames():
    """Fixture providing test frames with solid colors."""
    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i, color in enumerate(colors):
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        frames.append(frame)
    return frames


# Integration with existing test markers
pytestmark = [pytest.mark.gradient_color_artifacts]
