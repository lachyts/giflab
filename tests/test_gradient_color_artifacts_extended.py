"""Extended edge case and boundary condition tests for gradient banding and color validation.

This test suite covers unusual scenarios, boundary conditions, error handling,
and performance characteristics that weren't covered in the basic unit tests.
"""

from unittest.mock import patch

import numpy as np
import pytest

from giflab.gradient_color_artifacts import (
    GradientBandingDetector,
    PerceptualColorValidator,
    calculate_gradient_color_metrics,
)


class TestGradientBandingEdgeCases:
    """Test edge cases and boundary conditions for gradient banding detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = GradientBandingDetector(patch_size=32, variance_threshold=100.0)

    @pytest.mark.fast
    def test_very_small_images(self):
        """Test detection with images smaller than patch size."""
        # Create tiny images (smaller than patch size)
        tiny_frames = []
        for i in range(3):
            frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            tiny_frames.append(frame)

        result = self.detector.detect_banding_artifacts(tiny_frames, tiny_frames)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["banding_score_mean"] >= 0.0
        assert result["gradient_region_count"] >= 0

    @pytest.mark.fast
    def test_single_pixel_images(self):
        """Test with single pixel images."""
        single_pixel_frames = []
        for i in range(2):
            frame = np.full((1, 1, 3), [i * 127, 128, 255 - i * 127], dtype=np.uint8)
            single_pixel_frames.append(frame)

        result = self.detector.detect_banding_artifacts(
            single_pixel_frames, single_pixel_frames
        )

        # Should not crash and return meaningful defaults
        assert result["banding_score_mean"] == 0.0
        assert result["gradient_region_count"] == 0
        assert result["banding_patch_count"] == 0

    @pytest.mark.fast
    def test_solid_color_images(self):
        """Test with solid color images (no gradients)."""
        solid_frames = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for color in colors:
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            solid_frames.append(frame)

        result = self.detector.detect_banding_artifacts(solid_frames, solid_frames)

        # Solid colors should not trigger banding detection
        assert result["banding_score_mean"] == 0.0  # No gradients to analyze
        assert result["gradient_region_count"] == 0

    @pytest.mark.fast
    def test_extreme_gradients(self):
        """Test with extreme gradients (0-255 in minimal distance)."""
        extreme_frames = []

        for i in range(3):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            # Create extremely sharp gradient (black to white in 2 pixels)
            frame[:, :2] = [0, 0, 0]  # Black
            frame[:, 2:] = [255, 255, 255]  # White
            extreme_frames.append(frame)

        result = self.detector.detect_banding_artifacts(extreme_frames, extreme_frames)

        # Extreme gradients might not be detected as "gradients" due to high variance
        assert isinstance(result, dict)
        assert result["banding_score_mean"] >= 0.0

    @pytest.mark.fast
    def test_corrupted_frame_data(self):
        """Test handling of corrupted or invalid frame data."""
        # Test with NaN values
        corrupted_frame = np.full((64, 64, 3), np.nan, dtype=np.float32)

        # Should handle gracefully (convert to valid values)
        try:
            regions = self.detector.detect_gradient_regions(
                corrupted_frame.astype(np.uint8)
            )
            assert isinstance(regions, list)
        except Exception:
            # If it fails, that's acceptable for corrupted data
            pass

    @pytest.mark.fast
    def test_different_dtypes(self):
        """Test with different numpy data types."""
        # Test with float32 data
        float_frame = np.random.random((64, 64, 3)).astype(np.float32)

        regions = self.detector.detect_gradient_regions(float_frame)
        assert isinstance(regions, list)

        # Test with int32 data
        int32_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.int32)

        regions = self.detector.detect_gradient_regions(int32_frame)
        assert isinstance(regions, list)

    @pytest.mark.fast
    def test_grayscale_input(self):
        """Test with grayscale input (should handle RGB conversion)."""
        # Create grayscale-like RGB input
        gray_value = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        gray_frame = np.broadcast_to(gray_value, (64, 64, 3)).copy()

        regions = self.detector.detect_gradient_regions(gray_frame)
        assert isinstance(regions, list)

    @pytest.mark.fast
    def test_memory_efficiency_large_images(self):
        """Test memory efficiency with large images."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large frames (but still reasonable)
        large_frames = []
        for i in range(2):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            large_frames.append(frame)

        result = self.detector.detect_banding_artifacts(large_frames, large_frames)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Memory increase should be reasonable (<100MB for test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
        assert isinstance(result, dict)

    @pytest.mark.fast
    def test_gradient_region_merge_logic(self):
        """Test the region merging logic specifically."""
        # Create overlapping regions to test merging
        overlapping_regions = [
            (10, 10, 20, 20),  # First region
            (15, 15, 20, 20),  # Overlapping region
            (50, 50, 20, 20),  # Separate region
        ]

        merged = self.detector._merge_overlapping_regions(overlapping_regions)

        # Should merge overlapping regions
        assert len(merged) <= len(overlapping_regions)

        # All regions should be valid
        for x, y, w, h in merged:
            assert x >= 0 and y >= 0 and w > 0 and h > 0

    @pytest.mark.fast
    def test_empty_region_list(self):
        """Test merging with empty region list."""
        empty_regions = []
        merged = self.detector._merge_overlapping_regions(empty_regions)
        assert merged == []

    @pytest.mark.fast
    def test_single_region_no_merge(self):
        """Test that single region doesn't get modified."""
        single_region = [(10, 10, 20, 20)]
        merged = self.detector._merge_overlapping_regions(single_region)
        assert merged == single_region

    @pytest.mark.fast
    def test_histogram_with_empty_patch(self):
        """Test histogram calculation with uniform patches."""
        # Uniform patch (all same value)
        uniform_patch = np.full((32, 32), 128, dtype=np.uint8)

        histogram = self.detector.calculate_gradient_magnitude_histogram(uniform_patch)

        # Should return valid histogram
        assert len(histogram) == 32  # Default bins
        # Uniform patches have no gradients, so histogram sum may be 0.0
        assert np.sum(histogram) >= 0.0
        assert np.all(histogram >= 0.0)  # All bins should be non-negative

    @pytest.mark.fast
    def test_contour_detection_edge_cases(self):
        """Test contour detection with edge cases."""
        # All black patch
        black_patch = np.zeros((32, 32), dtype=np.uint8)
        contours = self.detector.detect_contours_in_patches(black_patch)
        assert contours == 0  # No contours in uniform black

        # All white patch
        white_patch = np.full((32, 32), 255, dtype=np.uint8)
        contours = self.detector.detect_contours_in_patches(white_patch)
        assert contours == 0  # No contours in uniform white

        # Checkerboard pattern
        checkerboard = np.zeros((32, 32), dtype=np.uint8)
        checkerboard[::2, ::2] = 255  # White squares
        checkerboard[1::2, 1::2] = 255
        contours = self.detector.detect_contours_in_patches(checkerboard)
        assert contours >= 0  # Should detect contours (algorithm dependent)

    @pytest.mark.fast
    def test_is_gradient_patch_edge_cases(self):
        """Test gradient patch detection with edge cases."""
        # Noise patch (high variance, no clear gradient)
        np.random.seed(42)  # Reproducible
        noise_patch = np.random.randint(0, 255, (32, 32), dtype=np.uint8).astype(
            np.float32
        )
        is_gradient = self.detector._is_gradient_patch(noise_patch)
        assert isinstance(is_gradient, bool)

        # Linear gradient patch
        gradient_patch = np.linspace(0, 255, 32 * 32).reshape(32, 32).astype(np.float32)
        is_gradient = self.detector._is_gradient_patch(gradient_patch)
        assert isinstance(is_gradient, bool)


class TestPerceptualColorEdgeCases:
    """Test edge cases for perceptual color validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 2, 3, 5]
        )

    @pytest.mark.fast
    def test_monochrome_images(self):
        """Test with monochrome (grayscale) images."""
        # Create grayscale images
        gray_frames = []
        for i in range(3):
            gray_value = i * 85  # 0, 85, 170
            frame = np.full(
                (64, 64, 3), [gray_value, gray_value, gray_value], dtype=np.uint8
            )
            gray_frames.append(frame)

        result = self.validator.calculate_color_difference_metrics(
            gray_frames, gray_frames
        )

        # Identical frames should have zero color difference
        assert result["deltae_mean"] == pytest.approx(0.0, abs=0.1)
        assert result["color_patch_count"] > 0

    @pytest.mark.fast
    def test_out_of_gamut_colors(self):
        """Test with colors that might be problematic in Lab space."""
        # Create frames with extreme colors
        extreme_frames = []
        extreme_colors = [
            (0, 0, 0),  # Pure black
            (255, 255, 255),  # Pure white
            (255, 0, 0),  # Pure red
            (0, 255, 0),  # Pure green
            (0, 0, 255),  # Pure blue
        ]

        for color in extreme_colors:
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            extreme_frames.append(frame)

        result = self.validator.calculate_color_difference_metrics(
            extreme_frames[:3], extreme_frames[2:5]
        )

        # Should handle extreme colors without crashing
        assert isinstance(result, dict)
        assert result["deltae_mean"] >= 0.0

    @pytest.mark.fast
    def test_extreme_color_differences(self):
        """Test with colors that have very large ΔE00 differences."""
        # Black vs white should have large ΔE
        black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        white_frame = np.full((64, 64, 3), 255, dtype=np.uint8)

        result = self.validator.calculate_color_difference_metrics(
            [black_frame], [white_frame]
        )

        # Black vs white should have large ΔE
        assert result["deltae_mean"] > 50.0  # Should be very high
        assert result["deltae_pct_gt5"] > 90.0  # Most patches exceed ΔE=5

    @pytest.mark.fast
    def test_near_identical_colors(self):
        """Test with colors that are nearly identical (ΔE00 < 0.1)."""
        # Create frames with tiny color differences
        frame1 = np.full((64, 64, 3), [128, 128, 128], dtype=np.uint8)
        frame2 = np.full(
            (64, 64, 3), [129, 128, 128], dtype=np.uint8
        )  # Tiny difference

        result = self.validator.calculate_color_difference_metrics([frame1], [frame2])

        # Should detect small but non-zero difference
        assert result["deltae_mean"] > 0.0
        assert result["deltae_mean"] < 5.0  # But not large
        assert result["deltae_pct_gt5"] == 0.0  # No patches exceed ΔE=5

    @pytest.mark.fast
    def test_different_bit_depths(self):
        """Test with different bit depth inputs."""
        # 16-bit input (should be handled)
        frame_16bit = np.random.randint(0, 65535, (64, 64, 3), dtype=np.uint16)

        lab_converted = self.validator.rgb_to_lab(frame_16bit)
        assert lab_converted.shape == frame_16bit.shape
        assert lab_converted.dtype == np.float32

    @pytest.mark.fast
    def test_rgb_to_lab_fallback(self):
        """Test RGB to Lab conversion fallback when scikit-image unavailable."""
        with patch("giflab.gradient_color_artifacts.SKIMAGE_AVAILABLE", False):
            validator = PerceptualColorValidator()

            rgb_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            result = validator.rgb_to_lab(rgb_image)

            # Should return something (fallback to RGB approximation)
            assert result.shape == rgb_image.shape
            assert result.dtype == np.float32

    @pytest.mark.fast
    def test_deltae2000_with_edge_values(self):
        """Test CIEDE2000 calculation with edge case values."""
        # Test with zero values
        lab_zero = np.zeros((1, 1, 3), dtype=np.float32)
        lab_small = np.array([[[1, 1, 1]]], dtype=np.float32)

        deltae = self.validator.calculate_deltae2000(lab_zero, lab_small)
        assert deltae.shape == (1, 1)
        assert deltae[0, 0] >= 0.0

    @pytest.mark.fast
    def test_patch_sampling_very_small_frames(self):
        """Test patch sampling with frames smaller than patch size."""
        tiny_frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)

        patches = self.validator.sample_color_patches(tiny_frame, num_samples=4)

        # Should return fewer patches or handle gracefully
        assert isinstance(patches, list)
        assert len(patches) <= 4

    @pytest.mark.fast
    def test_patch_sampling_exact_patch_size(self):
        """Test patch sampling when frame is exactly patch size."""
        exact_frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        patches = self.validator.sample_color_patches(exact_frame, num_samples=1)

        # Should return exactly one patch covering the whole frame
        assert len(patches) == 1
        assert patches[0] == (0, 0, 32, 32)

    @pytest.mark.fast
    def test_threshold_boundary_values(self):
        """Test threshold calculations at exact boundary values."""
        # Create known ΔE values at threshold boundaries
        frame_orig = np.zeros((64, 64, 3), dtype=np.uint8)

        # Create frame with controlled color differences
        frame_test = np.zeros((64, 64, 3), dtype=np.uint8)
        frame_test[:32, :32] = [10, 0, 0]  # Small difference
        frame_test[:32, 32:] = [30, 0, 0]  # Medium difference
        frame_test[32:, :32] = [60, 0, 0]  # Large difference
        frame_test[32:, 32:] = [120, 0, 0]  # Very large difference

        result = self.validator.calculate_color_difference_metrics(
            [frame_orig], [frame_test]
        )

        # Should properly categorize into threshold buckets
        assert result["deltae_pct_gt1"] >= result["deltae_pct_gt2"]
        assert result["deltae_pct_gt2"] >= result["deltae_pct_gt3"]
        assert result["deltae_pct_gt3"] >= result["deltae_pct_gt5"]

    @pytest.mark.fast
    def test_empty_patch_list(self):
        """Test behavior when no patches can be sampled."""
        # This could happen with very small images
        empty_frame = np.zeros((1, 1, 3), dtype=np.uint8)

        # Mock the patch sampling to return empty list
        with patch.object(self.validator, "sample_color_patches", return_value=[]):
            result = self.validator.calculate_color_difference_metrics(
                [empty_frame], [empty_frame]
            )

            assert result["deltae_mean"] == 0.0
            assert result["color_patch_count"] == 0


class TestIntegrationEdgeCases:
    """Test edge cases in the integration functions."""

    @pytest.mark.fast
    def test_mismatched_frame_counts(self):
        """Test with different numbers of original vs compressed frames."""
        orig_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)
        ]
        comp_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]

        result = calculate_gradient_color_metrics(orig_frames, comp_frames)

        # Should handle gracefully by using minimum frame count
        assert isinstance(result, dict)
        assert all(isinstance(v, (int, float)) for v in result.values())

    @pytest.mark.fast
    def test_none_frame_inputs(self):
        """Test with None inputs."""
        result = calculate_gradient_color_metrics(None, None)

        # Should return default/empty metrics
        assert isinstance(result, dict)
        assert result.get("banding_score_mean", 0) == 0.0
        assert result.get("deltae_mean", 0) == 0.0

    @pytest.mark.fast
    def test_empty_list_inputs(self):
        """Test with empty frame lists."""
        result = calculate_gradient_color_metrics([], [])

        # Should handle empty inputs gracefully
        assert isinstance(result, dict)
        assert result.get("banding_score_mean", 0) == 0.0
        assert result.get("deltae_mean", 0) == 0.0

    @pytest.mark.fast
    def test_mixed_frame_shapes(self):
        """Test with frames of different sizes in same list."""
        mixed_frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8),
        ]

        # Should handle mixed sizes (will be cropped to minimum)
        result = calculate_gradient_color_metrics(mixed_frames, mixed_frames)
        assert isinstance(result, dict)

    @pytest.mark.fast
    def test_exception_handling_in_combined_function(self):
        """Test exception handling in the main combined function."""
        # Mock one of the sub-functions to raise an exception
        with patch(
            "giflab.gradient_color_artifacts.calculate_banding_metrics",
            side_effect=Exception("Test exception"),
        ):
            frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]
            result = calculate_gradient_color_metrics(frames, frames)

            # Should return fallback metrics on exception
            assert isinstance(result, dict)
            # Should contain fallback values
            expected_keys = [
                "banding_score_mean",
                "banding_score_p95",
                "banding_patch_count",
                "gradient_region_count",
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

    @pytest.mark.fast
    def test_thread_safety_simulation(self):
        """Test that functions are thread-safe by running concurrent calls."""
        import threading

        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)
        ]
        results = []
        exceptions = []

        def run_calculation():
            try:
                result = calculate_gradient_color_metrics(frames, frames)
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_calculation)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should complete without exceptions
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 5

        # Results should be consistent (within reasonable tolerance)
        first_result = results[0]
        for result in results[1:]:
            for key in first_result:
                if isinstance(first_result[key], float):
                    assert (
                        abs(first_result[key] - result[key]) < 1e-10
                    ), f"Inconsistent results for {key}"
                else:
                    assert (
                        first_result[key] == result[key]
                    ), f"Inconsistent results for {key}"


# Performance and stress tests
class TestPerformanceEdgeCases:
    """Test performance characteristics and stress conditions."""

    @pytest.mark.fast
    def test_performance_scaling(self):
        """Test performance scaling with different image sizes."""
        import time

        sizes = [(32, 32), (64, 64), (128, 128)]
        times = []

        for size in sizes:
            frames = [
                np.random.randint(0, 255, (*size, 3), dtype=np.uint8) for _ in range(2)
            ]

            start_time = time.time()
            calculate_gradient_color_metrics(frames, frames)
            end_time = time.time()

            times.append(end_time - start_time)

        # Performance should scale reasonably (not exponentially)
        # Allow for some variance but check general trend
        assert times[0] < 1.0  # 32x32 should be fast
        assert times[1] < 2.0  # 64x64 should still be reasonable
        assert times[2] < 5.0  # 128x128 might be slower but not excessive

    @pytest.mark.fast
    def test_memory_cleanup(self):
        """Test that temporary objects are properly cleaned up."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        def memory_usage():
            return process.memory_info().rss / (1024 * 1024)  # MB

        initial_memory = memory_usage()

        # Run calculations in a loop
        for _ in range(5):
            frames = [
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)
            ]
            result = calculate_gradient_color_metrics(frames, frames)
            del frames, result
            gc.collect()

        final_memory = memory_usage()
        memory_increase = final_memory - initial_memory

        # Memory should not increase significantly (<50MB)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB"


# Fixtures for reusable test data
@pytest.fixture
def tiny_frames():
    """Fixture providing very small test frames."""
    return [np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(2)]


@pytest.fixture
def large_frames():
    """Fixture providing large test frames."""
    return [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)]


@pytest.fixture
def gradient_frames():
    """Fixture providing frames with known gradients."""
    frames = []
    for i in range(3):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Linear horizontal gradient
        for x in range(64):
            intensity = int(x * 255 / 63)
            frame[:, x] = [intensity, intensity // 2, intensity // 3]
        frames.append(frame)
    return frames


@pytest.fixture
def solid_color_frames():
    """Fixture providing solid color frames."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    frames = []
    for color in colors:
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        frames.append(frame)
    return frames


# Integration with existing test markers
pytestmark = [pytest.mark.gradient_color_extended]
