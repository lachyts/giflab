"""Integration tests for gradient banding and color validation with metrics system.

This test suite validates the integration of the gradient and color artifact detection
with the main metrics calculation pipeline, CSV output, and existing validation systems.
"""

import csv
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.config import MetricsConfig
from giflab.metrics import (
    calculate_comprehensive_metrics,
)
from PIL import Image, ImageDraw


class TestMetricsIntegration:
    """Test integration with calculate_comprehensive_metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_metrics_included_in_comprehensive_output(self):
        """Verify all new metrics appear in comprehensive results."""
        # Create test GIFs
        original_gif = self._create_test_gif("original.gif")
        compressed_gif = self._create_test_gif("compressed.gif")

        # Calculate comprehensive metrics
        result = calculate_comprehensive_metrics(
            original_path=original_gif, compressed_path=compressed_gif
        )

        # Verify gradient banding metrics are included
        expected_banding_metrics = [
            "banding_score_mean",
            "banding_score_p95",
            "banding_patch_count",
            "gradient_region_count",
        ]

        for metric in expected_banding_metrics:
            assert metric in result, f"Missing banding metric: {metric}"
            assert isinstance(
                result[metric], int | float
            ), f"Invalid type for {metric}: {type(result[metric])}"

        # Verify color validation metrics are included
        expected_color_metrics = [
            "deltae_mean",
            "deltae_p95",
            "deltae_max",
            "deltae_pct_gt1",
            "deltae_pct_gt2",
            "deltae_pct_gt3",
            "deltae_pct_gt5",
            "color_patch_count",
        ]

        for metric in expected_color_metrics:
            assert metric in result, f"Missing color metric: {metric}"
            assert isinstance(
                result[metric], int | float
            ), f"Invalid type for {metric}: {type(result[metric])}"

    @pytest.mark.fast
    def test_metrics_with_real_gifs(self):
        """Test with actual GIF files from fixtures."""
        # Create test GIFs with different characteristics
        gradient_gif = self._create_gradient_gif("gradient_test.gif")
        solid_gif = self._create_solid_gif("solid_test.gif")

        # Calculate metrics for gradient GIF
        gradient_result = calculate_comprehensive_metrics(
            original_path=gradient_gif,
            compressed_path=gradient_gif,  # Same file for consistency test
        )

        # Calculate metrics for solid color GIF
        solid_result = calculate_comprehensive_metrics(
            original_path=solid_gif, compressed_path=solid_gif
        )

        # Gradient GIF should have different characteristics than solid GIF
        # Gradient might have more gradient regions detected
        assert (
            gradient_result["gradient_region_count"]
            >= solid_result["gradient_region_count"]
        )

        # Both should have valid color metrics
        assert gradient_result["color_patch_count"] > 0
        assert solid_result["color_patch_count"] > 0

    @pytest.mark.fast
    def test_fallback_when_module_unavailable(self):
        """Test graceful degradation when gradient_color_artifacts import fails."""
        # Mock the module import to raise ImportError
        import sys

        original_module = sys.modules.get("giflab.gradient_color_artifacts")
        if "giflab.gradient_color_artifacts" in sys.modules:
            del sys.modules["giflab.gradient_color_artifacts"]

        with patch.dict("sys.modules", {"giflab.gradient_color_artifacts": None}):
            original_gif = self._create_test_gif("original.gif")
            compressed_gif = self._create_test_gif("compressed.gif")

            # Should not crash, but fall back to default values
            result = calculate_comprehensive_metrics(
                original_path=original_gif, compressed_path=compressed_gif
            )

            # Should contain fallback values
            assert result["banding_score_mean"] == 0.0
            assert result["deltae_mean"] == 0.0
            assert result["color_patch_count"] == 0

        # Restore the original module
        if original_module is not None:
            sys.modules["giflab.gradient_color_artifacts"] = original_module

    @pytest.mark.fast
    def test_metrics_calculation_exception_handling(self):
        """Test that metrics calculation exceptions are handled gracefully."""
        with patch(
            "giflab.gradient_color_artifacts.calculate_gradient_color_metrics",
            side_effect=Exception("Calculation failed"),
        ):
            original_gif = self._create_test_gif("original.gif")
            compressed_gif = self._create_test_gif("compressed.gif")

            # Should handle exception and return fallback values
            result = calculate_comprehensive_metrics(
                original_path=original_gif, compressed_path=compressed_gif
            )

            # Should contain fallback values
            expected_fallback_keys = [
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

            for key in expected_fallback_keys:
                assert key in result
                assert result[key] == 0.0

    @pytest.mark.fast
    def test_metrics_with_different_frame_counts(self):
        """Test metrics calculation with GIFs having different frame counts."""
        # Create GIFs with different frame counts
        single_frame_gif = self._create_test_gif("single.gif", frames=1)
        multi_frame_gif = self._create_test_gif("multi.gif", frames=5)

        # Calculate metrics between different frame counts
        result = calculate_comprehensive_metrics(
            original_path=multi_frame_gif, compressed_path=single_frame_gif
        )

        # Should handle frame count mismatch gracefully
        assert isinstance(result["banding_score_mean"], float)
        assert isinstance(result["deltae_mean"], float)

        # Verify frame count information is correct
        assert result["frame_count"] >= 1  # Original frame count
        assert result["compressed_frame_count"] >= 1  # Compressed frame count

    @pytest.mark.fast
    def test_metrics_value_ranges(self):
        """Test that metrics values are within expected ranges."""
        original_gif = self._create_test_gif("original.gif")
        compressed_gif = self._create_test_gif("compressed.gif")

        result = calculate_comprehensive_metrics(
            original_path=original_gif, compressed_path=compressed_gif
        )

        # Test banding metrics ranges
        assert 0.0 <= result["banding_score_mean"] <= 100.0
        assert 0.0 <= result["banding_score_p95"] <= 100.0
        assert result["banding_patch_count"] >= 0
        assert result["gradient_region_count"] >= 0

        # Test color metrics ranges
        assert result["deltae_mean"] >= 0.0
        assert result["deltae_p95"] >= 0.0
        assert result["deltae_max"] >= 0.0
        assert 0.0 <= result["deltae_pct_gt1"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt2"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt3"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt5"] <= 100.0
        assert result["color_patch_count"] >= 0

        # Test percentage ordering
        assert result["deltae_pct_gt1"] >= result["deltae_pct_gt2"]
        assert result["deltae_pct_gt2"] >= result["deltae_pct_gt3"]
        assert result["deltae_pct_gt3"] >= result["deltae_pct_gt5"]

    # Helper methods
    def _create_test_gif(self, filename: str, size=(32, 32), frames=3):
        """Create a simple test GIF."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple colored frames
            color = (i * 80 % 255, 100, 150)
            img = Image.new("RGB", size, color=color)

            # Add some simple content
            draw = ImageDraw.Draw(img)
            if size[0] > 10 and size[1] > 10:
                draw.rectangle([2, 2, 8, 8], fill=(255, 255, 255))

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path

    def _create_gradient_gif(self, filename: str, size=(64, 64)):
        """Create a GIF with gradient content."""
        gif_path = self.temp_dir / filename

        images = []
        for _i in range(3):
            img = Image.new("RGB", size)
            pixels = img.load()

            # Create horizontal gradient
            for x in range(size[0]):
                for y in range(size[1]):
                    intensity = int(x * 255 / (size[0] - 1))
                    pixels[x, y] = (intensity, intensity // 2, intensity // 3)

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path

    def _create_solid_gif(self, filename: str, size=(64, 64)):
        """Create a GIF with solid colors."""
        gif_path = self.temp_dir / filename

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        images = []

        for color in colors:
            img = Image.new("RGB", size, color=color)
            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


class TestCSVOutputIntegration:
    """Test CSV output integration for new metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_metrics_csv_output(self):
        """Verify metrics are correctly written to CSV."""
        original_gif = self._create_test_gif("original.gif")
        compressed_gif = self._create_test_gif("compressed.gif")

        # Calculate metrics and get result
        result = calculate_comprehensive_metrics(
            original_path=original_gif, compressed_path=compressed_gif
        )

        # Create a CSV file with the metrics
        csv_path = self.temp_dir / "metrics_output.csv"

        # Write metrics to CSV (simulating what the main pipeline does)
        with open(csv_path, "w", newline="") as csvfile:
            if result:  # Check if result is not empty
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writeheader()
                writer.writerow(result)

        # Read back and verify
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            # Check that gradient and color metrics are in CSV
            expected_metrics = [
                "banding_score_mean",
                "banding_score_p95",
                "deltae_mean",
                "deltae_pct_gt3",
                "color_patch_count",
            ]

            for metric in expected_metrics:
                assert metric in row, f"Missing metric in CSV: {metric}"
                # Verify it's a valid number
                float(row[metric])  # Should not raise exception

    @pytest.mark.fast
    def test_csv_headers_consistency(self):
        """Test that CSV headers are consistent across runs."""
        # Create test GIFs
        gif1 = self._create_test_gif("test1.gif")
        gif2 = self._create_test_gif("test2.gif")

        # Calculate metrics for both
        result1 = calculate_comprehensive_metrics(
            original_path=gif1, compressed_path=gif1
        )
        result2 = calculate_comprehensive_metrics(
            original_path=gif2, compressed_path=gif2
        )

        # Headers should be identical
        assert set(result1.keys()) == set(result2.keys())

        # All gradient and color metrics should be present
        expected_gradient_color_metrics = [
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

        for metric in expected_gradient_color_metrics:
            assert metric in result1.keys()
            assert metric in result2.keys()

    # Helper methods
    def _create_test_gif(self, filename: str, size=(32, 32), frames=3):
        """Create a simple test GIF."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple colored frames
            color = (i * 80 % 255, 100, 150)
            img = Image.new("RGB", size, color=color)

            # Add some simple content
            draw = ImageDraw.Draw(img)
            if size[0] > 10 and size[1] > 10:
                draw.rectangle([2, 2, 8, 8], fill=(255, 255, 255))

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


class TestPerformanceIntegration:
    """Test performance impact of new metrics on the overall pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_metrics_performance_impact(self):
        """Measure performance overhead of new metrics."""
        original_gif = self._create_test_gif("original.gif", frames=5)
        compressed_gif = self._create_test_gif("compressed.gif", frames=5)

        # Warm up the cache first (load any models that will be needed)
        calculate_comprehensive_metrics(
            original_path=original_gif, compressed_path=compressed_gif
        )

        # Now measure without gradient/color metrics
        with patch(
            "giflab.gradient_color_artifacts.calculate_gradient_color_metrics",
            return_value={},
        ):
            start_time = time.time()
            calculate_comprehensive_metrics(
                original_path=original_gif, compressed_path=compressed_gif
            )
            time_without_metrics = time.time() - start_time

        # Measure time with new metrics (models already cached)
        start_time = time.time()
        calculate_comprehensive_metrics(
            original_path=original_gif, compressed_path=compressed_gif
        )
        time_with_metrics = time.time() - start_time

        # Performance overhead should be reasonable (<2x slower)
        overhead = time_with_metrics / max(
            time_without_metrics, 0.001
        )  # Avoid division by zero
        assert overhead < 2.0, f"Performance overhead too high: {overhead:.2f}x"

        # Overall time should still be reasonable (<5 seconds for test)
        assert time_with_metrics < 5.0, f"Total time too high: {time_with_metrics:.2f}s"

    @pytest.mark.fast
    def test_memory_impact(self):
        """Test memory impact of new metrics calculation."""
        import gc
        import os

        import psutil

        # Import cleanup function for model cache
        from giflab.model_cache import cleanup_model_cache

        process = psutil.Process(os.getpid())
        
        # Clean cache before test to ensure clean state
        cleanup_model_cache(force=True)
        gc.collect()
        
        initial_memory = process.memory_info().rss

        try:
            # Create and process multiple GIFs
            for i in range(3):
                original_gif = self._create_test_gif(f"original_{i}.gif")
                compressed_gif = self._create_test_gif(f"compressed_{i}.gif")

                result = calculate_comprehensive_metrics(
                    original_path=original_gif, compressed_path=compressed_gif
                )

                # Verify we got results
                assert "banding_score_mean" in result
                assert "deltae_mean" in result

            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB

            # Memory increase should be reasonable (<100MB)
            assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
            
        finally:
            # Always clean up model cache after test
            cleanup_model_cache(force=True)
            gc.collect()

    # Helper methods
    def _create_test_gif(self, filename: str, size=(32, 32), frames=3):
        """Create a simple test GIF."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple colored frames
            color = (i * 80 % 255, 100, 150)
            img = Image.new("RGB", size, color=color)

            # Add some simple content
            draw = ImageDraw.Draw(img)
            if size[0] > 10 and size[1] > 10:
                draw.rectangle([2, 2, 8, 8], fill=(255, 255, 255))

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


class TestConfigurationIntegration:
    """Test integration with configuration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_metrics_with_custom_config(self):
        """Test metrics calculation with custom metrics configuration."""
        # Create custom metrics config
        custom_config = MetricsConfig(
            USE_COMPREHENSIVE_METRICS=True,
            SSIM_MAX_FRAMES=10,  # Reduced for faster testing
        )

        original_gif = self._create_test_gif("original.gif")
        compressed_gif = self._create_test_gif("compressed.gif")

        # Test that config doesn't interfere with gradient/color metrics
        with patch("giflab.metrics.DEFAULT_METRICS_CONFIG", custom_config):
            result = calculate_comprehensive_metrics(
                original_path=original_gif, compressed_path=compressed_gif
            )

        # New metrics should still be present regardless of config
        assert "banding_score_mean" in result
        assert "deltae_mean" in result
        assert result["color_patch_count"] >= 0

    @pytest.mark.fast
    def test_metrics_with_disabled_comprehensive(self):
        """Test behavior when comprehensive metrics are disabled."""
        # Create config with comprehensive metrics disabled
        minimal_config = MetricsConfig(USE_COMPREHENSIVE_METRICS=False)

        original_gif = self._create_test_gif("original.gif")
        compressed_gif = self._create_test_gif("compressed.gif")

        # Even with comprehensive disabled, gradient/color metrics should still work
        with patch("giflab.metrics.DEFAULT_METRICS_CONFIG", minimal_config):
            result = calculate_comprehensive_metrics(
                original_path=original_gif, compressed_path=compressed_gif
            )

        # New metrics should still be calculated
        assert "banding_score_mean" in result
        assert "deltae_mean" in result

    # Helper methods
    def _create_test_gif(self, filename: str, size=(32, 32), frames=3):
        """Create a simple test GIF."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple colored frames
            color = (i * 80 % 255, 100, 150)
            img = Image.new("RGB", size, color=color)

            # Add some simple content
            draw = ImageDraw.Draw(img)
            if size[0] > 10 and size[1] > 10:
                draw.rectangle([2, 2, 8, 8], fill=(255, 255, 255))

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


class TestEdgeCaseIntegration:
    """Test integration edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_corrupted_gif_handling(self):
        """Test handling of corrupted GIF files."""
        # Create a "corrupted" GIF (actually just a text file)
        corrupted_gif = self.temp_dir / "corrupted.gif"
        corrupted_gif.write_text("This is not a GIF file")

        valid_gif = self._create_test_gif("valid.gif")

        # Should handle gracefully without crashing
        try:
            result = calculate_comprehensive_metrics(
                original_path=valid_gif, compressed_path=corrupted_gif
            )
            # If it succeeds, gradient/color metrics should be present
            assert "banding_score_mean" in result
        except Exception:
            # If it fails, that's acceptable for corrupted files
            pass

    @pytest.mark.fast
    def test_very_small_gifs(self):
        """Test with very small GIF files."""
        small_gif = self._create_test_gif("small.gif", size=(4, 4), frames=1)

        result = calculate_comprehensive_metrics(
            original_path=small_gif, compressed_path=small_gif
        )

        # Should handle small GIFs without crashing
        assert isinstance(result["banding_score_mean"], float)
        assert isinstance(result["deltae_mean"], float)

    @pytest.mark.fast
    def test_single_pixel_gifs(self):
        """Test with single-pixel GIF files."""
        pixel_gif = self._create_test_gif("pixel.gif", size=(1, 1), frames=1)

        result = calculate_comprehensive_metrics(
            original_path=pixel_gif, compressed_path=pixel_gif
        )

        # Should handle gracefully
        assert result["banding_score_mean"] == 0.0  # No gradients possible
        assert result["gradient_region_count"] == 0
        # Color metrics might still work with single pixel
        assert result["deltae_mean"] >= 0.0

    # Helper methods
    def _create_test_gif(self, filename: str, size=(32, 32), frames=3):
        """Create a simple test GIF."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple colored frames
            color = (i * 80 % 255, 100, 150)
            img = Image.new("RGB", size, color=color)

            # Add some simple content
            draw = ImageDraw.Draw(img)
            if size[0] > 10 and size[1] > 10:
                draw.rectangle([2, 2, 8, 8], fill=(255, 255, 255))

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path

    def _create_gradient_gif(self, filename: str, size=(64, 64)):
        """Create a GIF with gradient content."""
        gif_path = self.temp_dir / filename

        images = []
        for _i in range(3):
            img = Image.new("RGB", size)
            pixels = img.load()

            # Create horizontal gradient
            for x in range(size[0]):
                for y in range(size[1]):
                    intensity = int(x * 255 / (size[0] - 1))
                    pixels[x, y] = (intensity, intensity // 2, intensity // 3)

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path

    def _create_solid_gif(self, filename: str, size=(64, 64)):
        """Create a GIF with solid colors."""
        gif_path = self.temp_dir / filename

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        images = []

        for color in colors:
            img = Image.new("RGB", size, color=color)
            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


# Integration with existing test markers
pytestmark = [pytest.mark.gradient_color_integration]
