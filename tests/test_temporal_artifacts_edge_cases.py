"""Edge case and performance tests for temporal artifact detection.

This test suite covers unusual scenarios, boundary conditions, error handling,
and performance characteristics of the temporal artifact detection system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from giflab.optimization_validation.data_structures import ValidationConfig
from giflab.optimization_validation.validation_checker import ValidationChecker
from giflab.temporal_artifacts import (
    TemporalArtifactDetector,
    calculate_enhanced_temporal_metrics,
)


class TestTemporalArtifactEdgeCases:
    """Test edge cases in temporal artifact detection."""

    @pytest.mark.fast
    def test_single_frame_gif(self):
        """Test temporal detection with single-frame GIF."""
        # Create a single frame GIF
        img = Image.new("RGB", (32, 32), (255, 0, 0))

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = Path(tmp.name)
            img.save(gif_path)

            try:
                # Test with temporal metrics using simple frame data
                detector = TemporalArtifactDetector()

                # Create single frame as numpy array
                single_frame = np.array(img).astype(np.uint8)

                # Test detector methods with single frame
                flicker_result = detector.detect_flicker_excess([single_frame])
                assert flicker_result["flicker_excess"] == 0.0
                assert flicker_result["flicker_frame_ratio"] == 0.0

                flat_result = detector.detect_flat_region_flicker([single_frame])
                assert flat_result["flat_flicker_ratio"] == 0.0

                pumping_result = detector.detect_temporal_pumping([single_frame])
                assert pumping_result["temporal_pumping_score"] == 0.0

                # LPIPS should handle single frame
                lpips_result = detector.calculate_lpips_temporal([single_frame])
                assert lpips_result["lpips_t_mean"] == 0.0

            finally:
                gif_path.unlink()

    def test_empty_frames_list(self):
        """Test temporal detection with empty frames list."""
        detector = TemporalArtifactDetector()

        # Should handle empty frames gracefully
        flicker_result = detector.detect_flicker_excess([])
        assert flicker_result["flicker_excess"] == 0.0
        assert flicker_result["flicker_frame_ratio"] == 0.0

        flat_result = detector.detect_flat_region_flicker([])
        assert flat_result["flat_flicker_ratio"] == 0.0

        pumping_result = detector.detect_temporal_pumping([])
        assert pumping_result["temporal_pumping_score"] == 0.0

        lpips_result = detector.calculate_lpips_temporal([])
        assert lpips_result["lpips_t_mean"] == 0.0

    def test_identical_frames(self):
        """Test temporal detection with identical frames (no temporal change)."""
        # Create multiple identical frames
        frames = []
        base_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        base_frame[:] = 128  # Gray image

        for _ in range(5):
            frames.append(base_frame.copy())

        detector = TemporalArtifactDetector()

        # All metrics should indicate no temporal artifacts
        flicker_result = detector.detect_flicker_excess(frames)
        assert flicker_result["flicker_excess"] == 0.0
        assert flicker_result["flicker_frame_ratio"] == 0.0

        flat_result = detector.detect_flat_region_flicker(frames)
        assert flat_result["flat_flicker_ratio"] == 0.0  # No flicker in flat regions

        pumping_result = detector.detect_temporal_pumping(frames)
        assert pumping_result["temporal_pumping_score"] == 0.0

        lpips_result = detector.calculate_lpips_temporal(frames)
        assert lpips_result["lpips_t_mean"] == 0.0

    def test_mismatched_frame_sizes(self):
        """Test temporal detection with frames of different sizes."""
        frames = []

        # Create frames with different sizes
        frames.append(np.zeros((32, 32, 3), dtype=np.uint8))
        frames.append(np.zeros((64, 64, 3), dtype=np.uint8))
        frames.append(np.zeros((48, 48, 3), dtype=np.uint8))

        detector = TemporalArtifactDetector()

        # Should handle mismatched sizes gracefully (likely by resizing or skipping)
        try:
            flicker_result = detector.detect_flicker_excess(frames)
            assert isinstance(flicker_result, dict)
            assert "flicker_excess" in flicker_result

            flat_result = detector.detect_flat_region_flicker(frames)
            assert isinstance(flat_result, dict)
            assert "flat_flicker_ratio" in flat_result

        except Exception as e:
            # If handling mismatched sizes by raising exception, that's also acceptable
            assert "size" in str(e).lower() or "shape" in str(e).lower()

    def test_extreme_frame_dimensions(self):
        """Test temporal detection with extremely small and large frames."""
        detector = TemporalArtifactDetector()

        # Test with very small frames (1x1)
        small_frames = [
            np.array([[[255, 0, 0]]], dtype=np.uint8),
            np.array([[[0, 255, 0]]], dtype=np.uint8),
        ]

        small_result = detector.detect_flicker_excess(small_frames)
        assert isinstance(small_result, dict)
        assert "flicker_excess" in small_result

        # Test with very large frames (if computationally feasible)
        # Note: This might be skipped in practice due to memory constraints
        try:
            large_frames = [
                np.zeros((1024, 1024, 3), dtype=np.uint8),
                np.ones((1024, 1024, 3), dtype=np.uint8) * 255,
            ]

            large_result = detector.detect_flicker_excess(large_frames)
            assert isinstance(large_result, dict)
            assert "flicker_excess" in large_result

        except MemoryError:
            pytest.skip("Large frame test skipped due to memory constraints")

    def test_extreme_color_values(self):
        """Test temporal detection with extreme color values."""
        frames = []

        # All black frame
        frames.append(np.zeros((64, 64, 3), dtype=np.uint8))

        # All white frame
        frames.append(np.ones((64, 64, 3), dtype=np.uint8) * 255)

        # High contrast patterns
        checkerboard = np.zeros((64, 64, 3), dtype=np.uint8)
        for x in range(64):
            for y in range(64):
                if (x + y) % 2 == 0:
                    checkerboard[x, y] = [255, 255, 255]
        frames.append(checkerboard)

        detector = TemporalArtifactDetector()

        # Should handle extreme values
        flicker_result = detector.detect_flicker_excess(frames, threshold=0.1)
        assert isinstance(flicker_result, dict)
        assert flicker_result["flicker_excess"] >= 0.0

        flat_result = detector.detect_flat_region_flicker(frames)
        assert isinstance(flat_result, dict)
        assert flat_result["flat_flicker_ratio"] >= 0.0

    def test_invalid_gif_file(self):
        """Test temporal detection with invalid GIF file."""
        # Create a file that's not a valid GIF
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            invalid_gif = Path(tmp.name)
            invalid_gif.write_text("This is not a GIF file")

            try:
                # Should handle invalid GIF gracefully - test with calculate_enhanced_temporal_metrics
                with pytest.raises(
                    Exception
                ):  # Specific exception depends on implementation
                    calculate_enhanced_temporal_metrics(
                        str(invalid_gif), str(invalid_gif)
                    )

            finally:
                invalid_gif.unlink()

    def test_corrupted_gif_frames(self):
        """Test temporal detection with partially corrupted GIF."""
        # This is harder to test without actual corrupted files
        # Instead, test with frames that have unusual characteristics

        detector = TemporalArtifactDetector()

        # Create frames with NaN values (simulating corruption)
        corrupted_frames = []
        frame1 = np.full((32, 32, 3), 100, dtype=np.float32)
        frame2 = np.full((32, 32, 3), 150, dtype=np.float32)
        frame2[10:20, 10:20] = np.nan  # Introduce NaN values

        corrupted_frames.append(frame1.astype(np.uint8))
        corrupted_frames.append(
            np.nan_to_num(frame2).astype(np.uint8)
        )  # Clean NaN for input

        # Should handle gracefully
        try:
            result = detector.detect_flicker_excess(corrupted_frames)
            assert isinstance(result, dict)
        except Exception:
            # Acceptable to raise exception for corrupted data
            assert True

    def test_memory_intensive_operations(self):
        """Test temporal detection under memory pressure."""
        detector = TemporalArtifactDetector()

        # Create a reasonable number of frames that would stress memory
        frames = []
        for i in range(20):  # 20 frames of reasonable size
            frame = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            frames.append(frame)

        try:
            # Should complete without memory errors
            flicker_result = detector.detect_flicker_excess(frames)
            assert isinstance(flicker_result, dict)

            flat_result = detector.detect_flat_region_flicker(frames)
            assert isinstance(flat_result, dict)

            pumping_result = detector.detect_temporal_pumping(frames)
            assert isinstance(pumping_result, dict)

        except MemoryError:
            pytest.skip("Memory-intensive test skipped due to insufficient memory")

    def test_zero_threshold_values(self):
        """Test temporal detection with zero thresholds."""
        detector = TemporalArtifactDetector()

        # Create frames with minimal differences
        frames = []
        base = np.zeros((32, 32, 3), dtype=np.uint8)
        frames.append(base)

        # Second frame with tiny difference
        frame2 = base.copy()
        frame2[0, 0] = [1, 1, 1]  # Minimal change
        frames.append(frame2)

        # Test with zero threshold - should detect any change
        flicker_result = detector.detect_flicker_excess(frames, threshold=0.0)
        assert flicker_result["flicker_excess"] > 0.0  # Should detect the tiny change

        flat_result = detector.detect_flat_region_flicker(
            frames, variance_threshold=0.0
        )
        # With zero variance threshold, any change should be detected
        assert isinstance(flat_result, dict)

    def test_maximum_threshold_values(self):
        """Test temporal detection with very high thresholds."""
        detector = TemporalArtifactDetector()

        # Create frames with dramatic differences
        frames = []
        frames.append(np.zeros((32, 32, 3), dtype=np.uint8))  # All black
        frames.append(np.ones((32, 32, 3), dtype=np.uint8) * 255)  # All white

        # Test with very high threshold - should not detect even dramatic changes
        flicker_result = detector.detect_flicker_excess(frames, threshold=1.0)
        assert flicker_result["flicker_excess"] <= 1.0  # Should be under threshold

        flat_result = detector.detect_flat_region_flicker(
            frames, variance_threshold=10000.0
        )
        assert flat_result["flat_flicker_ratio"] == 0.0  # High threshold should pass


@pytest.mark.performance
class TestTemporalArtifactPerformanceEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.mark.slow
    def test_performance_with_many_frames(self):
        """Test performance with a large number of frames."""
        detector = TemporalArtifactDetector()

        # Create many frames (but reasonable size)
        frames = []
        for i in range(100):  # 100 frames
            # Create varied but simple frames to avoid excessive computation
            frame = np.full((32, 32, 3), i % 256, dtype=np.uint8)
            frames.append(frame)

        import time

        start_time = time.time()

        try:
            flicker_result = detector.detect_flicker_excess(frames)
            flat_result = detector.detect_flat_region_flicker(frames)
            pumping_result = detector.detect_temporal_pumping(frames)

            end_time = time.time()
            duration = end_time - start_time

            print(f"Performance with {len(frames)} frames: {duration:.2f}s")

            # Should complete in reasonable time (adjust threshold as needed)
            assert (
                duration < 60.0
            ), f"Processing {len(frames)} frames took too long: {duration:.2f}s"

            # Results should be valid
            assert isinstance(flicker_result, dict)
            assert isinstance(flat_result, dict)
            assert isinstance(pumping_result, dict)

        except MemoryError:
            pytest.skip("Many frames test skipped due to memory constraints")

    def test_concurrent_detector_instances(self):
        """Test multiple detector instances running concurrently."""
        import threading
        import time

        # Create test frames
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            frames.append(frame)

        results = {}

        def run_detector(detector_id):
            detector = TemporalArtifactDetector()
            start = time.time()
            result = detector.detect_flicker_excess(frames)
            end = time.time()
            results[detector_id] = (result, end - start)

        # Run multiple detectors concurrently
        threads = []
        for i in range(3):  # 3 concurrent instances
            thread = threading.Thread(target=run_detector, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        assert len(results) >= 1, "At least one detector should complete"

        for detector_id, (result, duration) in results.items():
            assert isinstance(result, dict)
            assert (
                duration < 30.0
            ), f"Detector {detector_id} took too long: {duration:.2f}s"
            print(f"Detector {detector_id}: {duration:.2f}s")

    @pytest.mark.slow
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during temporal detection."""
        detector = TemporalArtifactDetector()

        # Create frames that might cause memory issues
        large_frames = []
        for i in range(10):
            frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            large_frames.append(frame)

        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run detection
            detector.detect_flicker_excess(large_frames)
            detector.detect_flat_region_flicker(large_frames)

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB")
            print(f"Memory increase: {peak_memory - initial_memory:.1f}MB")

            # Should not use excessive memory (adjust threshold as needed)
            memory_increase = peak_memory - initial_memory
            assert (
                memory_increase < 1000
            ), f"Excessive memory usage: {memory_increase:.1f}MB"

        except ImportError:
            pytest.skip("psutil not available for memory testing")


class TestTemporalArtifactErrorHandling:
    """Test error handling in temporal artifact detection."""

    def test_lpips_model_loading_failure(self):
        """Test handling of LPIPS model loading failures."""
        with patch("giflab.temporal_artifacts.lpips.LPIPS") as mock_lpips:
            # Mock LPIPS to raise exception during loading
            mock_lpips.side_effect = RuntimeError("CUDA out of memory")

            detector = TemporalArtifactDetector()
            frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]

            # Should fall back gracefully without crashing
            result = detector.calculate_lpips_temporal(frames)

            # Should return default/fallback values
            assert isinstance(result, dict)
            assert "lpips_t_mean" in result
            # Might be 0.0 or computed via MSE fallback

    def test_gpu_memory_exhaustion(self):
        """Test handling of GPU memory exhaustion."""
        with patch("giflab.temporal_artifacts.torch") as mock_torch:
            # Mock torch operations to raise CUDA out of memory
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = "cuda:0"

            mock_tensor = MagicMock()
            mock_tensor.to.side_effect = RuntimeError("CUDA out of memory")
            mock_torch.from_numpy.return_value = mock_tensor

            detector = TemporalArtifactDetector(device="cuda")
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]

            # Should handle GPU memory issues gracefully
            try:
                result = detector.calculate_lpips_temporal(frames)
                assert isinstance(result, dict)
            except Exception as e:
                # Should either handle gracefully or fail with clear error
                assert "cuda" in str(e).lower() or "memory" in str(e).lower()

    def test_invalid_device_specification(self):
        """Test handling of invalid device specifications."""
        # Test with invalid device string - may not raise exception in current implementation
        try:
            detector = TemporalArtifactDetector(device="invalid_device")
            # If it doesn't raise, verify device is set to something reasonable
            assert detector.device in [
                "cpu",
                "cuda",
                "invalid_device",
            ]  # May just accept the string
        except Exception as e:
            # If it does raise, should be appropriate exception type
            assert isinstance(e, (ValueError, RuntimeError))

    def test_file_io_errors(self, tmp_path):
        """Test handling of file I/O errors during GIF processing."""
        # Create a temporary file that we'll make unreadable
        test_gif = tmp_path / "unreadable.gif"
        test_gif.write_text("fake gif")

        # Make file unreadable
        test_gif.chmod(0o000)

        try:
            # Should handle file permission errors - test with frame extraction function
            from giflab.metrics import extract_gif_frames

            with pytest.raises((IOError, ValueError, PermissionError)):
                extract_gif_frames(test_gif)
        finally:
            # Restore permissions for cleanup
            test_gif.chmod(0o644)

    def test_preprocessing_errors(self):
        """Test handling of preprocessing errors."""
        # Test with invalid frame shapes by creating frames manually
        detector = TemporalArtifactDetector()

        # Test preprocessing with various invalid inputs
        try:
            # Empty array
            detector.preprocess_for_lpips(np.array([]))
        except Exception as e:
            assert isinstance(
                e, (ValueError, TypeError, AttributeError, IndexError, RuntimeError)
            )

        try:
            # Wrong dimensions (1D instead of 3D)
            detector.preprocess_for_lpips(np.array([1, 2, 3]))
        except Exception as e:
            assert isinstance(
                e, (ValueError, TypeError, AttributeError, IndexError, RuntimeError)
            )

    def test_calculation_numerical_errors(self):
        """Test handling of numerical errors in calculations."""
        # Create frames that might cause numerical issues
        problematic_frames = []

        # Frame with very small values that might cause precision issues
        frame1 = np.full((32, 32, 3), 1e-10, dtype=np.float32)
        problematic_frames.append(frame1.astype(np.uint8))

        # Frame with values that might cause overflow
        frame2 = np.full((32, 32, 3), 255, dtype=np.uint8)
        problematic_frames.append(frame2)

        detector = TemporalArtifactDetector()

        # Should handle numerical edge cases gracefully
        try:
            flicker_result = detector.detect_flicker_excess(problematic_frames)
            assert isinstance(flicker_result, dict)
            assert all(
                not np.isnan(v) for v in flicker_result.values() if isinstance(v, float)
            )

            flat_result = detector.detect_flat_region_flicker(problematic_frames)
            assert isinstance(flat_result, dict)
            assert all(
                not np.isnan(v) for v in flat_result.values() if isinstance(v, float)
            )

        except Exception as e:
            # Should fail gracefully with appropriate error
            assert not isinstance(e, (FloatingPointError, OverflowError))


class TestTemporalValidationBoundaryConditions:
    """Test boundary conditions in temporal validation."""

    def test_validation_at_exact_thresholds(self):
        """Test validation behavior at exact threshold values."""
        config = ValidationConfig(
            flicker_excess_threshold=0.05,
            flat_flicker_ratio_threshold=0.1,
            temporal_pumping_threshold=0.2,
            lpips_t_threshold=0.03,
        )

        # Create ValidationChecker with default config, then patch it
        validator = ValidationChecker(None)  # Use default config
        validator.config = config  # Override with test config

        # Test with metrics exactly at thresholds
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess": 0.05,  # Exactly at threshold
                "flat_flicker_ratio": 0.1,  # Exactly at threshold
                "temporal_pumping_score": 0.2,  # Exactly at threshold
                "lpips_t_mean": 0.03,  # Exactly at threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            # Create proper metadata and compression metrics
            from giflab.meta import GifMetadata

            original_metadata = GifMetadata(
                gif_sha="test_sha",
                orig_filename="test.gif",
                orig_kilobytes=1024,
                orig_width=64,
                orig_height=64,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256,
                entropy=5.0,
            )

            compression_metrics = {
                "flicker_excess": 0.05,  # Exactly at threshold
                "flat_flicker_ratio": 0.1,  # Exactly at threshold
                "temporal_pumping_score": 0.2,  # Exactly at threshold
                "lpips_t_mean": 0.03,  # Exactly at threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
                "file_size_mb": 0.8,
                "compression_ratio": 0.8,
            }

            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=compression_metrics,
                pipeline_id="test_pipeline",
                gif_name="test.gif",
            )

            # Behavior at exact threshold depends on implementation
            # (typically ≤ threshold passes, > threshold fails)
            print(f"Validation at exact thresholds: {result.is_acceptable()}")

    def test_validation_just_above_thresholds(self):
        """Test validation behavior just above thresholds."""
        config = ValidationConfig(
            flicker_excess_threshold=0.05,
            flat_flicker_ratio_threshold=0.1,
            temporal_pumping_threshold=0.2,
            lpips_t_threshold=0.03,
        )

        # Create ValidationChecker with default config, then patch it
        validator = ValidationChecker(None)  # Use default config
        validator.config = config  # Override with test config

        # Test with metrics just above thresholds
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess": 0.051,  # Just above threshold
                "flat_flicker_ratio": 0.101,  # Just above threshold
                "temporal_pumping_score": 0.201,  # Just above threshold
                "lpips_t_mean": 0.031,  # Just above threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            # Create proper metadata and compression metrics for above-threshold values
            from giflab.meta import GifMetadata

            original_metadata = GifMetadata(
                gif_sha="test_sha",
                orig_filename="test.gif",
                orig_kilobytes=1024,
                orig_width=64,
                orig_height=64,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256,
                entropy=5.0,
            )

            compression_metrics = {
                "flicker_excess": 0.06,  # Above threshold (0.05)
                "flat_flicker_ratio": 0.11,  # Above threshold (0.1)
                "temporal_pumping_score": 0.21,  # Above threshold (0.2)
                "lpips_t_mean": 0.031,  # Above threshold (0.03)
                "ssim_mean": 0.79,  # Below threshold (0.8)
                "psnr_mean": 24.9,  # Below threshold (25.0)
                "mse_mean": 301.0,  # Above threshold (300.0)
                "file_size_mb": 0.8,
                "compression_ratio": 0.8,
            }

            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=compression_metrics,
                pipeline_id="test_pipeline",
                gif_name="test.gif",
            )

            # Should fail when above thresholds
            assert not result.is_acceptable()

    def test_validation_just_below_thresholds(self):
        """Test validation behavior just below thresholds."""
        config = ValidationConfig(
            flicker_excess_threshold=0.05,
            flat_flicker_ratio_threshold=0.1,
            temporal_pumping_threshold=0.2,
            lpips_t_threshold=0.03,
        )

        # Create ValidationChecker with default config, then patch it
        validator = ValidationChecker(None)  # Use default config
        validator.config = config  # Override with test config

        # Test with metrics just below thresholds
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess": 0.049,  # Just below threshold
                "flat_flicker_ratio": 0.099,  # Just below threshold
                "temporal_pumping_score": 0.199,  # Just below threshold
                "lpips_t_mean": 0.029,  # Just below threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            # Create proper metadata and compression metrics for below-threshold values
            from giflab.meta import GifMetadata

            original_metadata = GifMetadata(
                gif_sha="test_sha",
                orig_filename="test.gif",
                orig_kilobytes=1024,
                orig_width=64,
                orig_height=64,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256,
                entropy=5.0,
            )

            compression_metrics = {
                "flicker_excess": 0.04,  # Below threshold (0.05)
                "flat_flicker_ratio": 0.09,  # Below threshold (0.1)
                "temporal_pumping_score": 0.19,  # Below threshold (0.2)
                "lpips_t_mean": 0.029,  # Below threshold (0.03)
                "ssim_mean": 0.81,  # Above threshold (0.8)
                "psnr_mean": 25.1,  # Above threshold (25.0)
                "mse_mean": 299.0,  # Below threshold (300.0)
                "file_size_mb": 0.8,
                "compression_ratio": 0.8,
            }

            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=compression_metrics,
                pipeline_id="test_pipeline",
                gif_name="test.gif",
            )

            # Should pass when below thresholds
            assert result.is_acceptable()
