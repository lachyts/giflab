"""Integration tests for ResizedFrameCache with metrics calculations."""

import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from giflab.caching.resized_frame_cache import get_resize_cache
from giflab.deep_perceptual_metrics import DeepPerceptualValidator
from giflab.metrics import calculate_ms_ssim, _resize_if_needed


class TestResizeCacheIntegrationWithMetrics:
    """Integration tests for resize cache with real metrics."""
    
    @pytest.fixture
    def test_frames(self):
        """Create test frames with different sizes."""
        frames = [
            np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8),  # Large
            np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8),  # Medium
            np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8),  # Small
        ]
        return frames
    
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        cache = get_resize_cache()
        cache.clear()
        yield
        cache.clear()
    
    def test_resize_if_needed_uses_cache(self, test_frames):
        """Test that _resize_if_needed function uses the resize cache."""
        frame1, frame2 = test_frames[0], test_frames[1]
        
        # Get initial cache stats
        cache = get_resize_cache()
        initial_stats = cache.get_stats()
        
        # First resize - should create cache entries
        resized1, resized2 = _resize_if_needed(frame1, frame2, use_cache=True)
        
        stats_after_first = cache.get_stats()
        assert stats_after_first["misses"] > initial_stats["misses"]
        
        # Second resize with same frames - should hit cache
        resized1_2, resized2_2 = _resize_if_needed(frame1, frame2, use_cache=True)
        
        stats_after_second = cache.get_stats()
        assert stats_after_second["hits"] > stats_after_first["hits"]
        
        # Results should be identical
        np.testing.assert_array_equal(resized1, resized1_2)
        np.testing.assert_array_equal(resized2, resized2_2)
    
    def test_resize_if_needed_without_cache(self, test_frames):
        """Test that _resize_if_needed can work without cache."""
        frame1, frame2 = test_frames[0], test_frames[1]
        
        cache = get_resize_cache()
        initial_stats = cache.get_stats()
        
        # Resize without cache
        resized1, resized2 = _resize_if_needed(frame1, frame2, use_cache=False)
        
        # Cache stats should not change
        final_stats = cache.get_stats()
        assert final_stats["hits"] == initial_stats["hits"]
        assert final_stats["misses"] == initial_stats["misses"]
        
        # Results should still be correct
        target_h = min(frame1.shape[0], frame2.shape[0])
        target_w = min(frame1.shape[1], frame2.shape[1])
        assert resized1.shape[:2] == (target_h, target_w)
        assert resized2.shape[:2] == (target_h, target_w)
    
    def test_ms_ssim_uses_cache(self, test_frames):
        """Test that MS-SSIM calculation uses the resize cache."""
        frame1, frame2 = test_frames[0], test_frames[1]
        
        cache = get_resize_cache()
        initial_stats = cache.get_stats()
        
        # First MS-SSIM calculation
        ms_ssim1 = calculate_ms_ssim(frame1, frame2, scales=3, use_cache=True)
        
        stats_after_first = cache.get_stats()
        # Should have multiple cache misses for different scales
        assert stats_after_first["misses"] > initial_stats["misses"]
        
        # Second MS-SSIM calculation with same frames
        ms_ssim2 = calculate_ms_ssim(frame1, frame2, scales=3, use_cache=True)
        
        stats_after_second = cache.get_stats()
        # Should have cache hits this time
        assert stats_after_second["hits"] > stats_after_first["hits"]
        
        # Results should be very close (floating point comparison)
        assert abs(ms_ssim1 - ms_ssim2) < 1e-6
    
    def test_ms_ssim_multi_scale_caching(self):
        """Test that MS-SSIM caches intermediate scale resizes."""
        # Create frames that will be downsampled multiple times
        frame1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        cache = get_resize_cache()
        cache.clear()
        
        # Calculate MS-SSIM with 4 scales
        ms_ssim = calculate_ms_ssim(frame1, frame2, scales=4, use_cache=True)
        
        stats = cache.get_stats()
        # Should have cached multiple scales (256x256, 128x128, 64x64 for each frame)
        assert stats["entries"] >= 6  # At least 3 scales * 2 frames
        assert stats["misses"] >= 6
    
    @pytest.mark.skipif(
        not pytest.importorskip("lpips", reason="LPIPS not available"),
        reason="LPIPS required for this test"
    )
    def test_lpips_downscale_uses_cache(self):
        """Test that LPIPS downscaling uses the resize cache."""
        # Create large frames that will be downscaled
        original_frames = [
            np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8) for _ in range(3)
        ]
        compressed_frames = [
            np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8) for _ in range(3)
        ]
        
        validator = DeepPerceptualValidator(
            downscale_size=512,
            use_resize_cache=True
        )
        
        cache = get_resize_cache()
        cache.clear()
        
        # First calculation - cache misses
        metrics1 = validator.calculate_deep_perceptual_metrics(
            original_frames[:2], compressed_frames[:2]
        )
        
        stats_after_first = cache.get_stats()
        assert stats_after_first["misses"] > 0
        
        # Second calculation with overlapping frames - should have cache hits
        metrics2 = validator.calculate_deep_perceptual_metrics(
            original_frames[1:], compressed_frames[1:]
        )
        
        stats_after_second = cache.get_stats()
        # Should have some cache hits for the overlapping frame
        assert stats_after_second["hits"] > 0
    
    def test_cache_performance_improvement(self, test_frames):
        """Test that caching provides measurable performance improvement."""
        frame1, frame2 = test_frames[0], test_frames[1]
        
        cache = get_resize_cache()
        cache.clear()
        
        # Time uncached operations
        start = time.perf_counter()
        for _ in range(10):
            _resize_if_needed(frame1, frame2, use_cache=False)
        uncached_time = time.perf_counter() - start
        
        # Prime the cache
        _resize_if_needed(frame1, frame2, use_cache=True)
        
        # Time cached operations
        start = time.perf_counter()
        for _ in range(10):
            _resize_if_needed(frame1, frame2, use_cache=True)
        cached_time = time.perf_counter() - start
        
        # Cached should be faster (at least 2x)
        assert cached_time < uncached_time / 2
    
    def test_concurrent_metric_calculations(self):
        """Test that concurrent metric calculations work correctly with cache."""
        import threading
        
        frames = [
            np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8) for _ in range(4)
        ]
        
        results = []
        
        def calculate_metrics(f1, f2):
            # Calculate both regular resize and MS-SSIM
            resized = _resize_if_needed(f1, f2, use_cache=True)
            ms_ssim = calculate_ms_ssim(f1, f2, scales=2, use_cache=True)
            results.append((resized, ms_ssim))
        
        # Run multiple calculations concurrently
        threads = []
        for i in range(0, len(frames), 2):
            t = threading.Thread(
                target=calculate_metrics,
                args=(frames[i], frames[i+1])
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All calculations should complete successfully
        assert len(results) == len(frames) // 2
        
        # Cache should have been used
        cache = get_resize_cache()
        stats = cache.get_stats()
        assert stats["entries"] > 0
    
    @patch('giflab.config.FRAME_CACHE', {
        'resize_cache_enabled': False
    })
    def test_metrics_work_with_cache_disabled(self, test_frames):
        """Test that metrics still work when resize cache is disabled globally."""
        frame1, frame2 = test_frames[0], test_frames[1]
        
        # These should still work even with cache disabled
        resized1, resized2 = _resize_if_needed(frame1, frame2)
        ms_ssim = calculate_ms_ssim(frame1, frame2)
        
        # Results should be valid
        assert resized1.shape == resized2.shape
        assert 0 <= ms_ssim <= 1
    
    def test_cache_with_different_interpolation_methods(self):
        """Test that different interpolation methods are cached separately."""
        frame = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        target_size = (200, 200)
        
        cache = get_resize_cache()
        cache.clear()
        
        # Use different interpolation methods
        from giflab.caching.resized_frame_cache import resize_frame_cached
        
        result_area = resize_frame_cached(frame, target_size, cv2.INTER_AREA)
        result_lanczos = resize_frame_cached(frame, target_size, cv2.INTER_LANCZOS4)
        result_cubic = resize_frame_cached(frame, target_size, cv2.INTER_CUBIC)
        
        stats = cache.get_stats()
        # Should have 3 different cache entries
        assert stats["entries"] == 3
        assert stats["misses"] == 3
        
        # Requesting again should hit cache
        resize_frame_cached(frame, target_size, cv2.INTER_AREA)
        resize_frame_cached(frame, target_size, cv2.INTER_LANCZOS4)
        resize_frame_cached(frame, target_size, cv2.INTER_CUBIC)
        
        final_stats = cache.get_stats()
        assert final_stats["hits"] == 3
    
    def test_cache_memory_efficiency(self):
        """Test that cache memory usage is efficient."""
        # Create frames of different sizes
        frames = [
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            for size in [100, 200, 300, 400, 500]
        ]
        
        cache = get_resize_cache()
        cache.clear()
        
        # Resize all frames to same size
        target_size = (150, 150)
        for frame in frames:
            from giflab.caching.resized_frame_cache import resize_frame_cached
            resize_frame_cached(frame, target_size, cv2.INTER_AREA)
        
        stats = cache.get_stats()
        
        # Calculate expected memory usage
        expected_memory_per_frame = 150 * 150 * 3  # bytes
        expected_total_mb = (expected_memory_per_frame * len(frames)) / (1024 * 1024)
        
        # Actual memory should be close to expected (within 20%)
        assert abs(stats["memory_mb"] - expected_total_mb) / expected_total_mb < 0.2
        
        # Should have correct number of entries
        assert stats["entries"] == len(frames)