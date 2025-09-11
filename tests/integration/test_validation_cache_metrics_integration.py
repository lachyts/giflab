"""Integration tests for ValidationCache with metrics calculations."""

import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from giflab.caching.validation_cache import ValidationCache, get_validation_cache, reset_validation_cache
from giflab.caching.metrics_integration import (
    calculate_ms_ssim_cached,
    calculate_ssim_cached,
    calculate_lpips_cached,
    calculate_gradient_color_cached,
    calculate_ssimulacra2_cached,
    integrate_validation_cache_with_metrics,
)


class TestMetricsIntegration:
    """Test integration of ValidationCache with metric calculations."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample frame arrays for testing."""
        np.random.seed(42)
        frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frames.append(frame)
        return frames
    
    @pytest.fixture
    def cache_config(self, temp_cache_dir):
        """Mock cache configuration."""
        return {
            "enabled": True,
            "memory_limit_mb": 10,
            "disk_path": temp_cache_dir / "test_cache.db",
            "disk_limit_mb": 50,
            "ttl_seconds": 3600,
            "cache_ssim": True,
            "cache_ms_ssim": True,
            "cache_lpips": True,
            "cache_gradient_color": True,
            "cache_ssimulacra2": True,
        }
    
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset cache singleton before each test."""
        reset_validation_cache()
        yield
        reset_validation_cache()
    
    def test_ms_ssim_caching(self, sample_frames, cache_config):
        """Test MS-SSIM calculation with caching."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                # Mock the actual MS-SSIM calculation
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    # First call should calculate
                    result1 = calculate_ms_ssim_cached(frame1, frame2, scales=5)
                    assert result1 == 0.95
                    assert mock_ms_ssim.call_count == 1
                    
                    # Second call should use cache
                    result2 = calculate_ms_ssim_cached(frame1, frame2, scales=5)
                    assert result2 == 0.95
                    assert mock_ms_ssim.call_count == 1  # No additional call
                    
                    # Different parameters should trigger new calculation
                    result3 = calculate_ms_ssim_cached(frame1, frame2, scales=3)
                    assert mock_ms_ssim.call_count == 2
    
    def test_ssim_caching(self, sample_frames, cache_config):
        """Test SSIM calculation with caching."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                # Mock the actual SSIM calculation
                with patch("skimage.metrics.structural_similarity") as mock_ssim:
                    mock_ssim.return_value = 0.88
                    
                    # First call should calculate
                    result1 = calculate_ssim_cached(frame1, frame2)
                    assert result1 == 0.88
                    assert mock_ssim.call_count == 1
                    
                    # Second call should use cache
                    result2 = calculate_ssim_cached(frame1, frame2)
                    assert result2 == 0.88
                    assert mock_ssim.call_count == 1  # No additional call
    
    def test_lpips_caching(self, sample_frames, cache_config):
        """Test LPIPS calculation with caching."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                # Mock the actual LPIPS calculation
                with patch("giflab.deep_perceptual_metrics.calculate_lpips_frames") as mock_lpips:
                    mock_lpips.return_value = [0.12]
                    
                    # First call should calculate
                    result1 = calculate_lpips_cached(frame1, frame2, net="alex")
                    assert result1 == 0.12
                    assert mock_lpips.call_count == 1
                    
                    # Second call should use cache
                    result2 = calculate_lpips_cached(frame1, frame2, net="alex")
                    assert result2 == 0.12
                    assert mock_lpips.call_count == 1  # No additional call
                    
                    # Different network should trigger new calculation
                    mock_lpips.return_value = [0.15]
                    result3 = calculate_lpips_cached(frame1, frame2, net="vgg")
                    assert result3 == 0.15
                    assert mock_lpips.call_count == 2
    
    def test_gradient_color_caching(self, sample_frames, cache_config):
        """Test gradient color metrics caching."""
        frames1 = sample_frames[:3]
        frames2 = sample_frames[1:4]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                # Mock the actual calculation
                mock_result = {
                    "gradient_score": 0.85,
                    "color_artifacts": 0.12,
                    "frame_scores": [0.8, 0.85, 0.9],
                }
                
                with patch("giflab.gradient_color_artifacts.calculate_gradient_color_metrics") as mock_calc:
                    mock_calc.return_value = mock_result
                    
                    # First call should calculate
                    result1 = calculate_gradient_color_cached(frames1, frames2)
                    assert result1 == mock_result
                    assert mock_calc.call_count == 1
                    
                    # Second call should use cache
                    result2 = calculate_gradient_color_cached(frames1, frames2)
                    assert result2 == mock_result
                    assert mock_calc.call_count == 1  # No additional call
    
    def test_ssimulacra2_caching(self, sample_frames, cache_config):
        """Test SSIMulacra2 metrics caching."""
        frames1 = sample_frames[:3]
        frames2 = sample_frames[1:4]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                # Mock the actual calculation
                mock_result = {
                    "mean_score": 75.5,
                    "min_score": 70.0,
                    "max_score": 80.0,
                    "std_score": 2.5,
                }
                
                with patch("giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics") as mock_calc:
                    mock_calc.return_value = mock_result
                    
                    # First call should calculate
                    result1 = calculate_ssimulacra2_cached(frames1, frames2)
                    assert result1 == mock_result
                    assert mock_calc.call_count == 1
                    
                    # Second call should use cache
                    result2 = calculate_ssimulacra2_cached(frames1, frames2)
                    assert result2 == mock_result
                    assert mock_calc.call_count == 1  # No additional call
    
    def test_cache_disabled_in_config(self, sample_frames):
        """Test that caching can be disabled via configuration."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        disabled_config = {
            "enabled": False,  # Cache disabled
            "cache_ms_ssim": True,
        }
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", disabled_config):
            with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                mock_ms_ssim.return_value = 0.95
                
                # Both calls should calculate (no caching)
                result1 = calculate_ms_ssim_cached(frame1, frame2)
                assert result1 == 0.95
                assert mock_ms_ssim.call_count == 1
                
                result2 = calculate_ms_ssim_cached(frame1, frame2)
                assert result2 == 0.95
                assert mock_ms_ssim.call_count == 2  # Called again
    
    def test_metric_specific_cache_disable(self, sample_frames):
        """Test disabling cache for specific metrics."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        config = {
            "enabled": True,
            "cache_ms_ssim": False,  # MS-SSIM caching disabled
            "cache_ssim": True,  # SSIM caching enabled
        }
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", config):
                # MS-SSIM should not cache
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    result1 = calculate_ms_ssim_cached(frame1, frame2)
                    result2 = calculate_ms_ssim_cached(frame1, frame2)
                    assert mock_ms_ssim.call_count == 2  # Called twice
                
                # SSIM should cache
                with patch("skimage.metrics.structural_similarity") as mock_ssim:
                    mock_ssim.return_value = 0.88
                    
                    result1 = calculate_ssim_cached(frame1, frame2)
                    result2 = calculate_ssim_cached(frame1, frame2)
                    assert mock_ssim.call_count == 1  # Called once
    
    def test_frame_indices_in_caching(self, sample_frames, cache_config):
        """Test that frame indices are properly used in cache keys."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    # Same frames but different indices should calculate separately
                    result1 = calculate_ms_ssim_cached(
                        frame1, frame2, frame_indices=(0, 1)
                    )
                    assert mock_ms_ssim.call_count == 1
                    
                    result2 = calculate_ms_ssim_cached(
                        frame1, frame2, frame_indices=(5, 6)
                    )
                    assert mock_ms_ssim.call_count == 2  # New calculation
                    
                    # Same indices should use cache
                    result3 = calculate_ms_ssim_cached(
                        frame1, frame2, frame_indices=(0, 1)
                    )
                    assert mock_ms_ssim.call_count == 2  # No new calculation
    
    def test_integrate_validation_cache(self, cache_config):
        """Test the integrate_validation_cache_with_metrics function."""
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            # Mock the metrics module
            mock_metrics = MagicMock()
            mock_metrics.calculate_ms_ssim = MagicMock(return_value=0.95)
            
            with patch("giflab.caching.metrics_integration.metrics", mock_metrics):
                # Call integration function
                integrate_validation_cache_with_metrics()
                
                # Verify that the function was wrapped
                assert hasattr(mock_metrics, "_original_calculate_ms_ssim")
    
    def test_cache_performance_improvement(self, sample_frames, cache_config):
        """Test that caching provides performance improvement."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                # Mock with delay to simulate computation
                def slow_calculation(f1, f2, scales=5):
                    time.sleep(0.01)  # 10ms delay
                    return 0.95
                
                with patch("giflab.metrics.calculate_ms_ssim", side_effect=slow_calculation):
                    # First call (with calculation)
                    start1 = time.time()
                    result1 = calculate_ms_ssim_cached(frame1, frame2)
                    time1 = time.time() - start1
                    
                    # Second call (from cache)
                    start2 = time.time()
                    result2 = calculate_ms_ssim_cached(frame1, frame2)
                    time2 = time.time() - start2
                    
                    # Cache should be significantly faster
                    assert time2 < time1 * 0.5  # At least 2x faster
                    assert result1 == result2 == 0.95
    
    def test_cache_invalidation_between_runs(self, sample_frames, temp_cache_dir):
        """Test that cache persists between application runs."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        cache_path = temp_cache_dir / "persist_test.db"
        
        config = {
            "enabled": True,
            "disk_path": cache_path,
            "cache_ms_ssim": True,
        }
        
        # First "run" - calculate and cache
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", config):
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    result1 = calculate_ms_ssim_cached(frame1, frame2)
                    assert mock_ms_ssim.call_count == 1
        
        # Reset cache singleton to simulate new run
        reset_validation_cache()
        
        # Second "run" - should load from disk cache
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", config):
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    result2 = calculate_ms_ssim_cached(frame1, frame2)
                    assert result2 == 0.95
                    assert mock_ms_ssim.call_count == 0  # Loaded from cache
    
    def test_concurrent_cache_access(self, sample_frames, cache_config):
        """Test concurrent access to cached metrics."""
        import threading
        
        frames = sample_frames
        results = []
        call_counts = []
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    def worker(frame_idx):
                        # Each thread calculates same metric
                        result = calculate_ms_ssim_cached(
                            frames[0], frames[1], scales=5
                        )
                        results.append(result)
                        call_counts.append(mock_ms_ssim.call_count)
                    
                    # Run multiple threads
                    threads = []
                    for i in range(10):
                        t = threading.Thread(target=worker, args=(i,))
                        threads.append(t)
                        t.start()
                    
                    for t in threads:
                        t.join()
                    
                    # All should get same result
                    assert all(r == 0.95 for r in results)
                    
                    # Should only calculate once (rest from cache)
                    assert mock_ms_ssim.call_count == 1
    
    @pytest.mark.parametrize("use_cache", [True, False])
    def test_cache_flag_parameter(self, sample_frames, cache_config, use_cache):
        """Test that use_validation_cache parameter works correctly."""
        frame1, frame2 = sample_frames[0], sample_frames[1]
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", cache_config):
            with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
                with patch("giflab.metrics.calculate_ms_ssim") as mock_ms_ssim:
                    mock_ms_ssim.return_value = 0.95
                    
                    # First call
                    result1 = calculate_ms_ssim_cached(
                        frame1, frame2, use_validation_cache=use_cache
                    )
                    assert mock_ms_ssim.call_count == 1
                    
                    # Second call
                    result2 = calculate_ms_ssim_cached(
                        frame1, frame2, use_validation_cache=use_cache
                    )
                    
                    if use_cache:
                        # Should use cache
                        assert mock_ms_ssim.call_count == 1
                    else:
                        # Should calculate again
                        assert mock_ms_ssim.call_count == 2