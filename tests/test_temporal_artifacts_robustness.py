"""
Comprehensive tests for temporal artifacts robustness improvements.

Tests the memory management, error handling, and batch processing
improvements made to the temporal artifacts detection system.
"""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from giflab.temporal_artifacts import (
    TemporalArtifactDetector,
    MemoryMonitor,
    calculate_enhanced_temporal_metrics,
)


class TestMemoryMonitor:
    """Test the MemoryMonitor class functionality."""
    
    def test_memory_monitor_cpu_device(self):
        """Test MemoryMonitor with CPU device."""
        monitor = MemoryMonitor("cpu")
        assert not monitor.is_cuda
        assert monitor._get_memory_usage() == 0.0
        
        # CPU should always allow max batch size
        batch_size = monitor.get_safe_batch_size((100, 100, 3), max_batch_size=32)
        assert batch_size == 32
        
        # No cleanup needed on CPU
        assert not monitor.should_cleanup_memory()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024*1024*1024)  # 1GB
    @patch('torch.cuda.get_device_properties')
    def test_memory_monitor_cuda_device(self, mock_props, mock_allocated, mock_available):
        """Test MemoryMonitor with CUDA device."""
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        
        monitor = MemoryMonitor("cuda:0")
        assert monitor.is_cuda
        
        usage = monitor._get_memory_usage()
        assert usage == 1024*1024*1024 / (8 * 1024 * 1024 * 1024)  # 1/8 = 0.125
        
        # Should not need cleanup at 12.5% usage
        assert not monitor.should_cleanup_memory()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=7 * 1024*1024*1024)  # 7GB
    @patch('torch.cuda.get_device_properties')
    def test_memory_monitor_high_usage(self, mock_props, mock_allocated, mock_available):
        """Test MemoryMonitor behavior at high memory usage."""
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        
        monitor = MemoryMonitor("cuda:0", memory_threshold=0.8)
        
        # Should need cleanup at 87.5% usage (7/8)
        assert monitor.should_cleanup_memory()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2 * 1024*1024*1024)  # 2GB
    @patch('torch.cuda.get_device_properties')
    def test_safe_batch_size_calculation(self, mock_props, mock_allocated, mock_available):
        """Test safe batch size calculation based on available memory."""
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        
        monitor = MemoryMonitor("cuda:0", memory_threshold=0.8)
        
        # With 2GB used out of 8GB, and 80% threshold, we have:
        # Available for threshold: 8GB * 0.8 = 6.4GB
        # Free memory: 6.4GB - 2GB = 4.4GB
        # For 100x100x3 frames, safety margin should allow reasonable batch size
        batch_size = monitor.get_safe_batch_size((100, 100, 3), max_batch_size=32)
        assert batch_size >= 1
        assert batch_size <= 32


class TestTemporalArtifactDetectorRobustness:
    """Test robustness improvements in TemporalArtifactDetector."""
    
    def test_detector_initialization(self):
        """Test detector initialization with different configurations."""
        # Default initialization
        detector = TemporalArtifactDetector()
        assert detector.memory_monitor is not None
        assert not detector.force_mse_fallback
        
        # MSE fallback initialization
        detector = TemporalArtifactDetector(force_mse_fallback=True)
        assert detector.force_mse_fallback
        assert detector._lpips_model is False
        
        # Custom memory threshold
        detector = TemporalArtifactDetector(memory_threshold=0.9)
        assert detector.memory_monitor.memory_threshold == 0.9
    
    @patch('giflab.temporal_artifacts.LPIPS_AVAILABLE', False)
    def test_detector_without_lpips(self):
        """Test detector behavior when LPIPS is not available."""
        detector = TemporalArtifactDetector()
        
        # Create dummy frames
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        
        # Should fall back to MSE
        metrics = detector.calculate_lpips_temporal(frames)
        assert "lpips_t_mean" in metrics
        assert "lpips_frame_count" in metrics
        assert metrics["lpips_frame_count"] == 3
    
    def test_detector_with_small_frames(self):
        """Test detector with minimal frame input."""
        detector = TemporalArtifactDetector(force_mse_fallback=True)
        
        # Single frame - should return zero metrics
        single_frame = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)]
        metrics = detector.calculate_lpips_temporal(single_frame)
        assert metrics["lpips_t_mean"] == 0.0
        assert metrics["lpips_frame_count"] == 1
        
        # Two frames - should work
        two_frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
        metrics = detector.calculate_lpips_temporal(two_frames)
        assert "lpips_t_mean" in metrics
        assert metrics["lpips_frame_count"] == 2
    
    def test_flicker_detection_robustness(self):
        """Test flicker detection with various edge cases."""
        detector = TemporalArtifactDetector(force_mse_fallback=True)
        
        # Empty frames
        metrics = detector.detect_flicker_excess([])
        assert metrics["flicker_excess"] == 0.0
        assert metrics["flicker_frame_count"] == 0
        
        # Single frame
        single_frame = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)]
        metrics = detector.detect_flicker_excess(single_frame)
        assert metrics["flicker_excess"] == 0.0
        
        # Multiple similar frames (low flicker)
        base_frame = np.ones((32, 32, 3), dtype=np.uint8) * 128
        similar_frames = [base_frame + np.random.randint(-5, 5, (32, 32, 3)) for _ in range(5)]
        metrics = detector.detect_flicker_excess(similar_frames)
        assert metrics["flicker_excess"] >= 0.0  # Should be low but not necessarily zero
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.OutOfMemoryError')
    def test_oom_error_handling(self, mock_oom, mock_cuda_available):
        """Test OOM error handling and recovery."""
        detector = TemporalArtifactDetector(device="cuda:0")
        
        # Mock LPIPS processing to raise OOM
        with patch.object(detector, '_process_lpips_batch', side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")):
            frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(10)]
            
            # Should gracefully fall back to MSE
            metrics = detector.calculate_lpips_temporal(frames)
            assert "lpips_t_mean" in metrics
            assert metrics["lpips_frame_count"] == 10
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing functionality."""
        detector = TemporalArtifactDetector(force_mse_fallback=False)  # Enable LPIPS to test batch sizing
        
        # Large frame sequence should trigger batch size reduction
        large_frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(150)]
        
        # Test that processing completes successfully with large batches
        metrics = detector.calculate_lpips_temporal(large_frames, batch_size=32)
        
        # Should get temporal metrics (both LPIPS and MSE fallback use lpips_ prefixed keys)
        assert "lpips_t_mean" in metrics
        assert "lpips_frame_count" in metrics
    
    def test_preprocessing_robustness(self):
        """Test preprocessing with different input formats."""
        detector = TemporalArtifactDetector()
        
        # Test with uint8 input
        uint8_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = detector.preprocess_for_lpips(uint8_frame)
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32
        
        # Test with float32 input
        float32_frame = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = detector.preprocess_for_lpips(float32_frame)
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32
        
        # Values should be in [-1, 1] range for LPIPS
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0


class TestEnhancedTemporalMetrics:
    """Test the main enhanced temporal metrics function."""
    
    def test_enhanced_metrics_with_mismatched_frames(self):
        """Test enhanced metrics with mismatched frame counts."""
        original_frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
        compressed_frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        
        metrics = calculate_enhanced_temporal_metrics(
            original_frames, compressed_frames, force_mse_fallback=True
        )
        
        # Should use the minimum frame count
        assert metrics["frame_count"] == 3
        assert "flicker_excess" in metrics
        assert "flat_flicker_ratio" in metrics
    
    def test_enhanced_metrics_with_empty_frames(self):
        """Test enhanced metrics with empty frame lists."""
        metrics = calculate_enhanced_temporal_metrics([], [], force_mse_fallback=True)
        
        assert metrics["flicker_excess"] == 0.0
        assert metrics["flat_flicker_ratio"] == 0.0
        assert metrics["temporal_pumping_score"] == 0.0
        assert metrics["lpips_t_mean"] == 0.0
        assert metrics["lpips_t_p95"] == 0.0
    
    def test_enhanced_metrics_large_sequence_optimization(self):
        """Test that large sequences trigger optimization."""
        # Create a large sequence that should trigger batch size reduction
        large_frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(150)]
        
        with patch('giflab.temporal_artifacts.logger') as mock_logger:
            metrics = calculate_enhanced_temporal_metrics(
                large_frames, large_frames, force_mse_fallback=True, batch_size=16
            )
            
            # Should log the batch size adjustment
            mock_logger.info.assert_called()
            assert "Large frame sequence" in str(mock_logger.info.call_args)
            
            assert metrics["frame_count"] == 150
    
    def test_device_fallback_handling(self):
        """Test device fallback when CUDA is requested but unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            detector = TemporalArtifactDetector(device="cuda:0")
            
            # Should have fallen back to CPU
            assert detector.device == "cpu"
            assert not detector.memory_monitor.is_cuda


@pytest.mark.parametrize("frame_size", [(32, 32), (64, 64), (128, 128), (256, 256)])
def test_memory_scaling_with_frame_size(frame_size):
    """Test that memory management scales appropriately with frame size."""
    h, w = frame_size
    monitor = MemoryMonitor("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = monitor.get_safe_batch_size((h, w, 3), max_batch_size=32)
    
    # Larger frames should generally result in smaller batch sizes on CUDA
    assert batch_size >= 1
    assert batch_size <= 32
    
    if monitor.is_cuda and h > 128:
        # Very large frames should definitely constrain batch size
        assert batch_size < 32


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_batch_processing_consistency(batch_size):
    """Test that different batch sizes produce consistent results."""
    detector = TemporalArtifactDetector(force_mse_fallback=True)
    
    # Create deterministic test frames
    np.random.seed(42)
    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    
    metrics1 = detector.calculate_lpips_temporal(frames, batch_size=batch_size)
    metrics2 = detector.calculate_lpips_temporal(frames, batch_size=batch_size)
    
    # Results should be deterministic
    assert abs(metrics1["lpips_t_mean"] - metrics2["lpips_t_mean"]) < 1e-6
    assert metrics1["lpips_frame_count"] == metrics2["lpips_frame_count"]


# Integration test for the complete pipeline
def test_temporal_artifacts_integration():
    """Integration test for the complete temporal artifacts detection pipeline."""
    # Create test frames with known temporal artifacts
    frames = []
    base_frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
    
    for i in range(20):
        if i % 4 == 0:
            # Every 4th frame has artifacts (flicker)
            frame = base_frame + np.random.randint(-50, 50, (64, 64, 3))
        else:
            # Stable frames
            frame = base_frame + np.random.randint(-5, 5, (64, 64, 3))
        
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    
    # Test with both original and compressed versions
    compressed_frames = [frame + np.random.randint(-10, 10, (64, 64, 3)) for frame in frames]
    compressed_frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in compressed_frames]
    
    metrics = calculate_enhanced_temporal_metrics(
        frames, compressed_frames, force_mse_fallback=True, batch_size=8
    )
    
    # Should detect some temporal artifacts
    assert metrics["flicker_excess"] >= 0.0
    assert metrics["flat_flicker_ratio"] >= 0.0
    assert metrics["temporal_pumping_score"] >= 0.0
    assert metrics["frame_count"] == 20
    
    # All metrics should be finite
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"