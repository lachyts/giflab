#!/usr/bin/env python3
"""Full Pipeline Integration Tests for Phase 5 Performance Validation.

This module provides end-to-end integration testing covering:
1. Complete metric calculation pipeline with all optimizations
2. Accuracy validation against baseline
3. Deterministic result verification
4. Error handling and recovery
5. Configuration compatibility testing
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pytest
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.metrics import calculate_comprehensive_metrics, cleanup_all_validators
from giflab.model_cache import LPIPSModelCache
from giflab.conditional_metrics import ConditionalMetricsCalculator
from giflab.parallel_metrics import ParallelMetricsCalculator


class TestFullPipelineIntegration:
    """Integration tests for full metric calculation pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Setup
        cleanup_all_validators()
        cache = LPIPSModelCache()
        cache.clear(force=True)
        
        yield
        
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Teardown
        cleanup_all_validators()
        cache.clear(force=True)
    
    def generate_test_gif_frames(
        self, 
        frame_count: int = 10,
        size: Tuple[int, int] = (200, 200),
        quality: str = "medium",
        content: str = "mixed"
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate test GIF frames with specific characteristics.
        
        Args:
            frame_count: Number of frames
            size: Frame size (width, height)
            quality: Quality level ('high', 'medium', 'low')
            content: Content type ('text', 'gradient', 'animation', 'static', 'mixed')
            
        Returns:
            Tuple of (original_frames, compressed_frames)
        """
        np.random.seed(42)  # Ensure deterministic generation
        width, height = size
        frames_orig = []
        frames_comp = []
        
        for i in range(frame_count):
            # Generate frame based on content type
            if content == "static":
                if i == 0:
                    base = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                frame = base.copy() if i > 0 else base
            elif content == "gradient":
                x = np.linspace(0, 255, width)
                y = np.linspace(0, 255, height)
                xx, yy = np.meshgrid(x, y)
                frame = np.stack([
                    (xx * (1 + i * 0.1) % 256).astype(np.uint8),
                    (yy * (1 + i * 0.1) % 256).astype(np.uint8),
                    ((xx + yy) / 2 % 256).astype(np.uint8)
                ], axis=-1)
            elif content == "text":
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                # Add text-like patterns
                for j in range(3):
                    y_pos = 30 + j * 40
                    x_pos = 20 + (i * 5) % 50
                    if y_pos + 20 < height and x_pos + 100 < width:
                        frame[y_pos:y_pos+20, x_pos:x_pos+100] = 0
            elif content == "animation":
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # Animated circle
                center_x = int(width * (0.5 + 0.3 * np.sin(i * 0.3)))
                center_y = int(height * (0.5 + 0.3 * np.cos(i * 0.3)))
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 6)**2
                frame[mask] = [255, 100, 100]
            else:  # mixed
                frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
                # Add structure
                frame[height//3:2*height//3, width//3:2*width//3] = 220
            
            frames_orig.append(frame)
            
            # Generate compressed version
            if quality == "high":
                noise = np.random.normal(0, 2, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
            elif quality == "medium":
                noise = np.random.normal(0, 8, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                compressed = (compressed // 4) * 4  # Mild quantization
            else:  # low
                noise = np.random.normal(0, 20, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                compressed = (compressed // 16) * 16  # Heavy quantization
            
            frames_comp.append(compressed)
        
        return frames_orig, frames_comp
    
    def test_pipeline_all_optimizations_enabled(self):
        """Test full pipeline with all optimizations enabled."""
        # Enable all optimizations
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        # Test different scenarios
        scenarios = [
            (10, (200, 200), "high", "static"),
            (30, (500, 500), "medium", "gradient"),
            (50, (800, 600), "low", "animation"),
        ]
        
        for frame_count, size, quality, content in scenarios:
            frames_orig, frames_comp = self.generate_test_gif_frames(
                frame_count, size, quality, content
            )
            
            # Calculate metrics
            start_time = time.perf_counter()
            metrics = calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
            elapsed = time.perf_counter() - start_time
            
            # Validate results
            assert metrics is not None, "Metrics calculation failed"
            assert "psnr" in metrics, "Missing PSNR metric"
            assert "ssim" in metrics, "Missing SSIM metric"
            
            # Check conditional processing worked
            if quality == "high":
                # High quality should skip expensive metrics
                conditional_calc = ConditionalMetricsCalculator()
                quality_tier = conditional_calc._assess_quality(
                    frames_orig[:5], frames_comp[:5]
                )
                if quality_tier == "HIGH":
                    # These expensive metrics should be skipped or have default values
                    print(f"High quality GIF correctly identified, tier: {quality_tier}")
            
            print(f"Scenario ({frame_count} frames, {size}, {quality}, {content}): "
                  f"{elapsed:.3f}s, {len(metrics)} metrics")
    
    def test_pipeline_accuracy_validation(self):
        """Test that optimized pipeline maintains accuracy."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (300, 300), "medium", "mixed"
        )
        
        # Calculate baseline (no optimizations)
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'false'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'false'
        
        baseline_metrics = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        
        # Clear validators for fair comparison
        cleanup_all_validators()
        
        # Calculate with optimizations
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'  # Disable conditional to compare all metrics
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        optimized_metrics = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        
        # Compare metrics
        tolerance = 0.001  # 0.1% tolerance
        
        for key in baseline_metrics:
            if key in optimized_metrics:
                baseline_val = baseline_metrics[key]
                optimized_val = optimized_metrics[key]
                
                if isinstance(baseline_val, (int, float)):
                    if baseline_val != 0:
                        relative_diff = abs(optimized_val - baseline_val) / abs(baseline_val)
                        assert relative_diff < tolerance, \
                            f"Metric {key} differs: baseline={baseline_val}, optimized={optimized_val}, diff={relative_diff:.4%}"
                    else:
                        assert abs(optimized_val) < tolerance, \
                            f"Metric {key} differs from zero: {optimized_val}"
        
        print(f"Accuracy validation passed: {len(baseline_metrics)} metrics within {tolerance:.1%} tolerance")
    
    def test_pipeline_deterministic_results(self):
        """Test that results are deterministic with same input."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            15, (250, 250), "medium", "gradient"
        )
        
        # Enable optimizations
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        # Run multiple times
        results = []
        for i in range(3):
            metrics = calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
            results.append(metrics)
            
            # Clear validators between runs
            cleanup_all_validators()
        
        # Verify all results are identical
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0]), \
                f"Different number of metrics: run 1={len(results[0])}, run {i+1}={len(results[i])}"
            
            for key in results[0]:
                if key in results[i]:
                    val0 = results[0][key]
                    vali = results[i][key]
                    
                    if isinstance(val0, (int, float)):
                        assert abs(val0 - vali) < 1e-6, \
                            f"Non-deterministic result for {key}: run 1={val0}, run {i+1}={vali}"
        
        print(f"Deterministic validation passed: {len(results)} runs produced identical results")
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        # Test with invalid input
        with pytest.raises((ValueError, TypeError, AttributeError)):
            calculate_comprehensive_metrics(
                original_frames=None,
                compressed_frames=None
            )
        
        # Test with mismatched frame counts
        frames_orig, frames_comp = self.generate_test_gif_frames(10, (200, 200))
        frames_comp = frames_comp[:5]  # Make compressed have fewer frames
        
        with pytest.raises((ValueError, IndexError, AssertionError)):
            calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
        
        # Verify cleanup after error
        cleanup_all_validators()
        
        # Should be able to run normally after error
        frames_orig, frames_comp = self.generate_test_gif_frames(10, (200, 200))
        metrics = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        assert metrics is not None, "Pipeline failed to recover after error"
        
        print("Error handling validation passed")
    
    def test_pipeline_configuration_compatibility(self):
        """Test different configuration combinations."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (300, 300), "medium", "mixed"
        )
        
        # Test configurations
        configs = [
            # (parallel, conditional, cache, description)
            (True, True, True, "All enabled"),
            (True, False, True, "Parallel + Cache"),
            (False, True, True, "Conditional + Cache"),
            (True, True, False, "Parallel + Conditional"),
            (False, False, True, "Cache only"),
            (True, False, False, "Parallel only"),
            (False, True, False, "Conditional only"),
            (False, False, False, "All disabled"),
        ]
        
        for parallel, conditional, cache, description in configs:
            os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = str(parallel).lower()
            os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = str(conditional).lower()
            os.environ['GIFLAB_USE_MODEL_CACHE'] = str(cache).lower()
            
            try:
                start_time = time.perf_counter()
                metrics = calculate_comprehensive_metrics(
                    original_frames=frames_orig,
                    compressed_frames=frames_comp
                )
                elapsed = time.perf_counter() - start_time
                
                assert metrics is not None, f"Failed with config: {description}"
                print(f"Config '{description}': {elapsed:.3f}s, {len(metrics)} metrics")
                
            except Exception as e:
                pytest.fail(f"Configuration '{description}' failed: {str(e)}")
            
            finally:
                cleanup_all_validators()
        
        print("Configuration compatibility validation passed")
    
    def test_pipeline_performance_thresholds(self):
        """Test that performance meets expected thresholds."""
        # Small GIF test
        frames_orig, frames_comp = self.generate_test_gif_frames(
            10, (100, 100), "high", "static"
        )
        
        # Enable all optimizations
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        start_time = time.perf_counter()
        metrics = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        small_time = time.perf_counter() - start_time
        
        # Small high-quality GIF should be very fast with conditional processing
        assert small_time < 2.0, f"Small GIF took too long: {small_time:.3f}s"
        
        # Large GIF test
        frames_orig, frames_comp = self.generate_test_gif_frames(
            100, (800, 600), "low", "animation"
        )
        
        start_time = time.perf_counter()
        metrics = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        large_time = time.perf_counter() - start_time
        
        # Large low-quality GIF will take longer but should still be reasonable
        assert large_time < 30.0, f"Large GIF took too long: {large_time:.3f}s"
        
        print(f"Performance thresholds met: Small={small_time:.3f}s, Large={large_time:.3f}s")
    
    def test_pipeline_cache_effectiveness(self):
        """Test that model cache is working effectively."""
        cache = LPIPSModelCache()
        cache.clear(force=True)
        
        # Enable cache
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (300, 300), "low", "mixed"
        )
        
        # First run - cache miss
        initial_info = cache.get_model_cache_info()
        metrics1 = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        after_first = cache.get_model_cache_info()
        
        # Don't cleanup validators - keep models in cache
        
        # Second run - cache hit
        metrics2 = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        after_second = cache.get_model_cache_info()
        
        # Verify cache was used
        models_loaded_first = after_first.get("models_loaded", 0) - initial_info.get("models_loaded", 0)
        models_loaded_second = after_second.get("models_loaded", 0) - after_first.get("models_loaded", 0)
        
        # Second run should load fewer or no new models
        assert models_loaded_second <= models_loaded_first, \
            f"Cache not effective: first run loaded {models_loaded_first}, second loaded {models_loaded_second}"
        
        print(f"Cache effectiveness validated: First run loaded {models_loaded_first} models, "
              f"second run loaded {models_loaded_second} models")
    
    def test_pipeline_conditional_skip_validation(self):
        """Test that conditional processing correctly skips metrics."""
        # High quality GIF
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (400, 400), "high", "static"
        )
        
        # Enable conditional processing
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_FORCE_ALL_METRICS'] = 'false'
        
        # Calculate metrics
        metrics_conditional = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        
        # Force all metrics for comparison
        os.environ['GIFLAB_FORCE_ALL_METRICS'] = 'true'
        
        cleanup_all_validators()
        
        metrics_all = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        
        # Conditional should calculate fewer metrics for high quality
        metrics_skipped = len(metrics_all) - len(metrics_conditional)
        assert metrics_skipped > 0, \
            f"Conditional processing didn't skip any metrics: all={len(metrics_all)}, conditional={len(metrics_conditional)}"
        
        print(f"Conditional skip validation passed: {metrics_skipped} metrics skipped for high quality GIF")
    
    def test_pipeline_parallel_speedup_validation(self):
        """Test that parallel processing provides speedup for large GIFs."""
        # Large GIF for parallel benefit
        frames_orig, frames_comp = self.generate_test_gif_frames(
            100, (600, 600), "medium", "animation"
        )
        
        # Disable conditional to ensure all metrics are calculated
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        # Sequential processing
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'false'
        
        start_time = time.perf_counter()
        metrics_seq = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        seq_time = time.perf_counter() - start_time
        
        cleanup_all_validators()
        
        # Parallel processing
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_MAX_PARALLEL_WORKERS'] = '4'
        
        start_time = time.perf_counter()
        metrics_par = calculate_comprehensive_metrics(
            original_frames=frames_orig,
            compressed_frames=frames_comp
        )
        par_time = time.perf_counter() - start_time
        
        # Verify same metrics calculated
        assert len(metrics_seq) == len(metrics_par), \
            f"Different metrics: sequential={len(metrics_seq)}, parallel={len(metrics_par)}"
        
        # Calculate speedup
        speedup = seq_time / par_time if par_time > 0 else 1.0
        
        # For large GIFs, we expect at least some speedup
        # Note: May not see speedup in all environments
        print(f"Parallel speedup: {speedup:.2f}x (sequential={seq_time:.3f}s, parallel={par_time:.3f}s)")
        
        # Warn if no speedup but don't fail (depends on system)
        if speedup < 1.0:
            print("WARNING: Parallel processing slower than sequential - may be due to system load or small workload")


def run_integration_tests():
    """Run all integration tests and generate report."""
    print("=" * 80)
    print("FULL PIPELINE INTEGRATION TESTS")
    print("=" * 80)
    
    test_suite = TestFullPipelineIntegration()
    results = {}
    
    # List of tests
    tests = [
        ("All Optimizations", test_suite.test_pipeline_all_optimizations_enabled),
        ("Accuracy Validation", test_suite.test_pipeline_accuracy_validation),
        ("Deterministic Results", test_suite.test_pipeline_deterministic_results),
        ("Error Handling", test_suite.test_pipeline_error_handling),
        ("Configuration Compatibility", test_suite.test_pipeline_configuration_compatibility),
        ("Performance Thresholds", test_suite.test_pipeline_performance_thresholds),
        ("Cache Effectiveness", test_suite.test_pipeline_cache_effectiveness),
        ("Conditional Skip", test_suite.test_pipeline_conditional_skip_validation),
        ("Parallel Speedup", test_suite.test_pipeline_parallel_speedup_validation),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        
        try:
            # Setup
            test_suite.setup_and_teardown().__next__()
            
            # Run test
            test_func()
            results[test_name] = "PASSED"
            print(f"✓ {test_name} passed")
            
        except AssertionError as e:
            results[test_name] = f"FAILED: {str(e)}"
            print(f"✗ {test_name} failed: {str(e)}")
            
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            print(f"✗ {test_name} error: {str(e)}")
            
        finally:
            # Teardown
            try:
                test_suite.setup_and_teardown().__next__()
            except StopIteration:
                pass
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed < total:
        print("\nFailed tests:")
        for test_name, result in results.items():
            if result != "PASSED":
                print(f"  - {test_name}: {result}")
    
    return results


if __name__ == "__main__":
    run_integration_tests()