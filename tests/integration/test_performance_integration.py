"""Integration tests for performance improvements.

These tests verify that the performance improvements work correctly
in realistic scenarios without requiring extensive benchmarking.
"""

import tempfile
import time
from pathlib import Path

import pytest
from giflab.multiprocessing_support import (
    ParallelFrameGenerator,
    get_optimal_worker_count,
)
from giflab.synthetic_gifs import SyntheticFrameGenerator, SyntheticGifGenerator
from PIL import Image


class TestPerformanceIntegration:
    """Integration tests for performance improvements."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.performance
    @pytest.mark.xdist_group("performance_tests")
    def test_vectorized_vs_serial_frame_generation(self):
        """Test that vectorized generation is faster than theoretical serial approach."""
        generator = SyntheticFrameGenerator()

        # Generate several frames and measure time
        frames_to_test = 10
        size = (200, 200)

        # Warm up to reduce timing variance
        generator.create_frame("complex_gradient", size, 0, frames_to_test)

        start_time = time.time()
        for i in range(frames_to_test):
            img = generator.create_frame("complex_gradient", size, i, frames_to_test)
            assert img.size == size

        vectorized_time = time.time() - start_time
        time_per_frame = vectorized_time / frames_to_test

        # With vectorization, should be fast (< 0.015s per frame for 200x200)
        # Allow some extra time for CI/parallel execution overhead
        assert (
            time_per_frame < 0.015
        ), f"Vectorized generation too slow: {time_per_frame:.4f}s/frame"

        # Test that larger images are still reasonable
        large_img = generator.create_frame("gradient", (400, 400), 0, 5)
        assert large_img.size == (400, 400)

    def test_multiprocessing_overhead_analysis(self):
        """Test that multiprocessing overhead is understood and appropriate."""
        # Small tasks where overhead dominates
        small_generator = ParallelFrameGenerator(max_workers=2)
        frame_generator = SyntheticFrameGenerator()

        # Small, fast tasks
        small_size = (50, 50)
        num_frames = 4

        # Time serial execution
        start_serial = time.time()
        serial_images = []
        for i in range(num_frames):
            img = frame_generator.create_frame("gradient", small_size, i, num_frames)
            serial_images.append(img)
        serial_time = time.time() - start_serial

        # Time parallel execution
        from giflab.synthetic_gifs import SyntheticGifSpec

        spec = SyntheticGifSpec("test", num_frames, small_size, "gradient", "Test")

        start_parallel = time.time()
        parallel_images = small_generator.generate_gif_frames_parallel(spec)
        parallel_time = time.time() - start_parallel

        # Verify both produce correct results
        assert len(serial_images) == num_frames
        assert len(parallel_images) == num_frames

        # For small, fast tasks, serial should often be faster due to overhead
        # But both should complete in reasonable time
        assert serial_time < 1.0
        assert parallel_time < 5.0  # Allow more time for process startup overhead

    def test_synthetic_gif_generation_integration(self):
        """Test complete synthetic GIF generation workflow."""
        generator = SyntheticGifGenerator(self.temp_dir)

        # Test with a small subset of specs
        test_specs = list(generator.synthetic_specs[:3])  # First 3 specs

        start_time = time.time()

        # Generate frames for each spec manually (testing the core functionality)
        frame_generator = SyntheticFrameGenerator()
        total_frames_generated = 0

        for spec in test_specs:
            images = []
            for frame_idx in range(spec.frames):
                img = frame_generator.create_frame(
                    spec.content_type, spec.size, frame_idx, spec.frames
                )
                images.append(img)
                total_frames_generated += 1

            # Verify we got the right number of frames
            assert len(images) == spec.frames

            # Save as GIF to test complete workflow
            gif_path = self.temp_dir / f"{spec.name}_test.gif"
            if images:
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=100,
                    loop=0,
                )
                assert gif_path.exists()
                assert gif_path.stat().st_size > 0

        total_time = time.time() - start_time
        avg_time_per_frame = total_time / total_frames_generated

        # Should generate frames very quickly with vectorization (allow for CI overhead)
        assert (
            avg_time_per_frame < 0.025
        ), f"Frame generation too slow: {avg_time_per_frame:.4f}s"
        assert total_frames_generated > 0

    def test_worker_count_optimization(self):
        """Test that optimal worker count calculation is sensible."""
        # Test different scenarios
        frame_workers = get_optimal_worker_count("frame_generation")
        pipeline_workers = get_optimal_worker_count("pipeline_execution")
        default_workers = get_optimal_worker_count("unknown")

        # All should be positive integers
        assert isinstance(frame_workers, int) and frame_workers > 0
        assert isinstance(pipeline_workers, int) and pipeline_workers > 0
        assert isinstance(default_workers, int) and default_workers > 0

        # Frame generation should use all CPUs (CPU-intensive)
        import multiprocessing as mp

        assert frame_workers == mp.cpu_count()

        # Pipeline execution should leave one CPU for coordination
        assert pipeline_workers == max(1, mp.cpu_count() - 1)

        # Default should be conservative
        assert default_workers == max(1, mp.cpu_count() // 2)

    def test_backward_compatibility_maintained(self):
        """Test that all performance improvements maintain backward compatibility."""
        # Original API should still work
        generator = SyntheticFrameGenerator()

        # Test all the main content types that were optimized
        content_types = ["gradient", "complex_gradient", "noise", "texture", "solid"]

        for content_type in content_types:
            # Should work with various sizes
            for size in [(50, 50), (100, 100), (200, 200)]:
                img = generator.create_frame(content_type, size, 0, 5)

                # Should return PIL Image as before
                assert isinstance(img, Image.Image)
                assert img.mode == "RGB"
                assert img.size == size

                # Should be deterministic for noise (same seed)
                if content_type == "noise":
                    img2 = generator.create_frame(content_type, size, 0, 5)
                    import numpy as np

                    assert np.array_equal(np.array(img), np.array(img2))

    def test_performance_regression_detection(self):
        """Test that performance hasn't regressed from expected levels."""
        generator = SyntheticFrameGenerator()

        # Test cases that should be very fast with vectorization
        test_cases = [
            ("gradient", (100, 100), 0.005),  # Should be very fast
            ("complex_gradient", (150, 150), 0.008),  # Slightly more complex
            ("noise", (100, 100), 0.005),  # Fast with vectorized random
            ("texture", (100, 100), 0.005),  # Fast with vectorized math
            ("solid", (150, 150), 0.003),  # Very fast block operations
        ]

        for content_type, size, max_time in test_cases:
            start = time.time()
            img = generator.create_frame(content_type, size, 0, 8)
            elapsed = time.time() - start

            assert (
                elapsed < max_time
            ), f"{content_type} took {elapsed:.4f}s, expected < {max_time}s"
            assert img.size == size


@pytest.mark.integration
class TestRealWorldPerformance:
    """Test performance in realistic usage scenarios."""

    def test_batch_frame_generation_performance(self):
        """Test performance when generating many frames (realistic batch scenario)."""
        generator = SyntheticFrameGenerator()

        # Simulate generating frames for multiple GIFs
        batch_specs = [
            ("gradient", (120, 120), 8),
            ("noise", (140, 140), 6),
            ("texture", (100, 100), 10),
            ("solid", (160, 160), 5),
        ]

        start_time = time.time()
        total_frames = 0

        for content_type, size, frame_count in batch_specs:
            for frame_idx in range(frame_count):
                img = generator.create_frame(content_type, size, frame_idx, frame_count)
                assert isinstance(img, Image.Image)
                total_frames += 1

        total_time = time.time() - start_time
        frames_per_second = total_frames / total_time

        # Should achieve high throughput with vectorization
        assert frames_per_second > 100, f"Low throughput: {frames_per_second:.1f} fps"
        assert total_frames == sum(spec[2] for spec in batch_specs)

    def test_memory_efficiency(self):
        """Test that vectorized operations don't use excessive memory."""
        generator = SyntheticFrameGenerator()

        # Generate several large images and verify they complete
        large_size = (500, 500)

        try:
            for i in range(5):
                img = generator.create_frame("complex_gradient", large_size, i, 5)
                assert img.size == large_size
                # Force garbage collection of previous image
                del img

        except MemoryError:
            pytest.fail("Vectorized generation should not cause memory errors")

    def test_concurrent_generation_safety(self):
        """Test that frame generation is safe under concurrent access."""
        import queue
        import threading

        generator = SyntheticFrameGenerator()
        results = queue.Queue()
        errors = queue.Queue()

        def worker(thread_id):
            try:
                for i in range(5):
                    img = generator.create_frame("gradient", (80, 80), i, 5)
                    results.put((thread_id, i, img.size))
            except Exception as e:
                errors.put((thread_id, str(e)))

        # Start multiple threads
        threads = []
        for t_id in range(3):
            thread = threading.Thread(target=worker, args=(t_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert errors.empty(), f"Concurrent access caused errors: {list(errors.queue)}"

        # Verify all results were produced
        assert results.qsize() == 15  # 3 threads × 5 frames each
