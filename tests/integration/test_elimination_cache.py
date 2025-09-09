"""Integration tests for pipeline elimination cache system.

These tests verify the caching system works correctly with real pipeline
data and handles edge cases properly.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from giflab.elimination_cache import PipelineResultsCache, get_git_commit
from giflab.elimination_errors import ErrorTypes


class TestPipelineResultsCache:
    """Integration tests for the PipelineResultsCache system."""

    @pytest.fixture
    def temp_cache_db(self):
        """Create a temporary cache database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def cache(self, temp_cache_db):
        """Create a cache instance for testing."""
        return PipelineResultsCache(temp_cache_db, git_commit="test_commit_123")

    @pytest.fixture
    def sample_pipeline_result(self):
        """Sample pipeline test result."""
        return {
            "success": True,
            "file_size_kb": 245.3,
            "original_size_kb": 1024.0,
            "compression_ratio": 4.17,
            "ssim_mean": 0.936,
            "ssim_std": 0.012,
            "render_time_ms": 1250,
            "composite_quality": 0.928,
            "pipeline_steps": ["color_reduce", "lossy_compress"],
            "tools_used": ["imagemagick", "gifsicle"],
        }

    @pytest.fixture
    def sample_test_params(self):
        """Sample test parameters."""
        return {"colors": 128, "lossy": 60, "frame_ratio": 0.8}

    def test_cache_initialization(self, temp_cache_db):
        """Test cache database initialization creates proper schema."""
        PipelineResultsCache(temp_cache_db, git_commit="test_init")

        # Verify tables exist
        with sqlite3.connect(temp_cache_db) as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('pipeline_results', 'pipeline_failures')
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

        assert "pipeline_results" in tables
        assert "pipeline_failures" in tables

        # Verify indexes exist
        with sqlite3.connect(temp_cache_db) as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_%'
            """
            )
            indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            "idx_cache_lookup",
            "idx_failure_lookup",
            "idx_failure_analysis",
        ]
        for expected_index in expected_indexes:
            assert expected_index in indexes

    def test_cache_key_generation(self, cache):
        """Test cache key generation is deterministic and unique."""
        params1 = {"colors": 128, "lossy": 60, "frame_ratio": 0.8}
        params2 = {"colors": 128, "lossy": 60, "frame_ratio": 0.8}
        params3 = {"colors": 64, "lossy": 60, "frame_ratio": 0.8}

        # Same parameters should generate same key
        key1 = cache._generate_cache_key("pipeline_A", "test.gif", params1)
        key2 = cache._generate_cache_key("pipeline_A", "test.gif", params2)
        assert key1 == key2

        # Different parameters should generate different keys
        key3 = cache._generate_cache_key("pipeline_A", "test.gif", params3)
        assert key1 != key3

        # Different pipeline should generate different key
        key4 = cache._generate_cache_key("pipeline_B", "test.gif", params1)
        assert key1 != key4

        # Different GIF should generate different key
        key5 = cache._generate_cache_key("pipeline_A", "other.gif", params1)
        assert key1 != key5

    def test_successful_result_caching(
        self, cache, sample_pipeline_result, sample_test_params
    ):
        """Test caching and retrieval of successful pipeline results."""
        pipeline_id = "imagemagick_floyd_128colors"
        gif_name = "smooth_gradient.gif"

        # Initially no cached result
        cached = cache.get_cached_result(pipeline_id, gif_name, sample_test_params)
        assert cached is None

        # Store result
        cache.queue_result(
            pipeline_id, gif_name, sample_test_params, sample_pipeline_result
        )
        cache.flush_batch(force=True)

        # Now should be cached
        cached = cache.get_cached_result(pipeline_id, gif_name, sample_test_params)
        assert cached is not None
        assert cached["success"] is True
        assert cached["file_size_kb"] == 245.3
        assert cached["ssim_mean"] == 0.936

    def test_failure_result_caching(self, cache, sample_test_params):
        """Test caching and retrieval of pipeline failures."""
        pipeline_id = "gifski_broken_pipeline"
        gif_name = "test.gif"

        error_info = {
            "error": "gifski: Only 1 valid frame found, but gifski requires at least 2 frames",
            "error_traceback": "Traceback: ... gifski error ...",
            "pipeline_steps": ["frame_reduce", "lossy_compress"],
            "tools_used": ["gifski"],
        }

        # Queue failure
        cache.queue_failure(pipeline_id, gif_name, sample_test_params, error_info)
        cache.flush_batch(force=True)

        # Query failures
        failures = cache.query_failures(error_type="gifski")
        assert len(failures) == 1

        failure = failures[0]
        assert failure["pipeline_id"] == pipeline_id
        assert failure["gif_name"] == gif_name
        assert failure["error_type"] == "gifski"
        assert "Only 1 valid frame" in failure["error_message"]
        assert failure["pipeline_steps"] == ["frame_reduce", "lossy_compress"]
        assert failure["tools_used"] == ["gifski"]

    def test_git_commit_invalidation(
        self, temp_cache_db, sample_pipeline_result, sample_test_params
    ):
        """Test that cache entries are invalidated when git commit changes."""
        pipeline_id = "test_pipeline"
        gif_name = "test.gif"

        # Create cache with first commit
        cache1 = PipelineResultsCache(temp_cache_db, git_commit="commit_123")
        cache1.queue_result(
            pipeline_id, gif_name, sample_test_params, sample_pipeline_result
        )
        cache1.flush_batch(force=True)

        # Verify it's cached
        cached = cache1.get_cached_result(pipeline_id, gif_name, sample_test_params)
        assert cached is not None

        # Create new cache with different commit
        cache2 = PipelineResultsCache(temp_cache_db, git_commit="commit_456")

        # Should not find cached result (invalidated)
        cached = cache2.get_cached_result(pipeline_id, gif_name, sample_test_params)
        assert cached is None

    def test_batch_processing(self, cache, sample_test_params):
        """Test batch processing of results and failures."""
        # Queue multiple results without forcing flush
        for i in range(15):  # More than default batch size (10)
            result = {
                "success": True,
                "file_size_kb": 100 + i,
                "ssim_mean": 0.9 + i * 0.001,
            }
            cache.queue_result(
                f"pipeline_{i}", f"gif_{i}.gif", sample_test_params, result
            )

        # Should auto-flush when batch size reached
        # Check that at least 10 results are in database
        with sqlite3.connect(cache.cache_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
            count = cursor.fetchone()[0]
            assert count >= 10

        # Force flush remaining
        cache.flush_batch(force=True)

        # Now all should be in database
        with sqlite3.connect(cache.cache_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
            count = cursor.fetchone()[0]
            assert count == 15

    def test_cache_statistics(self, cache, sample_pipeline_result, sample_test_params):
        """Test cache statistics reporting."""
        # Initially empty
        stats = cache.get_cache_stats()
        assert stats["total_results"] == 0
        assert stats["total_failures"] == 0
        assert stats["current_commit_results"] == 0

        # Add some results and failures
        cache.queue_result(
            "pipeline_1", "gif_1.gif", sample_test_params, sample_pipeline_result
        )
        cache.queue_failure(
            "pipeline_2",
            "gif_2.gif",
            sample_test_params,
            {
                "error": "Test error",
                "error_traceback": "Test traceback",
                "pipeline_steps": [],
                "tools_used": [],
            },
        )
        cache.flush_batch(force=True)

        # Check updated stats
        stats = cache.get_cache_stats()
        assert stats["total_results"] == 1
        assert stats["total_failures"] == 1
        assert stats["current_commit_results"] == 1
        assert stats["current_git_commit"] == "test_commit_123"
        assert stats["database_size_mb"] > 0

    def test_failure_querying_filters(self, cache, sample_test_params):
        """Test failure querying with various filters."""
        # Add different types of failures
        failures_data = [
            ("pipeline_1", "gif_1.gif", "gifski: frame error", "gifski"),
            ("pipeline_2", "gif_2.gif", "ffmpeg: encoding error", "ffmpeg"),
            ("pipeline_3", "gif_3.gif", "timeout occurred", "timeout"),
            ("pipeline_1", "gif_4.gif", "gifski: dimension error", "gifski"),
        ]

        for pipeline_id, gif_name, error_msg, expected_type in failures_data:
            cache.queue_failure(
                pipeline_id,
                gif_name,
                sample_test_params,
                {
                    "error": error_msg,
                    "error_traceback": f"Traceback for {error_msg}",
                    "pipeline_steps": ["step1"],
                    "tools_used": [expected_type],
                },
            )

        cache.flush_batch(force=True)

        # Test filtering by error type
        gifski_failures = cache.query_failures(error_type="gifski")
        assert len(gifski_failures) == 2

        ffmpeg_failures = cache.query_failures(error_type="ffmpeg")
        assert len(ffmpeg_failures) == 1

        # Test filtering by pipeline
        pipeline_1_failures = cache.query_failures(pipeline_id="pipeline_1")
        assert len(pipeline_1_failures) == 2

        # Test recent hours filter
        recent_failures = cache.query_failures(recent_hours=1)
        assert len(recent_failures) == 4  # All are recent

        # Test combined filters
        recent_gifski = cache.query_failures(error_type="gifski", recent_hours=1)
        assert len(recent_gifski) == 2

    def test_cache_clearing(self, cache, sample_pipeline_result, sample_test_params):
        """Test cache clearing functionality."""
        # Add some data
        cache.queue_result(
            "pipeline_1", "gif_1.gif", sample_test_params, sample_pipeline_result
        )
        cache.queue_failure(
            "pipeline_2",
            "gif_2.gif",
            sample_test_params,
            {
                "error": "Test error",
                "error_traceback": "Test traceback",
                "pipeline_steps": [],
                "tools_used": [],
            },
        )
        cache.flush_batch(force=True)

        # Verify data exists
        stats = cache.get_cache_stats()
        assert stats["total_results"] > 0
        assert stats["total_failures"] > 0

        # Clear cache
        cache.clear_cache()

        # Verify data is gone
        stats = cache.get_cache_stats()
        assert stats["total_results"] == 0
        assert stats["total_failures"] == 0

    def test_error_categorization_integration(self, cache, sample_test_params):
        """Test that error categorization works correctly in cache storage."""
        test_errors = [
            ("gifski: Only 1 valid frame found", ErrorTypes.GIFSKI),
            ("ffmpeg encoding failed", ErrorTypes.FFMPEG),
            ("ImageMagick convert error", ErrorTypes.IMAGEMAGICK),
            ("gifsicle optimization failed", ErrorTypes.GIFSICLE),
            ("Animately processing error", ErrorTypes.ANIMATELY),
            ("Command failed with timeout", ErrorTypes.COMMAND_EXECUTION),
            ("Connection timeout occurred", ErrorTypes.TIMEOUT),
            ("Unknown weird error", ErrorTypes.OTHER),
        ]

        for error_msg, expected_type in test_errors:
            cache.queue_failure(
                "test_pipeline",
                "test.gif",
                sample_test_params,
                {
                    "error": error_msg,
                    "error_traceback": f"Traceback for {error_msg}",
                    "pipeline_steps": [],
                    "tools_used": [],
                },
            )

        cache.flush_batch(force=True)

        # Check that errors were categorized correctly
        for error_msg, expected_type in test_errors:
            failures = cache.query_failures()
            matching = [f for f in failures if error_msg in f["error_message"]]
            assert len(matching) == 1
            assert matching[0]["error_type"] == expected_type

    def test_concurrent_access_safety(
        self, temp_cache_db, sample_pipeline_result, sample_test_params
    ):
        """Test that concurrent cache access is handled safely."""
        import threading

        results = []
        errors = []

        def cache_worker(worker_id):
            try:
                cache = PipelineResultsCache(
                    temp_cache_db, git_commit=f"commit_{worker_id}"
                )

                # Each worker stores some results
                for i in range(5):
                    result = sample_pipeline_result.copy()
                    result["file_size_kb"] = 100 + worker_id * 10 + i

                    cache.queue_result(
                        f"pipeline_{worker_id}_{i}",
                        f"gif_{worker_id}_{i}.gif",
                        sample_test_params,
                        result,
                    )

                cache.flush_batch(force=True)
                results.append(f"Worker {worker_id} completed")

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 3

        # Verify all data was stored
        PipelineResultsCache(temp_cache_db, git_commit="verify")
        with sqlite3.connect(temp_cache_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
            total_results = cursor.fetchone()[0]
            # Should have 3 workers * 5 results each = 15 total
            assert total_results == 15

    def test_legacy_store_method_compatibility(
        self, cache, sample_pipeline_result, sample_test_params
    ):
        """Test that legacy store_result method still works."""
        pipeline_id = "legacy_pipeline"
        gif_name = "legacy.gif"

        # Use legacy method
        cache.store_result(
            pipeline_id, gif_name, sample_test_params, sample_pipeline_result
        )

        # Should be retrievable
        cached = cache.get_cached_result(pipeline_id, gif_name, sample_test_params)
        assert cached is not None
        assert cached["success"] is True
        assert cached["file_size_kb"] == 245.3


class TestGitCommitUtility:
    """Tests for git commit hash utility function."""

    def test_get_git_commit_success(self):
        """Test successful git commit retrieval."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abcdef123456789\n"

            commit = get_git_commit()
            assert commit == "abcdef123456"  # Short hash (12 chars)

    def test_get_git_commit_failure(self):
        """Test git commit retrieval failure handling."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""

            commit = get_git_commit()
            assert commit == "unknown"

    def test_get_git_commit_exception(self):
        """Test git commit retrieval exception handling."""
        with mock.patch("subprocess.run", side_effect=Exception("Git not found")):
            commit = get_git_commit()
            assert commit == "unknown"


class TestCacheIntegrationScenarios:
    """Integration tests for realistic cache usage scenarios."""

    @pytest.fixture
    def elimination_cache_scenario(self, tmp_path):
        """Set up a realistic elimination cache scenario."""
        cache_db = tmp_path / "elimination_cache.db"
        cache = PipelineResultsCache(cache_db, git_commit="integration_test")
        return cache

    def test_typical_elimination_run_scenario(self, elimination_cache_scenario):
        """Test a typical pipeline elimination run scenario."""
        cache = elimination_cache_scenario

        # Simulate a pipeline elimination run
        synthetic_gifs = ["gradient.gif", "noise.gif", "contrast.gif"]
        pipelines = ["imagemagick_floyd", "gifsicle_O2", "gifski_quality80"]
        test_params_list = [
            {"colors": 128, "lossy": 60, "frame_ratio": 0.8},
            {"colors": 64, "lossy": 80, "frame_ratio": 0.5},
        ]

        successful_tests = 0
        failed_tests = 0

        for gif_name in synthetic_gifs:
            for pipeline_id in pipelines:
                for test_params in test_params_list:
                    # Simulate some successes and some failures
                    if "gifski" in pipeline_id and gif_name == "contrast.gif":
                        # Simulate gifski failure on high contrast
                        cache.queue_failure(
                            pipeline_id,
                            gif_name,
                            test_params,
                            {
                                "error": "gifski: Only 1 valid frame found",
                                "error_traceback": "Full traceback...",
                                "pipeline_steps": ["frame_reduce", "lossy_compress"],
                                "tools_used": ["gifski"],
                            },
                        )
                        failed_tests += 1
                    else:
                        # Simulate successful test
                        result = {
                            "success": True,
                            "file_size_kb": 150 + hash(pipeline_id + gif_name) % 200,
                            "ssim_mean": 0.85 + (hash(pipeline_id) % 100) / 1000,
                            "render_time_ms": 500 + hash(gif_name) % 1000,
                            "composite_quality": 0.82
                            + (hash(pipeline_id + gif_name) % 150) / 1000,
                        }
                        cache.queue_result(pipeline_id, gif_name, test_params, result)
                        successful_tests += 1

        # Flush all batched data
        cache.flush_batch(force=True)

        # Verify the elimination run results
        stats = cache.get_cache_stats()
        assert stats["total_results"] == successful_tests
        assert stats["total_failures"] == failed_tests

        # Verify we can query specific failure types
        gifski_failures = cache.query_failures(error_type="gifski")
        expected_gifski_failures = len(
            test_params_list
        )  # One failure per test param set
        assert len(gifski_failures) == expected_gifski_failures

        # Verify cached results are retrievable
        for gif_name in synthetic_gifs:
            for pipeline_id in pipelines:
                for test_params in test_params_list:
                    if not ("gifski" in pipeline_id and gif_name == "contrast.gif"):
                        cached = cache.get_cached_result(
                            pipeline_id, gif_name, test_params
                        )
                        assert cached is not None
                        assert cached["success"] is True

    def test_cache_performance_with_large_dataset(self, elimination_cache_scenario):
        """Test cache performance with a larger dataset."""
        cache = elimination_cache_scenario

        # Simulate a large elimination run
        num_gifs = 50
        num_pipelines = 20
        num_test_params = 4

        import time

        start_time = time.time()

        # Store many results
        for gif_idx in range(num_gifs):
            for pipeline_idx in range(num_pipelines):
                for param_idx in range(num_test_params):
                    gif_name = f"synthetic_{gif_idx}.gif"
                    pipeline_id = f"pipeline_{pipeline_idx}"
                    test_params = {
                        "colors": 64 + param_idx * 32,
                        "lossy": 20 + param_idx * 20,
                        "frame_ratio": 0.5 + param_idx * 0.125,
                    }

                    result = {
                        "success": True,
                        "file_size_kb": 100
                        + (gif_idx + pipeline_idx + param_idx) % 300,
                        "ssim_mean": 0.8 + ((gif_idx + pipeline_idx) % 200) / 1000,
                    }

                    cache.queue_result(pipeline_id, gif_name, test_params, result)

        # Flush and measure time
        cache.flush_batch(force=True)
        store_time = time.time() - start_time

        # Test retrieval performance
        retrieval_start = time.time()

        for gif_idx in range(min(10, num_gifs)):  # Test first 10 GIFs
            for pipeline_idx in range(min(5, num_pipelines)):  # Test first 5 pipelines
                gif_name = f"synthetic_{gif_idx}.gif"
                pipeline_id = f"pipeline_{pipeline_idx}"
                test_params = {"colors": 64, "lossy": 20, "frame_ratio": 0.5}

                cached = cache.get_cached_result(pipeline_id, gif_name, test_params)
                assert cached is not None

        retrieval_time = time.time() - retrieval_start

        # Performance assertions
        total_operations = num_gifs * num_pipelines * num_test_params

        # Should be able to store at least 100 results per second
        assert (
            store_time < total_operations / 100
        ), f"Store performance too slow: {store_time}s for {total_operations} operations"

        # Should be able to retrieve at least 1000 results per second
        retrievals_tested = 10 * 5  # 50 retrievals
        assert (
            retrieval_time < retrievals_tested / 1000
        ), f"Retrieval performance too slow: {retrieval_time}s for {retrievals_tested} operations"

        # Verify data integrity
        stats = cache.get_cache_stats()
        assert stats["total_results"] == total_operations
        assert stats["database_size_mb"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
