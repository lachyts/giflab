"""
Tests for cache effectiveness monitoring and analysis.

Comprehensive test suite covering:
- Cache effectiveness metrics collection and analysis
- Performance baseline framework functionality  
- Effectiveness analysis and recommendation generation
- Integration with memory monitoring systems
- Statistical analysis and confidence scoring

Phase 3.2 Implementation: Ensure cache effectiveness monitoring is robust and reliable.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from src.giflab.monitoring.cache_effectiveness import (
    CacheEffectivenessMonitor, CacheOperationType, CacheEffectivenessStats,
    BaselineComparison, get_cache_effectiveness_monitor
)
from src.giflab.monitoring.baseline_framework import (
    PerformanceBaselineFramework, BaselineTestMode, BaselineStatistics,
    PerformanceMeasurement, WorkloadScenario, baseline_performance_test
)
from src.giflab.monitoring.effectiveness_analysis import (
    CacheEffectivenessAnalyzer, CacheRecommendation, CacheEffectivenessAnalysis,
    analyze_cache_effectiveness
)
from src.giflab.monitoring.memory_monitor import MemoryPressureLevel


class TestCacheEffectivenessMonitor:
    """Test cache effectiveness monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test cache effectiveness monitor initialization."""
        monitor = CacheEffectivenessMonitor(
            max_operations_history=1000,
            time_window_minutes=30
        )
        
        assert monitor.max_operations_history == 1000
        assert monitor.time_window_minutes == 30
        assert len(monitor._operations) == 0
        assert len(monitor._cache_stats) == 0
    
    def test_record_cache_operation_hit(self):
        """Test recording cache hit operations."""
        monitor = CacheEffectivenessMonitor()
        
        monitor.record_operation(
            cache_type="frame_cache",
            operation=CacheOperationType.HIT,
            key="test_key_1",
            processing_time_ms=50.0
        )
        
        assert len(monitor._operations) == 1
        assert "frame_cache" in monitor._cache_stats
        
        stats = monitor._cache_stats["frame_cache"]
        assert stats.total_operations == 1
        assert stats.hits == 1
        assert stats.misses == 0
    
    def test_record_cache_operation_miss(self):
        """Test recording cache miss operations."""
        monitor = CacheEffectivenessMonitor()
        
        monitor.record_operation(
            cache_type="frame_cache",
            operation=CacheOperationType.MISS,
            key="test_key_2",
            processing_time_ms=150.0
        )
        
        stats = monitor._cache_stats["frame_cache"]
        assert stats.total_operations == 1
        assert stats.hits == 0
        assert stats.misses == 1
    
    def test_record_cache_eviction_with_memory_pressure(self):
        """Test recording cache evictions under memory pressure."""
        monitor = CacheEffectivenessMonitor()
        
        monitor.record_operation(
            cache_type="frame_cache",
            operation=CacheOperationType.EVICT,
            key="evicted_key",
            data_size_mb=10.0,
            memory_pressure_level=MemoryPressureLevel.CRITICAL
        )
        
        stats = monitor._cache_stats["frame_cache"]
        assert stats.evictions == 1
        assert stats.evictions_under_pressure == 1
    
    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        monitor = CacheEffectivenessMonitor()
        
        # Record mixed operations
        for i in range(8):  # 8 hits
            monitor.record_operation("cache", CacheOperationType.HIT, f"key_{i}")
        
        for i in range(2):  # 2 misses
            monitor.record_operation("cache", CacheOperationType.MISS, f"key_miss_{i}")
        
        effectiveness = monitor.get_cache_effectiveness("cache")
        assert effectiveness is not None
        assert effectiveness.hit_rate == 0.8  # 8/(8+2)
        assert effectiveness.hits == 8
        assert effectiveness.misses == 2
    
    def test_windowed_hit_rate_calculation(self):
        """Test time-windowed hit rate calculation."""
        monitor = CacheEffectivenessMonitor()
        
        # Record operations at different times
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0.0
            monitor.record_operation("cache", CacheOperationType.HIT, "old_key")
            
            # Move to 10 minutes later
            mock_time.return_value = 600.0
            for i in range(5):
                monitor.record_operation("cache", CacheOperationType.HIT, f"recent_key_{i}")
            monitor.record_operation("cache", CacheOperationType.MISS, "recent_miss")
            
            # Calculate windowed hit rate (should only include recent operations)
            hit_rate = monitor._calculate_windowed_hit_rate("cache", 5)  # 5 minute window
            assert hit_rate == 5/6  # 5 hits, 1 miss in recent window
    
    def test_baseline_comparison(self):
        """Test baseline performance comparison."""
        monitor = CacheEffectivenessMonitor()
        
        # Record baseline (non-cached) times (need at least 10 samples)
        baseline_times = [100, 110, 105, 95, 120, 115, 108, 102, 98, 125, 103, 107]
        for time_ms in baseline_times:
            monitor.record_baseline_performance("operation_a", time_ms)
        
        # Record cached times (need at least 10 samples)
        cached_times = [50, 55, 45, 60, 40, 48, 52, 58, 42, 46, 53, 49]
        for time_ms in cached_times:
            monitor.record_cached_performance("operation_a", time_ms)
        
        comparison = monitor.get_baseline_comparison("operation_a")
        assert comparison is not None
        assert comparison.operation_type == "operation_a"
        assert comparison.performance_improvement > 0.4  # Should be ~50% improvement
        assert comparison.sample_size_cached == 12
        assert comparison.sample_size_non_cached == 12
    
    def test_system_effectiveness_summary(self):
        """Test system-wide effectiveness summary."""
        monitor = CacheEffectivenessMonitor()
        
        # Record operations for multiple cache types
        for cache_type in ["frame_cache", "resize_cache"]:
            for i in range(10):
                monitor.record_operation(cache_type, CacheOperationType.HIT, f"key_{i}")
            for i in range(5):
                monitor.record_operation(cache_type, CacheOperationType.MISS, f"miss_{i}")
        
        summary = monitor.get_system_effectiveness_summary()
        assert summary["total_operations"] == 30  # 15 ops * 2 cache types
        assert summary["overall_hit_rate"] == 20/30  # 20 hits out of 30 total
        assert summary["cache_types"] == 2
    
    def test_memory_pressure_correlation(self):
        """Test memory pressure correlation calculation."""
        monitor = CacheEffectivenessMonitor()
        
        # Add some memory stats history so correlation calculation works
        from src.giflab.monitoring.memory_monitor import MemoryStats
        mock_stats = MemoryStats(
            process_memory_mb=100.0,
            process_memory_percent=5.0,
            system_memory_mb=2000.0,
            system_memory_percent=50.0,
            system_available_mb=2000.0,
            total_system_mb=4000.0,
            pressure_level=MemoryPressureLevel.CRITICAL,
            timestamp=time.time()
        )
        monitor.update_memory_pressure(mock_stats)
        
        # Record evictions under different pressure levels
        monitor.record_operation("cache", CacheOperationType.EVICT, "key1", 
                               memory_pressure_level=MemoryPressureLevel.CRITICAL)
        monitor.record_operation("cache", CacheOperationType.EVICT, "key2",
                               memory_pressure_level=MemoryPressureLevel.WARNING)
        monitor.record_operation("cache", CacheOperationType.EVICT, "key3",
                               memory_pressure_level=MemoryPressureLevel.NORMAL)
        
        correlation = monitor._calculate_pressure_correlation("cache")
        assert correlation == 2/3  # 2 out of 3 evictions under pressure
    
    def test_clear_statistics(self):
        """Test clearing cache statistics."""
        monitor = CacheEffectivenessMonitor()
        
        # Record some operations
        monitor.record_operation("cache1", CacheOperationType.HIT, "key1")
        monitor.record_operation("cache2", CacheOperationType.MISS, "key2")
        
        assert len(monitor._cache_stats) == 2
        assert len(monitor._operations) == 2
        
        # Clear specific cache
        monitor.clear_statistics("cache1")
        assert "cache1" not in monitor._cache_stats
        assert len(monitor._operations) == 1  # Only cache2 operation remains
        
        # Clear all statistics
        monitor.clear_statistics()
        assert len(monitor._cache_stats) == 0
        assert len(monitor._operations) == 0


class TestPerformanceBaselineFramework:
    """Test performance baseline framework functionality."""
    
    def test_framework_initialization(self):
        """Test baseline framework initialization."""
        framework = PerformanceBaselineFramework(
            test_mode=BaselineTestMode.AB_TESTING,
            ab_split_ratio=0.2
        )
        
        assert framework.test_mode == BaselineTestMode.AB_TESTING
        assert framework.ab_split_ratio == 0.2
        assert len(framework._measurements) == 0
    
    def test_ab_testing_mode(self):
        """Test A/B testing mode for baseline collection."""
        framework = PerformanceBaselineFramework(
            test_mode=BaselineTestMode.AB_TESTING,
            ab_split_ratio=0.5  # 50% split for testing
        )
        
        # Should alternate between baseline and cached testing
        results = []
        for i in range(10):
            should_baseline = framework.should_run_baseline_test("test_op")
            results.append(should_baseline)
        
        # Should have roughly 50% baseline tests
        baseline_count = sum(results)
        assert 3 <= baseline_count <= 7  # Allow some variance
    
    def test_record_performance_measurement(self):
        """Test recording performance measurements."""
        framework = PerformanceBaselineFramework()
        
        framework.record_performance(
            operation_type="gif_processing",
            processing_time_ms=100.0,
            cache_enabled=True,
            memory_usage_mb=50.0,
            metadata={"file_size": "large"}
        )
        
        measurements = framework._measurements["gif_processing"]
        assert len(measurements) == 1
        
        measurement = measurements[0]
        assert measurement.operation_type == "gif_processing"
        assert measurement.processing_time_ms == 100.0
        assert measurement.cache_enabled is True
        assert measurement.metadata["file_size"] == "large"
    
    def test_baseline_statistics_calculation(self):
        """Test baseline statistics calculation."""
        framework = PerformanceBaselineFramework(min_samples_for_analysis=15)  # Lower threshold for testing
        
        # Record cached measurements
        for i in range(20):
            framework.record_performance("test_op", 50.0 + i, True)
        
        # Record non-cached measurements
        for i in range(20):
            framework.record_performance("test_op", 100.0 + i, False)
        
        stats = framework.get_baseline_statistics("test_op")
        assert stats is not None
        assert stats.cached_samples == 20
        assert stats.non_cached_samples == 20
        assert stats.performance_improvement > 0.4  # Should be significant improvement
        assert stats.min_samples_met is True
    
    def test_workload_scenario_registration(self):
        """Test workload scenario registration and execution."""
        framework = PerformanceBaselineFramework()
        
        def setup_func():
            return {"data": "test_data"}
        
        def exec_func(context):
            time.sleep(0.001)  # Simulate processing
            return 1.0  # Return processing time
        
        scenario = WorkloadScenario(
            name="test_scenario",
            operation_type="test_operation",
            setup_function=setup_func,
            execution_function=exec_func,
            metadata={"type": "synthetic"}
        )
        
        framework.register_workload_scenario(scenario)
        assert "test_scenario" in framework._workload_scenarios
    
    def test_controlled_test_execution(self):
        """Test controlled test execution with workload scenarios."""
        framework = PerformanceBaselineFramework()
        
        call_count = 0
        
        def setup_func():
            return {"iteration": call_count}
        
        def exec_func(context):
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # Small delay to ensure measurable time
            return None  # Framework measures time automatically
        
        scenario = WorkloadScenario(
            name="controlled_test",
            operation_type="controlled_op",
            setup_function=setup_func,
            execution_function=exec_func
        )
        
        framework.register_workload_scenario(scenario)
        
        measurements = framework.run_controlled_test("controlled_test", iterations=5, cache_enabled=True)
        
        assert len(measurements) == 5
        assert call_count == 5
        assert all(m.cache_enabled for m in measurements)
        assert all(m.processing_time_ms > 0 for m in measurements)
    
    def test_comparative_test(self):
        """Test comparative testing with cached vs non-cached modes."""
        framework = PerformanceBaselineFramework()
        
        def setup_func():
            return {}
        
        def exec_func(context):
            return None
        
        scenario = WorkloadScenario(
            name="comparison_test",
            operation_type="comparison_op",
            setup_function=setup_func,
            execution_function=exec_func
        )
        
        framework.register_workload_scenario(scenario)
        
        cached_measurements, non_cached_measurements = framework.run_comparative_test("comparison_test", 10)
        
        assert len(cached_measurements) == 10
        assert len(non_cached_measurements) == 10
        assert all(m.cache_enabled for m in cached_measurements)
        assert not any(m.cache_enabled for m in non_cached_measurements)
    
    def test_performance_report_generation(self):
        """Test performance analysis report generation."""
        framework = PerformanceBaselineFramework()
        
        # Record sufficient data for meaningful analysis
        for i in range(50):
            framework.record_performance("report_test", 80.0 + i % 10, True)
            framework.record_performance("report_test", 120.0 + i % 15, False)
        
        report = framework.generate_performance_report()
        
        assert report["status"] == "active"
        assert report["collection_summary"]["total_operations"] == 100
        assert report["collection_summary"]["operation_types"] == 1
        assert "performance_analysis" in report
        assert "recommendations" in report
    
    def test_baseline_performance_context_manager(self):
        """Test baseline performance testing context manager."""
        framework = PerformanceBaselineFramework(test_mode=BaselineTestMode.PASSIVE)
        
        with baseline_performance_test(framework, "context_test") as cache_enabled:
            assert cache_enabled is True  # Should use cache in passive mode
            time.sleep(0.001)  # Simulate work
        
        # Should have recorded the measurement
        measurements = framework._measurements["context_test"]
        assert len(measurements) == 1
        assert measurements[0].cache_enabled is True


class TestCacheEffectivenessAnalyzer:
    """Test cache effectiveness analysis and recommendation generation."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = CacheEffectivenessAnalyzer(
            min_confidence_threshold=0.8,
            min_analysis_period_hours=2.0
        )
        
        assert analyzer.min_confidence_threshold == 0.8
        assert analyzer.min_analysis_period_hours == 2.0
    
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_insufficient_data_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test analysis with insufficient data."""
        # Mock insufficient data scenario
        mock_effectiveness.return_value.get_all_cache_stats.return_value = {}
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 10,  # Too few operations
            "overall_hit_rate": 0.0,
            "monitoring_duration_hours": 0.5  # Too short duration
        }
        mock_baseline.return_value.get_all_baseline_statistics.return_value = {}
        mock_baseline.return_value.generate_performance_report.return_value = {"performance_analysis": {}}
        mock_memory.return_value.get_system_effectiveness_summary.return_value = {}
        
        analyzer = CacheEffectivenessAnalyzer()
        analysis = analyzer.analyze_cache_effectiveness()
        
        assert analysis.recommendation == CacheRecommendation.INSUFFICIENT_DATA
        assert analysis.confidence_score < 0.2
        assert len(analysis.optimization_recommendations) > 0
    
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_excellent_performance_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test analysis with excellent cache performance."""
        # Mock excellent performance data
        mock_cache_stats = {
            "frame_cache": Mock(
                hit_rate=0.9,
                total_operations=1000,
                hits=900,
                misses=100,
                evictions=10,
                puts=200,
                total_data_cached_mb=100.0,
                cache_turnover_rate=0.05
            )
        }
        
        mock_effectiveness.return_value.get_all_cache_stats.return_value = mock_cache_stats
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 1000,
            "overall_hit_rate": 0.9,
            "monitoring_duration_hours": 5.0
        }
        
        mock_baseline_stats = {
            "gif_processing": Mock(
                operation_type="gif_processing",
                performance_improvement=0.4,  # 40% improvement
                statistical_significance=True,
                min_samples_met=True,
                cached_samples=100,
                non_cached_samples=100
            )
        }
        
        mock_baseline.return_value.get_all_baseline_statistics.return_value = mock_baseline_stats
        mock_baseline.return_value.generate_performance_report.return_value = {
            "performance_analysis": {"average_improvement": 0.4}
        }
        
        mock_memory.return_value.get_system_effectiveness_summary.return_value = {
            "efficiency_score": 0.9
        }
        
        analyzer = CacheEffectivenessAnalyzer()
        analysis = analyzer.analyze_cache_effectiveness()
        
        assert analysis.recommendation == CacheRecommendation.ENABLE_PRODUCTION
        assert analysis.confidence_score > 0.8
        assert analysis.overall_hit_rate == 0.9
        assert analysis.average_performance_improvement == 0.4
    
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_poor_performance_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test analysis with poor cache performance."""
        # Mock poor performance data
        mock_cache_stats = {
            "frame_cache": Mock(
                hit_rate=0.2,  # Very poor hit rate
                total_operations=1000,
                hits=200,
                misses=800,
                evictions=500,  # High eviction rate
                puts=600,
                total_data_cached_mb=50.0,
                cache_turnover_rate=0.8
            )
        }
        
        mock_effectiveness.return_value.get_all_cache_stats.return_value = mock_cache_stats
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 1000,
            "overall_hit_rate": 0.2,
            "monitoring_duration_hours": 5.0
        }
        
        mock_baseline_stats = {
            "gif_processing": Mock(
                operation_type="gif_processing",
                performance_improvement=-0.1,  # Performance regression
                statistical_significance=True,
                min_samples_met=True,
                cached_samples=100,
                non_cached_samples=100
            )
        }
        
        mock_baseline.return_value.get_all_baseline_statistics.return_value = mock_baseline_stats
        mock_baseline.return_value.generate_performance_report.return_value = {
            "performance_analysis": {"average_improvement": -0.1}
        }
        
        mock_memory.return_value.get_system_effectiveness_summary.return_value = {
            "efficiency_score": 0.2
        }
        
        analyzer = CacheEffectivenessAnalyzer()
        analysis = analyzer.analyze_cache_effectiveness()
        
        assert analysis.recommendation in [CacheRecommendation.KEEP_DISABLED, CacheRecommendation.PERFORMANCE_REGRESSION]
        assert analysis.overall_hit_rate == 0.2
        assert analysis.average_performance_improvement == -0.1
        assert len(analysis.risk_factors) > 0
    
    def test_hit_rate_analysis(self):
        """Test hit rate analysis scoring."""
        analyzer = CacheEffectivenessAnalyzer()
        
        # Test excellent hit rate
        cache_stats = {
            "cache1": Mock(hit_rate=0.9, total_operations=100)
        }
        analysis = analyzer._analyze_hit_rates(cache_stats, 0.9)
        assert analysis["overall_assessment"] == "excellent"
        assert analysis["hit_rate_score"] == 1.0
        
        # Test poor hit rate
        cache_stats = {
            "cache1": Mock(hit_rate=0.2, total_operations=100)
        }
        analysis = analyzer._analyze_hit_rates(cache_stats, 0.2)
        assert analysis["overall_assessment"] == "very_poor"
        assert analysis["hit_rate_score"] == 0.1
    
    def test_performance_analysis(self):
        """Test performance improvement analysis."""
        analyzer = CacheEffectivenessAnalyzer()
        
        # Test excellent performance improvement
        baseline_stats = {
            "op1": Mock(
                operation_type="op1",
                performance_improvement=0.5,
                statistical_significance=True
            )
        }
        analysis = analyzer._analyze_performance_impact(baseline_stats, 0.5)
        assert analysis["overall_assessment"] == "excellent"
        assert analysis["performance_score"] == 1.0
        
        # Test negative performance impact
        baseline_stats = {
            "op1": Mock(
                operation_type="op1", 
                performance_improvement=-0.2,
                statistical_significance=True
            )
        }
        analysis = analyzer._analyze_performance_impact(baseline_stats, -0.2)
        assert analysis["overall_assessment"] == "negative"
        assert analysis["performance_score"] == 0.0
    
    def test_memory_efficiency_analysis(self):
        """Test memory efficiency analysis."""
        analyzer = CacheEffectivenessAnalyzer()
        
        # Test excellent memory efficiency (low eviction rate)
        cache_stats = {
            "cache1": Mock(
                total_data_cached_mb=100.0,
                evictions=5,
                puts=100  # 5% eviction rate
            )
        }
        analysis = analyzer._analyze_memory_efficiency(cache_stats, {})
        assert analysis["overall_assessment"] == "excellent"
        assert analysis["efficiency_score"] == 1.0
        
        # Test poor memory efficiency (high eviction rate)
        cache_stats = {
            "cache1": Mock(
                total_data_cached_mb=100.0,
                evictions=80,
                puts=100  # 80% eviction rate
            )
        }
        analysis = analyzer._analyze_memory_efficiency(cache_stats, {})
        assert analysis["overall_assessment"] == "poor"
        assert analysis["efficiency_score"] == 0.1
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        analyzer = CacheEffectivenessAnalyzer()
        
        hit_rate_analysis = {"overall_hit_rate": 0.3}  # Poor hit rate
        performance_analysis = {"average_improvement": 0.02}  # Marginal improvement
        memory_analysis = {"cache_type_eviction_rates": {"cache1": 0.4}}  # High eviction
        
        cache_stats = {
            "cache1": Mock(cache_turnover_rate=0.9, hit_rate=0.2, total_operations=200)
        }
        
        recommendations = analyzer._generate_optimization_recommendations(
            hit_rate_analysis, performance_analysis, memory_analysis, cache_stats
        )
        
        assert len(recommendations) > 0
        assert any("hit rate" in rec.lower() for rec in recommendations)
        assert any("eviction" in rec.lower() for rec in recommendations)
    
    def test_cache_size_suggestions(self):
        """Test cache size optimization suggestions."""
        analyzer = CacheEffectivenessAnalyzer()
        
        cache_stats = {
            "high_eviction_cache": Mock(
                total_data_cached_mb=100.0,
                evictions=60,
                puts=100  # 60% eviction rate - suggest larger
            ),
            "low_eviction_cache": Mock(
                total_data_cached_mb=100.0, 
                evictions=5,
                puts=100,  # 5% eviction rate
                hit_rate=0.4  # But poor hit rate - suggest smaller
            )
        }
        
        memory_analysis = {"efficiency_score": 0.5}
        suggestions = analyzer._suggest_cache_sizes(cache_stats, memory_analysis)
        
        # High eviction cache should get size increase
        assert suggestions["high_eviction_cache"] > 100.0
        
        # Low eviction but poor hit rate cache should get size decrease
        assert suggestions["low_eviction_cache"] < 100.0


class TestIntegration:
    """Test integration between cache effectiveness components."""
    
    def test_effectiveness_monitor_singleton(self):
        """Test that effectiveness monitor singleton works correctly."""
        monitor1 = get_cache_effectiveness_monitor()
        monitor2 = get_cache_effectiveness_monitor()
        
        assert monitor1 is monitor2  # Should be same instance
    
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework') 
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_end_to_end_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test end-to-end cache effectiveness analysis."""
        # Setup mock data representing a realistic scenario
        mock_cache_stats = {
            "frame_cache": Mock(
                hit_rate=0.7,
                total_operations=500,
                hits=350,
                misses=150,
                evictions=25,
                puts=100,
                total_data_cached_mb=75.0,
                cache_turnover_rate=0.25
            )
        }
        
        mock_effectiveness.return_value.get_all_cache_stats.return_value = mock_cache_stats
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 500,
            "overall_hit_rate": 0.7,
            "monitoring_duration_hours": 3.0
        }
        
        mock_baseline_stats = {
            "frame_processing": Mock(
                operation_type="frame_processing",
                performance_improvement=0.25,  # 25% improvement
                statistical_significance=True,
                min_samples_met=True,
                cached_samples=50,
                non_cached_samples=50
            )
        }
        
        mock_baseline.return_value.get_all_baseline_statistics.return_value = mock_baseline_stats
        mock_baseline.return_value.generate_performance_report.return_value = {
            "performance_analysis": {"average_improvement": 0.25}
        }
        
        mock_memory.return_value.get_system_effectiveness_summary.return_value = {
            "efficiency_score": 0.75
        }
        
        # Run analysis
        analysis = analyze_cache_effectiveness()
        
        # Verify results
        assert analysis.recommendation in [CacheRecommendation.ENABLE_PRODUCTION, CacheRecommendation.ENABLE_WITH_MONITORING]
        assert analysis.confidence_score > 0.6
        assert analysis.overall_hit_rate == 0.7
        assert analysis.average_performance_improvement == 0.25
        assert len(analysis.optimization_recommendations) >= 0
    
    def test_thread_safety(self):
        """Test thread safety of cache effectiveness monitoring."""
        monitor = CacheEffectivenessMonitor()
        
        def record_operations(thread_id):
            for i in range(100):
                monitor.record_operation(
                    cache_type=f"cache_{thread_id}",
                    operation=CacheOperationType.HIT if i % 2 == 0 else CacheOperationType.MISS,
                    key=f"key_{thread_id}_{i}"
                )
        
        # Start multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=record_operations, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations were recorded correctly
        assert len(monitor._cache_stats) == 5  # 5 different cache types
        total_operations = sum(stats.total_operations for stats in monitor._cache_stats.values())
        assert total_operations == 500  # 100 operations * 5 threads
    
    def test_memory_pressure_integration(self):
        """Test integration with memory pressure monitoring."""
        monitor = CacheEffectivenessMonitor()
        
        # Add memory stats history for correlation calculation
        from src.giflab.monitoring.memory_monitor import MemoryStats
        mock_stats = MemoryStats(
            process_memory_mb=100.0,
            process_memory_percent=5.0,
            system_memory_mb=2000.0,
            system_memory_percent=50.0,
            system_available_mb=2000.0,
            total_system_mb=4000.0,
            pressure_level=MemoryPressureLevel.CRITICAL,
            timestamp=time.time()
        )
        monitor.update_memory_pressure(mock_stats)
        
        # Record evictions under different memory pressure levels
        pressure_levels = [
            MemoryPressureLevel.NORMAL,
            MemoryPressureLevel.WARNING,
            MemoryPressureLevel.CRITICAL,
            MemoryPressureLevel.EMERGENCY
        ]
        
        for i, level in enumerate(pressure_levels * 5):  # 20 total evictions
            monitor.record_operation(
                cache_type="pressure_test_cache",
                operation=CacheOperationType.EVICT,
                key=f"evict_key_{i}",
                memory_pressure_level=level
            )
        
        effectiveness = monitor.get_cache_effectiveness("pressure_test_cache")
        assert effectiveness is not None
        assert effectiveness.evictions == 20
        assert effectiveness.evictions_under_pressure == 15  # 3/4 of evictions under pressure
        
        # Correlation should be 0.75 (15 pressure evictions / 20 total evictions)
        correlation = monitor._calculate_pressure_correlation("pressure_test_cache")
        assert abs(correlation - 0.75) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])