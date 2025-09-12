"""
Tests for Phase 7: Performance Regression Detection System

This test suite provides comprehensive coverage of the continuous performance
monitoring and alerting system, including:

- PerformanceBaseline statistical calculations
- RegressionDetector threshold detection
- PerformanceHistory data management
- ContinuousMonitor background monitoring
- CLI integration testing

Test Categories:
- Unit tests for individual components
- Integration tests for system interactions
- Mock-based tests for external dependencies
- Performance tests for overhead validation
"""

import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest

from src.giflab.monitoring.performance_regression import (
    PerformanceBaseline,
    RegressionAlert,
    RegressionDetector,
    PerformanceHistory,
    ContinuousMonitor,
    create_performance_monitor
)
from src.giflab.monitoring.alerting import Alert, AlertLevel
from src.giflab.benchmarks.phase_4_3_benchmarking import BenchmarkResult, BenchmarkScenario


class TestPerformanceBaseline(unittest.TestCase):
    """Test PerformanceBaseline statistical calculations and serialization."""
    
    def setUp(self):
        self.baseline = PerformanceBaseline(
            scenario_name="test_scenario",
            mean_processing_time=10.0,
            std_processing_time=1.0,
            mean_memory_usage=100.0,
            std_memory_usage=10.0,
            sample_count=5,
            last_updated=datetime(2025, 1, 12, 12, 0, 0),
            confidence_level=0.95
        )
    
    def test_baseline_creation(self):
        """Test baseline creation with valid parameters."""
        self.assertEqual(self.baseline.scenario_name, "test_scenario")
        self.assertEqual(self.baseline.mean_processing_time, 10.0)
        self.assertEqual(self.baseline.sample_count, 5)
        self.assertEqual(self.baseline.confidence_level, 0.95)
    
    def test_control_limits_calculation(self):
        """Test statistical control limits calculation."""
        time_lower, time_upper, memory_lower, memory_upper = self.baseline.get_control_limits()
        
        # For 95% confidence level, z-score is 1.96
        expected_time_lower = max(0, 10.0 - 1.96 * 1.0)
        expected_time_upper = 10.0 + 1.96 * 1.0
        expected_memory_lower = max(0, 100.0 - 1.96 * 10.0)
        expected_memory_upper = 100.0 + 1.96 * 10.0
        
        self.assertAlmostEqual(time_lower, expected_time_lower, places=2)
        self.assertAlmostEqual(time_upper, expected_time_upper, places=2)
        self.assertAlmostEqual(memory_lower, expected_memory_lower, places=2)
        self.assertAlmostEqual(memory_upper, expected_memory_upper, places=2)
    
    def test_control_limits_99_percent(self):
        """Test control limits with 99% confidence level."""
        self.baseline.confidence_level = 0.99
        time_lower, time_upper, memory_lower, memory_upper = self.baseline.get_control_limits()
        
        # For 99% confidence level, z-score is 2.576
        expected_time_upper = 10.0 + 2.576 * 1.0
        self.assertAlmostEqual(time_upper, expected_time_upper, places=2)
    
    def test_serialization_deserialization(self):
        """Test JSON serialization and deserialization."""
        # Serialize to dict
        data = self.baseline.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data['scenario_name'], "test_scenario")
        self.assertEqual(data['mean_processing_time'], 10.0)
        self.assertEqual(data['last_updated'], '2025-01-12T12:00:00')
        
        # Deserialize from dict
        restored_baseline = PerformanceBaseline.from_dict(data)
        
        self.assertEqual(restored_baseline.scenario_name, self.baseline.scenario_name)
        self.assertEqual(restored_baseline.mean_processing_time, self.baseline.mean_processing_time)
        self.assertEqual(restored_baseline.last_updated, self.baseline.last_updated)
        self.assertEqual(restored_baseline.confidence_level, self.baseline.confidence_level)


class TestRegressionAlert(unittest.TestCase):
    """Test RegressionAlert creation and serialization."""
    
    def test_alert_creation(self):
        """Test regression alert creation."""
        detection_time = datetime.now()
        alert = RegressionAlert(
            scenario="test_scenario",
            metric_type="processing_time",
            current_value=15.0,
            baseline_mean=10.0,
            baseline_std=1.0,
            regression_severity=0.50,
            detection_time=detection_time,
            confidence_level=0.95
        )
        
        self.assertEqual(alert.scenario, "test_scenario")
        self.assertEqual(alert.metric_type, "processing_time")
        self.assertEqual(alert.regression_severity, 0.50)
        self.assertEqual(alert.detection_time, detection_time)
    
    def test_alert_serialization(self):
        """Test alert JSON serialization."""
        detection_time = datetime(2025, 1, 12, 15, 30, 0)
        alert = RegressionAlert(
            scenario="test_scenario",
            metric_type="memory_usage",
            current_value=150.0,
            baseline_mean=100.0,
            baseline_std=10.0,
            regression_severity=0.50,
            detection_time=detection_time,
            confidence_level=0.95
        )
        
        data = alert.to_dict()
        
        self.assertEqual(data['scenario'], "test_scenario")
        self.assertEqual(data['metric_type'], "memory_usage")
        self.assertEqual(data['regression_severity'], 0.50)
        self.assertEqual(data['detection_time'], '2025-01-12T15:30:00')


class TestPerformanceHistory(unittest.TestCase):
    """Test PerformanceHistory data management and trend analysis."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.history = PerformanceHistory(
            history_path=self.temp_dir,
            max_history_days=7
        )
        
        # Mock benchmark result
        self.mock_result = Mock(spec=BenchmarkResult)
        self.mock_result.processing_time = 10.0
        self.mock_result.mean_memory_usage = 100.0
        self.mock_result.success_rate = 1.0
        self.mock_result.total_files = 5
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_history_directory_creation(self):
        """Test that history directory is created."""
        self.assertTrue(self.temp_dir.exists())
        self.assertTrue(self.temp_dir.is_dir())
    
    def test_record_benchmark(self):
        """Test recording benchmark results."""
        self.history.record_benchmark("test_scenario", self.mock_result)
        
        history_file = self.temp_dir / "test_scenario_history.jsonl"
        self.assertTrue(history_file.exists())
        
        with open(history_file, 'r') as f:
            line = f.readline().strip()
            record = json.loads(line)
        
        self.assertEqual(record['processing_time'], 10.0)
        self.assertEqual(record['memory_usage'], 100.0)
        self.assertEqual(record['success_rate'], 1.0)
        self.assertEqual(record['total_files'], 5)
        self.assertIn('timestamp', record)
    
    def test_get_recent_history(self):
        """Test retrieving recent history records."""
        # Record multiple benchmark results
        for i in range(5):
            mock_result = Mock(spec=BenchmarkResult)
            mock_result.processing_time = 10.0 + i
            mock_result.mean_memory_usage = 100.0 + i
            mock_result.success_rate = 1.0
            mock_result.total_files = 5
            
            self.history.record_benchmark("test_scenario", mock_result)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get recent history
        records = self.history.get_recent_history("test_scenario", days=1)
        
        self.assertEqual(len(records), 5)
        self.assertEqual(records[0]['processing_time'], 10.0)
        self.assertEqual(records[-1]['processing_time'], 14.0)
        
        # Check sorted by timestamp
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in records]
        self.assertEqual(timestamps, sorted(timestamps))
    
    def test_trend_calculation(self):
        """Test performance trend calculation."""
        # Create trend data (improving performance - decreasing time)
        base_time = datetime.now() - timedelta(hours=5)
        
        for i in range(5):
            mock_result = Mock(spec=BenchmarkResult)
            mock_result.processing_time = 15.0 - i * 1.0  # Decreasing (improving)
            mock_result.mean_memory_usage = 100.0
            mock_result.success_rate = 1.0
            mock_result.total_files = 5
            
            # Mock timestamp for consistent trend
            with patch('src.giflab.monitoring.performance_regression.datetime') as mock_dt:
                mock_dt.now.return_value = base_time + timedelta(hours=i)
                mock_dt.fromisoformat = datetime.fromisoformat
                self.history.record_benchmark("trend_scenario", mock_result)
        
        # Calculate trend
        trend_slope = self.history.calculate_trend("trend_scenario", "processing_time", days=1)
        
        # Should be negative (improving performance)
        self.assertIsNotNone(trend_slope)
        self.assertLess(trend_slope, 0)
    
    def test_trend_insufficient_data(self):
        """Test trend calculation with insufficient data."""
        # Record only 2 data points (need at least 3)
        for i in range(2):
            self.history.record_benchmark("sparse_scenario", self.mock_result)
        
        trend_slope = self.history.calculate_trend("sparse_scenario", "processing_time", days=1)
        self.assertIsNone(trend_slope)
    
    def test_cleanup_old_records(self):
        """Test automatic cleanup of old records."""
        # Create history with very short retention (0 days)
        short_history = PerformanceHistory(
            history_path=self.temp_dir / "short",
            max_history_days=0  # Immediate cleanup
        )
        
        short_history.record_benchmark("cleanup_scenario", self.mock_result)
        
        # Small delay to ensure timestamp difference
        time.sleep(0.01)
        
        # Record another - should trigger cleanup
        short_history.record_benchmark("cleanup_scenario", self.mock_result)
        
        # Check that old records were cleaned up
        records = short_history.get_recent_history("cleanup_scenario", days=1)
        # Should have minimal records due to aggressive cleanup
        self.assertLessEqual(len(records), 2)


class TestRegressionDetector(unittest.TestCase):
    """Test RegressionDetector baseline management and regression detection."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.baseline_path = self.temp_dir / "baselines.json"
        self.detector = RegressionDetector(
            baseline_path=self.baseline_path,
            regression_threshold=0.10,  # 10%
            confidence_level=0.95
        )
        
        # Create mock benchmark results
        self.mock_results = []
        for i in range(5):
            result = Mock(spec=BenchmarkResult)
            result.processing_time = 10.0 + i * 0.5
            result.mean_memory_usage = 100.0 + i * 2.0
            self.mock_results.append(result)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_baseline_creation_and_persistence(self):
        """Test creating and persisting baselines."""
        self.detector.update_baseline("test_scenario", self.mock_results)
        
        # Check baseline was created
        self.assertIn("test_scenario", self.detector.baselines)
        baseline = self.detector.baselines["test_scenario"]
        
        self.assertEqual(baseline.scenario_name, "test_scenario")
        self.assertAlmostEqual(baseline.mean_processing_time, 11.0, places=1)  # Mean of 10-12
        self.assertEqual(baseline.sample_count, 5)
        
        # Check persistence
        self.assertTrue(self.baseline_path.exists())
        
        # Create new detector to test loading
        new_detector = RegressionDetector(
            baseline_path=self.baseline_path,
            regression_threshold=0.10,
            confidence_level=0.95
        )
        
        self.assertIn("test_scenario", new_detector.baselines)
        loaded_baseline = new_detector.baselines["test_scenario"]
        self.assertAlmostEqual(loaded_baseline.mean_processing_time, baseline.mean_processing_time, places=2)
    
    def test_insufficient_samples_baseline(self):
        """Test baseline creation with insufficient samples."""
        # Only 2 samples (need at least 3)
        insufficient_results = self.mock_results[:2]
        
        # Should log warning but not fail
        with patch('src.giflab.monitoring.performance_regression.logger') as mock_logger:
            self.detector.update_baseline("insufficient_scenario", insufficient_results, min_samples=3)
            mock_logger.warning.assert_called_once()
        
        # Should not have created baseline
        self.assertNotIn("insufficient_scenario", self.detector.baselines)
    
    def test_regression_detection_processing_time(self):
        """Test regression detection for processing time."""
        # Create baseline
        self.detector.update_baseline("perf_scenario", self.mock_results)
        
        # Create regressed result (50% slower)
        regressed_result = Mock(spec=BenchmarkResult)
        regressed_result.processing_time = 16.5  # 50% slower than 11.0 baseline
        regressed_result.mean_memory_usage = 100.0
        
        alerts = self.detector.detect_regressions("perf_scenario", regressed_result)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.scenario, "perf_scenario")
        self.assertEqual(alert.metric_type, "processing_time")
        self.assertGreater(alert.regression_severity, 0.10)  # Should exceed 10% threshold
        self.assertAlmostEqual(alert.current_value, 16.5, places=1)
    
    def test_regression_detection_memory_usage(self):
        """Test regression detection for memory usage."""
        # Create baseline
        self.detector.update_baseline("mem_scenario", self.mock_results)
        
        # Create regressed result (30% more memory usage)
        regressed_result = Mock(spec=BenchmarkResult)
        regressed_result.processing_time = 11.0  # Normal processing time
        regressed_result.mean_memory_usage = 130.0  # 30% more memory
        
        alerts = self.detector.detect_regressions("mem_scenario", regressed_result)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.metric_type, "memory_usage")
        self.assertGreater(alert.regression_severity, 0.20)  # Should be around 30%
    
    def test_no_regression_detection(self):
        """Test that no alerts are generated for good performance."""
        # Create baseline
        self.detector.update_baseline("good_scenario", self.mock_results)
        
        # Create good result (within baseline range)
        good_result = Mock(spec=BenchmarkResult)
        good_result.processing_time = 10.5  # Well within baseline
        good_result.mean_memory_usage = 95.0  # Slightly better than baseline
        
        alerts = self.detector.detect_regressions("good_scenario", good_result)
        
        self.assertEqual(len(alerts), 0)
    
    def test_no_baseline_available(self):
        """Test regression detection without baseline."""
        regressed_result = Mock(spec=BenchmarkResult)
        regressed_result.processing_time = 20.0
        regressed_result.mean_memory_usage = 200.0
        
        alerts = self.detector.detect_regressions("nonexistent_scenario", regressed_result)
        
        self.assertEqual(len(alerts), 0)
    
    def test_baseline_summary(self):
        """Test baseline summary generation."""
        # Add multiple baselines
        self.detector.update_baseline("scenario1", self.mock_results)
        self.detector.update_baseline("scenario2", self.mock_results)
        
        summary = self.detector.get_baseline_summary()
        
        self.assertEqual(summary['baseline_count'], 2)
        self.assertIn("scenario1", summary['scenarios'])
        self.assertIn("scenario2", summary['scenarios'])
        self.assertEqual(summary['regression_threshold'], 0.10)
        self.assertEqual(summary['confidence_level'], 0.95)
        self.assertIn('last_updated', summary)


class TestContinuousMonitor(unittest.TestCase):
    """Test ContinuousMonitor background monitoring and integration."""
    
    def setUp(self):
        # Create mock components
        self.mock_benchmarker = Mock()
        self.mock_detector = Mock()
        self.mock_history = Mock()
        self.mock_alert_manager = Mock()
        self.mock_metrics_collector = Mock()
        
        # Setup mock detector baselines
        self.mock_detector.baselines = {"test_scenario": Mock()}
        
        # Setup mock alert manager with iterable alerts
        self.mock_alert_manager.alerts = []
        
        self.monitor = ContinuousMonitor(
            benchmarker=self.mock_benchmarker,
            detector=self.mock_detector,
            history=self.mock_history,
            alert_manager=self.mock_alert_manager,
            metrics_collector=self.mock_metrics_collector,
            monitoring_interval=1  # 1 second for fast testing
        )
    
    def test_monitor_creation(self):
        """Test monitor creation and initial state."""
        self.assertFalse(self.monitor.enabled)
        self.assertIsNone(self.monitor.monitoring_thread)
        self.assertEqual(len(self.monitor.monitoring_scenarios), 1)  # Default scenario
    
    def test_monitor_start_stop(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        self.assertTrue(self.monitor.enabled)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        self.assertTrue(self.monitor.monitoring_thread.is_alive())
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.enabled)
    
    def test_monitoring_check_execution(self):
        """Test individual monitoring check execution."""
        # Setup mock benchmark results
        mock_result = Mock(spec=BenchmarkResult)
        mock_result.processing_time = 10.0
        mock_result.mean_memory_usage = 100.0
        self.mock_benchmarker.run_scenario.return_value = [mock_result]
        
        # Setup mock regression detection (no alerts)
        self.mock_detector.detect_regressions.return_value = []
        
        # Run monitoring check
        scenario = self.monitor.monitoring_scenarios[0]
        self.monitor._run_monitoring_check(scenario)
        
        # Verify calls
        self.mock_benchmarker.run_scenario.assert_called_once()
        self.mock_history.record_benchmark.assert_called_once_with(scenario.name, mock_result)
        self.mock_detector.detect_regressions.assert_called_once_with(scenario.name, mock_result)
        
        # Verify metrics recording
        self.mock_metrics_collector.record_timer.assert_called_once()
        self.mock_metrics_collector.record_gauge.assert_called_once()
    
    def test_regression_alert_sending(self):
        """Test sending regression alerts."""
        # Setup mock benchmark results with regression
        mock_result = Mock(spec=BenchmarkResult)
        mock_result.processing_time = 15.0  # Regressed
        mock_result.mean_memory_usage = 100.0
        self.mock_benchmarker.run_scenario.return_value = [mock_result]
        
        # Setup regression alert
        mock_alert = RegressionAlert(
            scenario="test_scenario",
            metric_type="processing_time",
            current_value=15.0,
            baseline_mean=10.0,
            baseline_std=1.0,
            regression_severity=0.50,
            detection_time=datetime.now(),
            confidence_level=0.95
        )
        self.mock_detector.detect_regressions.return_value = [mock_alert]
        
        # Mock alert manager
        self.mock_alert_manager.alerts = []
        
        # Run monitoring check
        scenario = self.monitor.monitoring_scenarios[0]
        self.monitor._run_monitoring_check(scenario)
        
        # Verify alert was added
        self.assertEqual(len(self.mock_alert_manager.alerts), 1)
        sent_alert = self.mock_alert_manager.alerts[0]
        self.assertIn("performance_regression", sent_alert.system)
        self.assertEqual(sent_alert.level, AlertLevel.CRITICAL)  # 50% regression
    
    def test_alert_severity_determination(self):
        """Test alert severity based on regression percentage."""
        # Test different regression levels
        severity_info = self.monitor._determine_alert_severity(0.10)  # 10%
        severity_warning = self.monitor._determine_alert_severity(0.30)  # 30%
        severity_critical = self.monitor._determine_alert_severity(0.60)  # 60%
        
        self.assertEqual(severity_info, AlertLevel.INFO)
        self.assertEqual(severity_warning, AlertLevel.WARNING)
        self.assertEqual(severity_critical, AlertLevel.CRITICAL)
    
    def test_monitoring_status(self):
        """Test monitoring status reporting."""
        status = self.monitor.get_monitoring_status()
        
        expected_keys = ['enabled', 'monitoring_interval', 'scenarios_monitored', 
                        'thread_alive', 'detector_baselines', 'recent_alerts']
        for key in expected_keys:
            self.assertIn(key, status)
        
        self.assertEqual(status['enabled'], False)
        self.assertEqual(status['monitoring_interval'], 1)
        self.assertEqual(status['scenarios_monitored'], 1)
    
    def test_monitoring_with_no_results(self):
        """Test monitoring check with no benchmark results."""
        # Setup empty results
        self.mock_benchmarker.run_scenario.return_value = []
        
        # Run monitoring check (should not crash)
        scenario = self.monitor.monitoring_scenarios[0]
        self.monitor._run_monitoring_check(scenario)
        
        # Verify no further calls were made
        self.mock_history.record_benchmark.assert_not_called()
        self.mock_detector.detect_regressions.assert_not_called()
    
    def test_monitoring_exception_handling(self):
        """Test monitoring resilience to exceptions."""
        # Setup benchmarker to raise exception
        self.mock_benchmarker.run_scenario.side_effect = Exception("Test exception")
        
        # Run monitoring check (should not crash)
        scenario = self.monitor.monitoring_scenarios[0]
        
        # Should not raise exception
        try:
            self.monitor._run_monitoring_check(scenario)
        except Exception as e:
            self.fail(f"Monitoring check raised exception: {e}")


class TestMonitorIntegration(unittest.TestCase):
    """Integration tests for the complete monitoring system."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.giflab.monitoring.performance_regression.get_alert_manager')
    @patch('src.giflab.monitoring.performance_regression.get_metrics_collector')
    @patch('src.giflab.monitoring.performance_regression.Phase43Benchmarker')
    def test_create_performance_monitor(self, mock_benchmarker_class, mock_get_metrics, mock_get_alerts):
        """Test performance monitor factory function."""
        # Setup mocks
        mock_alert_manager = Mock()
        mock_metrics_collector = Mock()
        mock_get_alerts.return_value = mock_alert_manager
        mock_get_metrics.return_value = mock_metrics_collector
        
        mock_benchmarker = Mock()
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Create monitor
        monitor = create_performance_monitor(data_dir=self.temp_dir)
        
        # Verify components
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor.benchmarker, mock_benchmarker)
        self.assertEqual(monitor.alert_manager, mock_alert_manager)
        self.assertEqual(monitor.metrics_collector, mock_metrics_collector)
        
        # Verify paths
        self.assertEqual(monitor.detector.baseline_path, self.temp_dir / "performance_baselines.json")
        self.assertEqual(monitor.history.history_path, self.temp_dir / "performance_history")
    
    def test_end_to_end_baseline_and_detection(self):
        """Test end-to-end baseline creation and regression detection."""
        # Create real components (with temp directories)
        baseline_path = self.temp_dir / "baselines.json"
        history_path = self.temp_dir / "history"
        
        detector = RegressionDetector(
            baseline_path=baseline_path,
            regression_threshold=0.20,  # 20% threshold
            confidence_level=0.95
        )
        
        history = PerformanceHistory(
            history_path=history_path,
            max_history_days=7
        )
        
        # Create baseline with mock results
        baseline_results = []
        for i in range(3):
            result = Mock(spec=BenchmarkResult)
            result.processing_time = 10.0 + i * 0.5
            result.mean_memory_usage = 100.0 + i * 2.0
            baseline_results.append(result)
        
        detector.update_baseline("integration_test", baseline_results)
        
        # Test good performance (no regression)
        good_result = Mock(spec=BenchmarkResult)
        good_result.processing_time = 10.5
        good_result.mean_memory_usage = 101.0
        
        history.record_benchmark("integration_test", good_result)
        alerts = detector.detect_regressions("integration_test", good_result)
        
        self.assertEqual(len(alerts), 0)  # No regression
        
        # Test regressed performance
        bad_result = Mock(spec=BenchmarkResult)
        bad_result.processing_time = 14.0  # ~27% regression
        bad_result.mean_memory_usage = 130.0  # ~28% regression
        
        history.record_benchmark("integration_test", bad_result)
        alerts = detector.detect_regressions("integration_test", bad_result)
        
        self.assertGreater(len(alerts), 0)  # Should have regressions
        
        # Verify alert details
        time_alert = next((a for a in alerts if a.metric_type == "processing_time"), None)
        memory_alert = next((a for a in alerts if a.metric_type == "memory_usage"), None)
        
        self.assertIsNotNone(time_alert)
        self.assertIsNotNone(memory_alert)
        self.assertGreater(time_alert.regression_severity, 0.20)
        self.assertGreater(memory_alert.regression_severity, 0.20)
        
        # Test history and trend
        records = history.get_recent_history("integration_test", days=1)
        self.assertEqual(len(records), 2)  # Good result + bad result
        
        trend = history.calculate_trend("integration_test", "processing_time", days=1)
        self.assertIsNotNone(trend)
        self.assertGreater(trend, 0)  # Degrading trend (positive slope)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)