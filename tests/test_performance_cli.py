"""
Tests for Phase 7: Performance Monitoring CLI Commands

This test suite covers the performance monitoring CLI commands:
- performance status
- performance baseline
- performance monitor
- performance history
- performance validate
- performance ci

Test Categories:
- CLI command execution and output
- JSON output format validation
- Error handling and edge cases
- Integration with performance monitoring system
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from click.testing import CliRunner
import pytest

from src.giflab.cli.performance_cmd import (
    performance,
    performance_status,
    performance_baseline,
    performance_monitor,
    performance_history,
    performance_validate,
    performance_ci
)


class TestPerformanceStatusCommand(unittest.TestCase):
    """Test 'performance status' command."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_status_disabled(self, mock_create_monitor):
        """Test status command when performance monitoring is disabled."""
        # Setup mocks
        mock_monitor = Mock()
        mock_monitor.get_monitoring_status.return_value = {
            'enabled': False,
            'monitoring_interval': 3600,
            'thread_alive': False
        }
        mock_monitor.detector.get_baseline_summary.return_value = {
            'baseline_count': 0,
            'scenarios': [],
            'last_updated': 'never',
            'regression_threshold': 0.10,
            'confidence_level': 0.95
        }
        mock_monitor.alert_manager.get_active_alerts.return_value = []
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_status)
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("DISABLED", result.output)
        self.assertIn("No performance baselines available", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_status_enabled_with_baselines(self, mock_create_monitor):
        """Test status command when performance monitoring is enabled with baselines."""
        # Setup mocks
        mock_monitor = Mock()
        mock_monitor.get_monitoring_status.return_value = {
            'enabled': True,
            'monitoring_interval': 3600,
            'thread_alive': True
        }
        mock_monitor.detector.get_baseline_summary.return_value = {
            'baseline_count': 2,
            'scenarios': ['scenario1', 'scenario2'],
            'last_updated': '2025-01-12T12:00:00',
            'regression_threshold': 0.10,
            'confidence_level': 0.95
        }
        mock_monitor.alert_manager.get_active_alerts.return_value = []
        mock_create_monitor.return_value = mock_monitor
        
        with patch.dict(os.environ, {'GIFLAB_ENABLE_PERFORMANCE_MONITORING': 'true'}):
            result = self.runner.invoke(performance_status)
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ENABLED", result.output)
        self.assertIn("RUNNING", result.output)
        self.assertIn("scenario1", result.output)
        self.assertIn("scenario2", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_status_json_output(self, mock_create_monitor):
        """Test status command with JSON output."""
        # Setup mocks
        mock_monitor = Mock()
        mock_monitor.get_monitoring_status.return_value = {
            'enabled': False,
            'monitoring_interval': 3600,
            'thread_alive': False
        }
        mock_monitor.detector.get_baseline_summary.return_value = {
            'baseline_count': 0,
            'scenarios': [],
            'last_updated': 'never',
            'regression_threshold': 0.10,
            'confidence_level': 0.95
        }
        mock_monitor.alert_manager.get_active_alerts.return_value = []
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_status, ['--json'])
        
        self.assertEqual(result.exit_code, 0)
        
        # Validate JSON output
        output_data = json.loads(result.output)
        self.assertIn('performance_monitoring', output_data)
        self.assertIn('baselines', output_data)
        self.assertIn('alerts', output_data)
        self.assertIn('configuration', output_data)
        self.assertFalse(output_data['performance_monitoring']['enabled'])
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_status_with_alerts(self, mock_create_monitor):
        """Test status command with active performance alerts."""
        # Create mock alert
        mock_alert = Mock()
        mock_alert.name = "performance_regression_test_scenario_processing_time"
        mock_alert.severity.name = "WARNING"
        mock_alert.message = "Performance regression detected"
        mock_alert.timestamp.isoformat.return_value = "2025-01-12T15:30:00"
        mock_alert.timestamp.strftime.return_value = "15:30:00"
        mock_alert.details = {
            'scenario': 'test_scenario',
            'regression_percentage': '25.5%'
        }
        
        # Setup mocks
        mock_monitor = Mock()
        mock_monitor.get_monitoring_status.return_value = {
            'enabled': True,
            'monitoring_interval': 3600,
            'thread_alive': True
        }
        mock_monitor.detector.get_baseline_summary.return_value = {
            'baseline_count': 1,
            'scenarios': ['test_scenario'],
            'last_updated': '2025-01-12T12:00:00',
            'regression_threshold': 0.10,
            'confidence_level': 0.95
        }
        mock_monitor.alert_manager.get_active_alerts.return_value = [mock_alert]
        mock_create_monitor.return_value = mock_monitor
        
        with patch.dict(os.environ, {'GIFLAB_ENABLE_PERFORMANCE_MONITORING': 'true'}):
            result = self.runner.invoke(performance_status)
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Recent Performance Alerts", result.output)
        self.assertIn("test_scenario", result.output)
        self.assertIn("WARNING", result.output)
        self.assertIn("25.5%", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_status_error_handling(self, mock_create_monitor):
        """Test status command error handling."""
        mock_create_monitor.side_effect = Exception("Test error")
        
        result = self.runner.invoke(performance_status)
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error getting performance status", result.output)


class TestPerformanceBaselineCommand(unittest.TestCase):
    """Test 'performance baseline' command."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_baseline_list_empty(self, mock_create_monitor):
        """Test baseline list with no existing baselines."""
        mock_monitor = Mock()
        mock_monitor.detector.get_baseline_summary.return_value = {
            'baseline_count': 0,
            'scenarios': []
        }
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_baseline, ['list'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No baselines available", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_baseline_list_with_data(self, mock_create_monitor):
        """Test baseline list with existing baselines."""
        from datetime import datetime
        
        # Create mock baseline
        mock_baseline = Mock()
        mock_baseline.mean_processing_time = 10.0
        mock_baseline.std_processing_time = 1.0
        mock_baseline.mean_memory_usage = 100.0
        mock_baseline.std_memory_usage = 5.0
        mock_baseline.sample_count = 5
        mock_baseline.last_updated = datetime(2025, 1, 12, 15, 30)
        
        mock_detector = Mock()
        mock_detector.get_baseline_summary.return_value = {
            'baseline_count': 1,
            'scenarios': ['test_scenario']
        }
        mock_detector.baselines = {'test_scenario': mock_baseline}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_baseline, ['list'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Performance Baselines", result.output)
        self.assertIn("test_scenario", result.output)
        self.assertIn("10.00s ± 1.00", result.output)
        self.assertIn("100.0MB ± 5.0", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_baseline_create(self, mock_create_monitor):
        """Test baseline creation."""
        # Create mock benchmark results
        mock_result = Mock()
        mock_result.processing_time = 10.0
        mock_result.mean_memory_usage = 100.0
        
        mock_benchmarker = Mock()
        mock_benchmarker.get_available_scenarios.return_value = []
        mock_benchmarker.run_scenario.return_value = [mock_result, mock_result, mock_result]
        
        mock_detector = Mock()
        mock_detector.regression_threshold = 0.1
        mock_detector.detector.regression_threshold = 3  # For min samples check
        mock_detector.baselines = {}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_monitor.benchmarker = mock_benchmarker
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_baseline, ['create', '--iterations', '3'])
        
        self.assertEqual(result.exit_code, 0)
        mock_detector.update_baseline.assert_called()
    
    @patch('src.giflab.cli.performance_cmd.console')
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_baseline_create_json_output(self, mock_create_monitor, mock_console):
        """Test baseline creation with JSON output."""
        # Setup mocks for successful baseline creation
        mock_result = Mock()
        mock_result.processing_time = 10.0
        mock_result.mean_memory_usage = 100.0
        
        mock_benchmarker = Mock()
        mock_benchmarker.get_available_scenarios.return_value = []
        mock_benchmarker.run_scenario.return_value = [mock_result, mock_result, mock_result]
        
        mock_detector = Mock()
        mock_detector.regression_threshold = 0.1
        mock_detector.detector.regression_threshold = 3
        mock_detector.baselines = {}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_monitor.benchmarker = mock_benchmarker
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_baseline, ['create', '--json'])
        
        self.assertEqual(result.exit_code, 0)
        # Should be valid JSON
        output_data = json.loads(result.output)
        self.assertIsInstance(output_data, dict)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_baseline_clear(self, mock_create_monitor):
        """Test baseline clearing."""
        mock_baselines = Mock()
        mock_baselines.__contains__ = Mock(return_value=True)  # For dict-like behavior
        
        mock_detector = Mock()
        mock_detector.baselines = mock_baselines
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_baseline, ['clear'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cleared all baselines", result.output)
        mock_baselines.clear.assert_called_once()
        mock_detector._save_baselines.assert_called_once()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_baseline_clear_specific(self, mock_create_monitor):
        """Test clearing specific baseline."""
        mock_detector = Mock()
        mock_detector.baselines = {'test_scenario': Mock(), 'other_scenario': Mock()}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_baseline, ['clear', '--scenario', 'test_scenario'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cleared baseline for test_scenario", result.output)


class TestPerformanceMonitorCommand(unittest.TestCase):
    """Test 'performance monitor' command."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_monitor_start_disabled(self, mock_create_monitor):
        """Test monitor start when performance monitoring is disabled."""
        mock_monitor = Mock()
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_monitor, ['start'])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Performance monitoring is disabled", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_monitor_start_no_baselines(self, mock_create_monitor):
        """Test monitor start with no baselines."""
        mock_detector = Mock()
        mock_detector.baselines = {}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        with patch.dict(os.environ, {'GIFLAB_ENABLE_PERFORMANCE_MONITORING': 'true'}):
            result = self.runner.invoke(performance_monitor, ['start'])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("No performance baselines available", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_monitor_start_success(self, mock_create_monitor):
        """Test successful monitor start."""
        mock_detector = Mock()
        mock_detector.baselines = {'test_scenario': Mock()}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_monitor.monitoring_interval = 3600
        mock_monitor.monitoring_scenarios = [Mock()]
        mock_create_monitor.return_value = mock_monitor
        
        with patch.dict(os.environ, {'GIFLAB_ENABLE_PERFORMANCE_MONITORING': 'true'}):
            result = self.runner.invoke(performance_monitor, ['start'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Performance monitoring started", result.output)
        mock_monitor.start_monitoring.assert_called_once()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_monitor_stop(self, mock_create_monitor):
        """Test monitor stop."""
        mock_monitor = Mock()
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_monitor, ['stop'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Performance monitoring stopped", result.output)
        mock_monitor.stop_monitoring.assert_called_once()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_monitor_status(self, mock_create_monitor):
        """Test monitor status."""
        mock_monitor = Mock()
        mock_monitor.get_monitoring_status.return_value = {
            'enabled': True,
            'monitoring_interval': 3600,
            'scenarios_monitored': 1,
            'detector_baselines': 2,
            'recent_alerts': 0
        }
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_monitor, ['status'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Continuous Monitoring Status", result.output)
        self.assertIn("RUNNING", result.output)


class TestPerformanceValidateCommand(unittest.TestCase):
    """Test 'performance validate' command."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_validate_phase6_disabled(self, mock_create_monitor):
        """Test validation when Phase 6 optimizations are disabled."""
        mock_monitor = Mock()
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_validate, ['--check-phase6'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Phase 6 optimizations are disabled", result.output)
        self.assertIn("UNKNOWN", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    @patch('src.giflab.cli.performance_cmd.Phase43Benchmarker')
    def test_validate_phase6_enabled_optimal(self, mock_benchmarker_class, mock_create_monitor):
        """Test validation with optimal Phase 6 performance."""
        from datetime import datetime
        
        # Mock benchmark result (fast performance)
        mock_result = Mock()
        mock_result.processing_time = 2.0  # Fast processing
        
        mock_benchmarker = Mock()
        mock_benchmarker.run_scenario.return_value = [mock_result]
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Mock baseline for comparison
        mock_baseline = Mock()
        mock_baseline.mean_processing_time = 10.0  # Baseline
        
        mock_detector = Mock()
        mock_detector.baselines = {'small_gif_basic': mock_baseline}
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        with patch.dict(os.environ, {'GIFLAB_ENABLE_PHASE6_OPTIMIZATION': 'true'}):
            result = self.runner.invoke(performance_validate, ['--check-phase6'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Phase 6 optimizations performing optimally", result.output)
        self.assertIn("EXCELLENT", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_validate_json_output(self, mock_create_monitor):
        """Test validation with JSON output."""
        mock_monitor = Mock()
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_validate, ['--json'])
        
        self.assertEqual(result.exit_code, 0)
        
        # Should be valid JSON
        output_data = json.loads(result.output)
        self.assertIn('timestamp', output_data)
        self.assertIn('overall_status', output_data)


class TestPerformanceCICommand(unittest.TestCase):
    """Test 'performance ci' command."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    @patch('src.giflab.cli.performance_cmd.Phase43Benchmarker')
    def test_ci_check_passing(self, mock_benchmarker_class, mock_create_monitor):
        """Test CI check with passing performance."""
        # Mock good benchmark result
        mock_result = Mock()
        mock_result.processing_time = 10.0
        mock_result.mean_memory_usage = 100.0
        mock_result.success_rate = 1.0
        
        mock_benchmarker = Mock()
        mock_benchmarker.run_scenario.return_value = [mock_result]
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Mock detector with no regressions
        mock_detector = Mock()
        mock_detector.detect_regressions.return_value = []
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_ci, ['check'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("CI Performance Check", result.output)
        self.assertIn("PASSED", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    @patch('src.giflab.cli.performance_cmd.Phase43Benchmarker')
    def test_ci_gate_failing(self, mock_benchmarker_class, mock_create_monitor):
        """Test CI gate with failing performance (should exit with code 1)."""
        from src.giflab.monitoring.performance_regression import RegressionAlert
        from datetime import datetime
        
        # Mock regressed benchmark result
        mock_result = Mock()
        mock_result.processing_time = 15.0  # Regressed
        mock_result.mean_memory_usage = 130.0  # Regressed
        mock_result.success_rate = 1.0
        
        mock_benchmarker = Mock()
        mock_benchmarker.run_scenario.return_value = [mock_result]
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Mock detector with significant regression
        mock_regression = RegressionAlert(
            scenario="test_scenario",
            metric_type="processing_time", 
            current_value=15.0,
            baseline_mean=10.0,
            baseline_std=1.0,
            regression_severity=0.50,  # 50% regression
            detection_time=datetime.now(),
            confidence_level=0.95
        )
        
        mock_detector = Mock()
        mock_detector.detect_regressions.return_value = [mock_regression]
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_ci, ['gate'])
        
        self.assertEqual(result.exit_code, 1)  # Should fail CI
        self.assertIn("CI Performance Check: FAILED", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_ci_json_output(self, mock_create_monitor):
        """Test CI command with JSON output."""
        mock_detector = Mock()
        mock_detector.detect_regressions.return_value = []
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        # Mock benchmarker to return empty results (no tests run)
        with patch('src.giflab.cli.performance_cmd.Phase43Benchmarker') as mock_benchmarker_class:
            mock_benchmarker = Mock()
            mock_benchmarker.run_scenario.return_value = []
            mock_benchmarker_class.return_value = mock_benchmarker
            
            result = self.runner.invoke(performance_ci, ['check', '--json'])
        
        self.assertEqual(result.exit_code, 0)
        
        # Should be valid JSON
        output_data = json.loads(result.output)
        self.assertIn('action', output_data)
        self.assertIn('threshold', output_data)
        self.assertIn('scenarios_tested', output_data)
        self.assertIn('overall_result', output_data)


class TestPerformanceHistoryCommand(unittest.TestCase):
    """Test 'performance history' command."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_history_no_records(self, mock_create_monitor):
        """Test history command with no historical records."""
        mock_history = Mock()
        mock_history.get_recent_history.return_value = []
        mock_history.calculate_trend.return_value = None
        
        mock_detector = Mock()
        mock_detector.baselines = {'test_scenario': Mock()}
        
        mock_monitor = Mock()
        mock_monitor.history = mock_history
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_history, ['--scenario', 'test_scenario'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No history available for test_scenario", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_history_with_records(self, mock_create_monitor):
        """Test history command with historical records."""
        from datetime import datetime
        
        # Mock historical records
        mock_records = [
            {
                'timestamp': '2025-01-12T10:00:00',
                'processing_time': 9.0,
                'memory_usage': 95.0,
                'success_rate': 1.0
            },
            {
                'timestamp': '2025-01-12T11:00:00',
                'processing_time': 10.0,
                'memory_usage': 100.0,
                'success_rate': 1.0
            }
        ]
        
        mock_history = Mock()
        mock_history.get_recent_history.return_value = mock_records
        mock_history.calculate_trend.return_value = 0.1  # Slightly degrading
        
        mock_detector = Mock()
        mock_detector.baselines = {'test_scenario': Mock()}
        
        mock_monitor = Mock()
        mock_monitor.history = mock_history
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_history, ['--scenario', 'test_scenario', '--trend'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Performance History", result.output)
        self.assertIn("test_scenario", result.output)
        self.assertIn("Records: 2", result.output)
        self.assertIn("Trend: 📉 Degrading", result.output)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_history_json_output(self, mock_create_monitor):
        """Test history command with JSON output."""
        mock_history = Mock()
        mock_history.get_recent_history.return_value = []
        mock_history.calculate_trend.return_value = None
        
        mock_detector = Mock()
        mock_detector.baselines = {'test_scenario': Mock()}
        
        mock_monitor = Mock()
        mock_monitor.history = mock_history
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_history, ['--json'])
        
        self.assertEqual(result.exit_code, 0)
        
        # Should be valid JSON
        output_data = json.loads(result.output)
        self.assertIsInstance(output_data, dict)
        self.assertIn('test_scenario', output_data)
    
    @patch('src.giflab.cli.performance_cmd.create_performance_monitor')
    def test_history_nonexistent_scenario(self, mock_create_monitor):
        """Test history command with nonexistent scenario."""
        mock_detector = Mock()
        mock_detector.baselines = {}  # No baselines
        
        mock_monitor = Mock()
        mock_monitor.detector = mock_detector
        mock_create_monitor.return_value = mock_monitor
        
        result = self.runner.invoke(performance_history, ['--scenario', 'nonexistent'])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("No history found for scenario: nonexistent", result.output)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)