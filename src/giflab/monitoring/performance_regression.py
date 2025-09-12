"""
Phase 7: Continuous Performance Monitoring & Alerting System

This module implements automated performance regression detection to protect
the transformational Phase 6 performance gains (5.04x speedup) and ensure
ongoing performance stability in production environments.

Key Components:
- PerformanceBaseline: Statistical baseline tracking with confidence intervals
- RegressionDetector: Automated regression analysis with configurable thresholds
- ContinuousMonitor: Background monitoring with alert integration
- PerformanceHistory: Historical performance data management

Architecture:
- Builds on Phase 4.3 benchmarking infrastructure
- Integrates with existing AlertManager from Phase 3.1
- Uses MetricsCollector for data persistence
- Maintains <1% performance overhead (matching memory monitoring)
"""

import json
import logging
import statistics
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from ..config import MONITORING
from .alerting import AlertManager, Alert, AlertLevel
from .metrics_collector import MetricsCollector
from ..benchmarks.phase_4_3_benchmarking import Phase43Benchmarker, BenchmarkResult, BenchmarkScenario

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Statistical baseline for performance regression detection."""
    scenario_name: str
    mean_processing_time: float
    std_processing_time: float
    mean_memory_usage: float
    std_memory_usage: float
    sample_count: int
    last_updated: datetime
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['last_updated'] = self.last_updated.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary (JSON deserialization)."""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)
    
    def get_control_limits(self) -> Tuple[float, float, float, float]:
        """Calculate statistical control limits for regression detection."""
        # 3-sigma control limits for processing time and memory
        z_score = 2.576 if self.confidence_level == 0.99 else 1.96  # 99% or 95%
        
        time_lower = max(0, self.mean_processing_time - z_score * self.std_processing_time)
        time_upper = self.mean_processing_time + z_score * self.std_processing_time
        
        memory_lower = max(0, self.mean_memory_usage - z_score * self.std_memory_usage)
        memory_upper = self.mean_memory_usage + z_score * self.std_memory_usage
        
        return time_lower, time_upper, memory_lower, memory_upper


@dataclass 
class RegressionAlert:
    """Performance regression alert details."""
    scenario: str
    metric_type: str  # 'processing_time' or 'memory_usage'
    current_value: float
    baseline_mean: float
    baseline_std: float
    regression_severity: float  # % regression from baseline
    detection_time: datetime
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['detection_time'] = self.detection_time.isoformat()
        return result


class PerformanceHistory:
    """Historical performance data management with trend analysis."""
    
    def __init__(self, history_path: Path, max_history_days: int = 30):
        self.history_path = history_path
        self.max_history_days = max_history_days
        self.lock = RLock()
        self._ensure_history_directory()
    
    def _ensure_history_directory(self):
        """Ensure history directory exists."""
        self.history_path.mkdir(parents=True, exist_ok=True)
    
    def record_benchmark(self, scenario: str, result: BenchmarkResult):
        """Record benchmark result with timestamp."""
        timestamp = datetime.now()
        
        with self.lock:
            history_file = self.history_path / f"{scenario}_history.jsonl"
            
            record = {
                'timestamp': timestamp.isoformat(),
                'processing_time': result.processing_time,
                'memory_usage': result.mean_memory_usage,
                'success_rate': result.success_rate,
                'total_files': result.total_files,
                'phase6_enabled': getattr(result, 'phase6_enabled', False)
            }
            
            # Append to history file (JSONL format)
            with open(history_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
            
            # Cleanup old records
            self._cleanup_old_records(history_file)
    
    def _cleanup_old_records(self, history_file: Path):
        """Remove records older than max_history_days."""
        if not history_file.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        temp_file = history_file.with_suffix('.tmp')
        
        try:
            with open(history_file, 'r') as infile, open(temp_file, 'w') as outfile:
                for line in infile:
                    try:
                        record = json.loads(line.strip())
                        record_time = datetime.fromisoformat(record['timestamp'])
                        if record_time >= cutoff_date:
                            outfile.write(line)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip malformed records
                        continue
            
            temp_file.replace(history_file)
            
        except Exception as e:
            logger.warning(f"Failed to cleanup history file {history_file}: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def get_recent_history(self, scenario: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent performance history for trend analysis."""
        history_file = self.history_path / f"{scenario}_history.jsonl"
        
        if not history_file.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = []
        
        with self.lock:
            try:
                with open(history_file, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            record_time = datetime.fromisoformat(record['timestamp'])
                            if record_time >= cutoff_date:
                                recent_records.append(record)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except FileNotFoundError:
                return []
        
        return sorted(recent_records, key=lambda x: x['timestamp'])
    
    def calculate_trend(self, scenario: str, metric: str = 'processing_time', days: int = 7) -> Optional[float]:
        """Calculate performance trend (slope) over recent history."""
        history = self.get_recent_history(scenario, days)
        
        if len(history) < 3:  # Need at least 3 data points for trend
            return None
        
        timestamps = [datetime.fromisoformat(record['timestamp']).timestamp() for record in history]
        values = [record.get(metric, 0) for record in history]
        
        if not all(v > 0 for v in values):  # Skip if any invalid values
            return None
        
        # Calculate linear regression slope
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


class RegressionDetector:
    """Automated performance regression detection with statistical analysis."""
    
    def __init__(self, 
                 baseline_path: Path,
                 regression_threshold: float = 0.10,  # 10% regression threshold
                 confidence_level: float = 0.95):
        self.baseline_path = baseline_path
        self.regression_threshold = regression_threshold
        self.confidence_level = confidence_level
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.lock = RLock()
        self._load_baselines()
    
    def _load_baselines(self):
        """Load existing performance baselines from disk."""
        if not self.baseline_path.exists():
            logger.info("No existing baselines found, will create new ones")
            return
        
        with self.lock:
            try:
                with open(self.baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                
                for scenario_name, data in baseline_data.items():
                    try:
                        self.baselines[scenario_name] = PerformanceBaseline.from_dict(data)
                        logger.debug(f"Loaded baseline for scenario: {scenario_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load baseline for {scenario_name}: {e}")
                        
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
                
            except Exception as e:
                logger.error(f"Failed to load baselines from {self.baseline_path}: {e}")
    
    def _save_baselines(self):
        """Save performance baselines to disk."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.lock:
            try:
                baseline_data = {
                    name: baseline.to_dict() 
                    for name, baseline in self.baselines.items()
                }
                
                with open(self.baseline_path, 'w') as f:
                    json.dump(baseline_data, f, indent=2)
                
                logger.debug(f"Saved {len(self.baselines)} baselines to {self.baseline_path}")
                
            except Exception as e:
                logger.error(f"Failed to save baselines: {e}")
    
    def update_baseline(self, scenario: str, results: List[BenchmarkResult], min_samples: int = 3):
        """Update performance baseline with new benchmark results."""
        if len(results) < min_samples:
            logger.warning(f"Insufficient samples ({len(results)}) for baseline update, need at least {min_samples}")
            return
        
        # Extract performance metrics
        processing_times = [r.processing_time for r in results]
        memory_usages = [r.mean_memory_usage for r in results]
        
        # Calculate statistics
        mean_time = statistics.mean(processing_times)
        std_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        mean_memory = statistics.mean(memory_usages)
        std_memory = statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
        
        # Create or update baseline
        baseline = PerformanceBaseline(
            scenario_name=scenario,
            mean_processing_time=mean_time,
            std_processing_time=std_time,
            mean_memory_usage=mean_memory,
            std_memory_usage=std_memory,
            sample_count=len(results),
            last_updated=datetime.now(),
            confidence_level=self.confidence_level
        )
        
        with self.lock:
            self.baselines[scenario] = baseline
            self._save_baselines()
        
        logger.info(f"Updated baseline for {scenario}: {mean_time:.3f}s ± {std_time:.3f}s, "
                   f"{mean_memory:.1f}MB ± {std_memory:.1f}MB (n={len(results)})")
    
    def detect_regressions(self, scenario: str, result: BenchmarkResult) -> List[RegressionAlert]:
        """Detect performance regressions by comparing against baseline."""
        if scenario not in self.baselines:
            logger.warning(f"No baseline available for scenario: {scenario}")
            return []
        
        baseline = self.baselines[scenario]
        alerts = []
        
        # Check processing time regression
        time_regression = self._calculate_regression_percentage(
            result.processing_time, baseline.mean_processing_time
        )
        
        if time_regression > self.regression_threshold:
            alerts.append(RegressionAlert(
                scenario=scenario,
                metric_type='processing_time',
                current_value=result.processing_time,
                baseline_mean=baseline.mean_processing_time,
                baseline_std=baseline.std_processing_time,
                regression_severity=time_regression,
                detection_time=datetime.now(),
                confidence_level=self.confidence_level
            ))
        
        # Check memory usage regression
        memory_regression = self._calculate_regression_percentage(
            result.mean_memory_usage, baseline.mean_memory_usage
        )
        
        if memory_regression > self.regression_threshold:
            alerts.append(RegressionAlert(
                scenario=scenario,
                metric_type='memory_usage',
                current_value=result.mean_memory_usage,
                baseline_mean=baseline.mean_memory_usage,
                baseline_std=baseline.std_memory_usage,
                regression_severity=memory_regression,
                detection_time=datetime.now(),
                confidence_level=self.confidence_level
            ))
        
        return alerts
    
    def _calculate_regression_percentage(self, current: float, baseline: float) -> float:
        """Calculate percentage regression (positive means worse performance)."""
        if baseline <= 0:
            return 0.0
        return max(0.0, (current - baseline) / baseline)
    
    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get summary of all performance baselines."""
        with self.lock:
            return {
                'baseline_count': len(self.baselines),
                'scenarios': list(self.baselines.keys()),
                'last_updated': max(
                    (b.last_updated for b in self.baselines.values()),
                    default=datetime.now()
                ).isoformat(),
                'regression_threshold': self.regression_threshold,
                'confidence_level': self.confidence_level
            }


class ContinuousMonitor:
    """Background performance monitoring with alert integration."""
    
    def __init__(self,
                 benchmarker: Phase43Benchmarker,
                 detector: RegressionDetector,
                 history: PerformanceHistory,
                 alert_manager: AlertManager,
                 metrics_collector: Optional[MetricsCollector] = None,
                 monitoring_interval: int = 3600):  # 1 hour default
        
        self.benchmarker = benchmarker
        self.detector = detector
        self.history = history
        self.alert_manager = alert_manager
        self.metrics_collector = metrics_collector
        self.monitoring_interval = monitoring_interval
        
        self.enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.lock = RLock()
        
        # Monitoring scenarios (lightweight subset for continuous monitoring)
        self.monitoring_scenarios = [
            BenchmarkScenario(
                name="continuous_small_gif",
                description="Small GIF continuous monitoring",
                frame_count=20,
                frame_size=(50, 50),
                metrics_to_test=["processing_time", "memory_usage"],
                expected_duration_range=(0.1, 2.0)
            )
        ]
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        with self.lock:
            if self.enabled:
                logger.warning("Performance monitoring already running")
                return
            
            self.enabled = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="PerformanceMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            
        logger.info(f"Started continuous performance monitoring (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        with self.lock:
            if not self.enabled:
                return
            
            self.enabled = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped continuous performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info("Performance monitoring loop started")
        
        while self.enabled:
            try:
                for scenario in self.monitoring_scenarios:
                    if not self.enabled:  # Check if we should stop
                        break
                    
                    self._run_monitoring_check(scenario)
                
                # Sleep with periodic checks for shutdown
                sleep_intervals = max(1, self.monitoring_interval // 10)
                for _ in range(sleep_intervals):
                    if not self.enabled:
                        break
                    time.sleep(self.monitoring_interval / sleep_intervals)
                    
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                # Sleep before retrying
                time.sleep(min(60, self.monitoring_interval))
        
        logger.info("Performance monitoring loop stopped")
    
    def _run_monitoring_check(self, scenario: BenchmarkScenario):
        """Run performance check for a single scenario."""
        try:
            logger.debug(f"Running monitoring check for scenario: {scenario.name}")
            
            # Run lightweight benchmark (single iteration)
            results = self.benchmarker.run_scenario(scenario, iterations=1)
            
            if not results:
                logger.warning(f"No results from monitoring check: {scenario.name}")
                return
            
            result = results[0]
            
            # Record in history
            self.history.record_benchmark(scenario.name, result)
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_timer(
                    f"performance_monitor.processing_time.{scenario.name}",
                    result.processing_time
                )
                self.metrics_collector.record_gauge(
                    f"performance_monitor.memory_usage.{scenario.name}",
                    result.mean_memory_usage
                )
            
            # Check for regressions
            regression_alerts = self.detector.detect_regressions(scenario.name, result)
            
            # Send alerts if regressions detected
            for reg_alert in regression_alerts:
                self._send_regression_alert(reg_alert)
                
        except Exception as e:
            logger.error(f"Failed monitoring check for {scenario.name}: {e}")
    
    def _send_regression_alert(self, reg_alert: RegressionAlert):
        """Send performance regression alert via AlertManager."""
        severity = self._determine_alert_severity(reg_alert.regression_severity)
        
        alert = Alert(
            level=severity,
            system=f"performance_regression_{reg_alert.scenario}",
            metric=reg_alert.metric_type,
            message=f"Performance regression detected in {reg_alert.scenario}",
            value=reg_alert.current_value,
            threshold=reg_alert.baseline_mean,
            timestamp=reg_alert.detection_time.timestamp(),
            tags={
                'scenario': reg_alert.scenario,
                'regression_percentage': f"{reg_alert.regression_severity * 100:.1f}%",
                'detection_time': reg_alert.detection_time.isoformat(),
                'confidence_level': str(reg_alert.confidence_level)
            }
        )
        
        self.alert_manager.alerts.append(alert)
        logger.warning(f"Performance regression alert: {alert.message} - "
                      f"{reg_alert.regression_severity * 100:.1f}% degradation")
    
    def _determine_alert_severity(self, regression_percentage: float) -> AlertLevel:
        """Determine alert severity based on regression percentage."""
        if regression_percentage >= 0.50:  # 50%+ regression
            return AlertLevel.CRITICAL
        elif regression_percentage >= 0.25:  # 25%+ regression
            return AlertLevel.WARNING
        else:  # 10%+ regression (minimum threshold)
            return AlertLevel.INFO
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics."""
        with self.lock:
            return {
                'enabled': self.enabled,
                'monitoring_interval': self.monitoring_interval,
                'scenarios_monitored': len(self.monitoring_scenarios),
                'thread_alive': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
                'detector_baselines': len(self.detector.baselines),
                'recent_alerts': len([
                    a for a in self.alert_manager.alerts
                    if 'performance_regression' in a.name and
                    (datetime.now() - a.timestamp).total_seconds() < 3600
                ])
            }


def create_performance_monitor(data_dir: Path = None) -> ContinuousMonitor:
    """Factory function to create configured performance monitor."""
    
    # Default paths
    if data_dir is None:
        data_dir = Path.cwd() / "performance_data"
    
    baseline_path = data_dir / "performance_baselines.json"
    history_path = data_dir / "performance_history"
    
    # Create components
    benchmarker = Phase43Benchmarker()
    detector = RegressionDetector(
        baseline_path=baseline_path,
        regression_threshold=MONITORING.get("performance", {}).get("regression_threshold", 0.10),
        confidence_level=MONITORING.get("performance", {}).get("confidence_level", 0.95)
    )
    history = PerformanceHistory(
        history_path=history_path,
        max_history_days=MONITORING.get("performance", {}).get("max_history_days", 30)
    )
    
    # Get existing monitoring components
    from .alerting import get_alert_manager
    from .metrics_collector import get_metrics_collector
    alert_manager = get_alert_manager()
    metrics_collector = get_metrics_collector()
    
    # Create monitor
    monitor = ContinuousMonitor(
        benchmarker=benchmarker,
        detector=detector,
        history=history,
        alert_manager=alert_manager,
        metrics_collector=metrics_collector,
        monitoring_interval=MONITORING.get("performance", {}).get("monitoring_interval", 3600)
    )
    
    logger.info("Created performance monitoring system")
    return monitor