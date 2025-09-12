#!/usr/bin/env python3
"""
Performance Regression Detection Suite for GifLab.

This module provides automated performance regression detection to prevent
performance-degrading changes from entering production. It establishes
baseline benchmarks, implements CI/CD gates, and provides continuous monitoring.
"""

import json
import hashlib
import time
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import numpy as np
from contextlib import contextmanager

# Local imports
from giflab.config import DEFAULT_METRICS_CONFIG, ENABLE_EXPERIMENTAL_CACHING
from giflab.monitoring import get_metrics_collector, MetricType
from giflab.caching import get_frame_cache, get_validation_cache
from giflab.synthetic_gifs import generate_gradient_frames
import cv2


class PerformanceMetric(Enum):
    """Performance metrics tracked for regression detection."""
    FRAME_CACHE_HIT_RATE = "frame_cache_hit_rate"
    FRAME_CACHE_LATENCY = "frame_cache_latency_ms"
    VALIDATION_CACHE_HIT_RATE = "validation_cache_hit_rate"
    VALIDATION_CACHE_LATENCY = "validation_cache_latency_ms"
    RESIZE_CACHE_BUFFER_REUSE = "resize_cache_buffer_reuse_rate"
    SAMPLING_SPEEDUP = "sampling_speedup_factor"
    MODULE_LOAD_TIME = "module_load_time_ms"
    TOTAL_VALIDATION_TIME = "total_validation_time_ms"
    MEMORY_USAGE_PEAK = "memory_usage_peak_mb"
    MEMORY_USAGE_AVERAGE = "memory_usage_avg_mb"


class RegressionThreshold(Enum):
    """Threshold levels for performance regression detection."""
    CRITICAL = 0.10  # 10% degradation - hard fail
    WARNING = 0.05   # 5% degradation - warning
    ACCEPTABLE = 0.02  # 2% variance - acceptable


@dataclass
class BenchmarkScenario:
    """Defines a benchmark scenario for testing."""
    name: str
    description: str
    frame_count: int
    frame_size: Tuple[int, int]
    compression_level: int
    metrics_to_validate: List[str]
    sampling_enabled: bool
    cache_warmup_rounds: int = 3
    measurement_rounds: int = 10
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    scenario_name: str
    timestamp: datetime
    git_commit: str
    metrics: Dict[str, float]
    environment: Dict[str, str]
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkResult':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for comparison."""
    scenario_name: str
    git_commit: str
    created_at: datetime
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sample_count: int
    environment: Dict[str, str]
    
    def compare(self, result: BenchmarkResult) -> Dict[str, float]:
        """Compare a result against this baseline."""
        comparison = {}
        for metric, baseline_value in self.metrics.items():
            if metric in result.metrics:
                result_value = result.metrics[metric]
                if baseline_value > 0:
                    # Calculate percentage change
                    change = (result_value - baseline_value) / baseline_value
                    comparison[metric] = change
                else:
                    comparison[metric] = 0.0
        return comparison


class RegressionDetector:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, baseline_dir: Optional[Path] = None):
        """Initialize the regression detector."""
        self.baseline_dir = baseline_dir or Path.home() / ".giflab" / "baselines"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.current_baselines: Dict[str, PerformanceBaseline] = {}
        self.load_baselines()
    
    def load_baselines(self) -> None:
        """Load baselines from disk."""
        baseline_file = self.baseline_dir / "current_baselines.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                for scenario_name, baseline_data in data.items():
                    baseline_data['created_at'] = datetime.fromisoformat(
                        baseline_data['created_at']
                    )
                    # Convert confidence interval lists back to tuples
                    baseline_data['confidence_intervals'] = {
                        k: tuple(v) for k, v in 
                        baseline_data['confidence_intervals'].items()
                    }
                    self.current_baselines[scenario_name] = PerformanceBaseline(
                        **baseline_data
                    )
    
    def save_baselines(self) -> None:
        """Save baselines to disk."""
        baseline_file = self.baseline_dir / "current_baselines.json"
        data = {}
        for scenario_name, baseline in self.current_baselines.items():
            baseline_dict = asdict(baseline)
            baseline_dict['created_at'] = baseline.created_at.isoformat()
            data[scenario_name] = baseline_dict
        
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_baseline(
        self, 
        scenario_name: str, 
        results: List[BenchmarkResult]
    ) -> PerformanceBaseline:
        """Update baseline from a set of benchmark results."""
        if not results:
            raise ValueError("Cannot create baseline from empty results")
        
        # Aggregate metrics across all results
        metrics_aggregated = {}
        confidence_intervals = {}
        
        for metric in results[0].metrics.keys():
            values = [r.metrics[metric] for r in results if metric in r.metrics]
            if values:
                metrics_aggregated[metric] = statistics.mean(values)
                # Calculate 95% confidence interval
                if len(values) > 1:
                    stdev = statistics.stdev(values)
                    margin = 1.96 * (stdev / (len(values) ** 0.5))
                    mean_val = metrics_aggregated[metric]
                    confidence_intervals[metric] = (
                        mean_val - margin, 
                        mean_val + margin
                    )
                else:
                    confidence_intervals[metric] = (values[0], values[0])
        
        baseline = PerformanceBaseline(
            scenario_name=scenario_name,
            git_commit=self._get_current_git_commit(),
            created_at=datetime.now(),
            metrics=metrics_aggregated,
            confidence_intervals=confidence_intervals,
            sample_count=len(results),
            environment=results[0].environment
        )
        
        self.current_baselines[scenario_name] = baseline
        self.save_baselines()
        return baseline
    
    def check_regression(
        self, 
        result: BenchmarkResult
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a result shows regression compared to baseline.
        
        Returns:
            Tuple of (has_regression, details_dict)
        """
        if result.scenario_name not in self.current_baselines:
            return False, {"status": "no_baseline", "scenario": result.scenario_name}
        
        baseline = self.current_baselines[result.scenario_name]
        comparison = baseline.compare(result)
        
        regressions = {"critical": [], "warning": [], "acceptable": []}
        
        for metric, change in comparison.items():
            # For hit rates, negative change is bad
            if "hit_rate" in metric:
                change = -change
            
            if abs(change) > RegressionThreshold.CRITICAL.value:
                regressions["critical"].append({
                    "metric": metric,
                    "change_percent": change * 100,
                    "baseline": baseline.metrics[metric],
                    "current": result.metrics[metric]
                })
            elif abs(change) > RegressionThreshold.WARNING.value:
                regressions["warning"].append({
                    "metric": metric,
                    "change_percent": change * 100,
                    "baseline": baseline.metrics[metric],
                    "current": result.metrics[metric]
                })
            elif abs(change) > RegressionThreshold.ACCEPTABLE.value:
                regressions["acceptable"].append({
                    "metric": metric,
                    "change_percent": change * 100,
                    "baseline": baseline.metrics[metric],
                    "current": result.metrics[metric]
                })
        
        has_regression = bool(regressions["critical"]) or bool(regressions["warning"])
        
        return has_regression, {
            "status": "regression_detected" if has_regression else "pass",
            "regressions": regressions,
            "comparison": comparison
        }
    
    def _get_current_git_commit(self) -> str:
        """Get the current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()[:8]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"


class BenchmarkSuite:
    """Main benchmark suite for performance regression testing."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the benchmark suite."""
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios = self._define_standard_scenarios()
        self.detector = RegressionDetector()
        self.metrics_collector = get_metrics_collector()
    
    def _define_standard_scenarios(self) -> List[BenchmarkScenario]:
        """Define standard benchmark scenarios."""
        return [
            BenchmarkScenario(
                name="small_gif_validation",
                description="Validation of small GIF (10 frames)",
                frame_count=10,
                frame_size=(256, 256),
                compression_level=60,
                metrics_to_validate=["ssim", "ms_ssim", "gradient_color"],
                sampling_enabled=False,
                tags=["quick", "ci"]
            ),
            BenchmarkScenario(
                name="medium_gif_processing",
                description="Processing medium GIF (100 frames)",
                frame_count=100,
                frame_size=(512, 512),
                compression_level=70,
                metrics_to_validate=["ssim", "ms_ssim", "lpips", "gradient_color"],
                sampling_enabled=True,
                tags=["standard", "ci"]
            ),
            BenchmarkScenario(
                name="large_gif_with_sampling",
                description="Large GIF with intelligent sampling (500 frames)",
                frame_count=500,
                frame_size=(800, 600),
                compression_level=80,
                metrics_to_validate=["ssim", "gradient_color", "ssimulacra2"],
                sampling_enabled=True,
                tags=["extended"]
            ),
            BenchmarkScenario(
                name="multi_metric_workflow",
                description="Multi-metric validation workflow",
                frame_count=50,
                frame_size=(640, 480),
                compression_level=75,
                metrics_to_validate=["ssim", "ms_ssim", "lpips", "gradient_color", "ssimulacra2"],
                sampling_enabled=False,
                tags=["comprehensive", "ci"]
            ),
            BenchmarkScenario(
                name="cache_stress_test",
                description="Cache system stress test",
                frame_count=200,
                frame_size=(400, 300),
                compression_level=85,
                metrics_to_validate=["ssim"],
                sampling_enabled=False,
                cache_warmup_rounds=10,
                measurement_rounds=20,
                tags=["stress", "cache"]
            )
        ]
    
    @contextmanager
    def _measure_performance(self) -> Dict[str, float]:
        """Context manager to measure performance metrics."""
        # Clear metrics
        self.metrics_collector.clear()
        
        # Get initial cache stats
        frame_cache = get_frame_cache()
        validation_cache = get_validation_cache()
        
        initial_frame_stats = frame_cache.get_stats()
        initial_val_stats = validation_cache.get_stats()
        
        # Measure memory before
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        
        metrics = {}
        try:
            yield metrics
        finally:
            # Calculate elapsed time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            metrics[PerformanceMetric.TOTAL_VALIDATION_TIME.value] = elapsed_ms
            
            # Get final cache stats
            final_frame_stats = frame_cache.get_stats()
            final_val_stats = validation_cache.get_stats()
            
            # Calculate cache hit rates
            frame_hits = final_frame_stats.hits - initial_frame_stats.hits
            frame_total = frame_hits + (final_frame_stats.misses - initial_frame_stats.misses)
            if frame_total > 0:
                metrics[PerformanceMetric.FRAME_CACHE_HIT_RATE.value] = (
                    frame_hits / frame_total
                )
            
            val_hits = final_val_stats.hits - initial_val_stats.hits
            val_total = val_hits + (final_val_stats.misses - initial_val_stats.misses)
            if val_total > 0:
                metrics[PerformanceMetric.VALIDATION_CACHE_HIT_RATE.value] = (
                    val_hits / val_total
                )
            
            # Memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            metrics[PerformanceMetric.MEMORY_USAGE_PEAK.value] = final_memory
            metrics[PerformanceMetric.MEMORY_USAGE_AVERAGE.value] = (
                (initial_memory + final_memory) / 2
            )
            
            # Get metrics from collector
            collector_metrics = self.metrics_collector.get_metrics(
                start_time=datetime.now() - timedelta(seconds=elapsed_ms/1000)
            )
            
            # Extract specific latencies if available
            for metric_data in collector_metrics:
                if metric_data['name'] == 'frame_cache.get':
                    if 'avg' in metric_data:
                        metrics[PerformanceMetric.FRAME_CACHE_LATENCY.value] = (
                            metric_data['avg']
                        )
                elif metric_data['name'] == 'validation_cache.get':
                    if 'avg' in metric_data:
                        metrics[PerformanceMetric.VALIDATION_CACHE_LATENCY.value] = (
                            metric_data['avg']
                        )
    
    def run_scenario(
        self, 
        scenario: BenchmarkScenario,
        verbose: bool = False
    ) -> BenchmarkResult:
        """Run a single benchmark scenario."""
        if verbose:
            print(f"\n🏃 Running scenario: {scenario.name}")
            print(f"   {scenario.description}")
        
        # Generate test GIF
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            compressed_path = Path(tmpdir) / "compressed.gif"
            
            # Create test GIF using available synthetic generation
            frames = generate_gradient_frames(
                num_frames=scenario.frame_count,
                width=scenario.frame_size[0],
                height=scenario.frame_size[1]
            )
            
            # Save as GIF (mock compression)
            # In real implementation, would use giflab compression
            import imageio
            imageio.mimsave(str(gif_path), frames, duration=100)
            imageio.mimsave(str(compressed_path), frames[::2], duration=100)  # Mock compression
            
            # Warm up caches
            if verbose:
                print(f"   Warming up caches ({scenario.cache_warmup_rounds} rounds)")
            
            for _ in range(scenario.cache_warmup_rounds):
                from giflab.validation import validate_optimization
                validate_optimization(
                    str(gif_path),
                    str(compressed_path),
                    metrics=scenario.metrics_to_validate,
                    sampling_enabled=scenario.sampling_enabled
                )
            
            # Run measurement rounds
            if verbose:
                print(f"   Running measurements ({scenario.measurement_rounds} rounds)")
            
            round_metrics = []
            for i in range(scenario.measurement_rounds):
                with self._measure_performance() as metrics:
                    try:
                        # Run validation
                        validation_results = validate_optimization(
                            str(gif_path),
                            str(compressed_path),
                            metrics=scenario.metrics_to_validate,
                            sampling_enabled=scenario.sampling_enabled
                        )
                        
                        # Add sampling speedup if applicable
                        if scenario.sampling_enabled and '_sampling_info' in validation_results:
                            sampling_info = validation_results['_sampling_info']
                            if sampling_info.get('sampling_applied'):
                                speedup = 1.0 / sampling_info.get('sampling_rate', 1.0)
                                metrics[PerformanceMetric.SAMPLING_SPEEDUP.value] = speedup
                        
                        round_metrics.append(metrics.copy())
                        
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️ Round {i+1} failed: {e}")
            
            # Aggregate metrics
            if not round_metrics:
                return BenchmarkResult(
                    scenario_name=scenario.name,
                    timestamp=datetime.now(),
                    git_commit=self.detector._get_current_git_commit(),
                    metrics={},
                    environment=self._get_environment_info(),
                    duration_ms=0,
                    success=False,
                    error_message="All measurement rounds failed"
                )
            
            aggregated_metrics = {}
            for metric_name in round_metrics[0].keys():
                values = [m[metric_name] for m in round_metrics if metric_name in m]
                if values:
                    aggregated_metrics[metric_name] = statistics.median(values)
            
            result = BenchmarkResult(
                scenario_name=scenario.name,
                timestamp=datetime.now(),
                git_commit=self.detector._get_current_git_commit(),
                metrics=aggregated_metrics,
                environment=self._get_environment_info(),
                duration_ms=aggregated_metrics.get(
                    PerformanceMetric.TOTAL_VALIDATION_TIME.value, 0
                ),
                success=True
            )
            
            if verbose:
                print(f"   ✅ Completed in {result.duration_ms:.2f}ms")
                print(f"   Cache hit rates: Frame={aggregated_metrics.get(PerformanceMetric.FRAME_CACHE_HIT_RATE.value, 0):.1%}, "
                      f"Validation={aggregated_metrics.get(PerformanceMetric.VALIDATION_CACHE_HIT_RATE.value, 0):.1%}")
            
            return result
    
    def run_ci_suite(self, tags: List[str] = ["ci"]) -> Tuple[bool, Dict[str, Any]]:
        """
        Run CI benchmark suite and check for regressions.
        
        Returns:
            Tuple of (success, report_dict)
        """
        scenarios_to_run = [
            s for s in self.scenarios 
            if any(tag in s.tags for tag in tags)
        ]
        
        print(f"\n🚀 Running CI Performance Regression Suite")
        print(f"   Scenarios: {len(scenarios_to_run)}")
        print(f"   Tags: {tags}")
        
        results = []
        regressions = []
        
        for scenario in scenarios_to_run:
            result = self.run_scenario(scenario, verbose=True)
            results.append(result)
            
            # Check for regression
            has_regression, details = self.detector.check_regression(result)
            if has_regression:
                regressions.append({
                    "scenario": scenario.name,
                    "details": details
                })
                print(f"   ⚠️ REGRESSION DETECTED in {scenario.name}")
                for level in ["critical", "warning"]:
                    for reg in details["regressions"][level]:
                        print(f"      {level.upper()}: {reg['metric']} "
                              f"changed by {reg['change_percent']:.1f}%")
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": self.detector._get_current_git_commit(),
            "scenarios_run": len(scenarios_to_run),
            "scenarios_passed": len(scenarios_to_run) - len(regressions),
            "has_regressions": bool(regressions),
            "regressions": regressions,
            "results": [r.to_json() for r in results]
        }
        
        # Save report
        report_file = self.output_dir / f"ci_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {report_file}")
        
        # Determine success based on critical regressions only
        has_critical = any(
            bool(r["details"]["regressions"]["critical"]) 
            for r in regressions
        )
        
        if has_critical:
            print("\n❌ CI FAILED: Critical performance regressions detected")
        elif regressions:
            print("\n⚠️ CI PASSED WITH WARNINGS: Non-critical regressions detected")
        else:
            print("\n✅ CI PASSED: No performance regressions detected")
        
        return not has_critical, report
    
    def establish_baselines(self, tags: List[str] = ["ci"]) -> None:
        """Establish new baselines from current performance."""
        scenarios_to_baseline = [
            s for s in self.scenarios 
            if any(tag in s.tags for tag in tags)
        ]
        
        print(f"\n📐 Establishing Performance Baselines")
        print(f"   Scenarios: {len(scenarios_to_baseline)}")
        
        for scenario in scenarios_to_baseline:
            print(f"\n   Processing: {scenario.name}")
            
            # Run multiple rounds for statistical significance
            results = []
            for i in range(5):
                print(f"      Round {i+1}/5...")
                result = self.run_scenario(scenario, verbose=False)
                if result.success:
                    results.append(result)
            
            if results:
                baseline = self.detector.update_baseline(scenario.name, results)
                print(f"   ✅ Baseline established with {len(results)} samples")
                print(f"      Commit: {baseline.git_commit}")
            else:
                print(f"   ❌ Failed to establish baseline")
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information."""
        import platform
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine()
        }


def main():
    """Main entry point for regression testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GifLab Performance Regression Detection"
    )
    parser.add_argument(
        "command",
        choices=["test", "baseline", "compare"],
        help="Command to run"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["ci"],
        help="Scenario tags to run"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    suite = BenchmarkSuite(output_dir=args.output_dir)
    
    if args.command == "test":
        # Run CI regression tests
        success, report = suite.run_ci_suite(tags=args.tags)
        sys.exit(0 if success else 1)
    
    elif args.command == "baseline":
        # Establish new baselines
        suite.establish_baselines(tags=args.tags)
    
    elif args.command == "compare":
        # Compare current performance to baselines
        print("Compare command not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()