#!/usr/bin/env python3
"""
Phase 4.3: Performance Benchmarking & Optimization Validation

This script establishes performance baselines and validates the impact of 
all Phase 1-3 architectural improvements using the existing monitoring infrastructure.
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tempfile

# Core GifLab imports
from giflab.config import ENABLE_EXPERIMENTAL_CACHING
from giflab.metrics import calculate_comprehensive_metrics
from giflab.synthetic_gifs import SyntheticFrameGenerator
from giflab.monitoring import (
    get_baseline_framework, 
    get_metrics_collector,
    BaselineTestMode
)


@dataclass
class BenchmarkScenario:
    """Test scenario for performance benchmarking."""
    name: str
    description: str
    frame_count: int
    frame_size: tuple[int, int]
    metrics_to_test: List[str]
    expected_duration_range: tuple[float, float]  # min, max seconds


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    scenario_name: str
    caching_enabled: bool
    processing_time_ms: float
    memory_usage_mb: float
    metrics_calculated: List[str]
    success: bool
    error_message: Optional[str] = None
    timestamp: float = 0.0


class Phase43Benchmarker:
    """Phase 4.3 Performance Benchmarking Implementation."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get monitoring components
        self.baseline_framework = get_baseline_framework()
        self.metrics_collector = get_metrics_collector()
        
        # Define standard test scenarios
        self.scenarios = self._define_scenarios()
        
        print(f"🚀 Phase 4.3 Benchmarker initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Caching enabled: {ENABLE_EXPERIMENTAL_CACHING}")
        print(f"   Test scenarios: {len(self.scenarios)}")
    
    def _define_scenarios(self) -> List[BenchmarkScenario]:
        """Define standardized performance test scenarios."""
        return [
            BenchmarkScenario(
                name="small_gif_basic",
                description="Small GIF with basic metrics (quick validation)",
                frame_count=10,
                frame_size=(256, 256),
                metrics_to_test=["ssim", "mse"],
                expected_duration_range=(0.5, 5.0)
            ),
            BenchmarkScenario(
                name="medium_gif_comprehensive",
                description="Medium GIF with comprehensive metrics",
                frame_count=50,
                frame_size=(512, 512),
                metrics_to_test=["ssim", "ms_ssim", "gradient_color"],
                expected_duration_range=(2.0, 15.0)
            ),
            BenchmarkScenario(
                name="large_gif_selective",
                description="Large GIF with selective high-value metrics",
                frame_count=100,
                frame_size=(800, 600),
                metrics_to_test=["ssim", "gradient_color"],
                expected_duration_range=(5.0, 30.0)
            ),
            BenchmarkScenario(
                name="memory_stress_test",
                description="Memory usage validation with moderate load",
                frame_count=200,
                frame_size=(640, 480),
                metrics_to_test=["ssim"],
                expected_duration_range=(8.0, 45.0)
            )
        ]
    
    def run_single_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run a single benchmark scenario."""
        print(f"  🔄 Running: {scenario.name}")
        print(f"     {scenario.description}")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Generate test data
                original_path = Path(tmpdir) / "original.gif"
                compressed_path = Path(tmpdir) / "compressed.gif"
                
                # Create synthetic GIF frames using the generator
                frame_generator = SyntheticFrameGenerator()
                frames = []
                for frame_idx in range(scenario.frame_count):
                    pil_frame = frame_generator.create_frame(
                        content_type="gradient",
                        size=(scenario.frame_size[0], scenario.frame_size[1]),
                        frame=frame_idx,
                        total_frames=scenario.frame_count
                    )
                    # Convert PIL to numpy array for compatibility
                    import numpy as np
                    frames.append(np.array(pil_frame))
                
                # Simple compression simulation (scale down by 10%)
                compressed_frames = []
                for frame in frames:
                    import cv2
                    h, w = frame.shape[:2]
                    new_size = (int(w * 0.9), int(h * 0.9))
                    compressed_frame = cv2.resize(frame, new_size)
                    compressed_frames.append(compressed_frame)
                
                # Save test files
                import imageio
                imageio.mimsave(str(original_path), frames, duration=100)
                imageio.mimsave(str(compressed_path), compressed_frames, duration=100)
                
                # Measure performance
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                start_time = time.perf_counter()
                
                # Run comprehensive metrics calculation
                results = calculate_comprehensive_metrics(
                    original_path,
                    compressed_path
                )
                
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_usage_mb = final_memory - initial_memory
                
                # Record with baseline framework
                self.baseline_framework.record_performance(
                    operation_type=scenario.name,
                    processing_time_ms=processing_time_ms,
                    cache_enabled=ENABLE_EXPERIMENTAL_CACHING,
                    memory_usage_mb=memory_usage_mb,
                    metadata={
                        "frame_count": scenario.frame_count,
                        "frame_size": scenario.frame_size,
                        "metrics_tested": scenario.metrics_to_test
                    }
                )
                
                return BenchmarkResult(
                    scenario_name=scenario.name,
                    caching_enabled=ENABLE_EXPERIMENTAL_CACHING,
                    processing_time_ms=processing_time_ms,
                    memory_usage_mb=memory_usage_mb,
                    metrics_calculated=list(results.keys()) if results else [],
                    success=True,
                    timestamp=time.time()
                )
                
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return BenchmarkResult(
                scenario_name=scenario.name,
                caching_enabled=ENABLE_EXPERIMENTAL_CACHING,
                processing_time_ms=0.0,
                memory_usage_mb=0.0,
                metrics_calculated=[],
                success=False,
                error_message=str(e),
                timestamp=time.time()
            )
    
    def run_baseline_suite(self, iterations: int = 5) -> Dict[str, Any]:
        """Run complete baseline performance suite."""
        print(f"\n📊 Running Phase 4.3 Baseline Performance Suite")
        print(f"   Iterations per scenario: {iterations}")
        print(f"   Total scenarios: {len(self.scenarios)}")
        print(f"   Caching mode: {'ENABLED' if ENABLE_EXPERIMENTAL_CACHING else 'DISABLED'}")
        
        all_results = []
        scenario_summaries = {}
        
        for scenario in self.scenarios:
            print(f"\n🎯 Testing scenario: {scenario.name}")
            scenario_results = []
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}")
                result = self.run_single_benchmark(scenario)
                scenario_results.append(result)
                all_results.append(result)
                
                if result.success:
                    print(f"     ✅ {result.processing_time_ms:.1f}ms, {result.memory_usage_mb:.1f}MB")
                else:
                    print(f"     ❌ Failed: {result.error_message}")
            
            # Analyze scenario performance
            successful_results = [r for r in scenario_results if r.success]
            if successful_results:
                times = [r.processing_time_ms for r in successful_results]
                memories = [r.memory_usage_mb for r in successful_results]
                
                scenario_summaries[scenario.name] = {
                    "success_rate": len(successful_results) / len(scenario_results),
                    "timing": {
                        "mean_ms": statistics.mean(times),
                        "median_ms": statistics.median(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "stddev_ms": statistics.stdev(times) if len(times) > 1 else 0.0
                    },
                    "memory": {
                        "mean_mb": statistics.mean(memories),
                        "max_mb": max(memories)
                    },
                    "within_expected_range": (
                        scenario.expected_duration_range[0] <= statistics.mean(times)/1000 <= scenario.expected_duration_range[1]
                    )
                }
                
                print(f"  📈 Summary: {statistics.mean(times):.1f}ms avg, {len(successful_results)}/{len(scenario_results)} success")
            else:
                scenario_summaries[scenario.name] = {
                    "success_rate": 0.0,
                    "error": "All iterations failed"
                }
        
        # Generate comprehensive report
        report = {
            "phase": "4.3",
            "test_type": "baseline_performance",
            "timestamp": time.time(),
            "configuration": {
                "caching_enabled": ENABLE_EXPERIMENTAL_CACHING,
                "iterations_per_scenario": iterations,
                "total_tests_run": len(all_results)
            },
            "scenario_summaries": scenario_summaries,
            "overall_analysis": self._analyze_overall_performance(all_results),
            "recommendations": self._generate_baseline_recommendations(scenario_summaries),
            "raw_results": [asdict(r) for r in all_results]
        }
        
        # Save report
        report_file = self.output_dir / f"phase_4_3_baseline_{'cached' if ENABLE_EXPERIMENTAL_CACHING else 'uncached'}_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Baseline report saved: {report_file}")
        return report
    
    def _analyze_overall_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze overall performance across all scenarios."""
        successful = [r for r in results if r.success]
        
        analysis = {
            "status": "success" if successful else "failed",
            "total_tests": len(results),
            "successful_tests": len(successful),
            "success_rate": len(successful) / len(results) if results else 0.0,
        }
        
        if not successful:
            analysis["message"] = "No successful test runs"
            analysis["performance_summary"] = {}
            return analysis
        
        times = [r.processing_time_ms for r in successful]
        memories = [r.memory_usage_mb for r in successful]
        
        analysis["performance_summary"] = {
            "mean_processing_time_ms": statistics.mean(times),
            "total_processing_time_sec": sum(times) / 1000,
            "mean_memory_usage_mb": statistics.mean(memories),
            "max_memory_usage_mb": max(memories)
        }
        
        return analysis
    
    def _generate_baseline_recommendations(self, summaries: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on baseline performance."""
        recommendations = []
        
        # Check if all scenarios completed successfully
        success_rates = [s.get("success_rate", 0) for s in summaries.values() if isinstance(s, dict)]
        if success_rates and min(success_rates) < 1.0:
            recommendations.append("Some test scenarios failed - investigate error conditions before optimization")
        
        # Check performance expectations
        within_range = [s.get("within_expected_range", False) for s in summaries.values() if isinstance(s, dict)]
        if within_range and not all(within_range):
            recommendations.append("Some operations exceeded expected duration - consider optimization targets")
        
        # Memory usage analysis
        memory_usages = []
        for s in summaries.values():
            if isinstance(s, dict) and "memory" in s:
                memory_usages.append(s["memory"]["max_mb"])
        
        if memory_usages and max(memory_usages) > 100:  # 100MB threshold
            recommendations.append("High memory usage detected - monitor memory pressure when enabling caching")
        
        # Baseline establishment recommendation
        if ENABLE_EXPERIMENTAL_CACHING:
            recommendations.append("Baselines established with caching enabled - compare with non-cached performance")
        else:
            recommendations.append("Baselines established without caching - ready for A/B testing with optimizations enabled")
        
        return recommendations


def main():
    """Main entry point for Phase 4.3 benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4.3: Performance Benchmarking & Optimization Validation")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per scenario")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--scenario", help="Run specific scenario only")
    
    args = parser.parse_args()
    
    benchmarker = Phase43Benchmarker(output_dir=args.output_dir)
    
    if args.scenario:
        scenarios = [s for s in benchmarker.scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"❌ Unknown scenario: {args.scenario}")
            print(f"Available scenarios: {[s.name for s in benchmarker.scenarios]}")
            return 1
        benchmarker.scenarios = scenarios
    
    # Run baseline suite
    report = benchmarker.run_baseline_suite(iterations=args.iterations)
    
    # Print summary
    print(f"\n🎉 Phase 4.3 Benchmarking Complete!")
    print(f"   Success rate: {report['overall_analysis']['success_rate']:.1%}")
    
    if report['overall_analysis']['status'] == 'success' and report['overall_analysis']['performance_summary']:
        print(f"   Total processing time: {report['overall_analysis']['performance_summary']['total_processing_time_sec']:.1f}s")
        print(f"   Mean memory usage: {report['overall_analysis']['performance_summary']['mean_memory_usage_mb']:.1f}MB")
    else:
        print(f"   Status: {report['overall_analysis']['status']}")
        if 'message' in report['overall_analysis']:
            print(f"   Message: {report['overall_analysis']['message']}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    return 0


if __name__ == "__main__":
    exit(main())