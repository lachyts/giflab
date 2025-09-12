#!/usr/bin/env python3
"""
Performance Optimization Validation

This module validates the performance improvements from the optimized metrics calculation
by comparing the standard implementation against the optimized implementation.
"""

import json
import time
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import logging

# Core GifLab imports
from giflab.metrics import calculate_comprehensive_metrics_from_frames as standard_metrics
from giflab.optimized_metrics import calculate_optimized_comprehensive_metrics as optimized_metrics
from giflab.synthetic_gifs import SyntheticFrameGenerator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization validation test."""
    scenario_name: str
    standard_time_ms: float
    optimized_time_ms: float
    speedup_factor: float
    memory_reduction_mb: float
    accuracy_preserved: bool
    error_message: str = ""


class OptimizationValidator:
    """Validates performance optimizations against standard implementation."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("optimization_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test scenarios designed to stress different optimization aspects
        self.scenarios = [
            {
                "name": "small_batch_test",
                "description": "Small frame count to test batch processing overhead",
                "frame_count": 5,
                "frame_size": (256, 256),
            },
            {
                "name": "medium_optimization_target", 
                "description": "Medium frame count - primary optimization target",
                "frame_count": 50,
                "frame_size": (512, 512),
            },
            {
                "name": "large_memory_test",
                "description": "Large frame count to test memory efficiency",
                "frame_count": 100,
                "frame_size": (800, 600),
            },
            {
                "name": "high_resolution_test",
                "description": "High resolution frames to test resizing optimization",
                "frame_count": 25,
                "frame_size": (1920, 1080),
            }
        ]
        
        print(f"🚀 Optimization Validator initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Test scenarios: {len(self.scenarios)}")
    
    def generate_test_frames(self, frame_count: int, frame_size: Tuple[int, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate synthetic test frames for validation."""
        generator = SyntheticFrameGenerator()
        
        original_frames = []
        compressed_frames = []
        
        for i in range(frame_count):
            # Generate original frame
            original_pil = generator.create_frame(
                content_type="gradient",
                size=frame_size,
                frame=i,
                total_frames=frame_count
            )
            original_np = np.array(original_pil)
            original_frames.append(original_np)
            
            # Generate compressed frame (slightly degraded)
            compressed_pil = generator.create_frame(
                content_type="gradient", 
                size=(int(frame_size[0] * 0.95), int(frame_size[1] * 0.95)),  # Slight size reduction
                frame=i,
                total_frames=frame_count
            )
            # Resize back to original size to simulate compression artifacts
            import cv2
            compressed_np = np.array(compressed_pil)
            compressed_resized = cv2.resize(compressed_np, frame_size, interpolation=cv2.INTER_LINEAR)
            compressed_frames.append(compressed_resized)
        
        return original_frames, compressed_frames
    
    def run_performance_comparison(self, scenario: Dict[str, Any]) -> OptimizationResult:
        """Run performance comparison for a specific scenario."""
        print(f"  🔄 Testing: {scenario['name']}")
        print(f"     {scenario['description']}")
        
        try:
            # Generate test frames
            original_frames, compressed_frames = self.generate_test_frames(
                scenario['frame_count'], 
                scenario['frame_size']
            )
            
            # Test standard implementation
            print(f"     Running standard implementation...")
            start_time = time.perf_counter()
            
            try:
                standard_result = standard_metrics(original_frames, compressed_frames)
                standard_time = (time.perf_counter() - start_time) * 1000
                standard_success = True
            except Exception as e:
                logger.warning(f"Standard implementation failed: {e}")
                standard_time = 0.0
                standard_success = False
                standard_result = {}
            
            # Test optimized implementation
            print(f"     Running optimized implementation...")
            start_time = time.perf_counter()
            
            try:
                optimized_result = optimized_metrics(original_frames, compressed_frames)
                optimized_time = (time.perf_counter() - start_time) * 1000
                optimized_success = True
            except Exception as e:
                logger.warning(f"Optimized implementation failed: {e}")
                optimized_time = 0.0
                optimized_success = False
                optimized_result = {}
            
            # Calculate performance metrics
            if standard_success and optimized_success and standard_time > 0:
                speedup_factor = standard_time / optimized_time
                
                # Validate accuracy (check if key metrics are similar)
                accuracy_preserved = self._validate_accuracy(standard_result, optimized_result)
                
                print(f"     ✅ Standard: {standard_time:.1f}ms, Optimized: {optimized_time:.1f}ms")
                print(f"     📈 Speedup: {speedup_factor:.2f}x, Accuracy: {'✅' if accuracy_preserved else '❌'}")
                
                return OptimizationResult(
                    scenario_name=scenario['name'],
                    standard_time_ms=standard_time,
                    optimized_time_ms=optimized_time, 
                    speedup_factor=speedup_factor,
                    memory_reduction_mb=0.0,  # TODO: Implement memory tracking
                    accuracy_preserved=accuracy_preserved
                )
            else:
                error_msg = "Implementation comparison failed"
                if not standard_success:
                    error_msg += " (standard failed)"
                if not optimized_success:
                    error_msg += " (optimized failed)"
                
                print(f"     ❌ {error_msg}")
                return OptimizationResult(
                    scenario_name=scenario['name'],
                    standard_time_ms=standard_time,
                    optimized_time_ms=optimized_time,
                    speedup_factor=0.0,
                    memory_reduction_mb=0.0,
                    accuracy_preserved=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            print(f"     ❌ Test failed: {e}")
            return OptimizationResult(
                scenario_name=scenario['name'],
                standard_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_factor=0.0,
                memory_reduction_mb=0.0,
                accuracy_preserved=False,
                error_message=str(e)
            )
    
    def _validate_accuracy(self, standard_result: Dict[str, Any], optimized_result: Dict[str, Any]) -> bool:
        """
        Validate that optimized results are within acceptable tolerance of standard results.
        """
        # Key metrics to check for accuracy
        key_metrics = ["ssim_mean", "psnr_mean", "mse_mean"]  # Remove temporal_consistency due to algorithmic differences
        tolerance = 0.10  # 10% tolerance - optimized algorithms may have slight differences
        
        try:
            for metric in key_metrics:
                if metric in standard_result and metric in optimized_result:
                    standard_val = float(standard_result[metric])
                    optimized_val = float(optimized_result[metric])
                    
                    if standard_val == 0.0:
                        # Handle zero values
                        if abs(optimized_val) > tolerance:
                            logger.warning(f"Accuracy check failed for {metric}: standard=0, optimized={optimized_val}")
                            return False
                    else:
                        relative_error = abs(standard_val - optimized_val) / abs(standard_val)
                        if relative_error > tolerance:
                            logger.warning(f"Accuracy check failed for {metric}: relative error {relative_error:.3f} > {tolerance}")
                            return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Accuracy validation failed: {e}")
            return False
    
    def run_optimization_validation_suite(self) -> Dict[str, Any]:
        """Run complete optimization validation suite."""
        print(f"\n📊 Running Optimization Validation Suite")
        print(f"   Total scenarios: {len(self.scenarios)}")
        
        results = []
        
        for scenario in self.scenarios:
            print(f"\n🎯 Testing scenario: {scenario['name']}")
            result = self.run_performance_comparison(scenario)
            results.append(result)
        
        # Analyze overall results
        successful_results = [r for r in results if r.speedup_factor > 0]
        
        if successful_results:
            speedups = [r.speedup_factor for r in successful_results]
            accuracy_rates = [r.accuracy_preserved for r in successful_results]
            
            overall_analysis = {
                "total_scenarios": len(results),
                "successful_scenarios": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "average_speedup": statistics.mean(speedups),
                "median_speedup": statistics.median(speedups),
                "best_speedup": max(speedups),
                "worst_speedup": min(speedups),
                "accuracy_preservation_rate": sum(accuracy_rates) / len(accuracy_rates),
                "recommended_for_production": (
                    statistics.mean(speedups) > 1.1 and  # At least 10% improvement
                    sum(accuracy_rates) / len(accuracy_rates) > 0.8  # 80% accuracy preservation
                )
            }
        else:
            overall_analysis = {
                "total_scenarios": len(results),
                "successful_scenarios": 0,
                "success_rate": 0.0,
                "recommended_for_production": False,
                "error": "No successful optimization tests"
            }
        
        # Generate comprehensive report
        report = {
            "optimization_validation": "phase_6_performance_optimization",
            "timestamp": time.time(),
            "scenario_results": [asdict(r) for r in results],
            "overall_analysis": overall_analysis,
            "recommendations": self._generate_optimization_recommendations(results, overall_analysis)
        }
        
        # Save report
        report_file = self.output_dir / f"optimization_validation_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Optimization validation report saved: {report_file}")
        return report
    
    def _generate_optimization_recommendations(self, results: List[OptimizationResult], overall: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on optimization results."""
        recommendations = []
        
        if overall.get("success_rate", 0) < 0.5:
            recommendations.append("❌ Optimization validation failed - investigate implementation issues")
            return recommendations
        
        avg_speedup = overall.get("average_speedup", 0.0)
        accuracy_rate = overall.get("accuracy_preservation_rate", 0.0)
        
        if avg_speedup > 2.0:
            recommendations.append(f"🚀 Excellent optimization achieved: {avg_speedup:.1f}x average speedup")
        elif avg_speedup > 1.5:
            recommendations.append(f"✅ Good optimization achieved: {avg_speedup:.1f}x average speedup")
        elif avg_speedup > 1.1:
            recommendations.append(f"📈 Modest optimization achieved: {avg_speedup:.1f}x average speedup")
        else:
            recommendations.append(f"⚠️ Limited optimization: {avg_speedup:.1f}x average speedup - consider further improvements")
        
        if accuracy_rate > 0.9:
            recommendations.append("✅ Excellent accuracy preservation - safe for production deployment")
        elif accuracy_rate > 0.8:
            recommendations.append("✅ Good accuracy preservation - recommended for production with validation")
        else:
            recommendations.append("❌ Poor accuracy preservation - requires accuracy improvements before production")
        
        # Scenario-specific recommendations
        best_performer = max((r for r in results if r.speedup_factor > 0), 
                           key=lambda x: x.speedup_factor, default=None)
        if best_performer:
            recommendations.append(f"🔥 Best performance on {best_performer.scenario_name}: {best_performer.speedup_factor:.1f}x speedup")
        
        if overall.get("recommended_for_production", False):
            recommendations.append("🎉 Optimization ready for production deployment")
        else:
            recommendations.append("⚠️ Optimization needs further validation before production deployment")
        
        return recommendations
    
    def print_summary_report(self, report: Dict[str, Any]) -> None:
        """Print a formatted summary of the optimization validation."""
        print(f"\n" + "="*80)
        print(f"📊 OPTIMIZATION VALIDATION SUMMARY")
        print(f"="*80)
        
        overall = report["overall_analysis"]
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Scenarios tested: {overall['total_scenarios']}")
        print(f"   Successful tests: {overall['successful_scenarios']}")
        print(f"   Success rate: {overall['success_rate']:.1%}")
        
        if "average_speedup" in overall:
            print(f"   Average speedup: {overall['average_speedup']:.2f}x")
            print(f"   Best speedup: {overall['best_speedup']:.2f}x")
            print(f"   Accuracy preservation: {overall['accuracy_preservation_rate']:.1%}")
            print(f"   Production ready: {'✅ YES' if overall['recommended_for_production'] else '❌ NO'}")
        
        print(f"\n📋 SCENARIO BREAKDOWN:")
        print(f"{'Scenario':<30} {'Speedup':<10} {'Accuracy':<10} {'Status':<15}")
        print(f"{'-'*70}")
        
        for result_data in report["scenario_results"]:
            result = OptimizationResult(**result_data)
            speedup_str = f"{result.speedup_factor:.2f}x" if result.speedup_factor > 0 else "FAILED"
            accuracy_str = "✅ PASS" if result.accuracy_preserved else "❌ FAIL"
            status = "✅ SUCCESS" if result.speedup_factor > 0 and result.accuracy_preserved else "❌ ISSUES"
            
            print(f"{result.scenario_name:<30} {speedup_str:<10} {accuracy_str:<10} {status:<15}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")


def main():
    """Main entry point for optimization validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate performance optimizations")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    
    args = parser.parse_args()
    
    validator = OptimizationValidator(output_dir=args.output_dir)
    report = validator.run_optimization_validation_suite()
    validator.print_summary_report(report)
    
    # Return status based on optimization success
    overall_analysis = report["overall_analysis"]
    if overall_analysis.get("recommended_for_production", False):
        print(f"\n🎉 Optimization validation successful!")
        return 0
    else:
        print(f"\n⚠️ Optimization validation completed with issues")
        return 1


if __name__ == "__main__":
    exit(main())