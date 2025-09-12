#!/usr/bin/env python3
"""
Phase 4.3 Performance Comparison Analysis

This script compares the performance between cached and non-cached configurations
to validate the impact of optimizations implemented in Phases 1-3.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceComparison:
    """Comparison between cached and non-cached performance."""
    scenario: str
    uncached_mean_ms: float
    cached_mean_ms: float
    performance_change_ms: float
    performance_change_percent: float
    uncached_memory_mb: float
    cached_memory_mb: float
    memory_change_mb: float
    memory_change_percent: float
    improvement: bool


class PerformanceAnalyzer:
    """Analyze performance differences between cached and non-cached configurations."""
    
    def __init__(self, results_dir: Path = Path("benchmark_results")):
        self.results_dir = results_dir
        
    def find_latest_reports(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Find the latest cached and uncached performance reports."""
        cached_files = list(self.results_dir.glob("*_cached_*.json"))
        uncached_files = list(self.results_dir.glob("*_uncached_*.json"))
        
        if not cached_files or not uncached_files:
            raise FileNotFoundError("Could not find both cached and uncached performance reports")
        
        # Get the most recent files
        latest_cached = max(cached_files, key=lambda f: f.stat().st_mtime)
        latest_uncached = max(uncached_files, key=lambda f: f.stat().st_mtime)
        
        print(f"📊 Comparing performance reports:")
        print(f"   Cached: {latest_cached.name}")
        print(f"   Uncached: {latest_uncached.name}")
        
        with open(latest_cached) as f:
            cached_data = json.load(f)
        with open(latest_uncached) as f:
            uncached_data = json.load(f)
            
        return uncached_data, cached_data
    
    def compare_scenario_performance(self, scenario: str, uncached_data: Dict, cached_data: Dict) -> PerformanceComparison:
        """Compare performance for a specific scenario."""
        uncached_summary = uncached_data["scenario_summaries"][scenario]
        cached_summary = cached_data["scenario_summaries"][scenario]
        
        # Extract timing data
        uncached_mean = uncached_summary["timing"]["mean_ms"]
        cached_mean = cached_summary["timing"]["mean_ms"]
        timing_change = cached_mean - uncached_mean
        timing_change_percent = (timing_change / uncached_mean) * 100
        
        # Extract memory data
        uncached_memory = uncached_summary["memory"]["mean_mb"]
        cached_memory = cached_summary["memory"]["mean_mb"]
        memory_change = cached_memory - uncached_memory
        memory_change_percent = (memory_change / uncached_memory) * 100 if uncached_memory > 0 else 0
        
        # Determine if this is an improvement (negative timing change = faster)
        improvement = timing_change < 0
        
        return PerformanceComparison(
            scenario=scenario,
            uncached_mean_ms=uncached_mean,
            cached_mean_ms=cached_mean,
            performance_change_ms=timing_change,
            performance_change_percent=timing_change_percent,
            uncached_memory_mb=uncached_memory,
            cached_memory_mb=cached_memory,
            memory_change_mb=memory_change,
            memory_change_percent=memory_change_percent,
            improvement=improvement
        )
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        uncached_data, cached_data = self.find_latest_reports()
        
        # Compare all scenarios
        scenarios = list(uncached_data["scenario_summaries"].keys())
        comparisons = []
        
        for scenario in scenarios:
            comparison = self.compare_scenario_performance(scenario, uncached_data, cached_data)
            comparisons.append(comparison)
        
        # Overall analysis
        timing_changes = [c.performance_change_percent for c in comparisons]
        memory_changes = [c.memory_change_percent for c in comparisons]
        improvements = [c for c in comparisons if c.improvement]
        degradations = [c for c in comparisons if not c.improvement]
        
        overall_analysis = {
            "total_scenarios": len(comparisons),
            "improvements": len(improvements),
            "degradations": len(degradations),
            "avg_timing_change_percent": statistics.mean(timing_changes),
            "avg_memory_change_percent": statistics.mean(memory_changes),
            "best_improvement_percent": min(timing_changes) if timing_changes else 0,
            "worst_degradation_percent": max(timing_changes) if timing_changes else 0,
            "total_time_uncached_sec": uncached_data["overall_analysis"]["performance_summary"].get("total_processing_time_sec", 0),
            "total_time_cached_sec": cached_data["overall_analysis"]["performance_summary"].get("total_processing_time_sec", 0)
        }
        
        if overall_analysis["total_time_uncached_sec"] > 0:
            overall_analysis["total_time_change_percent"] = (
                (overall_analysis["total_time_cached_sec"] - overall_analysis["total_time_uncached_sec"]) / 
                overall_analysis["total_time_uncached_sec"] * 100
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(comparisons, overall_analysis)
        
        return {
            "analysis_type": "phase_4_3_performance_comparison",
            "timestamp": uncached_data["timestamp"],
            "scenario_comparisons": [c.__dict__ for c in comparisons],
            "overall_analysis": overall_analysis,
            "recommendations": recommendations,
            "test_configuration": {
                "uncached_config": uncached_data["configuration"],
                "cached_config": cached_data["configuration"]
            }
        }
    
    def _generate_recommendations(self, comparisons: List[PerformanceComparison], overall: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on performance comparison."""
        recommendations = []
        
        # Overall performance assessment
        if overall["avg_timing_change_percent"] < -5:  # 5% improvement
            recommendations.append("🎉 Caching shows significant performance improvement - recommended for production")
        elif overall["avg_timing_change_percent"] < 0:  # Any improvement
            recommendations.append("✅ Caching shows modest performance improvement - consider enabling")
        elif overall["avg_timing_change_percent"] < 5:  # Less than 5% degradation
            recommendations.append("⚠️ Caching shows minimal performance impact - evaluate based on memory constraints")
        else:
            recommendations.append("❌ Caching shows performance degradation - keep disabled")
        
        # Memory analysis
        if overall["avg_memory_change_percent"] > 50:  # 50% memory increase
            recommendations.append("⚠️ High memory overhead detected - monitor memory pressure in production")
        elif overall["avg_memory_change_percent"] > 20:  # 20% memory increase
            recommendations.append("📊 Moderate memory overhead - acceptable for performance gains")
        
        # Scenario-specific recommendations
        best_performers = [c for c in comparisons if c.performance_change_percent < -10]  # 10% improvement
        if best_performers:
            scenarios = [c.scenario for c in best_performers]
            recommendations.append(f"🔥 High-benefit scenarios for selective caching: {', '.join(scenarios)}")
        
        worst_performers = [c for c in comparisons if c.performance_change_percent > 10]  # 10% degradation
        if worst_performers:
            scenarios = [c.scenario for c in worst_performers]
            recommendations.append(f"⚠️ Consider disabling caching for: {', '.join(scenarios)}")
        
        # Architecture recommendations
        if overall["improvements"] > overall["degradations"]:
            recommendations.append("🏗️ Consider implementing selective caching based on scenario performance")
        
        return recommendations
    
    def print_detailed_report(self, analysis: Dict[str, Any]) -> None:
        """Print a detailed performance comparison report."""
        print(f"\n" + "="*80)
        print(f"📊 PHASE 4.3 PERFORMANCE COMPARISON REPORT")
        print(f"="*80)
        
        overall = analysis["overall_analysis"]
        print(f"\n🎯 OVERALL PERFORMANCE IMPACT:")
        print(f"   Total scenarios tested: {overall['total_scenarios']}")
        print(f"   Scenarios improved: {overall['improvements']}")
        print(f"   Scenarios degraded: {overall['degradations']}")
        print(f"   Average timing change: {overall['avg_timing_change_percent']:+.1f}%")
        print(f"   Average memory change: {overall['avg_memory_change_percent']:+.1f}%")
        print(f"   Best improvement: {overall['best_improvement_percent']:.1f}%")
        print(f"   Worst degradation: {overall['worst_degradation_percent']:+.1f}%")
        
        if 'total_time_change_percent' in overall:
            print(f"   Total processing time change: {overall['total_time_change_percent']:+.1f}%")
        
        print(f"\n📋 SCENARIO BREAKDOWN:")
        print(f"{'Scenario':<25} {'Timing Change':<15} {'Memory Change':<15} {'Status':<10}")
        print(f"{'-'*70}")
        
        for comp_data in analysis["scenario_comparisons"]:
            comp = PerformanceComparison(**comp_data)
            timing_str = f"{comp.performance_change_percent:+.1f}%"
            memory_str = f"{comp.memory_change_percent:+.1f}%"
            status = "✅ IMPROVED" if comp.improvement else "❌ DEGRADED"
            print(f"{comp.scenario:<25} {timing_str:<15} {memory_str:<15} {status:<10}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📈 DETAILED PERFORMANCE DATA:")
        for comp_data in analysis["scenario_comparisons"]:
            comp = PerformanceComparison(**comp_data)
            print(f"\n   {comp.scenario}:")
            print(f"      Uncached: {comp.uncached_mean_ms:.1f}ms, {comp.uncached_memory_mb:.1f}MB")
            print(f"      Cached:   {comp.cached_mean_ms:.1f}ms, {comp.cached_memory_mb:.1f}MB")
            print(f"      Change:   {comp.performance_change_ms:+.1f}ms ({comp.performance_change_percent:+.1f}%), {comp.memory_change_mb:+.1f}MB ({comp.memory_change_percent:+.1f}%)")


def main():
    """Main entry point for performance comparison analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4.3 Performance Comparison Analysis")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark_results"), help="Directory containing benchmark results")
    parser.add_argument("--save-report", type=Path, help="Save detailed analysis to JSON file")
    
    args = parser.parse_args()
    
    try:
        analyzer = PerformanceAnalyzer(results_dir=args.results_dir)
        analysis = analyzer.generate_comprehensive_analysis()
        
        # Print detailed report
        analyzer.print_detailed_report(analysis)
        
        # Save report if requested
        if args.save_report:
            with open(args.save_report, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n💾 Detailed analysis saved to: {args.save_report}")
        
        # Return status based on overall performance
        overall_change = analysis["overall_analysis"]["avg_timing_change_percent"]
        if overall_change < -5:  # 5% improvement
            return 0  # Success - significant improvement
        elif overall_change > 10:  # 10% degradation
            return 1  # Failure - significant degradation
        else:
            return 0  # Success - acceptable performance
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())