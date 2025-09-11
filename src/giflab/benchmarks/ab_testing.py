#!/usr/bin/env python3
"""
A/B Testing Framework for GifLab optimization tuning.

This module provides capabilities to compare different optimization
configurations in production with statistical significance testing.
"""

import json
import hashlib
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import statistics
import math
from scipy import stats
import numpy as np

from giflab.config_manager import get_config_manager, ConfigManager
from giflab.monitoring import get_metrics_collector
from giflab.validation import validate_optimization


class VariantType(Enum):
    """Types of configuration variants for A/B testing."""
    CONTROL = "control"  # Current production configuration
    TREATMENT = "treatment"  # New configuration to test


class TrafficSplitStrategy(Enum):
    """Strategies for splitting traffic between variants."""
    RANDOM = "random"  # Random assignment
    HASH_BASED = "hash_based"  # Deterministic based on input hash
    WEIGHTED = "weighted"  # Weighted random assignment
    PROGRESSIVE = "progressive"  # Gradually increase treatment traffic


@dataclass
class ConfigurationVariant:
    """Represents a configuration variant for testing."""
    name: str
    variant_type: VariantType
    config_overrides: Dict[str, Any]
    description: str
    weight: float = 0.5  # Traffic weight (0.0 to 1.0)
    
    def apply(self, config_manager: ConfigManager) -> None:
        """Apply this variant's configuration overrides."""
        for path, value in self.config_overrides.items():
            config_manager.set(path, value)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        data['variant_type'] = self.variant_type.value
        return json.dumps(data, indent=2)


@dataclass
class ExperimentMetrics:
    """Metrics collected during an A/B test experiment."""
    variant_name: str
    sample_count: int
    metrics: Dict[str, List[float]]
    start_time: datetime
    end_time: Optional[datetime] = None
    
    def add_observation(self, metric_name: str, value: float) -> None:
        """Add a metric observation."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        self.sample_count = max(len(values) for values in self.metrics.values())
    
    def get_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    relative_change: float  # Percentage change
    p_value: float
    confidence_level: float
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    test_type: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ABTestExperiment:
    """Manages an A/B test experiment."""
    
    def __init__(
        self,
        name: str,
        control_variant: ConfigurationVariant,
        treatment_variants: List[ConfigurationVariant],
        split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM,
        confidence_level: float = 0.95,
        minimum_sample_size: int = 100
    ):
        """Initialize an A/B test experiment."""
        self.name = name
        self.control_variant = control_variant
        self.treatment_variants = treatment_variants
        self.split_strategy = split_strategy
        self.confidence_level = confidence_level
        self.minimum_sample_size = minimum_sample_size
        
        # Metrics collection
        self.metrics_by_variant: Dict[str, ExperimentMetrics] = {}
        self._initialize_metrics()
        
        # Experiment state
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.is_active = True
        
        # Results cache
        self._cached_results: Optional[Dict] = None
        self._results_cache_time: Optional[datetime] = None
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics collection for all variants."""
        self.metrics_by_variant[self.control_variant.name] = ExperimentMetrics(
            variant_name=self.control_variant.name,
            sample_count=0,
            metrics={},
            start_time=datetime.now()
        )
        
        for variant in self.treatment_variants:
            self.metrics_by_variant[variant.name] = ExperimentMetrics(
                variant_name=variant.name,
                sample_count=0,
                metrics={},
                start_time=datetime.now()
            )
    
    def assign_variant(self, input_identifier: str) -> ConfigurationVariant:
        """
        Assign a variant based on the split strategy.
        
        Args:
            input_identifier: Unique identifier for the input (e.g., file path)
            
        Returns:
            The assigned configuration variant
        """
        if not self.is_active:
            return self.control_variant
        
        if self.split_strategy == TrafficSplitStrategy.RANDOM:
            return self._assign_random()
        elif self.split_strategy == TrafficSplitStrategy.HASH_BASED:
            return self._assign_hash_based(input_identifier)
        elif self.split_strategy == TrafficSplitStrategy.WEIGHTED:
            return self._assign_weighted()
        elif self.split_strategy == TrafficSplitStrategy.PROGRESSIVE:
            return self._assign_progressive()
        else:
            return self.control_variant
    
    def _assign_random(self) -> ConfigurationVariant:
        """Random variant assignment."""
        all_variants = [self.control_variant] + self.treatment_variants
        return random.choice(all_variants)
    
    def _assign_hash_based(self, identifier: str) -> ConfigurationVariant:
        """Deterministic assignment based on input hash."""
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        all_variants = [self.control_variant] + self.treatment_variants
        index = hash_value % len(all_variants)
        return all_variants[index]
    
    def _assign_weighted(self) -> ConfigurationVariant:
        """Weighted random assignment."""
        all_variants = [self.control_variant] + self.treatment_variants
        weights = [v.weight for v in all_variants]
        return random.choices(all_variants, weights=weights, k=1)[0]
    
    def _assign_progressive(self) -> ConfigurationVariant:
        """Progressive rollout - gradually increase treatment traffic."""
        # Calculate days since experiment start
        days_elapsed = (datetime.now() - self.start_time).days
        
        # Start with 10% treatment, increase by 10% per day up to 50%
        treatment_percentage = min(0.5, 0.1 + (days_elapsed * 0.1))
        
        if random.random() < treatment_percentage and self.treatment_variants:
            return random.choice(self.treatment_variants)
        else:
            return self.control_variant
    
    def record_observation(
        self,
        variant_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """Record metric observations for a variant."""
        if variant_name not in self.metrics_by_variant:
            return
        
        experiment_metrics = self.metrics_by_variant[variant_name]
        for metric_name, value in metrics.items():
            experiment_metrics.add_observation(metric_name, value)
        
        # Invalidate results cache
        self._cached_results = None
    
    def calculate_statistical_significance(
        self,
        metric_name: str,
        treatment_variant_name: Optional[str] = None
    ) -> List[StatisticalTestResult]:
        """
        Calculate statistical significance for a metric.
        
        Args:
            metric_name: Name of the metric to test
            treatment_variant_name: Specific treatment to test (or all if None)
            
        Returns:
            List of statistical test results
        """
        results = []
        control_metrics = self.metrics_by_variant[self.control_variant.name]
        
        if metric_name not in control_metrics.metrics:
            return results
        
        control_values = control_metrics.metrics[metric_name]
        
        # Test each treatment variant
        variants_to_test = self.treatment_variants
        if treatment_variant_name:
            variants_to_test = [
                v for v in self.treatment_variants 
                if v.name == treatment_variant_name
            ]
        
        for variant in variants_to_test:
            variant_metrics = self.metrics_by_variant[variant.name]
            
            if metric_name not in variant_metrics.metrics:
                continue
            
            treatment_values = variant_metrics.metrics[metric_name]
            
            # Require minimum sample size
            if (len(control_values) < self.minimum_sample_size or 
                len(treatment_values) < self.minimum_sample_size):
                continue
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            control_mean = statistics.mean(control_values)
            treatment_mean = statistics.mean(treatment_values)
            
            # Calculate relative change
            if control_mean != 0:
                relative_change = ((treatment_mean - control_mean) / control_mean) * 100
            else:
                relative_change = 0
            
            result = StatisticalTestResult(
                metric_name=metric_name,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                relative_change=relative_change,
                p_value=p_value,
                confidence_level=self.confidence_level,
                is_significant=p_value < (1 - self.confidence_level),
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values),
                test_type="t-test"
            )
            
            results.append(result)
        
        return results
    
    def get_experiment_summary(self) -> Dict:
        """Get a comprehensive summary of the experiment."""
        # Use cache if available and recent
        if (self._cached_results and self._results_cache_time and 
            datetime.now() - self._results_cache_time < timedelta(minutes=5)):
            return self._cached_results
        
        summary = {
            "name": self.name,
            "status": "active" if self.is_active else "completed",
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_hours": (
                (self.end_time or datetime.now()) - self.start_time
            ).total_seconds() / 3600,
            "variants": {},
            "statistical_results": {},
            "recommendations": []
        }
        
        # Collect variant summaries
        for variant_name, metrics in self.metrics_by_variant.items():
            variant_summary = {
                "sample_count": metrics.sample_count,
                "metrics": {}
            }
            
            for metric_name in metrics.metrics.keys():
                variant_summary["metrics"][metric_name] = metrics.get_summary(metric_name)
            
            summary["variants"][variant_name] = variant_summary
        
        # Calculate statistical significance for key metrics
        key_metrics = [
            "total_validation_time_ms",
            "frame_cache_hit_rate",
            "validation_cache_hit_rate",
            "memory_usage_peak_mb"
        ]
        
        for metric in key_metrics:
            results = self.calculate_statistical_significance(metric)
            if results:
                summary["statistical_results"][metric] = [
                    r.to_dict() for r in results
                ]
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(
            summary["statistical_results"]
        )
        
        # Cache results
        self._cached_results = summary
        self._results_cache_time = datetime.now()
        
        return summary
    
    def _generate_recommendations(
        self, 
        statistical_results: Dict
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        significant_improvements = []
        significant_regressions = []
        
        for metric_name, results in statistical_results.items():
            for result in results:
                if result['is_significant']:
                    if result['relative_change'] > 5:
                        # Significant improvement
                        significant_improvements.append({
                            "metric": metric_name,
                            "change": result['relative_change'],
                            "variant": result['metric_name']
                        })
                    elif result['relative_change'] < -5:
                        # Significant regression
                        significant_regressions.append({
                            "metric": metric_name,
                            "change": result['relative_change'],
                            "variant": result['metric_name']
                        })
        
        if significant_improvements and not significant_regressions:
            recommendations.append(
                "✅ RECOMMEND: Deploy treatment configuration - significant improvements detected"
            )
            for imp in significant_improvements:
                recommendations.append(
                    f"  • {imp['metric']}: {imp['change']:.1f}% improvement"
                )
        elif significant_regressions:
            recommendations.append(
                "❌ RECOMMEND: Keep control configuration - regressions detected"
            )
            for reg in significant_regressions:
                recommendations.append(
                    f"  • {reg['metric']}: {abs(reg['change']):.1f}% regression"
                )
        else:
            recommendations.append(
                "⏳ RECOMMEND: Continue experiment - insufficient data for decision"
            )
            recommendations.append(
                f"  • Current sample size: {self.metrics_by_variant[self.control_variant.name].sample_count}"
            )
            recommendations.append(
                f"  • Minimum required: {self.minimum_sample_size}"
            )
        
        return recommendations
    
    def should_stop_experiment(self) -> Tuple[bool, str]:
        """
        Determine if the experiment should be stopped.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check for sufficient sample size
        control_count = self.metrics_by_variant[self.control_variant.name].sample_count
        
        if control_count < self.minimum_sample_size:
            return False, "Insufficient sample size"
        
        # Check for clear winner
        significant_results = []
        for metric_name in ["total_validation_time_ms", "frame_cache_hit_rate"]:
            results = self.calculate_statistical_significance(metric_name)
            significant_results.extend([r for r in results if r.is_significant])
        
        if len(significant_results) >= 2:
            # Multiple significant results - can make decision
            avg_change = statistics.mean([r.relative_change for r in significant_results])
            if abs(avg_change) > 10:
                return True, f"Clear winner detected with {avg_change:.1f}% average change"
        
        # Check for maximum duration (7 days)
        if datetime.now() - self.start_time > timedelta(days=7):
            return True, "Maximum experiment duration reached"
        
        return False, "Experiment ongoing"
    
    def stop_experiment(self) -> None:
        """Stop the experiment."""
        self.is_active = False
        self.end_time = datetime.now()
        
        # Mark all metrics as ended
        for metrics in self.metrics_by_variant.values():
            metrics.end_time = self.end_time


class ABTestingFramework:
    """Main framework for managing A/B tests."""
    
    def __init__(self, experiments_dir: Optional[Path] = None):
        """Initialize the A/B testing framework."""
        self.experiments_dir = experiments_dir or Path.home() / ".giflab" / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.active_experiments: Dict[str, ABTestExperiment] = {}
        self.config_manager = get_config_manager()
        self.metrics_collector = get_metrics_collector()
        
        # Load active experiments
        self._load_active_experiments()
    
    def _load_active_experiments(self) -> None:
        """Load active experiments from disk."""
        experiments_file = self.experiments_dir / "active_experiments.json"
        if experiments_file.exists():
            with open(experiments_file, 'r') as f:
                data = json.load(f)
                # Reconstruct experiments from saved data
                # (Simplified - in production would need full deserialization)
    
    def create_experiment(
        self,
        name: str,
        control_config: Dict[str, Any],
        treatment_configs: List[Dict[str, Any]],
        split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM,
        description: str = ""
    ) -> ABTestExperiment:
        """Create a new A/B test experiment."""
        # Create control variant
        control = ConfigurationVariant(
            name=f"{name}_control",
            variant_type=VariantType.CONTROL,
            config_overrides=control_config,
            description=f"Control variant for {name}"
        )
        
        # Create treatment variants
        treatments = []
        for i, config in enumerate(treatment_configs):
            treatment = ConfigurationVariant(
                name=f"{name}_treatment_{i+1}",
                variant_type=VariantType.TREATMENT,
                config_overrides=config,
                description=f"Treatment variant {i+1} for {name}"
            )
            treatments.append(treatment)
        
        # Create experiment
        experiment = ABTestExperiment(
            name=name,
            control_variant=control,
            treatment_variants=treatments,
            split_strategy=split_strategy
        )
        
        self.active_experiments[name] = experiment
        self._save_active_experiments()
        
        return experiment
    
    def run_validation_with_experiment(
        self,
        experiment_name: str,
        original_path: str,
        compressed_path: str,
        **validation_kwargs
    ) -> Tuple[Dict, str]:
        """
        Run validation with A/B test experiment.
        
        Returns:
            Tuple of (validation_results, variant_name)
        """
        if experiment_name not in self.active_experiments:
            # No experiment - run with default config
            results = validate_optimization(
                original_path, 
                compressed_path, 
                **validation_kwargs
            )
            return results, "default"
        
        experiment = self.active_experiments[experiment_name]
        
        # Assign variant
        variant = experiment.assign_variant(original_path)
        
        # Apply variant configuration
        original_config = self.config_manager.export_config()
        try:
            variant.apply(self.config_manager)
            
            # Run validation with variant config
            start_time = time.perf_counter()
            results = validate_optimization(
                original_path,
                compressed_path,
                **validation_kwargs
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Record observations
            metrics = {
                "total_validation_time_ms": elapsed_ms,
                "frame_cache_hit_rate": results.get('_cache_hit_rate', 0),
                "validation_cache_hit_rate": results.get('_validation_cache_hit_rate', 0),
                "memory_usage_peak_mb": results.get('_memory_peak_mb', 0)
            }
            experiment.record_observation(variant.name, metrics)
            
            # Check if experiment should stop
            should_stop, reason = experiment.should_stop_experiment()
            if should_stop:
                print(f"Stopping experiment '{experiment_name}': {reason}")
                experiment.stop_experiment()
                self._save_active_experiments()
            
            return results, variant.name
            
        finally:
            # Restore original configuration
            self.config_manager.import_config(original_config)
    
    def get_experiment_status(self, experiment_name: str) -> Dict:
        """Get the status of an experiment."""
        if experiment_name not in self.active_experiments:
            return {"error": f"Experiment '{experiment_name}' not found"}
        
        experiment = self.active_experiments[experiment_name]
        return experiment.get_experiment_summary()
    
    def finalize_experiment(
        self, 
        experiment_name: str,
        deploy_winner: bool = False
    ) -> Dict:
        """
        Finalize an experiment and optionally deploy the winner.
        
        Args:
            experiment_name: Name of the experiment
            deploy_winner: Whether to deploy the winning configuration
            
        Returns:
            Final experiment results
        """
        if experiment_name not in self.active_experiments:
            return {"error": f"Experiment '{experiment_name}' not found"}
        
        experiment = self.active_experiments[experiment_name]
        experiment.stop_experiment()
        
        # Get final results
        summary = experiment.get_experiment_summary()
        
        # Save results to disk
        results_file = self.experiments_dir / f"{experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Deploy winner if requested
        if deploy_winner and summary["recommendations"]:
            first_rec = summary["recommendations"][0]
            if "RECOMMEND: Deploy treatment" in first_rec:
                # Find best treatment variant
                best_variant = None
                best_improvement = 0
                
                for variant in experiment.treatment_variants:
                    # Calculate average improvement
                    improvements = []
                    for metric_results in summary["statistical_results"].values():
                        for result in metric_results:
                            if result["is_significant"] and result["relative_change"] > 0:
                                improvements.append(result["relative_change"])
                    
                    if improvements:
                        avg_improvement = statistics.mean(improvements)
                        if avg_improvement > best_improvement:
                            best_improvement = avg_improvement
                            best_variant = variant
                
                if best_variant:
                    # Apply winning configuration
                    best_variant.apply(self.config_manager)
                    summary["deployed_variant"] = best_variant.name
                    print(f"✅ Deployed winning variant: {best_variant.name}")
        
        # Remove from active experiments
        del self.active_experiments[experiment_name]
        self._save_active_experiments()
        
        return summary
    
    def _save_active_experiments(self) -> None:
        """Save active experiments to disk."""
        experiments_file = self.experiments_dir / "active_experiments.json"
        data = {
            name: {
                "name": exp.name,
                "start_time": exp.start_time.isoformat(),
                "is_active": exp.is_active,
                "sample_counts": {
                    v: m.sample_count 
                    for v, m in exp.metrics_by_variant.items()
                }
            }
            for name, exp in self.active_experiments.items()
        }
        
        with open(experiments_file, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """Example usage of A/B testing framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GifLab A/B Testing Framework")
    parser.add_argument("command", choices=["create", "status", "finalize", "list"])
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--control-config", type=json.loads, 
                       help="Control configuration as JSON")
    parser.add_argument("--treatment-config", type=json.loads,
                       help="Treatment configuration as JSON")
    parser.add_argument("--deploy-winner", action="store_true",
                       help="Deploy winning configuration")
    
    args = parser.parse_args()
    
    framework = ABTestingFramework()
    
    if args.command == "create":
        if not args.name or not args.control_config or not args.treatment_config:
            print("Error: create requires --name, --control-config, and --treatment-config")
            return 1
        
        experiment = framework.create_experiment(
            name=args.name,
            control_config=args.control_config,
            treatment_configs=[args.treatment_config],
            description="CLI-created experiment"
        )
        print(f"Created experiment: {experiment.name}")
    
    elif args.command == "status":
        if not args.name:
            print("Error: status requires --name")
            return 1
        
        status = framework.get_experiment_status(args.name)
        print(json.dumps(status, indent=2))
    
    elif args.command == "finalize":
        if not args.name:
            print("Error: finalize requires --name")
            return 1
        
        results = framework.finalize_experiment(args.name, args.deploy_winner)
        print(json.dumps(results, indent=2))
    
    elif args.command == "list":
        print("Active experiments:")
        for name in framework.active_experiments.keys():
            print(f"  - {name}")
    
    return 0


if __name__ == "__main__":
    exit(main())