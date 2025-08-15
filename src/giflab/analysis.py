"""Analysis tools for experimental compression results.

This module provides tools to analyze and visualize experimental compression
results, identify optimal strategies, detect anomalies, and generate
comprehensive reports for decision-making.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class StrategyComparison:
    """Comparison results between compression strategies."""

    strategy_name: str
    avg_compression_ratio: float
    avg_ssim: float
    avg_processing_time: float
    success_rate: float
    file_size_reduction: float
    quality_score: float

    def __str__(self) -> str:
        return f"{self.strategy_name}: {self.avg_compression_ratio:.2f}x compression, {self.avg_ssim:.3f} SSIM, {self.success_rate:.1%} success"


@dataclass
class AnomalyDetection:
    """Results of anomaly detection in compression data."""

    outliers: list[dict[str, Any]]
    suspicious_patterns: list[str]
    recommendations: list[str]

    def __str__(self) -> str:
        return f"Found {len(self.outliers)} outliers and {len(self.suspicious_patterns)} suspicious patterns"


class ExperimentalAnalyzer:
    """Analyzer for experimental compression results."""

    def __init__(self, results_csv: Path):
        """Initialize analyzer with results CSV file.

        Args:
            results_csv: Path to experimental results CSV file
        """
        self.results_csv = results_csv
        self.df = self._load_results()

    def _load_results(self) -> pd.DataFrame:
        """Load and preprocess results data."""
        try:
            df = pd.read_csv(self.results_csv)

            # Convert data types
            df["success"] = df["success"].astype(bool)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Calculate additional metrics
            df["compression_ratio"] = df["orig_kilobytes"] / df["kilobytes"].replace(
                0, 1
            )
            df["file_size_reduction"] = (
                df["orig_kilobytes"] - df["kilobytes"]
            ) / df["orig_kilobytes"]
            # Use composite quality directly as overall quality baseline
            # This avoids double-counting SSIM because composite_quality already
            # incorporates SSIM, MS-SSIM, PSNR and temporal consistency based
            # on the configurable weights in MetricsConfig.
            df["quality_score"] = df["composite_quality"]

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load results: {e}")

    def compare_strategies(
        self, quality_threshold: float = 0.8
    ) -> list[StrategyComparison]:
        """Compare compression strategies across all metrics.

        Args:
            quality_threshold: Minimum quality score for filtering

        Returns:
            List of strategy comparisons sorted by overall performance
        """
        successful_results = self.df[self.df["success"]]

        if successful_results.empty:
            return []

        comparisons = []

        for strategy in successful_results["strategy"].unique():
            strategy_data = successful_results[
                successful_results["strategy"] == strategy
            ]

            # Calculate metrics
            avg_compression = strategy_data["compression_ratio"].mean()
            avg_ssim = strategy_data["ssim"].mean()
            avg_processing_time = strategy_data["processing_time_ms"].mean()
            success_rate = len(strategy_data) / len(
                self.df[self.df["strategy"] == strategy]
            )
            avg_file_reduction = strategy_data["file_size_reduction"].mean()
            avg_quality = strategy_data["quality_score"].mean()

            comparison = StrategyComparison(
                strategy_name=strategy,
                avg_compression_ratio=avg_compression,
                avg_ssim=avg_ssim,
                avg_processing_time=avg_processing_time,
                success_rate=success_rate,
                file_size_reduction=avg_file_reduction,
                quality_score=avg_quality,
            )
            comparisons.append(comparison)

        # Sort by combined score (compression ratio + quality - processing time penalty)
        comparisons.sort(
            key=lambda x: (x.avg_compression_ratio * x.quality_score * x.success_rate)
            - (x.avg_processing_time / 10000),
            reverse=True,
        )

        return comparisons

    def detect_anomalies(self, threshold_std: float = 2.0) -> AnomalyDetection:
        """Detect anomalies in compression results.

        Args:
            threshold_std: Standard deviation threshold for outlier detection

        Returns:
            Anomaly detection results
        """
        successful_results = self.df[self.df["success"]]

        if successful_results.empty:
            return AnomalyDetection([], [], [])

        outliers = []
        suspicious_patterns = []
        recommendations = []

        # Detect outliers in compression ratio
        compression_mean = successful_results["compression_ratio"].mean()
        compression_std = successful_results["compression_ratio"].std()
        compression_outliers = successful_results[
            abs(successful_results["compression_ratio"] - compression_mean)
            > threshold_std * compression_std
        ]

        for _, row in compression_outliers.iterrows():
            outliers.append(
                {
                    "type": "compression_ratio",
                    "gif_sha": row["gif_sha"],
                    "strategy": row["strategy"],
                    "value": row["compression_ratio"],
                    "expected_range": f"{compression_mean - threshold_std * compression_std:.2f} - {compression_mean + threshold_std * compression_std:.2f}",
                }
            )

        # Detect quality outliers
        quality_mean = successful_results["quality_score"].mean()
        quality_std = successful_results["quality_score"].std()
        quality_outliers = successful_results[
            abs(successful_results["quality_score"] - quality_mean)
            > threshold_std * quality_std
        ]

        for _, row in quality_outliers.iterrows():
            outliers.append(
                {
                    "type": "quality_score",
                    "gif_sha": row["gif_sha"],
                    "strategy": row["strategy"],
                    "value": row["quality_score"],
                    "expected_range": f"{quality_mean - threshold_std * quality_std:.2f} - {quality_mean + threshold_std * quality_std:.2f}",
                }
            )

        # Detect suspicious patterns
        failure_rate = len(self.df[not self.df["success"]]) / len(self.df)
        if failure_rate > 0.1:
            suspicious_patterns.append(f"High failure rate: {failure_rate:.1%}")
            recommendations.append(
                "Investigate failed jobs and consider engine availability"
            )

        # Check for strategy-specific issues
        for strategy in self.df["strategy"].unique():
            strategy_data = self.df[self.df["strategy"] == strategy]
            strategy_success_rate = len(strategy_data[strategy_data["success"]]) / len(
                strategy_data
            )

            if strategy_success_rate < 0.8:
                suspicious_patterns.append(
                    f"Low success rate for {strategy}: {strategy_success_rate:.1%}"
                )
                recommendations.append(
                    f"Review {strategy} configuration and engine availability"
                )

        # Check for consistent poor performance
        if successful_results["quality_score"].mean() < 0.6:
            suspicious_patterns.append("Overall low quality scores")
            recommendations.append(
                "Consider adjusting compression parameters or quality thresholds"
            )

        return AnomalyDetection(outliers, suspicious_patterns, recommendations)

    def generate_performance_report(self, output_path: Path) -> dict[str, Any]:
        """Generate comprehensive performance report.

        Args:
            output_path: Path to save the report

        Returns:
            Dictionary containing report data
        """
        report = {
            "summary": self._generate_summary(),
            "strategy_comparison": [
                {
                    "strategy": comp.strategy_name,
                    "compression_ratio": comp.avg_compression_ratio,
                    "ssim": comp.avg_ssim,
                    "processing_time_ms": comp.avg_processing_time,
                    "success_rate": comp.success_rate,
                    "file_size_reduction": comp.file_size_reduction,
                    "quality_score": comp.quality_score,
                }
                for comp in self.compare_strategies()
            ],
            "anomalies": {
                "outliers": self.detect_anomalies().outliers,
                "suspicious_patterns": self.detect_anomalies().suspicious_patterns,
                "recommendations": self.detect_anomalies().recommendations,
            },
            "gif_analysis": self._analyze_by_gif(),
            "parameter_analysis": self._analyze_by_parameters(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total_jobs = len(self.df)
        successful_jobs = len(self.df[self.df["success"]])
        failed_jobs = total_jobs - successful_jobs

        if successful_jobs > 0:
            successful_data = self.df[self.df["success"]]
            avg_compression = successful_data["compression_ratio"].mean()
            avg_quality = successful_data["quality_score"].mean()
            avg_processing_time = successful_data["processing_time_ms"].mean()
        else:
            avg_compression = avg_quality = avg_processing_time = 0

        return {
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
            "avg_compression_ratio": avg_compression,
            "avg_quality_score": avg_quality,
            "avg_processing_time_ms": avg_processing_time,
            "unique_strategies": len(self.df["strategy"].unique()),
            "unique_gifs": len(self.df["gif_sha"].unique()),
        }

    def _analyze_by_gif(self) -> dict[str, Any]:
        """Analyze results by individual GIF."""
        successful_data = self.df[self.df["success"]]

        if successful_data.empty:
            return {}

        gif_analysis = {}

        for gif_sha in successful_data["gif_sha"].unique():
            gif_data = successful_data[successful_data["gif_sha"] == gif_sha]

            # Find best strategy for this GIF
            best_result = gif_data.loc[gif_data["quality_score"].idxmax()]
            worst_result = gif_data.loc[gif_data["quality_score"].idxmin()]

            gif_analysis[gif_sha] = {
                "orig_filename": gif_data["orig_filename"].iloc[0],
                "orig_size_kb": gif_data["orig_kilobytes"].iloc[0],
                "best_strategy": best_result["strategy"],
                "best_compression_ratio": best_result["compression_ratio"],
                "best_quality_score": best_result["quality_score"],
                "worst_strategy": worst_result["strategy"],
                "worst_compression_ratio": worst_result["compression_ratio"],
                "worst_quality_score": worst_result["quality_score"],
                "strategies_tested": len(gif_data["strategy"].unique()),
            }

        return gif_analysis

    def _analyze_by_parameters(self) -> dict[str, Any]:
        """Analyze results by compression parameters."""
        successful_data = self.df[self.df["success"]]

        if successful_data.empty:
            return {}

        analysis = {}

        # Analyze by lossy level
        if "lossy" in successful_data.columns:
            lossy_analysis = {}
            for lossy_level in successful_data["lossy"].unique():
                lossy_data = successful_data[successful_data["lossy"] == lossy_level]
                lossy_analysis[str(lossy_level)] = {
                    "count": len(lossy_data),
                    "avg_compression_ratio": lossy_data["compression_ratio"].mean(),
                    "avg_quality_score": lossy_data["quality_score"].mean(),
                    "avg_processing_time_ms": lossy_data["processing_time_ms"].mean(),
                }
            analysis["lossy_levels"] = lossy_analysis

        # Analyze by frame keep ratio
        if "frame_keep_ratio" in successful_data.columns:
            frame_analysis = {}
            for frame_ratio in successful_data["frame_keep_ratio"].unique():
                frame_data = successful_data[
                    successful_data["frame_keep_ratio"] == frame_ratio
                ]
                frame_analysis[str(frame_ratio)] = {
                    "count": len(frame_data),
                    "avg_compression_ratio": frame_data["compression_ratio"].mean(),
                    "avg_quality_score": frame_data["quality_score"].mean(),
                    "avg_processing_time_ms": frame_data["processing_time_ms"].mean(),
                }
            analysis["frame_keep_ratios"] = frame_analysis

        # Analyze by color count
        if "color_keep_count" in successful_data.columns:
            color_analysis = {}
            for color_count in successful_data["color_keep_count"].unique():
                color_data = successful_data[
                    successful_data["color_keep_count"] == color_count
                ]
                color_analysis[str(color_count)] = {
                    "count": len(color_data),
                    "avg_compression_ratio": color_data["compression_ratio"].mean(),
                    "avg_quality_score": color_data["quality_score"].mean(),
                    "avg_processing_time_ms": color_data["processing_time_ms"].mean(),
                }
            analysis["color_keep_counts"] = color_analysis

        return analysis

    def create_visualizations(self, output_dir: Path) -> list[Path]:
        """Create visualization charts for the experimental results.

        Args:
            output_dir: Directory to save visualization files

        Returns:
            List of paths to generated visualization files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files: list[Path] = []

        successful_data = self.df[self.df["success"]]

        if successful_data.empty:
            return generated_files

        # Set up matplotlib with fallback
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            # Fallback to available styles
            try:
                plt.style.use("seaborn")
            except OSError:
                plt.style.use("default")

        # 1. Strategy comparison chart
        if len(successful_data["strategy"].unique()) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Strategy Comparison", fontsize=16)

            # Compression ratio by strategy
            strategy_compression = successful_data.groupby("strategy")[
                "compression_ratio"
            ].mean()
            axes[0, 0].bar(strategy_compression.index, strategy_compression.values)
            axes[0, 0].set_title("Average Compression Ratio by Strategy")
            axes[0, 0].set_ylabel("Compression Ratio")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # Quality score by strategy
            strategy_quality = successful_data.groupby("strategy")[
                "quality_score"
            ].mean()
            axes[0, 1].bar(strategy_quality.index, strategy_quality.values)
            axes[0, 1].set_title("Average Quality Score by Strategy")
            axes[0, 1].set_ylabel("Quality Score")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Processing time by strategy
            strategy_time = successful_data.groupby("strategy")[
                "processing_time_ms"
            ].mean()
            axes[1, 0].bar(strategy_time.index, strategy_time.values)
            axes[1, 0].set_title("Average Processing Time by Strategy")
            axes[1, 0].set_ylabel("Processing Time (ms)")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Success rate by strategy
            strategy_success = self.df.groupby("strategy")["success"].mean()
            axes[1, 1].bar(strategy_success.index, strategy_success.values)
            axes[1, 1].set_title("Success Rate by Strategy")
            axes[1, 1].set_ylabel("Success Rate")
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            strategy_chart_path = output_dir / "strategy_comparison.png"
            plt.savefig(strategy_chart_path, dpi=300, bbox_inches="tight")
            plt.close()
            generated_files.append(strategy_chart_path)

        # 2. Parameter analysis charts
        if "lossy" in successful_data.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            # Scatter plot: compression ratio vs quality score, colored by lossy level
            scatter = ax.scatter(
                successful_data["compression_ratio"],
                successful_data["quality_score"],
                c=successful_data["lossy"],
                cmap="viridis",
                alpha=0.6,
            )

            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel("Quality Score")
            ax.set_title("Compression Ratio vs Quality Score (colored by Lossy Level)")

            plt.colorbar(scatter, label="Lossy Level")
            plt.tight_layout()

            scatter_path = output_dir / "compression_quality_scatter.png"
            plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close()
            generated_files.append(scatter_path)

        # 3. Distribution charts
        if len(successful_data) > 10:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Distribution Analysis", fontsize=16)

            # Compression ratio distribution
            axes[0, 0].hist(
                successful_data["compression_ratio"],
                bins=20,
                alpha=0.7,
                edgecolor="black",
            )
            axes[0, 0].set_title("Compression Ratio Distribution")
            axes[0, 0].set_xlabel("Compression Ratio")
            axes[0, 0].set_ylabel("Frequency")

            # Quality score distribution
            axes[0, 1].hist(
                successful_data["quality_score"], bins=20, alpha=0.7, edgecolor="black"
            )
            axes[0, 1].set_title("Quality Score Distribution")
            axes[0, 1].set_xlabel("Quality Score")
            axes[0, 1].set_ylabel("Frequency")

            # Processing time distribution
            axes[1, 0].hist(
                successful_data["processing_time_ms"],
                bins=20,
                alpha=0.7,
                edgecolor="black",
            )
            axes[1, 0].set_title("Processing Time Distribution")
            axes[1, 0].set_xlabel("Processing Time (ms)")
            axes[1, 0].set_ylabel("Frequency")

            # File size reduction distribution
            axes[1, 1].hist(
                successful_data["file_size_reduction"],
                bins=20,
                alpha=0.7,
                edgecolor="black",
            )
            axes[1, 1].set_title("File Size Reduction Distribution")
            axes[1, 1].set_xlabel("File Size Reduction")
            axes[1, 1].set_ylabel("Frequency")

            plt.tight_layout()
            distribution_path = output_dir / "distribution_analysis.png"
            plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
            plt.close()
            generated_files.append(distribution_path)

        return generated_files

    def get_recommendations(self) -> list[str]:
        """Get recommendations based on analysis results."""
        recommendations = []

        # Analyze strategy performance
        strategy_comparisons = self.compare_strategies()

        if strategy_comparisons:
            best_strategy = strategy_comparisons[0]
            recommendations.append(
                f"Best overall strategy: {best_strategy.strategy_name} "
                f"({best_strategy.avg_compression_ratio:.2f}x compression, "
                f"{best_strategy.avg_ssim:.3f} SSIM)"
            )

            # Check for poor performers
            for comp in strategy_comparisons:
                if comp.success_rate < 0.8:
                    recommendations.append(
                        f"Consider debugging {comp.strategy_name} - low success rate ({comp.success_rate:.1%})"
                    )

                if comp.avg_quality_score < 0.6:
                    recommendations.append(
                        f"Consider tuning {comp.strategy_name} - low quality score ({comp.avg_quality_score:.3f})"
                    )

        # Analyze anomalies
        anomalies = self.detect_anomalies()
        recommendations.extend(anomalies.recommendations)

        # General recommendations
        successful_data = self.df[self.df["success"]]
        if not successful_data.empty:
            avg_compression = successful_data["compression_ratio"].mean()
            if avg_compression < 2.0:
                recommendations.append(
                    "Consider more aggressive compression settings to improve compression ratios"
                )

            avg_quality = successful_data["quality_score"].mean()
            if avg_quality > 0.9:
                recommendations.append(
                    "High quality scores suggest room for more aggressive compression"
                )

        return recommendations


def analyze_results(
    results_csv: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Analyze experimental results and generate comprehensive report.

    Args:
        results_csv: Path to experimental results CSV file
        output_dir: Directory to save analysis outputs (optional)

    Returns:
        Dictionary containing analysis results
    """
    analyzer = ExperimentalAnalyzer(results_csv)

    # Generate analysis
    analysis = {
        "strategy_comparison": analyzer.compare_strategies(),
        "anomalies": analyzer.detect_anomalies(),
        "recommendations": analyzer.get_recommendations(),
    }

    # Generate outputs if directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate performance report
        report_path = output_dir / "performance_report.json"
        analyzer.generate_performance_report(report_path)

        # Generate visualizations
        viz_dir = output_dir / "visualizations"
        generated_charts = analyzer.create_visualizations(viz_dir)

        analysis["report_path"] = str(report_path)
        analysis["visualization_paths"] = [str(p) for p in generated_charts]

    return analysis
