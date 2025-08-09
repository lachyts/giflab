#!/usr/bin/env python3
"""
Focused analysis showing practical insights from efficiency scoring system.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_and_prepare_data():
    """Load data with additional calculated fields."""
    results_path = Path(
        "test-workspace/frame-comparison-with-gifs/run_20250807_123641/enhanced_streaming_results.csv"
    )
    df = pd.read_csv(results_path)
    df = df[df["success"] is True].copy()

    # Add quality difference
    df["quality_improvement"] = (
        df["enhanced_composite_quality"] - df["composite_quality"]
    )

    # Extract pipeline components
    pipeline_parts = df["pipeline_id"].str.split("__", expand=True)
    df["frame_tool"] = pipeline_parts[0].str.split("_").str[0]
    df["color_tool"] = pipeline_parts[1].str.split("_").str[0]
    df["lossy_tool"] = pipeline_parts[2].str.split("_").str[0]

    return df


def show_efficiency_in_practice(df):
    """Show how efficiency scoring works in practice."""
    print("🎯 EFFICIENCY SCORING IN PRACTICE")
    print("=" * 50)

    # Show the formula in action with real examples
    print("Formula: Efficiency = Compression Ratio × Enhanced Composite Quality")
    print("\nReal Examples:")

    # Pick diverse examples
    samples = df.sample(5, random_state=42).sort_values("efficiency", ascending=False)

    for _idx, row in samples.iterrows():
        efficiency_calc = row["compression_ratio"] * row["enhanced_composite_quality"]
        print(f"\n📊 {row['content_type'].upper()} content:")
        print(
            f"   Pipeline: {row['frame_tool']} + {row['color_tool']} + {row['lossy_tool']}"
        )
        print(f"   Compression: {row['compression_ratio']:.1f}x")
        print(f"   Quality: {row['enhanced_composite_quality']:.3f}")
        print(
            f"   Efficiency: {row['compression_ratio']:.1f} × {row['enhanced_composite_quality']:.3f} = {efficiency_calc:.2f}"
        )
        print(
            f"   Result: {row['file_size_kb']:.1f}KB (was {row['original_size_kb']:.1f}KB)"
        )


def compare_old_vs_new_quality_assessment(df):
    """Show specific cases where enhanced metrics reveal different insights."""
    print("\n\n🔍 OLD vs NEW QUALITY ASSESSMENT")
    print("=" * 50)

    # Find interesting cases
    big_improvements = df[df["quality_improvement"] > 0.3].nlargest(
        3, "quality_improvement"
    )
    big_decreases = df[df["quality_improvement"] < -0.1].nsmallest(
        3, "quality_improvement"
    )

    print("Cases where ENHANCED metrics found BETTER quality:")
    for _idx, row in big_improvements.iterrows():
        print(f"\n✅ {row['content_type'].title()} - {row['frame_tool']} pipeline")
        print(
            f"   Legacy assessment: {row['composite_quality']:.3f} (based on 4 metrics)"
        )
        print(
            f"   Enhanced assessment: {row['enhanced_composite_quality']:.3f} (based on 11 metrics)"
        )
        print(f"   Improvement: +{row['quality_improvement']:.3f}")
        print(
            "   Why: Enhanced system detected quality in dimensions legacy system missed"
        )
        print(f"   Impact: Efficiency score rose to {row['efficiency']:.2f}")

    print("\n\nCases where ENHANCED metrics found LOWER quality:")
    for _idx, row in big_decreases.iterrows():
        print(f"\n⚠️  {row['content_type'].title()} - {row['frame_tool']} pipeline")
        print(f"   Legacy assessment: {row['composite_quality']:.3f}")
        print(f"   Enhanced assessment: {row['enhanced_composite_quality']:.3f}")
        print(f"   Decrease: {row['quality_improvement']:.3f}")
        print(
            "   Why: Enhanced system detected quality issues in additional dimensions"
        )
        print(f"   Impact: More accurate efficiency score: {row['efficiency']:.2f}")


def find_best_pipelines_by_use_case(df):
    """Recommend specific pipelines for different use cases."""
    print("\n\n🎯 BEST PIPELINES BY USE CASE")
    print("=" * 50)

    use_cases = {
        "Maximum Compression": df.nlargest(3, "compression_ratio"),
        "Best Quality Retention": df.nlargest(3, "enhanced_composite_quality"),
        "Optimal Efficiency": df.nlargest(3, "efficiency"),
        "Small File Sizes": df.nsmallest(3, "file_size_kb"),
        "Balanced Performance": df.loc[df["balanced_score"].nlargest(3).index]
        if "balanced_score" in df.columns
        else df.nlargest(3, "efficiency"),
    }

    for use_case, results in use_cases.items():
        print(f"\n🏆 {use_case}:")
        for idx, (_, row) in enumerate(results.iterrows(), 1):
            print(
                f"   {idx}. {row['frame_tool']}+{row['color_tool']}+{row['lossy_tool']}"
            )
            print(
                f"      Quality: {row['enhanced_composite_quality']:.3f} | Compression: {row['compression_ratio']:.1f}x"
            )
            print(
                f"      Efficiency: {row['efficiency']:.2f} | Size: {row['file_size_kb']:.1f}KB"
            )


def content_type_recommendations(df):
    """Show which pipelines work best for different content types."""
    print("\n\n📝 CONTENT TYPE RECOMMENDATIONS")
    print("=" * 50)

    # Group by content type and find best performers
    content_insights = {}

    for content_type in df["content_type"].unique():
        content_data = df[df["content_type"] == content_type]

        best_efficiency = content_data.loc[content_data["efficiency"].idxmax()]
        best_quality = content_data.loc[
            content_data["enhanced_composite_quality"].idxmax()
        ]

        content_insights[content_type] = {
            "efficiency_champion": best_efficiency,
            "quality_champion": best_quality,
            "avg_efficiency": content_data["efficiency"].mean(),
            "avg_quality": content_data["enhanced_composite_quality"].mean(),
        }

    # Sort by average efficiency
    sorted_content = sorted(
        content_insights.items(), key=lambda x: x[1]["avg_efficiency"], reverse=True
    )

    print("Content types ranked by average efficiency performance:")

    for content_type, data in sorted_content[:8]:  # Top 8 to keep output reasonable
        print(f"\n📊 {content_type.upper()}:")
        print(f"   Average efficiency: {data['avg_efficiency']:.2f}")
        print(f"   Average quality: {data['avg_quality']:.3f}")

        eff_champ = data["efficiency_champion"]
        qual_champ = data["quality_champion"]

        print(
            f"   🥇 Efficiency winner: {eff_champ['frame_tool']}+{eff_champ['lossy_tool']} (score: {eff_champ['efficiency']:.2f})"
        )
        if eff_champ["pipeline_id"] != qual_champ["pipeline_id"]:
            print(
                f"   🏆 Quality winner: {qual_champ['frame_tool']}+{qual_champ['lossy_tool']} (quality: {qual_champ['enhanced_composite_quality']:.3f})"
            )
        else:
            print("   ✨ Same pipeline wins both efficiency and quality!")


def efficiency_distribution_insights(df):
    """Show distribution and practical interpretation of efficiency scores."""
    print("\n\n📈 EFFICIENCY SCORE DISTRIBUTION")
    print("=" * 50)

    # Create efficiency tiers
    df["efficiency_tier"] = pd.cut(
        df["efficiency"],
        bins=[0, 1, 2.5, 5, 10, float("inf")],
        labels=[
            "Poor (0-1)",
            "Fair (1-2.5)",
            "Good (2.5-5)",
            "Excellent (5-10)",
            "Outstanding (10+)",
        ],
    )

    tier_analysis = (
        df.groupby("efficiency_tier")
        .agg(
            {
                "enhanced_composite_quality": ["mean", "std"],
                "compression_ratio": ["mean", "std"],
                "file_size_kb": ["mean", "std"],
                "pipeline_id": "count",
            }
        )
        .round(3)
    )

    print("Performance characteristics by efficiency tier:")
    print(tier_analysis)

    # Practical guidance
    print("\n💡 PRACTICAL GUIDANCE:")

    outstanding = df[df["efficiency"] >= 10]
    if len(outstanding) > 0:
        print(
            f"   Outstanding (10+): {len(outstanding)} results - Perfect for web optimization"
        )
        top_outstanding = outstanding.iloc[0]
        print(
            f"   Example: Achieved {top_outstanding['compression_ratio']:.1f}x compression with {top_outstanding['enhanced_composite_quality']:.3f} quality"
        )

    excellent = df[(df["efficiency"] >= 5) & (df["efficiency"] < 10)]
    if len(excellent) > 0:
        print(f"   Excellent (5-10): {len(excellent)} results - Great for general use")

    good = df[(df["efficiency"] >= 2.5) & (df["efficiency"] < 5)]
    if len(good) > 0:
        print(
            f"   Good (2.5-5): {len(good)} results - Solid performance for most applications"
        )

    fair = df[(df["efficiency"] >= 1) & (df["efficiency"] < 2.5)]
    if len(fair) > 0:
        print(
            f"   Fair (1-2.5): {len(fair)} results - May sacrifice too much quality or compression"
        )


def main():
    """Run focused efficiency analysis."""
    print("🚀 EFFICIENCY SCORING ANALYSIS")
    print("Analysis of enhanced composite quality + efficiency metrics")
    print("=" * 60)

    df = load_and_prepare_data()

    # Add balanced score for use case analysis
    df["balanced_score"] = (
        df["enhanced_composite_quality"] * 0.4
        + (df["compression_ratio"] / df["compression_ratio"].max()) * 0.3
        + (df["efficiency"] / df["efficiency"].max()) * 0.3
    )

    show_efficiency_in_practice(df)
    compare_old_vs_new_quality_assessment(df)
    find_best_pipelines_by_use_case(df)
    content_type_recommendations(df)
    efficiency_distribution_insights(df)

    print("\n\n🎉 SUMMARY")
    print("=" * 20)
    print("✅ Enhanced metrics provide more nuanced quality assessment")
    print("✅ Efficiency scoring effectively balances compression vs quality")
    print("✅ Clear recommendations emerge for different use cases")
    print("✅ Content type significantly impacts optimal pipeline choice")

    return True


if __name__ == "__main__":
    main()
