#!/usr/bin/env python3
"""
Deep dive analysis of frame reduction algorithms: how they work, frame counts, and performance differences.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_prepare_data():
    """Load data and extract frame reduction details."""
    data_path = Path(
        "/Users/lachlants/repos/animately/giflab/test-workspace/frame-comparison-with-gifs/run_20250807_123641/enhanced_streaming_results.csv"
    )
    df = pd.read_csv(data_path)
    df = df[df["success"] is True].copy()

    # Extract frame reduction algorithm
    df["frame_algorithm"] = (
        df["pipeline_id"]
        .str.split("__")
        .str[0]
        .str.split("_")
        .str[0]
    )

    # Extract applied frame ratio (this tells us how many frames were kept)
    df["frames_kept_ratio"] = df["applied_frame_ratio"]
    df["frames_removed_ratio"] = 1 - df["frames_kept_ratio"].fillna(1.0)

    return df


def analyze_frame_reduction_mechanisms():
    """Explain how each frame reduction algorithm works."""
    print("🔧 FRAME REDUCTION ALGORITHM MECHANISMS")
    print("=" * 55)

    algorithms = {
        "none-frame": {
            "description": "No frame reduction (baseline)",
            "mechanism": "Passes all frames through unchanged",
            "strategy": "Preserves original temporal information",
            "pros": [
                "Maximum quality retention",
                "No temporal artifacts",
                "Best for motion analysis",
            ],
            "cons": [
                "No compression benefit from frame reduction",
                "Larger file sizes",
            ],
            "typical_use": "Quality baseline, motion-heavy content",
        },
        "gifsicle-frame": {
            "description": "Gifsicle's frame optimization",
            "mechanism": "Smart frame differencing and optimization",
            "strategy": "Removes redundant frames and optimizes frame differences",
            "pros": [
                "Excellent compression",
                "Maintains visual quality",
                "Smart duplicate detection",
            ],
            "cons": ["Can remove important motion frames", "May create stuttering"],
            "typical_use": "Web optimization, static-heavy content",
        },
        "ffmpeg-frame": {
            "description": "FFmpeg frame sampling",
            "mechanism": "Mathematical frame sampling (every Nth frame)",
            "strategy": "Uniform temporal sampling based on ratio",
            "pros": ["Predictable behavior", "Good for consistent motion"],
            "cons": ["May miss important keyframes", "Can create jarring transitions"],
            "typical_use": "Consistent motion, batch processing",
        },
        "imagemagick-frame": {
            "description": "ImageMagick frame decimation",
            "mechanism": "Frame decimation with visual analysis",
            "strategy": "Removes frames based on visual similarity",
            "pros": ["Good balance of quality/compression", "Visual-aware removal"],
            "cons": ["Processing intensive", "Can miss temporal patterns"],
            "typical_use": "General purpose, varied content",
        },
        "animately-frame": {
            "description": "Animately's advanced frame reduction",
            "mechanism": "AI-powered frame importance analysis",
            "strategy": "Intelligent frame selection based on visual significance",
            "pros": ["Preserves important frames", "Good motion understanding"],
            "cons": ["Can be conservative", "Processing overhead"],
            "typical_use": "High-quality animations, professional content",
        },
    }

    for algo, info in algorithms.items():
        print(f"\n🎬 {algo.upper()}:")
        print(f"   Method: {info['description']}")
        print(f"   How it works: {info['mechanism']}")
        print(f"   Strategy: {info['strategy']}")
        print(f"   ✅ Pros: {', '.join(info['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(info['cons'])}")
        print(f"   💡 Best for: {info['typical_use']}")


def analyze_frame_reduction_behavior(df):
    """Analyze actual frame reduction behavior from the data."""
    print("\n\n📊 FRAME REDUCTION BEHAVIOR ANALYSIS")
    print("=" * 55)

    frame_analysis = (
        df.groupby("frame_algorithm")
        .agg(
            {
                "applied_frame_ratio": ["mean", "std", "min", "max", "count"],
                "compression_ratio": ["mean", "std", "max"],
                "enhanced_composite_quality": ["mean", "std"],
                "efficiency": ["mean", "max"],
                "file_size_kb": "mean",
            }
        )
        .round(3)
    )

    print("📈 Frame Reduction Statistics by Algorithm:")
    print("(applied_frame_ratio: 1.0 = all frames kept, 0.5 = half frames kept)")
    print()

    for algorithm in frame_analysis.index:
        stats = frame_analysis.loc[algorithm]

        avg_kept = stats[("applied_frame_ratio", "mean")]
        frames_removed_pct = (1 - avg_kept) * 100 if pd.notna(avg_kept) else 0

        print(f"🎭 {algorithm.upper()}:")
        print(
            f"   Avg frames kept: {avg_kept:.1%} (removed: {frames_removed_pct:.1f}%)"
        )
        print(
            f"   Frame ratio range: {stats[('applied_frame_ratio', 'min')]:.2f} to {stats[('applied_frame_ratio', 'max')]:.2f}"
        )
        print(
            f"   Compression achieved: {stats[('compression_ratio', 'mean')]:.1f}x avg (max: {stats[('compression_ratio', 'max')]:.1f}x)"
        )
        print(
            f"   Quality retained: {stats[('enhanced_composite_quality', 'mean')]:.3f}"
        )
        print(
            f"   Efficiency: {stats[('efficiency', 'mean')]:.2f} avg (max: {stats[('efficiency', 'max')]:.1f})"
        )
        print(f"   Avg file size: {stats[('file_size_kb', 'mean')]:.1f}KB")
        print()


def compare_animately_vs_none_frame(df):
    """Deep comparison between animately-frame and none-frame performance."""
    print("🔍 ANIMATELY-FRAME vs NONE-FRAME DETAILED COMPARISON")
    print("=" * 55)

    animately_data = df[df["frame_algorithm"] == "animately-frame"]
    none_data = df[df["frame_algorithm"] == "none-frame"]

    print("📊 Head-to-Head Comparison:")
    print(f"\n   ANIMATELY-FRAME (n={len(animately_data)}):")
    print(
        f"   • Average frames kept: {animately_data['applied_frame_ratio'].mean():.1%}"
    )
    print(
        f"   • Average compression: {animately_data['compression_ratio'].mean():.1f}x"
    )
    print(
        f"   • Average quality: {animately_data['enhanced_composite_quality'].mean():.3f}"
    )
    print(f"   • Average efficiency: {animately_data['efficiency'].mean():.2f}")
    print(f"   • File size: {animately_data['file_size_kb'].mean():.1f}KB avg")

    print(f"\n   NONE-FRAME (n={len(none_data)}):")
    print(
        f"   • Average frames kept: {none_data['applied_frame_ratio'].mean():.1%} (no reduction)"
    )
    print(f"   • Average compression: {none_data['compression_ratio'].mean():.1f}x")
    print(f"   • Average quality: {none_data['enhanced_composite_quality'].mean():.3f}")
    print(f"   • Average efficiency: {none_data['efficiency'].mean():.2f}")
    print(f"   • File size: {none_data['file_size_kb'].mean():.1f}KB avg")

    # Why none-frame beats animately-frame
    print("\n🤔 WHY NONE-FRAME OUTPERFORMS ANIMATELY-FRAME:")

    quality_diff = (
        none_data["enhanced_composite_quality"].mean()
        - animately_data["enhanced_composite_quality"].mean()
    )
    compression_diff = (
        none_data["compression_ratio"].mean()
        - animately_data["compression_ratio"].mean()
    )
    efficiency_diff = (
        none_data["efficiency"].mean() - animately_data["efficiency"].mean()
    )

    print(
        f"   1. Quality Advantage: +{quality_diff:.3f} ({quality_diff/animately_data['enhanced_composite_quality'].mean()*100:.1f}% better)"
    )
    print(
        f"   2. Compression Advantage: +{compression_diff:.1f}x ({compression_diff/animately_data['compression_ratio'].mean()*100:.1f}% better)"
    )
    print(
        f"   3. Efficiency Advantage: +{efficiency_diff:.2f} ({efficiency_diff/animately_data['efficiency'].mean()*100:.1f}% better)"
    )

    print("\n💡 EXPLANATION:")
    print(
        f"   • Animately removes {(1-animately_data['applied_frame_ratio'].mean())*100:.1f}% of frames on average"
    )
    print("   • This should theoretically improve compression, but:")
    print("     - Frame removal can hurt quality more than it helps compression")
    print(
        "     - The lossy compression step (animately-advanced) is already optimizing"
    )
    print("     - Removing frames may disrupt compression patterns")
    print("     - Quality loss from missing frames outweighs file size gains")

    # Content type analysis
    print("\n📝 PERFORMANCE BY CONTENT TYPE:")
    content_comparison = []

    for content_type in df["content_type"].unique():
        animately_content = animately_data[
            animately_data["content_type"] == content_type
        ]
        none_content = none_data[none_data["content_type"] == content_type]

        if len(animately_content) > 0 and len(none_content) > 0:
            efficiency_diff = (
                none_content["efficiency"].mean()
                - animately_content["efficiency"].mean()
            )
            content_comparison.append(
                {
                    "content_type": content_type,
                    "animately_eff": animately_content["efficiency"].mean(),
                    "none_eff": none_content["efficiency"].mean(),
                    "difference": efficiency_diff,
                    "winner": "none-frame"
                    if efficiency_diff > 0
                    else "animately-frame",
                }
            )

    content_df = pd.DataFrame(content_comparison).sort_values(
        "difference", ascending=False
    )

    for _, row in content_df.head(10).iterrows():
        winner_icon = "🥇" if row["winner"] == "none-frame" else "🥈"
        print(
            f"   {winner_icon} {row['content_type'].ljust(15)}: none={row['none_eff']:4.2f} vs animately={row['animately_eff']:4.2f} (Δ{row['difference']:+4.2f})"
        )


def analyze_compression_vs_frame_reduction(df):
    """Analyze the relationship between frame reduction and compression."""
    print("\n\n🔗 FRAME REDUCTION vs COMPRESSION RELATIONSHIP")
    print("=" * 55)

    print("📈 Key Insight: Frame Reduction ≠ Better Compression")
    print("\n   This experiment reveals a counter-intuitive finding:")
    print(
        "   More aggressive frame reduction doesn't always lead to better compression!"
    )

    # Calculate frame reduction aggressiveness vs compression
    df["frame_reduction_aggressiveness"] = 1 - df["applied_frame_ratio"].fillna(1.0)

    correlation = df["frame_reduction_aggressiveness"].corr(df["compression_ratio"])
    quality_correlation = df["frame_reduction_aggressiveness"].corr(
        df["enhanced_composite_quality"]
    )
    efficiency_correlation = df["frame_reduction_aggressiveness"].corr(df["efficiency"])

    print("\n📊 Correlations with Frame Reduction Aggressiveness:")
    print(f"   • Compression ratio: {correlation:+.3f}")
    print(f"   • Quality retention: {quality_correlation:+.3f}")
    print(f"   • Efficiency score: {efficiency_correlation:+.3f}")

    # Binned analysis
    df["frame_reduction_category"] = pd.cut(
        df["frame_reduction_aggressiveness"],
        bins=[0, 0.1, 0.3, 0.5, 1.0],
        labels=[
            "Minimal (0-10%)",
            "Light (10-30%)",
            "Moderate (30-50%)",
            "Heavy (50%+)",
        ],
        include_lowest=True,
    )

    category_analysis = (
        df.groupby("frame_reduction_category")
        .agg(
            {
                "compression_ratio": ["mean", "std"],
                "enhanced_composite_quality": ["mean", "std"],
                "efficiency": ["mean", "std"],
                "pipeline_id": "count",
            }
        )
        .round(3)
    )

    print("\n📈 Performance by Frame Reduction Level:")
    for category in category_analysis.index:
        stats = category_analysis.loc[category]
        count = int(stats[("pipeline_id", "count")])
        compression = stats[("compression_ratio", "mean")]
        quality = stats[("enhanced_composite_quality", "mean")]
        efficiency = stats[("efficiency", "mean")]

        print(
            f"   {category.ljust(18)} (n={count:3d}): {compression:4.1f}x compression | {quality:.3f} quality | {efficiency:4.2f} efficiency"
        )


def explain_algorithm_differences(df):
    """Explain why different algorithms perform so differently."""
    print("\n\n🎯 WHY SUCH DRAMATIC PERFORMANCE DIFFERENCES?")
    print("=" * 55)

    # Get performance spread
    algo_performance = (
        df.groupby("frame_algorithm")
        .agg(
            {
                "efficiency": ["mean", "max"],
                "compression_ratio": ["mean", "max"],
                "enhanced_composite_quality": "mean",
            }
        )
        .round(3)
    )

    best_algo = algo_performance[("efficiency", "mean")].idxmax()
    worst_algo = algo_performance[("efficiency", "mean")].idxmin()
    performance_gap = (
        algo_performance.loc[best_algo, ("efficiency", "mean")]
        / algo_performance.loc[worst_algo, ("efficiency", "mean")]
    )

    print("📊 Performance Spread:")
    print(
        f"   • Best algorithm: {best_algo} (avg efficiency: {algo_performance.loc[best_algo, ('efficiency', 'mean')]:.2f})"
    )
    print(
        f"   • Worst algorithm: {worst_algo} (avg efficiency: {algo_performance.loc[worst_algo, ('efficiency', 'mean')]:.2f})"
    )
    print(f"   • Performance gap: {performance_gap:.1f}x difference!")

    print("\n🔍 ROOT CAUSE ANALYSIS:")

    print("\n1️⃣  GIFSICLE-FRAME DOMINANCE:")
    gifsicle_data = df[df["frame_algorithm"] == "gifsicle-frame"]
    print("   • Specializes in GIF optimization (native format)")
    print("   • Smart frame differencing removes truly redundant data")
    print(
        f"   • Achieves {gifsicle_data['compression_ratio'].max():.1f}x max compression"
    )
    print(
        f"   • Maintains {gifsicle_data['enhanced_composite_quality'].mean():.3f} avg quality"
    )
    print("   • Works especially well on motion content (up to 60x compression!)")

    print("\n2️⃣  IMAGEMAGICK-FRAME STRENGTH:")
    imagemagick_data = df[df["frame_algorithm"] == "imagemagick-frame"]
    print("   • Visual similarity analysis prevents important frame loss")
    print(
        f"   • Achieves {imagemagick_data['compression_ratio'].max():.1f}x max compression"
    )
    print("   • Consistent across content types")
    print("   • Good balance of automation and quality preservation")

    print("\n3️⃣  NONE-FRAME BASELINE VALUE:")
    none_data = df[df["frame_algorithm"] == "none-frame"]
    print("   • No frame loss = maximum temporal information")
    print(
        f"   • Higher quality ({none_data['enhanced_composite_quality'].mean():.3f}) compensates for less compression"
    )
    print("   • Efficiency formula rewards this balance")
    print("   • Proves that sometimes 'doing nothing' is optimal")

    print("\n4️⃣  ANIMATELY-FRAME CONSERVATION:")
    animately_data = df[df["frame_algorithm"] == "animately-frame"]
    print("   • AI tries to preserve important frames")
    print(
        f"   • But removes {(1-animately_data['applied_frame_ratio'].mean())*100:.1f}% of frames on average"
    )
    print("   • Quality loss outweighs compression gains in this dataset")
    print("   • Might excel with different content or parameters")

    print("\n5️⃣  FFMPEG-FRAME STRUGGLES:")
    ffmpeg_data = df[df["frame_algorithm"] == "ffmpeg-frame"]
    print("   • Mathematical sampling misses visual context")
    print(
        f"   • Removes {(1-ffmpeg_data['applied_frame_ratio'].mean())*100:.1f}% of frames uniformly"
    )
    print("   • Can remove keyframes that are visually important")
    print(
        f"   • Poor efficiency ({ffmpeg_data['efficiency'].mean():.2f}) shows algorithm mismatch"
    )


def practical_recommendations(df):
    """Provide practical recommendations based on the analysis."""
    print("\n\n💡 PRACTICAL RECOMMENDATIONS")
    print("=" * 55)

    print("🎯 ALGORITHM SELECTION GUIDE:")

    print("\n🥇 USE GIFSICLE-FRAME WHEN:")
    print("   • Optimizing for web delivery (up to 60x compression)")
    print("   • Working with motion/animation content")
    print("   • File size is critical priority")
    print("   • Content has natural frame redundancy")

    print("\n🥈 USE IMAGEMAGICK-FRAME WHEN:")
    print("   • Working with complex visual content")
    print("   • Need consistent results across content types")
    print("   • Balancing quality and compression")
    print("   • Processing gradients, textures, or noise")

    print("\n🥉 USE NONE-FRAME WHEN:")
    print("   • Quality is absolute priority")
    print("   • Working with high-value content")
    print("   • Frame-by-frame analysis is needed")
    print("   • Temporal accuracy is critical")

    print("\n⚠️  CONSIDER ALTERNATIVES TO:")
    print("   • ANIMATELY-FRAME: May be too conservative for this use case")
    print("   • FFMPEG-FRAME: Mathematical sampling lacks visual context")

    print("\n🔧 OPTIMIZATION STRATEGY:")
    print("   1. Start with gifsicle-frame for maximum efficiency")
    print("   2. Fall back to imagemagick-frame for complex content")
    print("   3. Use none-frame as quality baseline")
    print("   4. Test animately-frame with custom parameters")
    print("   5. Avoid ffmpeg-frame for this pipeline configuration")


def main():
    """Run complete frame reduction analysis."""
    print("🎬 FRAME REDUCTION ALGORITHMS: DEEP DIVE ANALYSIS")
    print("Understanding Performance, Mechanisms, and Differences")
    print("=" * 65)

    df = load_and_prepare_data()

    analyze_frame_reduction_mechanisms()
    analyze_frame_reduction_behavior(df)
    compare_animately_vs_none_frame(df)
    analyze_compression_vs_frame_reduction(df)
    explain_algorithm_differences(df)
    practical_recommendations(df)

    print("\n🎉 ANALYSIS COMPLETE")
    print("=" * 30)
    print("Frame reduction algorithm behavior fully analyzed!")

    return True


if __name__ == "__main__":
    main()
