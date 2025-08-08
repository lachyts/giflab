#!/usr/bin/env python3
"""
Comprehensive analysis of enhanced metrics vs legacy metrics with efficiency scoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_data():
    """Load the enhanced results data."""
    results_path = Path("test-workspace/frame-comparison-with-gifs/run_20250807_123641/enhanced_streaming_results.csv")
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    # Filter to successful results only
    return df[df['success'] == True].copy()

def analyze_metric_differences(df):
    """Analyze differences between legacy and enhanced composite quality."""
    print("=" * 60)
    print("LEGACY vs ENHANCED COMPOSITE QUALITY ANALYSIS")
    print("=" * 60)
    
    # Calculate differences
    df['quality_improvement'] = df['enhanced_composite_quality'] - df['composite_quality']
    
    print(f"\nDataset Overview:")
    print(f"• Total successful results: {len(df):,}")
    print(f"• Unique pipelines tested: {df['pipeline_id'].nunique()}")
    print(f"• Content types: {', '.join(df['content_type'].unique())}")
    
    print(f"\nLegacy Composite Quality Stats:")
    legacy_stats = df['composite_quality'].describe()
    print(f"• Range: {legacy_stats['min']:.3f} to {legacy_stats['max']:.3f}")
    print(f"• Mean: {legacy_stats['mean']:.3f} ± {legacy_stats['std']:.3f}")
    print(f"• Median: {legacy_stats['50%']:.3f}")
    
    print(f"\nEnhanced Composite Quality Stats:")
    enhanced_stats = df['enhanced_composite_quality'].describe()
    print(f"• Range: {enhanced_stats['min']:.3f} to {enhanced_stats['max']:.3f}")
    print(f"• Mean: {enhanced_stats['mean']:.3f} ± {enhanced_stats['std']:.3f}")
    print(f"• Median: {enhanced_stats['50%']:.3f}")
    
    print(f"\nQuality Improvement Analysis:")
    improvement_stats = df['quality_improvement'].describe()
    print(f"• Mean change: {improvement_stats['mean']:.3f}")
    print(f"• Standard deviation: {improvement_stats['std']:.3f}")
    print(f"• Biggest improvement: +{improvement_stats['max']:.3f}")
    print(f"• Biggest decrease: {improvement_stats['min']:.3f}")
    
    # Count direction of changes
    improved = (df['quality_improvement'] > 0.01).sum()
    degraded = (df['quality_improvement'] < -0.01).sum()
    similar = len(df) - improved - degraded
    
    print(f"\nChange Distribution:")
    print(f"• Improved (>+0.01): {improved} ({improved/len(df)*100:.1f}%)")
    print(f"• Similar (±0.01): {similar} ({similar/len(df)*100:.1f}%)")
    print(f"• Degraded (<-0.01): {degraded} ({degraded/len(df)*100:.1f}%)")

def analyze_efficiency_scores(df):
    """Analyze efficiency scores and their relationship to quality/compression."""
    print("\n" + "=" * 60)
    print("EFFICIENCY SCORE ANALYSIS")
    print("=" * 60)
    
    efficiency_stats = df['efficiency'].describe()
    print(f"\nEfficiency Score Statistics:")
    print(f"• Range: {efficiency_stats['min']:.3f} to {efficiency_stats['max']:.3f}")
    print(f"• Mean: {efficiency_stats['mean']:.3f} ± {efficiency_stats['std']:.3f}")
    print(f"• Median: {efficiency_stats['50%']:.3f}")
    
    # Categorize efficiency levels
    df['efficiency_category'] = pd.cut(df['efficiency'], 
                                     bins=[0, 1, 2.5, 5, 10, float('inf')],
                                     labels=['Low (0-1)', 'Moderate (1-2.5)', 'Good (2.5-5)', 'High (5-10)', 'Exceptional (10+)'])
    
    print(f"\nEfficiency Distribution:")
    efficiency_counts = df['efficiency_category'].value_counts().sort_index()
    for category, count in efficiency_counts.items():
        pct = count / len(df) * 100
        print(f"• {category}: {count} ({pct:.1f}%)")
    
    # Top performers by efficiency
    print(f"\nTop 10 Most Efficient Pipelines:")
    top_efficient = df.nlargest(10, 'efficiency')[['pipeline_id', 'enhanced_composite_quality', 'compression_ratio', 'efficiency', 'file_size_kb']]
    for idx, row in top_efficient.iterrows():
        print(f"• {row['pipeline_id'][:60]}")
        print(f"  Quality: {row['enhanced_composite_quality']:.3f}, Compression: {row['compression_ratio']:.1f}x, Efficiency: {row['efficiency']:.2f}, Size: {row['file_size_kb']:.1f}KB")

def analyze_by_pipeline_type(df):
    """Analyze performance by pipeline components."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BY PIPELINE TYPE")
    print("=" * 60)
    
    # Extract pipeline components
    df['frame_algorithm'] = df['pipeline_id'].str.split('__').str[0].str.split('_').str[0]
    df['color_algorithm'] = df['pipeline_id'].str.split('__').str[1].str.split('_').str[0]
    df['lossy_algorithm'] = df['pipeline_id'].str.split('__').str[2].str.split('_').str[0]
    
    print(f"\nFrame Reduction Algorithm Performance:")
    frame_analysis = df.groupby('frame_algorithm').agg({
        'enhanced_composite_quality': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'compression_ratio': ['mean', 'std'],
        'pipeline_id': 'count'
    }).round(3)
    
    frame_analysis.columns = ['Quality_Mean', 'Quality_Std', 'Efficiency_Mean', 'Efficiency_Std', 'Compression_Mean', 'Compression_Std', 'Count']
    print(frame_analysis.sort_values('Efficiency_Mean', ascending=False))
    
    print(f"\nColor Reduction Algorithm Performance:")
    color_analysis = df.groupby('color_algorithm').agg({
        'enhanced_composite_quality': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'compression_ratio': ['mean', 'std'],
        'pipeline_id': 'count'
    }).round(3)
    
    color_analysis.columns = ['Quality_Mean', 'Quality_Std', 'Efficiency_Mean', 'Efficiency_Std', 'Compression_Mean', 'Compression_Std', 'Count']
    print(color_analysis.sort_values('Efficiency_Mean', ascending=False))

def find_sweet_spot_pipelines(df):
    """Find pipelines that hit the sweet spot of quality vs compression."""
    print("\n" + "=" * 60)
    print("SWEET SPOT ANALYSIS")
    print("=" * 60)
    
    # Define quality and compression thresholds
    high_quality_threshold = df['enhanced_composite_quality'].quantile(0.75)  # Top 25%
    high_compression_threshold = df['compression_ratio'].quantile(0.75)  # Top 25%
    small_size_threshold = df['file_size_kb'].quantile(0.25)  # Bottom 25%
    
    print(f"Sweet Spot Criteria:")
    print(f"• High quality: ≥{high_quality_threshold:.3f} (top 25%)")
    print(f"• High compression: ≥{high_compression_threshold:.1f}x (top 25%)")
    print(f"• Small file size: ≤{small_size_threshold:.1f}KB (bottom 25%)")
    
    # Find sweet spot pipelines
    sweet_spot = df[
        (df['enhanced_composite_quality'] >= high_quality_threshold) &
        (df['compression_ratio'] >= high_compression_threshold) &
        (df['file_size_kb'] <= small_size_threshold)
    ]
    
    print(f"\nSweet Spot Results:")
    print(f"• Found {len(sweet_spot)} results meeting all criteria ({len(sweet_spot)/len(df)*100:.1f}%)")
    
    if len(sweet_spot) > 0:
        print(f"\nTop Sweet Spot Performers:")
        sweet_spot_sorted = sweet_spot.nlargest(5, 'efficiency')
        for idx, row in sweet_spot_sorted.iterrows():
            print(f"• {row['pipeline_id'][:60]}")
            print(f"  Quality: {row['enhanced_composite_quality']:.3f}, Compression: {row['compression_ratio']:.1f}x")
            print(f"  Efficiency: {row['efficiency']:.2f}, Size: {row['file_size_kb']:.1f}KB")
    
    # Alternative: High efficiency approach
    print(f"\nAlternative: Top 20 Most Efficient Pipelines:")
    top_efficiency = df.nlargest(20, 'efficiency')
    efficiency_summary = top_efficiency.agg({
        'enhanced_composite_quality': ['mean', 'min', 'max'],
        'compression_ratio': ['mean', 'min', 'max'],
        'file_size_kb': ['mean', 'min', 'max'],
        'efficiency': ['mean', 'min', 'max']
    }).round(3)
    print(efficiency_summary)

def compare_content_types(df):
    """Compare performance across different content types."""
    print("\n" + "=" * 60)
    print("CONTENT TYPE COMPARISON")
    print("=" * 60)
    
    content_analysis = df.groupby('content_type').agg({
        'enhanced_composite_quality': ['mean', 'std', 'min', 'max'],
        'efficiency': ['mean', 'std', 'min', 'max'],
        'compression_ratio': ['mean', 'std'],
        'file_size_kb': ['mean', 'std'],
        'pipeline_id': 'count'
    }).round(3)
    
    content_analysis.columns = [
        'Quality_Mean', 'Quality_Std', 'Quality_Min', 'Quality_Max',
        'Efficiency_Mean', 'Efficiency_Std', 'Efficiency_Min', 'Efficiency_Max',
        'Compression_Mean', 'Compression_Std',
        'Size_Mean', 'Size_Std',
        'Count'
    ]
    
    print("Performance by Content Type:")
    print(content_analysis.sort_values('Efficiency_Mean', ascending=False))
    
    # Best pipeline for each content type
    print(f"\nBest Pipeline by Content Type (by efficiency):")
    for content_type in df['content_type'].unique():
        content_data = df[df['content_type'] == content_type]
        best = content_data.loc[content_data['efficiency'].idxmax()]
        print(f"\n{content_type.upper()}:")
        print(f"• Pipeline: {best['pipeline_id']}")
        print(f"• Quality: {best['enhanced_composite_quality']:.3f}")
        print(f"• Efficiency: {best['efficiency']:.2f}")
        print(f"• Compression: {best['compression_ratio']:.1f}x")
        print(f"• File size: {best['file_size_kb']:.1f}KB")

def detailed_case_studies(df):
    """Show detailed case studies of interesting pipelines."""
    print("\n" + "=" * 60)
    print("DETAILED CASE STUDIES")
    print("=" * 60)
    
    # Case 1: Biggest quality improvement
    max_improvement = df.loc[df['quality_improvement'].idxmax()]
    print(f"CASE 1: Biggest Quality Improvement (+{max_improvement['quality_improvement']:.3f})")
    print(f"Pipeline: {max_improvement['pipeline_id']}")
    print(f"• Legacy quality: {max_improvement['composite_quality']:.3f}")
    print(f"• Enhanced quality: {max_improvement['enhanced_composite_quality']:.3f}")
    print(f"• Why improved: Enhanced system captures additional quality dimensions")
    print(f"• Efficiency: {max_improvement['efficiency']:.2f}")
    print(f"• Compression: {max_improvement['compression_ratio']:.1f}x")
    
    # Case 2: Highest efficiency
    max_efficiency = df.loc[df['efficiency'].idxmax()]
    print(f"\nCASE 2: Highest Efficiency Score ({max_efficiency['efficiency']:.2f})")
    print(f"Pipeline: {max_efficiency['pipeline_id']}")
    print(f"• Enhanced quality: {max_efficiency['enhanced_composite_quality']:.3f}")
    print(f"• Compression ratio: {max_efficiency['compression_ratio']:.1f}x")
    print(f"• File size: {max_efficiency['file_size_kb']:.1f}KB")
    print(f"• Why efficient: Excellent balance of high compression with good quality retention")
    
    # Case 3: Best balanced performer
    # Find pipeline with high scores in all dimensions
    df['balanced_score'] = (
        df['enhanced_composite_quality'] * 0.4 +
        (df['compression_ratio'] / df['compression_ratio'].max()) * 0.3 +
        (df['efficiency'] / df['efficiency'].max()) * 0.3
    )
    
    best_balanced = df.loc[df['balanced_score'].idxmax()]
    print(f"\nCASE 3: Most Balanced Performer (score: {best_balanced['balanced_score']:.3f})")
    print(f"Pipeline: {best_balanced['pipeline_id']}")
    print(f"• Enhanced quality: {best_balanced['enhanced_composite_quality']:.3f}")
    print(f"• Efficiency: {best_balanced['efficiency']:.2f}")
    print(f"• Compression: {best_balanced['compression_ratio']:.1f}x")
    print(f"• File size: {best_balanced['file_size_kb']:.1f}KB")
    print(f"• Why balanced: Strong performance across all key metrics")

def main():
    """Run comprehensive analysis."""
    print("Loading enhanced metrics data...")
    df = load_data()
    
    if df is None:
        return False
    
    print(f"Loaded {len(df)} successful results for analysis")
    
    # Add quality improvement column
    df['quality_improvement'] = df['enhanced_composite_quality'] - df['composite_quality']
    
    # Run all analyses
    analyze_metric_differences(df)
    analyze_efficiency_scores(df)
    analyze_by_pipeline_type(df)
    find_sweet_spot_pipelines(df)
    compare_content_types(df)
    detailed_case_studies(df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("• Enhanced metrics provide more nuanced quality assessment")
    print("• Efficiency scores effectively balance quality and compression")
    print("• Different pipeline components excel in different scenarios")
    print("• Content type significantly affects optimal pipeline choice")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)