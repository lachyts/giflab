# 🧪 Experimental Testing Framework

The experimental testing framework provides a structured way to test and compare different GIF compression workflows before running on large datasets. It's designed to help you understand the performance characteristics of different compression strategies and identify optimal settings.

## Overview

The experimental framework tests multiple compression strategies on a small set of diverse GIFs (~10 by default) to validate workflows, compare engines, and identify anomalies. This is essential for understanding what's worth testing before scaling to larger datasets.

## Quick Start

```bash
# Run experiment with default settings (10 GIFs, all strategies)
poetry run python -m giflab experiment

# Run experiment with custom settings
poetry run python -m giflab experiment --gifs 15 --strategies pure_gifsicle --strategies gifsicle_dithered

# Use your own sample GIFs
poetry run python -m giflab experiment --sample-gifs-dir data/my_samples --gifs 5
```

## Compression Strategy Model (Dynamic)

The experimental runner now generates strategies **dynamically** using the slot-based capability registry described in *next-tools-priority.md*.

• **Single-tool pipelines** – Each tool is tested in isolation for every slot it advertises.

• **Combined pipelines** – The runner automatically explores every valid combination of tools across Frame, Color and Lossy slots.  When two consecutive slots are handled by the same tool and that tool has `combines: true`, they are executed in a single pass without creating intermediate GIFs.

### Filtering Examples

Run only pipelines that use Gifsicle in any slot:

```bash
poetry run python -m giflab experiment --include-tool gifsicle
```

Run pipelines where Animately handles Color reduction (no matter what fills the other slots):

```bash
poetry run python -m giflab experiment --include color=animately
```

The legacy **pure_gifsicle**, **pure_animately**, and **animately_then_gifsicle** names still work as convenience aliases, but internally they resolve to the new dynamic model.

### Extended Gifsicle Options

The framework tests different gifsicle options to understand their impact:

#### Optimization Levels
- **basic**: `--optimize` (current default)
- **level1**: `-O1` (basic optimization)
- **level2**: `-O2` (better optimization)
- **level3**: `-O3` (best optimization)

#### Dithering Options
- **none**: `--no-dither` (sharp edges)
- **floyd**: `--dither` (Floyd-Steinberg dithering)
- **ordered**: `--dither=ordered` (ordered dithering)

## Test Dataset

The framework automatically generates 10 diverse sample GIFs covering different content types:

1. **simple_gradient**: Smooth color transitions
2. **complex_animation**: Moving shapes and objects
3. **text_content**: Text and UI elements
4. **photo_realistic**: Photo-like content with noise
5. **high_contrast**: High contrast patterns
6. **many_colors**: Full color spectrum
7. **small_frames**: Micro-animations (50x50px)
8. **large_frames**: Detailed content (300x200px)
9. **single_frame**: Static image
10. **rapid_motion**: Fast-moving elements

This ensures you're testing across different content types that may respond differently to compression.

## Command Line Options

```bash
poetry run python -m giflab experiment [OPTIONS]
```

### Options

- `--gifs, -g`: Number of test GIFs to generate (default: 10)
- `--workers, -j`: Number of worker processes (default: CPU count)
- `--sample-gifs-dir`: Use existing GIFs instead of generating new ones
- `--output-dir`: Directory to save results (default: data/experimental/results)
- `--strategies`: Strategies to test (default: all)
- `--no-analysis`: Disable detailed analysis report

### Strategy Selection

```bash
# Test only pure gifsicle strategies
poetry run python -m giflab experiment --strategies pure_gifsicle --strategies gifsicle_dithered

# Test workflow comparison
poetry run python -m giflab experiment --strategies pure_gifsicle --strategies animately_then_gifsicle

# Test all strategies (default)
poetry run python -m giflab experiment --strategies all
```

## Output Structure

The framework creates a timestamped results directory:

```
data/experimental/results/YYYYMMDD_HHMMSS/
├── results.csv          # Raw results data
├── analysis_report.json            # Comprehensive analysis
└── visualizations/                 # Charts and graphs
    ├── strategy_comparison.png
    ├── compression_quality_scatter.png
    └── distribution_analysis.png
```

## Results Analysis

The framework provides comprehensive analysis tools to understand the results:

### Automatic Analysis

Every experiment run generates:
- **Strategy comparison**: Performance metrics for each strategy
- **Anomaly detection**: Outliers and suspicious patterns
- **Recommendations**: Actionable insights for optimization
- **Visualizations**: Charts showing performance relationships

### Manual Analysis

You can also analyze results programmatically:

```python
from giflab.analysis import ExperimentalAnalyzer

# Load results
analyzer = ExperimentalAnalyzer(Path("data/experimental/results/20240101_120000/results.csv"))

# Compare strategies
comparisons = analyzer.compare_strategies()
for comp in comparisons:
    print(f"{comp.strategy_name}: {comp.avg_compression_ratio:.2f}x compression, {comp.avg_ssim:.3f} SSIM")

# Detect anomalies
anomalies = analyzer.detect_anomalies()
print(f"Found {len(anomalies.outliers)} outliers")

# Get recommendations
recommendations = analyzer.get_recommendations()
for rec in recommendations:
    print(f"• {rec}")
```

## Key Metrics

The framework measures:

### Performance Metrics
- **Compression ratio**: Original size / compressed size
- **File size reduction**: Percentage reduction in file size
- **Processing time**: Time taken for compression (ms)
- **Success rate**: Percentage of successful compressions

### Quality Metrics
- **SSIM**: Structural similarity index
- **Composite quality**: Combined quality score
- **MSE, RMSE**: Pixel-level error metrics
- **FSIM, GMSD**: Feature-based similarity metrics
- **Additional metrics**: Edge similarity, texture similarity, etc.

## Workflow Comparison

To specifically answer your question about comparing workflows:

### 1. Single-tool vs Combined-tool Comparison

```bash
# Compare a single-tool pipeline against a combined pipeline generated dynamically
poetry run python -m giflab experiment \
  --include-tool gifsicle --slots frame,color,lossy \
  --include frame=animately --include lossy=gifsicle
```

This will help you understand:
- **Compression efficiency**: Which workflow achieves better compression ratios
- **Quality preservation**: Which maintains better visual quality
- **Processing speed**: Which is faster to execute
- **Reliability**: Which has fewer failures

### 2. Dithering vs Non-dithering

```bash
# Test dithering options
poetry run python -m giflab experiment --strategies pure_gifsicle --strategies gifsicle_dithered
```

This will show:
- **Quality differences**: How dithering affects visual quality
- **File size impact**: How dithering affects compression
- **Content type sensitivity**: Which content types benefit from dithering

### 3. Optimization Level Impact

The framework tests different optimization levels to understand:
- **Time vs quality trade-offs**: How much extra time `-O3` takes
- **Compression improvements**: Whether higher optimization levels provide better compression
- **Diminishing returns**: When optimization stops providing benefits

## Anomaly Detection

The framework automatically detects:

### Outliers
- **Compression ratio outliers**: Unusually high/low compression ratios
- **Quality score outliers**: Unexpectedly high/low quality scores
- **Processing time outliers**: Unusually slow/fast processing

### Suspicious Patterns
- **High failure rates**: Strategies with low success rates
- **Inconsistent performance**: Strategies with high variance
- **Engine availability issues**: Missing or misconfigured engines

### Recommendations
- **Optimal strategy selection**: Which strategies work best overall
- **Parameter tuning**: Suggested parameter adjustments
- **Troubleshooting**: How to fix detected issues

## Best Practices

### 1. Start Small
- Use the default 10 GIFs for initial testing
- Focus on key strategies first
- Analyze results before scaling up

### 2. Use Your Own GIFs
```bash
# Test with your actual content
poetry run python -m giflab experiment --sample-gifs-dir data/my_test_gifs
```

### 3. Iterative Testing
- Test one workflow change at a time
- Compare results between runs
- Document findings for future reference

### 4. Consider Your Content
- Different content types may favor different strategies
- Test with representative samples of your actual data
- Pay attention to content-specific recommendations

## Troubleshooting

### Common Issues

1. **Engine not found**: Ensure gifsicle and animately are properly installed
2. **Permission errors**: Check file permissions in output directories
3. **Memory issues**: Reduce number of workers or GIF count
4. **Slow processing**: Use fewer strategies or smaller GIFs for initial testing

### Debug Mode

```bash
# Run with single worker for easier debugging
poetry run python -m giflab experiment --workers 1 --gifs 3
```

## Integration with Main Pipeline

After experimental testing, you can:

1. **Update configuration**: Modify compression settings based on results
2. **Select optimal strategy**: Choose the best-performing workflow
3. **Scale up**: Apply findings to larger datasets using the main pipeline

```bash
# After experiments show gifsicle_optimized is best:
# Update src/giflab/config.py to use -O3 optimization
# Then run on full dataset:
poetry run python -m giflab run data/raw data/
```

## Examples

### Example 1: Validate Current Workflow
```bash
# Test current implementation
poetry run python -m giflab experiment --strategies pure_gifsicle --gifs 10
```

### Example 2: Compare Workflows
```bash
# Compare pure gifsicle vs hybrid approach
poetry run python -m giflab experiment --strategies pure_gifsicle --strategies animately_then_gifsicle
```

### Example 3: Test Dithering Impact
```bash
# Test dithering options
poetry run python -m giflab experiment --strategies pure_gifsicle --strategies gifsicle_dithered --strategies gifsicle_ordered_dither
```

### Example 4: Use Your Own GIFs
```bash
# Test with your specific content
mkdir -p data/my_samples
cp /path/to/your/gifs/*.gif data/my_samples/
poetry run python -m giflab experiment --sample-gifs-dir data/my_samples
```

## Results Interpretation

### What to Look For

1. **Best overall strategy**: Highest combined score of compression ratio, quality, and success rate
2. **Content-specific performance**: Which strategies work best for which content types
3. **Anomalies**: Unexpected results that might indicate issues
4. **Trade-offs**: Balance between compression, quality, and processing time

### Making Decisions

- **High quality priority**: Choose strategies with highest SSIM/composite quality
- **High compression priority**: Choose strategies with highest compression ratios
- **Fast processing priority**: Choose strategies with lowest processing times
- **Reliability priority**: Choose strategies with highest success rates

The experimental framework gives you the data to make informed decisions about your GIF compression workflow before committing to processing large datasets. 