# 🎞️ GifLab

GIF compression and analysis laboratory for systematic performance evaluation.

## 👋 New to GifLab?

**🚀 Complete Beginners:** Start with our [**Beginner's Guide**](docs/guides/beginner.md) for step-by-step instructions and hand-holding through your first GIF analysis.

**🔧 Developers & Advanced Users:** Continue reading for technical details and quick reference.

---

## Overview

GifLab analyzes GIF compression by generating a grid of variants with different:

- **Frame keep ratios**: 1.00, 0.90, 0.80, 0.70, 0.50
- **Palette sizes**: 256, 128, 64 colors  
- **Lossy levels**: 0%, 60%, 100% (universal percentages, mapped to engine-specific ranges)
- **Production Engines**: **gifsicle**, **Animately** (main pipeline)
- **Experimental Engines**: **ImageMagick**, **FFmpeg**, **gifski** (experimental pipeline)

Each variant is measured for file size, comprehensive quality metrics, render time, and **efficiency score** (balanced 50/50 weighting of quality and compression).

### Dual Pipeline Architecture

GifLab uses a **two-pipeline approach** to balance stability and innovation:

#### 🏭 **Production Pipeline** (`run` command)
- **Engines**: gifsicle, Animately (proven, reliable)
- **Purpose**: Large-scale processing, production workflows
- **Usage**: `python -m giflab run data/raw`

#### 🧪 **Experimental Pipeline** (`experiment` command)  
- **Engines**: All 5 engines (ImageMagick, FFmpeg, gifski, gifsicle, Animately)
- **Purpose**: Comprehensive testing, pipeline elimination, finding optimal combinations
- **Traditional Usage**: `python -m giflab experiment --sampling representative`
- **🆕 Targeted Presets**: `python -m giflab experiment --preset frame-focus` (**93-99% more efficient!**)

| Engine | Pipeline | Color | Frame | Lossy | Best For |
|--------|----------|-------|-------|--------|----------|
| **gifsicle** | Both | ✅ | ✅ | ✅ | Fast, lightweight, widely compatible |
| **Animately** | Both | ✅ | ✅ | ✅ | Complex gradients, photo-realistic content |
| **ImageMagick** | Experimental | ✅ | ✅ | ✅ | General-purpose, wide format support |
| **FFmpeg** | Experimental | ✅ | ✅ | ✅ | High-quality video-based processing |
| **gifski** | Experimental | ❌ | ❌ | ✅ | Highest quality lossy compression |

## 🎯 **NEW: Targeted Experiment Presets**

**Transform your experiment workflow with 93-99% efficiency gains!**

Instead of generating all 935 possible pipeline combinations and sampling from them, targeted presets create only the specific combinations you need for focused research studies.

### Quick Start with Presets

```bash
# List all available presets
python -m giflab experiment --list-presets

# Compare all frame reduction algorithms (5 pipelines vs 935)
python -m giflab experiment --preset frame-focus

# Compare color quantization methods (17 pipelines vs 935)  
python -m giflab experiment --preset color-optimization

# Quick testing preset (2 pipelines)
python -m giflab experiment --preset quick-test
```

### Efficiency Comparison

| Approach | Pipelines Generated | Efficiency |
|----------|-------------------|------------|
| Traditional (generate all + sample) | 935 → 46 used | 95% waste |
| **🎯 Targeted: frame-focus** | **5 generated** | **99.5% efficient** |
| **🎯 Targeted: color-optimization** | **17 generated** | **98.2% efficient** |

### Available Research Presets

- **`frame-focus`**: Compare frame reduction algorithms (5 pipelines)
- **`color-optimization`**: Compare color quantization methods (17 pipelines)  
- **`lossy-quality-sweep`**: Evaluate lossy compression effectiveness (11 pipelines)
- **`tool-comparison-baseline`**: Fair engine comparison (64 pipelines)
- **`dithering-focus`**: Compare dithering algorithms (6 pipelines)
- **`png-optimization`**: Optimize PNG sequence workflows (4 pipelines)
- **`quick-test`**: Fast development testing (2 pipelines)

📚 **Learn more:** [Targeted Presets Quick Start Guide](docs/quickstart/targeted-presets-quickstart.md)

### Dual-Mode Workflow

GifLab is intentionally split into **two complementary modes**:

1. **Experimental Mode** – Generate a *small* synthetic or curated GIF set and exhaustively test *all* valid tool-pipelines (produced by the dynamic matrix generator).  This surfaces the most promising combinations **before** touching large datasets.
2. **Production Mode** – Run the *selected* top pipelines on full datasets (thousands of GIFs) to produce final compressed outputs and metric CSVs.

Workflow summary:

| Step | Command | Purpose |
|------|---------|---------|
| 1. Explore | `giflab experiment --sampling representative` | Tests pipeline combinations on synthetic GIFs, eliminates underperformers, writes results to `results/experiments/`. |
| 2. Analyse | Use notebooks / `giflab select-pipelines` | Analyze results and select top performing pipelines for production. |
| 3. Run | `giflab run data/raw/` | Executes chosen pipelines at scale, writing results and renders. |

The **Experiment → Analyse → Run** loop keeps production runs fast and data-driven.

> ⚠️ **Job-count warning**: With dynamic matrix mode enabled (default from S6
> onward) GifLab will test *every* combination of frame-ratio × palette size ×
> lossy level × engine optimisation flag × tool chain.  The default settings
> easily create several thousand pipeline runs for the 10 sample GIFs, which can
> take minutes on a modern laptop.  If you only want a subset, trim the
> `ExperimentalConfig` lists (e.g. `LOSSY_LEVELS`, `COLOR_KEEP_COUNTS`) or run
> with fewer sample GIFs.

## 🤖 ML-Driven Optimization Strategy

**GifLab's Vision**: Use machine learning to automatically select the optimal compression tool combination based on GIF characteristics.

### The Problem
Different GIF compression tools excel at different types of content:
- **Gifsicle**: Better for simple graphics, text, high-contrast content
- **Animately**: Superior for complex gradients, many colors, photo-realistic content
- **ImageMagick**: Versatile general-purpose tool with extensive format support
- **FFmpeg**: High-quality video-based processing with advanced filters
- **gifski**: Highest quality lossy compression using advanced algorithms
- **Hybrid approaches**: Can combine strengths of multiple tools

### The Solution
GifLab's experimental framework tests multiple algorithmic combinations to:

1. **Build a dataset** of GIF characteristics (colors, frames, complexity, content type) paired with optimal tool combinations
2. **Train ML models** to predict the best compression strategy for new GIFs
3. **Create intelligent routing** that automatically selects optimal tool chains
4. **Continuously improve** through feedback and expanded testing

### Research Approach
- **Comprehensive tool testing**: Expand beyond gifsicle/animately to include ImageMagick, FFmpeg, gifski, and other tools
- **Content classification**: Develop automated content type detection (text, photo, animation, etc.)
- **Performance prediction**: Train models to predict compression ratio and quality trade-offs
- **Algorithmic innovation**: Test novel tool combinations and parameter settings

*See [ML Strategy Documentation](docs/technical/ml-strategy.md) for detailed implementation plans.*

### Integration with Main Pipeline
After experimental validation, apply findings to your main compression pipeline:
```bash
# 1. Run experiments to identify optimal strategy
poetry run python -m giflab experiment --sampling representative

# 2. Select top performing pipelines
poetry run python -m giflab select-pipelines results/experiments/latest/enhanced_streaming_results.csv --top 3 -o winners.yaml

# 3. Run full pipeline with optimized settings
poetry run python -m giflab run data/raw --pipelines winners.yaml
```

📖 **For detailed documentation, see:** [Experimental Testing Guide](docs/guides/experimental-testing.md)

## 🗂️ Directory-Based Source Detection

GifLab automatically detects GIF sources based on directory structure, making it easy to organize and analyze GIFs from different platforms:

### Directory Structure
```
data/raw/
├── tenor/              # GIFs from Tenor platform
│   ├── love/           # "love" search results
│   ├── marketing/      # "marketing" search results
│   └── email_campaign/ # Email campaign GIFs
├── animately/          # GIFs from Animately platform (all user uploads)
│   ├── user_upload_1.gif
│   ├── user_upload_2.gif
│   └── user_upload_3.gif
├── tgif_dataset/       # GIFs from TGIF research dataset
│   ├── research_gif_1.gif
│   ├── research_gif_2.gif
│   └── research_gif_3.gif
└── unknown/            # Ungrouped GIFs
```

### Platform Naming Convention

| Platform | Directory Name | Type | Notes |
|----------|----------------|------|-------|
| Tenor | `tenor/` | Live Platform | Google's GIF search platform |
| Animately | `animately/` | Live Platform | Your compression platform |
| TGIF Dataset | `tgif_dataset/` | Research Dataset | Academic research dataset |
| Unknown | `unknown/` | Fallback | Unclassified or mixed sources |

**Why "tgif_dataset"?** The "_dataset" suffix distinguishes research datasets from live platforms, making the data source clear. Both `tgif/` and `tgif_dataset/` are supported.

**Why flat structure for some platforms?** Animately and TGIF files have uniform characteristics within each platform, so subdirectories don't add meaningful organization. Tenor search queries, however, create content differences worth preserving in the directory structure.

### Quick Start
```bash
# 1. Create directory structure
python -m giflab organize-directories data/raw/

# 2. Move GIFs to appropriate directories
# (manually or via collection scripts)

# 3. Run analysis with automatic source detection
python -m giflab run data/raw/

# 4. Optional: Disable source detection
python -m giflab run data/raw/ --no-detect-source-from-directory
```

### CSV Output
The resulting CSV includes source tracking columns:
- `source_platform`: Platform identifier (tenor, animately, tgif_dataset, unknown)
- `source_metadata`: JSON metadata with query, context, and collection details

📖 **For detailed documentation, see:** [Directory-Based Source Detection Guide](docs/guides/directory-source-detection.md)

## SSIM Quality Analysis

GifLab provides multiple SSIM (Structural Similarity Index) calculation modes optimized for different use cases:

### Performance Comparison

| Mode | Frames Sampled | Processing Time | Use Case |
|------|----------------|-----------------|----------|
| **Fast** | 3 keyframes | ~5ms | Massive datasets (10,000+ GIFs) |
| **Optimized (10)** | 10 frames | ~6ms | Production pipeline (1,000-10,000 GIFs) |
| **Optimized (20)** | 20 frames | ~8ms | Balanced accuracy/speed |
| **Full** | All frames | ~12ms | Research & analysis (small datasets) |

### Sampling Strategies

- **Uniform**: Evenly distributed frames across the GIF
- **Keyframe**: Strategic positions (start, 1/4, 1/2, 3/4, end) + uniform fill
- **Adaptive**: Content-aware sampling (future enhancement)

### Configuration Examples

```python
from giflab.config import MetricsConfig

# For large datasets (speed priority)
fast_config = MetricsConfig(
    SSIM_MODE="fast",                    # 3 frames only
    CALCULATE_FRAME_METRICS=False        # Skip detailed metrics
)

# For production pipeline (balanced)
production_config = MetricsConfig(
    SSIM_MODE="optimized",               # Sampled frames
    SSIM_MAX_FRAMES=15,                  # Good balance
    SSIM_SAMPLING_STRATEGY="uniform",    # Representative sampling
    CALCULATE_FRAME_METRICS=True         # Keep detailed metrics
)

# For research (accuracy priority)
research_config = MetricsConfig(
    SSIM_MODE="full",                    # All frames
    CALCULATE_FRAME_METRICS=True         # Full analysis
)
```

### Dataset Processing Estimates

| Dataset Size | Avg Frames | Fast Mode | Optimized Mode | Full Mode |
|-------------|------------|-----------|----------------|-----------|
| 1,000 GIFs  | 50         | **0.03h** | 0.15h         | 0.8h      |
| 5,000 GIFs  | 100        | **0.3h**  | 1.5h          | 8.3h      |
| 10,000 GIFs | 100        | **0.6h**  | 3.0h          | 16.7h     |

*Processing times include 24 variants per GIF (frame ratios × color counts × lossy levels × engines)*

## Expanded Quality Metrics (S10)

In addition to SSIM-based measures, GifLab now computes **eight complementary frame-level metrics** that capture color, texture, edge and gradient fidelity:

| Metric | Abbrev. | Higher-is-better? | Notes |
|--------|---------|-------------------|-------|
| Mean Squared Error | MSE | ❌ (lower) | Pixel-wise error |
| Root Mean Squared Error | RMSE | ❌ (lower) | Square-root of MSE |
| Feature Similarity Index | FSIM | ✅ | Gradient & phase features |
| Gradient Magnitude Similarity Deviation | GMSD | ❌ (lower) | Gradient-map deviation |
| Color-Histogram Correlation | CHIST | ✅ | Channel-wise histogram correlation |
| Edge-Map Jaccard Similarity | EDGE | ✅ | Edge overlap similarity |
| Texture-Histogram Correlation | TEXTURE | ✅ | LBP histogram correlation |
| Sharpness Similarity | SHARP | ✅ | Laplacian variance ratio |

All metrics are exposed via `giflab.metrics.calculate_comprehensive_metrics` and exported to CSV with mean/std/min/max descriptors (plus optional raw values).

## 🎯 Efficiency Scoring System

GifLab calculates an **efficiency score** (0-1 scale) that balances quality preservation with compression performance using equal 50/50 weighting:

```python
efficiency = (composite_quality^0.5) × (normalized_compression^0.5)
```

### Key Features:
- **Equal weighting**: 50% quality, 50% compression - neither dominates
- **Log-normalized compression**: Handles diminishing returns above 20x compression
- **Geometric mean**: Requires both good quality AND good compression for high scores
- **Bounded output**: Easy-to-interpret 0-1 scale

### Efficiency Scale:
| Range | Rating | Example Algorithms |
|--------|---------|-------------------|  
| 0.80-1.00 | **EXCELLENT** | imagemagick-frame (0.855) |
| 0.70-0.79 | **VERY GOOD** | gifsicle-frame (0.767) |
| 0.60-0.69 | **GOOD** | none-frame (0.642) |
| 0.50-0.59 | **FAIR** | ffmpeg-frame (0.569) |
| 0.00-0.49 | **POOR** | Investigate issues |

📖 **For technical details, see:** [Efficiency Calculation Guide](docs/technical/efficiency-calculation.md)

## 🎯 Pareto Frontier Analysis

GifLab uses **Pareto frontier analysis** to solve the fundamental challenge of comparing compression pipelines fairly when quality scores don't align. This mathematical approach identifies the optimal trade-offs between quality and file size without requiring subjective weightings.

### The Problem: Comparing Unlike Pipelines

Traditional pipeline comparison fails when pipelines achieve different quality levels:
- **Pipeline A**: 0.85 quality, 200KB  
- **Pipeline B**: 0.90 quality, 180KB
- **Pipeline C**: 0.88 quality, 250KB

*Which is better?* Simple rankings can't answer this fairly.

### The Solution: Mathematical Optimality

**Pareto frontier analysis** identifies pipelines that are **mathematically optimal** - where you cannot improve one metric (quality) without worsening another (file size).

#### Pareto Optimal Examples:
- **Pipeline B**: 0.90 quality, 180KB → **Optimal** (dominates A in both metrics)
- **Pipeline A**: 0.85 quality, 200KB → **Optimal** (best quality for larger sizes)  
- **Pipeline C**: 0.88 quality, 250KB → **Dominated** (B is better in both quality AND size)

### Pipeline Elimination Benefits

1. **Eliminates Subjectivity**: No need to choose quality-vs-size weights
2. **Mathematically Rigorous**: Clear, defensible elimination decisions  
3. **Content-Type Aware**: Different frontiers for different GIF types
4. **Multi-Metric Support**: Works with SSIM, MS-SSIM, composite quality scores

### Experimental Commands

```bash
# Run comprehensive experiments with Pareto analysis
poetry run python -m giflab experiment --sampling representative

# View experiment results and top performers
poetry run python -m giflab select-pipelines results/experiments/latest/enhanced_streaming_results.csv --top 5

# Use quick sampling for faster testing during development
poetry run python -m giflab experiment --sampling quick
```

### Interpretation Guide

**Pareto Frontier Points:**
- **Leftmost point**: "Best quality achievable (any file size)"
- **Rightmost point**: "Best compression achievable (acceptable quality)"  
- **Middle points**: "Optimal quality-size balance points"

**Dominated Pipelines:** 
- These are **never optimal** - always a better choice available
- Safe to eliminate from production considerations
- Identified automatically by elimination tests

### Quality-Aligned Efficiency Rankings

For specific quality targets, Pareto analysis provides definitive rankings:

```python
# At quality 0.85, which pipeline wins?
quality_85_winners = [
    ('animately_lossy40_128colors', 145KB),  # Winner
    ('gifsicle_O2_256colors', 167KB),        # Second  
    ('ffmpeg_floyd_steinberg', 189KB)        # Third
]
```

This answers your key question: **"Which pipeline provides better file size for the same quality?"**

📖 **For technical details, see:** [Pipeline Elimination Guide](docs/guides/elimination-results-tracking.md)

## Quick Start

```bash
# Install dependencies (requires Poetry)
poetry install

# Test all compression strategies with intelligent sampling
poetry run python -m giflab experiment --sampling representative

# Pick top 3 pipelines by SSIM
poetry run python -m giflab select-pipelines results/experiments/latest/enhanced_streaming_results.csv --top 3 -o winners.yaml

# Run production compression on full dataset with chosen pipelines
poetry run python -m giflab run data/raw --pipelines winners.yaml

# Add AI-generated tags
poetry run python -m giflab tag results.csv data/raw
```

## Pipeline Usage Examples

### 🏭 Production Pipeline (Reliable, Fast)
```bash
# Standard production processing with gifsicle + Animately
poetry run python -m giflab run data/raw

# Production with custom settings
poetry run python -m giflab run data/raw --workers 8 --resume

# Production with pipeline YAML (advanced)
poetry run python -m giflab run data/raw --pipelines custom_pipelines.yaml
```

### 🧪 Experimental Pipeline (All 5 Engines)
```bash
# Test all 5 engines with comprehensive sampling
poetry run python -m giflab experiment --sampling representative

# Quick experimental test for development
poetry run python -m giflab experiment --sampling quick

# Full comprehensive testing (slower but thorough)
poetry run python -m giflab experiment --sampling full

# Targeted testing with strategic synthetic GIFs
poetry run python -m giflab experiment --sampling targeted
```

### 🔬 Engine-Specific Testing
To test specific engines, use the experimental pipeline which automatically includes:
- **ImageMagick**: General-purpose processing
- **FFmpeg**: High-quality video-based compression  
- **gifski**: Premium lossy compression quality
- **gifsicle**: Fast, reliable baseline
- **Animately**: Complex gradient optimization

### 📊 Workflow Recommendations

**For Production Work:**
```bash
# Large datasets, proven reliability
poetry run python -m giflab run data/raw --workers 8 --resume
```

**For Engine Comparison:**
```bash  
# Compare all engines to find the best for your content
poetry run python -m giflab experiment --sampling representative
poetry run python -m giflab select-pipelines results/experiments/latest/enhanced_streaming_results.csv --top 3
```

**For Research & Development:**
```bash
# Full experimental testing with comprehensive analysis
poetry run python -m giflab experiment --sampling full --use-gpu
```

## Project Structure

```
giflab/
├─ data/                 # Data directories
├─ src/giflab/           # Python package
├─ notebooks/            # Analysis notebooks  
├─ tests/                # Test suite
└─ pyproject.toml        # Poetry configuration
```

## Development Status

This project is being developed in stages:

- **✅ S0**: Repo scaffold, Poetry, black/ruff, pytest
- **✅ S1**: Metadata extraction + SHA + file-name; tests
- **✅ S2**: Lossy compression functionality
- **✅ S3**: Frame reduction functionality  
- **✅ S4**: Color palette reduction functionality
- **✅ S5**: Quality metrics and SSIM analysis
- **⏳ S6-S10**: Additional functionality (see PROJECT_SCOPE.md)

## Requirements

- Python 3.11+
- Poetry for dependency management
- FFmpeg for video processing

## Cross-Platform Setup

Engine paths are configurable via environment variables or `src/giflab/config.py` under the `EngineConfig` class.

### Environment Variables (Recommended)
Set these environment variables to override default engine paths:
```bash
export GIFLAB_GIFSICLE_PATH=/usr/local/bin/gifsicle
export GIFLAB_ANIMATELY_PATH=/usr/local/bin/animately
export GIFLAB_IMAGEMAGICK_PATH=/usr/local/bin/magick
export GIFLAB_FFMPEG_PATH=/usr/local/bin/ffmpeg
export GIFLAB_FFPROBE_PATH=/usr/local/bin/ffprobe
export GIFLAB_GIFSKI_PATH=/usr/local/bin/gifski
```

### Tool Installation by Platform

#### macOS
```bash
# Install tools via Homebrew
brew install python@3.11 ffmpeg gifsicle imagemagick

# Install gifski for high-quality lossy compression
brew install gifski

# Animately binary included in repository (bin/darwin/arm64/animately)
# Automatically detected - no additional installation required
```

#### Linux/Ubuntu
```bash
# Install via package manager
sudo apt update
sudo apt install python3.11 ffmpeg gifsicle imagemagick-6.q16

# Install gifski via cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install gifski

# Animately: Download from releases and place in bin/linux/x86_64/
# See bin/linux/x86_64/PLACEHOLDER.md for instructions
```

#### Windows
```bash
# Using Chocolatey
choco install python ffmpeg gifsicle imagemagick

# Install gifski via cargo or download binary
winget install gifski

# Animately: Download from releases and place in bin/windows/x86_64/
# See bin/windows/x86_64/PLACEHOLDER.md for instructions
```

#### Repository Binaries
GifLab includes platform-specific binaries for tools not available via package managers:
- **Animately**: Pre-built binaries in `bin/<platform>/<architecture>/`
- **Automatic detection**: Tool discovery checks repository binaries before system PATH
- **Cross-platform support**: macOS ARM64, Linux x86_64, Windows x86_64

### Verification
```bash
# Test all engines are properly configured
poetry run python -c "from giflab.system_tools import get_available_tools; print(get_available_tools())"

# Run smoke tests to verify functionality
poetry run python -m pytest tests/test_engine_smoke.py -v
```

## 📈 Machine-Learning Dataset Best Practices  
*Why this matters: High-quality, well-documented metrics are the foundation of reliable ML models.*

When you create or extend a GifLab dataset for ML tasks, **follow the checklist below** (detailed rationale in *Section&nbsp;8* of `QUALITY_METRICS_EXPANSION_PLAN.md`). These rules apply to *every* future contribution:

1. **Deterministic extraction** – lock random seeds; metric functions must be pure.
2. **Schema validation** – export rows that pass `MetricRecordV1` (pydantic) validation.
3. **Version tagging** – record dataset version, `giflab` semver, and git commit hash.
4. **Canonical data-splits** – maintain *GIF-level* `train/val/test` JSON split files.
5. **Feature scaling** – persist `scaler.pkl` (z-score or min-max) alongside data.
6. **Missing-value handling** – encode unknown metrics as `np.nan`, not `0.0`.
7. **Outlier & drift reports** – auto-generate an HTML outlier summary and correlation dashboard.
8. **Reproducible pipeline** – provide a `make data` target that builds the dataset + EDA artifacts end-to-end.
9. **Comprehensive logs** – include parameter checksum and elapsed-time stats in every run.

> **🚦 Gate keeper:** Pull requests touching dataset code MUST tick all items or explain why they do not apply.

For implementation details, see `giflab/data_prep.py` (to be added) and the ML checklist in the plan document.

## 🧪 Testing & Development

GifLab maintains strict testing standards to ensure code quality and project cleanliness:

### Testing Guidelines
- **Unit/Integration Tests**: Use `tests/` directory with pytest
- **Manual Testing**: Use `test-workspace/` structure (never root directory!)
- **Debug Sessions**: Create organized sessions in `test-workspace/debug/`
- **Cleanup Protocols**: Regular cleanup prevents project pollution

### Quick Commands
```bash
# Development Testing (⚡ <30s, rapid iteration)
make test-fast

# Pre-commit Testing (🔄 <5min, comprehensive validation)  
make test-integration

# Release Testing (🔍 <30min, full coverage)
make test-full

# Create test workspace
make test-workspace

# Emergency cleanup of root directory mess
make clean-testing-mess

# Clean temporary files
make clean-temp
```

**📋 Full Guidelines**: See [Testing Best Practices](docs/guides/testing-best-practices.md) for comprehensive testing standards and protocols.

## 📚 Documentation

### Core
- **[Project Scope](SCOPE.md)** - Goals, requirements, and architecture overview

### User Guides
- **[Beginner's Guide](docs/guides/beginner.md)** - Step-by-step introduction for new users
- **[Setup Guide](docs/guides/setup.md)** - Installation and configuration instructions

### Technical Reference
- **[Metrics System](docs/technical/metrics-system.md)** - Comprehensive quality assessment framework
- **[EDA Framework](docs/technical/eda-framework.md)** - Data analysis and visualization tools
- **[ML Best Practices](docs/technical/ml-best-practices.md)** - Machine learning dataset preparation
- **[Testing Best Practices](docs/guides/testing-best-practices.md)** - Quality assurance and testing approaches
- **[Content Classification](docs/technical/content-classification.md)** - AI-powered content tagging system

### Research & Analysis
- **[Compression Research](docs/analysis/compression-research.md)** - Engine comparison and optimization strategies
- **[Implementation Lessons](docs/analysis/implementation-lessons.md)** - Development insights and best practices

## Common Commands & Troubleshooting

### 🗑️ Clear All Cached Data
If you need to reset all cached pipeline results (SQLite database):
```bash
poetry run python -m giflab experiment --clear-cache --estimate-time
```
This clears all cached test results from `results/cache/pipeline_results_cache.db`.

### 🔄 Force Fresh Results (Without Clearing Cache)
To run fresh tests while keeping cached data intact:
```bash
poetry run python -m giflab experiment --no-cache --sampling quick
```

### 📊 Check Cache Performance
Cache statistics are automatically shown in pipeline elimination output.

### 🔍 Debug Pipeline Failures
View detailed failure information:
```bash
poetry run python -m giflab debug-failures --summary
```

**📚 For more details:** See [Experimental Testing Guide](docs/guides/experimental-testing.md)

## License

MIT License – see LICENSE file for details.