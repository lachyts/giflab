# 🎞️ GifLab

GIF compression and analysis laboratory for systematic performance evaluation.

## Overview

GifLab analyzes GIF compression by generating a grid of variants with different:

- **Frame keep ratios**: 1.00, 0.90, 0.80, 0.70, 0.50
- **Palette sizes**: 256, 128, 64 colors  
- **Lossy levels**: 0, 40, 120
- **Engines**: gifsicle, animately

Each variant is measured for file size, SSIM quality, and render time.

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

## Quick Start

```bash
# Install dependencies (requires Poetry)
poetry install

# Run compression analysis
poetry run python -m giflab run data/raw data/

# Add AI-generated tags
poetry run python -m giflab tag results.csv data/raw
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
- gifsicle and animately-cli for compression
- FFmpeg for video processing

## Cross-Platform Setup

### macOS
```bash
brew install python@3.11 ffmpeg gifsicle
# Install animately-cli binary to PATH
```

**Engine Paths:**
- Animately engine: `/Users/lachlants/bin/launcher`
- Gifsicle: `gifsicle`

### Windows/WSL
```bash
choco install python ffmpeg gifsicle
# Or use WSL2 with Linux setup
```

## License

MIT License - see LICENSE file for details.