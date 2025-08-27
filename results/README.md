# Results Directory

This directory contains all analysis results, samples, and cached data in a clean, organized structure.

## 📁 Directory Structure

```
results/
├── runs/                      # Active analysis runs
│   ├── 001-frame-focus-22-08-25/
│   ├── 002-quick-test-22-08-25/
│   ├── latest -> 004-custom-experiment-22-08-25/
│   └── pipeline_results_cache.db
├── samples/                   # Test GIF samples (centralized)
│   ├── synthetic/            # Generated test content
│   └── real-world/           # Actual user content
├── archive/                   # Historical data (read-only)
│   └── experiments/          # Pre-consolidation experiment runs (171 runs, 116MB)
└── cache/                     # Additional shared cache files
    └── performance/
```

## 🎯 Analysis Organization

### Naming Convention
Analysis runs are organized with sequential numbering and descriptive names:
```
runs/{NNN-description-DD-MM-YY}/
├── streaming_results.csv      # Main results with 54+ fields
├── run_metadata.json         # Experiment configuration
├── elimination_progress.json # Progress tracking
└── visual_outputs/           # Generated visualizations
```

### Current Active Runs
Analysis runs are executed using the `giflab run` command and are saved in `results/runs/`:
- **Presets**: `--preset frame-focus`, `--preset quick-test`, etc.
- **Sampling**: `--sampling representative`, `--sampling quick`, etc.
- **Output**: Comprehensive 54+ field CSV with all engines and metrics

## 🔗 System Features

- **Latest symlink**: `results/runs/latest` points to most recent run
- **Cache system**: Database preserves results for efficiency
- **Tool integration**: Comprehensive analysis toolset:
  ```bash
  giflab view-failures results/runs/latest/
  giflab select-pipelines results/runs/latest/streaming_results.csv --top 3
  ```

## 🧹 Usage Guidelines

1. **Running Analysis**: Use `giflab run` - saves automatically to `results/runs/`
2. **Historical Data**: Access previous runs in `results/archive/experiments/` 
3. **Samples**: Test GIFs available in `results/samples/synthetic/`
4. **Cache**: Shared cache improves performance across runs

## 📊 Key Capabilities
- **All 5 engines**: gifsicle, Animately, ImageMagick, FFmpeg, gifski
- **54+ metrics**: Enhanced CSV output combining all previous pipeline fields
- **Targeted presets**: 14+ research presets for efficient analysis
- **Intelligent sampling**: representative, quick, full, targeted strategies
- **GPU acceleration**: Optional CUDA-accelerated quality metrics
- **Resume functionality**: Progress tracking and resume capability
- **Caching system**: Results cache preserves computation across runs

**Access the latest data:**
```bash
cd results/runs/latest/
# Main results: streaming_results.csv (54+ fields)
# Metadata: run_metadata.json
# Progress: elimination_progress.json
```

---
*The results directory provides a structured environment for GIF analysis and optimization, supporting efficient workflows through comprehensive tooling and caching.*