# Unified Results Directory

This directory contains all analysis results, samples, and cached data in a clean, organized structure following the pipeline consolidation.

## 📁 Directory Structure

```
results/
├── runs/                      # Active analysis runs (unified pipeline)
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
All new analysis runs use the unified `giflab run` command and are saved in `results/runs/`:
- **Presets**: `--preset frame-focus`, `--preset quick-test`, etc.
- **Sampling**: `--sampling representative`, `--sampling quick`, etc.
- **Output**: Comprehensive 54+ field CSV with all engines and metrics

## 🔗 Compatibility

- **Latest symlink**: `results/runs/latest` points to most recent run
- **Cache sharing**: Unified cache database preserves all historical results
- **Tool compatibility**: All analysis tools work with new structure:
  ```bash
  giflab view-failures results/runs/latest/
  giflab select-pipelines results/runs/latest/streaming_results.csv --top 3
  ```

## 🧹 Usage Guidelines

1. **Active Analysis**: Use `giflab run` - saves automatically to `results/runs/`
2. **Historical Data**: Access old experiment data in `results/archive/experiments/` 
3. **Samples**: Test GIFs available in `results/samples/synthetic/`
4. **Cache**: Shared cache benefits all runs and preserves historical results

## 🚀 Migration Summary

This unified system consolidates the pipeline architecture:
- ✅ Single `giflab run` command replaces dual pipeline system
- ✅ Unified output directory `results/runs/` (no more experiments vs runs confusion)
- ✅ Preserved all historical data in `results/archive/experiments/` (171 runs)
- ✅ Migrated cache database for seamless continuation
- ✅ Updated all tool references and documentation

## 📊 Key Capabilities

### Unified Pipeline Features
- **All 5 engines**: gifsicle, Animately, ImageMagick, FFmpeg, gifski
- **54+ metrics**: Enhanced CSV output combining all previous pipeline fields
- **Targeted presets**: 14+ research presets for efficient analysis
- **Intelligent sampling**: representative, quick, full, targeted strategies
- **GPU acceleration**: Optional CUDA-accelerated quality metrics
- **Resume functionality**: Progress tracking and resume capability
- **Comprehensive caching**: Results cache preserves computation across runs

**Access the latest data:**
```bash
cd results/runs/latest/
# Main results: streaming_results.csv (54+ fields)
# Metadata: run_metadata.json
# Progress: elimination_progress.json
```

---
*This unified system provides a single source of truth for all analysis results while preserving complete historical data and enabling seamless workflow continuation.*