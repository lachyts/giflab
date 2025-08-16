# Unified Results Directory

This directory contains all experimental results, samples, and cached data in a unified, organized structure.

## 📁 Directory Structure

```
results/
├── archive/                    # Historical data (read-only)
│   ├── legacy-runs/           # Pre-unification experiment_results content
│   └── pre-enhanced-metrics/  # Pre-unification test-workspace content
├── experiments/               # Active experiments (organized by type)
│   ├── frame-comparison/      # Frame algorithm studies
│   │   ├── enhanced-metrics-study_20250807/
│   │   │   └── run_20250807_123641/  # Key enhanced metrics experiment
│   │   └── latest -> enhanced-metrics-study_20250807/run_20250807_123641/
│   ├── matrix-analysis/       # Matrix pipeline experiments  
│   ├── quality-validation/    # Enhanced metrics validation
│   └── custom-studies/        # One-off investigations
├── samples/                   # Test GIF samples (centralized)
│   ├── synthetic/            # Generated test content
│   └── real-world/           # Actual user content
└── cache/                     # Shared cache files
    ├── pipeline_results_cache.db
    └── elimination_history_master.csv
```

## 🎯 Experiment Organization

### Naming Convention
Experiments are organized by type, then by study name and date:
```
experiments/{type}/{study-name}_{YYYYMMDD}/
└── run_{YYYYMMDD_HHMMSS}/
    ├── enhanced_streaming_results.csv
    ├── run_metadata.json
    └── analysis_outputs/
```

### Current Experiments
- **frame-comparison/**: Frame reduction algorithm studies
  - `enhanced-metrics-study_20250807/`: Key study using 11-metric enhanced quality system
- **matrix-analysis/**: Future matrix pipeline experiments
- **quality-validation/**: Enhanced metrics validation studies
- **custom-studies/**: One-off investigations

## 🔗 Compatibility

- **Legacy symlink**: `experiment_results -> results/archive/legacy-runs/`
- **Latest symlinks**: Each experiment type has a `latest` link to most recent run
- **Backward compatibility**: All existing analysis scripts continue to work

## 🧹 Usage Guidelines

1. **Active Work**: Use `experiments/{type}/` for ongoing studies
2. **Samples**: Add test GIFs to appropriate `samples/` subdirectory  
3. **Archives**: Historical data in `archive/` is read-only
4. **Cache**: Shared cache files benefit all experiments

## 🚀 Migration Summary

This unified system consolidates:
- ✅ All `experiment_results/` content → `archive/legacy-runs/`
- ✅ Key `test-workspace/` experiments → `archive/pre-enhanced-metrics/`  
- ✅ Enhanced metrics study → `experiments/frame-comparison/`
- ✅ GIF samples → `samples/synthetic/`
- ✅ Cache files → `cache/`
- ✅ Updated system defaults to use `results/experiments/`

## 📊 Key Results Available

### Enhanced Metrics Study (`experiments/frame-comparison/latest/`)
- **450 successful pipeline tests** with comprehensive 11-metric quality assessment
- **Frame reduction algorithm comparison** (gifsicle, imagemagick, animately, ffmpeg, none)
- **Enhanced composite quality** using research-based 11-metric weights
- **Efficiency scoring** balancing compression × quality
- **Content type analysis** across 14 different GIF categories

**Access the data:**
```bash
cd results/experiments/frame-comparison/latest/
# Main results: enhanced_streaming_results.csv
# Metadata: run_metadata.json
```

---
*This unified system provides single source of truth for all experimental results while preserving complete historical data.*