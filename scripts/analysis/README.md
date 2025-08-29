# Analysis Scripts

This directory contains analysis scripts for processing and understanding experimental results.

## 📊 Enhanced Metrics Analysis Scripts

### Core Analysis Scripts
- **`analyze_enhanced_metrics.py`** - Comprehensive analysis of enhanced vs legacy metrics
- **`efficiency_insights.py`** - Focused analysis of efficiency scoring system
- **`dataset_breakdown.py`** - Complete dataset breakdown with enhanced metrics
- **`key_findings.py`** - Key findings and actionable insights summary

### Frame Reduction Analysis
- **`frame_reduction_deep_dive.py`** - Deep dive into frame reduction algorithm behavior
- **`frame_reduction_insights.py`** - Clear explanations of frame reduction findings

### Utility Scripts
- **`process_existing_results.py`** - Process existing results to add enhanced metrics
- **`test_enhanced_metrics.py`** - Test and validate enhanced metrics system

## 🚀 Usage

### Quick Analysis
```bash
# Run from project root
cd /Users/lachlants/repos/animately/giflab

# Comprehensive enhanced metrics analysis
python scripts/analysis/analyze_enhanced_metrics.py

# Focused efficiency insights
python scripts/analysis/efficiency_insights.py

# Dataset breakdown
python scripts/analysis/dataset_breakdown.py
```

### Frame Reduction Analysis
```bash
# Deep dive into frame reduction algorithms
python scripts/analysis/frame_reduction_deep_dive.py

# Clear insights and explanations
python scripts/analysis/frame_reduction_insights.py
```

### Processing Scripts
```bash
# Add enhanced metrics to existing results
python scripts/analysis/process_existing_results.py

# Test enhanced metrics system
python scripts/analysis/test_enhanced_metrics.py
```

## 📁 Data Sources

All scripts are configured to work with the unified results directory:
- **Main data**: `results/runs/frame-comparison/latest/enhanced_streaming_results.csv`
- **Sample GIFs**: `results/samples/synthetic/`

## 🔍 Key Analyses Available

### Enhanced Metrics System (11-metric vs 4-metric)
- Comprehensive quality assessment using 11 dimensions
- Research-based weighting schemes
- More accurate quality predictions

### Efficiency Scoring
- `efficiency = compression_ratio × composite_quality`
- Balances compression and quality retention
- Clear performance tiers (Poor/Fair/Good/Excellent/Outstanding)

### Frame Reduction Algorithm Comparison
- 5 algorithms: gifsicle, imagemagick, animately, ffmpeg, none
- Counter-intuitive findings about frame reduction effectiveness
- Content-type specific recommendations

### Content Type Analysis
- 14 different GIF content types analyzed
- Performance varies dramatically by content (17x efficiency difference)
- Motion content compresses exceptionally well (up to 60x)

---
*These scripts provide comprehensive analysis of the enhanced metrics system and its insights into GIF compression behavior.*