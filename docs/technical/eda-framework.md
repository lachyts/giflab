# 📊 Data Analysis & EDA Framework

**Comprehensive guide to GifLab's data analysis capabilities and exploratory data analysis (EDA) framework.**

---

## 1. Overview

GifLab provides a complete data analysis pipeline for understanding GIF datasets and optimizing compression strategies. The system combines automated metric extraction, statistical analysis, and visualization tools.

### 1.1 Analysis Components

- **Automated EDA**: Correlation matrices, histograms, PCA analysis
- **Dataset Exploration**: File characteristics, quality distributions
- **Compression Optimization**: Parameter recommendation system
- **Performance Analysis**: Processing time and quality trade-offs

---

## 2. EDA Generation System

### 2.1 Automated Artifacts

The `giflab.eda.generate_eda()` function produces:

```python
artifacts = {
    'correlation_heatmap': 'correlation_heatmap.png',
    'pca_scree_plot': 'pca_scree_plot.png',
    'hist_ssim': 'hist_ssim.png',
    'hist_ms_ssim': 'hist_ms_ssim.png',
    # ... histogram for each metric
}
```

### 2.2 Implementation

```python
from giflab.eda import generate_eda
from pathlib import Path

# Generate EDA artifacts
csv_path = Path("data/csv/results.csv")
output_dir = Path("data/eda")
artifacts = generate_eda(csv_path, output_dir)

print(f"Generated {len(artifacts)} analysis artifacts")
```

### 2.3 Visualization Features

- **Correlation Analysis**: Pearson correlation heatmaps
- **Distribution Analysis**: Histograms for all numeric metrics
- **Dimensionality Analysis**: PCA scree plots for feature selection
- **Quality Assessment**: Statistical summaries and outlier detection

---

## 3. Dataset Exploration Framework

### 3.1 Notebook Structure

The `01_explore_dataset.ipynb` notebook provides:

#### Setup & Configuration
```python
# Standard analysis imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# GifLab specific tools
from giflab import meta, metrics, config
from giflab.meta import extract_gif_metadata
```

#### Analysis Sections
1. **Dataset Overview**: File counts, size distributions
2. **Quality Metrics**: SSIM, MS-SSIM, PSNR analysis
3. **Compression Analysis**: Engine comparison, parameter optimization
4. **Temporal Analysis**: Frame consistency, motion characteristics
5. **Content Classification**: Type-based quality patterns

### 3.2 Key Analysis Questions

The framework addresses:
- What are the optimal compression parameters for different content types?
- How do quality metrics correlate with human perception?
- Which engines perform best for specific GIF characteristics?
- What are the trade-offs between file size and quality?

---

## 4. Compression Research Integration

### 4.1 Parameter Optimization

Analysis of compression parameters across:
- **Frame keep ratios**: 1.00, 0.90, 0.80, 0.70, 0.50
- **Color reduction**: 256, 128, 64 colors
- **Lossy levels**: 0, 40, 120
- **Engines**: gifsicle, animately

### 4.2 Quality-Size Trade-offs

Research findings on:
- Optimal parameter combinations for different content types
- Quality thresholds for acceptable compression
- Engine-specific performance characteristics
- Temporal consistency preservation strategies

### 4.3 Content-Aware Recommendations

The system can recommend parameters based on:
- GIF content type (screen capture, photography, animation)
- Quality requirements (high, medium, low)
- Size constraints (target file size)
- Processing time limits

---

## 5. Infrastructure Generation

### 5.1 Seed JSON System

The `02_build_seed_json.ipynb` notebook generates:

#### Metadata Seeds
```json
{
  "version": "1.0",
  "gif_metadata": {
    "abc123...": {
      "orig_filename": "example.gif",
      "file_path": "data/raw/example.gif",
      "orig_kilobytes": 245.7,
      "dimensions": [640, 480],
      "frame_count": 30
    }
  }
}
```

#### Processing Seeds
```json
{
  "version": "1.0",
  "parameter_recommendations": {
    "screen_capture": {
      "optimal_lossy": 40,
      "optimal_colors": 128,
      "optimal_frame_ratio": 0.8
    }
  }
}
```

### 5.2 Lookup Tables

Generated lookup tables for:
- **Content classification**: GIF type predictions
- **Parameter recommendations**: Optimized compression settings
- **Quality thresholds**: Acceptable quality ranges
- **Performance estimates**: Processing time predictions

---

## 6. Statistical Analysis Tools

### 6.1 Correlation Analysis

```python
# Generate correlation matrix
correlation_matrix = df[numeric_columns].corr(method="pearson")

# Identify highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))
```

### 6.2 Outlier Detection

```python
from giflab.data_prep import clip_outliers

# Detect outliers using IQR method
for column in numeric_columns:
    outliers = clip_outliers(df[column], method="iqr", factor=1.5)
    outlier_count = len(df[df[column] != outliers])
    print(f"{column}: {outlier_count} outliers detected")
```

### 6.3 Feature Selection

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# PCA analysis
pca = PCA()
pca.fit(df[numeric_columns].fillna(0))
explained_variance = pca.explained_variance_ratio_

# Feature importance
selector = SelectKBest(score_func=f_regression, k=10)
selected_features = selector.fit_transform(X, y)
```

---

## 7. Performance Monitoring

### 7.1 Processing Metrics

Track analysis performance:
- **Dataset size**: Number of GIFs processed
- **Processing time**: Total analysis duration
- **Memory usage**: Peak memory consumption
- **Artifact generation**: Number of visualizations created

### 7.2 Quality Metrics

Monitor analysis quality:
- **Coverage**: Percentage of GIFs successfully analyzed
- **Accuracy**: Validation against ground truth
- **Consistency**: Reproducibility across runs
- **Completeness**: All required artifacts generated

---

## 8. Integration with ML Pipeline

### 8.1 Feature Engineering

The analysis framework supports:
- **Automated scaling**: MinMax and Z-score normalization
- **Missing value handling**: Imputation strategies
- **Feature selection**: PCA and correlation-based selection
- **Outlier treatment**: Robust statistical methods

### 8.2 Model Validation

Tools for model development:
- **Train/test splitting**: GIF-level stratification
- **Cross-validation**: Time-aware splitting
- **Performance metrics**: Regression and classification metrics
- **Hyperparameter tuning**: Grid search integration

### 8.3 Production Deployment

Ready-to-use components:
- **Batch processing**: Large dataset analysis
- **Real-time analysis**: Single GIF processing
- **API integration**: RESTful analysis endpoints
- **Monitoring**: Performance and quality tracking

