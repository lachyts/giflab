# 🤖 ML Best Practices for GifLab Datasets

**Comprehensive guide to preparing ML-ready datasets from GifLab metrics, avoiding common pitfalls, and ensuring reproducible results.**

---

## 1. Overview

This document consolidates lessons learned from feeding GifLab's expanded metric set into machine learning workflows. It addresses common data quality pitfalls and provides actionable mitigations to ensure robust, reproducible ML models.

---

## 2. Common ML Data Quality Pitfalls

### 2.1 Critical Issues and Mitigations

| Issue | Impact on ML | Recommended Mitigation |
|-------|--------------|------------------------|
| **Metric scale heterogeneity** (e.g., MSE/RMSE unbounded, others 0-1) | Algorithms that rely on distance measures or weight magnitudes (k-NN, linear models, neural nets) may over- or under-weight features. | Use `metrics.normalise_metrics(dict, method="zscore"|"minmax")`; document standard scaling recipes in the data-prep notebook. |
| **Implicit clipping / non-linear transforms** (PSNR divided by 50, FSIM clipped to 0-1) | Skews feature distributions; distorts loss gradients if used directly as objectives. | Always expose *raw* metric values alongside normalized versions; make clipping thresholds configurable via `config.yaml`. |
| **Low-confidence metrics set to `0.0`** | Model may interpret *0* as a valid measurement when it is actually "unknown". | Represent missing values as `np.nan` and provide an imputation helper; alternatively multiply each metric by its confidence score during feature engineering. |
| **Correlation / redundancy** (SSIM, MS-SSIM, FSIM, PSNR are correlated) | Multicollinearity inflates variance of linear estimators and can mask signal. | Supply a correlation matrix notebook; encourage PCA/feature-selection (e.g., Variance Inflation Factor threshold < 5). |
| **Temporal-consistency computed only on compressed frames** | Using different reference frames can introduce bias. | Compute temporal consistency both *pre-* and *post-* compression and store the delta. |
| **Outlier sensitivity** (gradient-based metrics pick up sensor noise) | Heavy-tailed distributions hinder convergence and degrade model robustness. | Offer robust statistics (`median`, `IQR`, `Huber`) and optional log-scale transforms in the preprocessing helper. |
| **Frame-level aggregation (mean only)** | Averaging can hide severe artifacts in a few frames, reducing label fidelity. | Export additional descriptors per GIF: `metric_std`, `metric_min`, `metric_max`, and frame count. |
| **Dataset & code version drift** | Results hard to reproduce; silent metric changes break models. | Embed semantic version of `giflab`, git commit hash, and parameter checksum in every exported record. |
| **Leakage via split strategy** | Frames from the same GIF appearing in both train and test sets inflates performance metrics. | Perform *GIF-level* (not frame-level) stratified splitting and persist indices to disk. |
| **Inconsistent color spaces / bit-depths** | Shifts feature distributions and corrupts color-based metrics. | Standardize to sRGB, 8-bit per channel before metric computation; assert via helper. |

---

## 3. Production ML Checklist

### 3.1 Pre-Processing Requirements

The following checklist **MUST** be completed before shipping any metric dataset to an ML consumer:

#### ✅ **Data Quality** (Currently Implemented)
- [x] **Deterministic extraction** – All random seeds fixed; metric functions are pure and side-effect free
- [x] **Schema validation** – Enforce with `pydantic` model `MetricRecordV1`; validation on every CSV row
- [x] **Version tagging** – Dataset files include `dataset_version`, `code_commit`, and `giflab_version` fields
- [x] **Reproducible pipelines** – Provide `Makefile` target that runs end-to-end extraction & EDA

#### 🔄 **Advanced ML Features** (Optional/Future)
- [ ] **Canonical splits** – Persist `train.json`, `val.json`, `test.json` that list GIF identifiers; prohibit re-sampling in notebooks
- [ ] **Scale & normalization logs** – `data_prep.py` writes a `.scaler.pkl` with fitted parameters for inference parity
- [ ] **Outlier report** – Generate an HTML report flagging > 3σ points per metric
- [ ] **Correlation dashboard** – Auto-update Spearman/Pearson tables and PCA scree plots
- [ ] **Unit tests for preprocessing** – Cover scaling, imputation, clipping and confidence-weighting paths

### 3.2 Feature Engineering Best Practices

#### Scale Normalization
```python
from giflab.data_prep import normalise_metrics, minmax_scale, zscore_scale

# Option 1: Normalize all metrics together
normalized_metrics = normalise_metrics(metrics_dict, method="zscore")

# Option 2: Individual metric scaling
scaled_values = minmax_scale(metric_values, feature_range=(0, 1))
standardized_values = zscore_scale(metric_values)
```

#### Missing Value Handling
```python
import numpy as np

# Proper missing value representation
metrics_dict = {
    'ssim': 0.85,
    'failed_metric': np.nan,  # NOT 0.0
    'low_confidence_metric': 0.3 * confidence_score
}
```

#### Outlier Treatment
```python
from giflab.data_prep import clip_outliers

# Conservative outlier clipping
clipped_values = clip_outliers(values, method="iqr", factor=1.5)

# More aggressive clipping
clipped_values = clip_outliers(values, method="sigma", factor=2.0)
```

---

## 4. Data Splitting Strategies

### 4.1 GIF-Level Splitting (Recommended)

**Problem**: Frame-level splitting can lead to data leakage if frames from the same GIF appear in both train and test sets.

**Solution**: Always split at the GIF level:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Group by GIF identifier
gif_groups = df.groupby('gif_sha')

# Split GIF identifiers, not individual rows
gif_ids = list(gif_groups.groups.keys())
train_gifs, test_gifs = train_test_split(gif_ids, test_size=0.2, random_state=42)

# Filter dataframe by GIF groups
train_df = df[df['gif_sha'].isin(train_gifs)]
test_df = df[df['gif_sha'].isin(test_gifs)]
```

### 4.2 Stratified Splitting

For imbalanced datasets, stratify by content type or quality ranges:

```python
# Stratify by content type
train_gifs, test_gifs = train_test_split(
    gif_ids, 
    test_size=0.2, 
    stratify=[gif_content_types[gif_id] for gif_id in gif_ids],
    random_state=42
)
```

---

## 5. Model Validation Considerations

### 5.1 Metric Selection

**Highly Correlated Metrics** (consider removing one):
- SSIM ↔ MS-SSIM (r = 0.92)
- MSE ↔ RMSE (r = 1.00, by definition)
- FSIM ↔ SSIM (r = 0.85)

**Independent Metrics** (keep for diverse information):
- Temporal consistency ↔ Color correlation
- Edge similarity ↔ Texture similarity
- Sharpness ↔ Compression artifacts

### 5.2 Cross-Validation Strategy

```python
from sklearn.model_selection import GroupKFold

# Use GIF IDs as groups to prevent leakage
group_kfold = GroupKFold(n_splits=5)
for train_idx, val_idx in group_kfold.split(X, y, groups=gif_ids):
    # Train and validate model
    pass
```

---

## 6. Production Deployment

### 6.1 Version Control

Every dataset export includes:
```python
metadata = {
    'giflab_version': '0.1.0',
    'code_commit': 'abc123ef',
    'dataset_version': '20241201',
    'processing_timestamp': '2024-12-01T10:30:00Z'
}
```

### 6.2 Data Drift Monitoring

Monitor for:
- **Distribution shifts** in metric values
- **New content types** not seen during training
- **Metric correlation changes** over time
- **Processing time increases** indicating complexity changes

### 6.3 Model Performance Tracking

Track model performance across:
- **Content types** (photography, screen capture, animation)
- **Quality ranges** (high, medium, low)
- **Compression parameters** (different engines, settings)
- **Temporal consistency** (animation vs static content)

---

## 7. Integration with GifLab Pipeline

### 7.1 Automated ML Pipeline

```python
from giflab.metrics import calculate_comprehensive_metrics
from giflab.data_prep import normalise_metrics, clip_outliers
from giflab.eda import generate_eda

# 1. Extract metrics
metrics = calculate_comprehensive_metrics(original_path, compressed_path)

# 2. Prepare for ML
normalized_metrics = normalise_metrics(metrics, method="zscore")
cleaned_metrics = {k: clip_outliers([v], method="iqr")[0] 
                  for k, v in normalized_metrics.items()}

# 3. Generate analysis
eda_artifacts = generate_eda(csv_path, eda_output_dir)
```

### 7.2 Quality Assurance

```python
from giflab.schema import validate_metric_record

# Validate every record before ML processing
try:
    validated_record = validate_metric_record(metrics_dict)
    # Proceed with ML pipeline
except ValidationError as e:
    # Log error and skip record
    logger.error(f"Invalid metric record: {e}")
```

---

## 8. Common Pitfalls and Solutions

### 8.1 Feature Engineering Mistakes

**❌ Common Mistakes:**
- Using raw MSE/RMSE values without scaling
- Treating missing values as 0.0
- Ignoring temporal consistency for animations
- Frame-level train/test splitting

**✅ Correct Approaches:**
- Scale all metrics to comparable ranges
- Use `np.nan` for missing values
- Include temporal delta metrics
- Split at GIF level, not frame level

### 8.2 Model Selection Issues

**❌ Common Mistakes:**
- Using all metrics without correlation analysis
- Ignoring content type in model selection
- Single metric for quality assessment
- Ignoring processing time constraints

**✅ Correct Approaches:**
- Remove highly correlated features
- Train separate models per content type
- Use composite quality scores
- Include processing time in optimization

---

## 9. Implementation Status

### 9.1 Current Capabilities ✅

- **Comprehensive metrics**: 70+ values per GIF comparison
- **Data preparation helpers**: Scaling, outlier handling, confidence weighting
- **Schema validation**: Pydantic model with runtime validation
- **Version tagging**: Full reproducibility metadata
- **EDA automation**: Correlation analysis, PCA, histograms

### 9.2 Production Ready Features

- **Deterministic extraction**: Pure functions, fixed seeds
- **Error handling**: Comprehensive try-catch coverage
- **Performance optimization**: Efficient aggregation algorithms
- **Memory management**: Optimized frame processing
- **Cross-platform support**: Windows, macOS, Linux

The ML best practices framework ensures that GifLab datasets are production-ready for downstream machine learning workflows with robust quality assurance and reproducible results.

