# Quality Metrics Expansion – Implementation Plan
Author: OpenAI Assistant  
Date: 2025-07-13

## 1  Context
`giflab.metrics` currently exposes SSIM, MS-SSIM, PSNR and temporal-consistency.  
To reach feature-parity with the reference evaluator we must add eight additional frame-level metrics:

| Metric                       | Abbrev. | Implemented? | Notes |
|------------------------------|---------|--------------|-------|
| Mean Squared Error           | MSE     |             | trivial via `skimage.metrics.mean_squared_error` |
| Root Mean Squared Error      | RMSE    |             | `sqrt(MSE)` |
| Feature Similarity Index     | FSIM    |             | custom implementation (gradients + phase-congruency) |
| Gradient Mag. Sim. Deviation | GMSD    |             | Prewitt gradients + STD of GMS map |
| Colour-Histogram Correlation | CHIST   |             | Earth-Mover’s/Correlation correlation per channel |
| Edge-Map Jaccard Similarity  | EDGE    |             | Canny + intersection / union |
| Texture-Histogram Correlation| TEXTURE |             | LBP (uniform, P=8,R=1) histograms |
| Sharpness Similarity         | SHARP   |             | Laplacian variance ratio |

*Weighting and composite scoring will **not** be added – we only expose raw metric values.*

---

## 2  Design Overview

1. **Metric Functions**  
   Add eight pure helper functions inside `giflab.metrics`.  
   • Signature: `def metric_name(frame1: np.ndarray, frame2: np.ndarray) -> float`  
   • No external state; raise `ValueError` on shape mismatch or invalid data.  
   • Unit-tested individually.

2. **Integration into Existing API**  
   Extend `calculate_comprehensive_metrics` so it computes and returns these eight metrics alongside the current ones.  
   The function signature remains unchanged; callers will automatically receive the expanded metric dictionary.

3. **Config Updates**  
   • None required – existing configuration stays valid.

4. **CLI / Pipeline Touchpoints**  
   • No interface changes are needed.  
   • Down-stream code continues to call `giflab.metrics.calculate_comprehensive_metrics`.

---

## 3  Implementation Steps

| Step | Task | Owner | Depends on | Time |
|------|------|-------|-----------|------|
| 1 | Implement the eight new frame-level metric functions (`metrics.py`) | Dev | — | 0.5 d |
| 2 | Extend `calculate_comprehensive_metrics` to compute *all* metrics **plus** aggregation descriptors (`std`, `min`, `max`) | Dev | 1 | 0.25 d |
| 3 | Add `raw_metrics` flag to `calculate_comprehensive_metrics` (expose un-normalised values) | Dev | 2 | 0.1 d |
| 4 | Introduce **`MetricRecordV1`** pydantic schema; validate every export row | Dev | 1-3 | 0.25 d |
| 5 | Create `giflab/data_prep.py` helpers for scaling, confidence-weighting, and outlier clipping | Dev | 1-4 | 0.35 d |
| 6 | Re-evaluate temporal-consistency metric (compute pre- & post-compression delta) | Dev | 1 | 0.2 d |
| 7 | Build **EDA artefacts**: exploratory notebook with distributions, correlations, PCA scree plots | Dev/QA | 5 | 0.25 d |
| 8 | Add `make data` target to automate dataset extraction + EDA | Dev | 5,7 | 0.1 d |
| 9 | **Unit & integration tests**: metrics, preprocessing helpers, schema validation, temporal-delta | QA | 1-8 | 0.5 d |
| 10 | **CI & Tooling**: add schema-validation step, notebook smoke-test, scikit-image pin, caching | Dev | 4,9 | 0.15 d |
| 11 | **Docs**: update README, QUALITY_METRICS_APPROACH.md, notebooks; embed ML checklist | Dev | 1-10 | 0.15 d |
| 12 | Code review & merge | Peers | 1-11 | 0.1 d |

*Total estimate: **2.8 developer-days + 0.5 QA-day***

---

## 4  Implementation Details

### 4.1  MSE / RMSE
```python
from skimage.metrics import mean_squared_error
def mse(frame1, frame2):
    _f1, _f2 = _resize_if_needed(frame1, frame2)
    return float(mean_squared_error(_f1, _f2))

def rmse(frame1, frame2):
    return math.sqrt(mse(frame1, frame2))
```

### 4.2  FSIM  
Follow the reference algorithm: gradients (np.gradient), gradient-magnitude similarity, phase-congruency similarity, weighted sum.

### 4.3  GMSD  
Use 3 × 3 Prewitt kernels (constant kernels reused), compute gradient magnitude maps, gms map, then return `np.std(gms)`.

### 4.4  Colour-Histogram Similarity  
`cv2.calcHist` → normalise → `cv2.compareHist(..., HISTCMP_CORREL)` per channel → mean of three.

### 4.5  Edge Similarity  
Convert to gray → `cv2.Canny(gray,50,150)` → boolean intersection/union → Jaccard.

### 4.6  Texture Similarity  
`skimage.feature.local_binary_pattern(gray, 8, 1, 'uniform')` → 10-bin histogram range 0-9 → normalise → `np.corrcoef(h1,h2)[0,1]` (fallback 1.0 if variance 0).

### 4.7  Sharpness Similarity  
`cv2.Laplacian(gray,CV_64F)` → `np.var` → `min(v1,v2) / max(v1,v2)`; when both zero → 1.0.

### 4.8  Helper: `_resize_if_needed`  
Re-use existing `resize_to_common_dimensions`, but internal helper avoids list/loop overhead for single frames.

### 4.9  GIF-Level Aggregation & `raw_metrics` Flag

To surface per-GIF statistics without hiding large per-frame deviations we calculate three descriptors **per metric**:

```python
metric_mean, metric_std, metric_min, metric_max = aggregate_metric(array_like)
```

Implementation notes:
• Computed over the *aligned* frame-level scores returned by each metric function.  
• `aggregate_metric` returns a `dict[str,float]` keyed as `metric`, `metric_std`, `metric_min`, `metric_max`.
• Stable for edge cases: if only one frame → descriptors set to 0 (std) / same value (min/max).

`raw_metrics` flag behaviour:
• **Default** = `False` – API & CSV emit *scaled/normalised* values per current behaviour.
• When `raw_metrics=True`, the function appends each metric's un-scaled raw value alongside the normalised one using key suffix `_raw`.
• Flag propagates through CLI and pipeline via `MetricsConfig.RAW_METRICS` boolean.

CSV/Schema impact:
• New columns `<metric>_std`, `<metric>_min`, `<metric>_max`, and optional `<metric>_raw` if flag enabled.  
• `MetricRecordV1` enforces nullable `float` with `ge=0`. Missing values encoded as `NaN`.

---

## 5  Testing Strategy

1. **Deterministic fixtures**: generate pairs of identical, slightly different, and very different 50 × 50 RGB arrays.  
2. Assert metric ranges  
   • Identical → value ≈ 1 (or 0 for MSE/RMSE)  
   • Totally different → value ≈ 0 (or high for MSE/RMSE)  
3. Negative tests: mismatched shapes, invalid dtype → expect `ValueError`.  
4. Config flag verifies that default results dict size doesn’t change.

---

## 6  Risk & Mitigations
• **Performance**: Added aggregation & temporal-delta increase runtime by ~7 %; mitigate with lazy evaluation flag.  
• **Third-party dependencies**: pin `scikit-image>=0.21`, `pydantic>=2.5`; licence review.  
• **Schema Drift**: `MetricRecordV1` changes may break legacy consumers ⇒ provide migration guide & versioned CSV.  
• **Scaling Bugs**: Incorrect scaler reuse can skew models ⇒ include golden-CSV fixture in tests.  
• **CI Overhead**: Notebook smoke-tests add minutes ⇒ run in parallel workflow job.

---

## 7  Done Definition
- [] All eight metrics **and aggregation descriptors** implemented and documented.
- [ ] `raw_metrics` flag operational.
- [ ] `MetricRecordV1` schema integrated; all exports validated.
- [ ] Data-prep helpers (`data_prep.py`) created with scaling, confidence, and outlier handling.
- [ ] EDA notebook & correlation dashboard generated.
- [ ] `make data` automation target functional.
- [ ] Temporal-consistency delta calculation in place.
- [ ] Unit-test suite green (`pytest -q`).
- [ ] CI pipeline passes.
- [ ] README & plan updated with metric table and ML checklist.
- [ ] Plan document updated with ✅ ticks and merged.

## 8  ML Data-Quality Considerations and Best Practices

Feeding the expanded metric set directly into a machine-learning workflow revealed several caveats that must be addressed to avoid data-quality pitfalls.  The table below consolidates the main issues, their potential impact on downstream models, and recommended mitigations.

| Issue | Impact on ML | Recommended Mitigation |
|-------|--------------|------------------------|
| **Metric scale heterogeneity** (e.g., MSE/RMSE un-bounded, others 0-1) | Algorithms that rely on distance measures or weight magnitudes (k-NN, linear models, neural nets) may over- or under-weight features. | Provide a dedicated helper `metrics.normalise_metrics(dict, method="zscore"|"minmax")`; document standard scaling recipes in the data-prep notebook. |
| **Implicit clipping / non-linear transforms** (PSNR divided by 50, FSIM clipped to 0-1) | Skews feature distributions; distorts loss gradients if used directly as objectives. | Always expose *raw* metric values alongside any normalised versions; make clipping thresholds configurable via `config.yaml`. |
| **Low-confidence metrics set to `0.0`** | Model may interpret *0* as a valid measurement when it is in fact "unknown". | Represent missing values as `np.nan` and provide an imputation helper; alternatively multiply each metric by its confidence score during feature engineering. |
| **Correlation / redundancy** (SSIM, MS-SSIM, FSIM, PSNR are correlated) | Multicollinearity inflates variance of linear estimators and can mask signal. | Supply a correlation matrix notebook; encourage PCA/feature-selection (e.g., Variance Inflation Factor threshold < 5). |
| **Temporal-consistency computed only on compressed frames** | Using different reference frames can introduce bias. | Compute temporal consistency both *pre-* and *post-* compression and store the delta. |
| **Outlier sensitivity** (gradient-based metrics pick up sensor noise) | Heavy-tailed distributions hinder convergence and degrade model robustness. | Offer robust statistics (`median`, `IQR`, `Huber`) and optional log-scale transforms in the preprocessing helper. |
| **Frame-level aggregation (mean only)** | Averaging can hide severe artefacts in a few frames, reducing label fidelity. | Export additional descriptors per GIF: `metric_std`, `metric_min`, `metric_max`, and frame count. |
| **Data-set & code version drift** | Results hard to reproduce; silent metric changes break models. | Embed semantic version of `giflab`, git commit hash, and parameter checksum in every exported record. |
| **Leakage via split strategy** | Frames from the same GIF appearing in both train and test sets inflates performance metrics. | Perform *GIF-level* (not frame-level) stratified splitting and persist indices to disk. |
| **Inconsistent colour spaces / bit-depths** | Shifts feature distributions and corrupts colour-based metrics. | Standardise to sRGB, 8-bit per channel before metric computation; assert via helper. |

### 8.1  Best-Practice Checklist

The following checklist SHOULD be ticked before shipping any metric dataset to an ML consumer:

- [ ] **Deterministic extraction** – All random seeds fixed; metric functions are pure and side-effect free.
- [ ] **Schema validation** – Enforce with `pydantic` model `MetricRecordV1`; CI fails on schema drift.
- [ ] **Version tagging** – Dataset files include `dataset_version`, `code_commit`, and `giflab_version` fields.
- [ ] **Canonical splits** – Persist `train.json`, `val.json`, `test.json` that list GIF identifiers; prohibit re-sampling in notebooks.
- [ ] **Scale & normalisation logs** – `data_prep.py` writes a `.scaler.pkl` with fitted parameters for inference parity.
- [ ] **Outlier report** – Generate an HTML report flagging > 3 σ points per metric.
- [ ] **Correlation dashboard** – Auto-update Spearman/Pearson tables and PCA scree plots.
- [ ] **Unit tests for preprocessing** – Cover scaling, imputation, clipping and confidence-weighting paths.
- [ ] **Reproducible pipelines** – Provide `Makefile` or `tox` target that runs end-to-end extraction & EDA.

### 8.2  Status Note

*Former “Follow-Up Tasks (Next Sprint)” items have been **merged into Section&nbsp;3** and the Done Definition above, making them part of this sprint’s scope.* 

