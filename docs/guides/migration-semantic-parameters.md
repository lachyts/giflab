# Semantic Parameters Reference

## Overview

GifLab uses semantic `applied_*` parameters instead of template-based `test_*` parameters for accurate ML analysis. This guide explains the parameter format and provides strategies for working with legacy data.

## Parameter Formats

### Legacy Format (Template Parameters)
```csv
test_colors,test_lossy,test_frame_ratio,pipeline_steps
16,80,1.0,3
16,80,0.5,3
```
**Issue**: Always showed template values, even for no-op steps

### Current Format (Semantic Parameters)  
```csv
applied_colors,applied_lossy,applied_frame_ratio,actual_pipeline_steps
16,80,,2
,,,1
```
**Benefit**: Only shows values when processing actually occurred

## Working with Data

### 1. Current Analysis Code

**✅ Recommended: Use backward-compatible code for legacy data**
```python
# Prefer new columns, fallback to old for compatibility
colors = row.get('applied_colors', row.get('test_colors', None))
lossy = row.get('applied_lossy', row.get('test_lossy', None))
frames = row.get('applied_frame_ratio', row.get('test_frame_ratio', None))
```

### 2. Working with Legacy CSV Data

**Option A: Keep Legacy Data Separate (Recommended)**
```bash
# Legacy data remains in original files
# New analysis runs generate semantic format automatically
```

**Option B: Legacy Data Conversion Script**
```python
import pandas as pd

# Load old data
df_old = pd.read_csv('elimination_history_master.csv')

# Convert template columns to semantic format
# (Requires pipeline reconstruction - complex)
df_new = convert_to_semantic_columns(df_old)

# Save converted data
df_new.to_csv('legacy_data_converted.csv', index=False)
```

### 3. Database Caches

**Automatic Support**: The system automatically handles both legacy and current column formats in cache databases.

## Data Interpretation

### Understanding NULL Values

In the semantic format, `NULL`/`None` values are semantically meaningful:

- **`applied_colors: NULL`** = No color reduction was applied (either no-op or failure)
- **`applied_colors: 16`** = Color reduction to 16 colors was actually applied

### ML Feature Engineering

```python
# Create processing type indicators
df['used_color_reduction'] = df['applied_colors'].notna()
df['used_frame_reduction'] = df['applied_frame_ratio'].notna()  
df['used_lossy_compression'] = df['applied_lossy'].notna()

# Count actual processing steps (excludes no-ops)
processing_complexity = df['actual_pipeline_steps']
```

## Legacy Data Handling

For legacy data analysis, use the template columns directly:

```python
# Working with legacy data
colors = row.get('test_colors', None)
lossy = row.get('test_lossy', None)
frames = row.get('test_frame_ratio', None)
```

## Data Validation

When working with semantic parameters, validate data quality:

```python
# Check that semantic columns make sense
assert df['actual_pipeline_steps'].max() <= df['pipeline_steps'].max()
assert df[df['success'] == False]['applied_colors'].isna().all()
```

## Summary

- **Current**: All pipeline runs use semantic parameters
- **Legacy Support**: Code supports both legacy and current column formats
- **Best Practice**: Use backward-compatible code when working with mixed datasets

For detailed information on semantic parameters, refer to the documentation in `docs/guides/testing-best-practices.md`. 