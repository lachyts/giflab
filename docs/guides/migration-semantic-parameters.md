# Migration Guide: Semantic Parameter Transition

## Overview

GifLab has transitioned from template-based `test_*` parameters to semantic `applied_*` parameters for better ML analysis accuracy. This guide covers migration strategies for existing data.

## What Changed

### Old Format (Template Parameters)
```csv
test_colors,test_lossy,test_frame_ratio,pipeline_steps
16,80,1.0,3
16,80,0.5,3
```
**Issue**: Always showed template values, even for no-op steps

### New Format (Semantic Parameters)  
```csv
applied_colors,applied_lossy,applied_frame_ratio,actual_pipeline_steps
16,80,,2
,,,1
```
**Benefit**: Only shows values when processing actually occurred

## Migration Strategies

### 1. For Analysis Code

**✅ Recommended: Use backward-compatible code**
```python
# Prefer new columns, fallback to old for compatibility
colors = row.get('applied_colors', row.get('test_colors', None))
lossy = row.get('applied_lossy', row.get('test_lossy', None))
frames = row.get('applied_frame_ratio', row.get('test_frame_ratio', None))
```

### 2. For CSV Data Migration

**Option A: Rename Master History File (Recommended)**
```bash
# Backup existing master file
mv elimination_results/elimination_history_master.csv \
   elimination_results/elimination_history_master_legacy.csv

# New runs will create a fresh master file with semantic columns
```

**Option B: Column Migration Script**
```python
import pandas as pd

# Load old data
df_old = pd.read_csv('elimination_history_master.csv')

# Create new semantic columns based on pipeline analysis
# (Requires pipeline reconstruction - complex)
df_new = migrate_to_semantic_columns(df_old)

# Save migrated data
df_new.to_csv('elimination_history_master_migrated.csv', index=False)
```

### 3. For Database Caches

**Automatic Migration**: The system automatically adds new columns to existing cache databases during initialization. No manual action required.

## Data Interpretation

### Understanding NULL Values

In the new format, `NULL`/`None` values are semantically meaningful:

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

## Rollback Strategy

If issues arise, you can temporarily revert to old column references:

```python
# Emergency fallback to old columns
colors = row.get('test_colors', None)
lossy = row.get('test_lossy', None)
frames = row.get('test_frame_ratio', None)
```

## Validation

After migration, validate data quality:

```python
# Check that semantic columns make sense
assert df['actual_pipeline_steps'].max() <= df['pipeline_steps'].max()
assert df[df['success'] == False]['applied_colors'].isna().all()
```

## Timeline

- **Immediate**: All new pipeline runs use semantic parameters
- **Transition**: Code supports both old and new column formats
- **Future**: Legacy `test_*` columns may be deprecated

For questions or issues during migration, refer to the semantic parameter documentation in `docs/guides/testing-best-practices.md`. 