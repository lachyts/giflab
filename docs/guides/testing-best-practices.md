# Testing Best Practices

## 🎯 Overview

This document establishes clear guidelines for testing in GifLab to maintain a clean, professional codebase and prevent root directory pollution.

## 📁 Directory Structure

### Proper Testing Locations

```
giflab/
├── tests/                    # ✅ Unit & integration tests (pytest)
│   ├── fixtures/            # Test GIF files for automated tests
│   └── test_*.py           # Automated test files
├── test-workspace/          # ✅ Manual testing & debugging
│   ├── manual/             # Manual testing sessions
│   ├── debug/              # Debug investigations  
│   ├── temp/               # Temporary files (auto-cleaned)
│   └── samples/            # Test samples & reference files
└── docs/guides/            # ✅ Testing documentation
```

### ❌ NEVER Put Testing Files In:
- **Root directory** (`/`) - Keep it clean!
- **src/** - Source code only
- **docs/** - Documentation only

## 🔧 Testing Workflows

### 1. **Unit/Integration Tests** (`tests/`)
```bash
# Development: Lightning-fast tests (<30s)
make test-fast

# Pre-commit: Comprehensive integration tests (<5min)
make test-integration

# Release: Full test matrix (<30min)  
make test-full

# Add new test files
pytest tests/test_new_feature.py

# Run specific tests
pytest tests/test_new_feature.py
```

### 2. **Manual Testing** (`test-workspace/manual/`)
```bash
# Create session directory
mkdir test-workspace/manual/feature-debugging-YYYYMMDD
cd test-workspace/manual/feature-debugging-YYYYMMDD

# Run your tests
python -m giflab experiment sample.gif

# Clean up when done (see cleanup section)
```

### 3. **Debug Investigations** (`test-workspace/debug/`)
```bash
# Create debug session
mkdir test-workspace/debug/pipeline-issue-investigation
cd test-workspace/debug/pipeline-issue-investigation

# Run debugging
python debug_script.py
```

## 🧹 Cleanup Protocols

### 1. **Auto-Cleanup** (temp files)
The `test-workspace/temp/` directory is automatically cleaned:
- Files older than 7 days
- Run `make clean-temp` or automated daily

### 2. **Manual Cleanup** (after testing)
```bash
# After completing manual testing session
cd test-workspace/manual/
rm -rf completed-session-name/

# Keep only active investigations
```

### 3. **Quarterly Cleanup**
- Review `test-workspace/` quarterly
- Archive important findings to `docs/analysis/`
- Delete outdated debug sessions

## 📋 Testing Session Checklist

### Before Starting Testing:
- [ ] Create properly named directory in `test-workspace/`
- [ ] Use descriptive session names: `feature-name-YYYYMMDD` or `bug-investigation-brief-desc`
- [ ] Document purpose in session README if investigation is complex

### After Completing Testing:
- [ ] Move important findings to appropriate `docs/` location
- [ ] Delete temporary files and unsuccessful experiments
- [ ] Clean up test session directory
- [ ] Update any relevant documentation

## 🚨 Emergency Cleanup

If you find testing files polluting the root directory:

```bash
# Run the cleanup script
make clean-testing-mess

# Or manually with confirmation
python scripts/clean_testing_workspace.py --interactive
```

## 🎯 Best Practices Summary

1. **Plan Before Testing** - Choose the right directory structure
2. **Use Descriptive Names** - `debug-ffmpeg-corruption-20240728` not `test1`
3. **Clean As You Go** - Don't let temporary files accumulate  
4. **Document Important Findings** - Move insights to `docs/`
5. **Respect the Root** - Keep it professional and clean

## 🔧 Tooling Support

### Makefile Targets
```bash
# Testing Commands
make test-fast           # Lightning-fast tests (<30s, development workflow)
make test-integration    # Integration tests (<5min, pre-commit validation)
make test-full           # Full test matrix (<30min, release validation)

# Workspace Management
make test-workspace      # Create proper test workspace structure
make clean-temp          # Clean temporary test files
make clean-testing-mess  # Emergency cleanup of root pollution
```

### VS Code/Cursor Integration
- Add `test-workspace/temp/` to `.gitignore`
- Configure workspace to suggest proper testing locations
- Use workspace snippets for testing directory creation

## 📝 Examples

### ✅ Good Testing Session
```bash
mkdir test-workspace/debug/gifski-memory-leak-20240728
cd test-workspace/debug/gifski-memory-leak-20240728
echo "Investigating gifski memory usage with large GIFs" > README.md
# ... run tests ...
# Document findings in docs/troubleshooting/
rm -rf ../gifski-memory-leak-20240728
```

### ❌ Bad Testing Practice  
```bash
# DON'T DO THIS - pollutes root directory
cd /
python -m giflab experiment sample.gif  # Creates files in root
mkdir debug_test                        # Clutters root
# ... leaves mess everywhere ...
```

# Pipeline Elimination Data for ML/Analysis

## Handling Null Values in Results

When using pipeline elimination results for machine learning or analysis, be aware of the data semantics:

### Parameter Fields Explained

**Applied parameter fields** (`applied_colors`, `applied_lossy`, `applied_frame_ratio`):
- **Successful pipelines**: Contain values ONLY if that processing step was actually applied
- **No-op steps**: Contain `None`/`NaN` (e.g., `none-color` step → `applied_colors: None`)
- **Failed pipelines**: Contain `None`/`NaN` (pipeline failed before completion)

**Pipeline step counts**:
- `pipeline_steps`: Total template steps (includes no-ops) - **avoid for ML**
- `actual_pipeline_steps`: Only actual processing steps - **use for ML**

### ❌ Wrong Approach
```python
# DON'T fill nulls with 0 - creates false patterns
df['applied_colors'].fillna(0)  # Implies "0 colors applied" vs "no color processing"
```

### ✅ Correct Approaches

**Option 1: Use applied parameter fields (Recommended)**
```python
# Use the applied_* fields - semantically correct, ML-ready
ml_features = [
    'applied_colors',         # Only has values when color reduction actually applied
    'applied_lossy',          # Only has values when lossy compression actually applied  
    'applied_frame_ratio',    # Only has values when frame reduction actually applied
    'actual_pipeline_steps',  # Count of actual processing steps (excludes no-ops)
    'success'                 # Success flag
]
X = results_df[ml_features]
```

**Option 2: Separate by actual processing type**
```python
# Analyze pipelines that actually used color reduction
color_pipelines = results_df[results_df['applied_colors'].notna()]
# Analyze pipelines that actually used lossy compression  
lossy_pipelines = results_df[results_df['applied_lossy'].notna()]
```

**Option 3: Feature engineering with processing flags**
```python
# Create boolean flags for what processing was actually applied
results_df['used_color_reduction'] = results_df['applied_colors'].notna()
results_df['used_frame_reduction'] = results_df['applied_frame_ratio'].notna()
results_df['used_lossy_compression'] = results_df['applied_lossy'].notna()

# Now you can analyze effectiveness of different processing combinations
processing_features = ['used_color_reduction', 'used_frame_reduction', 'used_lossy_compression']
```

## Data Types
The CSV loader uses proper nullable types:

**Parameter columns**:
- `applied_colors`: `Int64` (nullable integer) - actual applied color reduction
- `applied_lossy`: `Int64` (nullable integer) - actual applied lossy compression
- `applied_frame_ratio`: `float64` (nullable float) - actual applied frame reduction  

**Pipeline complexity**:
- `pipeline_steps`: `int` - total template steps (includes no-ops, for debugging)
- `actual_pipeline_steps`: `Int64` - count of real processing steps only (use for ML)

**Other fields**:
- `success`: `boolean` - pipeline completion status
- `error`: `string` - error messages for failed pipelines

This provides semantically correct, ML-ready features without confusing legacy columns.

---

*Following these practices ensures GifLab maintains a professional, navigable codebase that's easy for both humans and AI assistants to work with.* 