# GifLab Configuration Reference

## Overview

This document provides a comprehensive reference for all GifLab configuration options, particularly those related to Phase 3 performance optimizations. All configurations are controlled via environment variables.

## Environment Variables

### Core Performance Optimizations

#### `GIFLAB_USE_MODEL_CACHE`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enables singleton model caching to prevent duplicate model loading
- **Impact:** Reduces memory usage by ~50%, improves performance by 2x
- **Usage:** 
  ```bash
  export GIFLAB_USE_MODEL_CACHE=true
  ```

#### `GIFLAB_ENABLE_PARALLEL_METRICS`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enables parallel processing for frame-level metrics
- **Impact:** 10-50% speedup for large GIFs (100+ frames)
- **Usage:**
  ```bash
  export GIFLAB_ENABLE_PARALLEL_METRICS=true
  ```

#### `GIFLAB_ENABLE_CONDITIONAL_METRICS`
- **Type:** Boolean  
- **Default:** `true`
- **Description:** Enables intelligent metric skipping based on quality assessment
- **Impact:** 40-60% speedup for high-quality GIFs
- **Usage:**
  ```bash
  export GIFLAB_ENABLE_CONDITIONAL_METRICS=true
  ```

### Parallel Processing Configuration

#### `GIFLAB_MAX_PARALLEL_WORKERS`
- **Type:** Integer
- **Default:** CPU count
- **Range:** 1-32
- **Description:** Maximum number of parallel workers for metrics calculation
- **Recommendations:**
  - Small GIFs (<20 frames): 2
  - Medium GIFs (20-50 frames): 4
  - Large GIFs (100+ frames): 8+
- **Usage:**
  ```bash
  export GIFLAB_MAX_PARALLEL_WORKERS=4
  ```

#### `GIFLAB_CHUNK_STRATEGY`
- **Type:** String
- **Default:** `adaptive`
- **Options:** `adaptive`, `fixed`, `dynamic`
- **Description:** Strategy for distributing work across parallel workers
  - `adaptive`: Automatically adjusts based on frame count
  - `fixed`: Equal-sized chunks
  - `dynamic`: Load-balanced distribution
- **Usage:**
  ```bash
  export GIFLAB_CHUNK_STRATEGY=adaptive
  ```

#### `GIFLAB_PARALLEL_BATCH_SIZE`
- **Type:** Integer
- **Default:** 10
- **Range:** 1-100
- **Description:** Number of frames to process in each parallel batch
- **Usage:**
  ```bash
  export GIFLAB_PARALLEL_BATCH_SIZE=20
  ```

### Conditional Processing Configuration

#### `GIFLAB_QUALITY_HIGH_THRESHOLD`
- **Type:** Float
- **Default:** `0.9`
- **Range:** 0.0-1.0
- **Description:** Quality score above which expensive metrics are skipped
- **Note:** Lower values = more aggressive skipping
- **Usage:**
  ```bash
  export GIFLAB_QUALITY_HIGH_THRESHOLD=0.85
  ```

#### `GIFLAB_QUALITY_MEDIUM_THRESHOLD`
- **Type:** Float
- **Default:** `0.5`
- **Range:** 0.0-1.0
- **Description:** Quality score below which all metrics are calculated
- **Usage:**
  ```bash
  export GIFLAB_QUALITY_MEDIUM_THRESHOLD=0.5
  ```

#### `GIFLAB_QUALITY_SAMPLE_FRAMES`
- **Type:** Integer
- **Default:** `5`
- **Range:** 1-20
- **Description:** Number of frames to sample for quality assessment
- **Trade-off:** More frames = better assessment accuracy but slower
- **Usage:**
  ```bash
  export GIFLAB_QUALITY_SAMPLE_FRAMES=3
  ```

#### `GIFLAB_SKIP_EXPENSIVE_ON_HIGH_QUALITY`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Skip LPIPS, SSIMULACRA2, and other expensive metrics for high-quality GIFs
- **Usage:**
  ```bash
  export GIFLAB_SKIP_EXPENSIVE_ON_HIGH_QUALITY=true
  ```

#### `GIFLAB_USE_PROGRESSIVE_CALCULATION`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Calculate metrics progressively, stopping early if quality thresholds are met
- **Usage:**
  ```bash
  export GIFLAB_USE_PROGRESSIVE_CALCULATION=true
  ```

#### `GIFLAB_FORCE_ALL_METRICS`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Override conditional logic and calculate all metrics
- **Use Case:** Validation, testing, comprehensive analysis
- **Usage:**
  ```bash
  export GIFLAB_FORCE_ALL_METRICS=true
  ```

### Memory Management

#### `GIFLAB_CACHE_SIZE_MB`
- **Type:** Integer
- **Default:** `500`
- **Range:** 128-2048
- **Description:** Maximum memory allocated for model and frame caching (MB)
- **Usage:**
  ```bash
  export GIFLAB_CACHE_SIZE_MB=256
  ```

#### `GIFLAB_FORCE_MODEL_CLEANUP`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Force cleanup of cached models after each operation
- **Use Case:** Memory-constrained environments
- **Usage:**
  ```bash
  export GIFLAB_FORCE_MODEL_CLEANUP=true
  ```

#### `GIFLAB_MODEL_CACHE_TTL`
- **Type:** Integer
- **Default:** `3600`
- **Description:** Time-to-live for cached models in seconds
- **Usage:**
  ```bash
  export GIFLAB_MODEL_CACHE_TTL=1800
  ```

#### `GIFLAB_CLEANUP_INTERVAL_SECONDS`
- **Type:** Integer
- **Default:** `300`
- **Description:** Interval between automatic cache cleanup operations
- **Usage:**
  ```bash
  export GIFLAB_CLEANUP_INTERVAL_SECONDS=60
  ```

### Frame Processing

#### `GIFLAB_CACHE_FRAME_HASHES`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Cache frame hashes to avoid recalculation
- **Impact:** Speeds up repeated comparisons
- **Usage:**
  ```bash
  export GIFLAB_CACHE_FRAME_HASHES=true
  ```

#### `GIFLAB_FRAME_CACHE_SIZE`
- **Type:** Integer
- **Default:** `1000`
- **Description:** Maximum number of frame hashes to cache
- **Usage:**
  ```bash
  export GIFLAB_FRAME_CACHE_SIZE=500
  ```

#### `GIFLAB_ENABLE_MEMORY_MAPPING`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Use memory-mapped arrays for large frames
- **Use Case:** Very large GIFs or limited memory
- **Usage:**
  ```bash
  export GIFLAB_ENABLE_MEMORY_MAPPING=true
  ```

### Debugging and Monitoring

#### `GIFLAB_DEBUG`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable debug logging
- **Warning:** Significantly increases log volume
- **Usage:**
  ```bash
  export GIFLAB_DEBUG=true
  ```

#### `GIFLAB_LOG_LEVEL`
- **Type:** String
- **Default:** `INFO`
- **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Description:** Logging verbosity level
- **Usage:**
  ```bash
  export GIFLAB_LOG_LEVEL=DEBUG
  ```

#### `GIFLAB_ENABLE_PROFILING`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable performance profiling
- **Output:** Timing information for each operation
- **Usage:**
  ```bash
  export GIFLAB_ENABLE_PROFILING=true
  ```

#### `GIFLAB_LOG_PERFORMANCE_METRICS`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Log detailed performance metrics
- **Usage:**
  ```bash
  export GIFLAB_LOG_PERFORMANCE_METRICS=true
  ```

#### `GIFLAB_METRICS_OUTPUT_FILE`
- **Type:** String
- **Default:** None
- **Description:** File path for metrics output (JSON format)
- **Usage:**
  ```bash
  export GIFLAB_METRICS_OUTPUT_FILE=/tmp/giflab_metrics.json
  ```

#### `GIFLAB_LOG_MEMORY_USAGE`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Log memory usage statistics
- **Usage:**
  ```bash
  export GIFLAB_LOG_MEMORY_USAGE=true
  ```

#### `GIFLAB_LOG_CACHE_STATS`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Log cache hit/miss statistics
- **Usage:**
  ```bash
  export GIFLAB_LOG_CACHE_STATS=true
  ```

#### `GIFLAB_CACHE_STATS_INTERVAL`
- **Type:** Integer
- **Default:** `100`
- **Description:** Log cache statistics every N operations
- **Usage:**
  ```bash
  export GIFLAB_CACHE_STATS_INTERVAL=50
  ```

### Special Modes

#### `GIFLAB_DISABLE_ALL_OPTIMIZATIONS`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Disable all performance optimizations
- **Use Case:** Baseline testing, debugging
- **Usage:**
  ```bash
  export GIFLAB_DISABLE_ALL_OPTIMIZATIONS=true
  ```

#### `GIFLAB_RANDOM_SEED`
- **Type:** Integer
- **Default:** None (random)
- **Description:** Fixed random seed for reproducible results
- **Use Case:** Testing, debugging
- **Usage:**
  ```bash
  export GIFLAB_RANDOM_SEED=42
  ```

#### `GIFLAB_STRESS_TESTS`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable stress tests (requires explicit activation)
- **Usage:**
  ```bash
  export GIFLAB_STRESS_TESTS=1
  ```

### Deprecated Variables

These variables are deprecated and will be removed in future versions:

#### `GIFLAB_ENABLE_CACHING` (Deprecated)
- **Replaced by:** `GIFLAB_USE_MODEL_CACHE`
- **Migration:** Update scripts to use new variable name

#### `GIFLAB_PARALLEL_ENABLED` (Deprecated)
- **Replaced by:** `GIFLAB_ENABLE_PARALLEL_METRICS`
- **Migration:** Update scripts to use new variable name

## Configuration Profiles

### Production Profile
```bash
# Optimized for production performance
export GIFLAB_USE_MODEL_CACHE=true
export GIFLAB_ENABLE_PARALLEL_METRICS=true
export GIFLAB_MAX_PARALLEL_WORKERS=8
export GIFLAB_ENABLE_CONDITIONAL_METRICS=true
export GIFLAB_QUALITY_HIGH_THRESHOLD=0.9
export GIFLAB_CACHE_FRAME_HASHES=true
export GIFLAB_LOG_LEVEL=WARNING
```

### Development Profile
```bash
# Optimized for development and debugging
export GIFLAB_USE_MODEL_CACHE=true
export GIFLAB_ENABLE_PARALLEL_METRICS=false
export GIFLAB_ENABLE_CONDITIONAL_METRICS=true
export GIFLAB_DEBUG=true
export GIFLAB_LOG_LEVEL=DEBUG
export GIFLAB_ENABLE_PROFILING=true
```

### Testing Profile
```bash
# Optimized for consistent test results
export GIFLAB_USE_MODEL_CACHE=true
export GIFLAB_ENABLE_PARALLEL_METRICS=false
export GIFLAB_ENABLE_CONDITIONAL_METRICS=false
export GIFLAB_FORCE_ALL_METRICS=true
export GIFLAB_RANDOM_SEED=42
```

### Memory-Constrained Profile
```bash
# Optimized for low memory environments
export GIFLAB_USE_MODEL_CACHE=true
export GIFLAB_ENABLE_PARALLEL_METRICS=false
export GIFLAB_MAX_PARALLEL_WORKERS=2
export GIFLAB_ENABLE_CONDITIONAL_METRICS=true
export GIFLAB_CACHE_SIZE_MB=256
export GIFLAB_FORCE_MODEL_CLEANUP=true
```

## Configuration Validation

### Check Current Configuration
```python
#!/usr/bin/env python
"""Check current GifLab configuration"""

import os
import sys

# Define all configuration variables
CONFIG_VARS = [
    'GIFLAB_USE_MODEL_CACHE',
    'GIFLAB_ENABLE_PARALLEL_METRICS',
    'GIFLAB_ENABLE_CONDITIONAL_METRICS',
    'GIFLAB_MAX_PARALLEL_WORKERS',
    'GIFLAB_QUALITY_HIGH_THRESHOLD',
    'GIFLAB_QUALITY_MEDIUM_THRESHOLD',
    'GIFLAB_DEBUG',
]

print("GifLab Configuration:")
print("-" * 50)

for var in CONFIG_VARS:
    value = os.environ.get(var, "Not set (using default)")
    print(f"{var}: {value}")
```

### Validate Configuration
```python
#!/usr/bin/env python
"""Validate GifLab configuration"""

import os

def validate_config():
    errors = []
    warnings = []
    
    # Check numeric ranges
    workers = os.environ.get('GIFLAB_MAX_PARALLEL_WORKERS')
    if workers:
        try:
            w = int(workers)
            if w < 1 or w > 32:
                errors.append(f"GIFLAB_MAX_PARALLEL_WORKERS must be 1-32, got {w}")
        except ValueError:
            errors.append(f"GIFLAB_MAX_PARALLEL_WORKERS must be integer, got {workers}")
    
    # Check thresholds
    high_thresh = os.environ.get('GIFLAB_QUALITY_HIGH_THRESHOLD')
    if high_thresh:
        try:
            h = float(high_thresh)
            if h < 0 or h > 1:
                errors.append(f"GIFLAB_QUALITY_HIGH_THRESHOLD must be 0-1, got {h}")
        except ValueError:
            errors.append(f"GIFLAB_QUALITY_HIGH_THRESHOLD must be float, got {high_thresh}")
    
    # Check conflicts
    if os.environ.get('GIFLAB_DISABLE_ALL_OPTIMIZATIONS') == 'true':
        if os.environ.get('GIFLAB_ENABLE_PARALLEL_METRICS') == 'true':
            warnings.append("GIFLAB_ENABLE_PARALLEL_METRICS ignored when GIFLAB_DISABLE_ALL_OPTIMIZATIONS is true")
    
    return errors, warnings

errors, warnings = validate_config()

if errors:
    print("Configuration Errors:")
    for error in errors:
        print(f"  ✗ {error}")
    sys.exit(1)

if warnings:
    print("Configuration Warnings:")
    for warning in warnings:
        print(f"  ⚠ {warning}")

print("✓ Configuration valid")
```

## Best Practices

### 1. Start with Defaults
The default configuration is optimized for most use cases. Only adjust when needed.

### 2. Profile Before Tuning
Always measure baseline performance before adjusting configuration.

### 3. Change One Variable at a Time
When tuning, modify one variable at a time to understand its impact.

### 4. Document Custom Configurations
Always document why custom configurations were chosen.

### 5. Use Configuration Management
Store configurations in version control:
```bash
# .env.production
GIFLAB_USE_MODEL_CACHE=true
GIFLAB_ENABLE_PARALLEL_METRICS=true
GIFLAB_MAX_PARALLEL_WORKERS=8
```

### 6. Monitor After Changes
Always monitor performance metrics after configuration changes.

## Troubleshooting

### Configuration Not Taking Effect
1. Check variable spelling (case-sensitive)
2. Verify variable is exported: `export VAR=value`
3. Check for overrides in code
4. Restart application after changes

### Performance Not Improving
1. Enable profiling: `export GIFLAB_ENABLE_PROFILING=true`
2. Check optimization status in logs
3. Verify parallel workers are being used
4. Check if conditional metrics are being triggered

### Memory Issues
1. Reduce parallel workers
2. Enable forced cleanup
3. Reduce cache sizes
4. Enable memory mapping for large frames

## Version Compatibility

| Configuration Variable | Introduced | Deprecated | Removed |
|------------------------|------------|------------|---------|
| GIFLAB_USE_MODEL_CACHE | v2.0.0 | - | - |
| GIFLAB_ENABLE_PARALLEL_METRICS | v2.0.0 | - | - |
| GIFLAB_ENABLE_CONDITIONAL_METRICS | v2.0.0 | - | - |
| GIFLAB_ENABLE_CACHING | v1.0.0 | v2.0.0 | v3.0.0 |

## Related Documentation

- [Performance Tuning Guide](../guides/performance-tuning-guide.md)
- [Migration Guide](../guides/migration-guide.md)
- [Monitoring Setup](../technical/monitoring-setup.md)

---

*Last updated: 2025-09-09*
*Version: 1.0.0*
*For GifLab v2.0.0+ with Phase 3 Optimizations*