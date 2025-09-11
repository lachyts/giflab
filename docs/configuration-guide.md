# GifLab Configuration Guide

## Overview

The GifLab configuration management system provides a hierarchical, validated, and dynamically reloadable configuration framework optimized for different deployment scenarios. This guide covers configuration profiles, tuning guidelines, and best practices for production deployments.

## Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Configuration Profiles](#configuration-profiles)
3. [Dynamic Configuration](#dynamic-configuration)
4. [Tuning Guidelines](#tuning-guidelines)
5. [Validation Framework](#validation-framework)
6. [Monitoring and Alerts](#monitoring-and-alerts)
7. [Troubleshooting](#troubleshooting)

## Configuration Architecture

### Hierarchy

The configuration system follows a hierarchical override pattern:

1. **Defaults** - Base configuration from `config.py`
2. **Profiles** - Environment-specific overrides
3. **Environment Variables** - Runtime overrides
4. **API Overrides** - Programmatic changes

Each level overrides the previous, allowing fine-grained control.

### Components

- **ConfigManager** - Central configuration management singleton
- **ConfigValidator** - Type and range validation framework
- **ConfigProfiles** - Pre-defined environment configurations
- **ConfigFileWatcher** - Automatic reload on file changes

## Configuration Profiles

### Available Profiles

#### Development (`development`)
**Use Case:** Local development and debugging
- Aggressive caching (1GB memory, 5GB disk)
- Full frame processing (no sampling)
- Verbose logging and monitoring
- All metrics enabled for accuracy verification

```python
from giflab.config_manager import load_profile
load_profile("development")
```

#### Production (`production`)
**Use Case:** Stable production deployments
- Balanced caching (500MB memory, 2GB disk)
- Adaptive frame sampling for efficiency
- Lightweight monitoring (10% sampling)
- Optimized metrics for speed/accuracy balance

```python
load_profile("production")
```

#### High Memory (`high_memory`)
**Use Case:** Memory-rich environments (16GB+ RAM)
- Maximum caching (2GB memory, 10GB disk)
- No frame sampling - full processing
- Comprehensive metrics calculation
- Extended TTLs for better hit rates

```python
load_profile("high_memory")
```

#### Low Memory (`low_memory`)
**Use Case:** Constrained environments (<4GB RAM)
- Minimal caching (100MB memory, 500MB disk)
- Aggressive frame sampling (20% only)
- Basic metrics only
- Memory-only monitoring (no persistence)

```python
load_profile("low_memory")
```

#### High Throughput (`high_throughput`)
**Use Case:** Batch processing pipelines
- Optimized for speed over accuracy
- Progressive sampling strategy
- Monitoring disabled for overhead reduction
- Basic metrics only

```python
load_profile("high_throughput")
```

#### Interactive (`interactive`)
**Use Case:** Real-time preview generation
- Small, fast caches (300MB memory only)
- Minimal frame sampling (10%)
- Fastest metrics modes
- No monitoring overhead

```python
load_profile("interactive")
```

#### Testing (`testing`)
**Use Case:** Unit tests and CI/CD
- All caching disabled for reproducibility
- Strict validation enabled
- Deterministic settings
- Temporary directories

```python
load_profile("testing")
```

### Profile Selection Decision Tree

```
Start
  │
  ├─ Is this for testing? → Use "testing"
  │
  ├─ Is real-time response critical? → Use "interactive"
  │
  ├─ Is this batch processing?
  │   └─ Yes → Use "high_throughput"
  │
  ├─ Check available memory:
  │   ├─ < 4GB → Use "low_memory"
  │   ├─ > 16GB → Use "high_memory"
  │   └─ 4-16GB → Continue
  │
  └─ Default → Use "production"
```

## Dynamic Configuration

### Environment Variables

Override configuration using environment variables:

```bash
# Format: GIFLAB_CONFIG_<SECTION>_<KEY>=value
export GIFLAB_CONFIG_FRAME_CACHE_MEMORY_LIMIT_MB=1000
export GIFLAB_CONFIG_MONITORING_ENABLED=false
export GIFLAB_CONFIG_FRAME_SAMPLING_DEFAULT_STRATEGY=uniform
```

### Programmatic Updates

```python
from giflab.config_manager import get_config_manager

manager = get_config_manager()

# Get configuration value
cache_size = manager.get("FRAME_CACHE.memory_limit_mb")

# Set configuration value with validation
manager.set("FRAME_CACHE.memory_limit_mb", 1500, validate=True)

# Register change callback
def on_config_change(change):
    print(f"Config changed: {change.path} = {change.new_value}")

manager.register_change_callback(on_config_change)
```

### File Watching

Enable automatic reload on configuration file changes:

```python
from pathlib import Path
from giflab.config_manager import get_config_manager

manager = get_config_manager()

# Watch configuration directory
config_dir = Path("./config")
manager.start_file_watcher([config_dir])

# Stop watching
manager.stop_file_watcher()
```

### Signal-Based Reload

Send SIGHUP to reload configuration:

```bash
# Find GifLab process
ps aux | grep giflab

# Send reload signal
kill -HUP <pid>
```

## Tuning Guidelines

### Cache Tuning

#### Memory Cache Size
- **Formula:** Available RAM × 0.25 (conservative) to 0.5 (aggressive)
- **Example:** 8GB RAM → 2-4GB total cache
- **Distribution:** 60% frame cache, 30% validation cache, 10% resize cache

#### Disk Cache Size
- **Formula:** Available disk × 0.1 (conservative) to 0.2 (aggressive)
- **Minimum:** 2× memory cache size
- **Location:** Fast SSD preferred, separate from system disk

#### TTL Settings
- **Development:** 1 hour (rapid iteration)
- **Production:** 24-48 hours (stability)
- **High Traffic:** 6-12 hours (balance freshness/performance)

### Frame Sampling Tuning

#### Sampling Strategy Selection
- **Uniform:** Consistent motion, simple animations
- **Adaptive:** Variable motion, complex scenes
- **Progressive:** Unknown content, need confidence intervals
- **Scene-Aware:** Multi-scene content, transitions important

#### Sampling Rate Optimization
```python
# Based on GIF characteristics
if frame_count < 30:
    sampling_enabled = False  # Process all
elif frame_count < 100:
    sampling_rate = 0.5  # 50%
elif frame_count < 500:
    sampling_rate = 0.3  # 30%
else:
    sampling_rate = 0.2  # 20%
```

### Monitoring Tuning

#### Buffer Size
- **High Volume:** 50,000-100,000 events
- **Normal:** 10,000 events
- **Low Memory:** 1,000 events

#### Sampling Rate
- **Development:** 1.0 (100% - full visibility)
- **Production:** 0.1 (10% - reduce overhead)
- **High Load:** 0.05 (5% - minimal impact)

#### Alert Thresholds
```python
# Adjust based on workload patterns
alerts = {
    "cache_hit_rate_warning": 0.4,    # Expect 40%+ hits
    "cache_hit_rate_critical": 0.2,   # Alert below 20%
    "memory_usage_warning": 0.8,      # Warn at 80%
    "memory_usage_critical": 0.95,    # Critical at 95%
}
```

### Performance Optimization Matrix

| Workload Type | Cache Memory | Sampling | Monitoring | Metrics Mode |
|--------------|--------------|----------|------------|--------------|
| Development | High (1GB+) | Disabled | Full | Comprehensive |
| Production | Moderate (500MB) | Adaptive | Sampled | Optimized |
| Batch Processing | High (1GB+) | Progressive | Disabled | Fast |
| Interactive | Low (300MB) | Aggressive | Disabled | Fast |
| Memory Constrained | Minimal (100MB) | Uniform 20% | Memory-only | Fast |

## Validation Framework

### Pre-Flight Validation

Run validation before deployment:

```bash
# Validate current configuration
python scripts/validate_config.py current -v

# Validate specific profile
python scripts/validate_config.py profile production

# Compare profiles
python scripts/validate_config.py compare production high_memory
```

### Validation Rules

#### Type Validation
- Integer ranges (memory limits, TTLs)
- Float percentages (sampling rates, thresholds)
- Boolean flags (enabled/disabled)
- String enums (strategies, backends)

#### Resource Validation
- Memory availability checks
- Disk space verification
- Executable path validation
- Directory write permissions

#### Relationship Validation
- Total cache memory limits
- Metric weight summation
- Sampling configuration consistency
- Cache path uniqueness

### Custom Validators

Add application-specific validators:

```python
from giflab.config_manager import get_config_manager

manager = get_config_manager()
validator = manager._validator

# Add custom rule
def validate_gpu_memory(value):
    import torch
    if torch.cuda.is_available():
        available = torch.cuda.get_device_properties(0).total_memory
        return value <= available * 0.8
    return True

validator.add_validator(
    "metrics.LPIPS_BATCH_SIZE",
    validate_gpu_memory,
    "Batch size exceeds GPU memory"
)
```

## Monitoring and Alerts

### Key Metrics to Monitor

#### Cache Performance
- **Hit Rate:** Target >60% for frames, >40% for validation
- **Eviction Rate:** Should be <10% of insertions
- **Memory Usage:** Stay below 80% of limit
- **TTL Expiration:** Monitor for premature expiration

#### Processing Performance
- **Frame Extraction Time:** Baseline vs current
- **Validation Time:** Per-metric breakdowns
- **Sampling Accuracy:** Confidence intervals
- **Queue Depth:** For batch processing

### Alert Response Procedures

#### Low Cache Hit Rate (<20%)
1. Check TTL settings - may be too short
2. Verify cache size - may need increase
3. Analyze access patterns - may need different strategy
4. Check for cache invalidation bugs

#### High Memory Usage (>95%)
1. Reduce cache sizes immediately
2. Enable more aggressive sampling
3. Decrease batch sizes
4. Consider switching to low_memory profile

#### Performance Degradation (>1.5x baseline)
1. Check monitoring overhead - reduce sampling
2. Verify cache health - clear if corrupted
3. Review recent configuration changes
4. Check system resources (CPU, I/O)

## Troubleshooting

### Common Issues

#### Configuration Not Loading
```python
# Check for validation errors
from giflab.config_manager import get_config_manager

manager = get_config_manager()
metadata = manager.get_metadata()
print(f"Validation errors: {metadata.validation_errors}")
```

#### Cache Not Working
```bash
# Check cache statistics
giflab cache status

# Verify cache directory permissions
ls -la ~/.giflab_cache

# Clear corrupted cache
giflab cache clear --all
```

#### Memory Issues
```python
# Switch to low memory profile
from giflab.config_manager import load_profile
load_profile("low_memory")

# Or adjust specific settings
manager.set("FRAME_CACHE.memory_limit_mb", 100)
manager.set("FRAME_SAMPLING.enabled", True)
```

### Debug Commands

```bash
# Export current configuration
python scripts/validate_config.py export current_config.json

# Validate exported configuration
python scripts/validate_config.py file current_config.json -v

# List available profiles
python scripts/validate_config.py list

# Check configuration in Python
python -c "
from giflab.config_manager import get_config_manager
import json
m = get_config_manager()
print(json.dumps(m.config, indent=2))
"
```

### Performance Profiling

```python
# Profile configuration impact
import time
from giflab.config_manager import load_profile

# Baseline with minimal config
load_profile("interactive")
start = time.time()
# Run your workload
baseline_time = time.time() - start

# Test with different profile
load_profile("production")
start = time.time()
# Run same workload
production_time = time.time() - start

print(f"Performance impact: {production_time / baseline_time:.2f}x")
```

## Best Practices

1. **Start Conservative:** Begin with production profile and tune up
2. **Monitor First:** Collect baseline metrics before optimization
3. **Change One Thing:** Adjust one parameter at a time
4. **Document Changes:** Keep configuration changelog
5. **Test Thoroughly:** Validate changes under load
6. **Plan Rollback:** Export configuration before major changes
7. **Use Profiles:** Avoid manual tweaking in production

## Configuration Changelog Template

```markdown
## [Date] - Configuration Update

### Changed
- Increased FRAME_CACHE.memory_limit_mb from 500 to 750
- Enabled FRAME_SAMPLING with adaptive strategy

### Reason
- Cache hit rate was below 40%
- Large GIFs causing memory pressure

### Impact
- Cache hit rate improved to 65%
- Memory usage stable at 70%
- Processing time reduced by 30%

### Rollback
- Revert to profile: production_v1_backup
```

---

*Last Updated: January 2025*
*Version: 1.0.0*