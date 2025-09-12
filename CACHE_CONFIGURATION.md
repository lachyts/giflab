# GifLab Cache Configuration Guide

## Overview

GifLab uses multiple caching systems to improve performance. Understanding their configuration is crucial for both development and testing.

## Cache Types and Default States

### 1. Frame Cache (DISABLED by default)

**Purpose**: Caches extracted GIF frames to avoid repeated I/O operations
**Default State**: `DISABLED` 
**Configuration**: `FRAME_CACHE` in `src/giflab/config.py`

```python
ENABLE_EXPERIMENTAL_CACHING = False  # Master switch
FRAME_CACHE = {
    "enabled": ENABLE_EXPERIMENTAL_CACHING,  # False by default
    "memory_limit_mb": 500,
    "disk_limit_mb": 2000,
    "ttl_seconds": 86400,  # 24 hours
}
```

**To Enable**: Set environment variable or modify config
```bash
# Option 1: Environment variable (if implemented)
export GIFLAB_ENABLE_EXPERIMENTAL_CACHING=true

# Option 2: Modify config.py
ENABLE_EXPERIMENTAL_CACHING = True
```

### 2. Validation Cache (ENABLED by default)

**Purpose**: Caches metric calculation results (SSIM, LPIPS, etc.)
**Default State**: `ENABLED`
**Configuration**: `VALIDATION_CACHE` in `src/giflab/config.py`

```python
VALIDATION_CACHE = {
    "enabled": True,  # Enabled by default
    "memory_limit_mb": 100,
    "disk_limit_mb": 1000,
    "ttl_seconds": 172800,  # 48 hours
    "cache_ssim": True,
    "cache_ms_ssim": True,
    "cache_lpips": True,
    "cache_gradient_color": True,
    "cache_ssimulacra2": True,
}
```

## Testing Implications

### Frame Cache Tests
Frame cache integration tests use the `enable_cache` fixture:

```python
@pytest.fixture
def enable_cache(monkeypatch):
    """Enable frame cache for testing."""
    monkeypatch.setitem(FRAME_CACHE, "enabled", True)
    monkeypatch.setitem(FRAME_CACHE, "memory_limit_mb", 100)
    monkeypatch.setitem(FRAME_CACHE, "ttl_seconds", 3600)
```

**Expected Behavior**: 
- Tests should show cache misses/hits when `enable_cache` fixture is used
- Tests should show zero activity when `disable_cache` fixture is used

### Validation Cache Tests
Validation cache tests patch the import path:

```python
with patch("giflab.caching.validation_cache.VALIDATION_CACHE", cache_config):
```

**Issue**: `VALIDATION_CACHE` is defined in `giflab.config`, not `giflab.caching.validation_cache`
**Correct Path**: Should patch `giflab.config.VALIDATION_CACHE`

## Performance Impact

### When Caches Are Disabled
- **Frame Cache Disabled**: Each frame extraction reads from disk
- **Validation Cache Disabled**: Each metric calculation is recomputed
- **Performance Impact**: 10-100x slower operations

### When Caches Are Enabled  
- **Frame Cache**: ~100x faster repeated frame access
- **Validation Cache**: ~10x faster repeated metric calculations
- **Memory Usage**: Higher memory consumption
- **Disk Usage**: Cache files in `~/.giflab_cache/`

## Configuration Validation

The system validates cache configurations at startup:

```python
# config_validator.py checks:
- Memory limits are positive
- Disk paths are accessible
- TTL values are reasonable
- Total cache memory doesn't exceed system limits
```

## Troubleshooting

### Frame Cache Not Working
1. **Check if enabled**: `ENABLE_EXPERIMENTAL_CACHING = True`
2. **Check memory limits**: Ensure sufficient `memory_limit_mb`
3. **Check disk space**: Ensure sufficient disk space for cache
4. **Check permissions**: Cache directory must be writable

### Validation Cache Not Working  
1. **Check enabled flag**: `VALIDATION_CACHE["enabled"] = True`
2. **Check specific metric flags**: e.g., `cache_ssim`, `cache_lpips`
3. **Check TTL**: Ensure cache entries haven't expired
4. **Check memory pressure**: System may be evicting cache entries

### Test Failures
1. **Frame cache tests fail**: Usually means cache is disabled by default
2. **Validation cache tests fail**: Usually wrong import path in patches
3. **Performance tests fail**: Cache might not be properly initialized

## Best Practices

### Development
- **Enable frame cache** for faster local development
- **Keep validation cache enabled** for consistent performance
- **Monitor memory usage** with large datasets

### Production
- **Consider enabling frame cache** for repeated processing
- **Keep validation cache enabled** for better throughput
- **Set appropriate TTL values** based on data update frequency

### Testing
- **Use fixtures** to control cache state in tests
- **Reset caches** between tests to avoid interference
- **Patch correct import paths** for mocking cache behavior

## Related Files

- `src/giflab/config.py` - Main configuration
- `src/giflab/caching/frame_cache.py` - Frame cache implementation  
- `src/giflab/caching/validation_cache.py` - Validation cache implementation
- `tests/integration/test_frame_cache_integration.py` - Frame cache tests
- `tests/integration/test_validation_cache_metrics_integration.py` - Validation cache tests