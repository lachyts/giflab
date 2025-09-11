# GifLab Performance Monitoring Runbook

## Overview

This runbook provides operational guidance for the GifLab performance monitoring infrastructure implemented as part of Phase 5.2 of the validation performance optimization project.

## Architecture

### Components

1. **MetricsCollector**: Core metrics aggregation and collection
2. **Backends**: Storage backends (SQLite, In-Memory, StatsD)
3. **Decorators**: Easy instrumentation for code
4. **Integration**: Automatic instrumentation of cache systems
5. **Alerting**: Threshold monitoring and alert generation
6. **CLI**: Command-line interface for metrics viewing

### Monitored Systems

- **FrameCache**: 1700x+ speedup for cached frame extractions
- **ValidationCache**: 50-80% speedup for repeated validations
- **ResizedFrameCache**: 15-25% speedup for multi-metric validation
- **Frame Sampling**: 30-70% speedup for large GIFs
- **Lazy Imports**: Module loading optimization

## Quick Start

### Enable Monitoring

```python
# In your code
from giflab.monitoring import instrument_all_systems

# Enable instrumentation at startup
instrument_all_systems()
```

### View Metrics

```bash
# Current status
giflab metrics status

# Live monitoring dashboard
giflab metrics monitor

# Check for alerts
giflab metrics alerts

# Export metrics
giflab metrics export -o metrics.json --start "-1h"
```

## Configuration

### config.py Settings

```python
MONITORING = {
    "enabled": True,  # Enable/disable monitoring
    "backend": "sqlite",  # "memory", "sqlite", or "statsd"
    "buffer_size": 10000,  # In-memory buffer size
    "flush_interval": 10.0,  # Seconds between flushes
    "sampling_rate": 1.0,  # 1.0 = collect all metrics
    
    "sqlite": {
        "db_path": None,  # None = ~/.giflab_cache/metrics.db
        "retention_days": 7.0,
        "max_size_mb": 100.0,
    },
    
    "alerts": {
        "cache_hit_rate_warning": 0.4,  # 40%
        "cache_hit_rate_critical": 0.2,  # 20%
        "memory_usage_warning": 0.8,  # 80%
        "memory_usage_critical": 0.95,  # 95%
    },
}
```

## Metrics Reference

### Cache Metrics

| Metric | Type | Description | Normal Range |
|--------|------|-------------|--------------|
| `cache.frame.hits` | Counter | Frame cache hits | - |
| `cache.frame.misses` | Counter | Frame cache misses | - |
| `cache.frame.hit_rate` | Calculated | Hit rate percentage | 60-80% |
| `cache.frame.memory_usage_mb` | Gauge | Memory usage in MB | < 400MB |
| `cache.frame.operation.duration` | Timer | Operation latency | < 1ms (hit) |
| `cache.frame.evictions` | Counter | Cache evictions | < 10/min |

### Validation Cache Metrics

| Metric | Type | Description | Normal Range |
|--------|------|-------------|--------------|
| `cache.validation.hits` | Counter | Validation cache hits | - |
| `cache.validation.misses` | Counter | Validation cache misses | - |
| `cache.validation.hit_rate` | Calculated | Hit rate by metric type | 40-60% |
| `cache.validation.memory_usage_mb` | Gauge | Memory usage | < 80MB |

### Sampling Metrics

| Metric | Type | Description | Normal Range |
|--------|------|-------------|--------------|
| `sampling.frames_sampled_ratio` | Gauge | Sampling rate | 0.2-0.8 |
| `sampling.confidence_interval_width` | Gauge | CI width | < 0.1 |
| `sampling.speedup_factor` | Gauge | Performance gain | 1.5-5x |
| `sampling.strategy_usage` | Counter | Strategy distribution | - |

### Module Loading Metrics

| Metric | Type | Description | Normal Range |
|--------|------|-------------|--------------|
| `lazy_import.load_time` | Timer | Module load time | < 100ms |
| `lazy_import.load_count` | Counter | Load frequency | - |
| `lazy_import.fallback_used` | Counter | Fallback usage | 0 |

## Alert Response Procedures

### Low Cache Hit Rate Alert

**Symptoms**: Cache hit rate below 40% (warning) or 20% (critical)

**Investigation Steps**:
1. Check recent changes to codebase
2. Verify file modifications aren't invalidating cache
3. Check for memory pressure causing evictions
4. Review access patterns

**Resolution**:
```bash
# Check cache statistics
giflab cache status

# Warm cache with common files
giflab cache warm data/raw/*.gif

# Increase cache size if needed
# Edit config.py: FRAME_CACHE["memory_limit_mb"] = 1000
```

### High Memory Usage Alert

**Symptoms**: Memory usage above 80% (warning) or 95% (critical)

**Investigation Steps**:
1. Check current memory allocation
2. Review recent processing patterns
3. Look for memory leaks

**Resolution**:
```bash
# View current memory usage
giflab metrics status --system all

# Clear caches if necessary
giflab cache clear --all

# Reduce cache sizes in config.py if persistent
```

### Eviction Rate Spike Alert

**Symptoms**: Eviction rate > 3x normal

**Investigation Steps**:
1. Check for unusual access patterns
2. Verify cache size configuration
3. Look for large file processing

**Resolution**:
```bash
# Monitor eviction patterns
giflab metrics monitor --interval 1

# Adjust cache strategy
# Consider increasing memory limits or changing eviction policy
```

### Response Time Degradation Alert

**Symptoms**: Operation latency > 1.5x baseline

**Investigation Steps**:
1. Check system load
2. Verify disk I/O performance
3. Review recent code changes

**Resolution**:
```bash
# Profile slow operations
giflab metrics status --window 60

# Check for lock contention or I/O bottlenecks
```

## Performance Tuning

### Optimal Cache Sizes

Based on workload analysis:

```python
# For high-throughput scenarios
FRAME_CACHE = {
    "memory_limit_mb": 1000,  # Increase for large datasets
    "disk_limit_mb": 5000,
}

# For memory-constrained environments
FRAME_CACHE = {
    "memory_limit_mb": 200,  # Reduce memory footprint
    "disk_limit_mb": 2000,
}
```

### Sampling Strategy Selection

```python
# For fast preview (30-50% speedup)
FRAME_SAMPLING = {
    "default_strategy": "uniform",
    "uniform": {"sampling_rate": 0.5},
}

# For accuracy-focused (maintain >95% accuracy)
FRAME_SAMPLING = {
    "default_strategy": "progressive",
    "progressive": {"target_ci_width": 0.05},
}

# For motion-heavy content
FRAME_SAMPLING = {
    "default_strategy": "adaptive",
    "adaptive": {"base_rate": 0.2, "max_rate": 0.9},
}
```

## Grafana Dashboard Setup

### Import Dashboard

1. Access Grafana UI
2. Navigate to Dashboards → Import
3. Upload `src/giflab/monitoring/dashboards/giflab_performance.json`
4. Select data source (Prometheus/InfluxDB)
5. Configure refresh interval (10s recommended)

### Key Panels

- **Cache Hit Rates**: Real-time hit rate gauges
- **Operation Latency**: P95 response times
- **Memory Usage**: Cache memory consumption
- **Sampling Efficiency**: Speedup vs accuracy tradeoff
- **Alert Status**: Active alerts and thresholds

## Troubleshooting

### Metrics Not Appearing

```bash
# Verify monitoring is enabled
poetry run python -c "from giflab.config import MONITORING; print(MONITORING['enabled'])"

# Check instrumentation is active
poetry run python -c "from giflab.monitoring import instrument_all_systems; instrument_all_systems()"

# View backend statistics
giflab metrics backend-stats
```

### High Memory Usage

```bash
# Reduce buffer size
# Edit config.py: MONITORING["buffer_size"] = 5000

# Increase flush frequency
# Edit config.py: MONITORING["flush_interval"] = 5.0

# Switch to SQLite backend for persistence
# Edit config.py: MONITORING["backend"] = "sqlite"
```

### Database Growth

```bash
# Check database size
du -h ~/.giflab_cache/metrics.db

# Reduce retention period
# Edit config.py: MONITORING["sqlite"]["retention_days"] = 3.0

# Manual cleanup
sqlite3 ~/.giflab_cache/metrics.db "DELETE FROM metrics WHERE timestamp < strftime('%s', 'now', '-7 days');"
sqlite3 ~/.giflab_cache/metrics.db "VACUUM;"
```

## Maintenance Tasks

### Daily
- Review alert summary: `giflab metrics alerts`
- Check cache hit rates: `giflab metrics status`

### Weekly
- Export metrics for analysis: `giflab metrics export -o weekly_metrics.json`
- Review performance trends in Grafana
- Clean up old metrics if needed

### Monthly
- Review and adjust alert thresholds based on patterns
- Optimize cache sizes based on usage statistics
- Update sampling strategies for new workload patterns

## Integration with CI/CD

### Pre-deployment Checks

```yaml
# .github/workflows/performance.yml
- name: Check Performance Metrics
  run: |
    poetry run giflab metrics alerts --check all
    if [ $? -ne 0 ]; then
      echo "Performance alerts detected"
      exit 1
    fi
```

### Post-deployment Monitoring

```bash
# Monitor for 5 minutes after deployment
giflab metrics monitor --duration 300 --interval 10

# Export baseline metrics
giflab metrics export -o baseline.json

# Set up alerts for regression
giflab metrics alerts --window 600
```

## Custom Instrumentation

### Adding Metrics to New Code

```python
from giflab.monitoring import track_timing, track_counter, MetricTracker

@track_timing(metric_name="custom.operation.duration")
@track_counter(metric_name="custom.operation.calls")
def my_operation():
    tracker = MetricTracker("custom.operation")
    
    with tracker.timer("preprocessing"):
        # Preprocessing code
        pass
    
    tracker.counter("items_processed", 10)
    tracker.gauge("queue_size", 5)
    
    return result
```

### Creating Custom Alerts

```python
from giflab.monitoring.alerting import AlertRule, get_alert_manager

# Add custom rule
manager = get_alert_manager()
manager.add_rule(AlertRule(
    name="custom.metric",
    metric_pattern="custom.operation.duration",
    warning_threshold=1.0,  # 1 second
    critical_threshold=5.0,  # 5 seconds
    message_template="Custom operation slow: {value:.2f}s",
))
```

## Emergency Procedures

### Disable Monitoring (Performance Impact)

```python
# config.py
MONITORING = {
    "enabled": False,  # Disable all monitoring
}
```

### Clear All Metrics (Space Issues)

```bash
# Clear all metrics data
giflab metrics clear

# Remove metrics database
rm ~/.giflab_cache/metrics.db
```

### Reset Cache Systems

```bash
# Clear all caches
giflab cache clear --all
giflab cache resize-clear
giflab cache validation-clear

# Restart with fresh state
poetry run python -m giflab run --preset quick-test
```

## Support and Escalation

### Log Locations
- Metrics database: `~/.giflab_cache/metrics.db`
- Application logs: `logs/giflab.log`
- Cache directories: `~/.giflab_cache/`

### Debug Commands
```bash
# Verbose monitoring output
GIFLAB_LOG_LEVEL=DEBUG giflab metrics status

# Check specific system
giflab metrics status --system frame -w 60

# Export full metrics for analysis
giflab metrics export --start "-24h" -o debug_metrics.json
```

### Performance Baseline

Expected performance with monitoring enabled:
- Overhead: < 1% of request time
- Memory: < 50MB for monitoring infrastructure
- Disk: < 100MB for 7 days of metrics

---

*Last Updated: 2025-01-10*
*Version: 1.0.0*
*Part of GifLab Validation Performance Optimization Phase 5.2*