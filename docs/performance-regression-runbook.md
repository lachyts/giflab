# Performance Regression Detection Runbook

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Daily Operations](#daily-operations)
5. [Incident Response](#incident-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [A/B Testing Operations](#ab-testing-operations)

## Overview

The GifLab Performance Regression Detection System prevents performance-degrading changes from entering production through automated benchmarking, CI/CD gates, and continuous monitoring.

### Key Components
- **Benchmark Suite**: Standardized performance tests across different GIF sizes
- **Regression Detector**: Compares results against established baselines
- **CI/CD Integration**: Automated gates in GitHub Actions
- **Trend Analysis**: Weekly performance reports
- **A/B Testing**: Production configuration comparison

### Success Metrics
- Zero undetected regressions reaching production
- <5 minute CI runtime overhead
- 95% confidence in performance measurements
- <2% false positive rate

## Quick Start

### Running Performance Tests Locally

```bash
# Run quick CI benchmark suite
poetry run python -m giflab.benchmarks.regression_suite test --tags ci

# Establish new baselines
poetry run python -m giflab.benchmarks.regression_suite baseline --tags ci

# Run comprehensive benchmarks
poetry run python -m giflab.benchmarks.regression_suite test --tags comprehensive

# Compare against baselines
poetry run python -m giflab.benchmarks.regression_suite compare
```

### Viewing Results

Results are saved to `benchmark_results/` directory:
```bash
# View latest report
cat benchmark_results/ci_report_*.json | jq .

# Generate HTML report
python scripts/generate_performance_report.py \
  --report-file benchmark_results/ci_report_latest.json \
  --output-format html > report.html
```

## Architecture

### Performance Metrics Tracked

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `frame_cache_hit_rate` | Percentage of frame extractions from cache | >60% expected |
| `validation_cache_hit_rate` | Percentage of validation results from cache | >40% expected |
| `total_validation_time_ms` | End-to-end validation time | <10% variance |
| `memory_usage_peak_mb` | Maximum memory consumption | <20% increase |
| `sampling_speedup_factor` | Performance gain from frame sampling | >1.5x for large GIFs |

### Benchmark Scenarios

1. **small_gif_validation** (10 frames, 256x256)
   - Quick smoke test for CI
   - Validates basic caching functionality

2. **medium_gif_processing** (100 frames, 512x512)
   - Standard workload test
   - Includes all metrics (SSIM, MS-SSIM, LPIPS)

3. **large_gif_with_sampling** (500 frames, 800x600)
   - Tests sampling optimization
   - Memory efficiency validation

4. **multi_metric_workflow** (50 frames, 640x480)
   - Comprehensive metric calculation
   - Cache interaction testing

5. **cache_stress_test** (200 frames, 400x300)
   - Cache eviction behavior
   - Concurrent access patterns

## Daily Operations

### Morning Checklist

1. **Check CI Status**
   ```bash
   # View recent workflow runs
   gh run list --workflow=performance-regression.yml --limit 5
   
   # Check for failures
   gh run list --workflow=performance-regression.yml --status=failure
   ```

2. **Review Performance Alerts**
   ```bash
   # Check monitoring alerts
   poetry run giflab metrics alerts
   
   # View current cache performance
   poetry run giflab cache status
   ```

3. **Validate Baselines**
   ```bash
   # Check baseline age
   ls -la ~/.giflab/baselines/current_baselines.json
   
   # Verify baseline integrity
   python -c "import json; json.load(open('~/.giflab/baselines/current_baselines.json'))"
   ```

### Handling CI Failures

#### Performance Regression Detected

1. **Identify the regression**:
   ```bash
   # Download CI artifacts
   gh run download <RUN_ID> -n performance-reports
   
   # Analyze report
   python scripts/generate_performance_report.py \
     --report-file ci_report_*.json \
     --output-format markdown
   ```

2. **Investigate root cause**:
   - Check recent commits: `git log --oneline -10`
   - Review changes to optimization code
   - Verify configuration changes

3. **Decision tree**:
   ```
   Is regression >10% (CRITICAL)?
   ├─ YES → Block PR merge
   │   ├─ Revert changes
   │   └─ Fix and re-test
   └─ NO → Is it >5% (WARNING)?
       ├─ YES → Require justification
       │   ├─ Document reason
       │   └─ Update baselines if acceptable
       └─ NO → Allow merge with note
   ```

### Weekly Performance Review

Every Monday at 2 AM UTC, automated trend analysis runs:

1. **Review trend report**:
   ```bash
   # Find latest trend report issue
   gh issue list --label "automated-report" --limit 1
   ```

2. **Analyze degradations**:
   ```bash
   # Run manual trend analysis
   poetry run python scripts/analyze_performance_trends.py \
     --history-dir ./performance-history \
     --output trend_analysis.md \
     --format markdown
   ```

3. **Update baselines if needed**:
   ```bash
   # If performance improvements are stable
   poetry run python -m giflab.benchmarks.regression_suite baseline
   
   # Commit new baselines
   git add ~/.giflab/baselines/
   git commit -m "perf: Update performance baselines after improvements"
   ```

## Incident Response

### Critical Performance Regression in Production

**Severity**: P1  
**Response Time**: <15 minutes

1. **Immediate Actions**:
   ```bash
   # Verify the regression
   poetry run python -m giflab.benchmarks.regression_suite test --tags quick
   
   # Check monitoring metrics
   poetry run giflab metrics monitor --last-hour
   ```

2. **Rollback if necessary**:
   ```bash
   # Identify last known good commit
   git log --oneline --grep="perf:" -10
   
   # Create hotfix branch
   git checkout -b hotfix/performance-regression <GOOD_COMMIT>
   
   # Deploy hotfix
   make deploy-hotfix
   ```

3. **Root Cause Analysis**:
   - Profile the regression: `poetry run python tests/performance/profile_metrics.py`
   - Check cache hit rates: `poetry run giflab cache status`
   - Review configuration: `poetry run giflab config current`

4. **Post-Incident**:
   - Update runbook with findings
   - Add regression test for specific case
   - Schedule retrospective

### Gradual Performance Degradation

**Severity**: P2  
**Response Time**: <4 hours

1. **Identify trend**:
   ```bash
   # Analyze 30-day trends
   poetry run python scripts/analyze_performance_trends.py \
     --history-dir ./performance-history \
     --days-back 30
   ```

2. **Bisect to find cause**:
   ```bash
   # Use git bisect with performance test
   git bisect start
   git bisect bad HEAD
   git bisect good <30_DAYS_AGO_COMMIT>
   
   # Run at each step
   git bisect run poetry run python -m giflab.benchmarks.regression_suite test --tags quick
   ```

3. **Remediation**:
   - Fix identified issue
   - Update baselines
   - Add monitoring for specific metric

## Maintenance Procedures

### Updating Baselines

Baselines should be updated when:
- Intentional performance improvements are made
- Infrastructure changes affect all metrics equally
- Monthly scheduled update (first Monday)

```bash
# 1. Run baseline update
poetry run python -m giflab.benchmarks.regression_suite baseline --tags ci

# 2. Verify new baselines
cat ~/.giflab/baselines/current_baselines.json | jq .

# 3. Commit to repository
cp ~/.giflab/baselines/current_baselines.json ./baselines/
git add ./baselines/current_baselines.json
git commit -m "perf: Update baselines for <REASON>"

# 4. Upload to S3 (production)
aws s3 cp ./baselines/current_baselines.json \
  s3://giflab-performance/baselines/current.json
```

### Adding New Benchmark Scenarios

1. **Define scenario** in `src/giflab/benchmarks/regression_suite.py`:
   ```python
   BenchmarkScenario(
       name="new_scenario_name",
       description="What this tests",
       frame_count=100,
       frame_size=(512, 512),
       compression_level=70,
       metrics_to_validate=["ssim", "lpips"],
       sampling_enabled=True,
       tags=["ci", "new"]
   )
   ```

2. **Establish baseline**:
   ```bash
   poetry run python -m giflab.benchmarks.regression_suite baseline \
     --tags new
   ```

3. **Update CI workflow** if needed

### Cache Management

```bash
# View cache statistics
poetry run giflab cache status

# Clear specific cache
poetry run giflab cache clear --type frame
poetry run giflab cache clear --type validation

# Warm cache for testing
poetry run giflab cache warm tests/fixtures/*.gif

# Export cache metrics
poetry run giflab cache export-stats > cache_stats.json
```

## Troubleshooting Guide

### Common Issues

#### 1. Inconsistent Benchmark Results

**Symptoms**: High variance between runs, flaky CI

**Solutions**:
```bash
# Increase measurement rounds
poetry run python -m giflab.benchmarks.regression_suite test \
  --measurement-rounds 20

# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Run with process isolation
nice -n -20 poetry run python -m giflab.benchmarks.regression_suite test
```

#### 2. Cache Not Working

**Symptoms**: 0% cache hit rate, slow performance

**Debugging**:
```bash
# Check cache configuration
poetry run python -c "from giflab.config import FRAME_CACHE; print(FRAME_CACHE)"

# Verify cache directory
ls -la ~/.giflab_cache/

# Test cache directly
poetry run python -c "
from giflab.caching import get_frame_cache
cache = get_frame_cache()
print(cache.get_stats())
"
```

#### 3. Memory Issues During Benchmarks

**Symptoms**: OOM errors, system slowdown

**Solutions**:
```bash
# Reduce parallel execution
export GIFLAB_MAX_WORKERS=2

# Lower cache limits
export GIFLAB_CONFIG_FRAME_CACHE_MEMORY_LIMIT_MB=100

# Use memory profiling
poetry run python -m memory_profiler \
  tests/performance/benchmark_comprehensive.py
```

#### 4. CI Pipeline Timeouts

**Symptoms**: Workflow cancelled after 30 minutes

**Solutions**:
- Reduce scenarios with `--tags quick`
- Split into multiple jobs
- Increase timeout in workflow:
  ```yaml
  timeout-minutes: 45
  ```

### Performance Debugging Commands

```bash
# Profile specific scenario
poetry run python -m cProfile -o profile.stats \
  -m giflab.benchmarks.regression_suite test --tags quick

# Analyze profile
poetry run python -m pstats profile.stats

# Trace execution
poetry run python -m trace --trace \
  -m giflab.benchmarks.regression_suite test

# Memory profiling
poetry run mprof run python -m giflab.benchmarks.regression_suite test
poetry run mprof plot

# System monitoring during test
poetry run python tests/performance/profile_metrics.py &
MONITOR_PID=$!
poetry run python -m giflab.benchmarks.regression_suite test
kill $MONITOR_PID
```

## A/B Testing Operations

### Creating an Experiment

```bash
# Create cache optimization experiment
poetry run python -m giflab.benchmarks.ab_testing create \
  --name "cache_size_test" \
  --control-config '{"FRAME_CACHE.memory_limit_mb": 500}' \
  --treatment-config '{"FRAME_CACHE.memory_limit_mb": 1000}'
```

### Monitoring Experiments

```bash
# View experiment status
poetry run python -m giflab.benchmarks.ab_testing status \
  --name "cache_size_test"

# List all active experiments
poetry run python -m giflab.benchmarks.ab_testing list
```

### Finalizing Experiments

```bash
# Finalize and deploy winner
poetry run python -m giflab.benchmarks.ab_testing finalize \
  --name "cache_size_test" \
  --deploy-winner

# Just finalize without deployment
poetry run python -m giflab.benchmarks.ab_testing finalize \
  --name "cache_size_test"
```

### A/B Test Decision Criteria

| Metric | Improvement Required | Sample Size |
|--------|---------------------|-------------|
| Latency | >10% reduction | 1000+ |
| Cache Hit Rate | >5% increase | 500+ |
| Memory Usage | <20% increase | 200+ |

## Emergency Procedures

### Complete Performance System Failure

If all performance optimizations fail:

1. **Disable all caches**:
   ```bash
   export GIFLAB_CONFIG_FRAME_CACHE_ENABLED=false
   export GIFLAB_CONFIG_VALIDATION_CACHE_ENABLED=false
   export GIFLAB_CONFIG_FRAME_SAMPLING_ENABLED=false
   ```

2. **Use minimal configuration**:
   ```bash
   poetry run python -m giflab.config_manager load-profile low_memory
   ```

3. **Monitor and alert**:
   ```bash
   # Enable verbose logging
   export GIFLAB_LOG_LEVEL=DEBUG
   
   # Start monitoring
   poetry run giflab metrics monitor --alert-threshold 50
   ```

## Appendix

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GIFLAB_PERFORMANCE_CI` | Enable CI optimizations | `0` |
| `GIFLAB_BASELINE_DIR` | Baseline storage directory | `~/.giflab/baselines` |
| `GIFLAB_BENCHMARK_WORKERS` | Parallel benchmark workers | `4` |
| `GIFLAB_STRESS_TESTS` | Enable stress testing | `0` |

### Useful Queries

```sql
-- SQLite queries for performance database
-- Get average cache hit rate by day
SELECT 
  DATE(timestamp) as day,
  AVG(value) as avg_hit_rate
FROM metrics
WHERE name = 'frame_cache.hit_rate'
GROUP BY DATE(timestamp)
ORDER BY day DESC;

-- Find performance outliers
SELECT *
FROM metrics
WHERE name = 'total_validation_time_ms'
  AND value > (SELECT AVG(value) + 2 * STDEV(value) FROM metrics WHERE name = 'total_validation_time_ms')
ORDER BY timestamp DESC;
```

### Contact Information

- **Performance Team**: performance@giflab.io
- **On-Call**: Use PagerDuty escalation
- **Slack Channel**: #giflab-performance
- **Documentation**: https://docs.giflab.io/performance

---

*Last Updated: 2025-01-10*  
*Version: 1.0.0*  
*Next Review: 2025-02-10*