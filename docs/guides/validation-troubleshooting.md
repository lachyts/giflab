# 🛠️ Wrapper Validation System Troubleshooting Guide

This guide provides comprehensive troubleshooting procedures, common issues, solutions, and debugging workflows for the wrapper validation system.

---

## 📋 Quick Diagnosis Checklist

When experiencing validation issues, start with this checklist:

- [ ] **Is validation enabled?** Check `ENABLE_WRAPPER_VALIDATION=True`
- [ ] **Are there clear error messages?** Look in `result["validations"]`
- [ ] **Is the file path correct?** Verify input/output file paths exist
- [ ] **Is the configuration appropriate?** Check tolerances aren't too strict
- [ ] **Are dependencies available?** Ensure PIL/Pillow is installed

---

## 🚨 Common Issues & Solutions

### Issue 1: Validation Not Running

#### Symptoms
```python
result = wrapper.apply(input_path, output_path, params)
print("validations" in result)  # False - no validation metadata
```

#### Possible Causes & Solutions

**Cause A: Validation Disabled in Configuration**
```python
# Check configuration
from giflab.wrapper_validation import ValidationConfig
config = wrapper.validation_config  # Or your config source

if not config.ENABLE_WRAPPER_VALIDATION:
    print("❌ Validation is disabled")
    # Fix: Enable validation
    config.ENABLE_WRAPPER_VALIDATION = True
```

**Cause B: Integration Not Added to Wrapper**
```python
# Check if wrapper calls validation functions
# Wrapper should have this pattern:
def apply(self, input_path, output_path, params):
    result = self._compress(input_path, output_path, params)
    
    # Missing validation call - add this:
    return validate_wrapper_apply_result(self, input_path, output_path, params, result)
```

**Cause C: Import Error**
```python
try:
    from giflab.wrapper_validation.integration import validate_wrapper_apply_result
    print("✅ Validation imports work")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    # Fix: Check package installation and Python path
```

### Issue 2: Validation Always Fails

#### Symptoms
```python
result["validation_passed"]  # Always False
# All validations show is_valid=False
```

#### Possible Causes & Solutions

**Cause A: Tolerances Too Strict**
```python
# Check current tolerances
config = ValidationConfig()
print(f"Frame tolerance: {config.FRAME_RATIO_TOLERANCE}")  # Default: 0.05 (5%)
print(f"Color tolerance: {config.COLOR_COUNT_TOLERANCE}")  # Default: 2
print(f"FPS tolerance: {config.FPS_TOLERANCE}")           # Default: 0.1 (10%)

# Fix: Use more relaxed tolerances
relaxed_config = ValidationConfig(
    FRAME_RATIO_TOLERANCE=0.15,     # 15% instead of 5%
    COLOR_COUNT_TOLERANCE=10,       # 10 colors instead of 2
    FPS_TOLERANCE=0.2               # 20% instead of 10%
)
```

**Cause B: Incorrect Expected Values**
```python
# Debug expected vs actual values
for validation in result["validations"]:
    if not validation["is_valid"]:
        print(f"Validation: {validation['validation_type']}")
        print(f"Expected: {validation['expected']}")
        print(f"Actual: {validation['actual']}")
        print(f"Error: {validation['error_message']}")
        print("---")
```

**Cause C: Wrapper Parameters Don't Match Validation Expectations**
```python
# Frame reduction example
params = {"ratio": 0.5}  # Expecting 50% frame reduction

# But wrapper might not actually implement this ratio correctly
# Fix: Debug wrapper implementation or adjust expectations
```

### Issue 3: File Integrity Validation Fails

#### Symptoms
```python
# Validation error: "Cannot read output file as valid GIF"
```

#### Debugging Steps

**Step 1: Verify File Exists**
```python
from pathlib import Path

def debug_file_integrity(output_path):
    output_path = Path(output_path)
    
    if not output_path.exists():
        print(f"❌ File does not exist: {output_path}")
        return False
    
    print(f"✅ File exists: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")
    
    return True
```

**Step 2: Test File Reading**
```python
def test_file_reading(file_path):
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            print(f"✅ File readable as image")
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            
            if hasattr(img, 'n_frames'):
                print(f"Frames: {img.n_frames}")
            
            return True
            
    except Exception as e:
        print(f"❌ File reading failed: {e}")
        return False
```

**Step 3: Check File Corruption**
```python
def check_file_corruption(file_path):
    """Basic GIF header check."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(6)
            
        if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
            print("✅ Valid GIF header")
            return True
        else:
            print(f"❌ Invalid GIF header: {header}")
            return False
            
    except Exception as e:
        print(f"❌ Header check failed: {e}")
        return False
```

### Issue 4: Color Count Validation Inaccurate

#### Symptoms
```python
# Validation reports wrong color count
# Expected 32 colors, validation says 45 colors found
```

#### Debugging Steps

**Step 1: Manual Color Counting**
```python
from PIL import Image
import numpy as np

def debug_color_count(gif_path):
    """Manual color counting for comparison."""
    with Image.open(gif_path) as img:
        total_colors = set()
        
        # Check all frames
        for frame_idx in range(img.n_frames):
            img.seek(frame_idx)
            frame = img.convert('RGB')
            
            # Count unique colors in this frame
            frame_array = np.array(frame)
            frame_colors = set(map(tuple, frame_array.reshape(-1, 3)))
            total_colors.update(frame_colors)
        
        print(f"Manual count: {len(total_colors)} unique colors")
        return len(total_colors)
```

**Step 2: Compare Validation Methods**
```python
def compare_color_counting_methods(gif_path):
    """Compare different color counting approaches."""
    
    # Method 1: PIL palette
    with Image.open(gif_path) as img:
        if img.mode == 'P':
            palette_colors = len([c for c in img.palette.colors])
            print(f"Palette method: {palette_colors} colors")
    
    # Method 2: Unique pixel values
    manual_count = debug_color_count(gif_path)
    
    # Method 3: Validation system
    from giflab.wrapper_validation.core import WrapperOutputValidator
    validator = WrapperOutputValidator()
    validation_count = validator._count_unique_colors(gif_path)
    print(f"Validation system: {validation_count} colors")
    
    # Compare results
    print(f"Manual vs Validation difference: {abs(manual_count - validation_count)}")
```

### Issue 5: Performance Issues

#### Symptoms
```python
# Validation takes >100ms per operation
# Memory usage grows over time
# High CPU usage during validation
```

#### Performance Debugging

**Step 1: Profile Validation Time**
```python
import time
from giflab.wrapper_validation import WrapperOutputValidator

def profile_validation_performance(input_path, output_path):
    validator = WrapperOutputValidator()
    
    # Profile individual validations
    validations = [
        ("file_integrity", lambda: validator.validate_file_integrity(output_path, {})),
        ("frame_count", lambda: validator.validate_frame_reduction(input_path, output_path, 0.5, {})),
        ("color_count", lambda: validator.validate_color_reduction(input_path, output_path, 32, {})),
    ]
    
    for name, validation_func in validations:
        start_time = time.perf_counter()
        try:
            result = validation_func()
            end_time = time.perf_counter()
            print(f"{name}: {(end_time - start_time) * 1000:.2f}ms")
        except Exception as e:
            print(f"{name}: Failed - {e}")
```

**Step 2: Identify Large File Issues**
```python
def check_file_size_impact(file_paths):
    """Check if file size correlates with slow validation."""
    
    times = []
    sizes = []
    
    for file_path in file_paths:
        file_size = Path(file_path).stat().st_size
        
        start_time = time.perf_counter()
        # Run validation
        result = validator.validate_file_integrity(file_path, {})
        end_time = time.perf_counter()
        
        validation_time = (end_time - start_time) * 1000
        
        times.append(validation_time)
        sizes.append(file_size)
        
        print(f"File: {file_size:,} bytes -> {validation_time:.1f}ms")
    
    # Check correlation
    import statistics
    if len(times) > 3:
        size_sorted = sorted(zip(sizes, times))
        print(f"Smallest file time: {size_sorted[0][1]:.1f}ms")
        print(f"Largest file time: {size_sorted[-1][1]:.1f}ms")
```

**Step 3: Memory Profiling**
```python
import psutil
import os

def profile_memory_usage():
    """Monitor memory usage during validation."""
    process = psutil.Process(os.getpid())
    
    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024
    
    print(f"Initial memory: {get_memory_mb():.1f}MB")
    
    # Run many validations
    for i in range(100):
        result = validator.validate_file_integrity(test_file_path, {})
        
        if i % 10 == 0:
            current_memory = get_memory_mb()
            print(f"After {i+1} validations: {current_memory:.1f}MB")
```

---

## 🔍 Debugging Workflows

### Workflow 1: New Wrapper Integration Issues

**When**: Adding validation to a new wrapper class

```python
def debug_new_wrapper_integration(wrapper_class):
    """Step-by-step debugging for new wrapper integration."""
    
    print("=== Debugging New Wrapper Integration ===")
    
    # Step 1: Check wrapper structure
    wrapper = wrapper_class()
    
    if not hasattr(wrapper, 'apply'):
        print("❌ Wrapper missing apply() method")
        return
    
    print("✅ Wrapper has apply() method")
    
    # Step 2: Test basic operation (no validation)
    try:
        # Use dummy paths for testing
        test_input = Path("tests/fixtures/test_simple.gif")  
        test_output = Path("/tmp/debug_output.gif")
        
        if not test_input.exists():
            print(f"❌ Test input missing: {test_input}")
            return
        
        # Test wrapper core functionality
        result = wrapper._compress(test_input, test_output, {"test": True})
        print(f"✅ Core compression works: {result}")
        
    except Exception as e:
        print(f"❌ Core compression failed: {e}")
        return
    
    # Step 3: Test validation integration
    try:
        from giflab.wrapper_validation.integration import validate_wrapper_apply_result
        
        validated_result = validate_wrapper_apply_result(
            wrapper, test_input, test_output, {"test": True}, result
        )
        
        print("✅ Validation integration successful")
        print(f"Validations run: {len(validated_result.get('validations', []))}")
        
    except Exception as e:
        print(f"❌ Validation integration failed: {e}")
        import traceback
        traceback.print_exc()
```

### Workflow 2: Configuration Issues

**When**: Validation behavior doesn't match expectations

```python
def debug_configuration_issues(config):
    """Debug validation configuration problems."""
    
    print("=== Configuration Debug ===")
    
    # Check configuration values
    config_vars = [
        "ENABLE_WRAPPER_VALIDATION",
        "FRAME_RATIO_TOLERANCE", 
        "COLOR_COUNT_TOLERANCE",
        "FPS_TOLERANCE",
        "MIN_FILE_SIZE_BYTES",
        "MAX_FILE_SIZE_MB",
        "LOG_VALIDATION_FAILURES"
    ]
    
    for var in config_vars:
        value = getattr(config, var, "NOT_SET")
        print(f"{var}: {value}")
    
    # Check for problematic values
    warnings = []
    
    if config.FRAME_RATIO_TOLERANCE < 0.01:
        warnings.append("FRAME_RATIO_TOLERANCE very strict (<1%)")
    
    if config.COLOR_COUNT_TOLERANCE == 0:
        warnings.append("COLOR_COUNT_TOLERANCE exact matching (may cause failures)")
    
    if config.MAX_FILE_SIZE_MB < 1.0:
        warnings.append("MAX_FILE_SIZE_MB very restrictive (<1MB)")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Configuration looks reasonable")
```

### Workflow 3: Result Analysis

**When**: Validation results seem wrong or inconsistent

```python
def analyze_validation_results(result):
    """Detailed analysis of validation results."""
    
    print("=== Validation Result Analysis ===")
    
    # Basic structure check
    required_fields = ["validations", "validation_passed", "validation_summary"]
    for field in required_fields:
        if field in result:
            print(f"✅ Has {field}")
        else:
            print(f"❌ Missing {field}")
    
    if "validations" not in result:
        print("❌ No validation data to analyze")
        return
    
    # Analyze each validation
    validations = result["validations"]
    print(f"\nFound {len(validations)} validations:")
    
    for i, validation in enumerate(validations):
        print(f"\n--- Validation {i+1}: {validation['validation_type']} ---")
        print(f"Status: {'✅ PASS' if validation['is_valid'] else '❌ FAIL'}")
        
        if "expected" in validation:
            print(f"Expected: {validation['expected']}")
        if "actual" in validation:
            print(f"Actual: {validation['actual']}")
        
        if not validation["is_valid"] and "error_message" in validation:
            print(f"Error: {validation['error_message']}")
        
        # Check for detailed information
        if "details" in validation:
            details = validation["details"]
            print(f"Details: {details}")
            
            # Look for timing information
            if "validation_time_ms" in details:
                time_ms = details["validation_time_ms"]
                if time_ms > 50:
                    print(f"⚠️  Slow validation: {time_ms:.1f}ms")
```

---

## 📊 Diagnostic Tools

### Tool 1: Validation Health Check

```python
def validation_health_check(wrapper, input_path, output_path, params):
    """Comprehensive health check for validation system."""
    
    health_report = {
        "overall_status": "unknown",
        "issues": [],
        "recommendations": [],
        "test_results": {}
    }
    
    # Test 1: Basic functionality
    try:
        result = wrapper.apply(input_path, output_path, params)
        health_report["test_results"]["basic_operation"] = "PASS"
    except Exception as e:
        health_report["test_results"]["basic_operation"] = f"FAIL: {e}"
        health_report["issues"].append("Basic wrapper operation failed")
    
    # Test 2: Validation presence
    if "validations" in result:
        health_report["test_results"]["validation_integration"] = "PASS"
        
        # Test 3: Validation execution
        validations = result["validations"]
        if len(validations) > 0:
            health_report["test_results"]["validation_execution"] = "PASS"
        else:
            health_report["test_results"]["validation_execution"] = "FAIL: No validations run"
            health_report["issues"].append("Validation enabled but no validations executed")
        
    else:
        health_report["test_results"]["validation_integration"] = "FAIL"
        health_report["issues"].append("No validation metadata in result")
    
    # Test 4: Performance check
    import time
    start_time = time.perf_counter()
    result = wrapper.apply(input_path, output_path, params)
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    if total_time_ms < 100:
        health_report["test_results"]["performance"] = "PASS"
    elif total_time_ms < 500:
        health_report["test_results"]["performance"] = "ACCEPTABLE"
        health_report["recommendations"].append("Consider performance optimization")
    else:
        health_report["test_results"]["performance"] = "SLOW"
        health_report["issues"].append(f"Validation very slow: {total_time_ms:.1f}ms")
    
    # Overall assessment
    issues_count = len(health_report["issues"])
    if issues_count == 0:
        health_report["overall_status"] = "HEALTHY"
    elif issues_count <= 2:
        health_report["overall_status"] = "CONCERNS"
    else:
        health_report["overall_status"] = "UNHEALTHY"
    
    return health_report

def print_health_report(report):
    """Print formatted health report."""
    
    status_emoji = {
        "HEALTHY": "✅",
        "CONCERNS": "⚠️",
        "UNHEALTHY": "❌",
        "unknown": "❓"
    }
    
    print(f"\n{status_emoji.get(report['overall_status'], '❓')} Overall Status: {report['overall_status']}")
    
    print("\n📋 Test Results:")
    for test, result in report["test_results"].items():
        status = "✅" if result == "PASS" else ("⚠️" if result == "ACCEPTABLE" else "❌")
        print(f"  {status} {test}: {result}")
    
    if report["issues"]:
        print("\n🚨 Issues Found:")
        for issue in report["issues"]:
            print(f"  • {issue}")
    
    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
```

### Tool 2: Configuration Validator

```python
def validate_config_sanity(config):
    """Check configuration for common problems."""
    
    issues = []
    warnings = []
    recommendations = []
    
    # Critical issues
    if not config.ENABLE_WRAPPER_VALIDATION:
        issues.append("Validation is disabled")
    
    if config.FAIL_ON_VALIDATION_ERROR:
        issues.append("FAIL_ON_VALIDATION_ERROR=True will break pipelines")
    
    # Performance warnings
    if config.MAX_FILE_SIZE_MB > 100:
        warnings.append(f"MAX_FILE_SIZE_MB={config.MAX_FILE_SIZE_MB} may cause performance issues")
    
    if config.LOG_VALIDATION_FAILURES and "production" in os.getenv("ENV", "").lower():
        warnings.append("LOG_VALIDATION_FAILURES=True adds I/O overhead in production")
    
    # Tolerance warnings
    tolerance_checks = [
        ("FRAME_RATIO_TOLERANCE", 0.5, "very permissive"),
        ("FRAME_RATIO_TOLERANCE", 0.001, "very strict"),
        ("COLOR_COUNT_TOLERANCE", 50, "very permissive"),
        ("FPS_TOLERANCE", 0.5, "very permissive"),
        ("FPS_TOLERANCE", 0.001, "very strict")
    ]
    
    for param, threshold, description in tolerance_checks:
        value = getattr(config, param)
        if (description == "very permissive" and value > threshold) or \
           (description == "very strict" and value < threshold):
            warnings.append(f"{param}={value} is {description}")
    
    # Recommendations
    if config.MIN_FILE_SIZE_BYTES < 100:
        recommendations.append("Consider MIN_FILE_SIZE_BYTES >= 100 to skip tiny files")
    
    if not config.LOG_VALIDATION_FAILURES:
        recommendations.append("Enable LOG_VALIDATION_FAILURES during development")
    
    return {
        "issues": issues,
        "warnings": warnings, 
        "recommendations": recommendations,
        "overall_severity": "ERROR" if issues else ("WARNING" if warnings else "OK")
    }
```

---

## 🔧 Recovery Procedures

### Recovery 1: Emergency Validation Disable

**When**: Validation is causing production issues

```python
def emergency_disable_validation():
    """Emergency procedure to disable validation system."""
    
    print("🚨 EMERGENCY: Disabling validation system")
    
    # Method 1: Environment variable
    import os
    os.environ['GIFLAB_VALIDATION_EMERGENCY_DISABLE'] = 'true'
    
    # Method 2: Runtime config override
    from giflab.wrapper_validation import ValidationConfig
    emergency_config = ValidationConfig(
        ENABLE_WRAPPER_VALIDATION=False,
        LOG_VALIDATION_FAILURES=False
    )
    
    # Method 3: Apply to all wrappers
    # (Implementation depends on your wrapper management system)
    
    print("✅ Validation disabled. Monitor system recovery.")
    return emergency_config

# Usage in wrapper
def apply_with_emergency_check(self, input_path, output_path, params):
    if os.getenv('GIFLAB_VALIDATION_EMERGENCY_DISABLE'):
        # Skip validation entirely
        return self._compress(input_path, output_path, params)
    
    # Normal validation flow
    result = self._compress(input_path, output_path, params)
    return validate_wrapper_apply_result(self, input_path, output_path, params, result)
```

### Recovery 2: Reset Validation State

**When**: Validation system stuck or corrupted

```python
def reset_validation_system():
    """Reset validation system to known good state."""
    
    print("🔄 Resetting validation system...")
    
    # Clear any cached validation data
    import gc
    gc.collect()
    
    # Reset to default configuration
    from giflab.wrapper_validation import ValidationConfig
    default_config = ValidationConfig()
    
    # Clear validation-related environment variables
    validation_env_vars = [
        'GIFLAB_VALIDATION_EMERGENCY_DISABLE',
        'GIFLAB_VALIDATION_DEBUG',
        'GIFLAB_VALIDATION_PERFORMANCE_MODE'
    ]
    
    for var in validation_env_vars:
        if var in os.environ:
            del os.environ[var]
            print(f"Cleared environment variable: {var}")
    
    print("✅ Validation system reset complete")
    return default_config
```

---

## 📞 Getting Help

### Self-Service Debugging

1. **Enable Debug Logging**
   ```python
   import logging
   logging.getLogger("giflab.wrapper_validation").setLevel(logging.DEBUG)
   ```

2. **Run Health Check**
   ```python
   health_report = validation_health_check(wrapper, input_path, output_path, params)
   print_health_report(health_report)
   ```

3. **Validate Configuration**
   ```python
   config_issues = validate_config_sanity(your_config)
   print(f"Config status: {config_issues['overall_severity']}")
   ```

### Escalation Path

If self-service debugging doesn't resolve the issue:

1. **Gather Diagnostic Information**
   - Validation health check results
   - Configuration values
   - Error messages and stack traces
   - Performance measurements
   - Sample input/output files (if possible)

2. **Check Documentation**
   - [Configuration Reference](../reference/validation-config-reference.md)
   - [Integration Guide](wrapper-validation-integration.md)
   - [Performance Guide](../technical/validation-performance-guide.md)

3. **Report Issue**
   Include diagnostic information, expected behavior, actual behavior, and steps to reproduce.

---

## 🧪 Test Scenarios for Debugging

### Scenario 1: Minimal Test Case

```python
def create_minimal_test_case():
    """Create simplest possible validation test."""
    
    # Create minimal test files
    from PIL import Image
    
    # Simple 2-frame GIF
    frames = []
    for i in range(2):
        frame = Image.new('RGB', (10, 10), color=(i*127, i*127, i*127))
        frames.append(frame)
    
    test_input = Path("/tmp/minimal_test_input.gif")
    test_output = Path("/tmp/minimal_test_output.gif")
    
    frames[0].save(test_input, save_all=True, append_images=frames[1:], duration=100)
    frames[0].save(test_output, save_all=True, append_images=frames[1:], duration=100)
    
    return test_input, test_output
```

### Scenario 2: Edge Case Testing

```python
def test_edge_cases():
    """Test validation with edge case files."""
    
    edge_cases = [
        ("single_frame", create_single_frame_gif),
        ("empty_file", create_empty_file),
        ("corrupted_file", create_corrupted_gif),
        ("huge_file", create_large_gif),
        ("tiny_file", create_tiny_gif)
    ]
    
    results = {}
    
    for case_name, creator_func in edge_cases:
        try:
            test_file = creator_func()
            # Run validation on edge case
            result = run_validation_on_file(test_file)
            results[case_name] = "HANDLED" if "validations" in result else "FAILED"
        except Exception as e:
            results[case_name] = f"ERROR: {e}"
    
    return results
```

---

## ✅ Troubleshooting Checklist

When encountering validation issues:

- [ ] **Check validation is enabled** in configuration
- [ ] **Verify file paths** are correct and accessible  
- [ ] **Test with minimal example** to isolate the issue
- [ ] **Review tolerance settings** - may be too strict
- [ ] **Check for import errors** in validation modules
- [ ] **Enable debug logging** for detailed information
- [ ] **Run health check** to identify system issues
- [ ] **Validate configuration** for common problems
- [ ] **Test edge cases** to understand boundaries
- [ ] **Profile performance** if speed is an issue
- [ ] **Check file integrity** if files seem corrupted
- [ ] **Compare manual vs validation results** for accuracy issues

---

## 📚 Related Documentation

- [Wrapper Integration Guide](wrapper-validation-integration.md)
- [Configuration Reference](../reference/validation-config-reference.md)
- [Performance Optimization Guide](../technical/validation-performance-guide.md)
- [Testing Patterns Guide](../testing/validation-testing-patterns.md)
- [Quality Metrics Integration](../technical/validation-quality-integration.md)

---

## 🤖 Automated Diagnostics

For automated troubleshooting, use the diagnostic tools provided:

```python
# Quick health check
report = validation_health_check(wrapper, input_path, output_path, params)
print_health_report(report)

# Configuration validation
config_status = validate_config_sanity(config)
if config_status["overall_severity"] != "OK":
    print("⚠️ Configuration issues detected")
```