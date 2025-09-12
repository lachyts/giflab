# CLI Dependency Management and Troubleshooting Guide

This guide provides comprehensive troubleshooting procedures for dependency management using the new `giflab deps` command system and resolving import issues related to the conditional import architecture.

## Overview

The `giflab deps` command system was introduced in Phase 2.3 to provide comprehensive dependency checking, installation guidance, and troubleshooting support for the GifLab project's complex dependency requirements.

### Key Features

- **Comprehensive Dependency Checking**: Validates all core and optional dependencies
- **Rich CLI Output**: Beautiful tables and status indicators via Rich library
- **Installation Guidance**: Specific installation commands for each dependency
- **JSON Output**: Machine-readable format for automation and CI/CD
- **Real-time Status**: Integration with memory monitoring and system capabilities
- **Thread-Safe Operations**: Safe for concurrent dependency checking

---

## CLI Commands Reference

### `giflab deps check`

Check availability of all dependencies with detailed reporting.

**Usage**:
```bash
poetry run python -m giflab deps check [OPTIONS]
```

**Options**:
- `--verbose, -v`: Show detailed dependency tables and system capabilities
- `--json`: Output results in JSON format for automation
- `--help`: Show command help

**Examples**:
```bash
# Basic dependency check
poetry run python -m giflab deps check

# Detailed check with comprehensive tables
poetry run python -m giflab deps check --verbose

# JSON output for CI/CD integration
poetry run python -m giflab deps check --json

# Save JSON results to file
poetry run python -m giflab deps check --json > dependency_status.json
```

**Sample Output**:
```
Dependency Status Report
════════════════════════

Core Dependencies
┌─────────────────┬─────────┬────────────────┐
│ Dependency      │ Status  │ Details        │
├─────────────────┼─────────┼────────────────┤
│ PIL/Pillow      │ ✅ PASS │ Version 10.0.1 │
│ OpenCV (cv2)    │ ✅ PASS │ Version 4.8.1  │
│ NumPy           │ ✅ PASS │ Version 1.24.3 │
│ Poetry          │ ✅ PASS │ Version 1.6.1  │
└─────────────────┴─────────┴────────────────┘

Machine Learning
┌─────────────────┬─────────┬────────────────┐
│ Dependency      │ Status  │ Details        │
├─────────────────┼─────────┼────────────────┤
│ PyTorch         │ ✅ PASS │ Version 2.0.1  │
│ LPIPS           │ ✅ PASS │ Available      │
└─────────────────┴─────────┴────────────────┘

Visualization
┌─────────────────┬─────────┬───────────────────────┐
│ Dependency      │ Status  │ Details               │
├─────────────────┼─────────┼───────────────────────┤
│ Matplotlib      │ ✅ PASS │ Version 3.7.2         │
│ Seaborn         │ ❌ FAIL │ Not installed         │
│ Plotly          │ ❌ FAIL │ Import error          │
└─────────────────┴─────────┴───────────────────────┘

External Tools
┌─────────────────┬─────────┬────────────────┐
│ Tool            │ Status  │ Details        │
├─────────────────┼─────────┼────────────────┤
│ SSIMULACRA2     │ ❌ FAIL │ Not found      │
│ Animately       │ ✅ PASS │ Version 1.1.20 │
└─────────────────┴─────────┴────────────────┘

System Capabilities
• Performance Caching: ✅ Enabled
• ML Capabilities: ✅ Available (PyTorch + LPIPS)
• Advanced Metrics: ✅ Available
• Memory Monitoring: ✅ Active (72.3% system usage)
• Lazy Imports: 6/10 loaded

Overall Status: ⚠️ SOME ISSUES (2 missing optional dependencies)
```

### `giflab deps status`

Quick overview of dependency and system status.

**Usage**:
```bash
poetry run python -m giflab deps status
```

**Sample Output**:
```
📊 GifLab Dependency Status

Core Dependencies: ✅ PIL, ✅ OpenCV, ✅ NumPy, ✅ Poetry
ML Capabilities: ✅ PyTorch, ✅ LPIPS
Visualization: ✅ Matplotlib, ❌ Seaborn, ❌ Plotly  
External Tools: ❌ SSIMULACRA2, ✅ Animately

🧠 System Status: 
  Memory: 72.3% used (2.2GB / 8.0GB available)
  Caching: ✅ Enabled  
  Monitoring: ✅ Active
  
📦 Lazy Imports: 6/10 loaded (60% efficiency)

Overall: ⚠️ SOME ISSUES - Run 'giflab deps check --verbose' for details
```

### `giflab deps install-help`

Get installation guidance for dependencies.

**Usage**:
```bash
poetry run python -m giflab deps install-help [DEPENDENCY]
```

**Examples**:
```bash
# General installation help
poetry run python -m giflab deps install-help

# Specific dependency installation
poetry run python -m giflab deps install-help seaborn
poetry run python -m giflab deps install-help pytorch
poetry run python -m giflab deps install-help ssimulacra2
```

**Sample Output**:
```bash
# General help
📦 GifLab Installation Guide

Core Dependencies (Required):
pip install pillow opencv-python numpy
# or with Poetry:
poetry add pillow opencv-python numpy

Machine Learning (Optional):
pip install torch lpips
# or with Poetry:
poetry add torch lpips

Visualization (Optional):  
pip install matplotlib seaborn plotly
# or with Poetry:
poetry add matplotlib seaborn plotly

External Tools (Optional):
# SSIMULACRA2 - Follow installation guide at:
# https://github.com/cloudinary/ssimulacra2

# Animately - Contact repository maintainers

📋 After installation, run 'giflab deps check' to verify
```

```bash
# Specific dependency help
poetry run python -m giflab deps install-help seaborn

📦 Seaborn Installation Guide

Seaborn is a statistical data visualization library based on matplotlib.

Installation Options:
1. Poetry (Recommended):
   poetry add seaborn

2. Pip:
   pip install seaborn

3. Conda:
   conda install seaborn

Requirements:
- Python 3.7+
- matplotlib >= 3.1
- numpy >= 1.15
- pandas >= 0.25

Verification:
After installation, verify with:
poetry run python -c "import seaborn; print('✅ Seaborn installed successfully')"

Related: Run 'giflab deps check' to verify all dependencies
```

---

## Troubleshooting Common Issues

### Issue 1: ModuleNotFoundError for Core Dependencies

#### Symptoms
```bash
poetry run python -m giflab deps check
ModuleNotFoundError: No module named 'click'
# or
ModuleNotFoundError: No module named 'rich'
```

#### Diagnosis
This indicates that Poetry dependencies are not properly installed or the virtual environment is not activated.

#### Resolution Steps

**Step 1: Verify Poetry Installation**
```bash
poetry --version
# Should show: Poetry (version x.x.x)
```

**Step 2: Install Dependencies**
```bash
# From project root directory
poetry install
```

**Step 3: Verify Virtual Environment**
```bash
poetry env info
# Should show active virtual environment details
```

**Step 4: Test Installation**
```bash
poetry run python -c "import click, rich; print('✅ Core dependencies working')"
```

**Step 5: Alternative - Manual Environment Activation**
```bash
# If poetry run doesn't work
poetry shell
python -m giflab deps check
```

### Issue 2: Caching Import Errors

#### Symptoms
```bash
poetry run python -m giflab deps check --verbose

# Output shows:
🚨 Caching features unavailable due to import error.
Failed module: giflab.caching.resized_frame_cache
Error details: No module named 'giflab.caching.resized_frame_cache'
```

#### Diagnosis
The conditional import system is trying to import caching modules but they're missing or have circular dependency issues.

#### Resolution Steps

**Step 1: Check Feature Flag Status**
```bash
poetry run python -c "
from giflab.config import ENABLE_EXPERIMENTAL_CACHING
print(f'Experimental caching enabled: {ENABLE_EXPERIMENTAL_CACHING}')
"
```

**Step 2: Verify Caching Modules Exist**
```bash
find src -name "*caching*" -type f
# Should show caching module files
```

**Step 3: Test Manual Import**
```bash
poetry run python -c "
try:
    from giflab.caching import get_frame_cache
    print('✅ Caching import successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
"
```

**Step 4: Disable Caching if Problematic**
```bash
# Edit src/giflab/config.py
# Change: ENABLE_EXPERIMENTAL_CACHING = True
# To:     ENABLE_EXPERIMENTAL_CACHING = False
```

**Step 5: Verify Resolution**
```bash
poetry run python -m giflab deps check
# Should no longer show caching import errors
```

### Issue 3: Optional Dependency Missing (Expected)

#### Symptoms
```bash
poetry run python -m giflab deps check --verbose

# Output shows:
Visualization
┌─────────────────┬─────────┬───────────────────────┐
│ Seaborn         │ ❌ FAIL │ Not installed         │
│ Plotly          │ ❌ FAIL │ Import error          │
└─────────────────┴─────────┴───────────────────────┘
```

#### Diagnosis
Optional dependencies are missing but this may be expected behavior.

#### Resolution Steps

**Step 1: Determine if Dependencies are Needed**
```bash
# Check what features you plan to use
poetry run python -c "
from giflab.lazy_imports import is_seaborn_available, is_plotly_available
print(f'Seaborn needed: {is_seaborn_available()}')
print(f'Plotly needed: {is_plotly_available()}')
"
```

**Step 2: Install if Needed**
```bash
# For visualization features
poetry add seaborn plotly

# Verify installation
poetry run python -m giflab deps check
```

**Step 3: If Not Needed, Confirm System Still Works**
```bash
# Test core functionality  
poetry run python -c "from giflab.metrics import calculate_comprehensive_metrics; print('✅ Core functionality working')"
```

### Issue 4: External Tool Not Found

#### Symptoms
```bash
External Tools
┌─────────────────┬─────────┬────────────────┐
│ SSIMULACRA2     │ ❌ FAIL │ Not found      │
└─────────────────┴─────────┴────────────────┘
```

#### Diagnosis
External tools like SSIMULACRA2 require separate installation outside of Poetry.

#### Resolution Steps

**Step 1: Get Installation Instructions**
```bash
poetry run python -m giflab deps install-help ssimulacra2
```

**Step 2: Follow External Installation Guide**
```bash
# For SSIMULACRA2 - example steps
git clone https://github.com/cloudinary/ssimulacra2.git
cd ssimulacra2  
# Follow repository-specific build instructions
```

**Step 3: Verify Tool is in PATH**
```bash
which ssimulacra2
# or
ssimulacra2 --version
```

**Step 4: Test with GifLab**
```bash
poetry run python -m giflab deps check
# Should show tool as available
```

### Issue 5: Memory Monitoring Issues

#### Symptoms
```bash
poetry run python -m giflab deps status
# Memory monitoring shows errors or "unavailable"
```

#### Diagnosis
Memory monitoring system is failing or psutil dependency issues.

#### Resolution Steps

**Step 1: Check psutil Installation**
```bash
poetry run python -c "import psutil; print(f'psutil version: {psutil.__version__}')"
```

**Step 2: Test Memory Collection**
```bash
poetry run python -c "
from giflab.monitoring.memory_monitor import SystemMemoryMonitor
monitor = SystemMemoryMonitor()
stats = monitor.get_memory_stats()
print(f'Memory usage: {stats.memory_percent:.1%}')
"
```

**Step 3: Check Configuration**
```bash
poetry run python -c "
from giflab.config import MONITORING
print(f'Memory monitoring enabled: {MONITORING[\"memory_pressure\"][\"enabled\"]}')
"
```

**Step 4: Reset Memory Monitoring**
```bash
poetry run python -c "
from giflab.monitoring.memory_integration import reset_memory_monitoring_state
result = reset_memory_monitoring_state()
print(f'Reset successful: {result}')
"
```

### Issue 6: Performance Degradation

#### Symptoms
```bash
poetry run python -m giflab deps check
# Command takes >10 seconds to complete
```

#### Diagnosis
Dependency checking is taking too long, possibly due to network issues or system overload.

#### Resolution Steps

**Step 1: Profile Dependency Checking**
```bash
time poetry run python -m giflab deps check
# Note which phase takes longest
```

**Step 2: Check System Resources**
```bash
poetry run python -m giflab deps status
# Look at memory usage and system load
```

**Step 3: Use Faster Checking Mode**
```bash
# Skip verbose system checks
poetry run python -m giflab deps check --json | jq '.core_dependencies'
```

**Step 4: Check Network Dependencies** 
```bash
# If slow due to network checks, disable external checks temporarily
poetry run python -c "
import os
os.environ['GIFLAB_SKIP_NETWORK_CHECKS'] = '1'
# Then run dependency check
"
```

---

## Integration with Conditional Import System

### Understanding the Relationship

The `giflab deps` command system is deeply integrated with the conditional import architecture:

1. **Feature Flag Awareness**: Dependency checks respect `ENABLE_EXPERIMENTAL_CACHING` and other feature flags
2. **Import Error Reporting**: Provides detailed diagnostics for conditional import failures  
3. **Graceful Degradation**: Reports when systems are running in fallback mode
4. **Real-time Status**: Shows which optional features are active vs. disabled

### Diagnostic Workflows

**Workflow 1: Conditional Import Troubleshooting**
```bash
# Step 1: Check overall dependency status
poetry run python -m giflab deps check --verbose

# Step 2: Check specific import status  
poetry run python -c "
from giflab.metrics import CACHING_ENABLED, get_frame_cache
print(f'Caching enabled: {CACHING_ENABLED}')
print(f'Frame cache available: {get_frame_cache is not None}')
"

# Step 3: Test conditional imports manually
poetry run python -c "
from giflab.config import ENABLE_EXPERIMENTAL_CACHING
print(f'Feature flag: {ENABLE_EXPERIMENTAL_CACHING}')

if ENABLE_EXPERIMENTAL_CACHING:
    try:
        from giflab.caching import get_frame_cache
        print('✅ Conditional import successful')
    except ImportError as e:
        print(f'❌ Conditional import failed: {e}')
else:
    print('ℹ️ Experimental caching disabled by feature flag')
"
```

**Workflow 2: Lazy Import Diagnostics**
```bash
# Check lazy import efficiency
poetry run python -c "
from giflab.lazy_imports import (
    is_pil_available, is_matplotlib_available, 
    is_seaborn_available, is_plotly_available
)

imports = [
    ('PIL', is_pil_available()),
    ('Matplotlib', is_matplotlib_available()),
    ('Seaborn', is_seaborn_available()),
    ('Plotly', is_plotly_available())
]

loaded = sum(1 for _, available in imports if available)
print(f'Lazy imports loaded: {loaded}/{len(imports)}')

for name, available in imports:
    status = '✅' if available else '❌'
    print(f'  {status} {name}')
"
```

---

## Automation and CI/CD Integration

### JSON Output Format

The `--json` flag provides machine-readable output for automation:

```json
{
  "timestamp": "2025-01-11T15:30:00Z",
  "overall_status": "SOME_ISSUES",
  "core_dependencies": {
    "PIL": {"status": "PASS", "version": "10.0.1"},
    "cv2": {"status": "PASS", "version": "4.8.1"},
    "numpy": {"status": "PASS", "version": "1.24.3"}
  },
  "optional_dependencies": {
    "seaborn": {"status": "FAIL", "error": "Not installed"},
    "plotly": {"status": "FAIL", "error": "Import error"}
  },
  "external_tools": {
    "ssimulacra2": {"status": "FAIL", "error": "Not found in PATH"},
    "animately": {"status": "PASS", "version": "1.1.20"}
  },
  "system_capabilities": {
    "caching_enabled": true,
    "ml_available": true,
    "memory_monitoring_active": true,
    "memory_usage_percent": 72.3
  },
  "lazy_imports": {
    "loaded_count": 6,
    "total_count": 10,
    "efficiency_percent": 60.0
  }
}
```

### CI/CD Pipeline Integration

**GitHub Actions Example**:
```yaml
name: Dependency Check
on: [push, pull_request]

jobs:
  dependency-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install Poetry
      run: pip install poetry
    
    - name: Install dependencies
      run: poetry install
    
    - name: Check dependencies
      run: |
        poetry run python -m giflab deps check --json > deps_report.json
        
        # Parse results and set exit code
        python -c "
        import json, sys
        with open('deps_report.json') as f:
            report = json.load(f)
        
        if report['overall_status'] == 'ALL_PASS':
            print('✅ All dependencies available')
            sys.exit(0)
        elif report['overall_status'] == 'SOME_ISSUES':
            print('⚠️ Some optional dependencies missing')
            # Check if core dependencies are OK
            core_ok = all(dep['status'] == 'PASS' 
                         for dep in report['core_dependencies'].values())
            sys.exit(0 if core_ok else 1)
        else:
            print('❌ Critical dependency failures')
            sys.exit(1)
        "
    
    - name: Upload dependency report
      uses: actions/upload-artifact@v3
      with:
        name: dependency-report
        path: deps_report.json
```

**Pre-commit Hook Example**:
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Checking dependencies before commit..."

# Quick dependency check
if ! poetry run python -m giflab deps check --json > /tmp/deps_check.json; then
    echo "❌ Dependency check failed"
    exit 1
fi

# Parse results
CORE_OK=$(python3 -c "
import json
with open('/tmp/deps_check.json') as f:
    report = json.load(f)
core_ok = all(dep['status'] == 'PASS' for dep in report['core_dependencies'].values())
print('true' if core_ok else 'false')
")

if [ "$CORE_OK" != "true" ]; then
    echo "❌ Core dependencies missing - commit blocked"
    echo "Run 'poetry install' to install missing dependencies"
    exit 1
fi

echo "✅ Dependencies OK"
```

---

## Advanced Troubleshooting

### Debug Mode

Enable verbose logging for dependency checking:

```bash
# Set debug environment variable
export GIFLAB_DEBUG_DEPS=1

# Run with debug output
poetry run python -m giflab deps check --verbose 2>&1 | tee debug.log
```

### Manual Import Testing

For complex import issues, test imports manually:

```python
# create debug_imports.py
import sys
import traceback

def test_import(module_name, description=""):
    """Test import with detailed error reporting."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {e}")
        if "GIFLAB_DEBUG_DEPS" in os.environ:
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"⚠️ {module_name} - Unexpected error: {e}")
        traceback.print_exc()
        return False

# Test all imports
imports_to_test = [
    ("click", "CLI framework"),
    ("rich", "Rich text and beautiful formatting"),
    ("PIL", "Python Imaging Library"),
    ("cv2", "OpenCV computer vision"),
    ("numpy", "Numerical computing"),
    ("giflab.config", "GifLab configuration"),
    ("giflab.lazy_imports", "Lazy import system"),
    ("giflab.metrics", "Core metrics system"),
]

results = []
for module, desc in imports_to_test:
    result = test_import(module, desc)
    results.append((module, result))

# Summary
passed = sum(1 for _, result in results if result)
total = len(results)
print(f"\nSummary: {passed}/{total} imports successful")

if passed < total:
    print("Failed imports may need installation or have dependency conflicts")
```

### System Information Collection

Collect comprehensive system information for troubleshooting:

```bash
# Create system info script
poetry run python -c "
import sys, platform, os
from pathlib import Path

print('=== System Information ===')
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Working Directory: {os.getcwd()}')
print(f'Python Path: {sys.path[:3]}')  # First 3 paths

print('\n=== Poetry Information ===')
os.system('poetry --version')
os.system('poetry env info')

print('\n=== GifLab Project Structure ===')
project_root = Path('.')
key_paths = [
    'pyproject.toml',
    'src/giflab/__init__.py',
    'src/giflab/config.py',
    'src/giflab/metrics.py'
]

for path in key_paths:
    path_obj = Path(path)
    status = '✅' if path_obj.exists() else '❌'
    print(f'{status} {path}')

print('\n=== Environment Variables ===')
giflab_vars = {k: v for k, v in os.environ.items() if 'GIFLAB' in k}
for var, value in giflab_vars.items():
    print(f'{var}={value}')

print('\n=== Dependency Status ===')
"

# Follow up with dependency check
poetry run python -m giflab deps check --json | jq '.'
```

---

## Summary

The CLI dependency management system provides:

- **Comprehensive Checking**: Validates all dependencies with detailed reporting
- **User-Friendly Output**: Rich text formatting with clear status indicators
- **Automation Support**: JSON output for CI/CD integration
- **Installation Guidance**: Specific instructions for each dependency type
- **Integration Awareness**: Deep integration with conditional import system
- **Performance Monitoring**: Real-time system status and memory monitoring
- **Troubleshooting Support**: Detailed diagnostics and error resolution workflows

This system significantly improves the developer and user experience by providing clear visibility into dependency status and actionable guidance for resolving issues.

---

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Related Documentation: [Conditional Import Architecture](../technical/conditional-import-architecture.md), [Memory Monitoring Architecture](../technical/memory-monitoring-architecture.md), [Configuration Guide](../configuration-guide.md)*